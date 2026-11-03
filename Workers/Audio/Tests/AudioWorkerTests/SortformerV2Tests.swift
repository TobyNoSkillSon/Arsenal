import CryptoKit
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
import Testing
@testable import AudioWorker

/// Streaming Sortformer 4spk v2. Default tests run no model and no GPU work: folder ↔ adapter contract, presets ↔ the
/// model card, the frontend tables and NeMo's post-processing against the golden captured once from NeMo. The heavy
/// tests compare the MLX port with that capture per component and end to end.
@Suite struct SortformerV2Tests {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    static let folder = repo.appendingPathComponent("models/sortformer-4spk-v2")
    static let evidence = repo.appendingPathComponent("lab/evidence/sortformer-4spk-v2")

    static func golden() throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("gate/golden/golden.json"))) as? [String: Any])
    }

    static func segmentLines(_ segments: [SortformerV2PostProcessing.Segment]) -> [String] {
        segments.sorted { ($0.speaker, $0.start) < ($1.speaker, $1.start) }
            .map { String(format: "%.3f %.3f speaker_%d", $0.start, $0.end, $0.speaker) }
    }

    @Test func folderMatchesAdapterContract() throws {
        let manifest = try String(contentsOf: Self.folder.appendingPathComponent("model.toml"), encoding: .utf8)
        #expect(manifest.contains("architecture = \"sortformer_streaming\""))
        #expect(manifest.contains("fp16 = \"\(SortformerV2Conversion.recipe)\""))
        let registration = try #require(AudioWorker.registrations.first { $0.modelTypes.contains("sortformer_streaming") })
        let adapter = try #require(registration.make() as? SortformerV2Adapter)
        #expect(adapter.operations == ["diarize", "diarize_session"])
        for op in adapter.operations {
            #expect(manifest.contains("name = \"\(op)\""))
            for side in ["input", "output"] {
                let schema = try #require(
                    JSONSerialization.jsonObject(with: Data(contentsOf: Self.folder.appendingPathComponent("ops/\(op).\(side).json"))) as? [String: Any])
                #expect(schema["additionalProperties"] as? Bool == false)
                let properties = try #require(schema["properties"] as? [String: Any])
                if side == "input" {
                    // Every option the adapter accepts is declared, and nothing else.
                    let accepted = SortformerV2Adapter.optionKeys.union(op == "diarize" ? ["audio"] : ["audio", "action", "session_id"])
                    #expect(Set(properties.keys) == accepted)
                } else {
                    #expect(Set(["segments", "speakers", "rttm"]).isSubset(of: Set(properties.keys)))
                }
            }
        }
    }

    /// The model card's streaming table (all in 80 ms frames) and NeMo's `_check_streaming_parameters` minimum.
    @Test func presetsMatchModelCard() throws {
        let p = SortformerV2Geometry.presets
        #expect(p.count == 4)
        #expect(p["very_high"] == .init(chunk: 340, rightContext: 40, fifo: 40, updatePeriod: 300, cache: 188))
        #expect(p["high"] == .init(chunk: 124, rightContext: 1, fifo: 124, updatePeriod: 124, cache: 188))
        #expect(p["low"] == .init(chunk: 6, rightContext: 7, fifo: 188, updatePeriod: 144, cache: 188))
        #expect(p["ultra_low"] == .init(chunk: 3, rightContext: 1, fifo: 188, updatePeriod: 144, cache: 188))
        for (name, geometry) in p { #expect((try? geometry.validate()) != nil, "\(name)") }
        #expect(throws: (any Error).self) { try SortformerV2Geometry(chunk: 6, rightContext: 7, fifo: 188, updatePeriod: 144, cache: 15).validate() }
        let input = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: Self.folder.appendingPathComponent("ops/diarize.input.json"))) as? [String: Any])
        let latency = try #require((input["properties"] as? [String: Any])?["latency"] as? [String: Any])
        #expect(Set(latency["enum"] as? [String] ?? []) == Set(p.keys))
        #expect(latency["default"] as? String == SortformerV2Geometry.defaultPreset)
        // Overrides apply on top of the preset; an unknown preset is refused.
        let custom = try SortformerV2Adapter.geometry(["latency": "low", "chunk_frames": 10, "speaker_cache_frames": 32])
        #expect(custom == .init(chunk: 10, rightContext: 7, fifo: 188, updatePeriod: 144, cache: 32))
        #expect(throws: (any Error).self) { try SortformerV2Adapter.geometry(["latency": "instant"]) }
    }

    /// The Swift filterbank and window equal NeMo's float32 preprocessor buffers (librosa Slaney, symmetric Hann).
    @Test func frontendTablesMatchNeMo() throws {
        let frontend = try #require(try Self.golden()["frontend"] as? [String: Any])
        let features = SortformerV2Features()
        for case let entry as [Double] in try #require(frontend["filterbank_samples"] as? [Any]) {
            let value = features.filterbank[Int(entry[0]) * SortformerV2Features.bins + Int(entry[1])]
            #expect(abs(Double(value) - entry[2]) <= 1e-7 + 1e-5 * abs(entry[2]), "fb[\(entry[0]), \(entry[1])]")
        }
        for case let entry as [Double] in try #require(frontend["window_samples"] as? [Any]) {
            #expect(abs(Double(features.hann[56 + Int(entry[0])]) - entry[1]) <= 1e-6)
        }
        #expect(SortformerV2Features.frameCount(samples: 439_529) == 2_747)
        #expect(SortformerV2Features.completeFrames(available: 255) == 0)
        #expect(SortformerV2Features.completeFrames(available: 256) == 1)
    }

    /// NeMo's `ts_vad_post_processing` + `generate_diarization_output_lines` on NeMo's own probabilities.
    @Test func postProcessingReproducesNeMoSegments() throws {
        let streams = try #require(try Self.golden()["streams"] as? [String: Any])
        var checked = 0
        for tag in ["small", "card"] {
            let stream = try #require(streams[tag] as? [String: Any])
            let probabilities = try #require(stream["probabilities"] as? [Double]).map(Float.init)
            #expect(probabilities.count == (stream["frames"] as? Int ?? 0) * 4)
            let expected = try #require(stream["segments_nemo"] as? [String])
            #expect(Self.segmentLines(SortformerV2PostProcessing().segments(probabilities)) == expected, "\(tag)")
            checked += 1
        }
        #expect(checked == 2)
    }

    @Test func postProcessingOptionsAndOutputShape() throws {
        // One speaker on frames 2…5 and 7…9, another on frame 8 only.
        var p = [Float](repeating: 0, count: 12 * 4)
        for f in [2, 3, 4, 5, 7, 8, 9] { p[f * 4] = 0.9 }
        p[8 * 4 + 1] = 0.7
        let plain = SortformerV2PostProcessing().segments(p)
        #expect(plain == [.init(start: 0.16, end: 0.48, speaker: 0), .init(start: 0.56, end: 0.8, speaker: 0), .init(start: 0.64, end: 0.72, speaker: 1)])
        let filled = SortformerV2PostProcessing(minDurationOn: 0.1, minDurationOff: 0.1).segments(p)
        #expect(filled == [.init(start: 0.16, end: 0.8, speaker: 0)])
        let output = SortformerV2Adapter.output(p, post: SortformerV2PostProcessing(), duration: 0.75, includeProbabilities: true)
        #expect(output["speakers"]?.array?.compactMap(\.string) == ["SPEAKER_00", "SPEAKER_01"])
        #expect(output["segments"]?.array?.count == 3)
        #expect(output["segments"]?.array?[1]["end"]?.number == 0.75) // clipped to the recording
        #expect(output["probabilities"]?.array?.count == 12)
        #expect(output["rttm"]?.string?.hasPrefix("SPEAKER audio 1 0.160 0.320 <NA> <NA> SPEAKER_00") == true)
    }

    /// NeMo filters durations in float32 (torch tensors), so a segment exactly at `min_duration_on` is kept or dropped
    /// by its float32 length; reference outcomes from torch: 0.08–0.48 s is dropped at 0.4, 0.16–0.56 s is kept.
    @Test func durationFiltersCompareInFloat32LikeNeMo() {
        func track(_ frames: Range<Int>) -> [Float] {
            var p = [Float](repeating: 0, count: 12 * 4)
            for f in frames { p[f * 4] = 0.9 }
            return p
        }
        let filter = SortformerV2PostProcessing(minDurationOn: 0.4)
        #expect(filter.segments(track(1..<6)).isEmpty)
        #expect(filter.segments(track(2..<7)) == [.init(start: 0.16, end: 0.56, speaker: 0)])
    }

    @Test func abandonedSessionsFreeTheirSlot() throws {
        useTestMetallib()
        #expect(!SortformerV2Adapter.abandoned(lastUsed: 100, now: 100 + 599))
        #expect(SortformerV2Adapter.abandoned(lastUsed: 100, now: 100 + 600))
        // Through create itself (2 Nov 2026 review: the old test checked only the predicate). A one-layer random
        // network; creating a session runs no inference.
        let adapter = SortformerV2Adapter()
        adapter.network = SortformerV2Network(SortformerV2Dimensions(["conformer_layers": 1, "transformer_layers": 1]))
        var clock = 1_000.0
        adapter.now = { clock }
        try expectAbandonedSessionsFreeTheirSlot(adapter, advance: { clock += $0 })
    }

    @Test func conversionNamesAndNemoArchive() throws {
        let conv = try #require(try SortformerV2Conversion.target("encoder.pre_encode.conv.0.weight", shape: [256, 1, 3, 3]))
        #expect(conv.0 == "encoder.pre_encode.conv.0.weight" && conv.1 == [0, 2, 3, 1])
        #expect(try SortformerV2Conversion.target("encoder.layers.3.conv.depthwise_conv.weight", shape: [512, 1, 9])!.1 == [0, 2, 1])
        #expect(try SortformerV2Conversion.target("encoder.layers.3.conv.batch_norm.num_batches_tracked", shape: []) == nil)
        #expect(try SortformerV2Conversion.target("preprocessor.featurizer.fb", shape: [1, 128, 257]) == nil)
        #expect(try SortformerV2Conversion.target("transformer_encoder.layers.0.layer_norm_1.weight", shape: [192])!.1 == nil)
        #expect(throws: (any Error).self) { try SortformerV2Conversion.target("decoder.weight", shape: [4]) }
        // A two-member ustar archive in NeMo's layout ("./" names).
        func header(_ name: String, _ size: Int) -> Data {
            var h = Data(count: 512)
            h.replaceSubrange(0..<name.utf8.count, with: Data(name.utf8))
            let octal = String(format: "%011o", size)
            h.replaceSubrange(124..<135, with: Data(octal.utf8))
            h[156] = 0x30
            h.replaceSubrange(257..<262, with: Data("ustar".utf8))
            return h
        }
        var tar = header("./model_config.yaml", 5) + Data("a: 1\n".utf8) + Data(count: 507)
        tar += header("./model_weights.ckpt", 600) + Data(repeating: 7, count: 600) + Data(count: 424) + Data(count: 1024)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".nemo")
        try tar.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let members = try SortformerNemoArchive.members(handle)
        #expect(members == ["model_config.yaml": .init(offset: 512, size: 5), "model_weights.ckpt": .init(offset: 1536, size: 600)])
    }

    // MARK: - Heavy: parity with the NeMo capture

    static func relativeL2(_ a: [Float], _ b: [Float]) -> Double {
        var num = 0.0, den = 0.0
        for (x, y) in zip(a, b) { num += Double(x - y) * Double(x - y); den += Double(y) * Double(y) }
        return (num / max(den, 1e-30)).squareRoot()
    }
    static func number(_ value: Any?) -> Double { (value as? NSNumber)?.doubleValue ?? .infinity }
    static func maxAbs(_ a: [Float], _ b: [Float]) -> Double { zip(a, b).map { Double(abs($0 - $1)) }.max() ?? 0 }
    static func meanAbs(_ a: [Float], _ b: [Float]) -> Double { zip(a, b).map { Double(abs($0 - $1)) }.reduce(0, +) / Double(max(1, a.count)) }

    static func readWave(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        // The fixture is a canonical 44-byte-header PCM16 mono WAV written by Python's wave module.
        return stride(from: 44, to: data.count - 1, by: 2).map { Float(Int16(bitPattern: UInt16(data[$0]) | UInt16(data[$0 + 1]) << 8)) / 32_768 }
    }

    static func writeWave(_ samples: [Float], to url: URL) throws {
        var data = Data()
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + samples.count * 2); data.append(contentsOf: Array("WAVEfmt ".utf8))
        u32(16); u16(1); u16(1); u32(16_000); u32(32_000); u16(2); u16(16); data.append(contentsOf: Array("data".utf8)); u32(samples.count * 2)
        for x in samples { withUnsafeBytes(of: Int16(max(-32_768, min(32_767, (x * 32_768).rounded()))).littleEndian) { data.append(contentsOf: $0) } }
        try data.write(to: url)
    }

    /// Converts the pinned `.nemo` from the Hugging Face cache (APFS clone) with the Get conversion and loads the adapter.
    static func loadedAdapter(_ root: URL) throws -> SortformerV2Adapter {
        if let metallib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: metallib)
        }
        let nemo = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(
            ".cache/huggingface/hub/models--nvidia--diar_streaming_sortformer_4spk-v2/snapshots/84edd514b8ef68004c10086918cd62f2148cbd59/diar_streaming_sortformer_4spk-v2.nemo"
        ).resolvingSymlinksInPath()
        let source = root.appendingPathComponent("source"), weights = root.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: nemo, to: source.appendingPathComponent("diar_streaming_sortformer_4spk-v2.nemo"))
        _ = try SortformerV2Conversion.prepare(
            NativeConversionContext(source: source, assets: [:], destination: weights, precision: "fp16", half: .float16), NativeCancellation())
        let adapter = SortformerV2Adapter()
        try adapter.load(
            NativeLoadContext(["model_dir": .string(folder.path), "weights_dir": .string(weights.path), "dtype": "fp16", "config": ["architecture": "sortformer_streaming"]]),
            cancellation: NativeCancellation())
        return adapter
    }

    @Test(.heavy) func parityWithNeMoCapture() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("sortformer-parity-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root); Memory.clearCache() }
        let adapter = try Self.loadedAdapter(root)
        let network = try #require(adapter.network)
        let golden = try Self.golden()
        let tolerance = try #require(golden["tolerance"] as? [String: Any])
        let reference = try loadArrays(url: Self.evidence.appendingPathComponent("reference/reference.safetensors"))
        func ref(_ key: String) throws -> [Float] { try #require(reference[key]).asType(.float32).asArray(Float.self) }
        var report: [String: Any] = [:]

        // Counts before values: the fixture and its frame count.
        let wav = Self.evidence.appendingPathComponent("fixture-3spk.wav")
        let fixture = try #require(golden["fixture"] as? [String: Any])
        let digest = SHA256.hash(data: try Data(contentsOf: wav)).map { String(format: "%02x", $0) }.joined()
        #expect(digest == fixture["sha256"] as? String)
        let samples = try Self.readWave(wav)
        #expect(samples.count == fixture["samples"] as? Int)
        let frames = SortformerV2Features.frameCount(samples: samples.count)
        #expect(frames == 2_747)

        // 1. Frontend: [128, T] reference vs Swift [T, 128].
        let mel = adapter.features.features(samples, frames: 0..<frames)
        let referenceMel = try ref("features")
        var transposed = [Float](repeating: 0, count: mel.count)
        for t in 0..<frames { for m in 0..<128 { transposed[m * frames + t] = mel[t * 128 + m] } }
        report["features_max_abs"] = Self.maxAbs(transposed, referenceMel)
        report["features_relative_l2"] = Self.relativeL2(transposed, referenceMel)

        // 2. Components on the reference's own inputs.
        let pre = network.preEncode(try #require(reference["pre_encode_input"]).expandedDimensions(axis: 0))
        report["pre_encode_relative_l2"] = Self.relativeL2(pre.asType(.float32).asArray(Float.self), try ref("pre_encode_output"))
        let front = network.frontEnd(try #require(reference["encoder_input"]).expandedDimensions(axis: 0))
        report["encoder_relative_l2"] = Self.relativeL2(front.asType(.float32).asArray(Float.self), try ref("encoder_output"))
        let head = network.infer(try #require(reference["encoder_output"]).asType(.float16).expandedDimensions(axis: 0))
        report["head_max_abs"] = Self.maxAbs(head.asArray(Float.self), try ref("head_output"))

        // 3. One streaming cache step on NeMo's own state (small geometry, step 12: FIFO pop and cache compression).
        let small = SortformerV2Geometry(chunk: 10, rightContext: 2, fifo: 20, updatePeriod: 16, cache: 32)
        let stepStream = SortformerV2Stream(network: network, geometry: small)
        var state = SortformerStreamingState(config: small.updater)
        state.spkcache = try ref("small_step12_in_spkcache"); state.spkcacheLength = state.spkcache.count / 512
        state.fifo = try ref("small_step12_in_fifo"); state.fifoLength = state.fifo.count / 512
        state.spkcachePreds = reference["small_step12_in_spkcache_preds"]?.asType(.float32).asArray(Float.self)
        state.fifoPreds = reference["small_step12_in_fifo_preds"]?.asType(.float32).asArray(Float.self)
        state.meanSilenceEmbedding = try ref("small_step12_in_mean_sil")
        state.silenceFrameCount = Int(try #require(reference["small_step12_in_n_sil"]).asArray(Int32.self)[0])
        stepStream.replaceState(state)
        let chunk = try ref("small_step12_in_chunk")
        let core = try stepStream.step(chunk, frames: chunk.count / 128, leftFrames: 1, rightFrames: 2)
        let totalBefore = try ref("small_step12_out_total")
        report["step_chunk_probability_max_abs"] = Self.maxAbs(core, Array(totalBefore.suffix(core.count)))
        #expect(stepStream.state.spkcacheLength * 512 == (try ref("small_step12_out_spkcache")).count)
        report["step_spkcache_relative_l2"] = Self.relativeL2(stepStream.state.spkcache, try ref("small_step12_out_spkcache"))
        report["step_fifo_relative_l2"] = Self.relativeL2(stepStream.state.fifo, try ref("small_step12_out_fifo"))
        report["step_mean_silence_relative_l2"] = Self.relativeL2(stepStream.state.meanSilenceEmbedding, try ref("small_step12_out_mean_sil"))

        // 4. Whole streams from the Swift frontend: probabilities and NeMo-post-processed segments.
        let streams = try #require(golden["streams"] as? [String: Any])
        for tag in ["small", "card", "low"] {
            let entry = try #require(streams[tag] as? [String: Any]), config = try #require(entry["config"] as? [String: Int])
            let geometry = SortformerV2Geometry(
                chunk: config["chunk_len"]!, rightContext: config["chunk_right_context"]!, fifo: config["fifo_len"]!,
                updatePeriod: config["spkcache_update_period"]!, cache: config["spkcache_len"]!)
            let probabilities = try adapter.diarize(samples, geometry: geometry, cancellation: nil)
            let expected = try ref(tag + "_total_preds")
            #expect(probabilities.count == expected.count)
            report[tag + "_probability_max_abs"] = Self.maxAbs(probabilities, expected)
            report[tag + "_probability_mean_abs"] = Self.meanAbs(probabilities, expected)
            report[tag + "_segments_port"] = Self.segmentLines(SortformerV2PostProcessing().segments(probabilities))
            report[tag + "_segments_nemo"] = entry["segments_nemo"]
        }
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: Self.evidence.appendingPathComponent("parity-port.json"))
        FileHandle.standardError.write(data)

        #expect(Self.number(report["features_max_abs"]) <= Self.number(tolerance["features_max_abs"]))
        let componentBound = Self.number(tolerance["component_relative_l2"])
        for key in ["pre_encode_relative_l2", "encoder_relative_l2", "step_spkcache_relative_l2", "step_fifo_relative_l2"] {
            #expect(Self.number(report[key]) <= componentBound, "\(key)")
        }
        let probabilityBound = Self.number(tolerance["probability_max_abs"]), meanBound = Self.number(tolerance["probability_mean_abs"])
        for key in ["head_max_abs", "step_chunk_probability_max_abs", "small_probability_max_abs", "card_probability_max_abs", "low_probability_max_abs"] {
            #expect(Self.number(report[key]) <= probabilityBound, "\(key)")
        }
        for tag in ["small", "card", "low"] {
            #expect(Self.number(report[tag + "_probability_mean_abs"]) <= meanBound, "\(tag)")
            let port = (report[tag + "_segments_port"] as? [String] ?? []), nemo = (report[tag + "_segments_nemo"] as? [String] ?? [])
            #expect(port.count == nemo.count, "\(tag)")
            for (a, b) in zip(port, nemo) {
                let x = a.split(separator: " "), y = b.split(separator: " ")
                #expect(x[2] == y[2] && abs(Double(x[0])! - Double(y[0])!) <= 0.0801 && abs(Double(x[1])! - Double(y[1])!) <= 0.0801, "\(tag): \(a) vs \(b)")
            }
        }
    }

    static func ignore(_: WireFrame) {}
    static func push(_ id: String, _ piece: URL) -> JSONObject { ["action": "push", "session_id": .string(id), "audio": ["path": .string(piece.path)]] }

    /// A live session fed in uneven pieces ends with exactly the offline result at the same geometry.
    @Test(.heavy) func sessionMatchesOfflineDiarization() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("sortformer-session-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root); Memory.clearCache() }
        let adapter = try Self.loadedAdapter(root)
        let samples = try Self.readWave(Self.evidence.appendingPathComponent("fixture-3spk.wav"))
        let offline = try adapter.diarize(samples, geometry: SortformerV2Geometry.presets["low"]!, cancellation: nil)
        var pieces: [URL] = []
        var cursor = 0, size = 7_919
        while cursor < samples.count {
            let end = min(samples.count, cursor + size)
            let url = root.appendingPathComponent("piece-\(pieces.count).wav")
            try Self.writeWave(Array(samples[cursor..<end]), to: url)
            pieces.append(url)
            cursor = end
            size = size == 7_919 ? 24_007 : 7_919
        }
        let cancellation = NativeCancellation()
        let created = try adapter.run("diarize_session", input: ["action": "create", "latency": "low", "include_probabilities": true], cancellation: cancellation) { _ in }
        let id = try #require(created["session_id"]?.string)
        var lastConfirmed = 0
        for piece in pieces {
            let out = try adapter.run("diarize_session", input: Self.push(id, piece), cancellation: cancellation, emit: Self.ignore)
            let confirmed = out["probabilities"]?.array?.count ?? 0
            #expect(confirmed >= lastConfirmed)
            lastConfirmed = confirmed
        }
        #expect(lastConfirmed > 0 && lastConfirmed < offline.count / 4)
        let final = try adapter.run("diarize_session", input: ["action": "finish", "session_id": .string(id)], cancellation: cancellation) { _ in }
        #expect(final["done"]?.bool == true)
        let streamed = (final["probabilities"]?.array ?? []).flatMap { $0.array ?? [] }.compactMap(\.number)
        #expect(streamed.count == offline.count)
        let rounded = offline.map { (Double($0) * 10_000).rounded() / 10_000 }
        #expect(zip(streamed, rounded).allSatisfy { abs($0 - $1) <= 1e-9 })
        #expect(throws: (any Error).self) {
            _ = try adapter.run("diarize_session", input: Self.push(id, pieces[0]), cancellation: cancellation, emit: Self.ignore)
        }
    }
}

/// Two clients open sessions and vanish: a third create is refused (422) until ten idle minutes pass, then it succeeds,
/// and a session still in use keeps its slot.
/// Points MLX at the test metallib: the session tests build a small network, and run alone (not after a suite that sets
/// it) MLX otherwise fails to find its shaders.
func useTestMetallib() {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] {
        GPU.metallib = URL(fileURLWithPath: path)
    }
}

func expectAbandonedSessionsFreeTheirSlot(_ adapter: AudioAdapter, advance: (Double) -> Void) throws {
    func create(_ id: String) throws -> JSONObject {
        try adapter.run("diarize_session", input: ["action": "create", "session_id": .string(id)], cancellation: NativeCancellation(), emit: { _ in })
    }
    _ = try create("a"); _ = try create("b")
    let full = #expect(throws: WorkerFailure.self) { try create("c") }
    #expect(full?.code == "invalid_request")
    advance(599)
    #expect(throws: WorkerFailure.self) { try create("c") }
    advance(1)
    #expect(try create("c")["session_id"]?.string == "c")
    advance(300)
    // "c" is five minutes idle; the two abandoned ones are gone, so one more fits and then the slots are full again.
    #expect(try create("d")["session_id"]?.string == "d")
    #expect(throws: WorkerFailure.self) { try create("e") }
}
