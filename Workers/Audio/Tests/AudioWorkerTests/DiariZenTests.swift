import CryptoKit
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
import Testing
@testable import AudioWorker

/// DiariZen Large v2. Default tests run no model and no GPU work: folder ↔ adapter contract, the maker's powerset and
/// relative-position tables, and every host step of DiariZenPipeline (median filter, counting, clustering, reconstruction)
/// replayed on the maker's own captured values from the golden. Heavy tests convert the pinned checkpoints from the
/// Hugging Face cache and compare the MLX port per component and end to end with the maker's capture.
@Suite struct DiariZenTests {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    static let folder = repo.appendingPathComponent("models/diarizen-large-v2")
    static let evidence = repo.appendingPathComponent("lab/evidence/diarizen-large-v2")
    static let hub = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".cache/huggingface/hub")
    static let snapshot = hub.appendingPathComponent("models--BUT-FIT--diarizen-wavlm-large-s80-md-v2/snapshots/027b3e221b9d81dce2be794084f1dbd3ba2da403")
    static let wespeaker = hub.appendingPathComponent(
        "models--pyannote--wespeaker-voxceleb-resnet34-LM/snapshots/837717ddb9ff5507820346191109dc79c958d614/pytorch_model.bin")
    static let tags = ["fixture-3spk", "EN2002a_30s"]

    static func golden() throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("gate/golden/golden.json"))) as? [String: Any])
    }
    static func file(_ tag: String) throws -> [String: Any] {
        let files = try golden()["files"] as? [String: Any]
        return try #require(files?[tag] as? [String: Any])
    }

    /// [chunk][speaker][[start, end)] frame runs → [chunk][frame][speaker] binary.
    static func binary(_ runs: Any?, frames: Int) throws -> [[[Float]]] {
        let chunks = try #require(runs as? [[[[Int]]]])
        return chunks.map { speakers in
            var out = [[Float]](repeating: [Float](repeating: 0, count: speakers.count), count: frames)
            for (s, list) in speakers.enumerated() { for run in list { for t in run[0]..<run[1] { out[t][s] = 1 } } }
            return out
        }
    }
    static func halfs(_ base64: Any?) throws -> [Float] {
        let text = try #require(base64 as? String)
        let data = try #require(Data(base64Encoded: text))
        return stride(from: 0, to: data.count, by: 2).map { Float(Float16(bitPattern: UInt16(data[$0]) | UInt16(data[$0 + 1]) << 8)) }
    }
    static func segmentation(_ tag: String, filtered: Bool = true) throws -> [[[Float]]] {
        let f = try file(tag)
        return try binary(f[filtered ? "segmentation_runs" : "segmentation_raw_runs"], frames: try #require(f["frames_per_chunk"] as? Int))
    }
    static func embeddings(_ tag: String) throws -> [[[Float]]] {
        let f = try file(tag), values = try halfs(f["embeddings_f16_base64"]), chunks = try #require(f["chunks"] as? Int)
        #expect(values.count == chunks * 4 * 256)
        return (0..<chunks).map { c in (0..<4).map { s in Array(values[((c * 4 + s) * 256)..<((c * 4 + s + 1) * 256)]) } }
    }
    static func timeline(chunks: Int) -> Community1Timeline {
        Community1Timeline(
            chunks: chunks, chunkDuration: 16, chunkStep: 1.6, frameDuration: DiariZenPipeline.frameDuration, frameStep: DiariZenPipeline.frameStep,
            localSpeakers: 4)
    }
    /// Maker segments [start, end, label] with labels renamed the way the adapter names clusters.
    static func makerLines(_ tag: String) throws -> [String] {
        let segments = try #require(try file(tag)["segments_maker"] as? [[Any]])
        let labels = Community1Adapter.labelOrder(Set(segments.compactMap { Int($0[2] as? String ?? "") }))
        let names = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($0.element, String(format: "SPEAKER_%02d", $0.offset)) })
        return segments.map { segment in
            let start = (segment[0] as? NSNumber)?.doubleValue ?? -1, end = (segment[1] as? NSNumber)?.doubleValue ?? -1
            return String(format: "%.3f %.3f %@", start, end, names[Int(segment[2] as? String ?? "") ?? -1] ?? "?")
        }
    }
    static func lines(_ output: JSONObject) -> [String] {
        (output["segments"]?.array ?? []).map {
            String(format: "%.3f %.3f %@", $0["start"]?.number ?? -1, $0["end"]?.number ?? -1, $0["speaker"]?.string ?? "?")
        }
    }

    @Test func folderMatchesAdapterContract() throws {
        let manifest = try String(contentsOf: Self.folder.appendingPathComponent("model.toml"), encoding: .utf8)
        #expect(manifest.contains("architecture = \"diarizen\""))
        #expect(manifest.contains("fp16 = \"\(DiariZenConversion.recipe)\""))
        #expect(manifest.contains("id = \"\(DiariZenConversion.embeddingAsset)\""))
        #expect(manifest.contains("into = \"\(DiariZenConversion.embeddingDirectory)\""))
        #expect(manifest.contains("commercial = false"))
        let registration = try #require(AudioWorker.registrations.first { $0.modelTypes.contains("diarizen") })
        let adapter = try #require(registration.make() as? DiariZenAdapter)
        #expect(adapter.operations == ["diarize"])
        let input = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: Self.folder.appendingPathComponent("ops/diarize.input.json"))) as? [String: Any])
        #expect(input["additionalProperties"] as? Bool == false)
        #expect(Set((input["properties"] as? [String: Any] ?? [:]).keys) == ["audio"])
        let output = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: Self.folder.appendingPathComponent("ops/diarize.output.json"))) as? [String: Any])
        #expect(Set((output["properties"] as? [String: Any] ?? [:]).keys) == ["segments", "exclusive_segments", "speakers", "rttm"])
        #expect(Set(output["required"] as? [String] ?? []) == ["segments", "exclusive_segments", "speakers", "rttm"])
    }

    /// The maker's Powerset(4, 4) mapping, WavLM's relative-position buckets (torch float32 arithmetic) and frame counts.
    @Test func tablesMatchMaker() throws {
        let golden = try Self.golden()
        let powerset = try #require(golden["powerset"] as? [[Int]])
        #expect(powerset.count == 16)
        #expect(DiariZenPipeline.powerset.map { $0.map(Int.init) } == powerset)
        let expected = try #require(golden["relative_buckets_offset_minus798_to_798"] as? [Int])
        #expect(expected.count == 1_597)
        let table = DZAttention.buckets(length: 799)
        // Row 798 holds every offset from -798 (key 0) to 0; row 0 every offset from 0 to 798.
        let fromTable = (0..<799).map { Int(table[798 * 799 + $0]) } + (1..<799).map { Int(table[$0]) }
        #expect(fromTable == expected)
        #expect(DiariZenDimensions.s80.frames(samples: 256_000) == 799)
        #expect(DiariZenPipeline.chunkStarts(439_529) == (0..<9).map { $0 * 25_600 })
        #expect(DiariZenPipeline.chunkStarts(480_000).count == 10)
        #expect(DiariZenPipeline.chunkStarts(100) == [0])
        #expect(golden["min_num_samples"] as? Int == DiariZenPipeline.minimumEmbeddingSamples)
        #expect(golden["fbank_frames_per_window"] as? Int == 1_598)
        let field = try #require(golden["receptive_field"] as? [String: Double])
        #expect(field["duration"] == DiariZenPipeline.frameDuration && field["step"] == DiariZenPipeline.frameStep)
    }

    @Test func medianFilterReproducesMaker() throws {
        for tag in Self.tags {
            let raw = try Self.segmentation(tag, filtered: false), filtered = try Self.segmentation(tag)
            #expect(DiariZenPipeline.medianFilter(raw) == filtered, "\(tag)")
        }
        // scipy "reflect" repeats the edge sample, so the mirrored window gives the first three frames 6 of 11 votes.
        let edge: [[[Float]]] = [[[1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]]
        #expect(DiariZenPipeline.medianFilter(edge)[0].map { $0[0] } == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }

    @Test func speakerCountReproducesMaker() throws {
        for tag in Self.tags {
            let f = try Self.file(tag), segmentation = try Self.segmentation(tag)
            let count = Self.timeline(chunks: segmentation.count).speakerCount(segmentation)
            #expect(count.count == f["count_frames"] as? Int, "\(tag)")
            #expect(count == f["count"] as? [Int], "\(tag)")
        }
    }

    @Test func embeddingMasksExcludeOverlap() {
        // Speaker 0 alone on frames 0…4, with speaker 1 on 3…6; speaker 2 has one clean frame only (≤ 2), so keeps its full mask.
        var binary = [[Float]](repeating: [0, 0, 0, 0], count: 10)
        for t in 0...4 { binary[t][0] = 1 }
        for t in 3...6 { binary[t][1] = 1 }
        binary[8][2] = 1; binary[4][2] = 1
        let masks = DiariZenPipeline.embeddingMasks(binary, samples: 4_000)
        #expect(masks[0] == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        #expect(masks[1] == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
        #expect(masks[2] == [0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        #expect(masks[3] == [Float](repeating: 0, count: 10))
    }

    @Test func assignmentMatchesExhaustiveSearch() {
        var state: UInt64 = 42
        func random() -> Double { state = state &* 6_364_136_223_846_793_005 &+ 1; return Double(state >> 11) / Double(1 << 53) }
        var checked = 0
        for clusters in 1...7 {
            for _ in 0..<20 {
                let scores = (0..<4).map { _ in (0..<clusters).map { _ in random() } }
                let fast = DiariZenAssignment.maximize(scores), exact = Community1Clustering.assign(scores)
                func total(_ a: [Int]) -> Double { a.enumerated().reduce(0) { $0 + ($1.element >= 0 ? scores[$1.offset][$1.element] : 0) } }
                #expect(abs(total(fast) - total(exact)) < 1e-12)
                #expect(fast == exact)
                #expect(fast.filter { $0 >= 0 }.count == min(4, clusters))
                checked += 1
            }
        }
        #expect(checked == 140)
    }

    static var pldaAvailable: Bool { FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("plda/plda.npz").path) }
    static func plda() throws -> Community1PLDA {
        if let metallib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: metallib)
        }
        func vector(_ a: MLXArray) -> [Double] { a.asArray(Float.self).map { Double(Float(Float16($0))) } } // host-side FP16 rounding, no GPU
        let x = try C1Archive(snapshot.appendingPathComponent("plda/xvec_transform.npz").resolvingSymlinksInPath())
        let p = try C1Archive(snapshot.appendingPathComponent("plda/plda.npz").resolvingSymlinksInPath())
        let lda = vector(try x.numpy("lda")), tr = vector(try p.numpy("tr"))
        // The snapshot stores the PLDA tables at half precision; the test reads them the same way.
        return Community1PLDA(
            mean1: vector(try x.numpy("mean1")), mean2: vector(try x.numpy("mean2")), lda: (0..<256).map { Array(lda[($0 * 128)..<(($0 + 1) * 128)]) },
            mu: vector(try p.numpy("mu")), tr: (0..<128).map { Array(tr[($0 * 128)..<(($0 + 1) * 128)]) }, psi: vector(try p.numpy("psi")))
    }

    /// VBxClustering on the maker's embeddings and segmentation: the same training set, AHC start, VBx priors and hard
    /// clusters (after the maker's renumbering), with the PLDA tables the snapshot stores (FP16).
    @Test(.enabled(if: pldaAvailable, "skipped: the pinned DiariZen PLDA files are not in the Hugging Face cache")) func clusteringReproducesMaker() throws {
        let plda = try Self.plda()
        for tag in Self.tags {
            let f = try Self.file(tag)
            let clusters = try DiariZenPipeline.cluster(
                embeddings: try Self.embeddings(tag), segmentations: try Self.segmentation(tag), plda: plda, cancellation: NativeCancellation())
            #expect(clusters.train.map { [$0.chunk, $0.speaker] } == f["train"] as? [[Int]], "\(tag)")
            #expect(clusters.ahc == f["ahc"] as? [Int], "\(tag)")
            let priors = try #require(f["vbx_priors"] as? [Double])
            #expect(clusters.priors.count == priors.count)
            for (a, b) in zip(clusters.priors, priors) { #expect(abs(a - b) < 1e-3, "\(tag) prior \(a) vs \(b)") }
            #expect(clusters.hard == f["hard_clusters_raw"] as? [[Int]], "\(tag)")
        }
    }

    /// Reconstruction and binarisation from the maker's own clusters give the maker's segments.
    @Test func reconstructionReproducesMakerSegments() throws {
        for tag in Self.tags {
            let f = try Self.file(tag), segmentation = try Self.segmentation(tag)
            let hard = try #require(f["hard_clusters_raw"] as? [[Int]]), count = try #require(f["count"] as? [Int])
            let output = DiariZenAdapter.output(
                segmentation: segmentation, hard: hard, count: count.map { min($0, 20) }, timeline: Self.timeline(chunks: segmentation.count), uri: tag)
            let expected = try Self.makerLines(tag), actual = Self.lines(output)
            #expect(actual == expected, "\(tag): \(actual.filter { !expected.contains($0) }) vs \(expected.filter { !actual.contains($0) })")
            #expect(output["speaker_embeddings"] == nil)
            // Bug fixed 2 Nov: times rounded like Python's round(x, 3) on the exact binary value (0.0125 s → 0.013).
            #expect((output["exclusive_segments"]?.array?.count ?? 0) > 0)
            #expect(output["rttm"]?.string?.hasPrefix("SPEAKER \(tag) 1 ") == true)
        }
    }

    /// Bug fixed 2 Nov: the WeSpeaker Lightning checkpoint pickles pyannote's Specifications with NEWOBJ (0x81), which the
    /// reader refused; it now stays an opaque value and nothing is constructed.
    @Test func checkpointReaderKeepsNewObjectsOpaque() throws {
        // {"spec": m.C(), "n": 3} at protocol 2: PROTO, EMPTY_DICT, key, GLOBAL, EMPTY_TUPLE, NEWOBJ, SETITEM, key, BININT1, SETITEM, STOP.
        var bytes: [UInt8] = [0x80, 0x02, 0x7D, 0x58, 4, 0, 0, 0] + Array("spec".utf8) + [0x63] + Array("m\nC\n".utf8) + [0x29, 0x81, 0x73]
        bytes += [0x58, 1, 0, 0, 0] + Array("n".utf8) + [0x4B, 3, 0x73, 0x2E]
        guard case .dict(let box) = try TorchPickle.parse(Data(bytes)) else { Issue.record("not a dictionary"); return }
        #expect(box.items.count == 2)
        #expect(box.items[0].0 == .string("spec") && box.items[0].1 == .opaque("m.C"))
        #expect(box.items[1].0 == .string("n") && box.items[1].1 == .int(3))
    }

    /// The test's DER helper reproduces pyannote.metrics on the maker's own FP16 and FP32-embedding runs (0.1671).
    @Test func diarizationErrorMatchesPyannoteMetrics() throws {
        let golden = try Self.golden()
        let floor = try #require(golden["decision_floor"] as? [String: Any])
        let fp32 = try #require((floor["segments_maker_fp32_embeddings"] as? [String: [[Any]]])?["EN2002a_30s"])
            .map { ($0[0] as? Double ?? 0, $0[1] as? Double ?? 0, $0[2] as? String ?? "") }
        let fp16 = Self.parse(try Self.makerLines("EN2002a_30s"))
        let expected = try #require((floor["der_maker_fp32_embeddings_vs_maker_fp16"] as? [String: Double])?["EN2002a_30s"])
        #expect(abs(Self.diarizationError(reference: fp16, hypothesis: fp32) - expected) < 0.002)
        #expect(Self.diarizationError(reference: fp16, hypothesis: fp16) == 0)
    }

    /// A 0.3 s clip answered with a segment ending at 0.353 s (the zero-padded window's frames); segments now end with the audio.
    @Test func segmentsAreClippedToTheRecording() {
        var frames = [[Float]](repeating: [0, 0, 0, 0], count: 799)
        for t in 0..<18 { frames[t][0] = 1 }
        let output = DiariZenAdapter.output(
            segmentation: [frames], hard: [[0, 1, 2, 3]], count: Self.timeline(chunks: 1).speakerCount([frames]), timeline: Self.timeline(chunks: 1),
            uri: "short", duration: 0.3)
        #expect(Self.lines(output) == ["0.013 0.300 SPEAKER_00"])
        #expect(output["exclusive_segments"]?.array?.first?["end"]?.number == 0.3)
        let late = DiariZenAdapter.output(
            segmentation: [frames], hard: [[0, 1, 2, 3]], count: Self.timeline(chunks: 1).speakerCount([frames]), timeline: Self.timeline(chunks: 1),
            uri: "short", duration: 0.01)
        #expect(Self.lines(late).isEmpty)
    }

    /// Bug fixed 2 Nov: files with more than two channels were downmixed to two before taking channel 0; DiariZen reads
    /// channel 0 itself. A three-channel file whose channels differ must give exactly its first channel.
    @Test(arguments: [16_000, 48_000]) func multichannelFilesUseTheirFirstChannel(rate: Int) throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("diarizen-3ch-" + UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let frames = rate / 10, channels = 3
        var data = Data()
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + frames * channels * 2); data.append(contentsOf: Array("WAVEfmt ".utf8))
        u32(16); u16(1); u16(channels); u32(rate); u32(rate * channels * 2); u16(channels * 2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(frames * channels * 2)
        // Channel 0 is a 400 Hz tone; channels 1 and 2 are large constants that any mixing would leave as an offset.
        let first = (0..<frames).map { Int16((sin(2 * Double.pi * 400 * Double($0) / Double(rate)) * 8_000).rounded()) }
        for i in 0..<frames {
            for value in [first[i], Int16(12_000), Int16(12_000)] { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        }
        try data.write(to: url)
        let samples = try DiariZenAdapter.firstChannel(url)
        #expect(abs(samples.count - 1_600) <= 2, "\(samples.count) samples")
        let mean = samples.reduce(0, +) / Float(max(1, samples.count))
        #expect(abs(mean) < 0.01, "offset \(mean): channels were mixed")
        if rate == 16_000 {
            let worst = zip(samples, first).map { abs(Double($0) - Double($1) / 32_768) }.max() ?? 1
            #expect(worst < 1e-4, "largest difference \(worst)")
        }
    }

    /// Bug (2 Nov 2026 review): files with three or more channels took one unchecked read, so a NaN ran instead of
    /// being refused, and an unreadable file returned 503. Both are the caller's mistakes: 422.
    @Test func multichannelFilesAreCheckedLikeEveryOther() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("diarizen-check-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        // A 3-channel IEEE-float WAV with one NaN in channel 0.
        let frames = 1_600, channels = 3
        var data = Data()
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + frames * channels * 4); data.append(contentsOf: Array("WAVEfmt ".utf8))
        u32(16); u16(3); u16(channels); u32(16_000); u32(16_000 * channels * 4); u16(channels * 4); u16(32)
        data.append(contentsOf: Array("data".utf8)); u32(frames * channels * 4)
        for i in 0..<frames { for c in 0..<channels { u32(Int((c == 0 && i == 800 ? Float.nan : Float(0.1)).bitPattern)) } }
        let nan = directory.appendingPathComponent("nan-3ch.wav")
        try data.write(to: nan)
        let refused = #expect(throws: WorkerFailure.self) { try DiariZenAdapter.firstChannel(nan) }
        #expect(refused?.code == "invalid_request")
        let text = directory.appendingPathComponent("text.wav")
        try Data("not audio".utf8).write(to: text)
        let unreadable = #expect(throws: WorkerFailure.self) { try DiariZenAdapter.firstChannel(text) }
        #expect(unreadable?.code == "invalid_request")
    }

    @Test func conversionNames() throws {
        #expect(try DiariZenConversion.embeddingTarget("resnet.layer2.0.shortcut.0.weight", shape: [64, 32, 1, 1])! == ("embedding.layer2.0.shortcut.weight", [0, 2, 3, 1]))
        #expect(try DiariZenConversion.embeddingTarget("resnet.layer2.0.shortcut.1.running_var", shape: [64])! == ("embedding.layer2.0.shortcutBN.running_var", nil))
        #expect(try DiariZenConversion.embeddingTarget("resnet.bn1.num_batches_tracked", shape: []) == nil)
        #expect(throws: (any Error).self) { try DiariZenConversion.embeddingTarget("head.weight", shape: [1]) }
        #expect(
            try DiariZenConversion.segmentationTarget("conformer.conformer_layer.0.conv.depthwise_conv.weight", shape: [256, 1, 31])! == (
                "segmentation.conformer.conformer_layer.0.conv.depthwise_conv.weight", [0, 2, 1]
            ))
        #expect(
            try DiariZenConversion.segmentationTarget("wavlm_model.encoder.transformer.layers.0.attention.gru_rel_pos_const", shape: [1, 16, 1, 1])! == (
                "segmentation.wavlm_model.encoder.transformer.layers.0.attention.gru_rel_pos_const", nil
            ))
        #expect(throws: (any Error).self) { try DiariZenConversion.segmentationTarget("optimizer.state", shape: [1]) }
        // The pruned layers keep exactly the maker's head lists; four layers lost their attention.
        let d = DiariZenDimensions.s80
        #expect(d.remainingHeads.count == 24 && d.feedForward.count == 24)
        #expect(d.remainingHeads.enumerated().filter { $0.element.isEmpty }.map(\.offset) == [9, 12, 16, 17])
        #expect(d.remainingHeads.allSatisfy { $0 == $0.sorted() && Set($0).count == $0.count && $0.allSatisfy { (0..<16).contains($0) } })
    }

    // MARK: - Heavy: parity with the maker's capture

    static func relativeL2(_ a: [Float], _ b: [Float]) -> Double {
        var num = 0.0, den = 0.0
        for (x, y) in zip(a, b) { num += Double(x - y) * Double(x - y); den += Double(y) * Double(y) }
        return (num / max(den, 1e-30)).squareRoot()
    }
    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, x = 0.0, y = 0.0
        for (p, q) in zip(a, b) { dot += Double(p) * Double(q); x += Double(p) * Double(p); y += Double(q) * Double(q) }
        return dot / max((x * y).squareRoot(), 1e-30)
    }

    static func parse(_ lines: [String]) -> [(Double, Double, String)] {
        lines.map { line in
            let parts = line.split(separator: " ")
            return (Double(parts[0]) ?? 0, Double(parts[1]) ?? 0, String(parts[2]))
        }
    }

    /// pyannote's DER (collar 0, overlap scored) on a 1 ms grid: Σ(max(N_ref, N_hyp) − N_correct) / ΣN_ref under the
    /// speaker mapping that maximises overlap.
    static func diarizationError(reference: [(Double, Double, String)], hypothesis: [(Double, Double, String)]) -> Double {
        let end = (reference + hypothesis).map(\.1).max() ?? 0, steps = Int((end * 1000).rounded(.up)) + 1
        func grid(_ segments: [(Double, Double, String)]) -> ([String], [[Bool]]) {
            let labels = Array(Set(segments.map(\.2))).sorted()
            var active = [[Bool]](repeating: [Bool](repeating: false, count: steps), count: labels.count)
            for (start, stop, label) in segments {
                let row = labels.firstIndex(of: label)!
                for t in Int((start * 1000).rounded())..<max(Int((start * 1000).rounded()), Int((stop * 1000).rounded())) { active[row][t] = true }
            }
            return (labels, active)
        }
        let (refLabels, ref) = grid(reference), (hypLabels, hyp) = grid(hypothesis)
        let overlap = refLabels.indices.map { r in hypLabels.indices.map { h in Double((0..<steps).filter { ref[r][$0] && hyp[h][$0] }.count) } }
        let mapping = DiariZenAssignment.maximize(overlap)
        var errors = 0.0, total = 0.0
        for t in 0..<steps {
            let nRef = ref.filter { $0[t] }.count, nHyp = hyp.filter { $0[t] }.count
            let correct = refLabels.indices.filter { r in mapping[r] >= 0 && ref[r][t] && hyp[mapping[r]][t] }.count
            errors += Double(max(nRef, nHyp) - correct); total += Double(nRef)
        }
        return errors / max(total, 1)
    }

    /// Converts the pinned checkpoints from the Hugging Face cache (APFS clones) with the Get conversion and loads the adapter.
    static func loadedAdapter(_ root: URL) throws -> DiariZenAdapter {
        if let metallib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: metallib)
        }
        let fm = FileManager.default
        let source = root.appendingPathComponent("source"), asset = root.appendingPathComponent("asset"), weights = root.appendingPathComponent("weights")
        for dir in [source.appendingPathComponent("plda"), asset, weights] { try fm.createDirectory(at: dir, withIntermediateDirectories: true) }
        for file in ["pytorch_model.bin", "config.toml", "plda/plda.npz", "plda/xvec_transform.npz"] {
            try fm.copyItem(at: snapshot.appendingPathComponent(file).resolvingSymlinksInPath(), to: source.appendingPathComponent(file))
        }
        try fm.copyItem(at: wespeaker.resolvingSymlinksInPath(), to: asset.appendingPathComponent("pytorch_model.bin"))
        _ = try DiariZenConversion.prepare(
            NativeConversionContext(
                source: source, assets: [DiariZenConversion.embeddingAsset: asset], destination: weights, precision: "fp16", half: .float16),
            NativeCancellation())
        let adapter = DiariZenAdapter()
        try adapter.load(
            NativeLoadContext(["model_dir": .string(folder.path), "weights_dir": .string(weights.path), "dtype": "fp16", "config": ["architecture": "diarizen"]]),
            cancellation: NativeCancellation())
        return adapter
    }

    /// Bug fixed 2 Nov: the port applied the Transformer's layer norm after the positional convolution; the maker's
    /// extract_features path never applies it (relative error 0.98 at WavLM layer 0 before the fix). Layer 0's hidden
    /// state is checked by the heavy parity test; this pins the structure without the GPU.
    @Test func hiddenStatesSkipTheTransformerNorm() throws {
        let source = try String(contentsOf: Self.repo.appendingPathComponent("Workers/Audio/Sources/AudioWorker/DiariZen/DiariZenNetwork.swift"), encoding: .utf8)
        #expect(source.contains("var x = x + positional(x)"))
        #expect(!source.contains("layerNorm(x + positional(x))"))
        // Bug fixed 2 Nov: in-place `+=` on a hidden state that the caller had already stored rewrote that state.
        #expect(!source.contains("x += attention"))
        #expect(source.contains("if let attention { x = x + attention(layerNorm(x), bias: bias) }"))
    }

    /// Op-level probe: MLX's conv1d (grouped and per group) against an explicit convolution built from shifted matrix
    /// products, which shares no code with conv1d, at the positional convolution's geometry and in both half and float.
    @Test(.heavy, arguments: [(16, 64, 128, 40, true), (16, 64, 128, 799, true), (16, 64, 128, 799, false), (2, 8, 4, 12, false), (1, 64, 128, 799, true)])
    func convolutionMatchesShiftedMatmul(groups: Int, perGroup: Int, kernel: Int, length: Int, half: Bool) throws {
        if let metallib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: metallib)
        }
        let channels = groups * perGroup, dtype: DType = half ? .float16 : .float32
        MLXRandom.seed(7)
        let x = MLXRandom.normal([1, length, channels]).asType(dtype), w = (MLXRandom.normal([channels, kernel, perGroup]) * 0.05).asType(dtype)
        let pad = kernel / 2
        let padded = MLX.padded(x.asType(.float32), widths: [0, IntOrPair((pad, pad)), 0])
        let outLength = length + 2 * pad - kernel + 1
        var reference = MLXArray.zeros([1, outLength, channels])
        var parts: [MLXArray] = []
        for g in 0..<groups {
            let xs = padded[0..., 0..., (g * perGroup)..<((g + 1) * perGroup)], ws = w[(g * perGroup)..<((g + 1) * perGroup)].asType(.float32)
            var acc = MLXArray.zeros([1, outLength, perGroup])
            for k in 0..<kernel { acc += matmul(xs[0..., k..<(k + outLength), 0...], ws[0..., k, 0...].transposed()) }
            parts.append(acc)
        }
        reference = concatenated(parts, axis: -1)
        let grouped = conv1d(x, w, padding: pad, groups: groups).asType(.float32)
        let l2 = Self.relativeL2(grouped.asArray(Float.self), reference.asArray(Float.self))
        print("conv1d groups=\(groups) perGroup=\(perGroup) kernel=\(kernel) length=\(length) half=\(half): relative L2 vs shifted matmul \(l2)")
        // Float32 here differs by 7.7e-4 from the float32 matmul reference on M5 (summation order); half by 2e-4.
        #expect(l2 < 2e-3)
    }

    @Test(.heavy) func parityWithMakerCapture() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("diarizen-parity-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root); Memory.clearCache() }
        let adapter = try Self.loadedAdapter(root)
        let network = try #require(adapter.network)
        let golden = try Self.golden()
        let tolerances = try #require(golden["tolerance"] as? [String: Any])
        func bound(_ key: String) -> Double {
            // A missing bound fails the test instead of silently comparing against 0.
            guard let value = (tolerances[key] as? NSNumber)?.doubleValue else {
                Issue.record("golden tolerance lacks \(key)")
                return .nan
            }
            return value
        }
        let reference = try loadArrays(url: Self.evidence.appendingPathComponent("reference-fixture-3spk.safetensors"))
        func ref(_ key: String) throws -> [Float] { try #require(reference[key]).asType(.float32).asArray(Float.self) }
        func values(_ x: MLXArray?) throws -> [Float] { try #require(x).asType(.float32).asArray(Float.self) }
        var report: [String: Any] = [:]

        // Counts before values: the fixture and the window.
        let wav = Self.evidence.appendingPathComponent("fixture-3spk.wav")
        let f = try Self.file("fixture-3spk")
        #expect(SHA256.hash(data: try Data(contentsOf: wav)).map { String(format: "%02x", $0) }.joined() == f["sha256"] as? String)
        let samples = try DiariZenAdapter.firstChannel(wav)
        #expect(samples.count == f["samples"] as? Int)
        let window = Array(samples[0..<256_000])

        // 1. Segmentation network on window 0, per component.
        var trace = DZSegmentation.Trace()
        let logits = try network.segmentation(MLXArray(window, [1, 256_000]), dtype: .float16, trace: &trace, cancellation: NativeCancellation())
        eval(logits)
        func compare(_ name: String, _ ours: [Float], _ key: String) throws {
            let theirs = try ref(key)
            #expect(ours.count == theirs.count, "\(name) count")
            let l2 = Self.relativeL2(ours, theirs), cos = Self.cosine(ours, theirs)
            report[name] = ["relative_l2": l2, "cosine": cos]
            #expect(l2 <= bound(name + "_relative_l2"), "\(name) relative L2 \(l2)")
        }
        try compare("cnn", try values(trace.cnn), "c_cnn")
        try compare("projection", try values(trace.projection), "c_projection")
        for i in [0, 1, 9, 12, 16, 20, 24] { try compare("layer\(i)", try values(trace.states[i]), "c_layer\(i)") }
        try compare("weight_sum", try values(trace.weightSum), "c_weight_sum")
        try compare("proj_lnorm", try values(trace.projected), "c_proj_lnorm")
        for i in 0..<4 { try compare("conformer\(i)", try values(trace.conformer[i]), "c_conformer\(i)") }
        // The port's own window-0 trace, kept beside the maker's for op-level probes.
        var dump: [String: MLXArray] = ["cnn": try #require(trace.cnn), "projection": try #require(trace.projection), "logits": logits]
        for (i, state) in trace.states.enumerated() { dump["layer\(i)"] = state }
        dump["positional_weight"] = network.segmentation.wavlm.encoder.transformer.positional.conv.weight
        dump["positional_bias"] = try #require(network.segmentation.wavlm.encoder.transformer.positional.conv.bias)
        try save(arrays: dump, url: Self.evidence.appendingPathComponent("port-trace-window0.safetensors"))
        let ours = try values(logits), theirs = try ref("c_logits")
        try compare("logits", ours, "c_logits")
        func argmax(_ v: [Float]) -> [Int] { (0..<(v.count / 16)).map { t in (0..<16).max { v[t * 16 + $0] < v[t * 16 + $1] }! } }
        let agree = zip(argmax(ours), argmax(theirs)).filter { $0 == $1 }.count
        report["window0_powerset_argmax_agreement"] = Double(agree) / 799
        #expect(Double(agree) / 799 >= bound("argmax_agreement"))

        // 2. Embedding network on window 0 with the maker's segmentation masks.
        let fbank = try Community1Fbank.features(window)
        try compare("fbank", try values(fbank), "e_fbank")
        let embeddings = try adapter.embed(samples, segmentation: try Self.segmentation("fixture-3spk"), network: network, cancellation: NativeCancellation())
        let maker = try Self.embeddings("fixture-3spk")
        var worst = 1.0
        for (c, chunk) in embeddings.enumerated() { for (s, e) in chunk.enumerated() { worst = min(worst, Self.cosine(e, maker[c][s])) } }
        report["embedding_min_cosine"] = worst
        #expect(worst >= bound("embedding_min_cosine"))

        // 3. Per stage on both files: our segmentation, count, embeddings and clusters against the maker's.
        for tag in Self.tags {
            let entry = try Self.file(tag)
            let name = try #require((entry["file"] as? String)?.split(separator: "/").last.map(String.init))
            let audio = try DiariZenAdapter.firstChannel(Self.evidence.appendingPathComponent(name))
            let raw = try adapter.segment(audio, network: network, cancellation: NativeCancellation())
            let makerRaw = try Self.segmentation(tag, filtered: false), makerFiltered = try Self.segmentation(tag)
            #expect(raw.count == makerRaw.count)
            let flat = raw.flatMap { $0.flatMap { $0 } }, makerFlat = makerRaw.flatMap { $0.flatMap { $0 } }
            report["segmentation_raw_value_agreement_" + tag] = Double(zip(flat, makerFlat).filter { $0 == $1 }.count) / Double(max(1, flat.count))
            let filtered = DiariZenPipeline.medianFilter(raw)
            let ff = filtered.flatMap { $0.flatMap { $0 } }, mf = makerFiltered.flatMap { $0.flatMap { $0 } }
            report["segmentation_filtered_value_agreement_" + tag] = Double(zip(ff, mf).filter { $0 == $1 }.count) / Double(max(1, ff.count))
            let count = Self.timeline(chunks: filtered.count).speakerCount(filtered), makerCount = try #require(entry["count"] as? [Int])
            report["count_agreement_" + tag] = Double(zip(count, makerCount).filter { $0 == $1 }.count) / Double(max(1, count.count))
            #expect(raw.count == makerRaw.count && count.count == makerCount.count, "\(tag) counts")
            for key in ["segmentation_raw_value_agreement_", "segmentation_filtered_value_agreement_", "count_agreement_"] {
                let value = report[key + tag] as? Double ?? 0
                #expect(value >= bound(key + tag), "\(tag) \(key) \(value)")
            }
            // Embeddings from our audio with the maker's segmentation (embedding network alone).
            let ours = try adapter.embed(audio, segmentation: makerFiltered, network: network, cancellation: NativeCancellation())
            let maker = try Self.embeddings(tag)
            var low = 1.0
            for (c, chunk) in ours.enumerated() { for (sIndex, e) in chunk.enumerated() { low = min(low, Self.cosine(e, maker[c][sIndex])) } }
            report["embedding_min_cosine_" + tag] = low
            #expect(low >= bound("embedding_min_cosine"), "\(tag)")
            // Clusters from our embeddings on the maker's segmentation.
            let clusters = try DiariZenPipeline.cluster(
                embeddings: ours, segmentations: makerFiltered, plda: network.plda.value(), cancellation: NativeCancellation())
            let sameClusters = clusters.hard == (entry["hard_clusters_raw"] as? [[Int]] ?? [])
            report["hard_clusters_equal_on_maker_segmentation_" + tag] = sameClusters
            // Exact on the fixture; on the AMI clip VBx flips with embedding rounding (step 4 bounds the effect).
            if tag == "fixture-3spk" { #expect(sameClusters, "\(tag) clusters") }
        }

        // 4. End to end. The fixture must give exactly the maker's segments. On the AMI clip VBx flips with embedding
        // rounding alone (the maker's own FP16 and FP32-embedding runs differ), so the port must lie within that floor
        // of one of the two maker runs.
        let floor = try #require(golden["decision_floor"] as? [String: Any])
        let floorSegments = try #require(floor["segments_maker_fp32_embeddings"] as? [String: [[Any]]])
        for tag in Self.tags {
            let entry = try Self.file(tag)
            let name = try #require((entry["file"] as? String)?.split(separator: "/").last.map(String.init))
            let path = Self.evidence.appendingPathComponent(name)
            // Parity on the unclipped segments (the maker's pipeline does not crop to the audio length).
            let output = try adapter.diarize(path: path.path, clip: false, cancellation: NativeCancellation())
            let lines = Self.lines(output), expected = try Self.makerLines(tag)
            // The served result is the same with each segment ended at the recording's length.
            let served = try adapter.run("diarize", input: ["audio": ["path": .string(path.path)]], cancellation: NativeCancellation(), emit: { _ in })
            let duration = Double(try #require(entry["samples"] as? Int)) / 16_000
            let trimmed = Self.parse(lines).filter { $0.0 < duration }.map { String(format: "%.3f %.3f %@", $0.0, min($0.1, (duration * 1000).rounded() / 1000), $0.2) }
            #expect(Self.lines(served) == trimmed, "\(tag) served vs trimmed")
            report["served_segments_" + tag] = Self.lines(served)
            report["segments_" + tag] = lines
            report["segments_equal_" + tag] = lines == expected
            let ours = Self.parse(lines)
            let fp16 = Self.parse(expected), fp32 = try #require(floorSegments[tag]).map { ($0[0] as? Double ?? 0, $0[1] as? Double ?? 0, $0[2] as? String ?? "") }
            let der16 = Self.diarizationError(reference: fp16, hypothesis: ours), der32 = Self.diarizationError(reference: fp32, hypothesis: ours)
            report["der_vs_maker_fp16_" + tag] = der16
            report["der_vs_maker_fp32_embeddings_" + tag] = der32
            if tag == "fixture-3spk" {
                #expect(lines == expected, "\(tag)")
            } else {
                #expect(min(der16, der32) <= bound("end_to_end_der_bound_" + tag) + 1e-9, "\(tag) DER \(der16) / \(der32)")
            }
        }
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: Self.evidence.appendingPathComponent("parity-port.json"))
    }
}
