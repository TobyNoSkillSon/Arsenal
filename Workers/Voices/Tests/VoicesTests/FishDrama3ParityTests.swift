// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing
@testable import FishS1Codec
@testable import FishDrama3

/// Fixed-tensor parity of the native Drama 3 port against tensors captured once from the reference
/// (lab/drama3-port/scripts/capture_parity.py). Loads the real 24 GB checkpoint: heavy, opt-in, GPU.
///   DRAMA3_PARITY_DIR=<capture dir> DRAMA3_SNAPSHOT=<snapshot dir> scripts/test.sh --heavy --filter FishDrama3ParityTests
@Suite(.serialized) struct FishDrama3ParityTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) } }

    static let environment = ProcessInfo.processInfo.environment
    static var configured: Bool { environment["DRAMA3_PARITY_DIR"] != nil && environment["DRAMA3_SNAPSHOT"] != nil }

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a = a.asType(.float32).flattened(), b = b.asType(.float32).flattened()
        return ((a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))).item(Float.self)
    }
    nonisolated(unsafe) static var saved: [String: MLXArray] = [:]
    /// Results go to <capture>/native-report.txt as well as stdout: the test runner keeps stdout only for failures.
    static func note(_ line: String) {
        print("drama3-parity " + line)
        guard let dir = environment["DRAMA3_PARITY_DIR"] else { return }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("native-report.txt")
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd(); try? handle.write(contentsOf: Data((line + "\n").utf8)); try? handle.close()
        }
    }
    static func report(_ name: String, _ a: MLXArray, _ b: MLXArray) -> (cosine: Float, maxAbs: Float) {
        let a32 = a.asType(.float32), b32 = b.asType(.float32)
        let result = (cosine(a, b), abs(a32 - b32).max().item(Float.self))
        let relative = (sqrt(((a32 - b32) * (a32 - b32)).sum()) / sqrt((b32 * b32).sum())).item(Float.self)
        note("\(name): shape \(a.shape) cosine \(result.0) max|d| \(result.1) rel-L2 \(relative)")
        saved[name.replacingOccurrences(of: " ", with: "_")] = a32
        return result
    }

    /// Static layer replay: each backbone layer runs on the reference's own input to that layer, so its error is its own.
    @Test(.heavy, .enabled(if: FishDrama3ParityTests.configured)) func backboneLayersMatchOnReferenceInputs() async throws {
        let capture = URL(fileURLWithPath: Self.environment["DRAMA3_PARITY_DIR"]!)
        let file = capture.appendingPathComponent("drama3-layers.safetensors")
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        let ref = try MLX.loadArrays(url: file)
        #expect(ref.count == 40 * 3 + 1)
        let speech = try await Drama3Speech.load(directory: URL(fileURLWithPath: Self.environment["DRAMA3_SNAPSHOT"]!))
        let model = try #require(speech.model)
        let caches = model.backbone.makeCache()
        for (index, layer) in model.backbone.layers.enumerated() {
            let name = String(format: "layer_%02d", index)
            let input = try #require(ref[name + "_in"]).asType(.bfloat16)
            let normed = layer.inputLayerNorm(input)
            let mixer =
                layer.isLinear
                ? layer.linearAttn!(normed, mask: nil, cache: caches[index] as? MambaCache)
                : layer.selfAttn!(normed, mask: .causal, cache: caches[index])
            let refMixer = try #require(ref[name + "_mixer"])
            let mid = input + refMixer.asType(.bfloat16)
            let mlp = layer.mlp(layer.postAttentionLayerNorm(mid))
            eval(mixer, mlp)
            _ = Self.report(name + (layer.isLinear ? " gdn" : " attention"), mixer, refMixer)
            _ = Self.report(name + " moe", mlp, try #require(ref[name + "_mlp"]))
        }
        await speech.unload()
    }

    /// Static sampler check on saved reference logits (CPU): empirical draw frequencies, compared offline with the
    /// reference sampler's exact distribution.
    @Test(.heavy, .enabled(if: FishDrama3ParityTests.configured)) func samplerDrawsMatchTheReferenceDistribution() throws {
        let capture = URL(fileURLWithPath: Self.environment["DRAMA3_PARITY_DIR"]!)
        let file = capture.appendingPathComponent("drama3-decode.safetensors")
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        try Device.withDefaultDevice(.cpu) {
            let logits = try #require(try MLX.loadArrays(url: file, stream: .cpu)["forced_b_main"])[0..., 0]
            MLXRandom.seed(123)
            var counts: [[String: Int]] = []
            for row in 0..<logits.dim(0) {
                let tiled = repeated(logits[row].expandedDimensions(axis: 0), count: 4000, axis: 0)
                let draws = drama3SampleTopKTopP(tiled, temperature: 0.7, topK: 30, topP: 0.7).asArray(UInt32.self)
                counts.append(Dictionary(draws.map { (String($0), 1) }, uniquingKeysWith: +))
            }
            let data = try JSONSerialization.data(withJSONObject: counts, options: [.sortedKeys])
            try data.write(to: capture.appendingPathComponent("native-sampler-counts.json"))
            #expect(counts.count == logits.dim(0))
        }
    }

    /// Static decode replay: both implementations are forced along the same codes, so every step's logits are comparable.
    @Test(.heavy, .enabled(if: FishDrama3ParityTests.configured)) func teacherForcedDecodeMatchesTheReference() async throws {
        let capture = URL(fileURLWithPath: Self.environment["DRAMA3_PARITY_DIR"]!)
        let file = capture.appendingPathComponent("drama3-decode.safetensors")
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        let ref = try MLX.loadArrays(url: file)
        let parity = try MLX.loadArrays(url: capture.appendingPathComponent("drama3-parity.safetensors"))
        let speech = try await Drama3Speech.load(directory: URL(fileURLWithPath: Self.environment["DRAMA3_SNAPSHOT"]!))
        let model = try #require(speech.model)
        func forced(_ frames: MLXArray, _ codes: MLXArray, final: Bool = false) -> (MLXArray, MLXArray) {
            let caches = (backbone: model.backbone.makeCache(), tower: model.tower.makeCache())
            var hidden = model.prefill(frames, caches: caches, start: 0, stop: frames.dim(0))
            var previous = frames[(-1)...].expandedDimensions(axis: 0)
            var mains: [MLXArray] = [], towers: [MLXArray] = []
            for n in 0..<codes.dim(0) {
                mains.append(model.network.semanticHead(hidden[0..., 0]).asType(.float32))
                let row = codes[n].asArray(Int32.self)
                let main = MLXArray([row[0] + Int32(model.config.semanticBegin)])
                var heads: [MLXArray] = [], step = 0
                _ = model.tower.residualCodes(
                    hidden, c0: MLXArray([row[0]]).reshaped(1, 1), semantic: model.tower.isSemantic(main).expandedDimensions(axis: 1), frame: previous,
                    caches: caches.tower,
                    sample: { logits in
                        heads.append(logits[0..., 0].asType(.float32)); step += 1
                        return MLXArray([row[step]]).reshaped(1, 1)
                    })
                towers.append(concatenated(heads, axis: 0))
                let next = concatenated([main, MLXArray(row)]).asType(.int32).reshaped(1, 1, -1)
                hidden = model.backbone(embeddings: model.embed(next), cache: caches.backbone)
                previous = next
                eval(hidden, mains[mains.count - 1], towers[towers.count - 1])
            }
            if final { mains.append(model.network.semanticHead(hidden[0..., 0]).asType(.float32)) }
            return (stacked(mains), stacked(towers))
        }
        for (label, frames, codes) in [
            ("A", try #require(parity["prompt_frames"]), try #require(parity["greedy_codes"])),
            ("B", try #require(ref["unprompted_frames"]), try #require(ref["unprompted_codes"]))
        ] {
            let (mains, towers) = forced(frames, codes)
            let refMains = try #require(ref["forced_\(label.lowercased())_main"]), refTowers = try #require(ref["forced_\(label.lowercased())_tower"])
            #expect(mains.shape == refMains.shape && towers.shape == refTowers.shape)
            _ = Self.report("forced \(label) semantic logits", mains, refMains)
            _ = Self.report("forced \(label) tower logits", towers, refTowers)
            for n in 0..<mains.dim(0) {
                let a = mains[n], b = refMains[n]
                let rel = (sqrt(((a - b) * (a - b)).sum()) / sqrt((b * b).sum())).item(Float.self)
                let same = a.argMax(axis: -1).item(Int.self) == b.argMax(axis: -1).item(Int.self)
                let towerSame = (towers[n].argMax(axis: -1) .== refTowers[n].argMax(axis: -1)).asType(.float32).mean().item(Float.self)
                Self.note("forced \(label) step \(n): semantic rel-L2 \(rel) argmax \(same ? "same" : "DIFF") tower argmax agreement \(towerSame)")
            }
        }
        // C: P(EOS) along a reference run that ended by itself (unprompted voice, reference sampling).
        let eosFile = capture.appendingPathComponent("drama3-eos.safetensors")
        if FileManager.default.fileExists(atPath: eosFile.path) {
            let eos = try MLX.loadArrays(url: eosFile)
            let (mains, _) = forced(try #require(eos["eos_frames"]), try #require(eos["eos_codes"]), final: true)
            let refMains = try #require(eos["eos_main"])
            #expect(mains.shape == refMains.shape)
            _ = Self.report("forced C semantic logits", mains, refMains)
            let mine = softmax(mains[0..., 0], axis: -1)[0..., -1].asArray(Float.self)
            let theirs = softmax(refMains[0..., 0], axis: -1)[0..., -1].asArray(Float.self)
            Self.note("forced C P(eos) native \(mine.suffix(6).map { String(format: "%.4f", $0) }) reference \(theirs.suffix(6).map { String(format: "%.4f", $0) })")
            Self.note("forced C mean P(eos) native \(mine.reduce(0, +) / Float(mine.count)) reference \(theirs.reduce(0, +) / Float(theirs.count))")
        }
        await speech.unload()
    }

    @Test(.heavy, .enabled(if: FishDrama3ParityTests.configured)) func nativePortMatchesTheReference() async throws {
        let capture = URL(fileURLWithPath: Self.environment["DRAMA3_PARITY_DIR"]!)
        let snapshot = URL(fileURLWithPath: Self.environment["DRAMA3_SNAPSHOT"]!)
        let ref = try MLX.loadArrays(url: capture.appendingPathComponent("drama3-parity.safetensors"))
        let meta = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: capture.appendingPathComponent("drama3-parity.json"))) as? [String: Any])
        // Counts first.
        #expect(ref.count == 10)
        let speech = try await Drama3Speech.load(directory: snapshot)
        let model = try #require(speech.model)
        let codec = try #require(model.codec)

        // Codec encode of the fixed reference: codes are discrete, so compare the agreement rate.
        let samples = try #require(ref["reference_samples"]).asArray(Float.self)
        let codes = try codec.encodeCodes(samples)
        let refCodes = try #require(ref["reference_codes"])
        #expect(codes.shape == refCodes.shape)
        let agreement = (codes .== refCodes).asType(.float32).mean().item(Float.self)
        Self.note("reference codes agreement (FP16 codec) \(agreement)")
        // FP16 flips near-ties of the VQ argmax; the same port at the reference's float32 (test only) separates precision from
        // a port error, which would scatter codes almost completely.
        #expect(agreement >= 0.9)
        let exact = try FishS1Codec.load(url: snapshot.appendingPathComponent("codec.safetensors"), dtype: .float32)
        let exactAgreement = (exact.encode(samples) .== refCodes).asType(.float32).mean().item(Float.self)
        Self.note("reference codes agreement (float32 codec, test only) \(exactAgreement)")
        #expect(exactAgreement >= 0.98)

        // Prompt rows from the reference's codes and the native tokenizer must equal the reference's rows exactly.
        let prompt = try Drama3PromptBuilder.build(
            references: [(try #require(meta["reference_text"] as? String), Drama3Model.hostCodes(refCodes))], history: [],
            target: try #require(meta["target"] as? String), tokenizer: try #require(model.tokenizer), codebooks: 10, vocab: 4096,
            semanticBegin: model.config.semanticBegin)
        let refFrames = try #require(ref["prompt_frames"])
        #expect(prompt.count == refFrames.dim(0) && prompt.systemRows == meta["system_rows"] as? Int)
        let frames = MLXArray(prompt.values, [prompt.count, prompt.width])
        #expect((frames .== refFrames).all().item(Bool.self))

        // Backbone embeddings and the prefilled hidden state of every prompt row.
        let embeddings = model.embed(frames[..<64].expandedDimensions(axis: 0))
        #expect(Self.report("embeddings", embeddings, try #require(ref["embeddings_64"])).cosine > 0.9999)
        let caches = (backbone: model.backbone.makeCache(), tower: model.tower.makeCache())
        var rows: [MLXArray] = []
        for begin in stride(from: 0, to: prompt.count, by: Drama3Defaults.prefillChunk) {
            let end = min(begin + Drama3Defaults.prefillChunk, prompt.count), labelled = min(end, prompt.count - 1)
            let h = model.backbone(embeddings: model.embed(frames[begin..<end].expandedDimensions(axis: 0)), cache: caches.backbone)
            if labelled > begin {
                model.tower.prefill(
                    h[0..., ..<(labelled - begin)], frames: frames[begin..<labelled].expandedDimensions(axis: 0),
                    nextFrames: frames[(begin + 1)..<(labelled + 1)].expandedDimensions(axis: 0), caches: caches.tower)
            }
            eval([h] + caches.backbone.flatMap(\.state) + caches.tower.flatMap(\.state))
            rows.append(h)
        }
        let hidden = concatenated(rows, axis: 1)
        #expect(Self.report("prompt hidden", hidden, try #require(ref["prompt_hidden"])).cosine > 0.999)
        let logits = model.network.semanticHead(hidden[0..., -1]).asType(.float32)
        let refLogits = try #require(ref["head_logits"])
        _ = Self.report("semantic head logits", logits, refLogits)
        #expect(logits.argMax(axis: -1).item(Int.self) == refLogits.argMax(axis: -1).item(Int.self))

        // Tower, teacher-forced on the reference's first audio frame.
        let forced = refCodes[0..., 0].asArray(Int32.self)
        let main = MLXArray([forced[0] + Int32(model.config.semanticBegin)])
        var heads: [MLXArray] = [], step = 0
        _ = model.tower.residualCodes(
            hidden[0..., (-1)...], c0: MLXArray([forced[0]]).reshaped(1, 1), semantic: model.tower.isSemantic(main).expandedDimensions(axis: 1),
            frame: frames[(-1)...].expandedDimensions(axis: 0), caches: caches.tower,
            sample: { logits in
                heads.append(logits[0..., 0].asType(.float32)); step += 1
                return MLXArray([forced[step]]).reshaped(1, 1)
            })
        let towerLogits = concatenated(heads, axis: 0), refTower = try #require(ref["tower_logits"])
        #expect(Self.report("tower logits", towerLogits, refTower).cosine > 0.999)
        #expect((towerLogits.argMax(axis: -1) .== refTower.argMax(axis: -1)).all().item(Bool.self))

        // Greedy frames (temperature at its floor): every reference answer must be kept.
        let refGreedy = try #require(ref["greedy_codes"])
        let voice = try model.voice(references: [(samples, try #require(meta["reference_text"] as? String))])
        var greedy: [MLXArray] = []
        MLXRandom.seed(0) // the repetition-aware retry draws even at the temperature floor; same key order as the reference
        let floor = Drama3Sampling(temperature: 0, topK: 30, topP: 0.7, maxFrames: 16)
        try await model.frames(prompt, voice: voice, sampling: floor, isCancelled: { false }, emit: { greedy.append($0) })
        #expect(greedy.count == refGreedy.dim(0))
        if !greedy.isEmpty, greedy.count == refGreedy.dim(0) {
            let mine = stacked(greedy)
            Self.note("greedy code agreement \((mine .== refGreedy).asType(.float32).mean().item(Float.self))")
            // The first frame is fully determined; later frames compound bf16 kernel differences, so they are reported, not gated.
            #expect((mine[0] .== refGreedy[0]).all().item(Bool.self), "first greedy frame")
        }

        // Codec decode (FP16 here, float32 in the reference): whole and streamed in 3 + 5 frames.
        let eight = refCodes[0..., ..<8].expandedDimensions(axis: 0)
        codec.resetStreaming()
        let streamed = concatenated([codec.decodeStep(eight[.ellipsis, ..<3]), codec.decodeStep(eight[.ellipsis, 3...])])
        #expect(Self.report("codec streamed", streamed, try #require(ref["codec_streamed"])).cosine > 0.999)
        _ = Self.report("codec whole vs reference streamed", try #require(ref["codec_whole"]), try #require(ref["codec_streamed"]))
        if Self.environment["DRAMA3_PARITY_SAVE"] == "1" {
            Self.saved["native_codes"] = codes.asType(.int32)
            if !greedy.isEmpty { Self.saved["native_greedy"] = stacked(greedy).asType(.int32) }
            try MLX.save(arrays: Self.saved, url: capture.appendingPathComponent("native-outputs.safetensors"))
        }
        await speech.unload()
    }
}
