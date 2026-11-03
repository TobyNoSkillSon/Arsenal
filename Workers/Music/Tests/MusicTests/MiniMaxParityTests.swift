// SPDX-License-Identifier: MIT
import Foundation
import Testing
import IntegrationKit
import ArsenalWire
import ModelPrecision
import MLX
import MLXNN
@testable import MusicWorker

/// MiniMax Music 3 against the maker's diffusers modules, captured once on PyTorch MPS at BF16 (the maker's documented
/// dtype) and FP16, on the fixed inputs in `reference.safetensors`. Needs the prepared FP16 snapshot and the capture:
/// `MINIMAX_SNAPSHOT=<Get snapshot> MINIMAX_REFERENCE=<capture dir> scripts/test.sh --heavy --filter MiniMaxParityTests`.
///
/// Frozen before the first comparison (2 Nov 2026): for every component, Arsenal's FP16 output against the maker's BF16
/// output has relative L2 at most twice the maker's own FP16-vs-BF16 difference plus 0.001, and every output is
/// finite. The language model's guided semantic choice must keep the maker's top code and at least 45 of its top 50.
@Suite(.serialized) struct MiniMaxParityTests {
    static let env = ProcessInfo.processInfo.environment
    static let ready =
        env["MINIMAX_SNAPSHOT"] != nil && env["MINIMAX_REFERENCE"] != nil
        && FileManager.default.fileExists(atPath: (env["MINIMAX_REFERENCE"] ?? "") + "/chunks.safetensors")
    static let chunksReady = ready
    static let sourceReady = env["MINIMAX_SNAPSHOT"] != nil && env["MINIMAX_SOURCE"] != nil
    private var repo: URL { (0..<5).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() } }

    struct Metric: Encodable {
        let relL2Bf16: Double, cosineBf16: Double, maxAbsBf16: Double
        let relL2Fp16: Double?, floor: Double, floorSource: String, bound: Double, finite: Bool, count: Int
        var pass: Bool { finite && relL2Bf16 <= bound }
    }
    static func stats(_ a: MLXArray, _ b: MLXArray) -> (Double, Double, Double) {
        let x = a.asType(.float32).reshaped(-1), y = b.asType(.float32).reshaped(-1)
        let d = x - y
        let rel = (MLX.sqrt((d * d).sum()) / MLX.maximum(MLX.sqrt((y * y).sum()), MLXArray(Float(1e-12)))).item(Float.self)
        let cos = ((x * y).sum() / MLX.maximum(MLX.sqrt((x * x).sum()) * MLX.sqrt((y * y).sum()), MLXArray(Float(1e-12)))).item(Float.self)
        return (Double(rel), Double(cos), Double(MLX.abs(d).max().item(Float.self)))
    }
    /// The floor is the maker's FP16 output against its BF16 output. Where the maker's FP16 output is not finite (its vocoder:
    /// the snake's 1e-9 vanishes in FP16), the floor is the shared FP32-source policy's BF16 nearest against BF16 truncation.
    static func metric(_ ours: MLXArray, bf16: MLXArray, fp16: MLXArray, truncated: MLXArray? = nil) -> Metric {
        precondition(ours.shape == bf16.shape && bf16.shape == fp16.shape, "\(ours.shape) \(bf16.shape) \(fp16.shape)")
        let makerFp16Finite = MLX.all(MLX.isFinite(fp16)).item(Bool.self)
        let floorRef = makerFp16Finite ? fp16 : (truncated ?? fp16)
        let (rel, cos, maxAbs) = stats(ours, bf16), (relFp16, _, _) = stats(ours, fp16), (floor, _, _) = stats(floorRef, bf16)
        let finite = MLX.all(MLX.isFinite(ours.asType(.float32))).item(Bool.self)
        return Metric(
            relL2Bf16: rel, cosineBf16: cos, maxAbsBf16: maxAbs, relL2Fp16: makerFp16Finite ? relFp16 : nil, floor: floor,
            floorSource: makerFp16Finite ? "maker fp16" : (truncated == nil ? "maker fp16 (not finite)" : "maker bf16 truncation"),
            bound: 2 * floor + 0.001, finite: finite, count: ours.size)
    }

    static func load() throws -> MiniMaxAdapter {
        let adapter = MiniMaxAdapter()
        let repo = (0..<5).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() }
        try adapter.load(
            NativeLoadContext([
                "weights_dir": .string(env["MINIMAX_SNAPSHOT"]!), "model_dir": .string(repo.appendingPathComponent("models/minimax-music3").path),
                "dtype": "fp16", "config": ["architecture": "minimax_music3"]
            ]), cancellation: NativeCancellation())
        return adapter
    }
    static func write(_ name: String, _ value: some Encodable) throws {
        guard let out = env["MINIMAX_PARITY_OUT"] else { return }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        try encoder.encode(value).write(to: URL(fileURLWithPath: out).deletingLastPathComponent().appendingPathComponent(name))
    }

    /// Windows of 200 and 175 frames with overlap carry and crop/stitch, on the maker's own blocks and noise (8 steps), and the
    /// fusion encoder at 64 frames: both lengths where torch's nearest resampling differs from exact arithmetic.
    @Test(.heavy, .enabled(if: chunksReady)) func windowSplicingMatchesTheMakersBlocks() throws {
        try MiniMaxTests.useMetallib()
        let dir = URL(fileURLWithPath: Self.env["MINIMAX_REFERENCE"]!)
        let reference = try TensorArchive(dir.appendingPathComponent("chunks.safetensors"))
        func ref(_ key: String) throws -> MLXArray { try reference.array(key) }
        let adapter = try Self.load()
        let p = try #require(adapter.pipeline)
        var metrics: [String: Metric] = [:]
        let frames = try ref("fixture.frame_hiddens").asType(.float16)
        #expect(frames.shape == [1, 275, 8 * p.c.hidden] && MiniMaxPipeline.chunkStarts(275) == [0, 100])
        metrics["condition_64"] = Self.metric(
            p.condition(try ref("fixture.frame_hiddens_64").asType(.float16)), bf16: try ref("bf16.condition_64"), fp16: try ref("fp16.condition_64"))
        let noises = try (0..<2).map { try ref("fixture.noise_\($0)").transposed(0, 2, 1) }
        let chunks = try p.latents(
            frameHidden: frames, steps: 8, guidance: 1.7, key: MLXRandom.key(0), cancelled: { false }, progress: { _ in },
            noise: { k, length in
                precondition(noises[k].dim(1) == length, "noise \(k): \(noises[k].shape) for \(length) latents")
                return noises[k]
            })
        #expect(chunks.map { $0.dim(1) } == [p.c.latentLength(200), p.c.latentLength(175)])
        for (k, chunk) in chunks.enumerated() {
            metrics["chunk_\(k)"] = Self.metric(chunk.transposed(0, 2, 1), bf16: try ref("bf16.chunk_\(k)"), fp16: try ref("fp16.chunk_\(k)"))
        }
        let audio = try p.decode(chunks, cancelled: { false })
        let stitched = stacked([MLXArray(audio.left), MLXArray(audio.right)], axis: 0)[.newAxis]
        metrics["stitched_audio"] = Self.metric(stitched, bf16: try ref("bf16.audios"), fp16: try ref("fp16.audios"), truncated: try ref("bf16trunc.audios"))
        #expect(metrics.count == 4)
        for (name, m) in metrics.sorted(by: { $0.key < $1.key }) { #expect(m.pass, "\(name): \(m)") }
        try Self.write("parity-chunks.json", metrics)
        adapter.unload()
    }

    /// Get folds the vocoder's weight normalization after the FP16 source normalization. Each stored kernel must lie within
    /// twice the error of a single FP16 rounding of the kernel folded in float32 from the raw FP32 source.
    @Test(.heavy, .enabled(if: sourceReady)) func storedVocoderKernelsMatchAFloat32Fold() throws {
        try MiniMaxTests.useMetallib()
        let source = try TensorArchive(URL(fileURLWithPath: Self.env["MINIMAX_SOURCE"]!).appendingPathComponent("vocoder/diffusion_pytorch_model.safetensors"))
        let stored = try TensorArchive(URL(fileURLWithPath: Self.env["MINIMAX_SNAPSHOT"]!).appendingPathComponent("vocoder/diffusion_pytorch_model.safetensors"))
        struct Fold: Encodable { let stored: Double; let singleRounding: Double; let pass: Bool }
        var folds: [String: Fold] = [:]
        for key in source.entries.keys.sorted() where key.hasSuffix(".weight_v") {
            let gain = try source.array(String(key.dropLast(2)) + "_g").asType(.float32), v = try source.array(key).asType(.float32)
            let norm = MLX.sqrt((v.reshaped(v.dim(0), -1) * v.reshaped(v.dim(0), -1)).sum(axis: 1)).reshaped(gain.shape)
            let folded = MusicConversion.miniMaxValue(component: "vocoder", key: key, gain * v / norm)
            let kept = try stored.array(String(key.dropLast(9)) + ".weight")
            #expect(kept.dtype == .float16 && kept.shape == folded.shape, "\(key)")
            let (rel, _, _) = Self.stats(kept, folded), (floor, _, _) = Self.stats(folded.asType(.float16), folded)
            folds[key] = Fold(stored: rel, singleRounding: floor, pass: rel <= 2 * floor + 1e-6)
        }
        #expect(folds.count == 30)
        for (key, fold) in folds.sorted(by: { $0.key < $1.key }) { #expect(fold.pass, "\(key): \(fold.stored) vs \(fold.singleRounding)") }
        try Self.write("parity-vocoder-fold.json", folds)
    }

    @Test(.heavy, .enabled(if: ready)) func componentsMatchTheMakersCapture() throws {
        try MiniMaxTests.useMetallib()
        let referenceDir = URL(fileURLWithPath: Self.env["MINIMAX_REFERENCE"]!)
        let reference = try TensorArchive(referenceDir.appendingPathComponent("reference.safetensors"))
        let meta = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: referenceDir.appendingPathComponent("reference.json"))) as? [String: Any])
        func ref(_ key: String) throws -> MLXArray { try reference.array(key) }
        let adapter = try Self.load()
        let p = try #require(adapter.pipeline)
        var metrics: [String: Metric] = [:]
        func check(_ name: String, _ ours: MLXArray, _ key: String) throws {
            metrics[name] = Self.metric(ours, bf16: try ref("bf16." + key), fp16: try ref("fp16." + key))
        }

        // Counts before values: the prompt tokens and teacher-forced codes are the maker's exactly.
        let caption = try #require(meta["caption"] as? String), lyrics = try #require(meta["lyrics"] as? String)
        let tokens = try p.tokens(caption: caption, lyrics: lyrics)
        let expectedTokens = try #require(meta["text_ids"] as? [[Int]])
        #expect(tokens.shape == [2, expectedTokens[0].count] && tokens.asArray(Int32.self).map(Int.init) == expectedTokens.flatMap { $0 })
        let forced = try #require(meta["forced_codes"] as? [[Int]]).map { $0.map(Int32.init) }
        #expect(forced.count == 6 && forced.allSatisfy { $0.count == 8 })

        // Global model prefill and its guided semantic choice.
        let cache = p.lm.makeCache()
        let hidden = p.lm.hidden(p.lm.model.embedTokens(tokens), cache: cache)
        let last = hidden[0..., -1, 0...]
        try check("prefill_last_hidden", last, "prefill_last_hidden")
        try check("prefill_hidden_tail", hidden[0..., (-8)..., 0...], "prefill_hidden_tail")
        let logits = matmul(last, p.audioHead.T).asType(.float32)
        try check("prefill_audio_logits", logits, "prefill_audio_logits")
        let makerLogits = try ref("bf16.prefill_audio_logits")
        func guidedTop(_ l: MLXArray) -> [Int32] {
            let c = l[0..<1], u = l[1..<2]
            let g = u + (c - u) * MiniMaxPipeline.arGuidance
            return argSort(-g[0]).asType(.int32)[0..<50].asArray(Int32.self)
        }
        let oursTop = guidedTop(logits), makerTop = guidedTop(makerLogits)
        let overlap = Set(oursTop).intersection(makerTop).count
        #expect(oursTop[0] == makerTop[0] && overlap >= 45, "top \(oursTop[0]) vs \(makerTop[0]), overlap \(overlap)")

        // Teacher-forced frames: global and local hidden states per frame, and the local model's guided logits.
        p.depthLogitsTrace = []
        let ar = try p.frames(tokens: tokens, maxFrames: 6, seed: 0, forced: forced)
        #expect(ar.codes.map { $0 } == forced)
        try check("frame_hiddens", ar.hidden, "frame_hiddens")
        try check("depth_guided_logits", concatenated(p.depthLogitsTrace ?? [], axis: 0), "depth_guided_logits")
        p.depthLogitsTrace = nil

        // Fusion, flow transformer, one full window and the vocoder, each on the maker's BF16 inputs.
        try check("condition", p.condition(try ref("bf16.frame_hiddens").asType(.float16)), "condition")
        let noise = try ref("fixture.noise").transposed(0, 2, 1).asType(.float16)
        let condition = try ref("bf16.condition").asType(.float16)
        for t in ["0.0", "0.5", "0.9"] {
            let time = MLXArray([Float(t)!]).asType(.float16)
            try check("velocity_t" + t, p.transformer(noise, time: time, condition: condition).transposed(0, 2, 1), "velocity_t" + t)
            try check(
                "velocity_uncond_t" + t, p.transformer(noise, time: time, condition: MLXArray.zeros(like: condition)).transposed(0, 2, 1),
                "velocity_uncond_t" + t)
        }
        let window = try p.denoise(
            condition: condition, noise: noise, steps: 30, guidance: 1.7, previousLatent: nil, previousCondition: nil, cancelled: { false }, progress: { _ in })
        try check("window_latents", window.latents.transposed(0, 2, 1), "window_latents")
        let chunks = try TensorArchive(referenceDir.appendingPathComponent("chunks.safetensors"))
        let wave = p.vocoder(try ref("bf16.window_latents").transposed(0, 2, 1).asType(.float16))
        metrics["waveform"] = Self.metric(wave, bf16: try ref("bf16.waveform"), fp16: try ref("fp16.waveform"), truncated: try chunks.array("bf16trunc.waveform"))
        // Op-level probe (frozen before it ran): the maker's own vocoder at FP16 with only its snake in float32, as Arsenal
        // computes it, settles the vocoder despite BF16's wide rounding floor. Relative L2 at most 0.05.
        let probe = try TensorArchive(referenceDir.appendingPathComponent("vocoder-fp16fix.safetensors")).array("fp16fix.waveform")
        let (probeRel, probeCos, _) = Self.stats(wave, probe)
        #expect(probeRel <= 0.05, "vocoder against the maker's FP16 vocoder with a float32 snake: \(probeRel)")
        try Self.write("parity-vocoder-probe.json", ["relL2": probeRel, "cosine": probeCos, "bound": 0.05])

        #expect(metrics.count == 14)
        for (name, m) in metrics.sorted(by: { $0.key < $1.key }) { #expect(m.pass, "\(name): \(m)") }
        if let out = Self.env["MINIMAX_PARITY_OUT"] {
            struct Report: Encodable {
                let date: String, policy: String, topCode: [Int32], top50Overlap: Int, metrics: [String: Metric]
                let peakBytes: Int
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
            try encoder.encode(
                Report(
                    date: "2026-11-02", policy: "relL2(arsenal fp16, maker bf16) <= 2*relL2(maker fp16, maker bf16) + 0.001; finite; top code kept, top-50 overlap >= 45",
                    topCode: [oursTop[0], makerTop[0]], top50Overlap: overlap, metrics: metrics, peakBytes: Memory.peakMemory)
            ).write(to: URL(fileURLWithPath: out))
        }
        adapter.unload()
    }
}
