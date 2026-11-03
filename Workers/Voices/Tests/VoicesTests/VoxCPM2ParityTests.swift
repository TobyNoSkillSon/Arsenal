// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import Testing
import Tokenizers
@testable import VoxCPM2

/// Fixed-tensor parity of the VoxCPM2 port against tensors captured once from the maker's code (OpenBMB/VoxCPM on MPS,
/// float16 throughout): every component on the maker's own inputs, then a teacher-forced replay of a whole generation.
/// Loads the real 4.6 GB checkpoint (cast to FP16 in memory): heavy, opt-in, GPU.
///   VOXCPM2_PARITY_DIR=<capture dir> VOXCPM2_SNAPSHOT=<mlx-community snapshot> scripts/test.sh --heavy --filter VoxCPM2ParityTests
@Suite(.serialized) struct VoxCPM2ParityTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) } }

    static let environment = ProcessInfo.processInfo.environment
    static var configured: Bool { environment["VOXCPM2_PARITY_DIR"] != nil && environment["VOXCPM2_SNAPSHOT"] != nil }

    /// Tolerances, set before the first comparison (2 Nov 2026). Relative L2 against the maker's FP16 values.
    enum Tolerance {
        static let vaeEncode: Float = 0.02
        static let vaeDecode: Float = 0.05
        static let hidden: Float = 0.02 // LocEnc, prefill LMs, DiT velocity
        static let cfm: Float = 0.05 // ten Euler steps on fixed noise
        static let teacherForcedHidden: Float = 0.05 // worst step of the replay
        static let finalAudio: Float = 0.05
    }

    static func note(_ line: String) {
        print("voxcpm2-parity " + line)
        guard let dir = environment["VOXCPM2_PARITY_DIR"] else { return }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("native-report.txt")
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd(); try? handle.write(contentsOf: Data((line + "\n").utf8)); try? handle.close()
        }
    }

    /// Relative L2 and cosine of `a` against the reference `b`, logged.
    @discardableResult static func compare(_ name: String, _ a: MLXArray, _ b: MLXArray) -> Float {
        let a32 = a.asType(.float32).reshaped(-1), b32 = b.asType(.float32).reshaped(-1)
        let rel = (sqrt(((a32 - b32) * (a32 - b32)).sum()) / sqrt((b32 * b32).sum())).item(Float.self)
        let cos = ((a32 * b32).sum() / (sqrt((a32 * a32).sum()) * sqrt((b32 * b32).sum()))).item(Float.self)
        let maxAbs = abs(a32 - b32).max().item(Float.self)
        note("\(name): shape \(a.shape) rel-L2 \(rel) cosine \(cos) max|d| \(maxAbs)")
        return rel
    }

    nonisolated(unsafe) static var cached: (VoxCPM2Speech, [String: MLXArray], [String: Any])?
    static func setup() async throws -> (VoxCPM2Speech, [String: MLXArray], [String: Any]) {
        if let cached { return cached }
        let dir = URL(fileURLWithPath: environment["VOXCPM2_PARITY_DIR"]!)
        let ref = try MLX.loadArrays(url: dir.appendingPathComponent("ref-maker-fp16.safetensors"))
        let meta = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("ref-maker-fp16.json"))) as? [String: Any])
        let speech = try await VoxCPM2Speech.load(directory: URL(fileURLWithPath: environment["VOXCPM2_SNAPSHOT"]!))
        cached = (speech, ref, meta)
        return (speech, ref, meta)
    }

    @Test(.heavy, .enabled(if: VoxCPM2ParityTests.configured)) func tokensMatchTheMaker() async throws {
        let (speech, _, meta) = try await Self.setup()
        let tokens = try #require(meta["tokens"] as? [String: [Int]]), texts = try #require(meta["texts"] as? [String: String])
        #expect(tokens.count == 5)
        for (name, expected) in tokens.sorted(by: { $0.key < $1.key }) {
            let text = name == "pl_prompt" ? texts["pl_ref"]! + texts["pl"]! : texts[name]!
            let got = VoxCPM2Model.tokenize(text, tokenizer: speech.tokenizer)
            Self.note("tokens \(name): \(got.count) of \(expected.count), identical \(got == expected)")
            #expect(got == expected, "tokens \(name)")
        }
        // Fixed bug: continuation prompts tokenize transcript and target as one string, as the maker does; two token
        // lists joined differ at the seam ("książki.Dzień").
        let separate = VoxCPM2Model.tokenize(texts["pl_ref"]!, tokenizer: speech.tokenizer) + VoxCPM2Model.tokenize(texts["pl"]!, tokenizer: speech.tokenizer)
        Self.note("tokens pl_prompt tokenized separately: \(separate.count), equal to the maker's joint ids \(separate == tokens["pl_prompt"]!)")
        #expect(separate != tokens["pl_prompt"]!)
    }

    @Test(.heavy, .enabled(if: VoxCPM2ParityTests.configured)) func audioVAEMatchesTheMaker() async throws {
        let (speech, ref, _) = try await Self.setup()
        let model = try #require(speech.model)
        for side in ["right", "left"] {
            let input = try #require(ref["vae_enc_in_\(side)"]).transposed(0, 2, 1) // (1, N, 1)
            let feat = model.encodePadded(input)
            let expected = try #require(ref[side == "right" ? "ref_feat" : "prompt_feat"])
            #expect(feat.shape == expected.shape)
            #expect(Self.compare("vae encode (\(side) pad)", feat, expected) < Tolerance.vaeEncode)
        }
        let latent = try #require(ref["vae_enc_out_right"]).transposed(0, 2, 1) // (1, T', 64)
        let audio = model.audioVAE.decode(latent)
        let expected = try #require(ref["vae_dec_out"])
        #expect(audio.size == expected.size)
        #expect(Self.compare("vae decode", audio, expected) < Tolerance.vaeDecode)
    }

    /// The prompt for ultimate cloning, built by the port from the maker's tokens and features, equals the maker's.
    @Test(.heavy, .enabled(if: VoxCPM2ParityTests.configured)) func promptAndPrefillMatchTheMaker() async throws {
        let (speech, ref, meta) = try await Self.setup()
        let model = try #require(speech.model)
        let tokens = try #require(meta["tokens"] as? [String: [Int]])
        let p = model.prompt(
            text: tokens["pl_prompt"]!, promptFeat: try #require(ref["prompt_feat"]), refFeat: try #require(ref["ref_feat"]))
        #expect(p.tokens.map(Float.init) == ref["in_tokens"]!.asArray(Float.self))
        #expect(p.textMask == ref["in_text_mask"]!.asArray(Float.self))
        #expect(p.audioMask == ref["in_audio_mask"]!.asArray(Float.self))
        #expect(p.contextPatches == 3)
        #expect(Self.compare("prompt feats", p.feats, ref["in_feat"]![0]) < 1e-6)
        #expect(Self.compare("locenc", model.featEncoder(ref["in_feat"]!), ref["locenc_out"]!) < Tolerance.hidden)
        let state = model.prefill(p)
        let (raw, _) = model.baseLM(ref["combined_embed"]!.asType(model.dtype), causal: true)
        #expect(Self.compare("base lm prefill (maker input)", raw, ref["base_out_raw"]!) < Tolerance.hidden)
        #expect(Self.compare("base lm last hidden after fsq", state.lmHidden, ref["enc_outputs"]![0..., -1, 0...]) < Tolerance.hidden)
        #expect(Self.compare("residual lm last hidden", state.resHidden, ref["res_out"]![0..., -1, 0...]) < Tolerance.hidden)
    }

    @Test(.heavy, .enabled(if: VoxCPM2ParityTests.configured)) func ditAndFlowMatchingMatchOnFixedNoise() async throws {
        let (speech, ref, _) = try await Self.setup()
        let model = try #require(speech.model)
        let span = VoxUnifiedCFM.timeSpan(steps: 10)
        #expect(Self.compare("time span", MLXArray(span), ref["cfm_t_span"]!) < 1e-3)
        let z = ref["cfm_z"]!, mu = ref["cfm_mu"]!, cond = ref["cfm_cond"]!
        let v = model.featDecoder.estimator(
            MLX.concatenated([z, z]), mu: MLX.concatenated([mu, MLXArray.zeros(like: mu)]), t: ref["dit_t"]!, cond: MLX.concatenated([cond, cond]),
            dt: MLXArray.zeros([2]))
        #expect(Self.compare("dit velocity (cond + uncond)", v, ref["dit_v"]!) < Tolerance.hidden)
        let out = model.featDecoder.sample(mu: mu, steps: 10, patchSize: model.patchSize, cond: cond, cfgValue: 2, noise: z)
        #expect(Self.compare("cfm 10 steps cfg 2", out, ref["cfm_out"]!) < Tolerance.cfm)
    }

    /// Replays the maker's generation: every step's hidden states and stop decision from the maker's own patches, then
    /// the 48 kHz decode of the maker's latents with the context patches cut.
    @Test(.heavy, .enabled(if: VoxCPM2ParityTests.configured)) func teacherForcedGenerationMatchesTheMaker() async throws {
        let (speech, ref, meta) = try await Self.setup()
        let model = try #require(speech.model)
        let tokens = try #require(meta["tokens"] as? [String: [Int]])
        let p = model.prompt(text: tokens["pl_prompt"]!, promptFeat: ref["prompt_feat"]!, refFeat: ref["ref_feat"]!)
        let preds = ref["steps_pred"]!, steps = preds.dim(0)
        #expect(steps == (meta["ar"] as? [String: Any])?["steps"] as? Int)
        var state = model.prefill(p)
        var worst: [String: Float] = ["lm": 0, "res": 0, "mu": 0]
        var stopAgree = 0
        for i in 0..<steps {
            let lm = Self.relative(state.lmHidden, ref["steps_lm_hidden"]![i]), res = Self.relative(state.resHidden, ref["steps_res_hidden"]![i])
            let mu = Self.relative(model.ditConditioning(state), ref["steps_mu"]![i])
            worst["lm"] = max(worst["lm"]!, lm); worst["res"] = max(worst["res"]!, res); worst["mu"] = max(worst["mu"]!, mu)
            let stop = model.stopLogits(state).argMax(axis: -1).item(Int.self)
            if stop == ref["steps_stop_logits"]![i].argMax().item(Int.self) { stopAgree += 1 }
            state = model.step(state, patch: preds[i].expandedDimensions(axis: 0))
            eval(state.lmHidden, state.resHidden)
        }
        Self.note(
            "teacher-forced \(steps) steps: worst rel-L2 lm \(worst["lm"]!) residual \(worst["res"]!) dit-conditioning \(worst["mu"]!); stop decisions agree \(stopAgree)/\(steps)")
        #expect(stopAgree == steps)
        #expect(worst.values.allSatisfy { $0 < Tolerance.teacherForcedHidden })
        let context = p.feats[(p.feats.dim(0) - p.contextPatches)...]
        let audio = model.decode(context: context, patches: (0..<steps).map { preds[$0].expandedDimensions(axis: 0) })
        let full = ref["final_audio_full"]!.reshaped(-1)
        let skip = full.size - audio.count
        #expect(skip == 3 * model.patchSize * model.audioVAE.decodeChunkSize)
        #expect(Self.compare("final 48 kHz audio (maker latents)", MLXArray(audio), full[skip...]) < Tolerance.finalAudio)
    }

    /// FP16 range of the AudioVAE decoder on a long chunk (review, 2 Nov): the maker's 4.5 s latent tiled to about 60 s,
    /// decoded stage by stage in FP16; every stage stays finite and under half of FP16's range, and the audio matches an
    /// FP32 copy of the same decoder (test-only reference) within 2%. Bounds set before the run.
    @Test(.heavy, .enabled(if: VoxCPM2ParityTests.configured)) func audioDecoderStaysInFP16RangeOnALongChunk() async throws {
        let (speech, ref, _) = try await Self.setup()
        let model = try #require(speech.model)
        let short = try #require(ref["final_latent"]).transposed(0, 2, 1) // (1, T, 64)
        let latent = MLX.concatenated(Array(repeating: short, count: 1500 / short.dim(1) + 1), axis: 1)[0..., ..<1500, 0...]
        func stages(_ vae: VoxAudioVAE, _ z: MLXArray) -> (MLXArray, [Float]) {
            let decoder = vae.decoder
            var peaks: [Float] = []
            func record(_ h: MLXArray) { peaks.append(MLX.abs(h.asType(.float32)).max().item(Float.self)) }
            var h = (decoder.convIn as? VoxSequential).map { $0(z.asType(vae.dtype)) } ?? z
            record(h)
            let index = decoder.getSRIdx(MLXArray([Int32(vae.outSampleRate)]))
            for (block, cond) in zip(decoder.blocks, decoder.srCondLayers) {
                if let cond { h = cond(h, srCond: index); record(h) }
                h = block(h); eval(h); record(h)
            }
            h = decoder.convOut(decoder.snakeOut(h)); record(h)
            return (MLX.tanh(h).reshaped(-1), peaks)
        }
        let (audio16, peaks) = stages(model.audioVAE, latent)
        eval(audio16)
        Self.note("long-chunk decoder FP16: \(latent.dim(1)) frames, \(audio16.size) samples, stage peaks \(peaks.map { Int($0.rounded()) })")
        #expect(audio16.size == 1500 * model.audioVAE.decodeChunkSize)
        #expect(peaks.allSatisfy { $0.isFinite && $0 < VoxCPM2Cast.fp16Max / 2 })
        let vae32 = VoxAudioVAE(model.config.audioVaeConfig)
        try vae32.update(parameters: ModuleParameters.unflattened(model.audioVAE.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }), verify: [])
        vae32.decoder.srBoundaries = model.audioVAE.decoder.srBoundaries
        let (audio32, _) = stages(vae32, latent)
        #expect(Self.compare("long-chunk decoder FP16 vs FP32 (60 s)", audio16, audio32) < 0.02)
    }

    static func relative(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a32 = a.asType(.float32).reshaped(-1), b32 = b.asType(.float32).reshaped(-1)
        return (sqrt(((a32 - b32) * (a32 - b32)).sum()) / sqrt((b32 * b32).sum())).item(Float.self)
    }
}
