// SPDX-License-Identifier: MIT
// Whole-graph checks on small random weights: the prefix cache reproduces the reference single-pass block-causal
// forward, and every component runs end to end at the shapes the pipeline expects.
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Testing
@testable import QwenImageNative

@Suite struct GraphTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    /// QwenImage21Transformer2DModel.forward without a cache: one joint sequence, the mask (q ≥ k) or same image block,
    /// prefix tokens modulated from t = 0 and target tokens from t, norm_out per token.
    func referenceForward(_ m: QwenImageTransformer, text: MLXArray, conditions: [MLXArray], layout: SequenceLayout, latents: MLXArray, t: Float) -> MLXArray {
        let c = m.config
        let projected = m.textIn(text)
        var pieces: [MLXArray] = [], block: [Int] = [], blockID = 0
        for run in layout.runs {
            switch run {
            case .text(let range): pieces.append(projected[0..., range, 0...]); block += Array(repeating: -1, count: range.count)
            case .image(let index, let h, let w): pieces.append(m.imageIn(conditions[index])); block += Array(repeating: blockID, count: h * w); blockID += 1
            case .target(let h, let w): pieces.append(m.imageIn(latents)); block += Array(repeating: blockID, count: h * w)
            }
        }
        var x = concatenated(pieces, axis: 1)
        let l = x.dim(1), p = layout.prefixLength
        var mask = [Bool]()
        for i in 0..<l { for j in 0..<l { mask.append(j <= i || (block[i] >= 0 && block[i] == block[j])) } }
        let maskArray = MLXArray(mask, [l, l])
        let zero = DiTModulation(m.modulationRow(m.temb(0)), hidden: c.hidden), real = DiTModulation(m.modulationRow(m.temb(t)), hidden: c.hidden)
        func rows(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            concatenated([broadcast(a, to: [1, p, c.hidden]), broadcast(b, to: [1, l - p, c.hidden])], axis: 1)
        }
        let s1 = rows(zero.scale1, real.scale1), g1 = rows(zero.gate1, real.gate1), s2 = rows(zero.scale2, real.scale2), g2 = rows(zero.gate2, real.gate2)
        let rope = DiTRope.table(layout.positions, axes: c.axes)
        for b in m.blocks {
            let (q, k, v) = b.attn.qkv(b.norm(x) * s1, rope: rope)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 1 / Float(c.headDim).squareRoot(), mask: .array(maskArray))
            let h = x + g1 * b.attn.output(attended)
            x = DiTBlock.clipHalf(h + g2 * b.mlp(b.norm(h) * s2))
        }
        let target = x[0..., p..., 0...]
        return m.projOut(m.normOut(target, temb: m.temb(t)))
    }

    @Test(.heavy) func prefixCacheReproducesTheSinglePassForward() throws {
        MLXRandom.seed(5)
        let m = QwenImageTransformer(SmallConfigs.transformer)
        // Two condition images with text around them, then a 4 × 6 target.
        let slots = [false, false, false] + Array(repeating: true, count: 4) + [false, false] + Array(repeating: true, count: 2) + [false, false]
        let layout = try SequenceLayout(imageSlots: slots, conditions: [(4, 4), (2, 4)], target: (4, 6))
        let text = MLXRandom.normal([1, slots.count, 64])
        let conditions = [MLXRandom.normal([1, 16, 64]), MLXRandom.normal([1, 8, 64])]
        let latents = MLXRandom.normal([1, 24, 64])
        let cache = m.prefill(text: text, conditions: conditions, layout: layout)
        #expect(cache.length == layout.prefixLength && cache.keys.count == 2)
        for t: Float in [1000, 437.5] {
            let cached = m.denoise(latents, t: t, cache: cache, layout: layout)
            let reference = referenceForward(m, text: text, conditions: conditions, layout: layout, latents: latents, t: t)
            #expect(cached.shape == [1, 24, 64])
            let diff = abs(cached - reference).max().item(Float.self), scale = abs(reference).max().item(Float.self)
            #expect(diff <= 1e-4 * max(1, scale), "max |Δ| \(diff) at t=\(t)")
        }
    }

    @Test(.heavy) func vaeRoundTripShapes() {
        MLXRandom.seed(1)
        let vae = QwenImageVAE(SmallConfigs.vae)
        let image = MLXRandom.uniform(low: -1, high: 1, [1, 64, 96, 4])
        let z = vae.encode(image)
        #expect(z.shape == [1, 4, 6, 64])
        let decoded = vae.decode(z)
        #expect(decoded.shape == [1, 64, 96, 4])
        #expect(abs(decoded).max().item(Float.self) <= 1)
        #expect(MLX.all(MLX.isFinite(decoded)).item(Bool.self))
    }

    @Test(.heavy) func encoderReadsImagesThroughDeepstack() throws {
        MLXRandom.seed(2)
        let config = SmallConfigs.encoder
        let encoder = Qwen3VLEncoder(config)
        let bytes = ByteTokens(config: config)
        // A 64 × 96 image: 4 × 6 patches of 16 px → 2 × 3 merged slots.
        let rgb = (0..<(64 * 96 * 3)).map { UInt8($0 % 256) }
        let (pixels, grid) = try VisionPatches.make(rgb: rgb, width: 96, height: 64, config: config.vision)
        let layout = PromptLayout(prompt: "edit", imageSlots: [grid.patches / 4])
        let tokens = layout.tokens(bytes.encode, imagePad: config.imageToken)
        #expect(tokens.filter { $0 == Int32(config.imageToken) }.count == 6)
        let withImage = try encoder(tokens: tokens, pixels: pixels, grids: [grid])
        #expect(withImage.shape == [1, tokens.count, 64])
        // The vision features change the states of the text that follows the image (causal attention).
        let blank = try encoder(tokens: tokens, pixels: MLXArray.zeros(like: pixels), grids: [grid])
        let after = tokens.count - 1
        #expect(abs(withImage[0, after] - blank[0, after]).max().item(Float.self) > 0)
        // A wrong slot count is refused rather than mismatched.
        var missing = tokens
        missing.remove(at: missing.firstIndex(of: Int32(config.imageToken))!)
        #expect(throws: QwenImageError.self) { _ = try encoder(tokens: missing, pixels: pixels, grids: [grid]) }
    }

    @Test(.heavy) func pipelineGeneratesAndEditsOnRandomWeights() throws {
        MLXRandom.seed(3)
        let config = SmallConfigs.encoder, bytes = ByteTokens(config: config)
        let pipeline = QwenImagePipeline(
            encoder: Qwen3VLEncoder(config), transformer: QwenImageTransformer(SmallConfigs.transformer), vae: QwenImageVAE(SmallConfigs.vae),
            tokenize: bytes.encode, scheduler: SchedulerConfig())
        var r = GenerationRequest(prompt: "a red square")
        r.width = 64; r.height = 96; r.steps = 2; r.resolution = 64
        var steps: [Int] = []
        let image = try pipeline.generate(r) { step, total in
            steps.append(step); #expect(total == 2)
        }
        #expect(steps == [1, 2] && image.width == 64 && image.height == 96)
        // Edit: one RGBA reference; the output follows its aspect ratio at the resolution's area.
        r.width = nil; r.height = nil
        r.images = [try RGBAImage(width: 40, height: 20, bytes: (0..<(40 * 20 * 4)).map { UInt8(($0 * 7) % 256) })]
        r.resolution = 64
        let edited = try pipeline.generate(r)
        #expect(edited.width == 96 && edited.height == 32)
        // CFG doubles the branches and still returns one image.
        r.guidance = 2; r.negativePrompt = "blurry"
        #expect(try pipeline.generate(r).width == 96)
    }
}
