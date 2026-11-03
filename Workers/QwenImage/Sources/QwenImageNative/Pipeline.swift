// SPDX-License-Identifier: MIT
// QwenImage21Pipeline.__call__ in Swift: condition images are resized to the output resolution's area, read by the
// prompt encoder (over white) and the VAE (RGBA); the prompt is encoded once, its prefix keys/values cached, and the
// target latents denoised with Euler flow steps before the VAE decodes RGBA. `condition` needs only the prompt encoder
// and `render` only the transformer and VAE, so a caller may hold one stage's weights at a time.
import Foundation
import MLX
import MLXRandom

public struct GenerationRequest {
    public var prompt: String
    public var negativePrompt: String?
    /// True classifier-free guidance; 1 (the default) runs the conditional branch only.
    public var guidance: Float = 1
    public var images: [RGBAImage] = []
    public var width: Int?
    public var height: Int?
    public var steps: Int = 40
    public var seed: UInt64 = 0
    /// Side of the square whose area condition images and the default output size follow (output_resolution).
    public var resolution = 1024
    public init(prompt: String) { self.prompt = prompt }
}

public struct GenerationTimings: Sendable {
    public var encode = 0.0, prefill = 0.0, denoise = 0.0, decode = 0.0
}

/// The prompt encoder's output for one request: everything `render` needs.
public struct Conditioning {
    public struct Encoded { public let states: MLXArray; public let slots: [Bool] }
    public let positive: Encoded
    public let negative: Encoded?
    /// Condition images resized to the request's resolution area (RGBA).
    public let conditions: [RGBAImage]
}

public final class QwenImagePipeline {
    public var encoder: Qwen3VLEncoder?
    public var transformer: QwenImageTransformer?
    public var vae: QwenImageVAE?
    /// Text → token IDs, special tokens recognised, nothing added (the checkpoint tokenizer, or the lab byte tokenizer).
    public let tokenize: (String) -> [Int]
    public let scheduler: SchedulerConfig
    public let dropIndex: Int
    public private(set) var lastTimings = GenerationTimings()

    public init(
        encoder: Qwen3VLEncoder?, transformer: QwenImageTransformer?, vae: QwenImageVAE?, tokenize: @escaping (String) -> [Int], scheduler: SchedulerConfig
    ) {
        self.encoder = encoder; self.transformer = transformer; self.vae = vae; self.tokenize = tokenize; self.scheduler = scheduler
        dropIndex = PromptLayout.dropIndex(tokenize)
    }

    /// Output size: the request's, else the last condition image's aspect at the resolution's area, else square;
    /// floored to a multiple of 32 (two latent tokens).
    public static func outputSize(_ r: GenerationRequest) -> (width: Int, height: Int) {
        var w = r.width, h = r.height
        if let last = r.images.last, w == nil || h == nil {
            let size = ImageSizing.dimensions(area: r.resolution * r.resolution, ratio: Double(last.width) / Double(last.height))
            w = w ?? size.width; h = h ?? size.height
        }
        return ((w ?? r.resolution) / 32 * 32, (h ?? r.resolution) / 32 * 32)
    }

    /// The schedule a request runs: Turbo's saved sigmas, else `steps` shifted for the target token count.
    public func schedule(_ r: GenerationRequest) throws -> FlowSchedule {
        let size = Self.outputSize(r)
        return try FlowSchedule(scheduler, steps: r.steps, imageTokens: (size.width / 16) * (size.height / 16))
    }

    func encode(_ prompt: String, conditions: [RGBAImage], encoder: Qwen3VLEncoder) throws -> Conditioning.Encoded {
        var pixels: [MLXArray] = [], grids: [VisionGrid] = []
        for image in conditions {
            let (p, g) = try VisionPatches.make(rgb: ImageOps.overWhite(image), width: image.width, height: image.height, config: encoder.config.vision)
            pixels.append(p); grids.append(g)
        }
        let merge = encoder.config.vision.merge
        let layout = PromptLayout(prompt: prompt, imageSlots: grids.map { $0.patches / (merge * merge) })
        let tokens = layout.tokens(tokenize, imagePad: encoder.config.imageToken)
        let hidden = try encoder(tokens: tokens, pixels: pixels.isEmpty ? nil : concatenated(pixels, axis: 0), grids: grids)
        let states = hidden[0..., dropIndex..., 0...]
        eval(states)
        return .init(states: states, slots: tokens[dropIndex...].map { $0 == Int32(encoder.config.imageToken) })
    }

    /// Stage 1 (prompt encoder): resize condition images and encode the prompt (and the negative prompt under CFG).
    public func condition(_ r: GenerationRequest) throws -> Conditioning {
        guard let encoder else { throw QwenImageError("The prompt encoder is not loaded") }
        let clock = Date()
        let conditions = try r.images.map { image -> RGBAImage in
            let size = ImageSizing.dimensions(area: r.resolution * r.resolution, ratio: Double(image.width) / Double(image.height))
            return try ImageOps.resize(image, width: size.width, height: size.height)
        }
        let positive = try encode(r.prompt, conditions: conditions, encoder: encoder)
        let negative = r.guidance > 1 && r.negativePrompt != nil ? try encode(r.negativePrompt ?? "", conditions: conditions, encoder: encoder) : nil
        lastTimings.encode = Date().timeIntervalSince(clock)
        return Conditioning(positive: positive, negative: negative, conditions: conditions)
    }

    /// Stage 2 (transformer and VAE): prefill the prefix, denoise, decode. `progress(step, total)` may throw to cancel.
    public func render(_ c: Conditioning, _ r: GenerationRequest, progress: (Int, Int) throws -> Void = { _, _ in }) throws -> RGBAImage {
        guard let transformer, let vae else { throw QwenImageError("The transformer and VAE are not loaded") }
        let (width, height) = Self.outputSize(r)
        guard width >= 32, height >= 32 else { throw QwenImageError("The image must be at least 32 pixels on each side") }
        let schedule = try self.schedule(r)
        var clock = Date()
        let latents = c.conditions.map { image -> MLXArray in
            let z = vae.encode(ImageOps.vaeInput(image)); eval(z)
            return z.reshaped(1, -1, z.dim(3))
        }
        let grids = c.conditions.map { (height: $0.height / 16, width: $0.width / 16) }
        let target = (height: height / 16, width: width / 16)
        let layout = try SequenceLayout(imageSlots: c.positive.slots, conditions: grids, target: target)
        let cache = transformer.prefill(text: c.positive.states, conditions: latents, layout: layout)
        var negativeLayout: SequenceLayout?, negativeCache: PrefixCache?
        if let negative = c.negative {
            let l = try SequenceLayout(imageSlots: negative.slots, conditions: grids, target: target)
            negativeLayout = l; negativeCache = transformer.prefill(text: negative.states, conditions: latents, layout: l)
        }
        lastTimings.prefill = Date().timeIntervalSince(clock); clock = Date()

        var x = MLXRandom.normal([1, target.height * target.width, transformer.config.inChannels], key: MLXRandom.key(r.seed))
        for step in 0..<schedule.steps {
            let t = schedule.timestep(step)
            var v = transformer.denoise(x, t: t, cache: cache, layout: layout).asType(.float32)
            if let negativeCache, let negativeLayout {
                let u = transformer.denoise(x, t: t, cache: negativeCache, layout: negativeLayout).asType(.float32)
                v = u + r.guidance * (v - u)
            }
            // MLXArray += mutates shared storage in place; keep the explicit form.
            // swiftlint:disable:next shorthand_operator
            x = x + schedule.delta(step) * v
            eval(x)
            try progress(step + 1, schedule.steps)
        }
        lastTimings.denoise = Date().timeIntervalSince(clock); clock = Date()

        let decoded = vae.decode(x.reshaped(1, target.height, target.width, transformer.config.outChannels))
        let image = try ImageOps.bytes(decoded)
        lastTimings.decode = Date().timeIntervalSince(clock)
        return image
    }

    /// Generates one RGBA image with every component resident.
    public func generate(_ r: GenerationRequest, progress: (Int, Int) throws -> Void = { _, _ in }) throws -> RGBAImage {
        try render(condition(r), r, progress: progress)
    }
}
