// SPDX-License-Identifier: MIT
// Text-to-image and reference-image edits for the distilled FLUX 3 Image Klein (mflux `_Flux3KleinCore._generate` and
// `Flux3KleinEdit`): Qwen3-VL prompt features, seeded Gaussian latents [1, (H/16)·(W/16), 128] rounded to BF16 and held
// in float32, the 16-step flow-map schedule with one Euler jump per step (no guidance), then the FLUX.2 VAE decoder.
// References are VAE-encoded tokens with their own RoPE layer (−1, −2, …) and attend with time 0.
import Flux2Klein
import Foundation
import MLX
import MLXRandom
import Tokenizers

public struct Flux3Output {
    public let pixels: [UInt8] // RGB8, row-major, after the crop
    public let width: Int, height: Int
    public let latents: MLXArray // final float32 state [1, seq, 128]
}

public struct Flux3Request: Equatable {
    public var prompt: String
    public var width: Int?, height: Int?
    public var steps = Flux3Schedule.defaultSteps
    public var seed: UInt64 = 0
    public var references: [RGBImage] = []
    /// Set by `Flux3Pipeline.plan`: references prepared, size resolved and checked, and the crop back to a single
    /// reference's fitted box (x, y, width, height), if any.
    public fileprivate(set) var planned = false
    public fileprivate(set) var crop: [Int]?
    public init(prompt: String) { self.prompt = prompt }
}

public final class Flux3Pipeline {
    public let transformer: Flux3Transformer
    public let textEncoder: Flux3TextEncoder
    public let tokenizer: (any Tokenizer)?
    public let encoder: Flux3VAEEncoderHalf
    public let decoder: Flux2VAEDecoder
    public let dtype: DType

    public init(
        transformer: Flux3Transformer, textEncoder: Flux3TextEncoder, tokenizer: (any Tokenizer)?, encoder: Flux3VAEEncoderHalf,
        decoder: Flux2VAEDecoder, dtype: DType = .bfloat16
    ) {
        self.transformer = transformer; self.textEncoder = textEncoder; self.tokenizer = tokenizer
        self.encoder = encoder; self.decoder = decoder; self.dtype = dtype
    }

    /// Loads the vendor snapshot in BF16 (the release's precision; the float32 VAE is cast).
    public static func load(snapshot: URL, cancelled: () -> Bool = { false }) throws -> Flux3Pipeline {
        for file in Flux3Weights.requiredFiles where !FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(file).path) {
            throw Flux3Error("The local snapshot is incomplete: \(file); run Get again.")
        }
        let tokenizer = try Flux3Prompt.tokenizer(directory: snapshot.appendingPathComponent("text_encoder"))
        let text = try Flux3Weights.textEncoder(directory: snapshot.appendingPathComponent("text_encoder"))
        if cancelled() { throw CancellationError() }
        let transformer = try Flux3Weights.transformer(directory: snapshot.appendingPathComponent("transformer"))
        if cancelled() { throw CancellationError() }
        let vae = try Flux3Weights.vae(directory: snapshot.appendingPathComponent("vae"))
        return Flux3Pipeline(transformer: transformer, textEncoder: text, tokenizer: tokenizer, encoder: vae.encoder, decoder: vae.decoder)
    }

    /// Seeded noise as mflux draws it: normal(key(seed)) in float32 [1, gh·gw, 128], rounded to BF16, held in float32.
    public static func noise(seed: UInt64, gridH: Int, gridW: Int) -> MLXArray {
        MLXRandom.normal([1, gridH * gridW, 128], dtype: .float32, key: MLXRandom.key(seed)).asType(.bfloat16).asType(.float32)
    }

    /// Prompt → features [1, L, 20480] and its length.
    public func encode(_ prompt: String) throws -> MLXArray {
        guard let tokenizer else { throw Flux3Error("This pipeline has no tokenizer.") }
        let (ids, valid) = Flux3Prompt.tokenize(prompt, tokenizer: tokenizer)
        let features = textEncoder(MLXArray(ids.map(Int32.init), [1, ids.count]), valid: valid)
        eval(features)
        return features
    }

    /// RGB8 → NHWC [1, H, W, 3] in [-1, 1] (u8 / 127.5 − 1), in the pipeline's dtype.
    public func imageTensor(_ image: RGBImage) -> MLXArray {
        (MLXArray(image.pixels, [1, image.height, image.width, 3]).asType(.float32) / 127.5 - 1).asType(dtype)
    }

    /// Reference tokens [1, Σ, 128] and their ids (layer −1, −2, … in order).
    public func referenceTokens(_ references: [RGBImage]) -> (tokens: MLXArray, ids: [[Float]]) {
        var tokens: [MLXArray] = [], ids: [[Float]] = []
        for (i, image) in references.enumerated() {
            let t = encoder.tokens(imageTensor(image), statistics: decoder.statistics)
            eval(t)
            tokens.append(t)
            ids += Flux3Positions.imageIDs(gridH: image.height / 16, gridW: image.width / 16, layer: -(1 + i))
        }
        return (concatenated(tokens, axis: 1), ids)
    }

    /// Denoises `noise` (float32 state) over the flow-map schedule; `step(i, n)` runs after each step and may throw.
    public func denoise(
        noise: MLXArray, text: MLXArray, cond: (tokens: MLXArray, ids: [[Float]])?, gridH: Int, gridW: Int, steps: Int,
        step: (Int, Int) throws -> Void = { _, _ in }
    ) rethrows -> MLXArray {
        let times = Flux3Schedule.timesteps(steps: steps)
        let rope = Flux3RopeTables(
            image: Flux3Positions.imageIDs(gridH: gridH, gridW: gridW), cond: cond?.ids ?? [], text: Flux3Prompt.textIDs(text.dim(1)),
            config: transformer.config, dtype: dtype)
        var x = noise
        for i in 0..<steps {
            let v = transformer(x, text: text, cond: cond?.tokens, t: MLXArray([times[i]]), next: MLXArray([times[i + 1]]), rope: rope)
            // mflux: latents + dt · v with dt a Python float, so dt is rounded to v's BF16 and the product is BF16.
            x += MLXArray(times[i + 1] - times[i]).asType(v.dtype) * v
            eval(x)
            try step(i + 1, steps)
        }
        return x
    }

    /// Final state → NHWC image [1, H, W, 3] in [-1, 1], in the pipeline's dtype.
    public func decode(_ latents: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let image = decoder.decode(decoder.latents(state: latents, gridH: gridH, gridW: gridW, dtype: dtype), dtype: dtype)
        eval(image)
        return image
    }

    /// mflux `ImageUtil`: clip(x / 2 + 0.5, 0, 1) in the image dtype, then ×255 in float32, rounded half to even.
    public static func pixels(_ image: MLXArray) -> [UInt8] {
        let u8 = MLX.round(MLX.clip(image[0] / 2 + 0.5, min: 0, max: 1).asType(.float32) * 255).asType(.uint8)
        eval(u8)
        return u8.asArray(UInt8.self)
    }

    /// Prepares the references, resolves the output size as mflux does and checks it (multiples of 16, sides 64 to
    /// 4096, at most 4 megapixels). Pure host work, so the worker runs it first and refuses a bad request before any
    /// GPU work. Idempotent.
    public static func plan(_ request: Flux3Request) throws -> Flux3Request {
        if request.planned { return request }
        var r = request
        if r.references.isEmpty {
            r.width = r.width ?? 1024; r.height = r.height ?? 1024
        } else {
            let prepared = try Flux3References.prepare(r.references)
            let first = prepared.images[0]
            let resolved = Flux3Canvas.resolve(canvas: (first.width, first.height), box: prepared.box, width: r.width, height: r.height)
            r.references = prepared.images; r.width = resolved.width; r.height = resolved.height
            r.crop = resolved.crop.map { [$0.x, $0.y, $0.width, $0.height] }
        }
        guard let w = r.width, let h = r.height, w % 16 == 0, h % 16 == 0, (64...4096).contains(w), (64...4096).contains(h) else {
            throw Flux3Error("Width and height must be multiples of 16 from 64 to 4096.")
        }
        guard w * h <= Flux3Canvas.matchMaxPixels else {
            throw Flux3Error("The image would be \(w) × \(h), more than 4 megapixels; choose a smaller size.")
        }
        r.planned = true
        return r
    }

    /// One prompt's features and one set of reference tokens are kept, so the images of one request (n > 1) encode
    /// them once.
    private var promptCache: (prompt: String, features: MLXArray)?
    private var referenceCache: (images: [RGBImage], tokens: MLXArray, ids: [[Float]])?

    public func generate(_ request: Flux3Request, step: (Int, Int) throws -> Void = { _, _ in }) throws -> Flux3Output {
        let r = try Self.plan(request)
        let text: MLXArray
        if let cached = promptCache, cached.prompt == r.prompt {
            text = cached.features
        } else {
            text = try encode(r.prompt); promptCache = (r.prompt, text)
        }
        let width = r.width ?? 1024, height = r.height ?? 1024
        var cond: (tokens: MLXArray, ids: [[Float]])?
        if !r.references.isEmpty {
            if let cached = referenceCache, cached.images == r.references {
                cond = (cached.tokens, cached.ids)
            } else {
                let encoded = referenceTokens(r.references)
                referenceCache = (r.references, encoded.tokens, encoded.ids); cond = encoded
            }
        }
        let (gh, gw) = (height / 16, width / 16)
        let latents = try denoise(
            noise: Self.noise(seed: request.seed, gridH: gh, gridW: gw), text: text, cond: cond, gridH: gh, gridW: gw, steps: request.steps, step: step)
        let image = decode(latents, gridH: gh, gridW: gw)
        guard all(isFinite(image)).item(Bool.self) else { throw Flux3Error("Image generation produced nonfinite values; no image was written.") }
        var rgb = RGBImage(width: width, height: height, pixels: Self.pixels(image))
        if let crop = r.crop { rgb = rgb.cropped(x: crop[0], y: crop[1], width: crop[2], height: crop[3]) }
        return Flux3Output(pixels: rgb.pixels, width: rgb.width, height: rgb.height, latents: latents)
    }
}
