// SPDX-License-Identifier: MIT
// Text-to-image for the distilled FLUX.2 Klein 4B, following the vendor's diffusers Flux2KleinPipeline (0.41):
// Qwen3 conditioning, Gaussian latents [1, 128, H/16, W/16] packed to [1, seq, 128], the FlowMatchEuler schedule with
// the empirical mu, one transformer pass per step (the distilled model ignores guidance), then the VAE.
import Foundation
import MLX
import MLXRandom

public struct KleinPrecision: Sendable, Equatable, CustomStringConvertible {
    /// Weights of the text encoder and transformer (the vendor ships bf16).
    public var weights: DType = .bfloat16
    /// The text encoder's activations.
    public var encoder: DType = .bfloat16
    /// The transformer's residual stream.
    public var activations: DType = .bfloat16
    /// VAE weights and activations.
    public var vae: DType = .bfloat16
    public init() {}
    public init(weights: DType, encoder: DType, activations: DType, vae: DType) {
        self.weights = weights; self.encoder = encoder; self.activations = activations; self.vae = vae
    }
    public var description: String {
        "weights \(weights) · encoder \(encoder) · stream \(activations) · vae \(vae)"
    }
}

public enum KleinSchedule {
    /// diffusers `compute_empirical_mu`.
    public static func empiricalMu(seqLen: Int, steps: Int) -> Double {
        let (a1, b1, a2, b2) = (8.73809524e-05, 1.89833333, 0.00016927, 0.45666666)
        if seqLen > 4300 { return a2 * Double(seqLen) + b2 }
        let m200 = a2 * Double(seqLen) + b2, m10 = a1 * Double(seqLen) + b1
        let a = (m200 - m10) / 190.0
        return a * Double(steps) + (m200 - 200.0 * a)
    }
    /// sigmas = linspace(1, 1/N, N) as float32, exponential time shift by mu (float32 arithmetic as NumPy does it), a
    /// terminal 0 appended; timesteps = sigma · 1000.
    public static func sigmas(seqLen: Int, steps: Int) -> [Float] {
        let e = Float(Foundation.exp(empiricalMu(seqLen: seqLen, steps: steps)))
        var out = (0..<steps).map { i -> Float in
            // np.linspace: start + i·step, with the endpoint set exactly
            let s = Float(steps == 1 ? 1.0 : i == steps - 1 ? 1.0 / Double(steps) : 1.0 + Double(i) * ((1.0 / Double(steps) - 1.0) / Double(steps - 1)))
            return e / (e + (1 / s - 1))
        }
        out.append(0)
        return out
    }
}

public struct KleinTimings: Sendable {
    public var encode: Double = 0, steps: [Double] = [], decode: Double = 0, pixels: Double = 0, total: Double = 0
    public init() {}
}

public struct KleinOutput {
    public let pixels: [UInt8] // RGB8, row-major
    public let width: Int, height: Int
    public let latents: MLXArray // final packed latents [1, seq, 128], the selected half precision
    public let image: MLXArray // decoded [1, H, W, 3] in [-1, 1], the selected half precision, before clipping
    public let timings: KleinTimings
}

public struct KleinCancelled: Error {}

public final class KleinPipeline {
    public let transformer: Flux2Transformer
    public let textEncoder: KleinTextEncoder
    public let vae: Flux2VAEDecoder
    public let precision: KleinPrecision

    public init(transformer: Flux2Transformer, textEncoder: KleinTextEncoder, vae: Flux2VAEDecoder, precision: KleinPrecision) {
        self.transformer = transformer; self.textEncoder = textEncoder; self.vae = vae; self.precision = precision
    }

    public static func load(snapshot: URL, precision: KleinPrecision = .init()) throws -> KleinPipeline {
        for file in KleinWeights.requiredFiles where !FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(file).path) {
            throw KleinLoadError("The local snapshot is incomplete: \(file); run Get again.")
        }
        let tokenizer = try KleinTextEncoder.tokenizer(directory: snapshot.appendingPathComponent("tokenizer"))
        let encoder = try KleinWeights.textEncoder(snapshot: snapshot, dtype: precision.weights)
        let transformer = try KleinWeights.transformer(snapshot: snapshot, dtype: precision.weights)
        let vae = try KleinWeights.vae(snapshot: snapshot, dtype: precision.vae)
        return KleinPipeline(transformer: transformer, textEncoder: KleinTextEncoder(encoder: encoder, tokenizer: tokenizer), vae: vae, precision: precision)
    }

    /// Seeded Gaussian latents in the vendor's layout: [1, 128, gh, gw] drawn, then packed to [1, gh·gw, 128].
    public static func noise(seed: UInt64, width: Int, height: Int, dtype: DType = .bfloat16) -> MLXArray {
        let (gh, gw) = (height / 16, width / 16)
        let n = MLXRandom.normal([1, 128, gh, gw], dtype: dtype, key: MLXRandom.key(seed))
        return n.reshaped(1, 128, gh * gw).transposed(0, 2, 1)
    }

    /// Text ids (0, 0, 0, l) then image ids (0, h, w, 0): the joint sequence order is [txt, img].
    public static func ropeTables(textLength: Int, gridH: Int, gridW: Int, dtype: DType = .bfloat16) -> (MLXArray, MLXArray) {
        var ids = [Int32](); ids.reserveCapacity((textLength + gridH * gridW) * 4)
        for l in 0..<textLength { ids += [0, 0, 0, Int32(l)] }
        for h in 0..<gridH { for w in 0..<gridW { ids += [0, Int32(h), Int32(w), 0] } }
        let (c, s) = Flux2Rope.tables(MLXArray(ids, [ids.count / 4, 4]), dtype: dtype)
        return (c, s)
    }

    public func encode(_ prompt: String) -> MLXArray {
        let e = textEncoder.encode(prompt, activations: precision.encoder); eval(e); return e
    }

    /// Denoises from `noise` (packed the selected half precision) with `embeds` [512, 7680]; `step(i, n)` runs after each step and may throw
    /// to cancel. Returns the final packed latents in the selected half precision.
    public func denoise(
        embeds: MLXArray, noise: MLXArray, width: Int, height: Int, steps: Int,
        timings: inout KleinTimings, step: (Int, Int) throws -> Void
    ) rethrows -> MLXArray {
        let (gh, gw) = (height / 16, width / 16)
        let sigmas = KleinSchedule.sigmas(seqLen: gh * gw, steps: steps)
        let (cos, sin) = Self.ropeTables(textLength: embeds.dim(0), gridH: gh, gridW: gw, dtype: precision.activations)
        let encoder = embeds[.newAxis].asType(precision.activations)
        var x = noise.asType(precision.activations)
        for i in 0..<steps {
            let start = Date()
            let v = velocity(x, encoder: encoder, sigma: sigmas[i], cos: cos, sin: sin)
            x = x + MLXArray(Float16(sigmas[i + 1] - sigmas[i])).asType(precision.activations) * v // swiftlint:disable:this shorthand_operator
            eval(x)
            timings.steps.append(Date().timeIntervalSince(start))
            try step(i + 1, steps)
        }
        return x
    }

    /// One transformer pass at noise level `sigma` (timestep sigma · 1000), returned in the selected half precision.
    public func velocity(_ x: MLXArray, encoder: MLXArray, sigma: Float, cos: MLXArray, sin: MLXArray) -> MLXArray {
        transformer(x, encoder: encoder, timestep: MLXArray([Float16(sigma * 1000)]).asType(precision.activations), cos: cos, sin: sin, activations: precision.activations).asType(
            precision.activations)
    }

    /// Packed latents → [1, H, W, 3] the selected half precision in [-1, 1].
    public func decode(_ latents: MLXArray, width: Int, height: Int) -> MLXArray {
        let image = vae.decode(vae.latents(latents, gridH: height / 16, gridW: width / 16), dtype: precision.vae).asType(precision.activations)
        eval(image)
        return image
    }

    /// diffusers postprocess: (x / 2 + 0.5).clamp(0, 1), then round(x · 255) to uint8.
    public static func pixels(_ image: MLXArray) -> [UInt8] {
        let u8 = MLX.round(MLX.clip(image[0] / 2 + 0.5, min: 0, max: 1) * 255).asType(.uint8)
        eval(u8)
        return u8.asArray(UInt8.self)
    }

    public func generate(
        prompt: String, width: Int, height: Int, steps: Int, seed: UInt64, noise: MLXArray? = nil,
        step: (Int, Int) throws -> Void = { _, _ in }
    ) throws -> KleinOutput {
        var timings = KleinTimings()
        let started = Date()
        let embeds = encode(prompt)
        timings.encode = Date().timeIntervalSince(started)
        let latents = try denoise(
            embeds: embeds, noise: noise ?? Self.noise(seed: seed, width: width, height: height, dtype: precision.activations),
            width: width, height: height, steps: steps, timings: &timings, step: step)
        let decodeStart = Date()
        let image = decode(latents, width: width, height: height)
        timings.decode = Date().timeIntervalSince(decodeStart)
        guard all(isFinite(image)).item(Bool.self) else {
            throw KleinLoadError("Image generation produced nonfinite values; no image was written.")
        }
        let pixelStart = Date()
        let pixels = Self.pixels(image)
        timings.pixels = Date().timeIntervalSince(pixelStart)
        timings.total = Date().timeIntervalSince(started)
        return KleinOutput(pixels: pixels, width: width, height: height, latents: latents, image: image, timings: timings)
    }
}
