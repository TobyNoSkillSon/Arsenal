// SPDX-License-Identifier: MIT
// IDOL end to end (idol/pipeline.py IdolPipeline.__call__): prepared image → DINOv3 and FLUX.2 VAE tokens → 50 guided
// Euler steps of the denoiser → octree anchors → Gaussians. Random numbers come from MLX's generator, so a seed gives
// a different (equally valid) sample than the PyTorch reference with the same seed.
import CoreGraphics
import Flux2Klein
import Flux3Klein
import Foundation
import MLX
import MLXNN
import MLXRandom
import IntegrationKit

/// Converted weights, one folder: IDOL's three modules at the root (FP16; the latent positions FP32), DINOv3 ViT-H+ in
/// `components/idol-dinov3` (BF16) and the FLUX.2 VAE in `components/idol-flux2-vae/vae` (FP16).
public enum IdolFiles {
    public static let denoiser = "denoiser.safetensors", decoder = "decoder.safetensors", gaussianDecoder = "decoder_gs.safetensors"
    public static let dinov3 = "components/idol-dinov3/model.safetensors"
    public static let vae = "components/idol-flux2-vae/vae"
    /// Tensors kept in float32: the denoiser's fixed latent positions (98 KB; their embedding resolves 2^16 cycles).
    public static let float32Tensors: Set<String> = ["pos_pe"]
}

public enum IdolWeights {
    /// nn.Sequential children are numbered (`mlp.0`, `adaLN_modulation.1`); MLX reads a numbered key as an array index,
    /// so those children are named `l0`, `l1`, `l2` here. List indices (`blocks.3`) keep their numbers.
    public static func moduleKey(_ key: String) -> String {
        var parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        for i in parts.indices.dropFirst() where ["mlp", "adaLN_modulation"].contains(parts[i - 1]) && Int(parts[i]) != nil { parts[i] = "l" + parts[i] }
        return parts.joined(separator: ".")
    }

    /// Loads `file` into `module` by checkpoint name. `dtype` is the half every floating tensor must already have
    /// (`cast` converts instead, for lab and test sources); names in `float32` stay float32. Conv kernels are turned
    /// from PyTorch OIHW to MLX OHWI.
    public static func load(
        _ module: Module, file: URL, dtype: DType, float32: Set<String> = [], cast: Bool = false, rename: (String) -> String? = { $0 }
    ) throws {
        var updates: [String: MLXArray] = [:]
        for (key, raw) in try MLX.loadArrays(url: file) {
            guard let name = rename(key).map(moduleKey) else { continue }
            var value = raw
            if value.ndim == 4 { value = value.transposed(0, 2, 3, 1) }
            if [.float16, .bfloat16, .float32].contains(value.dtype) {
                let want: DType = float32.contains(name) ? .float32 : dtype
                guard cast || value.dtype == want else { throw IdolError("\(file.lastPathComponent): \(key) is \(value.dtype), expected \(want); Get converts the weights.") }
                value = value.asType(want)
            }
            updates[name] = value
        }
        try module.update(parameters: ModuleParameters.unflattened(updates), verify: [.allModelKeysSet, .noUnusedKeys, .shapeMismatch])
        eval(module.parameters())
    }
}

public struct IdolRequest: Sendable {
    public var margin = 1.2, seed: UInt64 = 42, steps = 50, guidance: Float = 3, rescaleT: Float = 3, gaussians = 262_144
    public init() {}
}

public struct IdolResult {
    public let gaussians: IdolGaussians
    public let camera: IdolCamera
    public let prepared: [UInt8]
}

public final class IdolPipeline {
    public let denoiser: IdolDenoiser, octree: IdolOctreeDecoder, gaussianDecoder: IdolGaussianDecoder
    public let dinov3: IdolDINOv3, vae: Flux3VAEEncoderHalf
    let statistics: (mean: MLXArray, variance: MLXArray)
    public let maxVoxelLevel: Int, imageSize: Int

    public init(
        denoiser: IdolDenoiser, octree: IdolOctreeDecoder, gaussianDecoder: IdolGaussianDecoder, dinov3: IdolDINOv3, vae: Flux3VAEEncoderHalf,
        statistics: (mean: MLXArray, variance: MLXArray), maxVoxelLevel: Int = 8, imageSize: Int = 1024
    ) {
        self.denoiser = denoiser; self.octree = octree; self.gaussianDecoder = gaussianDecoder; self.dinov3 = dinov3; self.vae = vae
        self.statistics = statistics; self.maxVoxelLevel = maxVoxelLevel; self.imageSize = imageSize
    }

    /// A miniature pipeline with random half weights and every stage of the real one (64 px image, 64 latent tokens,
    /// octree depth 4), for scenario tests without weights.
    public static func lab(seed: UInt64 = 7) throws -> IdolPipeline {
        var d = IdolDenoiserConfig()
        d.tokens = 64; d.channels = 64; d.condChannels = 64; d.blocks = 1; d.refinerBlocks = 1; d.heads = 4; d.headChannels = 16
        var c = IdolDecoderConfig()
        c.channels = 64; c.heads = 4; c.octreeBlocks = 1; c.gaussianBlocks = 1
        var v = IdolDINOv3Config()
        v.hidden = 64; v.layers = 1; v.heads = 4; v.intermediate = 96
        var vaeConfig = Flux3VAEConfig()
        vaeConfig.blockOut = [32, 32, 32, 32]
        let denoiser = IdolDenoiser(d), octree = IdolOctreeDecoder(c), gaussianDecoder = IdolGaussianDecoder(c), dinov3 = IdolDINOv3(v)
        let vae = Flux3VAEEncoderHalf(vaeConfig)
        let modules: [Module] = [denoiser, octree, gaussianDecoder, vae]
        for (i, module) in modules.enumerated() {
            try NativeParameters.initializeRandom(module, half: .float16, seed: seed + UInt64(i))
        }
        try NativeParameters.initializeRandom(dinov3, half: .bfloat16, seed: seed + 9)
        try denoiser.update(parameters: ModuleParameters.unflattened(["pos_pe": MLXRandom.uniform(0..<1, [1, d.tokens, 3], key: MLXRandom.key(seed))]), verify: .none)
        return IdolPipeline(
            denoiser: denoiser, octree: octree, gaussianDecoder: gaussianDecoder, dinov3: dinov3, vae: vae,
            statistics: (MLXArray.zeros([128]), MLXArray.ones([128])), maxVoxelLevel: 4, imageSize: 64)
    }

    /// Learned weights by module, for the precision check and inventory. The FP32 latent positions are a fixed
    /// buffer, not a learned weight, and are left out.
    public var parameters: [String: MLXArray] {
        var all: [String: MLXArray] = [:]
        for (prefix, module) in [("denoiser", denoiser), ("decoder", octree), ("decoder_gs", gaussianDecoder), ("dinov3", dinov3), ("vae", vae)] as [(String, Module)] {
            for (name, value) in module.parameters().flattened() where !(prefix == "denoiser" && IdolFiles.float32Tensors.contains(name)) {
                all[prefix + "." + name] = value
            }
        }
        return all
    }

    /// Loads the converted folder (see `IdolFiles`). `cast` accepts unconverted sources (lab and tests only).
    public static func load(_ root: URL, cast: Bool = false, cancellation: () throws -> Void = {}) throws -> IdolPipeline {
        let denoiser = IdolDenoiser(), octree = IdolOctreeDecoder(), gaussianDecoder = IdolGaussianDecoder(), dinov3 = IdolDINOv3()
        try IdolWeights.load(denoiser, file: root.appendingPathComponent(IdolFiles.denoiser), dtype: .float16, float32: IdolFiles.float32Tensors, cast: cast)
        try cancellation()
        try IdolWeights.load(octree, file: root.appendingPathComponent(IdolFiles.decoder), dtype: .float16, cast: cast)
        try IdolWeights.load(gaussianDecoder, file: root.appendingPathComponent(IdolFiles.gaussianDecoder), dtype: .float16, cast: cast)
        try cancellation()
        try IdolWeights.load(dinov3, file: root.appendingPathComponent(IdolFiles.dinov3), dtype: .bfloat16, cast: cast, rename: IdolDINOv3.moduleName)
        try cancellation()
        let (encoder, decoder) = try Flux3Weights.vae(directory: root.appendingPathComponent(IdolFiles.vae), dtype: .float16)
        let stats = decoder.statistics
        eval(stats.mean, stats.variance)
        return IdolPipeline(denoiser: denoiser, octree: octree, gaussianDecoder: gaussianDecoder, dinov3: dinov3, vae: encoder, statistics: stats)
    }

    /// The image tokens (ImageEncoder.__call__): feature1 DINOv3 [1, 4101, 1280], feature2 FLUX.2 VAE [1, 4101, 128]
    /// (five zero rows aligning it with DINOv3's class and register tokens). `rgb` is the prepared composite.
    public func condition(rgb: [UInt8], size: Int, key: MLXArray, checkpoint: () throws -> Void = {}) throws -> (MLXArray, MLXArray) {
        let image = MLXArray(rgb.map { Float($0) / 255 }, [1, size, size, 3])
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3]), std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
        let feature1 = try dinov3(((image - mean) / std).transposed(0, 3, 1, 2), checkpoint: checkpoint)
        try checkpoint()
        let feature2 = vaeTokens(image, key: key)
        eval(feature1, feature2)
        return (feature1, feature2)
    }

    /// A sample of the VAE posterior, patchified 2×2 and normalised by the VAE's batch-norm statistics.
    public func vaeTokens(_ image: MLXArray, key: MLXArray?) -> MLXArray {
        let moments = vae.moments(image * 2 - 1).asType(.float32)
        let mean = moments[.ellipsis, ..<32], logvar = MLX.clip(moments[.ellipsis, 32...], min: -30, max: 20)
        var latent = mean
        if let key { latent = mean + MLX.exp(0.5 * logvar) * MLXRandom.normal(mean.shape, key: key) }
        let stats = (statistics.mean.asType(.float32), statistics.variance.asType(.float32))
        let tokens = (Flux3VAEEncoderHalf.patchify(latent) - stats.0) / MLX.sqrt(stats.1 + 1e-4)
        return MLX.concatenated([MLXArray.zeros([tokens.dim(0), 5, tokens.dim(2)]), tokens], axis: 1)
    }

    /// The rescaled time grid from 1 to 0 (sampler.py).
    public static func timeGrid(steps: Int, rescale: Float) -> [Float] {
        (0...steps).map { i in
            let t = Float(1) - Float(i) / Float(steps)
            return rescale * t / (1 + (rescale - 1) * t)
        }
    }

    /// Euler integration from noise (t = 1) to data with classifier-free guidance: v = (1 + w)·v(cond) − w·v(0).
    public func sample(
        latent: MLXArray, camera: MLXArray, feature1: MLXArray, feature2: MLXArray, steps: Int, guidance: Float, rescale: Float,
        step: (Int) throws -> Void = { _ in }
    ) throws -> (latent: MLXArray, camera: MLXArray) {
        var z = latent, cam = camera
        let grid = Self.timeGrid(steps: steps, rescale: rescale)
        let guided = guidance > 0
        let f1 = guided ? MLX.concatenated([feature1, MLXArray.zeros(like: feature1)], axis: 0) : feature1
        let f2 = guided ? MLX.concatenated([feature2, MLXArray.zeros(like: feature2)], axis: 0) : feature2
        for i in 0..<steps {
            try step(i)
            let batch = guided ? 2 : 1
            let t = MLXArray([Float](repeating: 1000 * grid[i], count: batch))
            let zIn = guided ? MLX.concatenated([z, z], axis: 0) : z, camIn = guided ? MLX.concatenated([cam, cam], axis: 0) : cam
            let v = denoiser(latent: zIn, camera: camIn, t: t, feature1: f1, feature2: f2)
            var vz = v.latent, vc = v.camera
            if guided {
                vz = (1 + guidance) * vz[0..<1] - guidance * vz[1..<2]
                vc = (1 + guidance) * vc[0..<1] - guidance * vc[1..<2]
            }
            let dt = grid[i] - grid[i + 1]
            z -= vz * dt; cam -= vc * dt
            eval(z, cam)
        }
        return (z, cam)
    }

    /// The whole request. `progress(stage, fraction)` reports each stage.
    public func generate(_ image: CGImage, request r: IdolRequest, cancellation: () throws -> Void = {}, progress: (String, Double) -> Void = { _, _ in }) throws -> IdolResult {
        progress("Preparing image", 0)
        let rgb = try IdolImage.prepare(image, size: imageSize, margin: r.margin)
        try cancellation()
        let keys = MLXRandom.split(key: MLXRandom.key(r.seed), into: 4)
        progress("Reading image", 0.02)
        let (f1, f2) = try condition(rgb: rgb, size: imageSize, key: keys[0], checkpoint: cancellation)
        let noise = MLXRandom.normal([1, denoiser.config.tokens, denoiser.config.inChannels], key: keys[1])
        let cameraNoise = MLXRandom.normal([1, 1, denoiser.config.camChannels], key: keys[2])
        let sampled = try sample(
            latent: noise, camera: cameraNoise, feature1: f1, feature2: f2, steps: r.steps, guidance: r.guidance, rescale: r.rescaleT,
            step: { i in
                try cancellation(); progress("Shaping", 0.05 + 0.85 * Double(i) / Double(r.steps))
            })
        progress("Placing Gaussians", 0.9)
        let gaussians = try decode(sampled.latent, count: r.gaussians, key: keys[3], cancellation: cancellation)
        let camera = IdolCamera(latent: sampled.camera.reshaped([-1]).asArray(Float.self))
        progress("Done", 1)
        return IdolResult(gaussians: gaussians, camera: camera, prepared: rgb)
    }

    /// latent [1, 8192, 16] → Gaussians: count / 32 octree anchors, 32 Gaussians each.
    public func decode(_ latent: MLXArray, count: Int, key: MLXArray, cancellation: () throws -> Void = {}) throws -> IdolGaussians {
        let perAnchor = gaussianDecoder.perAnchor
        let anchors = try IdolOctree.sample(decoder: octree, latent: latent, points: max(1, count / perAnchor), level: maxVoxelLevel, key: key, cancellation: cancellation)
        try cancellation()
        let features = gaussianDecoder(anchors, latent: latent)
        let gaussians = gaussianDecoder.gaussians(anchors: anchors, features: features)
        eval(gaussians.xyz, gaussians.featuresDC, gaussians.scaling, gaussians.rotation, gaussians.opacity)
        return gaussians
    }
}
