// SPDX-License-Identifier: MIT
import Flux3Klein
import Foundation
import IntegrationKit
import MLX
import MLXNN
import Testing
@testable import MLXIdol

/// Parity with speridlabs/idol-sdk, captured once on the CPU at small sizes by `Workers/Geometry/scripts/idol_capture.py` (each module
/// in the precision idol-sdk uses). Heavy: IDOL_PARITY_DIR holds the capture, IDOL_WEIGHTS the speridlabs/idol
/// snapshot (FP32, cast here as Get does), IDOL_VAE a FLUX.2 VAE folder (klein's). The DINOv3 checks use random
/// weights in the real ViT-H+ layout; the trained ViT-H+ is gated.
/// Tolerance: BF16's, a relative RMS error under 2^-6 and cosine above 0.9995 for one pass.
@Suite(.serialized)
struct IdolParityTests {
    static let env = ProcessInfo.processInfo.environment
    static var enabled: Bool { env["IDOL_PARITY_DIR"] != nil && env["IDOL_WEIGHTS"] != nil }
    static func capture(_ name: String) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: URL(fileURLWithPath: env["IDOL_PARITY_DIR"]!).appendingPathComponent(name))
    }
    static func weights(_ file: String) -> URL { URL(fileURLWithPath: env["IDOL_WEIGHTS"]!).appendingPathComponent(file) }

    @discardableResult
    static func expectClose(_ name: String, _ candidate: MLXArray, _ reference: MLXArray, relative: Float = 1.0 / 64, cosine minimum: Float = 0.9995) -> Float {
        #expect(candidate.shape == reference.shape, "\(name) shape \(candidate.shape) != \(reference.shape)")
        let a = candidate.asType(.float32).reshaped([-1]), b = reference.asType(.float32).reshaped([-1])
        let cosine = ((a * b).sum() / (MLX.sqrt((a * a).sum()) * MLX.sqrt((b * b).sum()) + 1e-20)).item(Float.self)
        let rel = (MLX.sqrt(((a - b) * (a - b)).mean()) / (MLX.sqrt((b * b).mean()) + 1e-20)).item(Float.self)
        let maxAbs = MLX.abs(a - b).max().item(Float.self)
        print("PARITY \(name): cosine \(cosine) relRMS \(rel) maxAbs \(maxAbs)")
        #expect(cosine > minimum, "\(name) cosine \(cosine)")
        #expect(rel < relative, "\(name) relRMS \(rel)")
        return rel
    }

    @Test(.heavy, .enabled(if: enabled))
    func denoiserMatchesOnAShortSequence() throws {
        NativeRuntime.prepare()
        let ref = try Self.capture("ref.safetensors")
        let model = IdolDenoiser()
        try IdolWeights.load(model, file: Self.weights(IdolFiles.denoiser), dtype: .float16, float32: IdolFiles.float32Tensors, cast: true)
        let n = ref["den_latent"]!.dim(1)
        Self.expectClose("t_emb", model.tEmbedder(ref["den_t"]!), ref["den_t_emb"]!)
        let hx = model.inputLayer(ref["den_latent"]!).asType(.float16) + model.positionEmbedding(count: n)
        Self.expectClose("latent tokens + positions", hx, ref["den_hx0"]!)
        let rope = model.noiseRepo[0](ref["den_hx0"]!.asType(.float16))
        let rc = ref["den_repo0"]!
        Self.expectClose("RePo cos", rope.cos, rc[.ellipsis, 0])
        Self.expectClose("RePo sin", rope.sin, rc[.ellipsis, 1], relative: 1.0 / 32)
        func run(_ f1: MLXArray, _ f2: MLXArray) -> IdolVelocity {
            model(latent: ref["den_latent"]!, camera: ref["den_camera"]!, t: ref["den_t"]!, feature1: f1, feature2: f2)
        }
        let v = run(ref["den_feature1"]!, ref["den_feature2"]!)
        Self.expectClose("velocity latent", v.latent, ref["den_v_latent"]!)
        Self.expectClose("velocity camera", v.camera, ref["den_v_camera"]!)
        let zero = run(MLXArray.zeros(like: ref["den_feature1"]!), MLXArray.zeros(like: ref["den_feature2"]!))
        Self.expectClose("unconditional velocity latent", zero.latent, ref["den_vneg_latent"]!)
        // three guided Euler steps (batched guidance here, two passes in the reference)
        let pipeline = IdolPipeline(
            denoiser: model, octree: IdolOctreeDecoder(), gaussianDecoder: IdolGaussianDecoder(), dinov3: IdolDINOv3(IdolDINOv3Config()),
            vae: Flux3VAEEncoderHalf(Flux3VAEConfig()), statistics: (MLXArray.zeros([128]), MLXArray.ones([128])))
        let s = try pipeline.sample(
            latent: ref["den_latent"]!, camera: ref["den_camera"]!, feature1: ref["den_feature1"]!, feature2: ref["den_feature2"]!,
            steps: 3, guidance: 3, rescale: 3)
        Self.expectClose("3 guided steps latent", s.latent, ref["den_sample3_latent"]!)
        // five numbers, each step's difference amplified (1 + w) = 4× by guidance
        Self.expectClose("3 guided steps camera", s.camera, ref["den_sample3_camera"]!, relative: 1.0 / 16, cosine: 0.999)
    }

    @Test(.heavy, .enabled(if: enabled))
    func decodersMatch() throws {
        NativeRuntime.prepare()
        let ref = try Self.capture("ref.safetensors")
        let octree = IdolOctreeDecoder(), gs = IdolGaussianDecoder()
        try IdolWeights.load(octree, file: Self.weights(IdolFiles.decoder), dtype: .float16, cast: true)
        try IdolWeights.load(gs, file: Self.weights(IdolFiles.gaussianDecoder), dtype: .float16, cast: true)
        let logits = octree(ref["dec_cells"]!, resolution: ref["dec_level"]!.asType(.float32), latent: ref["dec_cond"]!)
        Self.expectClose("octree logits", logits, ref["dec_logits"]!)
        Self.expectClose("octree probabilities", softmax(logits, axis: -1), softmax(ref["dec_logits"]!, axis: -1))
        let features = gs(ref["gs_anchors"]!, latent: ref["dec_cond"]!)
        Self.expectClose("Gaussian features", features, ref["gs_features"]!)
        // activation alone, on the reference features
        let g = gs.gaussians(anchors: ref["gs_anchors"]!, features: ref["gs_features"]!)
        for (name, value) in [("xyz", g.xyz), ("features_dc", g.featuresDC), ("scaling", g.scaling), ("rotation", g.rotation), ("opacity", g.opacity)] {
            Self.expectClose("Gaussians \(name)", value, ref["gs_" + name]!.reshaped(value.shape), relative: 1e-4, cosine: 0.99999)
        }
    }

    /// The VAE posterior against the reference in BF16 (as idol-sdk runs it) and against an FP32 truth of the same
    /// encoder (`vae_fp32.safetensors`), for the BF16 and FP16 loads of the weights.
    @Test(.heavy, .enabled(if: enabled && env["IDOL_VAE"] != nil))
    func vaePosteriorMatches() throws {
        NativeRuntime.prepare()
        let ref = try Self.capture("ref.safetensors")
        let truth = try? Self.capture("vae_fp32.safetensors")["vae_mean_fp32"]!.transposed(0, 2, 3, 1)
        let image = ref["vae_image"]!.transposed(0, 2, 3, 1)
        for dtype in [DType.bfloat16, .float16] {
            let (encoder, decoder) = try Flux3Weights.vae(directory: URL(fileURLWithPath: Self.env["IDOL_VAE"]!), dtype: dtype)
            let moments = encoder.moments(image * 2 - 1).asType(.float32)
            let mean = moments[.ellipsis, ..<32]
            print("PARITY VAE \(dtype)")
            if let truth { Self.expectClose("VAE mean \(dtype) vs FP32", mean, truth, relative: 1.0 / 32, cosine: 0.999) }
            Self.expectClose("VAE mean \(dtype) vs idol-sdk BF16", mean, ref["vae_mean"]!.transposed(0, 2, 3, 1), relative: 1.0 / 16, cosine: 0.999)
            let std = MLX.exp(0.5 * MLX.clip(moments[.ellipsis, 32...], min: -30, max: 20))
            Self.expectClose("VAE std \(dtype)", std, ref["vae_std"]!.transposed(0, 2, 3, 1), relative: 1.0 / 32)
            let pipeline = IdolPipeline(
                denoiser: IdolDenoiser(), octree: IdolOctreeDecoder(), gaussianDecoder: IdolGaussianDecoder(), dinov3: IdolDINOv3(), vae: encoder,
                statistics: decoder.statistics)
            Self.expectClose("VAE tokens \(dtype)", pipeline.vaeTokens(image, key: nil), ref["vae_tokens_from_mean"]!, relative: 1.0 / 16, cosine: 0.999)
        }
    }

    /// DINOv3 in the ViT-H+ layout: a tiny two-layer model and one full-width H+ layer (random weights, BF16 as idol-sdk).
    @Test(.heavy, .enabled(if: env["IDOL_PARITY_DIR"] != nil))
    func dinov3MatchesTransformers() throws {
        NativeRuntime.prepare()
        for (file, config) in [("dino_tiny.safetensors", (128, 2, 4, 160)), ("dino_hplus1.safetensors", (1280, 1, 20, 5120))] {
            var c = IdolDINOv3Config()
            (c.hidden, c.layers, c.heads, c.intermediate) = config
            let model = IdolDINOv3(c)
            let url = URL(fileURLWithPath: Self.env["IDOL_PARITY_DIR"]!).appendingPathComponent(file)
            try IdolWeights.load(model, file: url, dtype: .bfloat16, cast: true, rename: { $0.hasPrefix("ref.") ? nil : IdolDINOv3.moduleName($0) })
            let ref = try MLX.loadArrays(url: url)
            let pixels = ref["ref.pixels"]!
            let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 3, 1, 1]), std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 3, 1, 1])
            let tokens = try model((pixels - mean) / std)
            Self.expectClose("DINOv3 \(file)", tokens, ref["ref.tokens"]!, relative: 1.0 / 32, cosine: 0.999)
        }
    }

    /// The trained ViT-H+ (IDOL_DINOV3: Meta's snapshot, FP32, cast here to BF16 as Get does) on one gallery picture at
    /// 256 px, against transformers in FP32 and in BF16 (`Workers/Geometry/scripts/idol_capture_dino.py`). The residual stream peaks at about
    /// 137,000 from layer 22, so FP16 is not an option; BF16 must stay as close to FP32 as idol-sdk's own BF16 run.
    @Test(.heavy, .enabled(if: env["IDOL_PARITY_DIR"] != nil && env["IDOL_DINOV3"] != nil))
    func dinov3TrainedWeightsMatch() throws {
        NativeRuntime.prepare()
        let model = IdolDINOv3()
        try IdolWeights.load(
            model, file: URL(fileURLWithPath: Self.env["IDOL_DINOV3"]!).appendingPathComponent("model.safetensors"), dtype: .bfloat16, cast: true,
            rename: IdolDINOv3.moduleName)
        let ref = try Self.capture("dino_real.safetensors")
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 3, 1, 1]), std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 3, 1, 1])
        let tokens = try model((ref["pixels"]! - mean) / std)
        let reference = Self.expectClose("DINOv3 ViT-H+ BF16 vs transformers BF16", tokens, ref["tokens_bf16"]!, relative: 1.0 / 8, cosine: 0.99)
        let truth = Self.expectClose("DINOv3 ViT-H+ BF16 vs transformers FP32", tokens, ref["tokens_fp32"]!, relative: 1.0 / 8, cosine: 0.99)
        let floor = Self.expectClose("transformers BF16 vs FP32 (idol-sdk's own error)", ref["tokens_bf16"]!, ref["tokens_fp32"]!, relative: 1.0 / 8, cosine: 0.99)
        #expect(truth < 1.5 * floor + 0.01, "MLX BF16 \(truth) vs idol-sdk BF16 \(floor) from FP32; apart \(reference)")
    }
}
