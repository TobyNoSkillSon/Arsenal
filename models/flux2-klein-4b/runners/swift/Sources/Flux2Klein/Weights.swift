// SPDX-License-Identifier: MIT
// Strict loading from the vendor's diffusers snapshot: every module parameter is filled and every checkpoint key that
// belongs to the decode path is consumed; a partial load would only show up as a broken image.
import Foundation
import MLX
import MLXNN
import ModelPrecision

public struct KleinLoadError: LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

public enum KleinWeights {
    /// The snapshot files the runner reads (all inside the vendor's pinned diffusers layout).
    public static let requiredFiles = [
        "transformer/diffusion_pytorch_model.safetensors", "text_encoder/config.json",
        "text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors",
        "vae/diffusion_pytorch_model.safetensors", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"
    ]

    static func arrays(_ directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory.resolvingSymlinksInPath(), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw KleinLoadError("No safetensors in \(directory.lastPathComponent)/") }
        var out: [String: MLXArray] = [:]
        for file in files { out.merge(try TensorArchive(file).arrays()) { _, new in new } }
        return out
    }

    static func apply(_ weights: [String: MLXArray], to model: Module, label: String) throws {
        try StreamedPrecision.prepare(model, weights: weights)
        let expected = Set(model.parameters().flattened().map(\.0)), given = Set(weights.keys)
        let missing = expected.subtracting(given).sorted(), unused = given.subtracting(expected).sorted()
        guard missing.isEmpty, unused.isEmpty else {
            throw KleinLoadError(
                "\(label) weights do not match: \(missing.count) missing (\(missing.prefix(3).joined(separator: ", "))), \(unused.count) unused (\(unused.prefix(3).joined(separator: ", ")))."
            )
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    public static func transformer(snapshot: URL, dtype: DType) throws -> Flux2Transformer {
        var weights: [String: MLXArray] = [:]
        for (key, value) in try arrays(snapshot.appendingPathComponent("transformer")) {
            weights[key.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")] = value.dtype == .uint32 ? value : value.asType(dtype)
        }
        let model = Flux2Transformer()
        try apply(weights, to: model, label: "Transformer")
        return model
    }

    /// Qwen3 layers 0..<27 and the embedding; deeper layers, the final norm and the tied head are never loaded.
    public static func textEncoder(snapshot: URL, dtype: DType) throws -> Qwen3TapEncoder {
        let directory = snapshot.appendingPathComponent("text_encoder")
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let keep = Qwen3TapEncoder.taps.max()!
        var weights: [String: MLXArray] = [:]
        for (key, value) in try arrays(directory) {
            guard key.hasPrefix("model.") else { continue }
            let name = String(key.dropFirst(6))
            if name.hasPrefix("norm.") { continue }
            if name.hasPrefix("layers."), let index = Int(name.dropFirst(7).prefix { $0 != "." }), index >= keep { continue }
            weights[name] = value.dtype == .uint32 ? value : value.asType(dtype)
        }
        let model = Qwen3TapEncoder(config)
        try apply(weights, to: model, label: "Text encoder")
        return model
    }

    /// Decoder, post-quant conv and the latent batch-norm statistics (kept in the VAE dtype); the encoder half is skipped.
    public static func vae(snapshot: URL, dtype: DType) throws -> Flux2VAEDecoder {
        var weights: [String: MLXArray] = [:]
        for (key, raw) in try arrays(snapshot.appendingPathComponent("vae")) {
            if key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") || key.hasSuffix("num_batches_tracked") { continue }
            var value = raw
            if value.ndim == 4 { value = value.transposed(0, 2, 3, 1) } // OIHW → OHWI
            switch key {
            case "bn.running_mean": weights["bn_mean"] = value.dtype == .uint32 ? value : value.asType(dtype)
            case "bn.running_var": weights["bn_var"] = value.dtype == .uint32 ? value : value.asType(dtype)
            default: weights[key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")] = value.dtype == .uint32 ? value : value.asType(dtype)
            }
        }
        let model = Flux2VAEDecoder()
        try apply(weights, to: model, label: "VAE")
        return model
    }
}

extension KleinWeights {
    /// Lab-only (precision report): MLX affine group-64 weight quantization of the transformer's attention and
    /// feed-forward Linears. Weights-only: QuantizedLinear dequantizes inside the matmul kernel and computes in the
    /// activation dtype. The embedders, modulation, norm_out and proj_out stay bf16.
    public static func quantize(_ model: Flux2Transformer, bits: Int) {
        let keep = ["x_embedder", "context_embedder", "proj_out", "norm_out", "time_guidance_embed", "double_stream_modulation", "single_stream_modulation"]
        MLXNN.quantize(
            model: model,
            filter: { path, module in
                guard module is Linear, !keep.contains(where: { path.hasPrefix($0) }) else { return nil }
                return (64, bits, .affine)
            })
        eval(model)
    }
}
