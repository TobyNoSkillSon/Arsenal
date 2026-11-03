// SPDX-License-Identifier: MIT
// Strict loading from the vendor's diffusers layout (transformer/, text_encoder/, vae/): every module parameter is
// filled and every checkpoint tensor that belongs to the served path is consumed; a partial load would only show as a
// broken image. The VAE ships in float32 and is cast to BF16 here, tensor by tensor; nothing else is up-cast.
import Flux2Klein
import Foundation
import MLX
import MLXNN
import ModelPrecision

public enum Flux3Weights {
    /// The snapshot files the runner reads (the add-on folder pins exactly these, plus the tokenizer's vocab files).
    public static let requiredFiles = [
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "text_encoder/config.json", "text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json", "vae/diffusion_pytorch_model.safetensors"
    ]

    static func archives(_ directory: URL) throws -> [URL] {
        let files = try FileManager.default.contentsOfDirectory(at: directory.resolvingSymlinksInPath(), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw Flux3Error("No safetensors in \(directory.lastPathComponent)/.") }
        return files
    }

    /// Checkpoint tensors of one component, renamed by `name` (nil skips a tensor without reading it), reshaped by
    /// `layout` and cast to `dtype`.
    static func tensors(
        _ directory: URL, dtype: DType, name: (String) -> String?, layout: (MLXArray) -> MLXArray = { $0 }
    ) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for file in try archives(directory) {
            let archive = try TensorArchive(file)
            for key in archive.entries.keys.sorted() {
                guard let target = name(key) else { continue }
                guard out[target] == nil else { throw Flux3Error("\(target) appears twice in \(directory.lastPathComponent)/.") }
                // One tensor's file bytes live at a time: without the pool every read's buffer stayed until the
                // component finished, so loading briefly held the weights twice (37.7 GB peak for 20 GB, 1 Nov).
                out[target] = try autoreleasepool {
                    let value = layout(try archive.array(key))
                    let cast = value.dtype == dtype || !value.dtype.isFloatingPoint ? value : value.asType(dtype)
                    eval(cast)
                    return cast
                }
            }
        }
        return out
    }

    static func apply(_ weights: [String: MLXArray], to model: Module, label: String) throws {
        let expected = Set(model.parameters().flattened().map(\.0)), given = Set(weights.keys)
        let missing = expected.subtracting(given).sorted(), unused = given.subtracting(expected).sorted()
        guard missing.isEmpty, unused.isEmpty else {
            throw Flux3Error(
                "\(label) weights do not match: \(missing.count) missing (\(missing.prefix(3).joined(separator: ", "))), \(unused.count) unused (\(unused.prefix(3).joined(separator: ", ")))."
            )
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    // MARK: Transformer

    public static func transformerName(_ key: String) -> String { key.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.") }

    public static func transformer(directory: URL, dtype: DType = .bfloat16) throws -> Flux3Transformer {
        let weights = try tensors(directory, dtype: dtype, name: transformerName)
        let model = Flux3Transformer(try Flux3TransformerConfig.inferred(from: weights.mapValues(\.shape)))
        try apply(weights, to: model, label: "Transformer")
        return model
    }

    // MARK: Text encoder

    /// `model.language_model.X` → `X` for the embedding and layers below the deepest tap; nil for the vision tower,
    /// deeper layers, the final norm and any head.
    public static func textEncoderName(_ key: String) -> String? {
        let prefix = "model.language_model."
        guard key.hasPrefix(prefix) else { return nil }
        let name = String(key.dropFirst(prefix.count))
        if name.hasPrefix("embed_tokens.") { return name }
        guard name.hasPrefix("layers."), let index = Int(name.dropFirst(7).prefix { $0 != "." }) else { return nil }
        return index < Flux3Prompt.taps.max()! ? name : nil
    }

    public static func textEncoder(directory: URL, dtype: DType = .bfloat16) throws -> Flux3TextEncoder {
        let config = try Flux3TextConfig(json: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let weights = try tensors(directory, dtype: dtype, name: textEncoderName)
        let model = Flux3TextEncoder(config)
        try apply(weights, to: model, label: "Text encoder")
        return model
    }

    // MARK: VAE

    /// diffusers VAE names → (half, module name): encoder and quant_conv to the encoder half, decoder, post_quant_conv
    /// and the batch-norm statistics to Flux2Klein's decoder half; num_batches_tracked is dropped.
    public static func vaeName(_ key: String) -> (encoder: Bool, name: String)? {
        if key.hasSuffix("num_batches_tracked") { return nil }
        if key == "bn.running_mean" { return (false, "bn_mean") }
        if key == "bn.running_var" { return (false, "bn_var") }
        let name = key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
        if key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") { return (true, name) }
        if key.hasPrefix("decoder.") || key.hasPrefix("post_quant_conv.") { return (false, name) }
        return nil
    }

    public static func vae(directory: URL, dtype: DType = .bfloat16) throws -> (encoder: Flux3VAEEncoderHalf, decoder: Flux2VAEDecoder) {
        var halves: [Bool: [String: MLXArray]] = [true: [:], false: [:]]
        // torch OIHW → MLX OHWI
        let all = try tensors(
            directory, dtype: dtype, name: { key in vaeName(key).map { ($0.encoder ? "E:" : "D:") + $0.name } },
            layout: { $0.ndim == 4 ? $0.transposed(0, 2, 3, 1) : $0 })
        for (key, value) in all { halves[key.hasPrefix("E:")]![String(key.dropFirst(2))] = value }
        let config = try Flux3VAEConfig.inferred(from: halves[true]!.mapValues(\.shape))
        let encoder = Flux3VAEEncoderHalf(config), decoder = Flux2VAEDecoder(blockOut: config.blockOut)
        try apply(halves[true]!, to: encoder, label: "VAE encoder")
        try apply(halves[false]!, to: decoder, label: "VAE decoder")
        return (encoder, decoder)
    }
}

extension DType {
    var isFloatingPoint: Bool { [.float16, .bfloat16, .float32, .float64].contains(self) }
}
