// SPDX-License-Identifier: MIT
// Maps the maker's checkpoint (alibabasglab/MossFormer2_SE_48K last_best_checkpoint.pt, a PyTorch state dict of
// ClearerVoice-Studio's TestNet) onto this port's module tree, and loads the converted half safetensors.
import Foundation
import MLX
import MLXNN

public enum MossFormer2Weights {
    public static let fileName = "model.safetensors"
    /// The maker state dict holds 929 tensors (55,262,794 values); 24 rotary `freqs` buffers and the positional
    /// `inv_freq` buffer are recomputed in code, so 904 tensors become parameters.
    public static let sourceTensorCount = 929
    public static let parameterCount = 904

    public enum Layout: Equatable, Sendable {
        /// As stored; a 3-D tensor is a PyTorch Conv1d weight [out, in, k] and becomes MLX's [out, k, in].
        case keep
        /// The FSMN memory Conv2d [C, 1, k, 1] → [C, k, 1, 1] (this port squeezes it into a depthwise conv1d).
        case conv2d
        /// [n] → [n, 1] for GlobalLayerNorm's per-channel affine.
        case column
        /// [1] → [] for PReLU's single slope.
        case scalar
    }

    /// Module key and layout for one maker key, or nil for a buffer the port recomputes.
    public static func map(_ source: String) -> (key: String, layout: Layout)? {
        guard source.hasPrefix("mossformer.") else { return nil }
        if source.hasSuffix(".rotary_pos_emb.freqs") || source == "mossformer.pos_enc.inv_freq" { return nil }
        var key = "model." + source
        let renames = [
            (".conv1.0.", ".conv1."), (".conv1.1.weight", ".prelu.weight"), (".mdl.0.", ".norm."), (".mdl.1.", ".linear."),
            (".mdl.3.sequential.1.conv.weight", ".conv_module.weight"), ("mossformer.output.0.", "mossformer.output."),
            ("mossformer.output_gate.0.", "mossformer.output_gate.")
        ]
        for (from, to) in renames { key = key.replacingOccurrences(of: from, with: to) }
        if key.hasSuffix("prelu.weight") { return (key, .scalar) }
        if source == "mossformer.norm.weight" || source == "mossformer.norm.bias" { return (key, .column) }
        if key.hasSuffix(".fsmn.conv1.weight") { return (key, .conv2d) }
        return (key, .keep)
    }

    /// Applies `layout` to a tensor read from the checkpoint.
    public static func arrange(_ array: MLXArray, layout: Layout) -> MLXArray {
        switch layout {
        case .keep: return array.ndim == 3 ? array.transposed(0, 2, 1) : array
        case .conv2d: return array.transposed(0, 2, 3, 1)
        case .column: return array.reshaped(-1, 1)
        case .scalar: return array.reshaped([])
        }
    }

    /// Builds the network from converted safetensors; every module parameter must be present with its exact shape.
    public static func load(_ url: URL, dtype: DType) throws -> MossFormer2SE {
        let model = MossFormer2SE()
        let arrays = try MLX.loadArrays(url: url)
        guard arrays.count == parameterCount else { throw MossFormer2Error("MossFormer2 weights hold \(arrays.count) tensors, expected \(parameterCount)") }
        try model.update(parameters: ModuleParameters.unflattened(arrays.mapValues { $0.asType(dtype) }), verify: .all)
        eval(model)
        return model
    }
}

public struct MossFormer2Error: Error, LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}
