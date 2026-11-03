// swiftlint:disable shorthand_operator
// SPDX-License-Identifier: MIT
// EmbeddingGemma 2's vision tower: Transformers' Gemma4VisionModel (models/gemma4/modeling_gemma4.py, 5.22.1) and
// EmbeddingGemma2MultimodalEmbedder, in MLX. Layout follows mlx-swift-lm's Gemma4 vision port (Libraries/MLXVLM/
// Models/Gemma4.swift @ bd4b7434, MIT). Only real patches are encoded: the maker's padding patches are masked out as
// keys and dropped after pooling, so they never change a real output.
import ArsenalWire
import Foundation
import IntegrationKit
import MLX
import MLXFast
import MLXNN

struct EmbeddingGemma2VisionConfiguration: Codable, Sendable {
    let hiddenSize: Int
    let layers: Int
    let heads: Int
    let headDim: Int
    let intermediateSize: Int
    let patchSize: Int
    let poolingKernelSize: Int
    let positionEmbeddingSize: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let standardize: Bool
    let clipped: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", layers = "num_hidden_layers", heads = "num_attention_heads", headDim = "head_dim"
        case intermediateSize = "intermediate_size", patchSize = "patch_size", poolingKernelSize = "pooling_kernel_size"
        case positionEmbeddingSize = "position_embedding_size", rmsNormEps = "rms_norm_eps", ropeParameters = "rope_parameters"
        case standardize, clipped = "use_clipped_linears", kvHeads = "num_key_value_heads", modelType = "model_type"
    }
    private struct Rope: Codable { let rope_theta: Float; let rope_type: String }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize); layers = try c.decode(Int.self, forKey: .layers)
        heads = try c.decode(Int.self, forKey: .heads); headDim = try c.decode(Int.self, forKey: .headDim)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize); patchSize = try c.decode(Int.self, forKey: .patchSize)
        poolingKernelSize = try c.decode(Int.self, forKey: .poolingKernelSize)
        positionEmbeddingSize = try c.decode(Int.self, forKey: .positionEmbeddingSize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        let rope = try c.decode(Rope.self, forKey: .ropeParameters)
        ropeTheta = rope.rope_theta
        standardize = try c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
        clipped = try c.decodeIfPresent(Bool.self, forKey: .clipped) ?? false
        let kv = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        let type = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_vision"
        // The supported layout: the snapshot's own (no standardization, no clipping, MHA, axial RoPE, 3×3 pooling).
        guard type == "gemma4_vision", rope.rope_type == "axial", !standardize, !clipped, kv == heads, headDim % 4 == 0,
            heads * headDim == hiddenSize, patchSize == EG2Vision.patch, poolingKernelSize == EG2Vision.pooling
        else {
            throw DecodingError.dataCorruptedError(forKey: .modelType, in: c, debugDescription: "Unsupported EmbeddingGemma 2 vision layout.")
        }
    }

    func encode(to encoder: any Encoder) throws {}
}

/// `Gemma4ClippableLinear` without clipping (the vision config's `use_clipped_linears = false`): the weight sits under `.linear`.
final class EG2Linear: Module, UnaryLayer {
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ input: Int, _ output: Int) { _linear.wrappedValue = Linear(input, output, bias: false) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(x) }
}

/// RMSNorm without a learned scale, normalized in float32 and returned in the input's dtype.
func eg2NormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: .float32), eps: eps).asType(x.dtype)
}

final class EG2VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: EG2Linear
    @ModuleInfo(key: "k_proj") var kProj: EG2Linear
    @ModuleInfo(key: "v_proj") var vProj: EG2Linear
    @ModuleInfo(key: "o_proj") var oProj: EG2Linear
    @ModuleInfo(key: "q_norm") var qNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EmbeddingGemma2RMSNorm
    let heads: Int, headDim: Int, eps: Float

    init(_ c: EmbeddingGemma2VisionConfiguration) {
        heads = c.heads; headDim = c.headDim; eps = c.rmsNormEps
        _qProj.wrappedValue = EG2Linear(c.hiddenSize, c.heads * c.headDim)
        _kProj.wrappedValue = EG2Linear(c.hiddenSize, c.heads * c.headDim)
        _vProj.wrappedValue = EG2Linear(c.hiddenSize, c.heads * c.headDim)
        _oProj.wrappedValue = EG2Linear(c.heads * c.headDim, c.hiddenSize)
        _qNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
    }

    /// `x`: `[1, L, hidden]`; `cos`/`sin`: `[L, headDim]` in the activation dtype, laid out x-x-y-y.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = x.dim(1)
        func rope(_ t: MLXArray) -> MLXArray {
            // apply_multidimensional_rope: rotate_half within each axis' half of the head.
            let half = headDim / 2, quarter = headDim / 4
            var parts: [MLXArray] = []
            for axis in 0..<2 {
                let part = t[.ellipsis, (axis * half)..<((axis + 1) * half)]
                let rotated = concatenated([-part[.ellipsis, quarter...], part[.ellipsis, ..<quarter]], axis: -1)
                let c = cos[0..., (axis * half)..<((axis + 1) * half)].expandedDimensions(axis: 1)
                let s = sin[0..., (axis * half)..<((axis + 1) * half)].expandedDimensions(axis: 1)
                parts.append(part * c + rotated * s)
            }
            return concatenated(parts, axis: -1)
        }
        let q = rope(qNorm(qProj(x).reshaped(1, length, heads, headDim))).transposed(0, 2, 1, 3)
        let k = rope(kNorm(kProj(x).reshaped(1, length, heads, headDim))).transposed(0, 2, 1, 3)
        let v = eg2NormNoScale(vProj(x).reshaped(1, length, heads, headDim), eps: eps).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1, mask: .none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(1, length, -1))
    }
}

final class EG2VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: EG2Linear
    @ModuleInfo(key: "up_proj") var up: EG2Linear
    @ModuleInfo(key: "down_proj") var down: EG2Linear
    init(_ c: EmbeddingGemma2VisionConfiguration) {
        _gate.wrappedValue = EG2Linear(c.hiddenSize, c.intermediateSize)
        _up.wrappedValue = EG2Linear(c.hiddenSize, c.intermediateSize)
        _down.wrappedValue = EG2Linear(c.intermediateSize, c.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

final class EG2VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: EG2VisionAttention
    @ModuleInfo(key: "mlp") var mlp: EG2VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardNorm: EmbeddingGemma2RMSNorm
    init(_ c: EmbeddingGemma2VisionConfiguration) {
        _attention.wrappedValue = EG2VisionAttention(c)
        _mlp.wrappedValue = EG2VisionMLP(c)
        _inputNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _preFeedforwardNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postFeedforwardNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + postAttentionNorm(attention(inputNorm(x), cos: cos, sin: sin))
        return h + postFeedforwardNorm(mlp(preFeedforwardNorm(h)))
    }
}

final class EG2VisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [EG2VisionLayer]
    init(_ c: EmbeddingGemma2VisionConfiguration) { _layers.wrappedValue = (0..<c.layers).map { _ in EG2VisionLayer(c) } }
}

final class EG2PatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var table: MLXArray
    init(_ c: EmbeddingGemma2VisionConfiguration) {
        _inputProj.wrappedValue = Linear(3 * c.patchSize * c.patchSize, c.hiddenSize, bias: false)
        _table.wrappedValue = MLXArray.zeros([2, c.positionEmbeddingSize, c.hiddenSize])
    }
}

/// The vision tower and its embedder (`vision_tower.*`, `embed_vision.*`).
final class EmbeddingGemma2VisionTower: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: EG2PatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: EG2VisionEncoder
    let config: EmbeddingGemma2VisionConfiguration

    init(_ c: EmbeddingGemma2VisionConfiguration) {
        config = c
        _patchEmbedder.wrappedValue = EG2PatchEmbedder(c)
        _encoder.wrappedValue = EG2VisionEncoder(c)
    }

    /// Axial RoPE tables for row-major positions (x = column, y = row): `[L, headDim]`, layout x-x-y-y.
    func rotary(columns: Int, rows: Int, dtype: DType, headDim: Int? = nil) -> (MLXArray, MLXArray) {
        let dim = headDim ?? config.headDim, quarter = dim / 4, spatial = dim / 2
        let exponents = MLXArray(stride(from: 0, to: spatial, by: 2).map { Float($0) / Float(spatial) })
        let inverse = 1 / MLX.pow(MLXArray(config.ropeTheta), exponents) // [quarter]
        let xs = MLXArray((0..<(rows * columns)).map { Float($0 % columns) }), ys = MLXArray((0..<(rows * columns)).map { Float($0 / columns) })
        let fx = xs.reshaped(-1, 1) * inverse.reshaped(1, quarter), fy = ys.reshaped(-1, 1) * inverse.reshaped(1, quarter)
        let cos = concatenated([MLX.cos(fx), MLX.cos(fx), MLX.cos(fy), MLX.cos(fy)], axis: -1).asType(dtype)
        let sin = concatenated([MLX.sin(fx), MLX.sin(fx), MLX.sin(fy), MLX.sin(fy)], axis: -1).asType(dtype)
        return (cos, sin)
    }

    /// Encoder output of one picture's real patches, `[rows·columns, hidden]` in the weights' dtype.
    func encode(_ patches: EG2Patches, cancellation: NativeCancellation) throws -> MLXArray {
        let dtype = patchEmbedder.inputProj.weight.dtype, count = patches.rows * patches.columns
        var h = patchEmbedder.inputProj((2 * (patches.pixels - 0.5)).asType(dtype)).reshaped(1, count, -1)
        let xs = MLXArray((0..<count).map { Int32($0 % patches.columns) }), ys = MLXArray((0..<count).map { Int32($0 / patches.columns) })
        h = h + (patchEmbedder.table[0].take(xs, axis: 0) + patchEmbedder.table[1].take(ys, axis: 0)).expandedDimensions(axis: 0)
        let (cos, sin) = rotary(columns: patches.columns, rows: patches.rows, dtype: dtype)
        for layer in encoder.layers {
            try cancellation.check()
            h = layer(h, cos: cos, sin: sin)
            eval(h)
        }
        return h[0]
    }

    /// Gemma4VisionPooler: k×k average pooling over the patch grid (float32 accumulation, rounded to the activation
    /// dtype as the maker does), then × √hidden, returned in float32: the scaled magnitude can pass FP16's range.
    func pool(_ hidden: MLXArray, columns: Int, rows: Int) -> MLXArray {
        let k = config.poolingKernelSize, d = hidden.dim(-1)
        let grid = hidden.asType(.float32).reshaped(rows / k, k, columns / k, k, d)
        let pooled = grid.mean(axes: [1, 3]).reshaped(rows * columns / (k * k), d).asType(hidden.dtype)
        return pooled.asType(.float32) * Float(config.hiddenSize).squareRoot()
    }
}

/// EmbeddingGemma2MultimodalEmbedder: RMSNorm without scale, then a linear map into the text width. The norm reads the
/// float32 soft tokens, so a vision magnitude beyond FP16's range is normalized before any half cast.
final class EmbeddingGemma2Embedder: Module {
    @ModuleInfo(key: "embedding_projection") var projection: Linear
    let eps: Float
    init(input: Int, output: Int, eps: Float) {
        self.eps = eps
        _projection.wrappedValue = Linear(input, output, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = MLXFast.rmsNorm(x.asType(.float32), weight: MLXArray.ones([x.dim(-1)], dtype: .float32), eps: eps)
        return projection(normed.asType(projection.weight.dtype))
    }
}

// swiftlint:enable shorthand_operator
