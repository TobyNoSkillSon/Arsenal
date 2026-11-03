// SPDX-License-Identifier: MIT
// MiniCPM-4 transformer used four times by VoxCPM2 (text-semantic LM, residual acoustic LM, local encoder, local DiT).
// From mlx-audio-swift e8450402 (MIT), Sources/MLXAudioTTS/Models/VoxCPM2/MiniCPM.swift, checked against the maker's
// voxcpm/modules/minicpm4/model.py. Changes: rotary embedding applied in float32 and cast back to the activation dtype
// (as the maker does), MLX's fused causal mask instead of a float32 -1e9 matrix, rotary tables kept out of the
// parameter tree (the mlx-community release ships them as tensors; they are recomputed here).

import Foundation
import MLX
import MLXFast
import MLXNN

final class VoxRMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dims: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dims])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { MLXFast.rmsNorm(x, weight: weight, eps: eps) }
}

/// LongRoPE tables. With max_position_embeddings == original_max_position_embeddings (VoxCPM2) the scaling factor is 1
/// and the short factors apply, as in the maker's `MiniCPMLongRoPE`.
struct VoxCPMLongRoPE {
    let invFreq: [Float]
    let factors: [Float]
    let scalingFactor: Float

    init(_ config: VoxCPM2LMConfig) {
        let dim = config.kvChannels ?? (config.hiddenSize / config.numAttentionHeads)
        let scale = Float(config.maxPositionEmbeddings) / Float(config.originalMaxPositionEmbeddings)
        scalingFactor = Foundation.sqrt(1 + Foundation.log(max(scale, 1)) / Foundation.log(Float(config.originalMaxPositionEmbeddings)))
        invFreq = (0..<dim / 2).map { 1 / Foundation.pow(config.ropeTheta, Float(2 * $0) / Float(dim)) }
        let chosen = config.maxPositionEmbeddings > config.originalMaxPositionEmbeddings ? config.ropeLongFactor : config.ropeShortFactor
        factors = chosen.isEmpty ? Array(repeating: 1, count: dim / 2) : chosen
    }

    /// cos and sin, float32, shape (L, dim), for positions offset ..< offset + length.
    func callAsFunction(offset: Int, length: Int) -> (MLXArray, MLXArray) {
        let t = MLXArray(offset..<(offset + length)).asType(.float32).expandedDimensions(axis: 1)
        let freqs = t * (MLXArray(invFreq) / MLXArray(factors)).expandedDimensions(axis: 0)
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb) * scalingFactor, MLX.sin(emb) * scalingFactor)
    }
}

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return MLX.concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

/// q, k: (B, H, L, D); cos, sin: (L, D) float32. Computed in float32, returned in the input dtype (maker behaviour).
func voxApplyRotary(_ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let dtype = q.dtype
    let q32 = q.asType(.float32), k32 = k.asType(.float32)
    let qe = q32 * cos + rotateHalf(q32) * sin
    let ke = k32 * cos + rotateHalf(k32) * sin
    return (qe.asType(dtype), ke.asType(dtype))
}

/// Per-layer key/value cache, (B, H_kv, T, D).
typealias VoxKV = (keys: MLXArray, values: MLXArray)

final class VoxAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: VoxCPM2LMConfig) {
        numHeads = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        headDim = config.kvChannels ?? (config.hiddenSize / config.numAttentionHeads)
        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rope: (MLXArray, MLXArray)?, causal: Bool, cache: VoxKV?) -> (MLXArray, VoxKV) {
        let b = x.dim(0), l = x.dim(1)
        var q = qProj(x).reshaped(b, l, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        if let rope { (q, k) = voxApplyRotary(q, k, cos: rope.0, sin: rope.1) }
        if let cache {
            k = MLX.concatenated([cache.keys, k], axis: 2)
            v = MLX.concatenated([cache.values, v], axis: 2)
        }
        // A cached step attends to every earlier position; a prefill is causal (LMs) or full (encoder, DiT).
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = causal && cache == nil && l > 1 ? .causal : .none
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Foundation.sqrt(Float(headDim)), mask: mask)
        return (oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1)), (k, v))
    }
}

final class VoxMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: VoxCPM2LMConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

final class VoxMiniCPMDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VoxAttention
    @ModuleInfo var mlp: VoxMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: VoxRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: VoxRMSNorm
    let residualScale: Float

    init(_ config: VoxCPM2LMConfig) {
        _selfAttn.wrappedValue = VoxAttention(config)
        _mlp.wrappedValue = VoxMLP(config)
        _inputLayernorm.wrappedValue = VoxRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = VoxRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        residualScale = config.useMup ? config.scaleDepth / Foundation.sqrt(Float(config.numHiddenLayers)) : 1
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rope: (MLXArray, MLXArray)?, causal: Bool, cache: VoxKV?) -> (MLXArray, VoxKV) {
        let (h, kv) = selfAttn(inputLayernorm(x), rope: rope, causal: causal, cache: cache)
        let mid = residualScale == 1 ? x + h : x + h * residualScale
        let m = mlp(postAttentionLayernorm(mid))
        return (residualScale == 1 ? mid + m : mid + m * residualScale, kv)
    }
}

final class VoxMiniCPMModel: Module {
    let config: VoxCPM2LMConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    let layers: [VoxMiniCPMDecoderLayer]
    @ModuleInfo var norm: VoxRMSNorm
    private let rope: VoxCPMLongRoPE?

    init(_ config: VoxCPM2LMConfig) {
        self.config = config
        _embedTokens.wrappedValue = config.vocabSize > 0 ? Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize) : nil
        layers = (0..<config.numHiddenLayers).map { _ in VoxMiniCPMDecoderLayer(config) }
        _norm.wrappedValue = VoxRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        rope = config.noRope ? nil : VoxCPMLongRoPE(config)
        super.init()
    }

    /// inputs (B, L, H). `causal` selects the LM (causal prefill, then cached steps) or the bidirectional encoder/DiT.
    func callAsFunction(_ inputs: MLXArray, causal: Bool, cache: [VoxKV]? = nil) -> (MLXArray, [VoxKV]) {
        var h = inputs
        let offset = cache?.first?.keys.dim(2) ?? 0
        let tables = rope.map { $0(offset: offset, length: inputs.dim(1)) }
        var next: [VoxKV] = []
        next.reserveCapacity(layers.count)
        for (i, layer) in layers.enumerated() {
            let kv: VoxKV
            (h, kv) = layer(h, rope: tables, causal: causal, cache: cache?[i])
            next.append(kv)
        }
        return (norm(h), next)
    }
}
