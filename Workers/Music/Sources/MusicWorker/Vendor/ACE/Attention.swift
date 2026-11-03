//
//  Attention.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  ACE-Step's attention: grouped-query heads with RMS-normed queries and
//  keys, RoPE on self-attention only. Used by every encoder layer and by the
//  DiT for both self- and cross-attention. Adapted from mere.run (MIT, see
//  THIRD_PARTY_NOTICES.md) and checked against upstream's MLX decoder.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// Grouped-query attention with QK RMS norm.
final class ACEAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(hiddenSize: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float, theta: Float, bias: Bool) {
        numHeads = heads
        numKVHeads = kvHeads
        self.headDim = headDim
        scale = pow(Float(headDim), -0.5)
        ropeTheta = theta
        _qProj.wrappedValue = Linear(hiddenSize, heads * headDim, bias: bias)
        _kProj.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: bias)
        _vProj.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: bias)
        _oProj.wrappedValue = Linear(heads * headDim, hiddenSize, bias: bias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
    }

    convenience init(_ c: ACEStepConfiguration) {
        self.init(
            hiddenSize: c.hiddenSize, heads: c.numAttentionHeads, kvHeads: c.numKeyValueHeads,
            headDim: c.headDim, eps: c.rmsNormEps, theta: c.ropeTheta, bias: c.attentionBias)
    }

    // Keys and values for cross-attention over `encoder`, reusable across diffusion steps.
    func crossKeysValues(_ encoder: MLXArray) -> (MLXArray, MLXArray) {
        let b = encoder.dim(0), l = encoder.dim(1)
        let k = kNorm(kProj(encoder).reshaped(b, l, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(encoder).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        return (k, v)
    }

    // Self-attention (RoPE from `rope`) or, with `keysValues`, cross-attention.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        rope: RotaryTable? = nil, keysValues: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let b = x.dim(0), l = x.dim(1)
        var q = qNorm(qProj(x).reshaped(b, l, numHeads, headDim)).transposed(0, 2, 1, 3)
        var k: MLXArray
        let v: MLXArray
        if let keysValues {
            (k, v) = keysValues
        } else {
            k = kNorm(kProj(x).reshaped(b, l, numKVHeads, headDim)).transposed(0, 2, 1, 3)
            v = vProj(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
            if let rope {
                q = rope.apply(q)
                k = rope.apply(k)
            }
        }
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

// Rotate-half RoPE tables for one sequence length, as upstream builds them
// (`inv_freq = 1 / base^(2i/d)`, `cos/sin` of `[freqs, freqs]`).
struct RotaryTable {
    let cos: MLXArray
    let sin: MLXArray

    init(length: Int, headDim: Int, theta: Float) {
        let half = headDim / 2
        let exponents = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) })
        let invFreq = 1.0 / MLX.pow(MLXArray(theta), exponents)
        let positions = MLXArray((0..<length).map(Float.init))
        let freqs = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let both = MLX.concatenated([freqs, freqs], axis: -1)
        cos = MLX.cos(both).reshaped(1, 1, length, half * 2)
        sin = MLX.sin(both).reshaped(1, 1, length, half * 2)
    }

    func apply(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let rotated = MLX.concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return (x * cos.asType(x.dtype)) + (rotated * sin.asType(x.dtype))
    }
}

// Qwen3's SwiGLU feed-forward.
final class ACEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

// A pre-norm transformer encoder layer (bidirectional).
final class ACEEncoderLayer: Module {
    let sliding: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: ACEAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: ACEMLP

    init(_ c: ACEStepConfiguration, index: Int) {
        sliding = c.isSliding(layer: index)
        _selfAttn.wrappedValue = ACEAttention(c)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = ACEMLP(hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize)
    }

    func callAsFunction(
        _ x: MLXArray, full: MLXFast.ScaledDotProductAttentionMaskMode,
        slidingMask: MLXFast.ScaledDotProductAttentionMaskMode, rope: RotaryTable
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: sliding ? slidingMask : full, rope: rope)
        return h + mlp(postAttentionLayerNorm(h))
    }
}
