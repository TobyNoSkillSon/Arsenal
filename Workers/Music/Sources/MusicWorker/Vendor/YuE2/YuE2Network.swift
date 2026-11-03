//
//  YuE2Network.swift
//  YuE2Music
//
//  Created by David Sherlock on 2026.
//
//  YuE2's AR–NAR mixture of transformers (modeling_yue2.py): 28 Qwen3-shaped
//  layers, each with two of everything — an AR path (causal, writes the
//  score and the 25 Hz song tokens) and a NAR path (bidirectional, predicts
//  flow-matching velocities for the 64-channel acoustic latents) — over one
//  shared attention. Heads: the LM head, llm2vae / vae2llm, a timestep MLP
//  and a fixed sinusoidal position table for latent frames. bf16, as shipped.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// RMS norm as upstream writes it: the mean square in float32, cast back, times the weight.
final class YuERMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let scale = MLX.rsqrt((f * f).mean(axis: -1, keepDims: true) + eps).asType(x.dtype)
        return x * scale * weight
    }
}

// Rotate-half RoPE from float32 angles, cast to the activations' type (upstream's `_apply_rotary`).
struct YuERotary {
    let cos: MLXArray // [1, L, 1, D/2]
    let sin: MLXArray

    init(positions: Range<Int>, headDim: Int, base: Float) {
        let exponents = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) })
        let invFreq = 1.0 / MLX.pow(MLXArray(base), exponents)
        let pos = MLXArray(positions.map(Float.init)).expandedDimensions(axis: 1)
        let angles = pos * invFreq.expandedDimensions(axis: 0)
        cos = MLX.cos(angles).reshaped(1, positions.count, 1, headDim / 2)
        sin = MLX.sin(angles).reshaped(1, positions.count, 1, headDim / 2)
    }

    // `x` `[B, L, H, D]`.
    func apply(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let c = cos.asType(x.dtype), s = sin.asType(x.dtype)
        let x1 = x[.ellipsis, ..<half], x2 = x[.ellipsis, half...]
        return MLX.concatenated([x1 * c - x2 * s, x2 * c + x1 * s], axis: -1)
    }
}

final class YuEAttention: Module {
    let heads: Int, kvHeads: Int
    let headDim = 128

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: YuERMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: YuERMSNorm

    init(hidden: Int, eps: Float) {
        heads = hidden / 128; kvHeads = heads / 2
        _qProj.wrappedValue = Linear(hidden, heads * 128, bias: false)
        _kProj.wrappedValue = Linear(hidden, kvHeads * 128, bias: false)
        _vProj.wrappedValue = Linear(hidden, kvHeads * 128, bias: false)
        _oProj.wrappedValue = Linear(heads * 128, hidden, bias: false)
        _qNorm.wrappedValue = YuERMSNorm(128, eps: eps)
        _kNorm.wrappedValue = YuERMSNorm(128, eps: eps)
    }

    // Q `[B, H, L, D]`, K and V `[B, Hkv, L, D]` with RoPE applied.
    func project(_ x: MLXArray, rope: YuERotary) -> (MLXArray, MLXArray, MLXArray) {
        let b = x.dim(0), l = x.dim(1)
        let q = rope.apply(qNorm(qProj(x).reshaped(b, l, heads, headDim))).transposed(0, 2, 1, 3)
        let k = rope.apply(kNorm(kProj(x).reshaped(b, l, kvHeads, headDim))).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        return (q, k, v)
    }

    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: pow(Float(headDim), -0.5), mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(q.dim(0), q.dim(2), -1))
    }
}

final class YuEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hidden: Int, intermediate: Int) {
        _gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

// One mixture-of-transformers layer.
final class YuELayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: YuERMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: YuEAttention
    @ModuleInfo(key: "nar_input_layernorm") var narInputLayerNorm: YuERMSNorm
    @ModuleInfo(key: "nar_self_attn") var narSelfAttn: YuEAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: YuERMSNorm
    @ModuleInfo(key: "mlp") var mlp: YuEMLP
    @ModuleInfo(key: "nar_pre_mlp_layernorm") var narPreMLPLayerNorm: YuERMSNorm
    @ModuleInfo(key: "nar_mlp") var narMLP: YuEMLP

    init(hidden: Int, intermediate: Int, eps: Float) {
        _inputLayerNorm.wrappedValue = YuERMSNorm(hidden, eps: eps)
        _selfAttn.wrappedValue = YuEAttention(hidden: hidden, eps: eps)
        _narInputLayerNorm.wrappedValue = YuERMSNorm(hidden, eps: eps)
        _narSelfAttn.wrappedValue = YuEAttention(hidden: hidden, eps: eps)
        _postAttentionLayerNorm.wrappedValue = YuERMSNorm(hidden, eps: eps)
        _mlp.wrappedValue = YuEMLP(hidden: hidden, intermediate: intermediate)
        _narPreMLPLayerNorm.wrappedValue = YuERMSNorm(hidden, eps: eps)
        _narMLP.wrappedValue = YuEMLP(hidden: hidden, intermediate: intermediate)
    }

    // The AR path with a KV cache (`mask` causal for a prefill, none for one token).
    func autoregressive(_ x: MLXArray, rope: YuERotary, cache: KVCache, mask: MLXFast.ScaledDotProductAttentionMaskMode)
        -> MLXArray
    {
        let (q, k, v) = selfAttn.project(inputLayerNorm(x), rope: rope)
        let (keys, values) = cache.update(keys: k, values: v)
        let h = x + selfAttn.attend(q, keys, values, mask: mask)
        return h + mlp(postAttentionLayerNorm(h))
    }

    // The AR path's prefill for the acoustic stage: causal over the whole
    // prefix, returning the output and this layer's keys and values.
    func prefill(_ x: MLXArray, rope: YuERotary) -> (MLXArray, MLXArray, MLXArray) {
        let (q, k, v) = selfAttn.project(inputLayerNorm(x), rope: rope)
        let h = x + selfAttn.attend(q, k, v, mask: .causal)
        return (h + mlp(postAttentionLayerNorm(h)), k, v)
    }

    // The NAR path: latent frames attend (unmasked) over the AR keys and their own.
    func acoustic(_ x: MLXArray, rope: YuERotary, arKeys: MLXArray, arValues: MLXArray) -> MLXArray {
        let (q, k, v) = narSelfAttn.project(narInputLayerNorm(x), rope: rope)
        let keys = MLX.concatenated([arKeys, k], axis: 2), values = MLX.concatenated([arValues, v], axis: 2)
        let h = x + narSelfAttn.attend(q, keys, values, mask: .none)
        return h + narMLP(narPreMLPLayerNorm(h))
    }
}

// The checkpoint: backbone (`model.`), heads and NAR auxiliaries.
final class YuE2Network: Module {
    @ModuleInfo(key: "model") var backbone: YuEBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "llm2vae") var llm2vae: Linear
    @ModuleInfo(key: "vae2llm") var vae2llm: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: YuETimeEmbedder
    @ModuleInfo(key: "latent_pos_embed") var latentPositions: YuELatentPositions

    let hidden: Int

    init(hidden: Int = 2048, layers: Int = 28, intermediate: Int = 6144) {
        self.hidden = hidden
        _backbone.wrappedValue = YuEBackbone(hidden: hidden, layers: layers, intermediate: intermediate)
        _lmHead.wrappedValue = Linear(hidden, YuE2Protocol.vocabularySize, bias: false)
        _llm2vae.wrappedValue = Linear(hidden, 64, bias: true)
        _vae2llm.wrappedValue = Linear(64, hidden, bias: true)
        _timeEmbedder.wrappedValue = YuETimeEmbedder(hidden: hidden)
        _latentPositions.wrappedValue = YuELatentPositions(frames: 24_576, hidden: hidden)
    }
}

final class YuEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [YuELayer]
    @ModuleInfo(key: "norm") var norm: YuERMSNorm

    init(hidden: Int = 2048, layers: Int = 28, intermediate: Int = 6144) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: YuE2Protocol.vocabularySize, dimensions: hidden)
        _layers.wrappedValue = (0..<layers).map { _ in YuELayer(hidden: hidden, intermediate: intermediate, eps: 1e-6) }
        _norm.wrappedValue = YuERMSNorm(hidden, eps: 1e-6)
    }
}

// Sinusoidal timestep (no ×1000: t is 0–1) → Linear → SiLU → Linear
// (checkpoint `time_embedder.mlp.0` / `.mlp.2`, renamed on load).
final class YuETimeEmbedder: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(hidden: Int) {
        _linear1.wrappedValue = Linear(256, hidden, bias: true)
        _linear2.wrappedValue = Linear(hidden, hidden, bias: true)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let freqs = MLX.exp(-Float(Foundation.log(10_000.0)) * MLXArray(0..<128).asType(.float32) / 128)
        let args = t.asType(.float32).expandedDimensions(axis: -1) * freqs
        let embedded = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1).asType(.float16)
        return linear2(silu(linear1(embedded)))
    }
}

// The fixed position table, shipped in the checkpoint as `pe`.
final class YuELatentPositions: Module {
    let pe: MLXArray

    init(frames: Int, hidden: Int) {
        pe = MLXArray.zeros([frames, hidden], dtype: .float16)
    }
}
