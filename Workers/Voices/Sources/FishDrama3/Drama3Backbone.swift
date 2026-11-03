// SPDX-License-Identifier: MIT
// Drama 3's text backbone: the Qwen3.5-MoE text model driven by frame embeddings instead of token IDs.
// Adapted from mlx-swift-lm bd4b7434 `Libraries/MLXLLM/Models/{Qwen35,Qwen3Next}.swift` (MIT; its layers are internal
// and its model takes token IDs only). Reference: mlx-audio dad2087f `fish_drama3/backbone.py` over the vendored
// `mlx_audio/lm/models/qwen3_5.py`. Every norm with a zero-centred gamma is stored as `1 + w` in float32 and computed
// in float32 (Fp32RMSNorm), as in the reference.
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// RMSNorm with a float32 gamma, computed in float32 and cast back (backbone.py `Fp32RMSNorm`).
final class Fp32RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float) {
        weight = MLXArray.ones([dimensions], dtype: .float32)
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps).asType(x.dtype)
    }
}

/// qwen3_next.py `Qwen3NextRMSNormGated`: rms_norm with gamma, then silu(gate) in float32.
final class Drama3RMSNormGated: Module {
    let weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float) {
        weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ hidden: MLXArray, gate: MLXArray) -> MLXArray {
        let x = MLXFast.rmsNorm(hidden, weight: weight, eps: eps)
        return (silu(gate.asType(.float32)) * x).asType(hidden.dtype)
    }
}

final class Drama3GatedDeltaNet: Module {
    let numVHeads: Int, numKHeads: Int, headKDim: Int, headVDim: Int, keyDim: Int, valueDim: Int, convKernel: Int, convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Drama3RMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: Drama3TextConfig) {
        numVHeads = c.linearValueHeads; numKHeads = c.linearKeyHeads; headKDim = c.linearKeyHeadDim; headVDim = c.linearValueHeadDim
        keyDim = headKDim * numKHeads; valueDim = headVDim * numVHeads; convKernel = c.convKernel; convDim = keyDim * 2 + valueDim
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernel, stride: 1, padding: 0, dilation: 1, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(c.hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(c.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(c.hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(c.hiddenSize, numVHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLXArray.zeros([numVHeads])
        _norm.wrappedValue = Drama3RMSNormGated(dimensions: headVDim, eps: c.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        let batch = inputs.dim(0), length = inputs.dim(1)
        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(batch, length, numVHeads, headVDim)
        let b = inProjB(inputs), a = inProjA(inputs)
        let convState = cache?[0] ?? MLXArray.zeros([batch, convKernel - 1, convDim], dtype: inputs.dtype)
        if let mask { qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, MLXArray(0, dtype: qkv.dtype)) }
        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache { cache[0] = contiguous(convInput[0..., (-(convKernel - 1))..., 0...]) }
        let convOut = silu(conv1d(convInput))
        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = parts[0].reshaped(batch, length, numKHeads, headKDim)
        let k = parts[1].reshaped(batch, length, numKHeads, headKDim)
        let v = parts[2].reshaped(batch, length, numVHeads, headVDim)
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed = MLXArray(invScale * invScale).asType(q.dtype) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed = MLXArray(invScale).asType(k.dtype) * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)
        // The Metal kernel runs only on the GPU; on the CPU (tests) the reference's ops path runs instead, as in gated_delta.py.
        let (out, state) =
            Device.defaultDevice().deviceType == .cpu
            ? drama3GatedDeltaOps(q: qNormed, k: kNormed, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: cache?[1], mask: mask)
            : gatedDeltaUpdate(q: qNormed, k: kNormed, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: cache?[1], mask: mask)
        if let cache {
            cache[1] = state
            cache.advance(length)
        }
        return outProj(norm(out, gate: z).reshaped(batch, length, -1))
    }
}

/// gated_delta.py `gated_delta_ops`: the sequential ops reference of the gated delta recurrence. q, k [B, T, Hk, Dk];
/// v [B, T, Hv, Dv]; a, b [B, T, Hv]; state [B, Hv, Dv, Dk] float32. Returns y [B, T, Hv, Dv] and the new state.
func drama3GatedDeltaOps(
    q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray, state: MLXArray?, mask: MLXArray?
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b)
    let g = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    let batch = q.dim(0), steps = q.dim(1), keyHeads = q.dim(2), keyDim = q.dim(3), valueHeads = v.dim(2), valueDim = v.dim(3)
    var state = state ?? MLXArray.zeros([batch, valueHeads, valueDim, keyDim], dtype: .float32)
    var q = q, k = k
    if valueHeads / keyHeads > 1 {
        q = repeated(q, count: valueHeads / keyHeads, axis: -2)
        k = repeated(k, count: valueHeads / keyHeads, axis: -2)
    }
    var ys: [MLXArray] = []
    for t in 0..<steps {
        let old = state
        let qt = q[0..., t], kt = k[0..., t], vt = v[0..., t]
        // New arrays, never in-place updates: `old` and the caller's cache must keep the previous state.
        // swiftlint:disable:next shorthand_operator
        state = state * expandedDimensions(g[0..., t], axes: [-1, -2])
        let memory = (state * expandedDimensions(kt, axis: -2)).sum(axis: -1)
        let delta = (vt - memory) * expandedDimensions(beta[0..., t], axis: -1)
        // swiftlint:disable:next shorthand_operator
        state = state + expandedDimensions(kt, axis: -2) * expandedDimensions(delta, axis: -1)
        ys.append((state * expandedDimensions(qt, axis: -2)).sum(axis: -1).asType(qt.dtype))
        if let mask { state = MLX.where(expandedDimensions(mask[0..., t], axes: [1, 2, 3]), state, old) }
    }
    return (stacked(ys, axis: 1), state)
}

final class Drama3Attention: Module {
    let heads: Int, kvHeads: Int, scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Fp32RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Fp32RMSNorm
    let rope: RoPE

    init(_ c: Drama3TextConfig) {
        heads = c.attentionHeads; kvHeads = c.kvHeads; scale = pow(Float(c.headDim), -0.5)
        _qProj.wrappedValue = Linear(c.hiddenSize, heads * c.headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, kvHeads * c.headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, kvHeads * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * c.headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = Fp32RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = Fp32RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        rope = RoPE(dimensions: max(1, Int(Float(c.headDim) * c.partialRotary)), traditional: false, base: c.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1)
        let split = qProj(x).reshaped(batch, length, heads, -1).split(parts: 2, axis: -1)
        let gate = split[1].reshaped(batch, length, -1)
        var queries = qNorm(split[0]).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)
        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)
        let output = attentionWithCacheUpdate(queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
            .transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return oProj(output * sigmoid(gate))
    }
}

final class Drama3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    init(dimensions: Int, hidden: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hidden, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

final class Drama3SparseMoe: Module, UnaryLayer {
    let normTopK: Bool, topK: Int
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Drama3MLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ c: Drama3TextConfig) {
        normTopK = c.normTopK; topK = c.expertsPerToken
        _gate.wrappedValue = Linear(c.hiddenSize, c.experts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(inputDims: c.hiddenSize, hiddenDims: c.moeIntermediate, numExperts: c.experts)
        _sharedExpert.wrappedValue = Drama3MLP(dimensions: c.hiddenSize, hidden: c.sharedExpertIntermediate)
        _sharedExpertGate.wrappedValue = Linear(c.hiddenSize, 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = softmax(gate(x), axis: -1, precise: true)
        let kth = gates.dim(-1) - topK
        let inds = argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
        var scores = takeAlong(gates, inds, axis: -1)
        if normTopK { scores /= scores.sum(axis: -1, keepDims: true) }
        let routed = (switchMLP(x, inds) * expandedDimensions(scores, axis: -1)).sum(axis: -2)
        return routed + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

final class Drama3DecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: Drama3Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Drama3GatedDeltaNet?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Fp32RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Fp32RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Drama3SparseMoe

    init(_ c: Drama3TextConfig, index: Int) {
        isLinear = (index + 1) % c.fullAttentionInterval != 0
        if isLinear { _linearAttn.wrappedValue = Drama3GatedDeltaNet(c) } else { _selfAttn.wrappedValue = Drama3Attention(c) }
        _inputLayerNorm.wrappedValue = Fp32RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Fp32RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = Drama3SparseMoe(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXFast.ScaledDotProductAttentionMaskMode, ssmMask: MLXArray?, cache: KVCache?) -> MLXArray {
        let r =
            isLinear
            ? linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
            : selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// qwen3_5.py `Qwen3_5TextModel` with `input_embeddings`; returns the final-normed hidden states.
final class Drama3TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Drama3DecoderLayer]
    let norm: Fp32RMSNorm
    let ssmIndex = 0
    let fullIndex: Int

    init(_ c: Drama3TextConfig) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        layers = (0..<c.layers).map { Drama3DecoderLayer(c, index: $0) }
        norm = Fp32RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        fullIndex = c.fullAttentionInterval - 1
        super.init()
    }

    func callAsFunction(embeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        var h = embeddings
        let attentionMask = createAttentionMask(h: h, cache: cache[fullIndex])
        let ssmMask = createSSMMask(h: h, cache: cache[ssmIndex] as? MambaCache)
        for (layer, c) in zip(layers, cache) {
            h = layer(h, attentionMask: layer.isLinear ? .none : attentionMask, ssmMask: layer.isLinear ? ssmMask : nil, cache: c)
        }
        return norm(h)
    }

    func makeCache() -> [KVCache] { layers.map { $0.isLinear ? MambaCache() : KVCacheSimple() } }
}

/// The reference's `backbone` (a Qwen3.5 TextModel whose vocabulary head was removed): keeps the `backbone.model.*` key path.
final class Drama3Backbone: Module {
    let model: Drama3TextModel
    init(_ c: Drama3TextConfig) {
        model = Drama3TextModel(c)
        super.init()
    }
}
