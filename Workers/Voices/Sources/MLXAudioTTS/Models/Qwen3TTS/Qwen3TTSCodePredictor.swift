import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

// MARK: - Code Predictor Attention

private let compiledCodePredictorSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    compile(shapeless: true) { gate, up in
        silu(gate) * up
    }
}()

final class CodePredictorAttention: Module {
    /// Vireo fast path: fused rotate-half RoPE kernel (bit-identical; see VireoFusedRoPE).
    var fusedRope = false
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Foundation.sqrt(Float(headDim))

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray),
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(batch, seqLen, numKvHeads, headDim)
        var v = vProj(x).reshaped(batch, seqLen, numKvHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)

        let (cosVal, sinVal) = positionEmbeddings
        if fusedRope {
            (q, k) = VireoFusedRoPE.apply(q: q, k: k, cos: cosVal, sin: sinVal)
            q = q.transposed(0, 2, 1, 3)
            k = k.transposed(0, 2, 1, 3)
            v = v.transposed(0, 2, 1, 3)
        } else {
            q = q.transposed(0, 2, 1, 3)
            k = k.transposed(0, 2, 1, 3)
            v = v.transposed(0, 2, 1, 3)
            let cosE = expandedDimensions(cosVal, axis: 1)
            let sinE = expandedDimensions(sinVal, axis: 1)
            q = q * cosE + cpRotateHalf(q) * sinE
            k = k * cosE + cpRotateHalf(k) * sinE
        }

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1))
    }
}

func cpRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

// MARK: - Code Predictor MLP

final class CodePredictorMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3TTSTalkerCodePredictorConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(compiledCodePredictorSwiGLU(gateProj(x), upProj(x)))
    }
}

// MARK: - Code Predictor Decoder Layer

final class CodePredictorDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CodePredictorAttention
    @ModuleInfo var mlp: CodePredictorMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        _selfAttn.wrappedValue = CodePredictorAttention(config: config, layerIdx: layerIdx)
        _mlp.wrappedValue = CodePredictorMLP(config: config)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray),
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        var out = x + selfAttn(inputLayernorm(x), positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        out = out + mlp(postAttentionLayernorm(out))
        return out
    }
}

// MARK: - Code Predictor Model (inner)

final class CodePredictorModel: Module {
    /// Vireo fast path: reuse rotary tables and causal masks per (offset, length, dtype).
    var cacheTables = false { didSet { if !cacheTables { tables = [:] } } }
    private var tables: [String: ((MLXArray, MLXArray), MLXArray?)] = [:]
    let config: Qwen3TTSTalkerCodePredictorConfig
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]
    let layers: [CodePredictorDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    let rotaryEmb: Qwen3TTSRotaryEmbedding

    init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        _codecEmbedding.wrappedValue = (0 ..< config.numCodeGroups - 1).map { _ in
            Embedding(embeddingCount: config.vocabSize, dimensions: talkerHiddenSize)
        }
        self.layers = (0 ..< config.numHiddenLayers).map { CodePredictorDecoderLayer(config: config, layerIdx: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmb = Qwen3TTSRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (inputsEmbeds.dim(0), inputsEmbeds.dim(1), inputsEmbeds.dim(2))

        let offset: Int = if let firstCache = cache?.first {
            firstCache.offset
        } else {
            0
        }

        let posIds: MLXArray
        if let positionIds {
            posIds = positionIds
        } else {
            let pos = MLXArray(Int32(offset) ..< Int32(offset + seqLen)).reshaped(1, seqLen)
            posIds = broadcast(pos, to: [batch, seqLen])
        }

        // Vireo fast path: the code predictor restarts at position 0 for every frame, so the rotary tables and
        // the 2-token causal mask repeat; reuse them (same values, computed once per model and dtype).
        let memoKey = "\(offset)|\(seqLen)|\(batch)|\(inputsEmbeds.dtype)"
        let posEmbeddings: (MLXArray, MLXArray)
        var causalMask = mask
        if cacheTables, positionIds == nil, mask == nil, let hit = tables[memoKey] {
            posEmbeddings = hit.0
            causalMask = hit.1
        } else {
            posEmbeddings = rotaryEmb(inputsEmbeds, positionIds: posIds)
            if causalMask == nil, seqLen > 1 {
                causalMask = MultiHeadAttention.createAdditiveCausalMask(seqLen).asType(inputsEmbeds.dtype)
            }
            if cacheTables, positionIds == nil, mask == nil {
                eval(posEmbeddings.0, posEmbeddings.1)
                if let causalMask { eval(causalMask) }
                tables[memoKey] = (posEmbeddings, causalMask)
            }
        }

        var x = inputsEmbeds
        for (i, layer) in layers.enumerated() {
            x = layer(x, positionEmbeddings: posEmbeddings, mask: causalMask, cache: cache?[i])
            VoiceIntakeCapture.observe("layer-\(i)", x)
        }
        return norm(x)
    }

    func makeCache() -> [any KVCache] {
        layers.map { _ in
            let cache = KVCacheSimple()
            // Vireo fast path (with the tables): a frame holds at most numCodeGroups positions (2 + 14 sub-steps), so
            // a frame-sized buffer instead of 256 slots; each sub-step's slice update then copies 32 KB, not 512 KB,
            // because the frame's graph still holds the previous buffer. Same keys/values, bit-identical.
            if cacheTables { cache.step = config.numCodeGroups }
            return cache
        }
    }
}

// MARK: - Code Predictor (public)

final class Qwen3TTSCodePredictor: Module {
    let config: Qwen3TTSTalkerCodePredictorConfig
    let numCodeGroups: Int
    let talkerHiddenSize: Int

    @ModuleInfo(key: "small_to_mtp_projection") var projection: Linear?
    @ModuleInfo var model: CodePredictorModel
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]

    var codecEmbedding: [Embedding] { model.codecEmbedding }

    init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self.numCodeGroups = config.numCodeGroups
        self.talkerHiddenSize = talkerHiddenSize

        if config.hiddenSize != talkerHiddenSize {
            _projection.wrappedValue = Linear(talkerHiddenSize, config.hiddenSize, bias: true)
        } else {
            _projection.wrappedValue = nil
        }

        _model.wrappedValue = CodePredictorModel(config: config, talkerHiddenSize: talkerHiddenSize)
        _lmHead.wrappedValue = (0 ..< config.numCodeGroups - 1).map { _ in
            Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil,
        generationStep: Int = 0
    ) -> (MLXArray, [any KVCache]?, Int) {
        var embeds = inputsEmbeds
        if let proj = projection {
            embeds = proj(embeds)
        }

        let x = model(embeds, positionIds: positionIds, mask: mask, cache: cache)
        let logits = lmHead[generationStep](x)
        return (logits, cache, generationStep + 1)
    }

    func makeCache() -> [any KVCache] {
        model.makeCache()
    }
}
