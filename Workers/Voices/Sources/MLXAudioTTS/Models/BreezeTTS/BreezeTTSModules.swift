@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN
import Foundation
import SmallMGEMM

final class BreezeAudioEmbedding: Module {
    let numCodebooks: Int
    let vocabSize: Int

    @ModuleInfo(key: "embed_audio_tokens") var embedAudioTokens: Embedding
    @ModuleInfo(key: "audio_embeds_projector") var audioEmbedsProjector: Linear?

    init(numCodebooks: Int, vocabSize: Int, audioEmbedSize: Int, hiddenSize: Int) {
        self.numCodebooks = numCodebooks
        self.vocabSize = vocabSize
        _embedAudioTokens.wrappedValue = Embedding(
            embeddingCount: numCodebooks * vocabSize,
            dimensions: audioEmbedSize
        )
        _audioEmbedsProjector.wrappedValue = audioEmbedSize == hiddenSize
            ? nil
            : Linear(audioEmbedSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ codebooks: MLXArray) -> MLXArray {
        precondition(
            codebooks.ndim == 3 && codebooks.dim(-1) == numCodebooks,
            "Breeze audio codebooks must have shape [batch, time, numCodebooks]"
        )
        let offsets = MLX.arange(numCodebooks).reshaped([1, 1, numCodebooks]) * vocabSize
        var hidden = embedAudioTokens(codebooks + offsets)
        if let audioEmbedsProjector {
            hidden = audioEmbedsProjector(hidden)
        }
        return hidden.sum(axis: -2)
    }
}

final class BreezeBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: BreezeAudioEmbedding
    @ModuleInfo var layers: [Qwen3TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    init(config: BreezeTTSConfig) {
        let backbone = config.backboneConfig
        let qwen = Qwen3Configuration(
            hiddenSize: backbone.hiddenSize,
            hiddenLayers: backbone.numHiddenLayers,
            intermediateSize: backbone.intermediateSize,
            attentionHeads: backbone.numAttentionHeads,
            kvHeads: backbone.numKeyValueHeads,
            headDim: backbone.headDim,
            vocabularySize: backbone.vocabSize,
            rmsNormEps: backbone.rmsNormEps,
            ropeTheta: backbone.ropeTheta,
            ropeScaling: backbone.ropeScaling,
            maxPositionEmbeddings: backbone.maxPositionEmbeddings
        )
        _embedTokens.wrappedValue = BreezeAudioEmbedding(
            numCodebooks: config.numCodebooks,
            vocabSize: config.audioVocabSize,
            audioEmbedSize: config.audioEmbedSize,
            hiddenSize: backbone.hiddenSize
        )
        _layers.wrappedValue = (0..<backbone.numHiddenLayers).map { _ in Qwen3TransformerBlock(qwen) }
        _norm.wrappedValue = RMSNorm(dimensions: backbone.hiddenSize, eps: backbone.rmsNormEps)
    }

    func callAsFunction(
        inputIDs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        precondition((inputIDs == nil) != (inputEmbeddings == nil), "Pass input IDs or embeddings")
        var hidden = inputEmbeddings ?? embedTokens(inputIDs!)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
            VoiceIntakeCapture.observe("layer-\(index)", hidden)
        }
        return norm(hidden)
    }

    func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }
}

final class BreezeDepthAttention: Module {
    let config: BreezeDepthDecoderConfig
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear

    init(config: BreezeDepthDecoderConfig) {
        self.config = config
        self.scale = Foundation.pow(Float(config.headDim), -0.5)
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta, scale: 1)
        _query.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        _output.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache? = nil) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let offset = cache?.offset ?? 0
        let queries = rope(
            query(x).reshaped([batch, length, config.numAttentionHeads, config.headDim])
                .transposed(0, 2, 1, 3),
            offset: offset
        )
        var keys = rope(
            key(x).reshaped([batch, length, config.numKeyValueHeads, config.headDim])
                .transposed(0, 2, 1, 3),
            offset: offset
        )
        var values = value(x)
            .reshaped([batch, length, config.numKeyValueHeads, config.headDim])
            .transposed(0, 2, 1, 3)
        if let cache { (keys, values) = cache.update(keys: keys, values: values) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: length > 1 ? .causal : .none
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped([batch, length, -1]))
    }
}

final class BreezeDepthMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(config: BreezeDepthDecoderConfig) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class BreezeDepthLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: BreezeDepthAttention
    @ModuleInfo var mlp: BreezeDepthMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(config: BreezeDepthDecoderConfig) {
        _attention.wrappedValue = BreezeDepthAttention(config: config)
        _mlp.wrappedValue = BreezeDepthMLP(config: config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache? = nil) -> MLXArray {
        let attended = x + attention(inputNorm(x), cache: cache)
        return attended + mlp(postAttentionNorm(attended))
    }
}

final class BreezeDepthModel: Module {
    let config: BreezeDepthDecoderConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "backbone_hidden_state_projector") var backboneProjector: Linear?
    @ModuleInfo(key: "inputs_embeds_projector") var inputProjector: Linear
    @ModuleInfo var layers: [BreezeDepthLayer]
    @ModuleInfo var norm: RMSNorm

    init(config: BreezeDepthDecoderConfig) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.numCodebooks * config.vocabSize,
            dimensions: config.audioEmbedSize
        )
        _backboneProjector.wrappedValue = config.backboneHiddenSize == config.audioEmbedSize
            ? nil
            : Linear(config.backboneHiddenSize, config.audioEmbedSize, bias: false)
        _inputProjector.wrappedValue = Linear(config.audioEmbedSize, config.hiddenSize, bias: false)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in BreezeDepthLayer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokenIDs: MLXArray, backboneHiddenState: MLXArray) -> MLXArray {
        precondition(tokenIDs.ndim == 2, "Depth token IDs must have shape [batch, time]")
        precondition(backboneHiddenState.ndim == 2, "Backbone hidden state must have shape [batch, hidden]")
        precondition(tokenIDs.dim(0) == backboneHiddenState.dim(0), "Depth batch sizes must match")
        let length = tokenIDs.dim(1)
        let positions = MLX.maximum(MLX.arange(length) - 1, MLXArray(0))
        var embeddings = embedTokens(tokenIDs + positions.reshaped([1, length]) * config.vocabSize)
        var backbone = backboneHiddenState
        if let backboneProjector {
            backbone = backboneProjector(backbone)
        }
        embeddings = concatenated([
            backbone.expandedDimensions(axis: 1),
            embeddings[0..., 1..., 0...],
        ], axis: 1)
        var hidden = inputProjector(embeddings)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden)
            VoiceIntakeCapture.observeLastToken("layer-\(index)", hidden)
        }
        return norm(hidden)
    }

    // Vireo (fast path candidate): incremental depth decoding with a KV cache. `prime` runs the first two
    // positions (projected backbone state, first-codebook embedding); `step` one new token at position `p`
    // (= cache offset). Same maths as recomputing the whole prefix; different GEMM shapes (M = batch).
    func makeCache() -> [KVCache] {
        layers.map { _ in
            let cache = KVCacheSimple()
            // Fused step: frame-sized buffers (prime writes 2 positions, each step one; the kernel returns new ones).
            if fused { cache.step = VireoBreezeFused.capacity }
            return cache
        }
    }

    /// Vireo fast path (`VIREO_BREEZE_FUSED`, see `VireoBreezeFused`): one-token steps with fused kernels.
    var fused = false
    /// Opt-in (VIREO_BREEZE_SMALLM=1): in the fused step, the FFN of dense (BF16/FP16) layers through the family's
    /// SmallMGEMM GEMV: gate and up in one pass with the SiLU-gate epilogue, down with the residual epilogue.
    var smallMFFN = false

    func prime(firstToken: MLXArray, backboneHiddenState: MLXArray, cache: [KVCache]) -> MLXArray {
        var backbone = backboneHiddenState
        if let backboneProjector { backbone = backboneProjector(backbone) }
        let first = embedTokens(firstToken.reshaped([-1, 1]))
        let embeddings = concatenated([backbone.expandedDimensions(axis: 1), first], axis: 1)
        return run(inputProjector(embeddings), cache: cache)
    }

    func step(token: MLXArray, position: Int, cache: [KVCache]) -> MLXArray {
        let input = inputProjector(embedTokens(token.reshaped([-1, 1]) + (position - 1) * config.vocabSize))
        if fused, let simple = cache as? [KVCacheSimple], let first = simple.first,
           first.innerState().first?.dim(2) == VireoBreezeFused.capacity, first.offset < VireoBreezeFused.capacity {
            return runFused(input.squeezed(axis: 1), cache: simple)
        }
        return run(input, cache: cache)
    }

    /// The fused one-token step: per layer RMSNorm, q/k/v, fused attention (writes the next KV buffers), o, add,
    /// RMSNorm, gate/up, compiled SiLU-gate, down, add; then the final norm. `x` is `[rows, hidden]`.
    private func runFused(_ x: MLXArray, cache: [KVCacheSimple]) -> MLXArray {
        var residual = x
        for (index, pair) in zip(layers, cache).enumerated() {
            let (layer, layerCache) = pair
            let normed = layer.inputNorm(residual)
            let attention = layer.attention
            let state = layerCache.innerState()
            let offset = layerCache.offset
            let (attended, keys, values) = VireoBreezeFused.attention(
                q: attention.query(normed), k: attention.key(normed), v: attention.value(normed),
                keys: state[0], values: state[1], offset: offset,
                queryHeads: config.numAttentionHeads, kvHeads: config.numKeyValueHeads, headDim: config.headDim,
                ropeBase: config.ropeTheta)
            layerCache.state = [keys, values]
            layerCache.offset = offset + 1
            residual = residual + attention.output(attended)
            let mlp = layer.mlp
            let normed2 = layer.postAttentionNorm(residual)
            if smallMFFN, let next = Self.smallMFFN(normed2, residual: residual, mlp: mlp) {
                residual = next
            } else {
                residual = residual + mlp.down(VireoBreezeFused.siluGate(mlp.gate(normed2), mlp.up(normed2)))
            }
            VoiceIntakeCapture.observeLastToken("layer-\(index)", residual)
        }
        return norm(residual)
    }

    /// residual + down(silu(gate(x)) · up(x)) through SmallMGEMM's GEMV (dense weights only); nil → stock ops.
    static func smallMFFN(_ x: MLXArray, residual: MLXArray, mlp: BreezeDepthMLP) -> MLXArray? {
        guard !(mlp.gate is QuantizedLinear), !(mlp.up is QuantizedLinear), !(mlp.down is QuantizedLinear),
              mlp.gate.bias == nil, mlp.up.bias == nil, mlp.down.bias == nil,
              let hidden = SmallMGEMM.matmul(x, .dense(mlp.gate.weight), epilogue: .siluGate(up: .dense(mlp.up.weight))),
              let out = SmallMGEMM.matmul(hidden, .dense(mlp.down.weight), epilogue: .residual(residual))
        else { return nil }
        return out.reshaped(residual.shape)
    }


    private func run(_ input: MLXArray, cache: [KVCache]) -> MLXArray {
        var hidden = input
        for (index, pair) in zip(layers, cache).enumerated() {
            hidden = pair.0(hidden, cache: pair.1)
            VoiceIntakeCapture.observeLastToken("layer-\(index)", hidden)
        }
        return norm(hidden[0..., (-1)..., 0...]).squeezed(axis: 1)
    }
}

final class BreezeCodebooksHead: Module {
    @ModuleInfo var heads: [Linear]

    init(headCount: Int, hiddenSize: Int, vocabSize: Int) {
        _heads.wrappedValue = (0..<headCount).map { _ in Linear(hiddenSize, vocabSize, bias: false) }
    }
}

final class BreezeDepthDecoder: Module {
    @ModuleInfo var model: BreezeDepthModel
    @ModuleInfo(key: "codebooks_head") var codebooksHead: BreezeCodebooksHead

    init(config: BreezeDepthDecoderConfig) {
        _model.wrappedValue = BreezeDepthModel(config: config)
        _codebooksHead.wrappedValue = BreezeCodebooksHead(
            headCount: config.numCodebooks - 1,
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize
        )
    }

    func nextLogits(tokenIDs: MLXArray, backboneHiddenState: MLXArray) -> MLXArray {
        let headIndex = tokenIDs.dim(1) - 2
        precondition(headIndex >= 0 && headIndex < codebooksHead.heads.count, "Invalid depth head")
        let hidden = model(tokenIDs, backboneHiddenState: backboneHiddenState)[0..., -1, 0...]
        return codebooksHead.heads[headIndex](hidden)
    }

    /// Logits for depth head `headIndex` from a last hidden state (`BreezeDepthModel.prime`/`step`).
    func logits(_ hidden: MLXArray, headIndex: Int) -> MLXArray {
        codebooksHead.heads[headIndex](hidden)
    }
}
