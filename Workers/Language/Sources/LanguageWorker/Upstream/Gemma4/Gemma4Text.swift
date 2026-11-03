// SPDX-License-Identifier: MIT
// PORTED-FROM: ml-explore/mlx-swift-lm bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57 Libraries/MLXVLM/Models/Gemma4.swift (MIT,
// adjacent LICENSE-upstream), split into Gemma4Support/Gemma4Text/Gemma4Vision; types renamed ArsenalGemma4*.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Text

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported).
final class ArsenalGemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported).
final class ArsenalGemma4RMSNormZeroShift: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

final class ArsenalGemma4TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: ArsenalGemma4TextConfiguration, layerIdx: Int) {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        let isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0
        let useDoubleWide = config.useDoubleWideMLP && isKVSharedLayer
        let hiddenDimensions = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class ArsenalGemma4TextRouter: Module {
    let topKExperts: Int
    let config: ArsenalGemma4TextConfiguration
    private let rootSize: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(config: ArsenalGemma4TextConfiguration) {
        guard let numExperts = config.numExperts, let topKExperts = config.topKExperts else {
            fatalError("Gemma 4 MoE router requires numExperts and topKExperts")
        }

        self.topKExperts = topKExperts
        self.config = config
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let normed = MLXFast.rmsNorm(
            x, weight: (scale * rootSize).asType(x.dtype), eps: config.rmsNormEps)

        let scores = proj(normed)

        let topKIndices = MLX.argPartition(scores, kth: -topKExperts, axis: -1)[
            .ellipsis, (-topKExperts)...,
        ]
        var topKWeights = MLX.takeAlong(scores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1)
        // swiftlint:disable:next shorthand_operator
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

final class ArsenalGemma4TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(config: ArsenalGemma4TextConfiguration) {
        guard let numExperts = config.numExperts,
            let moeIntermediateSize = config.moeIntermediateSize
        else {
            fatalError("Gemma 4 MoE experts require numExperts and moeIntermediateSize")
        }

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let topK = topKIndices.dim(-1)

        let expertOutput = switchGLU(
            x.reshaped(batch * length, hidden),
            topKIndices.reshaped(batch * length, topK)
        )
        let weights = topKWeights.reshaped(batch * length, topK).asType(expertOutput.dtype)
        return weightedExpertSum(expertOutput, weights).reshaped(batch, length, hidden)
    }
}

final class ArsenalGemma4ScaledLinear: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = MLXArray.zeros([outFeatures, inFeatures])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x.matmul(weight.transposed())) * scalar
    }
}

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported).
/// Use `kvSharedOnly: true` in the constructor to skip building local K/V
/// projections (the drafter consumes the target's K/V via `sharedKV` instead).
final class ArsenalGemma4TextAttention: Module {
    let config: ArsenalGemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let isKVSharedLayer: Bool
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "k_norm") var kNorm: ArsenalGemma4RMSNormZeroShift?
    @ModuleInfo(key: "v_norm") var vNorm: ArsenalGemma4RMSNormNoScale?
    @ModuleInfo var rope: OffsetLayer

    init(config: ArsenalGemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.headDim =
            layerType == "full_attention" && config.globalHeadDim > 0
            ? config.globalHeadDim : config.headDim
        self.numHeads = config.attentionHeads
        self.useKEqV = config.attentionKEqV && !isSliding
        self.numKVHeads =
            useKEqV ? (config.globalKVHeads ?? config.kvHeads) : config.kvHeads
        self.scale = 1.0

        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self.isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        if !kvSharedOnly {
            self._kProj.wrappedValue = Linear(
                config.hiddenSize, numKVHeads * headDim, bias: false)
            if !useKEqV {
                self._vProj.wrappedValue = Linear(
                    config.hiddenSize, numKVHeads * headDim, bias: false)
            }
            self._kNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
                dimensions: headDim, eps: config.rmsNormEps)
            self._vNorm.wrappedValue = ArsenalGemma4RMSNormNoScale(eps: config.rmsNormEps)
        }
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: headDim, eps: config.rmsNormEps)

        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        sharedKV: ArsenalGemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, ArsenalGemma4SharedKVState?, Int) {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries)

        let currentOffset: Int
        let kvState: ArsenalGemma4SharedKVState?

        if let sharedKV {
            currentOffset = offset ?? 0
            kvState = sharedKV
        } else {
            // Non-`kvSharedOnly` path: K/V projections must be present. If they
            // are nil here the layer was built with `kvSharedOnly: true` and the
            // caller forgot to pass `sharedKV` — a configuration bug.
            guard let kProj, let kNorm, let vNorm else {
                fatalError(
                    "Gemma 4 attention called without sharedKV on a kvSharedOnly layer")
            }
            currentOffset = cache?.offset ?? 0
            var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
            var values =
                if useKEqV {
                    keys
                } else {
                    vProj!(x).reshaped(batch, length, numKVHeads, headDim)
                }
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            values = vNorm(values).transposed(0, 2, 1, 3)
            keys = rope(keys, offset: currentOffset)
            if let quantizedCache = cache as? QuantizedKVCacheProtocol {
                let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
                    keys: keys, values: values)
                kvState = .quantized(
                    keys: quantizedKeys,
                    values: quantizedValues,
                    groupSize: quantizedCache.groupSize,
                    bits: quantizedCache.bits,
                    mode: quantizedCache.mode
                )
            } else {
                if let cache {
                    (keys, values) = cache.update(keys: keys, values: values)
                }
                kvState = .regular(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: currentOffset)

        guard let kvState else {
            fatalError("Gemma 4 attention expected a KV state")
        }
        let localMask = arsenalGemma4AdjustAttentionMask(mask, keyLength: kvState.sequenceLength)

        let output: MLXArray =
            switch kvState {
            case .regular(let keys, let values):
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: localMask
                )
            case .quantized(let keys, let values, let groupSize, let bits, let mode):
                quantizedScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: keys,
                    quantizedValues: values,
                    scale: scale,
                    mask: localMask,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }

        return (
            oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)),
            kvState,
            currentOffset
        )
    }
}

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported).
final class ArsenalGemma4TextDecoderLayer: Module {
    let layerType: String
    let enableMoE: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: ArsenalGemma4TextAttention
    @ModuleInfo var mlp: ArsenalGemma4TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "router") var router: ArsenalGemma4TextRouter?
    @ModuleInfo(key: "experts") var experts: ArsenalGemma4TextExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: ArsenalGemma4RMSNormZeroShift?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: ArsenalGemma4RMSNormZeroShift?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: ArsenalGemma4RMSNormZeroShift?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: ArsenalGemma4RMSNormZeroShift?
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: ArsenalGemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.layerType = config.layerTypes[layerIdx]
        self.enableMoE = config.enableMoEBlock
        self._selfAttention.wrappedValue = ArsenalGemma4TextAttention(
            config: config, layerIdx: layerIdx, kvSharedOnly: kvSharedOnly)
        self._mlp.wrappedValue = ArsenalGemma4TextMLP(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.enableMoEBlock {
            self._router.wrappedValue = ArsenalGemma4TextRouter(config: config)
            self._experts.wrappedValue = ArsenalGemma4TextExperts(config: config)
            self._postFeedforwardLayerNorm1.wrappedValue = ArsenalGemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = ArsenalGemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = ArsenalGemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: ArsenalGemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, ArsenalGemma4SharedKVState?, Int) {
        var residual = x
        var h = inputLayerNorm(x)
        let (attentionOutput, kvState, attentionOffset) = selfAttention(
            h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = attentionOutput
        h = postAttentionLayerNorm(h)
        h = residual + h

        residual = h
        if enableMoE,
            let router,
            let experts,
            let postFeedforwardLayerNorm1,
            let postFeedforwardLayerNorm2,
            let preFeedforwardLayerNorm2
        {
            var dense = preFeedforwardLayerNorm(h)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let (topKIndices, topKWeights) = router(h)
            var sparse = preFeedforwardLayerNorm2(h)
            sparse = experts(sparse, topKIndices: topKIndices, topKWeights: topKWeights)
            sparse = postFeedforwardLayerNorm2(sparse)

            h = dense + sparse
        } else {
            h = preFeedforwardLayerNorm(h)
            h = mlp(h)
        }
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if let perLayerInputGate, let perLayerProjection, let postPerLayerInputNorm,
            let perLayerInput
        {
            residual = h
            var gated = perLayerInputGate(h)
            gated = geluApproximate(gated)
            // swiftlint:disable:next shorthand_operator
            gated = gated * perLayerInput
            gated = perLayerProjection(gated)
            gated = postPerLayerInputNorm(gated)
            h = residual + gated
        }

        return (h * layerScalar, kvState, attentionOffset)
    }
}

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported) (for target-side
/// `embed_tokens` / `embed_scale` / `layer_types` access during drafter bind).
final class ArsenalGemma4TextBackbone: Module {
    let config: ArsenalGemma4TextConfiguration
    let firstKVSharedLayerIdx: Int
    let layerIdxToCacheIdx: [Int]
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int
    let embedScale: Float
    let embedTokensPerLayerScale: Float
    let perLayerProjectionScale: Float
    private let _perLayerInputScale: MLXArray

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [ArsenalGemma4TextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: ArsenalGemma4RMSNormZeroShift?

    init(_ config: ArsenalGemma4TextConfiguration) {
        self.config = config
        self.firstKVSharedLayerIdx = config.hiddenLayers - config.numKVSharedLayers
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(max(config.hiddenSizePerLayerInput, 1)), 0.5)
        self._perLayerInputScale = rsqrt(MLXArray(2.0))

        let concreteLayers = Array(config.layerTypes.prefix(firstKVSharedLayerIdx))
        let sharedFullIdx = concreteLayers.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayers.lastIndex(of: "sliding_attention") ?? 0

        var cacheMap: [Int] = []
        cacheMap.reserveCapacity(config.hiddenLayers)
        for (idx, layerType) in config.layerTypes.enumerated() {
            if idx < firstKVSharedLayerIdx {
                cacheMap.append(idx)
            } else {
                cacheMap.append(layerType == "full_attention" ? sharedFullIdx : sharedSlidingIdx)
            }
        }
        layerIdxToCacheIdx = cacheMap
        firstFullCacheIdx = concreteLayers.firstIndex(of: "full_attention") ?? 0
        firstSlidingCacheIdx = concreteLayers.firstIndex(of: "sliding_attention") ?? 0

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.hiddenLayers).map {
            ArsenalGemma4TextDecoderLayer(config: config, layerIdx: $0)
        }
        self._norm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.hiddenSizePerLayerInput > 0 {
            self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabularySizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            self.perLayerProjectionScale = 1.0
        }

        super.init()
    }

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else {
            fatalError("Per-layer inputs requested for a model without embed_tokens_per_layer")
        }
        let validMask =
            logicalAnd(
                inputIds .>= 0, inputIds .< config.vocabularySizePerLayerInput)
        let tokens = MLX.where(validMask, inputIds, MLXArray.zeros(like: inputIds))
        var result = embedTokensPerLayer(tokens)
        result = (result * MLXArray(embedTokensPerLayerScale, dtype: .float32)).asType(result.dtype)
        return result.reshaped(
            Array(inputIds.shape) + [config.hiddenLayers, config.hiddenSizePerLayerInput]
        )
    }

    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray? {
        guard let perLayerModelProjection, let perLayerProjectionNorm else {
            return nil
        }

        var perLayerProjection = perLayerModelProjection(inputsEmbeds) * perLayerProjectionScale
        perLayerProjection = perLayerProjection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.hiddenLayers, config.hiddenSizePerLayerInput
            ]
        )
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)

        guard let perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + perLayerInputs)
            * _perLayerInputScale.asType(inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        emitDrafterState: Bool = false
    ) -> (hidden: MLXArray, sharedKV: [String: (MLXArray, MLXArray)]?) {
        // Tolerate callers that hand us a 1D `(L,)` token array instead
        // of the canonical 2D `(B, L)` produced by `GemmaVisionProcessor.prepare`.
        // The downstream `perLayerInputs` indexing path (`finalPerLayerInputs[
        // 0..., 0..., idx, 0...]`) requires 4D shapes; with 1D inputs the
        // model otherwise crashes inside `MLXArray.subscript.getter`
        // → `mlx_array_dim` → `_mlx_error`. This expansion is zero-copy
        // and behaves identically when the caller already passed 2D.
        let inputs = inputs.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        let inputsEmbeds = inputsEmbeds.map {
            $0.ndim == 2 ? $0.expandedDimensions(axis: 0) : $0
        }

        let h0: MLXArray
        if let inputsEmbeds {
            h0 = inputsEmbeds
        } else if let inputs {
            let embeddings = embedTokens(inputs)
            h0 = (embeddings * MLXArray(embedScale, dtype: .float32)).asType(embeddings.dtype)
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }

        let processedPerLayerInputs: MLXArray?
        if config.hiddenSizePerLayerInput > 0 {
            if let perLayerInputs {
                processedPerLayerInputs = perLayerInputs
            } else if let inputs {
                processedPerLayerInputs = getPerLayerInputs(inputs)
            } else {
                processedPerLayerInputs = nil
            }
        } else {
            processedPerLayerInputs = nil
        }
        let finalPerLayerInputs = projectPerLayerInputs(h0, perLayerInputs: processedPerLayerInputs)

        let hasExplicitCache = cache != nil
        let localCache =
            cache ?? Array(repeating: nil as KVCache?, count: max(firstKVSharedLayerIdx, 1))
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            fullMask = mask
            slidingMask = mask
        } else {
            let tokenTypeIds = tokenTypeIds.map {
                $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0
            }
            let hasAudioTokens =
                if let tokenTypeIds {
                    (tokenTypeIds .== 3).asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let hasVisualTokens =
                if let tokenTypeIds {
                    ((tokenTypeIds .== 1) | (tokenTypeIds .== 2))
                        .asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let useBidirectionalVision =
                config.useBidirectionalAttention == "vision"
                && tokenTypeIds != nil
                && hasVisualTokens
                && !hasAudioTokens
                && h0.dim(1) > 1

            fullMask = createAttentionMask(
                h: h0,
                cache: firstFullCacheIdx < localCache.count ? localCache[firstFullCacheIdx] : nil,
                returnArray: useBidirectionalVision)
            slidingMask = createAttentionMask(
                h: h0,
                cache: firstSlidingCacheIdx < localCache.count
                    ? localCache[firstSlidingCacheIdx] : nil,
                windowSize: config.slidingWindow,
                returnArray: useBidirectionalVision
            )
            if useBidirectionalVision, let tokenTypeIds {
                fullMask = arsenalGemma4ApplyBlockwiseBidirectionalOverlay(
                    fullMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: nil
                )
                slidingMask = arsenalGemma4ApplyBlockwiseBidirectionalOverlay(
                    slidingMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: config.slidingWindow
                )
            }
        }

        var h = h0
        var intermediates = [(kv: ArsenalGemma4SharedKVState?, offset: Int?)](
            repeating: (nil, nil), count: config.hiddenLayers)
        for (idx, layer) in layers.enumerated() {
            let sourceIdx = layerIdxToCacheIdx[idx]
            let layerCache: KVCache? =
                if idx < firstKVSharedLayerIdx, sourceIdx < localCache.count {
                    localCache[sourceIdx]
                } else {
                    nil
                }
            let layerMask =
                if layer.layerType == "full_attention" {
                    fullMask
                } else {
                    slidingMask
                }
            let layerInput: MLXArray? =
                if let finalPerLayerInputs {
                    finalPerLayerInputs[0..., 0..., idx, 0...]
                } else {
                    nil
                }
            let (output, kvState, attentionOffset) = layer(
                h,
                mask: layerMask,
                cache: layerCache,
                perLayerInput: layerInput,
                sharedKV: idx >= firstKVSharedLayerIdx
                    ? intermediates[sourceIdx].kv : nil,
                offset: idx >= firstKVSharedLayerIdx
                    ? intermediates[sourceIdx].offset : nil
            )
            h = output
            intermediates[idx] = (kvState, attentionOffset)
        }
        let finalHidden = norm(h)

        guard emitDrafterState else {
            return (finalHidden, nil)
        }

        // Walk intermediates from the last layer backward; for each unique
        // `layer_type`, take the first `.regular` K/V encountered. Quantized
        // cases are skipped — the iterator treats absent `sharedKV` as a
        // signal to fall back to single-token generation (R8/R13 limitation,
        // documented).
        var sharedKV: [String: (MLXArray, MLXArray)] = [:]
        var seenTypes = Set<String>()
        let targetTypes: Set<String> = ["full_attention", "sliding_attention"]
        for idx in stride(from: layers.count - 1, through: 0, by: -1) {
            let layerType = layers[idx].layerType
            guard targetTypes.contains(layerType), !seenTypes.contains(layerType) else {
                continue
            }
            if case .regular(let keys, let values) = intermediates[idx].kv {
                sharedKV[layerType] = (keys, values)
                seenTypes.insert(layerType)
            }
            if seenTypes == targetTypes { break }
        }
        // Treat partial coverage (e.g. only one layer_type populated, or
        // quantized cache for the other) as no-emit — iterator falls back.
        return (finalHidden, seenTypes == targetTypes ? sharedKV : nil)
    }
}

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported) (the MTP drafter
/// reaches `embed_tokens` / `embed_scale` / `config.layer_types` through this).
final class ArsenalGemma4TextLanguageModel: Module, KVCacheDimensionProvider {
    let config: ArsenalGemma4TextConfiguration
    let finalLogitSoftcapping: Float?

    @ModuleInfo(key: "model") var model: ArsenalGemma4TextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    var kvHeads: [Int] {
        (0..<config.hiddenLayers).map { idx in
            let layerType = config.layerTypes[idx]
            if config.attentionKEqV && layerType == "full_attention" {
                return config.globalKVHeads ?? config.kvHeads
            } else {
                return config.kvHeads
            }
        }
    }

    init(_ config: ArsenalGemma4TextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._model.wrappedValue = ArsenalGemma4TextBackbone(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        return config.layerTypes.prefix(config.hiddenLayers - config.numKVSharedLayers).map { layerType in
            if layerType == "full_attention" {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: slidingWindow, keep: 0)
            }
        }
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        emitDrafterState: Bool = false
    ) -> LMOutput {
        let (hidden, sharedKV) = model(
            inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache?.map { $0 as KVCache? },
            perLayerInputs: perLayerInputs,
            tokenTypeIds: tokenTypeIds,
            emitDrafterState: emitDrafterState
        )
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else {
            logits = model.embedTokens.asLinear(hidden)
        }
        let softcappedLogits: MLXArray
        if let finalLogitSoftcapping, finalLogitSoftcapping > 0 {
            let scale = MLXArray(finalLogitSoftcapping)
            softcappedLogits = tanh(logits / scale) * scale
        } else {
            softcappedLogits = logits
        }

        guard emitDrafterState, let sharedKV else {
            return LMOutput(logits: softcappedLogits)
        }
        var state = LMOutput.State()
        state[mtpLastHiddenStatesKey] = hidden
        state[mtpSharedKVStatesKey] = sharedKV
        return LMOutput(logits: softcappedLogits, state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count + 1)

        for (key, value) in weights {
            if key.contains("rotary_emb") {
                continue
            }
            // Drop redundant k_proj/v_proj/k_norm for KV-shared layers: they reuse an
            // earlier layer's K/V and own no K projection or K norm, so the module tree
            // has none. QAT checkpoints already omit these; some (PTQ) checkpoints still
            // ship them, and keeping them would be an unexpected weight.
            // Scope: text backbone only — the vision/audio towers share the
            // `layers.N.self_attn.{k,v}_proj` naming, so without these guards the drop
            // would amputate tower layers >= firstKVSharedLayer.
            if firstKVSharedLayer > 0,
                !key.contains("vision_tower"),
                !key.contains("audio_tower"),
                key.contains("self_attn.k_proj")
                    || key.contains("self_attn.v_proj")
                    || key.contains("self_attn.k_norm"),
                let layerIdx = Self.decoderLayerIndex(in: key),
                layerIdx >= firstKVSharedLayer
            {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            }
            if newKey.hasPrefix("language_model."),
                !newKey.hasPrefix("language_model.model."),
                !newKey.hasPrefix("language_model.lm_head.")
            {
                let rest = String(newKey.dropFirst("language_model.".count))
                newKey = "language_model.model.\(rest)"
            }

            if newKey.hasSuffix(".experts.down_proj") {
                newKey = newKey.replacingOccurrences(
                    of: ".experts.down_proj",
                    with: ".experts.switch_glu.down_proj.weight"
                )
            }

            if newKey.hasSuffix(".experts.gate_up_proj") {
                let mid = value.dim(-2) / 2
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.gate_proj.weight"
                    )
                ] = value[.ellipsis, ..<mid, 0...]
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.up_proj.weight"
                    )
                ] = value[.ellipsis, mid..., 0...]
                continue
            }

            sanitized[newKey] = value
        }

        if config.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.hasPrefix("language_model.lm_head.")
            }
        } else if sanitized["language_model.lm_head.weight"] == nil,
            let embedWeight = sanitized["language_model.model.embed_tokens.weight"]
        {
            sanitized["language_model.lm_head.weight"] = embedWeight
        }

        return sanitized
    }

    /// Extract `N` from a weight key shaped like `…layers.N.…`, else nil.
    private static func decoderLayerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let digits = key[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }
}
