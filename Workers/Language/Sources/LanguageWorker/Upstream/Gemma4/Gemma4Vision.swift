// SPDX-License-Identifier: MIT
// PORTED-FROM: ml-explore/mlx-swift-lm bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57 Libraries/MLXVLM/Models/Gemma4.swift (MIT,
// adjacent LICENSE-upstream), split into Gemma4Support/Gemma4Text/Gemma4Vision; types renamed ArsenalGemma4*.
// Local changes follow Transformers modeling_gemma4.py / image_processing_gemma4.py: each image is encoded at its own
// aspect-preserving size (multiples of 48 px within the 2,520-patch budget) and yields (rows/3)·(columns/3) soft tokens,
// where the pinned port forced 800×800 (a 50-patch side that collides in the 3×3 pooler) and required exactly 280; the
// pooler averages by position and scales by sqrt(hidden) in float32 like HF; the patch projection reads its half dtype
// from the position table because int8/int4 tiers pack input_proj.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

// MARK: - Vision

private final class ArsenalGemma4ClippableLinear: Module, UnaryLayer {
    let useClipping: Bool

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?

    init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool) {
        self.useClipping = useClipping
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)
        if useClipping {
            self._inputMin.wrappedValue = MLXArray(-Float.infinity)
            self._inputMax.wrappedValue = MLXArray(Float.infinity)
            self._outputMin.wrappedValue = MLXArray(-Float.infinity)
            self._outputMax.wrappedValue = MLXArray(Float.infinity)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let clippedInput =
            if let inputMin, let inputMax {
                clip(x, min: inputMin, max: inputMax)
            } else {
                x
            }
        let projected = linear(clippedInput)
        if let outputMin, let outputMax {
            return clip(projected, min: outputMin, max: outputMax)
        }
        return projected
    }
}

private final class ArsenalGemma4VisionRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat.square(), axis: -1, keepDims: true)
        let normalized = xFloat * rsqrt(variance + eps)
        return (normalized * weight.asType(.float32)).asType(x.dtype)
    }
}

private final class ArsenalGemma4VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat.square(), axis: -1, keepDims: true)
        return (xFloat * rsqrt(variance + eps)).asType(x.dtype)
    }
}

private final class ArsenalGemma4VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let ropeBaseFrequency: Float

    @ModuleInfo(key: "q_proj") var qProj: ArsenalGemma4ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: ArsenalGemma4ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: ArsenalGemma4ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: ArsenalGemma4ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: ArsenalGemma4VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: ArsenalGemma4VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var vNorm: ArsenalGemma4VisionRMSNormNoScale

    init(config: ArsenalGemma4VisionConfiguration) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.keyValueHeads
        self.headDim = config.headDim
        self.hiddenSize = config.hiddenSize
        self.ropeBaseFrequency = config.ropeParameters["rope_theta"]?.asFloat() ?? 100.0

        self._qProj.wrappedValue = ArsenalGemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._kProj.wrappedValue = ArsenalGemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numKVHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._vProj.wrappedValue = ArsenalGemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numKVHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._oProj.wrappedValue = ArsenalGemma4ClippableLinear(
            inFeatures: numHeads * headDim,
            outFeatures: hiddenSize,
            useClipping: config.useClippedLinears
        )
        self._qNorm.wrappedValue = ArsenalGemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = ArsenalGemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = ArsenalGemma4VisionRMSNormNoScale(eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
        var values = vProj(x).reshaped(batch, length, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)
        values = vNorm(values)

        queries = arsenalGemma4ApplyMultiDimensionalRoPE(
            queries, positions: positions, baseFrequency: ropeBaseFrequency)
        keys = arsenalGemma4ApplyMultiDimensionalRoPE(
            keys, positions: positions, baseFrequency: ropeBaseFrequency)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let mask {
                .array(mask)
            } else {
                .none
            }
        let output = arsenalGemma4EnsureFusedSDPA(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

private final class ArsenalGemma4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: ArsenalGemma4ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: ArsenalGemma4ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: ArsenalGemma4ClippableLinear

    init(config: ArsenalGemma4VisionConfiguration) {
        self._gateProj.wrappedValue = ArsenalGemma4ClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useClipping: config.useClippedLinears
        )
        self._upProj.wrappedValue = ArsenalGemma4ClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useClipping: config.useClippedLinears
        )
        self._downProj.wrappedValue = ArsenalGemma4ClippableLinear(
            inFeatures: config.intermediateSize,
            outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private final class ArsenalGemma4VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: ArsenalGemma4VisionAttention
    @ModuleInfo var mlp: ArsenalGemma4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: ArsenalGemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: ArsenalGemma4RMSNormZeroShift

    init(config: ArsenalGemma4VisionConfiguration) {
        self._selfAttention.wrappedValue = ArsenalGemma4VisionAttention(config: config)
        self._mlp.wrappedValue = ArsenalGemma4VisionMLP(config: config)
        self._inputLayerNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = ArsenalGemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attentionOutput = selfAttention(normed, positions: positions, mask: mask)
        let h = x + postAttentionLayerNorm(attentionOutput)
        let ff = mlp(preFeedforwardLayerNorm(h))
        return h + postFeedforwardLayerNorm(ff)
    }
}

private final class ArsenalGemma4VisionPatchEmbedder: Module {
    let patchSize: Int
    let hiddenSize: Int
    let positionEmbeddingSize: Int

    @ModuleInfo(key: "input_proj") var inputProjection: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    init(config: ArsenalGemma4VisionConfiguration) {
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize
        self.positionEmbeddingSize = config.positionEmbeddingSize
        self._inputProjection.wrappedValue = Linear(
            3 * patchSize * patchSize, hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = MLXArray.ones([
            2, positionEmbeddingSize, hiddenSize
        ])
        super.init()
    }

    /// `patches`: `[1, n, 3·P·P]` patch pixels in [0, 1] ordered (py, px, c) like image_processing_gemma4.py;
    /// `positions`: `[1, n, 2]` (x, y) patch indices.
    func callAsFunction(_ patches: MLXArray, positions: MLXArray) -> MLXArray {
        // Int8/int4 tiers pack input_proj (uint32); the position table keeps the half dtype.
        let hiddenStates = inputProjection((2 * (patches - 0.5)).asType(positionEmbeddingTable.dtype))
        let batch = positions.dim(0)
        let seqLen = positions.dim(1)

        let xIndices = positions[0..., 0..., 0].flattened().asType(.int32)
        let yIndices = positions[0..., 0..., 1].flattened().asType(.int32)
        let xEmbeddings = take(positionEmbeddingTable[0], xIndices, axis: 0)
            .reshaped(batch, seqLen, hiddenSize)
        let yEmbeddings = take(positionEmbeddingTable[1], yIndices, axis: 0)
            .reshaped(batch, seqLen, hiddenSize)
        return hiddenStates + xEmbeddings + yEmbeddings
    }
}

/// Transformers ArsenalGemma4VisionPooler (modeling_gemma4.py): average each k×k block of patch positions, then scale by
/// sqrt(hidden_size) in float32 because the scaled activations can exceed the float16 range.
private final class ArsenalGemma4VisionPooler: Module {
    let kernel: Int
    let rootHiddenSize: Float

    init(config: ArsenalGemma4VisionConfiguration) {
        self.kernel = config.poolingKernelSize
        self.rootHiddenSize = pow(Float(config.hiddenSize), 0.5)
        super.init()
    }

    /// `hiddenStates`: `[1, rows·columns, D]` in row-major patch order; rows and columns are multiples of the kernel.
    /// Returns `[1, (rows/k)·(columns/k), D]` in float32, ordered like HF's `x//k + (columns//k)·(y//k)` kernel index.
    func callAsFunction(_ hiddenStates: MLXArray, columns: Int, rows: Int) -> MLXArray {
        ArsenalGemma4Pooling.pool(hiddenStates, columns: columns, rows: rows, kernel: kernel, scale: rootHiddenSize)
    }
}

enum ArsenalGemma4Pooling {
    /// k×k average by patch position, then `scale` (sqrt(hidden_size)), both in float32.
    static func pool(_ hiddenStates: MLXArray, columns: Int, rows: Int, kernel: Int, scale: Float) -> MLXArray {
        let width = hiddenStates.dim(-1)
        let blocks = hiddenStates.asType(.float32).reshaped(rows / kernel, kernel, columns / kernel, kernel, width)
        let pooled = blocks.mean(axes: [1, 3]).reshaped(1, (rows / kernel) * (columns / kernel), width)
        return pooled * MLXArray(scale, dtype: .float32)
    }
}

private final class ArsenalGemma4VisionTransformerModel: Module {
    @ModuleInfo(key: "layers") var layers: [ArsenalGemma4VisionTransformerBlock]

    init(config: ArsenalGemma4VisionConfiguration) {
        self._layers.wrappedValue = (0..<config.hiddenLayers).map { _ in
            ArsenalGemma4VisionTransformerBlock(config: config)
        }
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = hiddenStates
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

private final class ArsenalGemma4VisionModel: Module {
    let config: ArsenalGemma4VisionConfiguration
    let patchSize: Int
    let poolingKernelSize: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: ArsenalGemma4VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: ArsenalGemma4VisionTransformerModel
    @ModuleInfo(key: "pooler") var pooler: ArsenalGemma4VisionPooler
    @ModuleInfo(key: "std_bias") var standardizationBias: MLXArray?
    @ModuleInfo(key: "std_scale") var standardizationScale: MLXArray?

    init(config: ArsenalGemma4VisionConfiguration) {
        self.config = config
        self.patchSize = config.patchSize
        self.poolingKernelSize = config.poolingKernelSize
        self._patchEmbedder.wrappedValue = ArsenalGemma4VisionPatchEmbedder(config: config)
        self._encoder.wrappedValue = ArsenalGemma4VisionTransformerModel(config: config)
        self._pooler.wrappedValue = ArsenalGemma4VisionPooler(config: config)
        if config.standardize {
            self._standardizationBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            self._standardizationScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }
        super.init()
    }

    /// One image of `columns × rows` patches: `patches` `[1, n, 3·P·P]`, `positions` `[1, n, 2]`. Returns its
    /// `n / k²` soft tokens `[n / k², D]` in the working dtype. Transformers pads every image to the processor's
    /// patch budget, masks the padding out of attention and strips it after pooling; encoding the real patches
    /// alone gives the same rows.
    func callAsFunction(_ patches: MLXArray, positions: MLXArray, columns: Int, rows: Int) -> MLXArray {
        var hiddenStates = patchEmbedder(patches, positions: positions)
        hiddenStates = encoder(hiddenStates, positions: positions, mask: nil)
        var pooled = pooler(hiddenStates, columns: columns, rows: rows)
        // Standardize in float32 (std_bias cancels the large values), then return to the working dtype.
        if let standardizationBias, let standardizationScale {
            pooled = (pooled - standardizationBias.asType(.float32)) * standardizationScale.asType(.float32)
        }
        return pooled.asType(hiddenStates.dtype)[0]
    }
}

private final class ArsenalGemma4MultimodalEmbedder: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm: ArsenalGemma4RMSNormNoScale

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = ArsenalGemma4RMSNormNoScale(eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(embeddingPreProjectionNorm(x))
    }
}

// MARK: - Model

public final class ArsenalGemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: ArsenalGemma4VisionModel
    /// Module-internal — also reached by the pinned `Gemma4Assistant.swift` (not ported) (drafter `bind()`
    /// walks here to cache the target's input embeddings, embed scale, and
    /// per-layer type metadata).
    @ModuleInfo(key: "language_model") var languageModel: ArsenalGemma4TextLanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: ArsenalGemma4MultimodalEmbedder

    public let config: ArsenalGemma4Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: ArsenalGemma4Configuration) {
        self.config = config
        self._visionTower.wrappedValue = ArsenalGemma4VisionModel(config: config.visionConfiguration)
        self._languageModel.wrappedValue = ArsenalGemma4TextLanguageModel(config.textConfiguration)
        self._embedVision.wrappedValue = ArsenalGemma4MultimodalEmbedder(
            embeddingDim: config.visionConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            eps: config.visionConfiguration.rmsNormEps
        )
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        image: LMInput.ProcessedImage? = nil
    ) throws -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds =
            (inputsEmbeds
            * MLXArray(pow(Float(config.textConfiguration.hiddenSize), 0.5), dtype: .float32))
            .asType(inputsEmbeds.dtype)

        var perLayerInputs: MLXArray?
        if config.textConfiguration.hiddenSizePerLayerInput > 0 {
            let imageMask = inputIds .== config.imageTokenId
            let audioMask =
                if let audioTokenId = config.audioTokenId {
                    inputIds .== audioTokenId
                } else {
                    MLXArray.zeros(like: imageMask)
                }
            let textMask = logicalNot(logicalOr(imageMask, audioMask))
            let perLayerTokens = MLX.where(textMask, inputIds, MLXArray.zeros(like: inputIds))
            perLayerInputs = languageModel.model.getPerLayerInputs(perLayerTokens)
        }

        guard let image else {
            return (inputsEmbeds, perLayerInputs)
        }
        // GemmaVisionProcessor: patches [B, maxN, 3·P·P], position ids [B, maxN, 2] (-1 padding), one grid per image.
        let kernel = config.visionConfiguration.poolingKernelSize
        guard let positionIds = image.positionIds, let frames = image.frames, frames.count == image.pixels.dim(0),
            frames.allSatisfy({ $0.h % kernel == 0 && $0.w % kernel == 0 && $0.h * $0.w <= image.pixels.dim(1) })
        else {
            throw VLMError.processing("Gemma 4 image patches need a patch grid that is a multiple of the pooling kernel.")
        }
        var features: [MLXArray] = []
        for (index, frame) in frames.enumerated() {
            let count = frame.h * frame.w
            features.append(
                visionTower(
                    image.pixels[index..<index + 1, ..<count, 0...], positions: positionIds[index..<index + 1, ..<count, 0...],
                    columns: frame.w, rows: frame.h))
        }
        let imageFeatures = embedVision(concatenated(features, axis: 0)).asType(inputsEmbeds.dtype)

        let imageMask = inputIds .== config.imageTokenId
        let expectedImageTokens = imageMask.asType(.int32).sum().item(Int.self)

        if expectedImageTokens != imageFeatures.dim(0) {
            throw ArsenalGemma4Error.imageTokenCountMismatch(
                expectedVisionTokens: imageFeatures.dim(0), actualPromptTokens: expectedImageTokens)
        }

        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = broadcast(imageMaskExpanded, to: inputsEmbeds.shape)
        inputsEmbeds = arsenalGemma4MaskedScatter(
            inputTensor: inputsEmbeds,
            mask: imageMaskExpanded,
            source: imageFeatures
        )

        return (inputsEmbeds, perLayerInputs)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let convertedCache = cache.map { $0 }
        if let image = input.image {
            let (inputsEmbeds, perLayerInputs) = try getInputEmbeddings(
                inputIds: input.text.tokens, image: image)
            let result = languageModel(
                nil,
                cache: convertedCache,
                inputsEmbeds: inputsEmbeds,
                perLayerInputs: perLayerInputs,
                tokenTypeIds: arsenalGemma4TokenTypeIds(
                    inputIds: input.text.tokens,
                    imageTokenId: config.imageTokenId,
                    videoTokenId: nil,
                    audioTokenId: config.audioTokenId)
            )
            return .logits(result)
        } else {
            return arsenalGemma4PrepareTextOnly(
                input, cache: convertedCache, windowSize: windowSize, languageModel: languageModel)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let logits = languageModel(inputs, cache: cache?.map { $0 })
        return logits.logits
    }

    /// MTP-aware `LanguageModel` entry point. Reads `mtpEmitFlagKey` from
    /// the incoming `state` and threads it through to `ArsenalGemma4TextLanguageModel`;
    /// the returned `LMOutput` carries `mtpLastHiddenStatesKey` and
    /// `mtpSharedKVStatesKey` populated when the flag is set, empty otherwise.
    /// Overrides the protocol-extension default at `LanguageModel` which
    /// would discard `state`.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let emit = state?[mtpEmitFlagKey] ?? false
        return languageModel(
            input.tokens, cache: cache?.map { $0 },
            emitDrafterState: emit
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = languageModel.sanitize(weights: weights)

        // This port currently supports text + vision only.
        sanitized = sanitized.filter { key, _ in
            !key.contains("audio_tower") && !key.contains("embed_audio")
        }

        if !config.visionConfiguration.useClippedLinears {
            sanitized = sanitized.filter { key, _ in
                !key.contains("input_min")
                    && !key.contains("input_max")
                    && !key.contains("output_min")
                    && !key.contains("output_max")
            }
        }

        return sanitized
    }
}
