// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm2_vl
//
// PORTED-FROM: ml-explore/mlx-swift-lm bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57 Libraries/MLXVLM/Models/LFM2VL.swift (MIT,
// adjacent LICENSE-upstream). Local changes, following Transformers modeling_siglip2.py / modeling_lfm2_vl.py and
// mlx-vlm lfm2_vl/vision.py: the vision tower takes the pixel attention mask so padded patches (a thumbnail shorter
// than its tiles) never enter attention; position embeddings are resized with antialiased bilinear interpolation
// (Siglip2 uses F.interpolate bilinear, align_corners=False, antialias=True) instead of bicubic; an image-token /
// feature count mismatch throws instead of fatalError; the half dtype is read from the position table, since int8/int4
// tiers pack patch_embedding. The processor is LFMVisionProcessor; types are renamed so
// the pinned MLXVLM LFM2VL stays unambiguous.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

// MARK: - Vision

private enum Vision {

    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(dims: Int, numHeads: Int, bias: Bool = true) {
            precondition(
                dims % numHeads == 0,
                "The input feature dimensions should be divisible by the number of heads")

            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._kProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._vProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._outProj.wrappedValue = Linear(dims, dims, bias: bias)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            // swiftlint:disable:next identifier_name
            let (B, L, _) = (queries.dim(0), queries.dim(1), queries.dim(2))
            // swiftlint:disable:next identifier_name
            let S = keys.dim(1)

            queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale,
                mask: mask.map { .array($0) } ?? .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return outProj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear

        init(config: ArsenalLFM2VLConfiguration.VisionConfiguration) {
            self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(geluApproximate(fc1(x)))
        }
    }

    fileprivate class EncoderLayer: Module {
        let embedDim: Int
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(config: ArsenalLFM2VLConfiguration.VisionConfiguration) {
            self.embedDim = config.hiddenSize

            self._selfAttn.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numAttentionHeads, bias: true)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: embedDim, eps: config.layerNormEps)
            self.mlp = MLP(config: config)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: embedDim, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            var r = selfAttn(layerNorm1(x), mask: mask)
            var h = x + r
            r = mlp(layerNorm2(h))
            // swiftlint:disable:next shorthand_operator
            h = h + r
            return h
        }
    }

    fileprivate class Encoder: Module {
        var layers: [EncoderLayer]

        init(config: ArsenalLFM2VLConfiguration.VisionConfiguration, visionFeatureLayer: Int = -1) {
            // Determine how many layers to create
            let numLayers: Int

            // visionFeatureLayer == -1 means use all layers
            // Other negative values are Python-style indices from the end (e.g., -2 = second to last)
            if visionFeatureLayer == -1 {
                numLayers = config.numHiddenLayers
            } else {
                // Convert negative indices to positive (e.g., -2 with 27 layers -> 25)
                let actualLayer =
                    visionFeatureLayer < 0
                    ? config.numHiddenLayers + visionFeatureLayer
                    : visionFeatureLayer

                if actualLayer >= 0 && actualLayer < config.numHiddenLayers {
                    numLayers = actualLayer + 1
                } else {
                    numLayers = config.numHiddenLayers
                }
            }
            self.layers = (0..<numLayers).map { _ in EncoderLayer(config: config) }
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false, mask: MLXArray? = nil)
            -> [MLXArray]?
        {
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil

            var h = x
            for layer in layers {
                h = layer(h, mask: mask)
                if outputHiddenStates {
                    encoderStates?.append(h)
                }
            }

            return encoderStates
        }
    }

    fileprivate class VisionEmbeddings: Module {
        let config: ArsenalLFM2VLConfiguration.VisionConfiguration
        let embedDim: Int
        let imageSize: Int
        let patchSize: Int
        let numPatches: Int
        let positionEmbeddingSize: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(config: ArsenalLFM2VLConfiguration.VisionConfiguration) {
            self.config = config
            self.embedDim = config.hiddenSize
            self.imageSize = config.imageSize
            self.patchSize = config.patchSize
            self.numPatches = config.numPatches
            self.positionEmbeddingSize = Int(sqrt(Double(numPatches)))

            self._patchEmbedding.wrappedValue = Linear(
                config.numChannels * patchSize * patchSize,
                embedDim
            )
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPatches, dimensions: embedDim)
        }

        /// Resize positional embeddings like Siglip2VisionEmbeddings.resize_positional_embeddings
        static func resizePositionalEmbeddings(
            positionalEmbeddings: MLXArray,
            spatialShapes: MLXArray,
            maxLength: Int
        ) -> MLXArray {
            let batchSize = spatialShapes.dim(0)
            let embedDim = positionalEmbeddings.dim(-1)
            let sourceDtype = positionalEmbeddings.dtype
            let table = positionalEmbeddings.asType(.float32)

            var rows: [MLXArray] = []
            for i in 0..<batchSize {
                let shape = spatialShapes[i]
                let targetH = shape[0].item(Int.self)
                let targetW = shape[1].item(Int.self)

                let resizedEmbeddings = Siglip2PositionResize.resize(table, height: targetH, width: targetW)
                    .reshaped(targetH * targetW, embedDim).asType(sourceDtype)

                let numPositions = targetH * targetW
                if numPositions < maxLength {
                    // Fill remaining positions with the first embedding
                    let fill = broadcast(resizedEmbeddings[0..<1], to: [maxLength - numPositions, embedDim])
                    rows.append(concatenated([resizedEmbeddings, fill], axis: 0))
                } else {
                    rows.append(resizedEmbeddings[0..<maxLength])
                }
            }

            return stacked(rows, axis: 0)
        }

        func callAsFunction(_ pixelValues: MLXArray, spatialShapes: MLXArray) -> MLXArray {
            // Int8/int4 tiers pack patch_embedding (uint32), so the half dtype comes from the unpacked position table.
            let targetDtype = positionEmbedding.weight.dtype
            let patchEmbeds = patchEmbedding(pixelValues.asType(targetDtype))

            let positionalEmbeddings = positionEmbedding.weight.reshaped(
                positionEmbeddingSize, positionEmbeddingSize, -1
            )

            let resizedPositionalEmbeddings = VisionEmbeddings.resizePositionalEmbeddings(
                positionalEmbeddings: positionalEmbeddings,
                spatialShapes: spatialShapes,
                maxLength: pixelValues.dim(1)
            )

            let embeddings = patchEmbeds + resizedPositionalEmbeddings
            return embeddings
        }
    }

    fileprivate class VisionModel: Module {
        let modelType: String

        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

        init(config: ArsenalLFM2VLConfiguration.VisionConfiguration, visionFeatureLayer: Int = -1) {
            self.modelType = config.modelType

            self.embeddings = VisionEmbeddings(config: config)
            self.encoder = Encoder(config: config, visionFeatureLayer: visionFeatureLayer)
            self._postLayernorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(
            _ x: MLXArray,
            outputHiddenStates: Bool = false,
            spatialShapes: MLXArray,
            pixelAttentionMask: MLXArray? = nil
        ) -> (encoderOutputs: [MLXArray]?, embeddings: MLXArray, lastHiddenState: MLXArray) {
            var embeds = embeddings(x, spatialShapes: spatialShapes)
            embeds = embeds.asType(embeddings.positionEmbedding.weight.dtype)

            // Siglip2 turns pixel_attention_mask into an additive key mask: padded rows never reach a real patch.
            let mask = pixelAttentionMask.map { valid in
                MLX.where(
                    valid[0..., .newAxis, .newAxis, 0...].asType(.bool), MLXArray(Float(0)), MLXArray(-Float.infinity)
                ).asType(embeds.dtype)
            }
            let encoderOutputs = encoder(embeds, outputHiddenStates: outputHiddenStates, mask: mask)
            let lastHiddenState = postLayernorm(encoderOutputs?.last ?? embeds)

            return (encoderOutputs, embeds, lastHiddenState)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_ids") {
                    continue
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Language Model Components (LFM2)

private enum Language {

    fileprivate class LFM2Attention: Module {
        let scale: Float
        let headDim: Int
        let heads: Int
        let kvHeads: Int

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        let rope: RoPE

        init(_ config: ArsenalLFM2VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self._qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            self._kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)

            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            // swiftlint:disable:next identifier_name
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = qLayerNorm(queries.reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
            keys = kLayerNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return outProj(output)
        }
    }

    fileprivate class LFM2ShortConv: Module {
        let lCache: Int
        let hiddenSize: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: ArsenalLFM2VLConfiguration.TextConfiguration, layerIdx: Int) {
            self.lCache = config.convLCache
            self.hiddenSize = config.hiddenSize
            let bias = config.convBias

            self._conv.wrappedValue = Conv1d(
                inputChannels: config.hiddenSize,
                outputChannels: config.hiddenSize,
                kernelSize: lCache,
                groups: config.hiddenSize,
                bias: bias
            )

            self._inProj.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: bias)
            self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: bias)
        }

        func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
            let BCx = inProj(x)
            let BCxSplit = BCx.split(parts: 3, axis: -1)
            // swiftlint:disable:next identifier_name
            let B = BCxSplit[0]
            // swiftlint:disable:next identifier_name
            let C = BCxSplit[1]
            let xPart = BCxSplit[2]
            // swiftlint:disable:next identifier_name
            var Bx = B * xPart

            var state: MLXArray?
            if let cache {
                state = cache[0]
            }
            if state == nil {
                state = MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
            }

            Bx = concatenated([state!, Bx], axis: -2)
            if let cache {
                cache[0] = contiguous(Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...])
                cache.advance(x.dim(1))
            }

            let convOut = conv(Bx)
            let y = C * convOut
            return outProj(y)
        }
    }

    fileprivate class LFM2MLP: Module, UnaryLayer {
        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear
        @ModuleInfo(key: "w3") var w3: Linear

        init(_ config: ArsenalLFM2VLConfiguration.TextConfiguration) {
            var adjustedFFDim = config.blockFFDim

            if config.blockAutoAdjustFFDim {
                adjustedFFDim = Int(Float(2 * adjustedFFDim) / 3.0)
                adjustedFFDim = Int(config.blockFFNDimMultiplier * Float(adjustedFFDim))
                adjustedFFDim =
                    config.blockMultipleOf
                    * ((adjustedFFDim + config.blockMultipleOf - 1) / config.blockMultipleOf)
            }

            self._w1.wrappedValue = Linear(config.blockDim, adjustedFFDim, bias: false)
            self._w2.wrappedValue = Linear(adjustedFFDim, config.blockDim, bias: false)
            self._w3.wrappedValue = Linear(config.blockDim, adjustedFFDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    fileprivate class LFM2DecoderLayer: Module {
        let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var attention: LFM2Attention?
        @ModuleInfo(key: "conv") var conv: LFM2ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: LFM2MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: ArsenalLFM2VLConfiguration.TextConfiguration, layerIdx: Int) {
            self.isAttentionLayer = config.fullAttnIdxs.contains(layerIdx)

            if isAttentionLayer {
                self._attention.wrappedValue = LFM2Attention(config)
            } else {
                self._conv.wrappedValue = LFM2ShortConv(config, layerIdx: layerIdx)
            }

            self._feedForward.wrappedValue = LFM2MLP(config)
            self._operatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let r: MLXArray
            if isAttentionLayer {
                r = attention!(operatorNorm(x), mask: mask, cache: cache)
            } else {
                r = conv!(operatorNorm(x), cache: cache as? MambaCache)
            }
            let h = x + r
            let out = h + feedForward(ffnNorm(h))
            return out
        }
    }

    fileprivate class LFM2ModelInner: Module {
        let config: ArsenalLFM2VLConfiguration.TextConfiguration
        let vocabularySize: Int
        let numHiddenLayers: Int

        let layers: [LFM2DecoderLayer]

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: ArsenalLFM2VLConfiguration.TextConfiguration) {
            self.config = config
            self.vocabularySize = config.vocabularySize
            self.numHiddenLayers = config.hiddenLayers

            precondition(vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize, dimensions: config.hiddenSize)

            self.layers = (0..<numHiddenLayers).map { i in
                LFM2DecoderLayer(config, layerIdx: i)
            }

            self._embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbeddings ?? embedTokens(inputs)

            let mask =
                mask
                ?? {
                    let firstAttnIdx = config.fullAttnIdxs.first ?? 0
                    let c =
                        cache != nil && firstAttnIdx < cache!.count ? cache![firstAttnIdx] : nil
                    return createAttentionMask(h: h, cache: c)
                }()

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return embeddingNorm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: ArsenalLFM2VLConfiguration.TextConfiguration
        let modelType: String
        let model: LFM2ModelInner

        var kvHeads: [Int]

        init(_ config: ArsenalLFM2VLConfiguration.TextConfiguration) {
            self.config = config
            self.modelType = config.modelType

            self.model = LFM2ModelInner(config)

            self.kvHeads = (0..<config.hiddenLayers).map { layerIdx in
                config.fullAttnIdxs.contains(layerIdx) ? config.kvHeads : 0
            }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            mask: MLXArray? = nil,
            cache: [KVCache]? = nil,
            inputsEmbeds: MLXArray? = nil
        ) -> LMOutput {
            var out = model(
                inputs ?? MLXArray([0]), cache: cache, inputEmbeddings: inputsEmbeds)
            out = model.embedTokens.asLinear(out)
            return LMOutput(logits: out)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (name, param) in weights {
                var sanitizedParam = param

                if name.contains("conv.weight") {
                    if param.shape[param.shape.count - 1] > param.dim(1) {
                        sanitizedParam = param.transposed(0, 2, 1)
                    }
                }

                sanitizedWeights[name] = sanitizedParam
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Multi-modal Projector

private class Lfm2VlMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm?
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(config: ArsenalLFM2VLConfiguration) {
        let inChannels =
            config.visionConfiguration.hiddenSize
            * (config.downsampleFactor * config.downsampleFactor)

        if config.projectorUseLayernorm {
            self._layerNorm.wrappedValue = LayerNorm(dimensions: inChannels)
        }

        self._linear1.wrappedValue = Linear(
            inChannels, config.projectorHiddenSize, bias: config.projectorBias)
        self._linear2.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize,
            bias: config.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let layerNorm {
            x = layerNorm(x)
        }
        x = linear1(x)
        x = gelu(x)
        x = linear2(x)
        return x
    }
}

// MARK: - PixelUnshuffleBlock

private class PixelUnshuffleBlock: Module, UnaryLayer {
    let factor: Int

    init(factor: Int) {
        self.factor = factor
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        var (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        // Pad width if necessary
        if w % factor != 0 {
            let padW = factor - (w % factor)
            let padding = MLXArray.zeros([n, padW, h, c], dtype: x.dtype)
            x = concatenated([x, padding], axis: 1)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }

        // Pad height if necessary
        if h % factor != 0 {
            let padH = factor - (h % factor)
            let padding = MLXArray.zeros([n, w, padH, c], dtype: x.dtype)
            x = concatenated([x, padding], axis: 2)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }

        x = x.reshaped(n, w, h / factor, c * factor)
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped(n, h / factor, w / factor, c * factor * factor)
        x = x.transposed(0, 2, 1, 3)

        return x
    }
}

// MARK: - Model

/// LFM2 VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class ArsenalLFM2VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector: Lfm2VlMultiModalProjector
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "pixel_unshuffle") private var pixelUnshuffle: PixelUnshuffleBlock?

    public let config: ArsenalLFM2VLConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers.map { $0 as Module }
    }

    public init(_ config: ArsenalLFM2VLConfiguration) {
        self.config = config

        self._visionModel.wrappedValue = Vision.VisionModel(
            config: config.visionConfiguration,
            visionFeatureLayer: config.visionFeatureLayer
        )

        if config.downsampleFactor > 1 {
            self._pixelUnshuffle.wrappedValue = PixelUnshuffleBlock(factor: config.downsampleFactor)
        }

        self._multiModalProjector.wrappedValue = Lfm2VlMultiModalProjector(config: config)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray?,
        spatialShapes: MLXArray?,
        pixelAttentionMask: MLXArray?
    ) throws -> MLXArray {
        // Ensure inputIds has batch dimension
        var batchedInputIds = inputIds
        if inputIds.ndim == 1 {
            batchedInputIds = inputIds.expandedDimensions(axis: 0)
        }

        var inputsEmbeds = languageModel.model.embedTokens(batchedInputIds)

        // Ensure embeddings have batch dimension
        if inputsEmbeds.ndim == 2 {
            inputsEmbeds = inputsEmbeds.expandedDimensions(axis: 0)
        }

        guard let pixelValues, let spatialShapes, let pixelAttentionMask else {
            return inputsEmbeds
        }

        // Get the output hidden states from the vision model
        // Only a batch with padded rows needs the key mask.
        let padded = pixelAttentionMask.sum().item(Int.self) < pixelAttentionMask.size
        let visionOutput = visionModel(
            pixelValues, outputHiddenStates: true, spatialShapes: spatialShapes,
            pixelAttentionMask: padded ? pixelAttentionMask : nil)
        let hiddenStates = visionOutput.lastHiddenState

        // Get feature lengths from attention mask
        let imgFeatureLengths = sum(pixelAttentionMask, axis: 1)

        var imageFeatures = [MLXArray]()

        for imgIdx in 0..<hiddenStates.dim(0) {
            var feature = hiddenStates[imgIdx]
            let featureLength = imgFeatureLengths[imgIdx].item(Int.self)

            // Slice to valid features
            feature = feature[0..<featureLength].expandedDimensions(axis: 0)

            // Get spatial dimensions
            let featureOrgH = spatialShapes[imgIdx, 0].item(Int.self)
            let featureOrgW = spatialShapes[imgIdx, 1].item(Int.self)

            // Reshape to spatial dimensions
            feature = feature.reshaped(1, featureOrgH, featureOrgW, -1)

            // Apply pixel unshuffle if configured
            if let pixelUnshuffle {
                feature = pixelUnshuffle(feature)
            }

            // Project to language model dimension
            var imgEmbedding = multiModalProjector(feature)

            // Flatten back
            imgEmbedding = imgEmbedding.reshaped(-1, imgEmbedding.dim(-1))
            imageFeatures.append(imgEmbedding)
        }

        let concatenatedImageFeatures = concatenated(imageFeatures, axis: 0)

        // Merge image features with text embeddings
        return try mergeInputIdsWithImageFeatures(
            imageFeatures: concatenatedImageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds,
            imageTokenIndex: config.imageTokenIndex
        )
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int
    ) throws -> MLXArray {
        // Find image token positions
        var imageIndices = [Int]()
        for (i, v) in inputIds.flattened().asArray(Int.self).enumerated() {
            // swiftlint:disable:next for_where
            if v == imageTokenIndex {
                imageIndices.append(i)
            }
        }

        let nImageFeatures = imageFeatures.dim(0)
        if imageIndices.count != nImageFeatures {
            throw VLMError.processing(
                "Image features and image tokens do not match: tokens: \(imageIndices.count), features \(nImageFeatures)"
            )
        }

        // Make sure shapes match before assignment
        var result = inputsEmbeds
        if result.ndim == 2 {
            result = result.expandedDimensions(axis: 0)
        }

        // Assign image features to the image token positions
        if imageFeatures.ndim == 2 {
            let reshapedFeatures = imageFeatures.expandedDimensions(axis: 0)
            result[0..., MLXArray(imageIndices), 0...] = reshapedFeatures
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.embeddings.positionEmbedding.weight.dtype

        // Get image data if available
        let pixelValues = input.image?.pixels.asType(dtype)

        var spatialShapes: MLXArray?
        var pixelAttentionMask: MLXArray?

        if pixelValues != nil, let frames = input.image?.frames, !frames.isEmpty {
            // Extract spatial shapes from frames (THW format where t=1 for images)

            // Convert frames to spatial shapes array [numImages, 2]
            let shapeArrays = frames.map { MLXArray([$0.h, $0.w]) }
            spatialShapes = stacked(shapeArrays, axis: 0)

            // Create attention mask based on actual feature lengths per image
            var maskArrays = [MLXArray]()
            for frame in frames {
                let numPatches = frame.h * frame.w
                let imageMask = MLXArray.ones([numPatches]).asType(.int32)
                maskArrays.append(imageMask)
            }
            // Stack masks - for now assuming single batch processing
            if maskArrays.count == 1 {
                pixelAttentionMask = maskArrays[0].expandedDimensions(axis: 0)
            } else {
                // For multiple images, we need to pad to max length
                let maxLen = maskArrays.map { $0.dim(0) }.max() ?? 0
                let paddedMasks = maskArrays.map { mask -> MLXArray in
                    if mask.dim(0) < maxLen {
                        let padding = MLXArray.zeros([maxLen - mask.dim(0)]).asType(.int32)
                        return concatenated([mask, padding], axis: 0)
                    }
                    return mask
                }
                pixelAttentionMask = stacked(paddedMasks, axis: 0)
            }
        } else if let pixels = pixelValues {
            // Fallback: infer spatial shapes from pixel dimensions (assumes square)
            let numPatches = pixels.dim(1)
            let side = Int(sqrt(Double(numPatches)))
            spatialShapes = MLXArray([side, side]).expandedDimensions(axis: 0)
            pixelAttentionMask = MLXArray.ones([1, numPatches]).asType(.int32)
        }

        let inputEmbeddings = try getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: pixelValues,
            spatialShapes: spatialShapes,
            pixelAttentionMask: pixelAttentionMask
        )

        let result = withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let prefillStepSize = windowSize ?? 512
            let totalPositions = inputEmbeddings.dim(1)
            var processed = 0
            while totalPositions - processed > 1 {
                let chunkLength = min(prefillStepSize, totalPositions - processed - 1)
                let range = processed..<(processed + chunkLength)
                _ = languageModel(
                    nil, cache: cache, inputsEmbeds: inputEmbeddings[0..., range, 0...])
                asyncEval(cache)
                processed += chunkLength
            }
            eval(cache)

            let result = languageModel(
                nil, cache: cache, inputsEmbeds: inputEmbeddings[0..., processed..., 0...])
            return result
        }

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func transformKey(_ key: String) -> String {
            var key = key

            if key.contains("vision_tower") {
                key =
                    key
                    .replacingOccurrences(of: "model.", with: "")
                    .replacingOccurrences(of: "vision_encoder", with: "encoder")
                    .replacingOccurrences(of: "vision_embeddings", with: "embeddings")
                    .replacingOccurrences(of: "vision_post_layernorm", with: "post_layernorm")
            }

            if key.contains("language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            }

            if key.contains("multi_modal_projector") {
                key = key.replacingOccurrences(
                    of: "model.multi_modal_projector", with: "multi_modal_projector")
            }

            return key
        }

        var sanitizedWeights = [String: MLXArray]()
        for (k, v) in weights {
            let newKey = transformKey(k)

            // Handle conv weight transposition
            var value = v
            if newKey.contains("conv.weight") {
                if v.shape[v.shape.count - 1] > v.dim(1) {
                    value = v.transposed(0, 2, 1)
                }
            }

            sanitizedWeights[newKey] = value
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let textConfig = config.textConfiguration
        return (0..<textConfig.hiddenLayers).map { layerIdx in
            if textConfig.fullAttnIdxs.contains(layerIdx) {
                KVCacheSimple()
            } else {
                MambaCache()
            }
        }
    }
}

// MARK: - Configuration

/// Configuration for ``ArsenalLFM2VL``
public struct ArsenalLFM2VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let vocabularySize: Int
        private let _normEps: Float?
        public var normEps: Float { _normEps ?? 1e-5 }
        private let _convBias: Bool?
        public var convBias: Bool { _convBias ?? false }
        private let _convLCache: Int?
        public var convLCache: Int { _convLCache ?? 3 }
        private let _blockDim: Int?
        public var blockDim: Int { _blockDim ?? hiddenSize }
        private let _blockFFDim: Int?
        public var blockFFDim: Int { _blockFFDim ?? hiddenSize }
        private let _blockMultipleOf: Int?
        public var blockMultipleOf: Int { _blockMultipleOf ?? 256 }
        private let _blockFFNDimMultiplier: Float?
        public var blockFFNDimMultiplier: Float { _blockFFNDimMultiplier ?? 1.0 }
        private let _blockAutoAdjustFFDim: Bool?
        public var blockAutoAdjustFFDim: Bool { _blockAutoAdjustFFDim ?? true }
        private let _fullAttnIdxs: [Int]?
        private let layerTypes: [String]?
        public var fullAttnIdxs: [Int] {
            if let fullAttnIdxs = _fullAttnIdxs {
                return fullAttnIdxs
            }

            if let layerTypes {
                return layerTypes.enumerated().compactMap { index, layerType in
                    layerType == "full_attention" ? index : nil
                }
            }

            return Array(0..<hiddenLayers)
        }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1000000.0 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case _normEps = "norm_eps"
            case _convBias = "conv_bias"
            case _convLCache = "conv_L_cache"
            case _blockDim = "block_dim"
            case _blockFFDim = "block_ff_dim"
            case _blockMultipleOf = "block_multiple_of"
            case _blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case _blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
            case _fullAttnIdxs = "full_attn_idxs"
            case layerTypes = "layer_types"
            case _ropeTheta = "rope_theta"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        private let _numChannels: Int?
        public var numChannels: Int { _numChannels ?? 3 }
        private let _imageSize: Int?
        public var imageSize: Int { _imageSize ?? 224 }
        private let _patchSize: Int?
        public var patchSize: Int { _patchSize ?? 16 }
        private let _numPatches: Int?
        public var numPatches: Int { _numPatches ?? 256 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case _numChannels = "num_channels"
            case _imageSize = "image_size"
            case _patchSize = "patch_size"
            case _numPatches = "num_patches"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _downsampleFactor: Int?
    public var downsampleFactor: Int { _downsampleFactor ?? 2 }
    private let _imageTokenId: Int?
    public var imageTokenIndex: Int { _imageTokenId ?? 396 }
    private let _projectorBias: Bool?
    public var projectorBias: Bool { _projectorBias ?? true }
    private let _projectorHiddenSize: Int?
    public var projectorHiddenSize: Int { _projectorHiddenSize ?? 2560 }
    private let _projectorUseLayernorm: Bool?
    public var projectorUseLayernorm: Bool { _projectorUseLayernorm ?? true }
    private let _visionFeatureLayer: Int?
    /// Which vision encoder layer to use for features. -1 means use all layers (default).
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    private let _doImageSplitting: Bool?
    public var doImageSplitting: Bool { _doImageSplitting ?? true }
    private let _maxImageTokens: Int?
    public var maxImageTokens: Int { _maxImageTokens ?? 256 }
    private let _maxNumPatches: Int?
    public var maxNumPatches: Int { _maxNumPatches ?? 1024 }
    private let _minImageTokens: Int?
    public var minImageTokens: Int { _minImageTokens ?? 64 }
    private let _minTiles: Int?
    public var minTiles: Int { _minTiles ?? 2 }
    private let _useThumbnail: Bool?
    public var useThumbnail: Bool { _useThumbnail ?? false }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _downsampleFactor = "downsample_factor"
        case _imageTokenId = "image_token_id"
        case _projectorBias = "projector_bias"
        case _projectorHiddenSize = "projector_hidden_size"
        case _projectorUseLayernorm = "projector_use_layernorm"
        case _visionFeatureLayer = "vision_feature_layer"
        case _doImageSplitting = "do_image_splitting"
        case _maxImageTokens = "max_image_tokens"
        case _maxNumPatches = "max_num_patches"
        case _minImageTokens = "min_image_tokens"
        case _minTiles = "min_tiles"
        case _useThumbnail = "use_thumbnail"
    }
}

/// Siglip2VisionEmbeddings.resize_positional_embeddings: F.interpolate(bilinear, align_corners=False, antialias=True)
/// over the learned square grid, as two dense passes. The weights are torch's `_upsample_bilinear2d_aa` windowed
/// triangle filter, the same computation as Pillow's BILINEAR (`PILResample.coefficients`); upsampling reduces to
/// plain bilinear.
enum Siglip2PositionResize {
    /// `[target, source]` interpolation weights.
    static func matrix(source: Int, target: Int) -> MLXArray {
        let plan = PILResample.coefficients(inSize: source, outSize: target, filter: .bilinear)
        var values = [Float](repeating: 0, count: target * source)
        for row in 0..<target {
            for tap in 0..<plan.count[row] {
                values[row * source + plan.start[row] + tap] = Float(plan.weights[row * plan.kernel + tap])
            }
        }
        return MLXArray(values, [target, source])
    }

    /// `[H, W, D]` → `[height, width, D]` in float32.
    static func resize(_ table: MLXArray, height: Int, width: Int) -> MLXArray {
        let (sourceH, sourceW, dimensions) = (table.dim(0), table.dim(1), table.dim(2))
        // [H', H] · [H, W·D] → [H', W, D]; then [W', W] · [W, D] for each of the H' rows → [H', W', D].
        let vertical = matmul(matrix(source: sourceH, target: height), table.asType(.float32).reshaped(sourceH, sourceW * dimensions))
            .reshaped(height, sourceW, dimensions)
        return matmul(matrix(source: sourceW, target: width), vertical)
    }
}
