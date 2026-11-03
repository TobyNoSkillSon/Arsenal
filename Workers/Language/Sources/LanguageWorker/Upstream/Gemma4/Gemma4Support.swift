// SPDX-License-Identifier: MIT
// PORTED-FROM: ml-explore/mlx-swift-lm bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57 Libraries/MLXVLM/Models/Gemma4.swift (MIT,
// adjacent LICENSE-upstream), split into Gemma4Support/Gemma4Text/Gemma4Vision; types renamed ArsenalGemma4*.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

enum ArsenalGemma4Error: LocalizedError {
    case imageTokenCountMismatch(expectedVisionTokens: Int, actualPromptTokens: Int)
    case multimodalTokenCountMismatch(kind: String, featureTokens: Int, promptTokens: Int)

    var errorDescription: String? {
        switch self {
        case .imageTokenCountMismatch(let expectedVisionTokens, let actualPromptTokens):
            return
                "Gemma 4 image token count mismatch: vision encoder produced \(expectedVisionTokens) soft tokens, but the prompt contains \(actualPromptTokens) image tokens."
        case .multimodalTokenCountMismatch(let kind, let featureTokens, let promptTokens):
            return
                "Gemma 4 \(kind) token count mismatch: encoder produced \(featureTokens) soft tokens, but the prompt contains \(promptTokens) \(kind) tokens."
        }
    }
}

func arsenalGemma4BuildLayerTypes(hiddenLayers: Int, slidingWindowPattern: Int) -> [String] {
    let pattern =
        Array(repeating: "sliding_attention", count: max(slidingWindowPattern - 1, 0))
        + ["full_attention"]
    guard !pattern.isEmpty else { return Array(repeating: "full_attention", count: hiddenLayers) }
    var result: [String] = []
    result.reserveCapacity(hiddenLayers)
    while result.count < hiddenLayers {
        result.append(contentsOf: pattern)
    }
    return Array(result.prefix(hiddenLayers))
}

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported).
func arsenalGemma4DefaultTextRopeParameters() -> [String: [String: StringOrNumber]] {
    [
        "full_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(1_000_000.0),
            "rope_type": .string("proportional")
        ],
        "sliding_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(10_000.0),
            "rope_type": .string("default")
        ]
    ]
}

func arsenalGemma4DefaultVisionRopeParameters() -> [String: StringOrNumber] {
    [
        "rope_theta": .float(100.0),
        "rope_type": .string("default")
    ]
}

func arsenalGemma4MaskedScatter(
    inputTensor: MLXArray, mask: MLXArray, source: MLXArray
) -> MLXArray {
    let flattenedInput = inputTensor.flattened()
    let flattenedMask = mask.flattened().asArray(Bool.self)
    let flattenedSource = source.flattened()

    let targetIndices = flattenedMask.enumerated().compactMap { idx, value in
        value ? Int32(idx) : nil
    }
    guard !targetIndices.isEmpty else {
        return inputTensor
    }

    guard flattenedSource.dim(0) == targetIndices.count else {
        fatalError(
            "Masked scatter shape mismatch. source=\(flattenedSource.dim(0)) mask=\(targetIndices.count)"
        )
    }

    let result = flattenedInput
    result[MLXArray(targetIndices, [targetIndices.count])] = flattenedSource
    return result.reshaped(inputTensor.shape)
}

func arsenalGemma4RotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape[x.shape.count - 1] / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

func arsenalGemma4ApplyMultiDimensionalRoPE(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float
) -> MLXArray {
    let headDim = inputs.shape[inputs.ndim - 1]
    if positions.ndim == 2 {
        let half = headDim / 2
        let freqExponents =
            (2.0 / Float(headDim)) * MLXArray(0..<half).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let sinusoid = positions.asType(.float32).expandedDimensions(axis: -1) / timescale
        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)
        return inputs * cosValue + arsenalGemma4RotateHalf(inputs) * sinValue
    }

    let numDimensions = positions.shape[positions.ndim - 1]
    let channelsPerDimension = 2 * (headDim / (2 * numDimensions))
    let halfPerDimension = channelsPerDimension / 2

    var parts: [MLXArray] = []
    parts.reserveCapacity(numDimensions)

    for d in 0..<numDimensions {
        let start = d * channelsPerDimension
        let end = start + channelsPerDimension
        let part = inputs[.ellipsis, start..<end]

        let freqExponents =
            (2.0 / Float(channelsPerDimension)) * MLXArray(0..<halfPerDimension).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let dimPositions = positions[.ellipsis, d..<d + 1].asType(.float32)
        let sinusoid = dimPositions / timescale

        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)

        parts.append(part * cosValue + arsenalGemma4RotateHalf(part) * sinValue)
    }

    return concatenated(parts, axis: -1)
}

func arsenalGemma4EnsureFusedSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let fusedDims = [64, 80, 128]
    let d = queries.dim(queries.ndim - 1)
    let target = fusedDims.first(where: { d <= $0 }) ?? d

    if target == d {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    let paddedQueries = MLX.padded(
        queries, widths: [0, 0, 0, .init((0, target - d))])
    let paddedKeys = MLX.padded(
        keys, widths: [0, 0, 0, .init((0, target - d))])
    let paddedValues = MLX.padded(
        values, widths: [0, 0, 0, .init((0, target - d))])

    return MLXFast.scaledDotProductAttention(
        queries: paddedQueries, keys: paddedKeys, values: paddedValues, scale: scale, mask: mask
    )[.ellipsis, ..<d]
}

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported).
enum ArsenalGemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            return keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            return keys.0.dim(-2)
        }
    }
}

/// Module-internal — also consumed by the pinned `Gemma4Assistant.swift` (not ported).
func arsenalGemma4AdjustAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    keyLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mask {
    case .array(let maskArray):
        let maskLength = maskArray.dim(-1)
        guard maskLength > keyLength else {
            return mask
        }
        let start = maskLength - keyLength
        return .array(maskArray[.ellipsis, start...])
    case .arrays, .causal, .none:
        return mask
    }
}

func arsenalGemma4TokenTypeIds(
    inputIds: MLXArray,
    imageTokenId: Int?,
    videoTokenId: Int?,
    audioTokenId: Int?
) -> MLXArray {
    var tokenTypeIds = MLXArray.zeros(like: inputIds).asType(.int32)
    if let imageTokenId {
        tokenTypeIds = MLX.where(inputIds .== imageTokenId, MLXArray(1), tokenTypeIds)
    }
    if let videoTokenId {
        tokenTypeIds = MLX.where(inputIds .== videoTokenId, MLXArray(2), tokenTypeIds)
    }
    if let audioTokenId {
        tokenTypeIds = MLX.where(inputIds .== audioTokenId, MLXArray(3), tokenTypeIds)
    }
    return tokenTypeIds
}

func arsenalGemma4TextOnlyPromptTokens(_ input: LMInput) -> MLXArray {
    let tokens = input.text.tokens
    if tokens.ndim == 2, tokens.dim(0) == 1 {
        return tokens[0]
    }
    if tokens.ndim == 1 {
        return tokens
    }
    return tokens.flattened()
}

func arsenalGemma4PrepareTextOnly(
    _ input: LMInput,
    cache: [any KVCache],
    windowSize: Int?,
    languageModel: ArsenalGemma4TextLanguageModel
) -> PrepareResult {
    let prefillStepSize = max(windowSize ?? 512, 1)
    let y = arsenalGemma4TextOnlyPromptTokens(input).expandedDimensions(axis: 0)
    let convertedCache = cache.map { $0 }
    let totalPositions = y.dim(1)

    var processed = 0
    while totalPositions - processed > 1 {
        let chunkLength = min(prefillStepSize, totalPositions - processed - 1)
        _ = languageModel(
            y[0..., processed..<(processed + chunkLength)],
            cache: convertedCache
        )
        asyncEval(cache)
        processed += chunkLength
    }

    eval(cache)
    return .logits(languageModel(y[0..., processed...], cache: convertedCache))
}

func arsenalGemma4BlockSequenceIdsForMask(_ tokenTypeIds: MLXArray) -> MLXArray {
    let isVision = (tokenTypeIds .== 1) | (tokenTypeIds .== 2)
    let sequenceLength = isVision.dim(1)
    guard sequenceLength > 0 else {
        return MLXArray.zeros(like: tokenTypeIds).asType(.int32) - 1
    }

    let previous =
        if sequenceLength == 1 {
            MLXArray.zeros(like: isVision)
        } else {
            concatenated(
                [
                    MLXArray.zeros(like: isVision[0..., ..<1]),
                    isVision[0..., ..<(sequenceLength - 1)]
                ],
                axis: 1
            )
        }
    let starts = isVision & logicalNot(previous)
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray.zeros(like: groupIds) - 1)
}

func arsenalGemma4ApplyBlockwiseBidirectionalOverlay(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    tokenTypeIds: MLXArray,
    sequenceLength: Int,
    windowSize: Int?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let baseMask: MLXArray
    switch mask {
    case .array(let array):
        baseMask = array
    case .arrays(let arrays):
        guard let first = arrays.first else {
            return mask
        }
        baseMask = first
    case .causal:
        baseMask = createCausalMask(n: sequenceLength, offset: 0, windowSize: windowSize)
    case .none:
        baseMask = MLXArray.ones([sequenceLength, sequenceLength], dtype: .bool)
    }

    guard tokenTypeIds.dim(1) == baseMask.dim(-1) else {
        return mask
    }

    let blockSequenceIds = arsenalGemma4BlockSequenceIdsForMask(tokenTypeIds)
    let queryBlocks = expandedDimensions(blockSequenceIds, axis: -1)
    let keyBlocks = expandedDimensions(blockSequenceIds, axis: -2)
    let sameBlock = (queryBlocks .!= -1) & (queryBlocks .== keyBlocks)
    return .array(baseMask | expandedDimensions(sameBlock, axis: 1))
}

func arsenalGemma4CompactPrefixRows(features: MLXArray, validMask: MLXArray) -> MLXArray {
    let maskRows = validMask.asArray(Bool.self)
    let batch = validMask.dim(0)
    let length = validMask.dim(1)
    var rows: [MLXArray] = []
    rows.reserveCapacity(batch)

    for batchIdx in 0..<batch {
        let start = batchIdx * length
        let count = maskRows[start..<start + length].reduce(0) { $0 + ($1 ? 1 : 0) }
        if count > 0 {
            rows.append(features[batchIdx, ..<count, 0...])
        }
    }

    guard !rows.isEmpty else {
        return features.reshaped(-1, features.dim(-1))[..<0, 0...]
    }
    return concatenated(rows, axis: 0)
}
// MARK: - Configuration

public struct ArsenalGemma4TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let globalKVHeads: Int?
    public let headDim: Int
    public let globalHeadDim: Int
    public let vocabularySize: Int
    public let vocabularySizePerLayerInput: Int
    public let numKVSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTraditional: Bool
    public let finalLogitSoftcapping: Float?
    public let useDoubleWideMLP: Bool
    public let enableMoEBlock: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let attentionKEqV: Bool
    public let useBidirectionalAttention: String?
    public let layerTypes: [String]
    public let ropeParameters: [String: [String: StringOrNumber]]
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case vocabularySizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionKEqV = "attention_k_eq_v"
        case useBidirectionalAttention = "use_bidirectional_attention"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_text"
        let isUnified = modelType == "gemma4_unified_text" || modelType == "gemma4_unified"
        hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize)
            ?? (isUnified ? 3840 : 1536)
        hiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers)
            ?? (isUnified ? 48 : 35)
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize)
            ?? (isUnified ? 15_360 : 6144)
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads)
            ?? (isUnified ? 16 : 8)
        kvHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.kvHeads) ?? (isUnified ? 8 : 1)
        globalKVHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalKVHeads)
            ?? (isUnified ? 1 : nil)
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalHeadDim) ?? 512
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySize) ?? 262_144
        vocabularySizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySizePerLayerInput)
            ?? vocabularySize
        numKVSharedLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numKVSharedLayers)
            ?? (isUnified ? 0 : 20)
        hiddenSizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSizePerLayerInput)
            ?? (isUnified ? 0 : 256)
        slidingWindow =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindow)
            ?? (isUnified ? 1024 : 512)
        slidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindowPattern)
            ?? (isUnified ? 6 : 5)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.ropeTraditional) ?? false
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: CodingKeys.finalLogitSoftcapping) ?? 30.0
        useDoubleWideMLP =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useDoubleWideMLP) ?? true
        enableMoEBlock =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.enableMoEBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.topKExperts)
        moeIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: CodingKeys.moeIntermediateSize)
        attentionKEqV =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.attentionKEqV) ?? isUnified
        useBidirectionalAttention =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.useBidirectionalAttention)
            ?? (isUnified ? "vision" : nil)
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: CodingKeys.ropeParameters)
            ?? (isUnified
                ? [
                    "full_attention": [
                        "partial_rotary_factor": .float(0.25),
                        "rope_theta": .float(1_000_000.0),
                        "rope_type": .string("proportional")
                    ],
                    "sliding_attention": [
                        "rope_theta": .float(10_000.0),
                        "rope_type": .string("default")
                    ]
                ] : arsenalGemma4DefaultTextRopeParameters())
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: CodingKeys.layerTypes)
            ?? arsenalGemma4BuildLayerTypes(
                hiddenLayers: hiddenLayers, slidingWindowPattern: slidingWindowPattern)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
    }
}

public struct ArsenalGemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let rmsNormEps: Float
    public let defaultOutputLength: Int
    public let positionEmbeddingSize: Int
    public let poolingKernelSize: Int
    public let useClippedLinears: Bool
    public let standardize: Bool
    public let ropeParameters: [String: StringOrNumber]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_vision"
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers) ?? 16
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize) ?? 768
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize) ?? 3072
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads) ?? 12
        keyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.keyValueHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 64
        patchSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize) ?? 16
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        defaultOutputLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.defaultOutputLength) ?? 280
        positionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.positionEmbeddingSize) ?? 10_240
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize) ?? 3
        useClippedLinears =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useClippedLinears) ?? false
        standardize = try c.decodeIfPresent(Bool.self, forKey: CodingKeys.standardize) ?? false
        ropeParameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: CodingKeys.ropeParameters)
            ?? arsenalGemma4DefaultVisionRopeParameters()
    }
}

public struct ArsenalGemma4Configuration: Codable, Sendable {
    public let textConfiguration: ArsenalGemma4TextConfiguration
    public let visionConfiguration: ArsenalGemma4VisionConfiguration
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let tieWordEmbeddings: Bool

    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var hiddenSize: Int { _hiddenSize ?? textConfiguration.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case quantization
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case tieWordEmbeddings = "tie_word_embeddings"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(
            ArsenalGemma4TextConfiguration.self, forKey: CodingKeys.textConfiguration)
        visionConfiguration = try c.decode(
            ArsenalGemma4VisionConfiguration.self, forKey: CodingKeys.visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4"
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: CodingKeys.quantization)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId)
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId)
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerImage)
            ?? visionConfiguration.defaultOutputLength
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings)
            ?? textConfiguration.tieWordEmbeddings
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._vocabularySize)
        _hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._hiddenSize)
        _padTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys._padTokenId)
    }
}
