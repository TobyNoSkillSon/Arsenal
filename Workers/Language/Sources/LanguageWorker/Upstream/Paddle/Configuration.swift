import Foundation

struct PaddleOCRVLVisionConfig: Codable, Sendable {
    var hiddenSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numChannels: Int
    var imageSize: Int
    var patchSize: Int
    var hiddenAct: String
    var layerNormEps: Float
    var attentionDropout: Float
    var intermediateSize: Int
    var projectorHiddenAct: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case hiddenAct = "hidden_act"
        case layerNormEps = "layer_norm_eps"
        case attentionDropout = "attention_dropout"
        case intermediateSize = "intermediate_size"
        case projectorHiddenAct = "projector_hidden_act"
    }

    init(
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numChannels: Int = 3,
        imageSize: Int = 448,
        patchSize: Int = 14,
        hiddenAct: String = "gelu",
        layerNormEps: Float = 1e-6,
        attentionDropout: Float = 0.0,
        intermediateSize: Int = 4096,
        projectorHiddenAct: String = "gelu"
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numChannels = numChannels
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.hiddenAct = hiddenAct
        self.layerNormEps = layerNormEps
        self.attentionDropout = attentionDropout
        self.intermediateSize = intermediateSize
        self.projectorHiddenAct = projectorHiddenAct
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        self.imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 448
        self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        self.layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        self.projectorHiddenAct = try container.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
    }
}

struct PaddleOCRVLTextConfig: Decodable, Sendable {
    enum RopeKeys: String, CodingKey { case sections = "mrope_section" }
    var headDim: Int
    var mropeSection: [Int]
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var hiddenAct: String
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(
        headDim: Int = 128,
        mropeSection: [Int] = [16, 24, 24],
        vocabSize: Int = 103424,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 3072,
        numHiddenLayers: Int = 18,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 2,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 32768,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 500_000,
        tieWordEmbeddings: Bool = false
    ) {
        self.headDim = headDim
        self.mropeSection = mropeSection
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.mropeSection = (try? container.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeScaling).decode([Int].self, forKey: .sections)) ?? [16, 24, 24]
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 103424
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 18
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

struct PaddleOCRVLConfig: Decodable, Sendable {
    var visionConfig: PaddleOCRVLVisionConfig
    var textConfig: PaddleOCRVLTextConfig
    var imageTokenIndex: Int
    var visionStartTokenId: Int
    var visionEndTokenId: Int
    var visionTokenId: Int

    enum CodingKeys: String, CodingKey {
        case visionConfig = "vision_config"
        case textConfig = "text_config"
        case imageTokenIndex = "image_token_index"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case visionTokenId = "vision_token_id"
    }

    init(
        visionConfig: PaddleOCRVLVisionConfig = PaddleOCRVLVisionConfig(),
        textConfig: PaddleOCRVLTextConfig = PaddleOCRVLTextConfig(),
        imageTokenIndex: Int = 100295,
        visionStartTokenId: Int = 101305,
        visionEndTokenId: Int = 101306,
        visionTokenId: Int = 100295
    ) {
        self.visionConfig = visionConfig
        self.textConfig = textConfig
        self.imageTokenIndex = imageTokenIndex
        self.visionStartTokenId = visionStartTokenId
        self.visionEndTokenId = visionEndTokenId
        self.visionTokenId = visionTokenId
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.visionConfig = try container.decodeIfPresent(PaddleOCRVLVisionConfig.self, forKey: .visionConfig) ?? PaddleOCRVLVisionConfig()
        self.textConfig = try container.decodeIfPresent(PaddleOCRVLTextConfig.self, forKey: .textConfig) ?? decoder.singleValueContainer().decode(PaddleOCRVLTextConfig.self)
        self.imageTokenIndex = try container.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 100295
        self.visionStartTokenId = try container.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 101305
        self.visionEndTokenId = try container.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 101306
        self.visionTokenId = try container.decodeIfPresent(Int.self, forKey: .visionTokenId) ?? 100295
    }

    static func load(from directory: URL) throws -> PaddleOCRVLConfig {
        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)

        let raw = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] ?? [:]
        let decoder = JSONDecoder()

        let visionConfig: PaddleOCRVLVisionConfig
        if let visionDict = raw["vision_config"] as? [String: Any] {
            let visionData = try JSONSerialization.data(withJSONObject: visionDict, options: [])
            visionConfig = try decoder.decode(PaddleOCRVLVisionConfig.self, from: visionData)
        } else {
            visionConfig = PaddleOCRVLVisionConfig()
        }

        let textConfig: PaddleOCRVLTextConfig
        if let textDict = raw["text_config"] as? [String: Any] {
            let textData = try JSONSerialization.data(withJSONObject: textDict, options: [])
            textConfig = try decoder.decode(PaddleOCRVLTextConfig.self, from: textData)
        } else {
            textConfig = try decoder.decode(PaddleOCRVLTextConfig.self, from: data)
        }

        let imageTokenIndex = (raw["vision_token_id"] as? Int) ?? (raw["image_token_id"] as? Int) ?? 100295
        let visionStartTokenId = (raw["vision_start_token_id"] as? Int) ?? 101305
        let visionEndTokenId = (raw["vision_end_token_id"] as? Int) ?? 101306
        let visionTokenId = (raw["vision_token_id"] as? Int) ?? (raw["image_token_id"] as? Int) ?? 100295

        return PaddleOCRVLConfig(
            visionConfig: visionConfig,
            textConfig: textConfig,
            imageTokenIndex: imageTokenIndex,
            visionStartTokenId: visionStartTokenId,
            visionEndTokenId: visionEndTokenId,
            visionTokenId: visionTokenId
        )
    }
}

struct PaddleOCRVLTokens {
    static let imageToken = "<image>"
    static let visionStartToken = "<|vision_start|>"
    static let visionEndToken = "<|vision_end|>"
    static let visionPadToken = "<|vision_pad|>"
}

enum PaddleOCRTask: String, CaseIterable, Sendable {
    case ocr
    case table
    case formula
    case chart

    var prompt: String {
        switch self {
        case .ocr: return "OCR:"
        case .table: return "Table Recognition:"
        case .formula: return "Formula Recognition:"
        case .chart: return "Chart Recognition:"
        }
    }
}
