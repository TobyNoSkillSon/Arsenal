import Foundation
import MLX
import MLXNN
public struct GraniteQuantizationConfiguration: Decodable, Sendable, Equatable {
    /// Default number of source values sharing a scale and bias.
    public var groupSize: Int
    /// Packed weight bit width.
    public var bits: Int
    /// MLX quantization mode.
    public var mode: QuantizationMode
    /// Optional per-module overrides. Keys use the flattened MLX module path,
    /// for example `encoder.input_linear`.
    public var groupSizes: [String: Int] = [:]

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size", bits, mode
        case groupSizes = "group_sizes"
    }

    /// Creates an affine quantization configuration.
    /// - Parameters:
    ///   - groupSize: Default number of source values sharing scale and bias.
    ///   - bits: Packed weight width supported by the converted checkpoint.
    ///   - mode: MLX quantization representation, normally affine.
    ///   - groupSizes: Per-module group-size overrides keyed by flattened path.
    public init(
        groupSize: Int, bits: Int, mode: QuantizationMode,
        groupSizes: [String: Int] = [:]
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.groupSizes = groupSizes
    }

    /// Decodes quantization settings from a converted checkpoint configuration.
    /// - Parameter decoder: Decoder positioned at the quantization object.
    /// - Throws: `DecodingError` when required values are missing or malformed.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        bits = try container.decode(Int.self, forKey: .bits)
        mode = try container.decode(QuantizationMode.self, forKey: .mode)
        groupSizes = try container.decodeIfPresent([String: Int].self, forKey: .groupSizes) ?? [:]
    }

    /// Returns the per-module group size override or the default group size.
    /// - Parameter modulePath: Flattened MLX module path, for example
    ///   `encoder.input_linear`.
    /// - Returns: Matching override, or ``groupSize`` when no override exists.
    public func groupSize(for modulePath: String) -> Int {
        groupSizes[modulePath] ?? groupSize
    }
}

/// Runtime architecture configuration for Granite Speech 5.0 TurboCTC.
public struct GraniteModelConfiguration: Decodable, Sendable {
    /// Checkpoint architecture identifier.
    public var modelType: String = "granite_speech5_ctc"
    /// CTC tokenizer vocabulary size.
    public var vocabSize: Int = 16_384
    /// Encoder hidden width.
    public var hiddenSize: Int = 1_024
    /// Number of Conformer layers.
    public var numHiddenLayers: Int = 16
    /// Required input audio sample rate.
    public var sampleRate: Int = 16_000
    /// CTC output frames per second.
    public var outputFrameRate: Double = 12.5
    /// Feed-forward hidden width.
    public var intermediateSize: Int = 4_096
    /// Number of attention heads.
    public var numAttentionHeads: Int = 8
    /// Width of each attention head.
    public var headDimension: Int = 128
    /// Local attention context size.
    public var contextSize: Int = 128
    /// Convolution module expansion multiplier.
    public var convExpansionFactor: Int = 2
    /// Depthwise convolution kernel width.
    public var convKernelSize: Int = 7
    /// Encoder layers that perform temporal subsampling.
    public var subsampleLayers: [Int] = [0, 1]
    /// Relative-position embedding capacity.
    public var maxPositionEmbeddings: Int = 512
    /// Weight quantization settings, or `nil` for floating-point weights.
    public var quantization: GraniteQuantizationConfiguration?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type", vocabSize = "vocab_size", encoderConfig = "encoder_config"
        case quantization
    }

    private enum EncoderKeys: String, CodingKey {
        case hiddenSize = "hidden_size", numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size", numAttentionHeads = "num_attention_heads"
        case headDimension = "head_dim", contextSize = "context_size"
        case convExpansionFactor = "conv_expansion_factor", convKernelSize = "conv_kernel_size"
        case subsampleLayers = "subsample_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    /// Creates the default Granite Speech 5.0 configuration.
    public init() {}

    /// Decodes architecture settings from a converted checkpoint configuration.
    /// - Parameter decoder: Decoder positioned at the model configuration object.
    /// - Throws: `DecodingError` when a present configuration value has an
    ///   incompatible type or nested structure.
    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try root.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        vocabSize = try root.decodeIfPresent(Int.self, forKey: .vocabSize) ?? vocabSize
        quantization = try root.decodeIfPresent(
            GraniteQuantizationConfiguration.self, forKey: .quantization)
        if root.contains(.encoderConfig) {
            let encoder = try root.nestedContainer(keyedBy: EncoderKeys.self, forKey: .encoderConfig)
            hiddenSize = try encoder.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            numHiddenLayers = try encoder.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? numHiddenLayers
            intermediateSize = try encoder.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            numAttentionHeads = try encoder.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? numAttentionHeads
            headDimension = try encoder.decodeIfPresent(Int.self, forKey: .headDimension) ?? headDimension
            contextSize = try encoder.decodeIfPresent(Int.self, forKey: .contextSize) ?? contextSize
            convExpansionFactor = try encoder.decodeIfPresent(Int.self, forKey: .convExpansionFactor) ?? convExpansionFactor
            convKernelSize = try encoder.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? convKernelSize
            subsampleLayers = try encoder.decodeIfPresent([Int].self, forKey: .subsampleLayers) ?? subsampleLayers
            maxPositionEmbeddings = try encoder.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? maxPositionEmbeddings
        }
    }
}

/// A decoded word with approximate CTC timing.
public struct GraniteWord: Codable, Sendable, Equatable {
    /// Decoded word text.
    public let text: String
    /// Word onset in seconds.
    public let start: Double
    /// Word end in seconds.
    public let end: Double

    /// Creates a timed word.
    /// - Parameters:
    ///   - text: Display spelling for the decoded word.
    ///   - start: Approximate word onset in seconds from the audio start.
    ///   - end: Approximate exclusive word end in seconds from the audio start.
    public init(text: String, start: Double, end: Double) {
        self.text = text
        self.start = start
        self.end = end
    }
}
