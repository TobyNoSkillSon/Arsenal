// SPDX-License-Identifier: MIT
import Foundation

public struct QwenImageError: Error, LocalizedError, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
    public var description: String { message }
}

/// `transformer/config.json` (QwenImage21Transformer2DModel).
public struct TransformerConfig: Equatable, Sendable {
    public var layers = 32, heads = 32, headDim = 128, inChannels = 64, outChannels = 64, contextDim = 4096, mlpRatio = 3
    public var axes = [16, 56, 56]
    public var eps: Float = 1e-6
    public var causalCondition = true
    public init() {}
    public var hidden: Int { heads * headDim }
    public init(json: [String: Any]) throws {
        guard json["_class_name"] as? String == "QwenImage21Transformer2DModel" else { throw QwenImageError("transformer/config.json is not a Qwen-Image 2.1 transformer") }
        layers = json["num_layers"] as? Int ?? layers
        heads = json["num_attention_heads"] as? Int ?? heads
        headDim = json["attention_head_dim"] as? Int ?? headDim
        inChannels = json["in_channels"] as? Int ?? inChannels
        outChannels = json["out_channels"] as? Int ?? inChannels
        contextDim = json["context_in_dim"] as? Int ?? contextDim
        mlpRatio = json["mlp_ratio"] as? Int ?? mlpRatio
        axes = json["axes_dims_rope"] as? [Int] ?? axes
        eps = (json["eps"] as? NSNumber)?.floatValue ?? eps
        causalCondition = json["causal_condition"] as? Bool ?? true
        guard (json["patch_size"] as? Int ?? 1) == 1, axes.count == 3, axes.reduce(0, +) == headDim, axes.allSatisfy({ $0 % 2 == 0 }) else {
            throw QwenImageError("Unsupported Qwen-Image transformer geometry")
        }
        guard causalCondition else { throw QwenImageError("The prefix cache needs causal_condition = true") }
    }
}

/// `vae/config.json` (AutoencoderKLQwenImage21, the image specialization of Wan's VAE).
public struct VAEConfig: Equatable, Sendable {
    public var baseDim = 96, decoderBaseDim = 144, zDim = 64, resBlocks = 2, inChannels = 4, outChannels = 4
    public var dimMult = [1, 2, 4, 8, 8]
    public var temporalDownsample = [false, true, true, true]
    public var latentsMean: [Float] = Array(repeating: 0, count: 64)
    public var latentsStd: [Float] = Array(repeating: 1, count: 64)
    public init() {}
    public var spatialScale: Int { 1 << (dimMult.count - 1) }
    public init(json: [String: Any]) throws {
        guard json["_class_name"] as? String == "AutoencoderKLQwenImage21" else { throw QwenImageError("vae/config.json is not a Qwen-Image 2.1 VAE") }
        baseDim = json["base_dim"] as? Int ?? baseDim
        decoderBaseDim = json["decoder_base_dim"] as? Int ?? baseDim
        zDim = json["z_dim"] as? Int ?? zDim
        resBlocks = json["num_res_blocks"] as? Int ?? resBlocks
        inChannels = json["in_channels"] as? Int ?? inChannels
        outChannels = json["out_channels"] as? Int ?? outChannels
        dimMult = json["dim_mult"] as? [Int] ?? dimMult
        temporalDownsample = json["temperal_downsample"] as? [Bool] ?? temporalDownsample
        latentsMean = (json["latents_mean"] as? [NSNumber])?.map(\.floatValue) ?? []
        latentsStd = (json["latents_std"] as? [NSNumber])?.map(\.floatValue) ?? []
        guard json["is_residual"] as? Bool ?? true, (json["attn_scales"] as? [Any] ?? []).isEmpty, json["patch_size"] is NSNull || json["patch_size"] == nil,
            latentsMean.count == zDim, latentsStd.count == zDim, temporalDownsample.count == dimMult.count - 1, inChannels == 4, outChannels == 4
        else { throw QwenImageError("Unsupported Qwen-Image VAE configuration") }
    }
}

/// `scheduler/scheduler_config.json` plus the pipeline's saved `sample_sigmas` (Turbo's 8-step schedule).
public struct SchedulerConfig: Equatable, Sendable {
    public var baseImageSeqLen = 256, maxImageSeqLen = 8192, trainTimesteps = 1000
    public var baseShift = 0.5, maxShift = 0.9, shift = 1.0
    public var shiftTerminal: Double? = 0.02
    public var dynamicShifting = true
    public var sampleSigmas: [Double]?
    public init() {}
    public init(json: [String: Any], pipeline: [String: Any]) throws {
        guard json["_class_name"] as? String == "FlowMatchEulerDiscreteScheduler" else { throw QwenImageError("Unsupported scheduler") }
        func number(_ key: String) -> Double? { (json[key] as? NSNumber)?.doubleValue }
        baseImageSeqLen = json["base_image_seq_len"] as? Int ?? baseImageSeqLen
        maxImageSeqLen = json["max_image_seq_len"] as? Int ?? maxImageSeqLen
        trainTimesteps = json["num_train_timesteps"] as? Int ?? trainTimesteps
        baseShift = number("base_shift") ?? baseShift
        maxShift = number("max_shift") ?? maxShift
        shift = number("shift") ?? shift
        shiftTerminal = number("shift_terminal").flatMap { $0 == 0 ? nil : $0 }
        dynamicShifting = json["use_dynamic_shifting"] as? Bool ?? false
        sampleSigmas = (pipeline["sample_sigmas"] as? [NSNumber])?.map(\.doubleValue)
        let flags = ["use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "invert_sigmas", "stochastic_sampling"]
        guard flags.allSatisfy({ json[$0] as? Bool != true }), (json["time_shift_type"] as? String ?? "exponential") == "exponential" else {
            throw QwenImageError("Unsupported scheduler options")
        }
        if let sampleSigmas { guard !sampleSigmas.isEmpty, sampleSigmas.allSatisfy({ $0 > 0 && $0 <= 1 }) else { throw QwenImageError("Invalid sample_sigmas") } }
    }
}

/// `text_encoder/config.json` (Qwen3VLForConditionalGeneration).
public struct Qwen3VLConfig: Equatable, Sendable {
    public struct Text: Equatable, Sendable {
        public var hidden = 4096, layers = 36, heads = 32, kvHeads = 8, headDim = 128, intermediate = 12288, vocab = 151936
        public var eps: Float = 1e-6
        public var ropeTheta: Float = 5_000_000
        public var mropeSection = [24, 20, 20]
    }
    public struct Vision: Equatable, Sendable {
        public var depth = 27, hidden = 1152, heads = 16, intermediate = 4304, patch = 16, temporalPatch = 2, merge = 2, outHidden = 4096
        public var positions = 2304, inChannels = 3
        public var deepstack = [8, 16, 24]
        public var activation = "gelu_pytorch_tanh"
    }
    public var text = Text(), vision = Vision()
    public var imageToken = 151655, visionStart = 151652, visionEnd = 151653
    public init() {}
    public init(json: [String: Any]) throws {
        guard json["model_type"] as? String == "qwen3_vl", let t = json["text_config"] as? [String: Any], let v = json["vision_config"] as? [String: Any] else {
            throw QwenImageError("text_encoder/config.json is not a Qwen3-VL model")
        }
        text.hidden = t["hidden_size"] as? Int ?? text.hidden
        text.layers = t["num_hidden_layers"] as? Int ?? text.layers
        text.heads = t["num_attention_heads"] as? Int ?? text.heads
        text.kvHeads = t["num_key_value_heads"] as? Int ?? text.kvHeads
        text.headDim = t["head_dim"] as? Int ?? text.hidden / text.heads
        text.intermediate = t["intermediate_size"] as? Int ?? text.intermediate
        text.vocab = t["vocab_size"] as? Int ?? text.vocab
        text.eps = (t["rms_norm_eps"] as? NSNumber)?.floatValue ?? text.eps
        let scaling = t["rope_scaling"] as? [String: Any] ?? t["rope_parameters"] as? [String: Any] ?? [:]
        text.ropeTheta = (t["rope_theta"] as? NSNumber ?? scaling["rope_theta"] as? NSNumber)?.floatValue ?? text.ropeTheta
        text.mropeSection = scaling["mrope_section"] as? [Int] ?? text.mropeSection
        guard scaling["mrope_interleaved"] as? Bool ?? true, t["attention_bias"] as? Bool != true, (t["hidden_act"] as? String ?? "silu") == "silu",
            text.mropeSection.count == 3, text.mropeSection.reduce(0, +) == text.headDim / 2
        else { throw QwenImageError("Unsupported Qwen3-VL text configuration") }
        vision.depth = v["depth"] as? Int ?? vision.depth
        vision.hidden = v["hidden_size"] as? Int ?? vision.hidden
        vision.heads = v["num_heads"] as? Int ?? vision.heads
        vision.intermediate = v["intermediate_size"] as? Int ?? vision.intermediate
        vision.patch = v["patch_size"] as? Int ?? vision.patch
        vision.temporalPatch = v["temporal_patch_size"] as? Int ?? vision.temporalPatch
        vision.merge = v["spatial_merge_size"] as? Int ?? vision.merge
        vision.outHidden = v["out_hidden_size"] as? Int ?? vision.outHidden
        vision.positions = v["num_position_embeddings"] as? Int ?? vision.positions
        vision.inChannels = v["in_channels"] as? Int ?? vision.inChannels
        vision.deepstack = v["deepstack_visual_indexes"] as? [Int] ?? vision.deepstack
        vision.activation = v["hidden_act"] as? String ?? vision.activation
        imageToken = json["image_token_id"] as? Int ?? imageToken
        visionStart = json["vision_start_token_id"] as? Int ?? visionStart
        visionEnd = json["vision_end_token_id"] as? Int ?? visionEnd
        guard ["gelu_pytorch_tanh", "gelu_new", "gelu_fast"].contains(vision.activation),
            Int(Double(vision.positions).squareRoot()) * Int(Double(vision.positions).squareRoot()) == vision.positions
        else {
            throw QwenImageError("Unsupported Qwen3-VL vision configuration")
        }
    }
}

public enum ConfigFile {
    public static func json(_ url: URL) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else { throw QwenImageError("Invalid JSON: \(url.lastPathComponent)") }
        return object
    }
}
