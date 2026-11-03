// SPDX-License-Identifier: MIT
// Fish Audio Drama 3 (`model_type = fish_drama3`): configuration. Reference: mlx-audio
// `mlx_audio/tts/models/fish_drama3/{config,backbone}.py` at dad2087f (MIT).
import Foundation

public struct Drama3Error: LocalizedError, Equatable, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// The Qwen3.5-MoE text backbone's shape (`text_config`).
public struct Drama3TextConfig: Equatable, Sendable {
    public var hiddenSize: Int
    public var layers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var linearKeyHeads: Int
    public var linearValueHeads: Int
    public var linearKeyHeadDim: Int
    public var linearValueHeadDim: Int
    public var convKernel: Int
    public var experts: Int
    public var expertsPerToken: Int
    public var moeIntermediate: Int
    public var sharedExpertIntermediate: Int
    public var normTopK: Bool
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var partialRotary: Float
    public var vocabSize: Int
    public var fullAttentionInterval: Int = 4

    public init(
        hiddenSize: Int, layers: Int, attentionHeads: Int, kvHeads: Int, headDim: Int, linearKeyHeads: Int, linearValueHeads: Int,
        linearKeyHeadDim: Int, linearValueHeadDim: Int, convKernel: Int, experts: Int, expertsPerToken: Int, moeIntermediate: Int,
        sharedExpertIntermediate: Int, normTopK: Bool = true, rmsNormEps: Float = 1e-6, ropeTheta: Float = 1e7, partialRotary: Float = 0.25,
        vocabSize: Int
    ) {
        self.hiddenSize = hiddenSize; self.layers = layers; self.attentionHeads = attentionHeads; self.kvHeads = kvHeads; self.headDim = headDim
        self.linearKeyHeads = linearKeyHeads; self.linearValueHeads = linearValueHeads; self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim; self.convKernel = convKernel; self.experts = experts; self.expertsPerToken = expertsPerToken
        self.moeIntermediate = moeIntermediate; self.sharedExpertIntermediate = sharedExpertIntermediate; self.normTopK = normTopK
        self.rmsNormEps = rmsNormEps; self.ropeTheta = ropeTheta; self.partialRotary = partialRotary; self.vocabSize = vocabSize
    }

    /// The settings Qwen3.5 hardcodes (backbone.py FIXED_TEXT_SETTINGS); a checkpoint that differs would load and compute something else.
    static let fixed: [String: Any] = [
        "attn_output_gate": true, "hidden_act": "silu", "layernorm_zero_centered_gamma": true, "rotary_interleaved": false,
        "shared_expert_gate": true, "use_qk_norm": true
    ]

    init(json t: [String: Any]) throws {
        func int(_ key: String) throws -> Int {
            guard let v = t[key] as? Int else { throw Drama3Error("text_config.\(key) is missing") }
            return v
        }
        let layers = try int("num_hidden_layers")
        let full = (0..<layers).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        guard (t["layer_types"] as? [String]) == full else { throw Drama3Error("expected three linear-attention layers per full one") }
        if let mlp = t["mlp_layer_types"] as? [String], mlp != Array(repeating: "moe", count: layers) {
            throw Drama3Error("expected a mixture-of-experts MLP in every layer")
        }
        for (key, value) in Self.fixed {
            let ok: Bool
            if let b = value as? Bool { ok = (t[key] as? Bool) == b } else { ok = (t[key] as? String) == (value as? String) }
            guard ok else { throw Drama3Error("unsupported text_config \(key)=\(String(describing: t[key]))") }
        }
        let rope = t["rope_parameters"] as? [String: Any] ?? [:]
        if let type = rope["rope_type"] as? String ?? rope["type"] as? String, type != "default" { throw Drama3Error("unsupported rope type \(type)") }
        let hidden = try int("hidden_size"), heads = try int("num_attention_heads")
        self.init(
            hiddenSize: hidden, layers: layers, attentionHeads: heads, kvHeads: try int("num_key_value_heads"),
            headDim: t["head_dim"] as? Int ?? hidden / heads, linearKeyHeads: try int("linear_num_key_heads"),
            linearValueHeads: try int("linear_num_value_heads"), linearKeyHeadDim: try int("linear_key_head_dim"),
            linearValueHeadDim: try int("linear_value_head_dim"), convKernel: try int("linear_conv_kernel_dim"), experts: try int("num_experts"),
            expertsPerToken: try int("num_experts_per_tok"), moeIntermediate: try int("moe_intermediate_size"),
            sharedExpertIntermediate: try int("shared_expert_intermediate_size"), normTopK: t["norm_topk_prob"] as? Bool ?? true,
            rmsNormEps: Float(t["rms_norm_eps"] as? Double ?? 1e-6), ropeTheta: Float(rope["rope_theta"] as? Double ?? 1e7),
            partialRotary: Float(rope["partial_rotary_factor"] as? Double ?? 0.25), vocabSize: try int("vocab_size"))
        guard linearValueHeads % linearKeyHeads == 0 else { throw Drama3Error("num_v_heads must be divisible by num_k_heads") }
    }
}

/// The audio tower that samples codebooks 1…9 of every frame (`audio_decoder_config`).
public struct Drama3TowerConfig: Equatable, Sendable {
    public var dim: Int
    public var heads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var intermediate: Int
    public var vocabSize: Int
    public var codebooks: Int
    public var sharedLayers: Int
    public var backboneCodebooks: Int
    public var normEps: Float = 1e-6
    public var ropeBase: Float = 1e7
    public var rotaryPercent: Float = 0.25

    public init(
        dim: Int, heads: Int, kvHeads: Int, headDim: Int, intermediate: Int, vocabSize: Int, codebooks: Int, sharedLayers: Int,
        backboneCodebooks: Int, normEps: Float = 1e-6, ropeBase: Float = 1e7, rotaryPercent: Float = 0.25
    ) {
        self.dim = dim; self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim; self.intermediate = intermediate
        self.vocabSize = vocabSize; self.codebooks = codebooks; self.sharedLayers = sharedLayers; self.backboneCodebooks = backboneCodebooks
        self.normEps = normEps; self.ropeBase = ropeBase; self.rotaryPercent = rotaryPercent
    }

    init(json a: [String: Any]) throws {
        func int(_ key: String) throws -> Int {
            guard let v = a[key] as? Int else { throw Drama3Error("audio_decoder_config.\(key) is missing") }
            return v
        }
        self.init(
            dim: try int("dim"), heads: try int("n_head"), kvHeads: try int("n_local_heads"), headDim: try int("head_dim"),
            intermediate: try int("intermediate_size"), vocabSize: try int("vocab_size"), codebooks: try int("num_codebooks"),
            sharedLayers: try int("shared_enrichment_layers"), backboneCodebooks: try int("vq_backbone_codebooks"),
            normEps: Float(a["norm_eps"] as? Double ?? 1e-6), ropeBase: Float(a["rope_base"] as? Double ?? 1e7),
            rotaryPercent: Float(a["rotary_percent"] as? Double ?? 0.25))
        // The reference tower rotates with traditional (interleaved) RoPE and gates attention; anything else computes something else.
        if let interleaved = a["rotary_interleaved"] as? Bool, !interleaved { throw Drama3Error("unsupported tower rotary_interleaved=false") }
        if let gate = a["attention_output_gate"] as? Bool, !gate { throw Drama3Error("unsupported tower attention_output_gate=false") }
        if let moe = a["use_moe"] as? Bool, moe { throw Drama3Error("unsupported tower use_moe=true") }
    }
}

/// Per-module quantization declared by the checkpoint (`config.quantization`, MLX affine).
public struct Drama3Quantization: Equatable, Sendable {
    public var groupSize: Int
    public var bits: Int
    public var modules: [String: Spec]
    public struct Spec: Equatable, Sendable {
        public var groupSize: Int
        public var bits: Int
    }
    public func spec(_ path: String) -> Spec { modules[path] ?? Spec(groupSize: groupSize, bits: bits) }

    init?(json q: [String: Any]?) {
        guard let q else { return nil }
        groupSize = q["group_size"] as? Int ?? 64
        bits = q["bits"] as? Int ?? 4
        var modules: [String: Spec] = [:]
        for (key, value) in q {
            guard let entry = value as? [String: Any] else { continue }
            modules[key] = Spec(groupSize: entry["group_size"] as? Int ?? groupSize, bits: entry["bits"] as? Int ?? bits)
        }
        self.modules = modules
        if let mode = q["mode"] as? String, mode != "affine" { return nil }
    }
}

public struct Drama3Config: Equatable, Sendable {
    public var text: Drama3TextConfig
    public var tower: Drama3TowerConfig
    public var semanticBegin: Int
    public var semanticEnd: Int
    public var eosID: Int
    public var quantization: Drama3Quantization?

    public init(text: Drama3TextConfig, tower: Drama3TowerConfig, semanticBegin: Int, semanticEnd: Int, eosID: Int, quantization: Drama3Quantization? = nil) {
        self.text = text; self.tower = tower; self.semanticBegin = semanticBegin; self.semanticEnd = semanticEnd; self.eosID = eosID
        self.quantization = quantization
    }

    public init(data: Data) throws {
        guard let c = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw Drama3Error("config.json is not an object") }
        guard c["model_type"] as? String == "fish_drama3" else { throw Drama3Error("config.json is not a fish_drama3 checkpoint") }
        let template = c["speech_template_version"] as? String ?? "reference-bracket-v1"
        guard template == "reference-bracket-v1" else { throw Drama3Error("unsupported speech_template_version \(template)") }
        guard let text = c["text_config"] as? [String: Any], let tower = c["audio_decoder_config"] as? [String: Any],
            let begin = c["semantic_start_token_id"] as? Int, let end = c["semantic_end_token_id"] as? Int, let eos = c["eos_token_id"] as? Int
        else { throw Drama3Error("config.json lacks the Drama 3 fields") }
        if let mode = (c["quantization"] as? [String: Any])?["mode"] as? String, mode != "affine" {
            throw Drama3Error("unsupported quantization mode \(mode)")
        }
        self.init(
            text: try Drama3TextConfig(json: text), tower: try Drama3TowerConfig(json: tower), semanticBegin: begin, semanticEnd: end, eosID: eos,
            quantization: Drama3Quantization(json: c["quantization"] as? [String: Any] ?? c["quantization_config"] as? [String: Any]))
        guard end - begin + 1 == self.tower.vocabSize else { throw Drama3Error("semantic band does not match the tower vocabulary") }
    }
}
