// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import ArsenalWire

/// Normalize versioned configuration spellings without changing checkpoint tensors.
enum LanguageGraph {
    static func configuration(_ data: Data) throws -> Data {
        var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        if root["model_type"] as? String == "qwen3_vl", var text = root["text_config"] as? [String: Any] {
            text["tie_word_embeddings"] = text["tie_word_embeddings"] ?? root["tie_word_embeddings"] ?? false
            if let rope = text["rope_parameters"] as? [String: Any] {
                text["rope_scaling"] = rope; text["rope_theta"] = rope["rope_theta"]
            }
            root["text_config"] = text
            if let eos = root["eos_token_id"] as? Int { root["eos_token_id"] = [eos] }
        }
        return try JSONSerialization.data(withJSONObject: root)
    }
    static func make(_ data: Data, type: String) throws -> any LanguageModel {
        let normalized = try configuration(data)
        let decoder = JSONDecoder()
        switch type {
        case "qwen2_5_vl": return TeleOCRVL(try decoder.decode(TeleOCRVLConfiguration.self, from: normalized))
        case "qwen3_5": return Qwen35(try decoder.decode(Qwen35Configuration.self, from: normalized))
        case "qwen3_vl": return Qwen3VL(try decoder.decode(Qwen3VLConfiguration.self, from: normalized))
        case "lfm2_vl", "lfm2-vl": return ArsenalLFM2VL(try decoder.decode(ArsenalLFM2VLConfiguration.self, from: normalized))
        case "gemma4": return ArsenalGemma4(try decoder.decode(ArsenalGemma4Configuration.self, from: normalized))
        default: throw WorkerFailure(code: "unsupported_model", message: "Unsupported language architecture: \(type).")
        }
    }
    static func removeUnusedSharedKV(_ model: Module, config: Data) throws {
        let raw = try JSONSerialization.jsonObject(with: config) as? [String: Any] ?? [:]
        guard raw["model_type"] as? String == "gemma4", let text = raw["text_config"] as? [String: Any], let count = text["num_hidden_layers"] as? Int,
            let shared = text["num_kv_shared_layers"] as? Int, shared > 0
        else { return }
        for (name, module) in model.namedModules() where name.hasPrefix("language_model.model.layers.") && name.hasSuffix(".self_attn") {
            let pieces = name.split(separator: ".")
            if pieces.count == 5, let index = Int(pieces[3]), index >= count - shared {
                try module.update(modules: ModuleChildren(values: ["k_proj": .none, "v_proj": .none, "k_norm": .none]), verify: .none)
            }
        }
    }
}
