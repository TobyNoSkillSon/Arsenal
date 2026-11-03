// SPDX-License-Identifier: MIT
import Foundation

/// The worker's only approved model identity; dependency registries are not a catalog of allowed models.
public struct QwenConfiguration {
    public let maximumContext: Int
    public let bf16Source: Bool
    public init(_ config: [String: Any]) throws {
        guard config["model_type"] as? String == "qwen3_5",
            let text = config["text_config"] as? [String: Any],
            text["num_hidden_layers"] as? Int == 64,
            text["hidden_size"] as? Int == 5120,
            text["vocab_size"] as? Int == 248320,
            text["full_attention_interval"] as? Int == 4
        else {
            throw ChatFailure("unsupported", "This native chat runner supports only Qwen3.8-27B's approved text architecture.")
        }
        let layers = text["layer_types"] as? [String]
        if let layers {
            guard layers.count == 64,
                layers.enumerated().allSatisfy({ index, type in
                    type == ((index + 1).isMultiple(of: 4) ? "full_attention" : "linear_attention")
                })
            else { throw ChatFailure("unsupported", "Layer types do not match the approved hybrid-attention configuration.") }
        }
        maximumContext = text["max_position_embeddings"] as? Int ?? 262144
        let dtype = text["dtype"] as? String ?? text["torch_dtype"] as? String ?? config["torch_dtype"] as? String
        bf16Source = config["quantization"] == nil && dtype == "bfloat16"
    }
}
