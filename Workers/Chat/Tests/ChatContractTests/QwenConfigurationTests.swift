// SPDX-License-Identifier: MIT
import Testing
import ChatContract

struct QwenConfigurationTests {
    func vendor() -> [String: Any] {
        [
            "model_type": "qwen3_5",
            "text_config": [
                "num_hidden_layers": 64, "hidden_size": 5120,
                "vocab_size": 248320, "full_attention_interval": 4, "max_position_embeddings": 262144, "dtype": "bfloat16"
            ]
        ]
    }
    @Test func nestedVendorDtype() throws {
        let config = try QwenConfiguration(vendor())
        #expect(config.bf16Source); #expect(config.maximumContext == 262144)
    }
    @Test func quantizedIsNotSource() throws {
        var config = vendor(); config["quantization"] = ["bits": 4, "group_size": 64]
        #expect(!(try QwenConfiguration(config)).bf16Source)
    }
    @Test func rejectsUnapprovedArchitecture() throws {
        #expect(throws: (any Error).self) { try QwenConfiguration(["model_type": "other"]) }
        var config = vendor(); var text = try #require(config["text_config"] as? [String: Any]); text["num_hidden_layers"] = 32; config["text_config"] = text
        #expect(throws: (any Error).self) { try QwenConfiguration(config) }
    }
}
