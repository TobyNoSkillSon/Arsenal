// SPDX-License-Identifier: MIT
// Heavy (GPU arrays): blocked attention queries give the same output as one block, so long prefixes can be split.
@testable import ElephantEngine
import Foundation
import MLX
import MLXNN
import Testing

@Suite struct AttentionBlockTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    @Test(.heavy) func blockedQueriesMatchOneBlock() throws {
        let json: [String: Any] = [
            "text_config": [
                "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 96, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 32,
                "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "vocab_size": 32, "full_attention_interval": 4,
                "max_position_embeddings": 1024, "rope_parameters": ["rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]]
            ],
            "modeling_overrides": ["attention": ["full_attention": "block_bidirectional", "linear_attention": "causal"], "readout": "slot_lm"]
        ]
        let attention = Attention(try ElephantConfig(json: json))
        /// Fixed pseudo-random values (no RNG state): sin of a scaled index.
        func fixed(_ shape: [Int], _ seed: Float, _ scale: Float = 1) -> MLXArray {
            let n = shape.reduce(1, *)
            return MLXArray((0..<n).map { scale * Foundation.sin(Float($0) * 0.731 + seed) }, shape).asType(.float16)
        }
        var updates: [String: MLXArray] = [:]
        for (i, (name, value)) in attention.parameters().flattened().enumerated() {
            updates[name] = name.hasSuffix("norm.weight") ? MLXArray.ones(value.shape).asType(.float16) : fixed(value.shape, Float(i), 0.1)
        }
        attention.update(parameters: ModuleParameters.unflattened(updates))
        let x = fixed([1, 11, 64], 0.5)
        let past = LayerState.attention(keys: fixed([1, 2, 5, 32], 1.5), values: fixed([1, 2, 5, 32], 2.5))
        attention.queryBlock = 1024
        let (whole, _) = attention(x, previous: past, rope: .offset(5))
        attention.queryBlock = 4 // 11 rows: blocks of 4, 4 and 3
        let (blocked, state) = attention(x, previous: past, rope: .offset(5))
        #expect(whole.shape == [1, 11, 64] && blocked.shape == whole.shape)
        guard case .attention(let k, _) = state else { Issue.record("no attention state"); return }
        #expect(k.dim(2) == 16)
        let diff = abs(whole.asType(.float32) - blocked.asType(.float32)).max().item(Float.self)
        // Different block sizes may take different MLX attention kernels: equal up to FP16 rounding.
        let scale = abs(whole.asType(.float32)).max().item(Float.self)
        #expect(diff <= 2e-3 * Swift.max(scale, 1), "max |diff| \(diff) of \(scale)")
    }
}
