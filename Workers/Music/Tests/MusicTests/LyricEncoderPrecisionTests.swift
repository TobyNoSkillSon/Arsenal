// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
@testable import MusicWorker

/// ACE-Step's lyric encoder was trained in BF16. With the trained weights its last layer's residual stream reaches
/// about 48,000 on "[Instrumental]" and passes FP16's 65,504 on four or more lyric lines, so FP16 execution turned the
/// encoder output into NaN and every song with real lyrics failed with "the diffusion produced NaN or infinite latents"
/// (31 Oct 2026). Constructed weights reproduce that residual size without the model.
struct LyricEncoderPrecisionTests {
    @Test(.heavy) func residualsBeyondHalfRangeStayFiniteAndReturnHalf() throws {
        let metallib = try #require(
            ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"],
            "the GPU check needs the pinned no-fast-math metallib")
        setTestMetallib(URL(fileURLWithPath: metallib))
        let c = try JSONDecoder().decode(
            ACEStepConfiguration.self,
            from: JSONSerialization.data(withJSONObject: [
                "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                "text_hidden_dim": 128, "num_lyric_encoder_hidden_layers": 1, "num_attention_pooler_hidden_layers": 1, "num_timbre_encoder_hidden_layers": 1,
                "timbre_fix_frame": 16, "is_turbo": true
            ]))
        let encoder = LyricEncoder(c)
        // Attention contributes nothing; the embedding puts 40,000 on every channel and the MLP adds about 40,000 more:
        // RMS-normed input 1, gate and up 128 × 0.078125 = 10, down 256 × silu(10)·10 × 1.5625 ≈ 39,998.
        var values: [String: MLXArray] = [:]
        for (key, value) in encoder.parameters().flattened() {
            let fill: Float =
                switch key {
                case "embed_tokens.bias": 40_000
                case let k where k.hasSuffix("gate_proj.weight") || k.hasSuffix("up_proj.weight"): 0.078125
                case let k where k.hasSuffix("down_proj.weight"): 1.5625
                case let k where k.hasSuffix("norm.weight"): 1
                default: 0
                }
            values[key] = MLXArray.full(value.shape, values: MLXArray(fill)).asType(.float16)
        }
        try encoder.update(parameters: ModuleParameters.unflattened(values), verify: .all)
        let half = MLXArray.full([1, 4, c.hiddenSize], values: MLXArray(Float(40_000))).asType(.float16)
        #expect(MLX.isInf(half + half * Float(0.99995)).all().item(Bool.self), "the constructed sum must overflow FP16")

        let embeddings = MLXArray.zeros([1, 4, c.textHiddenDim], dtype: .float16)
        let out = encoder(embeddings, keep: MLXArray.ones([1, 4], dtype: .int32))
        #expect(out.dtype == .float16)
        #expect(MLX.isNaN(out).any().item(Bool.self) == false)
        #expect(MLX.isInf(out).any().item(Bool.self) == false)
        // Every channel is equal, so the final RMS norm gives 1.
        #expect(abs(MLX.abs(out.asType(.float32)).max().item(Float.self) - 1) < 1e-2)
    }
}
