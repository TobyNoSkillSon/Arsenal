// SPDX-License-Identifier: MIT
// LCO-Embedding Omni (Qwen2_5OmniThinkerForConditionalGeneration): configuration and the Qwen2.5 text model with
// multimodal RoPE. Structure follows mlx-swift-lm's Qwen25VL.swift (MIT, port of mlx-vlm's qwen2_5_vl); numerics follow
// Transformers' modeling_qwen2_5_omni.py, which casts the rotary cos/sin to the activation dtype before applying them.
import Foundation
import IntegrationKit
import MLX
import MLXFast
import MLXNN

struct LCOConfiguration: Decodable {
    struct Text: Decodable {
        let hiddenSize, intermediateSize, layers, heads, kvHeads, vocabularySize, maxPositions: Int
        let rmsNormEps, ropeTheta: Float
        let mropeSection: [Int]
        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size", intermediateSize = "intermediate_size", layers = "num_hidden_layers"
            case heads = "num_attention_heads", kvHeads = "num_key_value_heads", vocabularySize = "vocab_size"
            case maxPositions = "max_position_embeddings", rmsNormEps = "rms_norm_eps", ropeTheta = "rope_theta", ropeScaling = "rope_scaling"
        }
        enum ScalingKeys: String, CodingKey { case mropeSection = "mrope_section" }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize); intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            layers = try c.decode(Int.self, forKey: .layers); heads = try c.decode(Int.self, forKey: .heads)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads); vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
            maxPositions = try c.decodeIfPresent(Int.self, forKey: .maxPositions) ?? 32768
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            let scaling = try c.nestedContainer(keyedBy: ScalingKeys.self, forKey: .ropeScaling)
            mropeSection = try scaling.decode([Int].self, forKey: .mropeSection)
        }
        init(hiddenSize: Int, intermediateSize: Int, layers: Int, heads: Int, kvHeads: Int, vocabularySize: Int, mropeSection: [Int]) {
            self.hiddenSize = hiddenSize; self.intermediateSize = intermediateSize; self.layers = layers; self.heads = heads
            self.kvHeads = kvHeads; self.vocabularySize = vocabularySize; maxPositions = 32768; rmsNormEps = 1e-6; ropeTheta = 1_000_000
            self.mropeSection = mropeSection
        }
    }
    struct Vision: Decodable {
        let depth, hiddenSize, intermediateSize, heads, outHiddenSize, patchSize, spatialMergeSize, temporalPatchSize, windowSize, inChannels: Int
        let fullAttentionBlocks: [Int]
        let tokensPerSecond: Int
        enum CodingKeys: String, CodingKey {
            case depth, hiddenSize = "hidden_size", intermediateSize = "intermediate_size", heads = "num_heads", outHiddenSize = "out_hidden_size"
            case patchSize = "patch_size", spatialMergeSize = "spatial_merge_size", temporalPatchSize = "temporal_patch_size"
            case windowSize = "window_size", inChannels = "in_channels", fullAttentionBlocks = "fullatt_block_indexes", tokensPerSecond = "tokens_per_second"
        }
    }
    struct Audio: Decodable {
        let dModel, heads, ffn, layers, melBins, maxSourcePositions, nWindow, outputDim: Int
        enum CodingKeys: String, CodingKey {
            case dModel = "d_model", heads = "encoder_attention_heads", ffn = "encoder_ffn_dim", layers = "encoder_layers"
            case melBins = "num_mel_bins", maxSourcePositions = "max_source_positions", nWindow = "n_window", outputDim = "output_dim"
        }
    }
    let text: Text
    let vision: Vision
    let audio: Audio
    let imageToken, videoToken, audioToken, visionStart, audioStart: Int
    let positionIdPerSeconds: Int
    enum CodingKeys: String, CodingKey {
        case text = "text_config", vision = "vision_config", audio = "audio_config"
        case imageToken = "image_token_index", videoToken = "video_token_index", audioToken = "audio_token_index"
        case visionStart = "vision_start_token_id", audioStart = "audio_start_token_id", positionIdPerSeconds = "position_id_per_seconds"
    }
}

final class LCOTextAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    let heads, kvHeads, headDim: Int
    init(_ c: LCOConfiguration.Text) {
        heads = c.heads; kvHeads = c.kvHeads; headDim = c.hiddenSize / c.heads
        _q.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: true)
        _k.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: true)
        _v.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: true)
        _o.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var qs = q(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        var ks = k(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let vs = v(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        qs = qs * cos + LCORope.rotateHalf(qs) * sin
        ks = ks * cos + LCORope.rotateHalf(ks) * sin
        let out = MLXFast.scaledDotProductAttention(queries: qs, keys: ks, values: vs, scale: pow(Float(headDim), -0.5), mask: l > 1 ? .causal : .none)
        return o(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class LCOTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ width: Int, _ hidden: Int) {
        _gate.wrappedValue = Linear(width, hidden, bias: false); _up.wrappedValue = Linear(width, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, width, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class LCOTextLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LCOTextAttention
    @ModuleInfo var mlp: LCOTextMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ c: LCOConfiguration.Text) {
        _attention.wrappedValue = LCOTextAttention(c); _mlp.wrappedValue = LCOTextMLP(c.hiddenSize, c.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), cos: cos, sin: sin)
        return h + mlp(postNorm(h))
    }
}

enum LCORope {
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }
    /// Qwen2_5OmniRotaryEmbedding: positions [3, L] → cos, sin [1, 1, L, headDim] in `dtype`. The half-dimension
    /// frequencies take section i from axis i % 3 (temporal, height, width), then repeat for the rotated half.
    static func cosSin(positions: [[Int]], headDim: Int, theta: Float, sections: [Int], dtype: DType) -> (MLXArray, MLXArray) {
        let length = positions[0].count
        let inv = (0..<(headDim / 2)).map { 1 / pow(Double(theta), Double(2 * $0) / Double(headDim)) }
        var axis = [Int](), start = 0
        for (i, s) in sections.enumerated() { axis += Array(repeating: i % 3, count: s); start += s }
        precondition(start == headDim / 2, "mrope sections must cover half the head dimension")
        var angles = [Float](repeating: 0, count: length * headDim)
        for t in 0..<length {
            for j in 0..<(headDim / 2) {
                // FP32 angle, as the maker computes `position.float() * inv_freq` before the cast.
                let a = Float(positions[axis[j]][t]) * Float(inv[j])
                angles[t * headDim + j] = a; angles[t * headDim + headDim / 2 + j] = a
            }
        }
        let array = MLXArray(angles, [1, 1, length, headDim])
        return (MLX.cos(array).asType(dtype), MLX.sin(array).asType(dtype))
    }
}

final class LCOTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [LCOTextLayer]
    @ModuleInfo var norm: RMSNorm
    let config: LCOConfiguration.Text
    /// Lab FP16 range check: when set, the largest |hidden| over every layer output of the last forward.
    var traceMaximum = false
    private(set) var observedMaximum: Float = 0
    init(_ c: LCOConfiguration.Text) {
        config = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0..<c.layers).map { _ in LCOTextLayer(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    /// The final normed hidden state of the last token (`hidden_states[-1][:, -1]`, the ST last-token pooling).
    func lastHidden(_ embeds: MLXArray, positions: [[Int]], cancellation: NativeCancellation) throws -> MLXArray {
        let (cos, sin) = LCORope.cosSin(
            positions: positions, headDim: config.hiddenSize / config.heads, theta: config.ropeTheta, sections: config.mropeSection, dtype: embeds.dtype)
        var h = embeds
        observedMaximum = 0
        for layer in layers {
            try cancellation.check()
            h = layer(h, cos: cos, sin: sin)
            eval(h)
            if traceMaximum { observedMaximum = max(observedMaximum, abs(h).max().asType(.float32).item(Float.self)) }
        }
        return norm(h[0..., (h.dim(1) - 1)...])[0, 0]
    }
}
