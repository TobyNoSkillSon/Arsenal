// SPDX-License-Identifier: MIT
// Klein's conditioner: Qwen3-4B run over the chat-templated prompt padded to 512 tokens; hidden states after layers
// 9, 18 and 27 are concatenated into [512, 7680]. Layers past 27, the final norm and the head never run.
// Adapted from xocialize/flux2-klein-swift (MIT) `TextEncoder.swift` / `Adapted/Qwen3Encoder.swift`, which copy-adapted
// mlx-swift-lm's Qwen3 (MIT); rebuilt on MLXFast RoPE/SDPA so this package needs no mlx-swift-lm.
import Foundation
import Hub
import MLX
import MLXFast
import MLXNN
import Tokenizers

public struct Qwen3Configuration: Codable, Sendable {
    public var hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int, attentionHeads: Int, kvHeads: Int, headDim: Int
    public var rmsNormEps: Float, vocabularySize: Int, ropeTheta: Float
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", hiddenLayers = "num_hidden_layers", intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads", kvHeads = "num_key_value_heads", headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps", vocabularySize = "vocab_size", ropeTheta = "rope_theta"
    }
}

// Qwen's explicit RMS normalization keeps each arithmetic boundary in the activation dtype.
final class QwenHalfRMSNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Eager half squaring overflows for valid Qwen activations above 256. The stock
        // fused kernel accumulates its reduction safely without constructing fp32 arrays.
        if x.dtype == .float16 { return super.callAsFunction(x) }
        let normalized = x * rsqrt(mean(square(x), axis: -1, keepDims: true) + eps)
        return weight * normalized
    }
}

final class Qwen3Attention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, theta: Float
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: QwenHalfRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: QwenHalfRMSNorm
    init(_ c: Qwen3Configuration) {
        heads = c.attentionHeads; kvHeads = c.kvHeads; headDim = c.headDim; theta = c.ropeTheta
        _wq.wrappedValue = Linear(c.hiddenSize, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(c.hiddenSize, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = QwenHalfRMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = QwenHalfRMSNorm(dimensions: headDim, eps: c.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(wq(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(wk(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let scale = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float16($0) / Float16(headDim) }).asType(.bfloat16)
        let inv = 1 / MLX.pow(MLXArray(Float16(1)).asType(.bfloat16) * Double(theta), scale)
        let angles = MLXArray(0..<l).asType(.bfloat16)[.newAxis, 0..., .newAxis] * inv[.newAxis, .newAxis, 0...]
        let angles2 = concatenated([angles, angles], axis: -1)
        let c = MLX.cos(angles2).asType(x.dtype)[0..., .newAxis, 0..., 0...], s = MLX.sin(angles2).asType(x.dtype)[0..., .newAxis, 0..., 0...]
        func rotate(_ a: MLXArray) -> MLXArray {
            let parts = split(a, parts: 2, axis: -1)
            return a * c + concatenated([-parts[1], parts[0]], axis: -1) * s
        }
        q = rotate(q); k = rotate(k)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: .array(mask))
        return wo(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class Qwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    init(_ c: Qwen3Configuration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class Qwen3Block: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3Attention
    @ModuleInfo var mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: QwenHalfRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: QwenHalfRMSNorm
    init(_ c: Qwen3Configuration) {
        _attention.wrappedValue = Qwen3Attention(c); _mlp.wrappedValue = Qwen3MLP(c)
        _inputNorm.wrappedValue = QwenHalfRMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = QwenHalfRMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask)
        return h + mlp(postNorm(h))
    }
}

public final class Qwen3TapEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embed: Embedding
    @ModuleInfo var layers: [Qwen3Block]
    public static let taps = [9, 18, 27]
    public init(_ c: Qwen3Configuration) {
        _embed.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0..<Self.taps.max()!).map { _ in Qwen3Block(c) }
    }
    /// ids [1, L] → [1, L, 3·hidden]; HF hidden_states[k] is the stream after k layers.
    public func callAsFunction(_ ids: MLXArray, mask: MLXArray, activations: DType, trace: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        var h = embed(ids).asType(activations)
        let mask = mask.asType(h.dtype)
        var caps: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask); trace?("layer-\(i)", h)
            if Self.taps.contains(i + 1) { caps.append(h) }
        }
        return concatenated(caps, axis: -1)
    }
}

public final class KleinTextEncoder {
    public let encoder: Qwen3TapEncoder
    public let tokenizer: any Tokenizer
    public static let maxLength = 512
    public static let padToken = 151643 // <|endoftext|>, the tokenizer's pad_token
    public init(encoder: Qwen3TapEncoder, tokenizer: any Tokenizer) { self.encoder = encoder; self.tokenizer = tokenizer }

    /// Loads the tokenizer from the snapshot's `tokenizer/` (local files only).
    public static func tokenizer(directory: URL) throws -> any Tokenizer {
        let decoder = JSONDecoder()
        let config = try decoder.decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")))
        let data = try decoder.decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
        return try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
    }

    /// The vendor's `apply_chat_template(add_generation_prompt=True, enable_thinking=False)` for one user message.
    public static func chatPrompt(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// Right-padded ids (truncated to 512, as the vendor's tokenizer call does) and the count of real tokens.
    public func tokenize(_ prompt: String) -> (ids: [Int], valid: Int) {
        var ids = tokenizer.encode(text: Self.chatPrompt(prompt), addSpecialTokens: false)
        if ids.count > Self.maxLength { ids = Array(ids.prefix(Self.maxLength)) }
        let valid = ids.count
        return (ids + Array(repeating: Self.padToken, count: Self.maxLength - valid), valid)
    }

    /// Additive [1, 1, L, L] mask: causal, and padding keys masked (HF's attention_mask). No row is fully masked.
    public static func mask(length: Int, valid: Int) -> MLXArray {
        var host = [Float16](repeating: 0, count: length * length)
        for i in 0..<length { for j in 0..<length where j > i || j >= valid { host[i * length + j] = -Float16.infinity } }
        return MLXArray(host, [1, 1, length, length])
    }

    /// prompt → [512, 7680].
    public func encode(_ prompt: String, activations: DType, length: Int = maxLength) -> MLXArray {
        let (allIDs, allValid) = tokenize(prompt)
        let ids = Array(allIDs.prefix(length)), valid = Swift.min(allValid, length)
        return encoder(MLXArray(ids.map(Int32.init), [1, ids.count]), mask: Self.mask(length: ids.count, valid: valid), activations: activations)[0]
    }
}
