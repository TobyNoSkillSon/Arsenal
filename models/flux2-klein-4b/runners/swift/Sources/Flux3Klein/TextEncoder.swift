// SPDX-License-Identifier: MIT
// FLUX 3's conditioner: the Qwen3-VL-4B-Instruct language tower over the chat-templated prompt, right-padded to the next
// multiple of 80 tokens (at most 4096). Hidden states after layers 4, 8, …, 32 are concatenated into [L, 8 · 2560 = 20480].
// Layers past 32, the final norm, the head and the vision tower never load. Text-only positions are equal on all three
// MRoPE axes, so the interleaved MRoPE reduces to plain 1-D RoPE (mflux passes mrope_section=None).
// mflux reference: models/flux3/model/flux3_text_encoder/prompt_encoder.py, models/flux2/.../qwen3_text_encoder.py,
// models/common_models/qwen3_vl/*. Differences, all within BF16 rounding: RMSNorm uses the fused kernel (float32
// accumulation inside) and attention runs on BF16 inputs (mflux casts q/k/v to float32 first).
import Foundation
import Hub
import MLX
import MLXFast
import MLXNN
import Tokenizers

public enum Flux3Prompt {
    /// Hidden-state indices (HF `hidden_states[k]` is the stream after k layers).
    public static let taps = [4, 8, 12, 16, 20, 24, 28, 32]
    public static let maxLength = 4096
    public static let padMultiple = 80
    public static let padToken = 151643 // <|endoftext|>, the tokenizer's pad_token

    /// Qwen3-VL-Instruct's `apply_chat_template([{role: user, content: prompt}], add_generation_prompt=True)`.
    public static func chatPrompt(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    /// The padded length for `count` real tokens: the next multiple of 80, capped at 4096 (mflux `tokenize`).
    public static func paddedLength(_ count: Int) -> Int {
        Swift.min((Swift.max(count, 0) + padMultiple - 1) / padMultiple * padMultiple, maxLength)
    }

    /// Right-padded ids and the real-token count; real tokens are truncated at 4096 as the tokenizer call does.
    public static func pad(_ ids: [Int]) -> (ids: [Int], valid: Int) {
        let real = Array(ids.prefix(maxLength))
        let length = Swift.max(paddedLength(real.count), real.count)
        return (real + Array(repeating: padToken, count: length - real.count), real.count)
    }

    /// cumsum(mask) − 1, with padding positions set to 1 (mflux `positions`).
    public static func positions(length: Int, valid: Int) -> [Int32] {
        (0..<length).map { $0 < valid ? Int32($0) : 1 }
    }

    /// Text RoPE ids (0, 0, 0, position) for the DiT.
    public static func textIDs(_ length: Int) -> [[Float]] { (0..<length).map { [0, 0, 0, Float($0)] } }

    /// Loads the tokenizer from the snapshot's `text_encoder/` (local files only).
    public static func tokenizer(directory: URL) throws -> any Tokenizer {
        let decoder = JSONDecoder()
        let config = try decoder.decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")))
        let data = try decoder.decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
        return try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
    }

    public static func tokenize(_ prompt: String, tokenizer: any Tokenizer) -> (ids: [Int], valid: Int) {
        pad(tokenizer.encode(text: chatPrompt(prompt), addSpecialTokens: false))
    }
}

final class Flux3QwenAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    init(_ c: Flux3TextConfig) {
        heads = c.heads; kvHeads = c.kvHeads; headDim = c.headDim
        _wq.wrappedValue = Linear(c.hidden, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(c.hidden, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(c.hidden, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, c.hidden, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: c.eps)
    }
    /// x [B, L, hidden]; mask additive [1, 1, L, L]; cos/sin [1, 1, L, headDim] in the activation dtype (rotate-half).
    func callAsFunction(_ x: MLXArray, mask: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        func rotate(_ a: MLXArray) -> MLXArray {
            let half = split(a, parts: 2, axis: -1)
            return a * cos + concatenated([-half[1], half[0]], axis: -1) * sin
        }
        let q = rotate(qNorm(wq(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3))
        let k = rotate(kNorm(wk(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3))
        let v = wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: .array(mask))
        return wo(out.transposed(0, 2, 1, 3).reshaped(b, l, heads * headDim))
    }
}

final class Flux3QwenMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: Flux3TextConfig) {
        _gate.wrappedValue = Linear(c.hidden, c.intermediate, bias: false)
        _up.wrappedValue = Linear(c.hidden, c.intermediate, bias: false)
        _down.wrappedValue = Linear(c.intermediate, c.hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

public final class Flux3QwenLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Flux3QwenAttention
    @ModuleInfo var mlp: Flux3QwenMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ c: Flux3TextConfig) {
        _attention.wrappedValue = Flux3QwenAttention(c); _mlp.wrappedValue = Flux3QwenMLP(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.eps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.eps)
    }
    public func callAsFunction(_ x: MLXArray, mask: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), mask: mask, cos: cos, sin: sin)
        return h + mlp(postNorm(h))
    }
}

/// The language tower up to the deepest tap. Parameter names follow the checkpoint after `model.language_model.`.
public final class Flux3TextEncoder: Module {
    public let config: Flux3TextConfig
    @ModuleInfo(key: "embed_tokens") var embed: Embedding
    @ModuleInfo public var layers: [Flux3QwenLayer]

    public init(_ c: Flux3TextConfig) {
        config = c
        _embed.wrappedValue = Embedding(embeddingCount: c.vocabulary, dimensions: c.hidden)
        _layers.wrappedValue = (0..<Flux3Prompt.taps.max()!).map { _ in Flux3QwenLayer(c) }
    }

    /// Additive [1, 1, L, L] mask: causal, and padding keys masked (mflux `Qwen3TextEncoder`). No row is fully masked.
    public static func mask(length: Int, valid: Int, dtype: DType) -> MLXArray {
        let i = MLXArray(0..<length).reshaped(length, 1), j = MLXArray(0..<length).reshaped(1, length)
        let blocked = (j .> i) .|| (j .>= MLXArray(valid))
        return MLX.where(blocked, MLXArray(-Float.infinity).asType(dtype), MLXArray(Float(0)).asType(dtype)).reshaped(1, 1, length, length)
    }

    /// cos/sin [1, 1, L, headDim]: float32 angles (as mflux's Qwen3TextRotaryEmbedding), cast to `dtype`.
    public func rope(positions: [Int32], dtype: DType) -> (MLXArray, MLXArray) {
        let d = config.headDim
        let exponent = MLXArray(stride(from: 0, to: d, by: 2).map { Float($0) }) / Float(d)
        let inv = 1 / MLX.pow(MLXArray(Float(config.ropeTheta)), exponent)
        let freqs = MLXArray(positions).asType(.float32).reshaped(-1, 1) * inv.reshaped(1, -1)
        let emb = concatenated([freqs, freqs], axis: -1)
        let l = positions.count
        return (MLX.cos(emb).asType(dtype).reshaped(1, 1, l, d), MLX.sin(emb).asType(dtype).reshaped(1, 1, l, d))
    }

    /// ids [1, L] (real tokens first) → prompt features [1, L, 8 · hidden] in the weights' dtype.
    public func callAsFunction(_ ids: MLXArray, valid: Int) -> MLXArray {
        let length = ids.dim(1)
        var h = embed(ids)
        let mask = Self.mask(length: length, valid: valid, dtype: h.dtype)
        let (cos, sin) = rope(positions: Flux3Prompt.positions(length: length, valid: valid), dtype: h.dtype)
        var taps: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cos: cos, sin: sin)
            if Flux3Prompt.taps.contains(i + 1) { taps.append(h) }
        }
        return concatenated(taps, axis: -1)
    }
}
