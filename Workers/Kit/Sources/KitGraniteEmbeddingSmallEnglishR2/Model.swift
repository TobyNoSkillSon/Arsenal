// SPDX-License-Identifier: MIT
// ibm-granite/granite-embedding-small-english-r2 (Apache-2.0): a ModernBERT encoder that turns English text into 384-number vectors.
// Ported from transformers modeling_modernbert.py (ModernBertModel) at the pinned revision. Pooling is the CLS token (modules.json),
// and the output is unnormalized unless `normalize` is true (the model card's own default).
import Foundation
import MLX
import MLXFast
import MLXNN
import Tokenizers

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "embeddings"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["embed"]

final class Model {
    let net: Encoder
    let tokenizer: Tokenizer
    let config: ModernBertConfig

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, *.safetensors, tokenizer files) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = ModernBertConfig(try Files.json(weightsDir))
        net = Encoder(config)
        try Weights.apply(try Weights.load(weightsDir), to: net)
        tokenizer = try Text.tokenizer(weightsDir)
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let texts = try inputTexts(params)
        let normalize = params.bool("normalize", default: false)
        let truncate = params.bool("truncate", default: true)
        if let dimensions = raw(params, "dimensions"), dimensions != config.hidden {
            throw RunnerError.invalid("dimensions must be \(config.hidden): this model has no smaller output size.")
        }
        var vectors: [[Double]] = []
        var inputTokens = 0
        for text in texts {
            var ids = tokenizer.encode(text: text)
            if ids.count > config.maxPositions {
                guard truncate else {
                    throw RunnerError.invalid("An input has \(ids.count) tokens; the limit is \(config.maxPositions). Set truncate to true to cut it.")
                }
                ids = Array(ids.prefix(config.maxPositions - 1)) + [config.sepToken]
            }
            inputTokens += ids.count
            var cls = net(ids)[0, 0].asType(.float32)
            if normalize { cls = cls / sqrt(sum(cls * cls)) }
            vectors.append(list(cls))
        }
        return ["embeddings": vectors, "input_tokens": inputTokens]
    }

    /// The text inputs: a string, or {"type": "text", "text": ...}. Images, audio and mixed content are not supported.
    private func inputTexts(_ params: Params) throws -> [String] {
        guard let items = params.raw["input"] as? [Any], !items.isEmpty else {
            throw RunnerError.invalid("input must be a nonempty list of text.")
        }
        return try items.enumerated().map { index, item in
            var text: String?
            if let string = item as? String { text = string }
            if let object = item as? [String: Any], object["type"] as? String == "text" { text = object["text"] as? String }
            guard let text else {
                throw RunnerError.invalid("input[\(index)] must be text. This model embeds text only.")
            }
            guard !text.isEmpty else { throw RunnerError.invalid("input[\(index)] is empty.") }
            return text
        }
    }

    private func raw(_ params: Params, _ key: String) -> Int? { (params.raw[key] as? NSNumber)?.intValue }
}

struct ModernBertConfig {
    let hidden: Int
    let heads: Int
    let layers: Int
    let intermediate: Int
    let normEps: Float
    let globalEvery: Int
    let slidingHalfWindow: Int
    let globalTheta: Double
    let localTheta: Double
    let maxPositions: Int
    let sepToken: Int

    init(_ raw: [String: Any]) {
        func int(_ key: String, _ fallback: Int) -> Int { (raw[key] as? NSNumber)?.intValue ?? fallback }
        func double(_ key: String, _ fallback: Double) -> Double { (raw[key] as? NSNumber)?.doubleValue ?? fallback }
        hidden = int("hidden_size", 384)
        heads = int("num_attention_heads", 12)
        layers = int("num_hidden_layers", 12)
        intermediate = int("intermediate_size", 1536)
        normEps = Float(double("norm_eps", 1e-5))
        globalEvery = int("global_attn_every_n_layers", 3)
        // The checkpoint's local_attention is the full window; attention reaches half of it on each side.
        slidingHalfWindow = int("local_attention", 128) / 2
        globalTheta = double("global_rope_theta", 80000)
        localTheta = double("local_rope_theta", 10000)
        maxPositions = int("max_position_embeddings", 8192)
        sepToken = int("sep_token_id", 50282)
    }
}

/// The encoder: token embeddings, 12 ModernBERT layers and the final norm.
final class Encoder: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embeddings
    @ModuleInfo(key: "layers") var layers: [EncoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(_ config: ModernBertConfig) {
        _embeddings.wrappedValue = Embeddings(config)
        _layers.wrappedValue = (0..<config.layers).map { EncoderLayer(config, index: $0) }
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.hidden, eps: config.normEps, bias: false)
        self.config = config
    }

    let config: ModernBertConfig

    /// Last hidden state [1, L, hidden] for one unpadded sequence of token ids.
    func callAsFunction(_ ids: [Int]) -> MLXArray {
        let length = ids.count
        let input = MLXArray(ids.map(Int32.init), [1, length])
        var hidden = embeddings(input)
        let headDim = config.hidden / config.heads
        let globalRope = rotary(length: length, headDim: headDim, theta: config.globalTheta)
        let localRope = rotary(length: length, headDim: headDim, theta: config.localTheta)
        let slidingMask = slidingWindowMask(length: length, halfWindow: config.slidingHalfWindow)
        for layer in layers {
            if layer.isSliding {
                hidden = layer(hidden, cos: localRope.cos, sin: localRope.sin, mask: slidingMask)
            } else {
                hidden = layer(hidden, cos: globalRope.cos, sin: globalRope.sin, mask: nil)
            }
        }
        return finalNorm(hidden)
    }
}

final class Embeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: ModernBertConfig) {
        _tokEmbeddings.wrappedValue = Embedding(embeddingCount: 50368, dimensions: config.hidden)
        _norm.wrappedValue = LayerNorm(dimensions: config.hidden, eps: config.normEps, bias: false)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray { norm(tokEmbeddings(ids)) }
}

final class EncoderLayer: Module {
    /// Layer 0 has no attention norm (the checkpoint has none for it).
    @ModuleInfo(key: "attn_norm") var attnNorm: UnaryLayer
    @ModuleInfo(key: "attn") var attn: Attention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: GeGLU
    let isSliding: Bool

    init(_ config: ModernBertConfig, index: Int) {
        isSliding = index % config.globalEvery != 0
        let norm: UnaryLayer = index == 0
            ? Identity() : LayerNorm(dimensions: config.hidden, eps: config.normEps, bias: false)
        _attnNorm.wrappedValue = norm
        _attn.wrappedValue = Attention(hidden: config.hidden, heads: config.heads)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: config.hidden, eps: config.normEps, bias: false)
        _mlp.wrappedValue = GeGLU(hidden: config.hidden, intermediate: config.intermediate)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        var hidden = x + attn(attnNorm(x), cos: cos, sin: sin, mask: mask)
        hidden = hidden + mlp(mlpNorm(hidden))
        return hidden
    }
}

final class Attention: Module {
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear
    let heads: Int
    let headDim: Int

    init(hidden: Int, heads: Int) {
        self.heads = heads
        headDim = hidden / heads
        _wqkv.wrappedValue = Linear(hidden, 3 * hidden, bias: false)
        _wo.wrappedValue = Linear(hidden, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        let batch = x.dim(0), count = x.dim(1)
        let qkv = wqkv(x).reshaped([batch, count, 3, heads, headDim])
        func part(_ index: Int) -> MLXArray { qkv[0..., 0..., index].transposed(0, 2, 1, 3) }
        let queries = rotate(part(0), cos: cos, sin: sin)
        let keys = rotate(part(1), cos: cos, sin: sin)
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask.map { .array($0) } ?? .none
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: part(2),
            scale: pow(Float(headDim), -0.5), mask: maskMode)
        return wo(attended.transposed(0, 2, 1, 3).reshaped([batch, count, heads * headDim]))
    }
}

final class GeGLU: Module {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear
    let intermediate: Int

    init(hidden: Int, intermediate: Int) {
        self.intermediate = intermediate
        _wi.wrappedValue = Linear(hidden, 2 * intermediate, bias: false)
        _wo.wrappedValue = Linear(intermediate, hidden, bias: false)
    }

    /// The maker's GLU: the first half goes through GELU, then multiplies the second half.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = wi(x)
        let value = projected[0..., 0..., 0 ..< intermediate]
        let gate = projected[0..., 0..., intermediate...]
        return wo(gelu(value) * gate)
    }
}

/// Rotary embedding tables for one theta: cos and sin of shape [length, headDim], in float32.
func rotary(length: Int, headDim: Int, theta: Double) -> (cos: MLXArray, sin: MLXArray) {
    let half = headDim / 2
    let inverse = (0 ..< half).map { Float(1.0 / pow(theta, Double(2 * $0) / Double(headDim))) }
    var angles = [Float]()
    angles.reserveCapacity(length * half)
    for position in 0 ..< length {
        for frequency in inverse { angles.append(Float(position) * frequency) }
    }
    let table = MLXArray(angles, [length, half])
    let full = concatenated([table, table], axis: -1)
    return (MLX.cos(full), MLX.sin(full))
}

/// Rotates [batch, heads, length, headDim] queries or keys: x * cos + rotate_half(x) * sin, in float32, then back to x's dtype.
func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let values = x.asType(.float32)
    let half = values.dim(-1) / 2
    let rotated = concatenated([-values[0..., 0..., 0..., half...], values[0..., 0..., 0..., ..<half]], axis: -1)
    return (values * cos + rotated * sin).asType(x.dtype)
}

/// Additive mask [length, length]: 0 where two tokens are at most `halfWindow` apart, -inf elsewhere.
func slidingWindowMask(length: Int, halfWindow: Int) -> MLXArray {
    let positions = MLXArray((0 ..< length).map { Float($0) })
    let distance = abs(positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0))
    let outside = greater(distance, MLXArray(Float(halfWindow)))
    return which(outside, MLXArray(-Float.infinity), MLXArray(Float(0))).asType(.float16)
}
