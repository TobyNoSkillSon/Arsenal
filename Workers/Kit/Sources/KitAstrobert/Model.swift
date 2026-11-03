// SPDX-License-Identifier: MIT
// astroBERT (adsabs/astroBERT): a BERT-base encoder pretrained on astrophysics papers.
// Ported from transformers' modeling_bert.py (BertModel: embeddings, encoder; the pooler and the MLM/NSP heads are unused).
// Embeddings follow the maker's Tutorial 0 (0_Embeddings.ipynb): the last hidden state, mean-pooled over every token
// of the sequence ([CLS] and [SEP] included, as its attention-mask mean does).
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
    let net: BertEncoder
    let tokenizer: Tokenizer
    let clsId: Int, sepId: Int
    let maxTokens: Int
    let width: Int

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, *.safetensors, tokenizer files) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        func int(_ key: String) throws -> Int {
            guard let value = config[key] as? Int else { throw RunnerError.failed("config.json has no \(key).") }
            return value
        }
        width = try int("hidden_size")
        maxTokens = try int("max_position_embeddings")
        net = BertEncoder(
            vocab: try int("vocab_size"), hidden: width, layers: try int("num_hidden_layers"), heads: try int("num_attention_heads"),
            intermediate: try int("intermediate_size"), positions: maxTokens, types: try int("type_vocab_size"),
            eps: Float(config["layer_norm_eps"] as? Double ?? 1e-12))
        // The checkpoint is BertForPreTraining: keep bert.embeddings.* and bert.encoder.*; drop the pooler, the MLM/NSP
        // heads (cls.*) and the position_ids buffer.
        try Weights.apply(try Weights.load(weightsDir), to: net) { key in
            guard key.hasPrefix("bert.embeddings.") || key.hasPrefix("bert.encoder."), !key.hasSuffix("position_ids") else { return nil }
            return String(key.dropFirst("bert.".count))
        }
        tokenizer = try Text.tokenizer(weightsDir)
        guard let cls = tokenizer.convertTokenToId("[CLS]"), let sep = tokenizer.convertTokenToId("[SEP]") else {
            throw RunnerError.failed("The tokenizer has no [CLS] or [SEP] token.")
        }
        clsId = cls
        sepId = sep
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        guard let items = params.raw["input"] as? [Any], !items.isEmpty else {
            throw RunnerError.invalid("input must be a nonempty list of texts.")
        }
        let texts: [String] = try items.map { item in
            if let text = item as? String { return text }
            if let object = item as? [String: Any], object["type"] as? String == "text", let text = object["text"] as? String { return text }
            throw RunnerError.invalid("astroBERT reads text only; images, audio and mixed content are not supported.")
        }
        if params.raw["instruction"] != nil {
            throw RunnerError.invalid("astroBERT takes no instruction; send the text alone.")
        }
        if let dimensions = params.raw["dimensions"], (dimensions as? NSNumber)?.intValue != width {
            throw RunnerError.invalid("astroBERT embeddings have \(width) dimensions; it cannot shorten them.")
        }
        // task is accepted for the kind's sake: astroBERT embeds queries and documents the same way.
        let normalize = params.bool("normalize", default: true)
        let truncate = params.bool("truncate", default: false)

        var vectors: [[Double]] = []
        var total = 0
        for text in texts {
            try params.checkCancelled()
            var pieces = tokenizer.tokenize(text: Self.clean(text)).compactMap { tokenizer.convertTokenToId($0) }
            if pieces.count > maxTokens - 2 {
                guard truncate else {
                    throw RunnerError.invalid("A text is \(pieces.count + 2) tokens, past astroBERT's \(maxTokens); send truncate: true to cut it.")
                }
                pieces = Array(pieces.prefix(maxTokens - 2))
            }
            let ids = [clsId] + pieces + [sepId]
            total += ids.count
            let hidden = net(MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0))[0].asType(.float32)  // [T, H]
            var vector = hidden.mean(axis: 0)
            if normalize { vector = vector / maximum(sqrt((vector * vector).sum()), MLXArray(Float(1e-12))) }
            vectors.append(list(vector))
        }
        return ["embeddings": vectors, "input_tokens": total]
    }

    /// The parts of the tokenizer's BertNormalizer clean_text that swift-transformers skips: remove NUL, U+FFFD and
    /// control/format characters, and turn every whitespace (newlines included) into a space.
    static func clean(_ text: String) -> String {
        var scalars = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            if scalar.properties.isWhitespace { scalars.append(" "); continue }
            if scalar.value == 0 || scalar.value == 0xFFFD { continue }
            switch scalar.properties.generalCategory {
            case .control, .format: continue
            default: scalars.append(scalar)
            }
        }
        return String(scalars)
    }
}

// MARK: - BERT (modeling_bert.py), named after the checkpoint's keys under bert.

final class BertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var words: Embedding
    @ModuleInfo(key: "position_embeddings") var positions: Embedding
    @ModuleInfo(key: "token_type_embeddings") var types: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(vocab: Int, hidden: Int, positions: Int, types: Int, eps: Float) {
        _words.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden)
        _positions.wrappedValue = Embedding(embeddingCount: positions, dimensions: hidden)
        _types.wrappedValue = Embedding(embeddingCount: types, dimensions: hidden)
        _norm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let length = ids.dim(1)
        let position = MLXArray(Int32(0)..<Int32(length)).expandedDimensions(axis: 0)
        // One segment: token_type_ids are all 0.
        let x = words(ids) + positions(position) + types(MLXArray.zeros(like: ids))
        return norm(x)
    }
}

final class SelfAttention: Module {
    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear
    init(hidden: Int) {
        _query.wrappedValue = Linear(hidden, hidden)
        _key.wrappedValue = Linear(hidden, hidden)
        _value.wrappedValue = Linear(hidden, hidden)
    }
}

final class DenseNorm: Module {
    @ModuleInfo var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm
    init(inputs: Int, hidden: Int, eps: Float) {
        _dense.wrappedValue = Linear(inputs, hidden)
        _norm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }
    /// BertSelfOutput / BertOutput: LayerNorm(dense(x) + residual); dropout is off at inference.
    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray { norm(dense(x) + residual) }
}

final class Attention: Module {
    @ModuleInfo(key: "self") var attention: SelfAttention
    @ModuleInfo var output: DenseNorm
    let heads: Int
    init(hidden: Int, heads: Int, eps: Float) {
        self.heads = heads
        _attention.wrappedValue = SelfAttention(hidden: hidden)
        _output.wrappedValue = DenseNorm(inputs: hidden, hidden: hidden, eps: eps)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, h) = (x.dim(0), x.dim(1), x.dim(2))
        let d = h / heads
        func split(_ y: MLXArray) -> MLXArray { y.reshaped([b, t, heads, d]).transposed(0, 2, 1, 3) }
        // One unpadded sequence per call, so no attention mask is needed.
        let context = MLXFast.scaledDotProductAttention(
            queries: split(attention.query(x)), keys: split(attention.key(x)), values: split(attention.value(x)),
            scale: 1 / Float(d).squareRoot(), mask: nil)
        return output(context.transposed(0, 2, 1, 3).reshaped([b, t, h]), residual: x)
    }
}

final class Intermediate: Module {
    @ModuleInfo var dense: Linear
    init(hidden: Int, intermediate: Int) { _dense.wrappedValue = Linear(hidden, intermediate) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(dense(x)) }  // hidden_act "gelu" is the exact erf GELU
}

final class BertLayer: Module {
    @ModuleInfo var attention: Attention
    @ModuleInfo var intermediate: Intermediate
    @ModuleInfo var output: DenseNorm
    init(hidden: Int, heads: Int, intermediate: Int, eps: Float) {
        _attention.wrappedValue = Attention(hidden: hidden, heads: heads, eps: eps)
        _intermediate.wrappedValue = Intermediate(hidden: hidden, intermediate: intermediate)
        _output.wrappedValue = DenseNorm(inputs: intermediate, hidden: hidden, eps: eps)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = attention(x)
        return output(intermediate(a), residual: a)
    }
}

final class Encoder: Module {
    @ModuleInfo var layer: [BertLayer]
    init(count: Int, hidden: Int, heads: Int, intermediate: Int, eps: Float) {
        _layer.wrappedValue = (0..<count).map { _ in BertLayer(hidden: hidden, heads: heads, intermediate: intermediate, eps: eps) }
    }
}

/// BertModel without the pooler: token ids [1, T] in, last hidden state [1, T, H] out.
final class BertEncoder: Module {
    @ModuleInfo var embeddings: BertEmbeddings
    @ModuleInfo var encoder: Encoder

    init(vocab: Int, hidden: Int, layers: Int, heads: Int, intermediate: Int, positions: Int, types: Int, eps: Float) {
        _embeddings.wrappedValue = BertEmbeddings(vocab: vocab, hidden: hidden, positions: positions, types: types, eps: eps)
        _encoder.wrappedValue = Encoder(count: layers, hidden: hidden, heads: heads, intermediate: intermediate, eps: eps)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var x = embeddings(ids)
        for block in encoder.layer { x = block(x) }
        return x
    }
}
