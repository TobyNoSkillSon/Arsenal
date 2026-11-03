// SPDX-License-Identifier: MIT
// Geneformer V2-104M (ctheodoris/Geneformer, subfolder Geneformer-V2-104M): a BERT encoder (transformers BertModel,
// relu, absolute positions, 4,096 tokens) over a cell's rank value encoding, ported from transformers' modeling_bert.py.
// Input follows the maker's TranscriptomeTokenizer for the V2 series: the cell's genes as Ensembl IDs, highest rank
// first, mapped through the 104M token dictionary (geneformer/token_dictionary_gc104M.pkl, read here with a small
// pickle reader); genes missing from the dictionary are skipped, the list is cut to 4,094 genes and wrapped in
// <cls> ... <eos>. Output follows the maker's EmbExtractor defaults: emb_mode "cls" at emb_layer -1 (the CLS token
// of the second-to-last layer), plus emb_mode "cell" (the mean over the gene tokens) from the same layer.
import Foundation
import MLX
import MLXNN

let modelKind = "local.cells"
let modelOps = ["embed_cells"]

let variantFolder = "Geneformer-V2-104M"
let dictionaryFile = "geneformer/token_dictionary_gc104M.pkl"

final class Model {
    let net: BertNet
    let tokens: [String: Int]
    let clsId: Int, eosId: Int, padId: Int, maskId: Int
    let maxTokens: Int, vocabSize: Int

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let dir = weightsDir.appendingPathComponent(variantFolder)
        let config = try Files.json(dir)
        guard (config["hidden_act"] as? String ?? "relu") == "relu", (config["position_embedding_type"] as? String ?? "absolute") == "absolute" else {
            throw RunnerError.failed("Only Geneformer V2 checkpoints (relu, absolute positions) are supported.")
        }
        let hidden = config["hidden_size"] as? Int ?? 768
        vocabSize = config["vocab_size"] as? Int ?? 20275
        maxTokens = config["max_position_embeddings"] as? Int ?? 4096
        net = BertNet(
            vocab: vocabSize, hidden: hidden, intermediate: config["intermediate_size"] as? Int ?? 3072,
            layers: config["num_hidden_layers"] as? Int ?? 12, heads: config["num_attention_heads"] as? Int ?? 12,
            positions: maxTokens, types: config["type_vocab_size"] as? Int ?? 2,
            eps: Float(config["layer_norm_eps"] as? Double ?? 1e-12))
        // The masked-LM head (cls.*) is not used for embeddings.
        try Weights.apply(try Weights.load(dir), to: net) { key in
            guard key.hasPrefix("bert."), !key.hasSuffix("position_ids") else { return nil }
            return String(key.dropFirst(5))
        }
        let data = try Data(contentsOf: weightsDir.appendingPathComponent(dictionaryFile))
        tokens = try Pickle.stringIntDictionary(data)
        guard let cls = tokens["<cls>"], let eos = tokens["<eos>"], let pad = tokens["<pad>"], let mask = tokens["<mask>"], tokens.count > 1000 else {
            throw RunnerError.failed("\(dictionaryFile) is not a Geneformer V2 token dictionary.")
        }
        clsId = cls; eosId = eos; padId = pad; maskId = mask
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        var genes: [Int] = []
        var skipped = 0
        if params.raw["genes"] != nil && params.raw["gene_tokens"] != nil {
            throw RunnerError.invalid("Give either genes or gene_tokens, not both.")
        }
        if params.raw["genes"] != nil {
            for id in try params.strings("genes") {
                let key = id.trimmingCharacters(in: .whitespaces).uppercased()
                // Version suffixes (ENSG00000141510.17) are not in the dictionary.
                let bare = key.split(separator: ".").first.map(String.init) ?? key
                if let token = tokens[bare], bare.hasPrefix("ENSG") { genes.append(token) } else { skipped += 1 }
            }
        } else if params.raw["gene_tokens"] != nil {
            for value in try params.numbers("gene_tokens") {
                // Range-check before converting: Int(1e20) or Int(.nan) would trap.
                guard value >= 0, value < Double(vocabSize), let token = Int(exactly: value),
                      ![clsId, eosId, padId, maskId].contains(token) else {
                    throw RunnerError.invalid("gene_tokens must be gene token ids from the 104M dictionary (integers 4-\(vocabSize - 1)); found \(value).")
                }
                genes.append(token)
            }
        } else {
            throw RunnerError.invalid("Give genes (Ensembl gene IDs, highest rank first) or gene_tokens (token ids).")
        }
        guard !genes.isEmpty else {
            throw RunnerError.invalid("None of the \(skipped) genes is in Geneformer's dictionary: use human Ensembl gene IDs such as ENSG00000141510.")
        }
        let truncated = max(0, genes.count - (maxTokens - 2))
        if truncated > 0 { genes = Array(genes.prefix(maxTokens - 2)) }
        let ids = [clsId] + genes + [eosId]
        let lastLayer = params.bool("last_layer", default: false)
        let hidden = try net(MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0), stopAfter: lastLayer ? nil : 1, params: params)[0]
        let h = hidden.asType(.float32)
        let cls = h[0]
        let mean = h[1 ..< (ids.count - 1)].mean(axis: 0)
        eval(cls, mean)
        return [
            "embedding": list(cls), "mean_embedding": list(mean), "dimensions": cls.dim(0),
            "genes_used": genes.count, "genes_skipped": skipped, "genes_truncated": truncated,
        ]
    }
}

// MARK: pickle (protocol 2-5 dict of str -> int, as the Geneformer token dictionaries are saved)

enum Pickle {
    enum Value { case mark, string(String), int(Int), dict([String: Int]) }

    static func stringIntDictionary(_ data: Data) throws -> [String: Int] {
        let bytes = [UInt8](data)
        var i = 0, stack: [Value] = [], memo: [Int: Value] = [:]
        func fail() -> RunnerError { RunnerError.failed("token dictionary: unsupported pickle content at byte \(i).") }
        func read(_ n: Int) throws -> [UInt8] {
            guard i + n <= bytes.count else { throw fail() }
            defer { i += n }
            return Array(bytes[i ..< i + n])
        }
        func uint(_ b: [UInt8]) -> Int { b.enumerated().reduce(0) { $0 | (Int($1.element) << (8 * $1.offset)) } }
        func string(_ n: Int) throws -> Value {
            guard let s = String(bytes: try read(n), encoding: .utf8) else { throw fail() }
            return .string(s)
        }
        func setItems(from start: Int) throws {
            guard start > 0, case .dict(var dict) = stack[start - 1] else { throw fail() }
            var k = start
            while k + 1 < stack.count {
                guard case .string(let key) = stack[k], case .int(let value) = stack[k + 1] else { throw fail() }
                dict[key] = value; k += 2
            }
            stack.removeSubrange((start - 1)...)
            stack.append(.dict(dict))
        }
        while i < bytes.count {
            let op = bytes[i]; i += 1
            switch op {
            case 0x80: _ = try read(1)                                   // PROTO
            case 0x95: _ = try read(8)                                   // FRAME
            case 0x7d: stack.append(.dict([:]))                          // EMPTY_DICT
            case 0x28: stack.append(.mark)                               // MARK
            case 0x94: guard let top = stack.last else { throw fail() }; memo[memo.count] = top  // MEMOIZE
            case 0x71: guard let top = stack.last else { throw fail() }; memo[uint(try read(1))] = top  // BINPUT
            case 0x72: guard let top = stack.last else { throw fail() }; memo[uint(try read(4))] = top  // LONG_BINPUT
            case 0x68: guard let v = memo[uint(try read(1))] else { throw fail() }; stack.append(v)  // BINGET
            case 0x6a: guard let v = memo[uint(try read(4))] else { throw fail() }; stack.append(v)  // LONG_BINGET
            case 0x8c: stack.append(try string(uint(try read(1))))       // SHORT_BINUNICODE
            case 0x58: stack.append(try string(uint(try read(4))))       // BINUNICODE
            case 0x4b: stack.append(.int(uint(try read(1))))             // BININT1
            case 0x4d: stack.append(.int(uint(try read(2))))             // BININT2
            case 0x4a: stack.append(.int(Int(Int32(bitPattern: UInt32(uint(try read(4)))))))  // BININT
            case 0x75:                                                   // SETITEMS
                guard let mark = stack.lastIndex(where: { if case .mark = $0 { return true }; return false }) else { throw fail() }
                stack.remove(at: mark)
                try setItems(from: mark)
            case 0x73: try setItems(from: stack.count - 2)               // SETITEM
            case 0x2e:                                                   // STOP
                guard case .dict(let dict) = stack.last else { throw fail() }
                return dict
            default: throw fail()
            }
        }
        throw fail()
    }
}

// MARK: network (names follow the checkpoint, prefix "bert." removed)

final class BertNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: BertEmbeddings
    @ModuleInfo(key: "encoder") var encoder: BertEncoder

    init(vocab: Int, hidden: Int, intermediate: Int, layers: Int, heads: Int, positions: Int, types: Int, eps: Float) {
        _embeddings.wrappedValue = BertEmbeddings(vocab: vocab, hidden: hidden, positions: positions, types: types, eps: eps)
        _encoder.wrappedValue = BertEncoder(hidden: hidden, intermediate: intermediate, layers: layers, heads: heads, eps: eps)
    }

    /// ids [1, L] -> hidden state [1, L, hidden] after all layers but the last `stopAfter` (nil: every layer).
    func callAsFunction(_ ids: MLXArray, stopAfter: Int?, params: Params) throws -> MLXArray {
        var x = embeddings(ids)
        let count = encoder.layer.count - (stopAfter ?? 0)
        for layer in encoder.layer.prefix(count) {
            x = layer(x)
            try params.checkCancelled()
        }
        return x
    }
}

final class BertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    init(vocab: Int, hidden: Int, positions: Int, types: Int, eps: Float) {
        _wordEmbeddings.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden)
        _positionEmbeddings.wrappedValue = Embedding(embeddingCount: positions, dimensions: hidden)
        _tokenTypeEmbeddings.wrappedValue = Embedding(embeddingCount: types, dimensions: hidden)
        _layerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let length = ids.dim(1)
        let positions = MLXArray(Int32(0) ..< Int32(length)).expandedDimensions(axis: 0)
        // token_type_ids are all 0 (one segment).
        let x = wordEmbeddings(ids) + positionEmbeddings(positions) + tokenTypeEmbeddings.weight[0]
        return layerNorm(x)
    }
}

final class BertEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [BertLayer]
    init(hidden: Int, intermediate: Int, layers: Int, heads: Int, eps: Float) {
        _layer.wrappedValue = (0 ..< layers).map { _ in BertLayer(hidden: hidden, intermediate: intermediate, heads: heads, eps: eps) }
    }
}

final class DenseNorm: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    init(_ inDim: Int, _ outDim: Int, eps: Float) {
        _dense.wrappedValue = Linear(inDim, outDim)
        _layerNorm.wrappedValue = LayerNorm(dimensions: outDim, eps: eps)
    }
    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray { layerNorm(dense(x) + residual) }
}

final class Dense: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ inDim: Int, _ outDim: Int) { _dense.wrappedValue = Linear(inDim, outDim) }
}

final class BertSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    init(hidden: Int) {
        _query.wrappedValue = Linear(hidden, hidden)
        _key.wrappedValue = Linear(hidden, hidden)
        _value.wrappedValue = Linear(hidden, hidden)
    }
}

final class BertAttention: Module {
    @ModuleInfo(key: "self") var attn: BertSelfAttention
    @ModuleInfo(key: "output") var output: DenseNorm
    let heads: Int
    init(hidden: Int, heads: Int, eps: Float) {
        self.heads = heads
        _attn.wrappedValue = BertSelfAttention(hidden: hidden)
        _output.wrappedValue = DenseNorm(hidden, hidden, eps: eps)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = d / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, l, heads, headDim]).transposed(0, 2, 1, 3) }
        let o = MLXFast.scaledDotProductAttention(
            queries: split(attn.query(x)), keys: split(attn.key(x)), values: split(attn.value(x)),
            scale: Float(1 / Double(headDim).squareRoot()), mask: nil)
        return output(o.transposed(0, 2, 1, 3).reshaped([b, l, d]), residual: x)
    }
}

final class BertLayer: Module {
    @ModuleInfo(key: "attention") var attention: BertAttention
    @ModuleInfo(key: "intermediate") var intermediate: Dense
    @ModuleInfo(key: "output") var output: DenseNorm
    init(hidden: Int, intermediate inner: Int, heads: Int, eps: Float) {
        _attention.wrappedValue = BertAttention(hidden: hidden, heads: heads, eps: eps)
        _intermediate.wrappedValue = Dense(hidden, inner)
        _output.wrappedValue = DenseNorm(inner, hidden, eps: eps)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = attention(x)
        return output(relu(intermediate.dense(a)), residual: a)
    }
}
