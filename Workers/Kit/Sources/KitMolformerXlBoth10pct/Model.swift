// SPDX-License-Identifier: MIT
// MoLFormer-XL (ibm-research/MoLFormer-XL-both-10pct), ported from the repository's modeling_molformer.py
// (MolformerModel): a 12-layer encoder with rotary positions and linear attention through a fixed random feature map
// (the checkpoint's feature_map.weight, used as is: the card's deterministic_eval=True). The embedding is the
// card's pooler_output: the final LayerNorm output averaged over every token, <bos> and <eos> included.
// Tokenizer: tokenizer.json's own rule, written out here: the SMILES regex picks the tokens (other characters are
// dropped; this port refuses such input instead), each token maps through the WordLevel vocabulary (unknown -> <unk>), wrapped in <bos> ... <eos>.
import Foundation
import MLX
import MLXNN

let modelKind = "local.molecules"
let modelOps = ["embed_molecule"]

final class Model {
    let net: MolformerNet
    let vocab: [String: Int]
    let bosId: Int, eosId: Int, unkId: Int
    let maxTokens: Int
    let pattern: NSRegularExpression

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        let hidden = config["hidden_size"] as? Int ?? 768
        net = MolformerNet(
            vocab: config["vocab_size"] as? Int ?? 2362, hidden: hidden,
            intermediate: config["intermediate_size"] as? Int ?? 768,
            layers: config["num_hidden_layers"] as? Int ?? 12, heads: config["num_attention_heads"] as? Int ?? 12,
            features: config["num_random_features"] as? Int ?? 32,
            eps: Float(config["layer_norm_eps"] as? Double ?? 1e-12),
            attentionEps: Float(config["linear_attention_eps"] as? Double ?? 1e-6))
        guard (config["feature_map_kernel"] as? String ?? "relu") == "relu", (config["hidden_act"] as? String ?? "gelu") == "gelu" else {
            throw RunnerError.failed("Only MoLFormer checkpoints with a relu feature map and gelu are supported.")
        }
        // The masked-LM head is not used for embeddings.
        try Weights.apply(try Weights.load(weightsDir), to: net) { key in
            guard key.hasPrefix("molformer.") else { return nil }
            return String(key.dropFirst("molformer.".count))
        }
        let tokenizer = try Files.json(weightsDir, "tokenizer.json")
        guard let model = tokenizer["model"] as? [String: Any], let words = model["vocab"] as? [String: Int],
              let pre = tokenizer["pre_tokenizer"] as? [String: Any], let steps = pre["pretokenizers"] as? [[String: Any]],
              let regex = ((steps.last?["pattern"] as? [String: Any])?["Regex"] as? String),
              let bos = words["<bos>"], let eos = words["<eos>"], let unk = words["<unk>"]
        else { throw RunnerError.failed("tokenizer.json is not MoLFormer's SMILES tokenizer.") }
        vocab = words; bosId = bos; eosId = eos; unkId = unk
        pattern = try NSRegularExpression(pattern: regex)
        // Training dropped molecules longer than 202 tokens (the card), <bos> and <eos> included.
        maxTokens = config["max_position_embeddings"] as? Int ?? 202
    }

    func tokens(_ smiles: String) -> [String] {
        let range = NSRange(smiles.startIndex..., in: smiles)
        return pattern.matches(in: smiles, range: range).compactMap { Range($0.range, in: smiles).map { String(smiles[$0]) } }
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        let smiles = try params.string("smiles").trimmingCharacters(in: .whitespacesAndNewlines)
        // Inner whitespace would be skipped by the tokenizer, joining "CCO CCN" into one molecule.
        guard !smiles.contains(where: { $0.isWhitespace }) else {
            throw RunnerError.invalid("smiles must be one SMILES string without spaces; send one molecule per request.")
        }
        let pieces = tokens(smiles)
        guard !pieces.isEmpty else { throw RunnerError.invalid("smiles holds no SMILES tokens (atoms, bonds, rings or branches).") }
        // The tokenizer silently drops characters outside SMILES; a request with any is not a SMILES string.
        let read = pieces.joined().count, given = smiles.filter { !$0.isWhitespace }.count
        guard read == given else {
            throw RunnerError.invalid("smiles is not a SMILES string: \(given - read) of its \(given) characters are not SMILES (e.g. lowercase letters other than aromatic b c n o p s).")
        }
        let ids = [bosId] + pieces.map { vocab[$0] ?? unkId } + [eosId]
        guard ids.count <= maxTokens else {
            throw RunnerError.invalid("smiles has \(pieces.count) tokens; MoLFormer was trained on molecules of at most \(maxTokens - 2).")
        }
        let unknown = pieces.filter { vocab[$0] == nil }
        let hidden = try net(MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0), params: params)[0]
        let pooled = hidden.asType(.float32).mean(axis: 0)
        eval(pooled)
        var out: [String: Any] = ["embedding": list(pooled), "dimensions": pooled.dim(0), "tokens": pieces.count]
        if !unknown.isEmpty { out["unknown_tokens"] = unknown }
        return out
    }
}

// MARK: network (names follow the checkpoint, prefix "molformer." removed)

final class MolformerNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: MolformerEmbeddings
    @ModuleInfo(key: "encoder") var encoder: MolformerEncoder
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    let headDim: Int

    init(vocab: Int, hidden: Int, intermediate: Int, layers: Int, heads: Int, features: Int, eps: Float, attentionEps: Float) {
        headDim = hidden / heads
        _embeddings.wrappedValue = MolformerEmbeddings(vocab: vocab, hidden: hidden)
        _encoder.wrappedValue = MolformerEncoder(
            hidden: hidden, intermediate: intermediate, layers: layers, heads: heads, features: features, eps: eps, attentionEps: attentionEps)
        _layerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }

    /// ids [1, L] -> final hidden state [1, L, hidden] (after the model's LayerNorm).
    func callAsFunction(_ ids: MLXArray, params: Params) throws -> MLXArray {
        var x = embeddings.wordEmbeddings(ids)
        // Rotary tables as MolformerRotaryEmbedding builds them: float32 angles, cos/sin cast to the activations' dtype.
        let length = ids.dim(1)
        let invFreq = 1 / pow(MLXArray(Float(10000)), MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) }))
        let freqs = MLXArray((0 ..< length).map(Float.init)).expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let angles = concatenated([freqs, freqs], axis: -1)
        let cos = MLX.cos(angles).asType(x.dtype), sin = MLX.sin(angles).asType(x.dtype)
        for layer in encoder.layer {
            x = layer(x, cos: cos, sin: sin)
            try params.checkCancelled()
        }
        return layerNorm(x)
    }
}

final class MolformerEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    init(vocab: Int, hidden: Int) { _wordEmbeddings.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden) }
}

final class MolformerEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [MolformerLayer]
    init(hidden: Int, intermediate: Int, layers: Int, heads: Int, features: Int, eps: Float, attentionEps: Float) {
        _layer.wrappedValue = (0 ..< layers).map { _ in
            MolformerLayer(hidden: hidden, intermediate: intermediate, heads: heads, features: features, eps: eps, attentionEps: attentionEps)
        }
    }
}

/// BERT-style dense + LayerNorm(dense(x) + residual).
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

final class FeatureMap: Module {
    /// [head_dim, num_random_features]: the orthogonal random projection saved in the checkpoint.
    var weight: MLXArray
    init(headDim: Int, features: Int) { weight = MLXArray.zeros([headDim, features]) }
}

final class MolformerSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "feature_map") var featureMap: FeatureMap
    let heads: Int, eps: Float
    init(hidden: Int, heads: Int, features: Int, eps: Float) {
        self.heads = heads; self.eps = eps
        _query.wrappedValue = Linear(hidden, hidden)
        _key.wrappedValue = Linear(hidden, hidden)
        _value.wrappedValue = Linear(hidden, hidden)
        _featureMap.wrappedValue = FeatureMap(headDim: hidden / heads, features: features)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = d / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, l, heads, headDim]).transposed(0, 2, 1, 3) }
        func rotary(_ t: MLXArray) -> MLXArray {
            let half = headDim / 2
            let rotated = concatenated([-t[.ellipsis, half...], t[.ellipsis, ..<half]], axis: -1)
            return t * cos + rotated * sin
        }
        // relu random features of the rotated queries and keys (all tokens are real: no padding mask).
        let q = relu(matmul(rotary(split(query(x))), featureMap.weight))
        let k = relu(matmul(rotary(split(key(x))), featureMap.weight))
        let v = split(value(x))
        // Linear attention: (q (k^T v)) / clamp(q . sum_k, eps). The sums over tokens run in float32.
        let q32 = q.asType(.float32), k32 = k.asType(.float32)
        let kv = matmul(k32.transposed(0, 1, 3, 2), v.asType(.float32))
        let norm = maximum(matmul(q32, k32.sum(axis: -2).expandedDimensions(axis: -1)), MLXArray(eps))
        let context = (matmul(q32, kv) / norm).asType(x.dtype)
        return context.transposed(0, 2, 1, 3).reshaped([b, l, d])
    }
}

final class MolformerAttention: Module {
    @ModuleInfo(key: "self") var attn: MolformerSelfAttention
    @ModuleInfo(key: "output") var output: DenseNorm
    init(hidden: Int, heads: Int, features: Int, eps: Float, attentionEps: Float) {
        _attn.wrappedValue = MolformerSelfAttention(hidden: hidden, heads: heads, features: features, eps: attentionEps)
        _output.wrappedValue = DenseNorm(hidden, hidden, eps: eps)
    }
}

final class MolformerLayer: Module {
    @ModuleInfo(key: "attention") var attention: MolformerAttention
    @ModuleInfo(key: "intermediate") var intermediate: Dense
    @ModuleInfo(key: "output") var output: DenseNorm
    init(hidden: Int, intermediate inner: Int, heads: Int, features: Int, eps: Float, attentionEps: Float) {
        _attention.wrappedValue = MolformerAttention(hidden: hidden, heads: heads, features: features, eps: eps, attentionEps: attentionEps)
        _intermediate.wrappedValue = Dense(hidden, inner)
        _output.wrappedValue = DenseNorm(inner, hidden, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let a = attention.output(attention.attn(x, cos: cos, sin: sin), residual: x)
        // ACT2FN["gelu"] is the exact erf form (MLXNN.gelu).
        return output(gelu(intermediate.dense(a)), residual: a)
    }
}
