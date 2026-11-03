// SPDX-License-Identifier: MIT
// ESM-2 protein language model (facebook/esm2_*), ported from transformers' modeling_esm.py (EsmModel).
// One file serves every ESM-2 size: layer count, width and heads come from config.json.
// Tokenizer: the kit's Text.tokenizer needs tokenizer.json, which ESM-2 repos do not ship (EsmTokenizer reads
// vocab.txt), so this file has a per-character tokenizer built from vocab.txt: <cls> residues <eos>.
import Foundation
import MLX
import MLXNN

let modelKind = "local.science"
let modelOps = ["embed_sequence"]

final class Model {
    let net: EsmNet
    let vocab: [Character: Int]
    let clsId: Int, eosId: Int
    let maxResidues: Int

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        guard (config["position_embedding_type"] as? String ?? "absolute") == "rotary",
              (config["emb_layer_norm_before"] as? Bool ?? false) == false
        else { throw RunnerError.failed("Only ESM-2 checkpoints (rotary, no embedding LayerNorm) are supported.") }
        let hidden = config["hidden_size"] as? Int ?? 320
        let heads = config["num_attention_heads"] as? Int ?? 20
        net = EsmNet(
            vocab: config["vocab_size"] as? Int ?? 33, hidden: hidden,
            intermediate: config["intermediate_size"] as? Int ?? hidden * 4,
            layers: config["num_hidden_layers"] as? Int ?? 6, heads: heads,
            eps: Float(config["layer_norm_eps"] as? Double ?? 1e-5),
            ropeBase: Float(config["rope_theta"] as? Double ?? 10000),
            tokenDropout: config["token_dropout"] as? Bool ?? true)
        // The checkpoint also holds the masked-LM head, the contact head, unused absolute position embeddings
        // and per-layer rotary inv_freq copies; the encoder does not use them.
        try Weights.apply(try Weights.load(weightsDir), to: net) { key in
            guard key.hasPrefix("esm."), !key.hasSuffix("inv_freq"), !key.hasSuffix("position_ids"),
                  !key.contains("position_embeddings"), !key.contains("contact_head") else { return nil }
            return String(key.dropFirst(4))
        }
        let lines = try String(contentsOf: weightsDir.appendingPathComponent("vocab.txt"), encoding: .utf8)
            .split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
        var map: [Character: Int] = [:]
        for (i, token) in lines.enumerated() where token.count == 1 {
            let c = Character(token)
            if c.isLetter && c.isUppercase { map[c] = i }
        }
        guard let cls = lines.firstIndex(of: "<cls>"), let eos = lines.firstIndex(of: "<eos>"), map.count >= 20 else {
            throw RunnerError.failed("vocab.txt is not an ESM vocabulary.")
        }
        vocab = map; clsId = cls; eosId = eos
        // ESM-2 was trained on crops of 1,022 residues (1,024 tokens with <cls> and <eos>).
        maxResidues = (config["max_position_embeddings"] as? Int ?? 1026) - 4
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        let raw = try params.string("sequence")
        let residues = raw.filter { !$0.isWhitespace }
        guard !residues.isEmpty else { throw RunnerError.invalid("sequence must hold at least one amino-acid letter.") }
        var ids = [clsId]
        for c in residues {
            guard let id = vocab[c] else {
                let allowed = String(vocab.keys.sorted())
                throw RunnerError.invalid("sequence may hold only uppercase amino-acid letters (\(allowed)); found \"\(c)\".")
            }
            ids.append(id)
        }
        ids.append(eosId)
        guard residues.count <= maxResidues else {
            throw RunnerError.invalid("sequence has \(residues.count) residues; ESM-2 takes at most \(maxResidues).")
        }
        let hidden = try net(MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0), params: params)[0]
        let perResidue = hidden[1 ..< (ids.count - 1)].asType(.float32)
        let pooled = perResidue.mean(axis: 0)
        eval(pooled)
        var out: [String: Any] = ["embedding": list(pooled), "dimensions": pooled.dim(0), "residues": residues.count]
        if params.bool("per_residue", default: false) {
            let rows = perResidue.asArray(Float.self), width = pooled.dim(0)
            out["per_residue"] = (0 ..< residues.count).map { r in rows[(r * width) ..< ((r + 1) * width)].map(Double.init) }
        }
        return out
    }
}

// MARK: network (names follow the checkpoint, prefix "esm." removed)

final class EsmNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: EsmEmbeddings
    @ModuleInfo(key: "encoder") var encoder: EsmEncoder
    let headDim: Int, ropeBase: Float, tokenDropout: Bool

    init(vocab: Int, hidden: Int, intermediate: Int, layers: Int, heads: Int, eps: Float, ropeBase: Float, tokenDropout: Bool) {
        headDim = hidden / heads
        self.ropeBase = ropeBase
        self.tokenDropout = tokenDropout
        _embeddings.wrappedValue = EsmEmbeddings(vocab: vocab, hidden: hidden)
        _encoder.wrappedValue = EsmEncoder(hidden: hidden, intermediate: intermediate, layers: layers, heads: heads, eps: eps)
    }

    /// ids [1, L] -> last hidden state [1, L, hidden] (after emb_layer_norm_after).
    func callAsFunction(_ ids: MLXArray, params: Params) throws -> MLXArray {
        var x = embeddings.wordEmbeddings(ids)
        // token_dropout: the model was trained with 15% x 80% of tokens masked to zero, and scales embeddings
        // by (1 - 0.12) / (1 - observed mask ratio). Requests never contain <mask>, so the ratio is 0.
        if tokenDropout { x = x * Float(1 - 0.15 * 0.8) }
        // Rotary tables as transformers builds them: float32 angles, cos/sin cast to the activations' dtype.
        let length = ids.dim(1)
        let invFreq = 1 / pow(MLXArray(ropeBase), MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) }))
        let freqs = MLXArray((0 ..< length).map(Float.init)).expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let angles = concatenated([freqs, freqs], axis: -1)
        let cos = MLX.cos(angles).asType(x.dtype).asType(.float32)
        let sin = MLX.sin(angles).asType(x.dtype).asType(.float32)
        for layer in encoder.layer {
            x = layer(x, cos: cos, sin: sin)
            try params.checkCancelled()
        }
        return encoder.embLayerNormAfter(x)
    }
}

final class EsmEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    init(vocab: Int, hidden: Int) { _wordEmbeddings.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden) }
}

final class EsmEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [EsmLayer]
    @ModuleInfo(key: "emb_layer_norm_after") var embLayerNormAfter: LayerNorm
    init(hidden: Int, intermediate: Int, layers: Int, heads: Int, eps: Float) {
        _layer.wrappedValue = (0 ..< layers).map { _ in EsmLayer(hidden: hidden, intermediate: intermediate, heads: heads, eps: eps) }
        _embLayerNormAfter.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }
}

final class Dense: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ inDim: Int, _ outDim: Int) { _dense.wrappedValue = Linear(inDim, outDim) }
}

final class EsmSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    init(hidden: Int) {
        _query.wrappedValue = Linear(hidden, hidden)
        _key.wrappedValue = Linear(hidden, hidden)
        _value.wrappedValue = Linear(hidden, hidden)
    }
}

final class EsmAttention: Module {
    @ModuleInfo(key: "self") var attn: EsmSelfAttention
    @ModuleInfo(key: "output") var output: Dense
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    let heads: Int
    init(hidden: Int, heads: Int, eps: Float) {
        self.heads = heads
        _attn.wrappedValue = EsmSelfAttention(hidden: hidden)
        _output.wrappedValue = Dense(hidden, hidden)
        _layerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = d / heads
        let h = layerNorm(x)
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, l, heads, headDim]).transposed(0, 2, 1, 3) }
        // ESM scales the query before the rotary embedding; attention then uses scale 1.
        var q = split(attn.query(h)) * Float(1 / Double(headDim).squareRoot())
        var k = split(attn.key(h))
        let v = split(attn.value(h))
        q = rotary(q, cos: cos, sin: sin)
        k = rotary(k, cos: cos, sin: sin)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1, mask: nil)
        return output.dense(o.transposed(0, 2, 1, 3).reshaped([b, l, d])) + x
    }

    /// rotate_half rotary embedding, computed in float32 and cast back (as apply_rotary_pos_emb does).
    private func rotary(_ t: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let f = t.asType(.float32)
        let half = f.dim(-1) / 2
        let rotated = concatenated([-f[.ellipsis, half...], f[.ellipsis, ..<half]], axis: -1)
        return (f * cos + rotated * sin).asType(t.dtype)
    }
}

final class EsmLayer: Module {
    @ModuleInfo(key: "attention") var attention: EsmAttention
    @ModuleInfo(key: "intermediate") var intermediate: Dense
    @ModuleInfo(key: "output") var output: Dense
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    init(hidden: Int, intermediate inner: Int, heads: Int, eps: Float) {
        _attention.wrappedValue = EsmAttention(hidden: hidden, heads: heads, eps: eps)
        _intermediate.wrappedValue = Dense(hidden, inner)
        _output.wrappedValue = Dense(inner, hidden)
        _layerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let a = attention(x, cos: cos, sin: sin)
        // ESM's gelu is the exact erf form (MLXNN.gelu).
        return output.dense(gelu(intermediate.dense(layerNorm(a)))) + a
    }
}
