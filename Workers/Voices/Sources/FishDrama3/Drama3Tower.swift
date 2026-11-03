// SPDX-License-Identifier: MIT
// Drama 3's audio tower: samples codebooks 1…9 of a frame, attending over time. Reference: mlx-audio dad2087f
// `fish_drama3/tower.py`. Each of the 4 shared + 9 residual tower layers keeps a KV cache across frames; all run at the
// same temporal position and the codebook index selects the layer, not the RoPE position. A frame row is
// [main token, c0, …, c9], where c0 is the main token's offset in the semantic band.
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Gated GQA. `wqkv` rows are [all Q | all gates | K | V].
final class TowerAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, qWidth: Int, kvWidth: Int, rotaryDims: Int, ropeBase: Float
    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Fp32RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Fp32RMSNorm

    init(_ c: Drama3TowerConfig) {
        heads = c.heads; kvHeads = c.kvHeads; headDim = c.headDim; qWidth = c.heads * c.headDim; kvWidth = c.kvHeads * c.headDim
        rotaryDims = Int(Float(c.headDim) * c.rotaryPercent); ropeBase = c.ropeBase
        _wqkv.wrappedValue = Linear(c.dim, 2 * qWidth + 2 * kvWidth, bias: false)
        _wo.wrappedValue = Linear(qWidth, c.dim, bias: false)
        _qNorm.wrappedValue = Fp32RMSNorm(dimensions: c.headDim, eps: c.normEps)
        _kNorm.wrappedValue = Fp32RMSNorm(dimensions: c.headDim, eps: c.normEps)
        super.init()
    }

    /// Rotated in float32, as in the reference.
    func rope(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(x.asType(.float32), dimensions: rotaryDims, traditional: true, base: ropeBase, scale: 1, offset: offset).asType(x.dtype)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1)
        let parts = MLX.split(wqkv(x), indices: [qWidth, 2 * qWidth, 2 * qWidth + kvWidth], axis: -1)
        let q = qNorm(parts[0].reshaped(batch, length, heads, headDim))
        let k = kNorm(parts[2].reshaped(batch, length, kvHeads, headDim))
        let v = parts[3].reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        let offset = cache.offset
        let (keys, values) = cache.update(keys: rope(k.transposed(0, 2, 1, 3), offset: offset), values: v)
        let out = MLXFast.scaledDotProductAttention(
            queries: rope(q.transposed(0, 2, 1, 3), offset: offset), keys: keys, values: values, scale: pow(Float(headDim), -0.5),
            mask: length > 1 ? .causal : .none)
        return wo(out.transposed(0, 2, 1, 3).reshaped(batch, length, qWidth) * sigmoid(parts[1]))
    }
}

final class TowerMLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear
    init(_ c: Drama3TowerConfig) {
        _w1.wrappedValue = Linear(c.dim, c.intermediate, bias: false)
        _w2.wrappedValue = Linear(c.intermediate, c.dim, bias: false)
        _w3.wrappedValue = Linear(c.dim, c.intermediate, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class TowerBlock: Module {
    @ModuleInfo(key: "attention") var attention: TowerAttention
    @ModuleInfo(key: "attention_norm") var attentionNorm: Fp32RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: Fp32RMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: TowerMLP
    init(_ c: Drama3TowerConfig) {
        _attention.wrappedValue = TowerAttention(c)
        _attentionNorm.wrappedValue = Fp32RMSNorm(dimensions: c.dim, eps: c.normEps)
        _ffnNorm.wrappedValue = Fp32RMSNorm(dimensions: c.dim, eps: c.normEps)
        _feedForward.wrappedValue = TowerMLP(c)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let h = x + attention(attentionNorm(x), cache: cache)
        return h + feedForward(ffnNorm(h))
    }
}

/// Mixes a code embedding (zero on text rows) into the hidden state, then runs one block.
final class TowerLayer: Module {
    let enrichment: Bool
    let vocabSize: Int
    // Enrichment and residual layers name their embedding differently.
    @ModuleInfo(key: "semantic_embedding") var semanticEmbedding: Embedding?
    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding?
    @ModuleInfo(key: "enorm") var enorm: Fp32RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: Fp32RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "transformer_layer") var transformerLayer: TowerBlock
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: Fp32RMSNorm

    init(_ c: Drama3TowerConfig, enrichment: Bool) {
        self.enrichment = enrichment; vocabSize = c.vocabSize
        let embedding = Embedding(embeddingCount: c.vocabSize, dimensions: c.dim)
        if enrichment { _semanticEmbedding.wrappedValue = embedding } else { _codebookEmbedding.wrappedValue = embedding }
        _enorm.wrappedValue = Fp32RMSNorm(dimensions: c.dim, eps: c.normEps)
        _hnorm.wrappedValue = Fp32RMSNorm(dimensions: c.dim, eps: c.normEps)
        _ehProj.wrappedValue = Linear(2 * c.dim, c.dim, bias: false)
        _transformerLayer.wrappedValue = TowerBlock(c)
        _finalLayerNorm.wrappedValue = Fp32RMSNorm(dimensions: c.dim, eps: c.normEps)
        super.init()
    }

    /// hidden [B, T, D], codes [B, T], semantic [B, T] bool.
    func callAsFunction(_ hidden: MLXArray, codes: MLXArray, semantic: MLXArray, cache: KVCache, extra: MLXArray? = nil) -> MLXArray {
        let table = (enrichment ? semanticEmbedding : codebookEmbedding)!
        var e = table(clip(codes, min: 0, max: vocabSize - 1))
        e = MLX.where(expandedDimensions(semantic, axis: -1), e, MLXArray.zeros(like: e))
        if let extra { e += extra }
        let mixed = ehProj(concatenated([enorm(e), hnorm(hidden)], axis: -1))
        return finalLayerNorm(transformerLayer(mixed, cache: cache)).asType(mixed.dtype)
    }
}

final class Drama3Tower: Module {
    let vocabSize: Int, codebooks: Int, backboneCodebooks: Int, semanticBegin: Int, semanticEnd: Int
    /// One table for all codebooks; codebook i uses rows [i · vocab, (i+1) · vocab).
    @ModuleInfo(key: "codebook_embeddings") var codebookEmbeddings: Embedding
    @ModuleInfo(key: "shared_layers") var sharedLayers: [TowerLayer]
    @ModuleInfo(key: "residual_layers") var residualLayers: [TowerLayer]
    @ModuleInfo(key: "codebook_heads") var codebookHeads: [Linear]

    init(_ c: Drama3TowerConfig, semanticBegin: Int, semanticEnd: Int) {
        vocabSize = c.vocabSize; codebooks = c.codebooks; backboneCodebooks = c.backboneCodebooks
        self.semanticBegin = semanticBegin; self.semanticEnd = semanticEnd
        _codebookEmbeddings.wrappedValue = Embedding(embeddingCount: c.vocabSize * c.codebooks, dimensions: c.dim)
        _sharedLayers.wrappedValue = (0..<c.sharedLayers).map { _ in TowerLayer(c, enrichment: true) }
        _residualLayers.wrappedValue = (0..<(c.codebooks - 1)).map { _ in TowerLayer(c, enrichment: false) }
        _codebookHeads.wrappedValue = (0..<(c.codebooks - 1)).map { _ in Linear(c.dim, c.vocabSize, bias: false) }
        super.init()
    }

    func makeCache() -> [KVCache] { (0..<(sharedLayers.count + residualLayers.count)).map { _ in KVCacheSimple() } }

    func isSemantic(_ tokens: MLXArray) -> MLXArray { (tokens .>= semanticBegin) .&& (tokens .<= semanticEnd) }

    /// Sum of the embeddings of codebooks first..<end for frames [B, T, 1 + codebooks].
    func embeddingSum(_ frames: MLXArray, first: Int, end: Int) -> MLXArray {
        let offsets = MLXArray((first..<end).map { Int32($0 * vocabSize) })
        let rows = frames[.ellipsis, (1 + first)..<(1 + end)] + offsets
        return codebookEmbeddings(rows).sum(axis: -2)
    }

    /// Backbone input: semantic rows add the first residual codebooks to the token embedding and rescale; text rows keep it.
    func embedBackbone(_ frames: MLXArray, text: MLXArray) -> MLXArray {
        let count = backboneCodebooks
        let mixed = (text + embeddingSum(frames, first: 1, end: 1 + count)) * MLXArray(pow(Float(1 + count), -0.5)).asType(text.dtype)
        return MLX.where(expandedDimensions(isSemantic(frames[.ellipsis, 0]), axis: -1), mixed, text)
    }

    /// The first enrichment layer also sees the frame's codebooks that the backbone input left out.
    func enrich(_ hidden: MLXArray, c0: MLXArray, semantic: MLXArray, frames: MLXArray, caches: [KVCache]) -> MLXArray {
        var unseen = embeddingSum(frames, first: 1 + backboneCodebooks, end: codebooks)
        unseen = MLX.where(expandedDimensions(isSemantic(frames[.ellipsis, 0]), axis: -1), unseen, MLXArray.zeros(like: unseen))
        var hidden = hidden
        for (index, layer) in sharedLayers.enumerated() {
            hidden = layer(hidden, codes: c0, semantic: semantic, cache: caches[index], extra: index == 0 ? unseen : nil)
        }
        return hidden
    }

    /// Teacher-forces the caches over a prompt; row t is labelled by row t+1.
    func prefill(_ hidden: MLXArray, frames: MLXArray, nextFrames: MLXArray, caches: [KVCache]) {
        let semantic = isSemantic(nextFrames[.ellipsis, 0])
        var hidden = enrich(hidden, c0: nextFrames[.ellipsis, 1], semantic: semantic, frames: frames, caches: caches)
        let offset = sharedLayers.count
        for (index, layer) in residualLayers.enumerated() {
            hidden = layer(hidden, codes: nextFrames[.ellipsis, index + 1], semantic: semantic, cache: caches[offset + index])
        }
    }

    /// Codes c0…c9 of one frame, sampling c1…c9 with `sample(logits)`.
    func residualCodes(
        _ hidden: MLXArray, c0: MLXArray, semantic: MLXArray, frame: MLXArray, caches: [KVCache], sample: (MLXArray) -> MLXArray
    ) -> [MLXArray] {
        var hidden = enrich(hidden, c0: c0, semantic: semantic, frames: frame, caches: caches)
        var codes = [c0]
        let offset = sharedLayers.count
        for (index, (layer, head)) in zip(residualLayers, codebookHeads).enumerated() {
            hidden = layer(hidden, codes: codes[codes.count - 1], semantic: semantic, cache: caches[offset + index])
            codes.append(sample(head(hidden)))
        }
        return codes
    }
}
