// SPDX-License-Identifier: MIT
// Grounding DINO's two backbones, named after the checkpoint: a Swin-B image backbone (transformers SwinBackbone with
// always_partition=True: window 12, shift 6 on odd blocks, features before downsampling of stages 2-4, each LayerNorm'd)
// and a BERT-base text encoder that takes Grounding DINO's per-phrase attention mask and position ids.
import Foundation
import MLX
import MLXNN

// MARK: - Swin

let swinWindow = 12

final class SwinSelfAttention: Module {
    var query: Linear, key: Linear, value: Linear
    var relative_position_bias_table: MLXArray
    init(_ dim: Int, heads: Int) {
        query = Linear(dim, dim); key = Linear(dim, dim); value = Linear(dim, dim)
        relative_position_bias_table = MLXArray.zeros([(2 * swinWindow - 1) * (2 * swinWindow - 1), heads])
        super.init()
    }
}

final class SwinSelfOutput: Module { var dense: Linear; init(_ dim: Int) { dense = Linear(dim, dim); super.init() } }

final class SwinAttentionBlock: Module {
    @ModuleInfo(key: "self") var selfAttention: SwinSelfAttention
    var output: SwinSelfOutput
    init(_ dim: Int, heads: Int) {
        _selfAttention.wrappedValue = SwinSelfAttention(dim, heads: heads)
        output = SwinSelfOutput(dim)
        super.init()
    }
}

final class DenseOnly: Module { var dense: Linear; init(_ i: Int, _ o: Int) { dense = Linear(i, o); super.init() } }

/// Relative position index of a 12×12 window (SwinRelativePositionBias._create_relative_position_index), [144 * 144].
let swinRelativeIndex: MLXArray = {
    let w = swinWindow
    var index = [Int32]()
    for i in 0..<(w * w) {
        for j in 0..<(w * w) {
            let dh = i / w - j / w + w - 1, dw = i % w - j % w + w - 1
            index.append(Int32(dh * (2 * w - 1) + dw))
        }
    }
    return MLXArray(index)
}()

final class SwinLayer: Module {
    var attention: SwinAttentionBlock
    var layernorm_before: LayerNorm, layernorm_after: LayerNorm
    var intermediate: DenseOnly, output: DenseOnly
    let heads: Int, shift: Int

    init(_ dim: Int, heads: Int, shift: Int) {
        attention = SwinAttentionBlock(dim, heads: heads)
        layernorm_before = LayerNorm(dimensions: dim, eps: 1e-5); layernorm_after = LayerNorm(dimensions: dim, eps: 1e-5)
        intermediate = DenseOnly(dim, 4 * dim); output = DenseOnly(4 * dim, dim)
        self.heads = heads; self.shift = shift
        super.init()
    }

    /// Cyclic-shift mask for a padded map (get_attn_mask): [windows, 144, 144], 0 or -100.
    static func shiftMask(height: Int, width: Int, shift: Int) -> MLXArray {
        let w = swinWindow
        func region(_ i: Int, _ n: Int) -> Int { (i >= n - w ? 1 : 0) + (i >= n - shift ? 1 : 0) }
        let rows = height / w, cols = width / w
        var labels = [Int](repeating: 0, count: rows * cols * w * w)
        for wr in 0..<rows {
            for wc in 0..<cols {
                for i in 0..<w {
                    for j in 0..<w {
                        labels[((wr * cols + wc) * w + i) * w + j] = region(wr * w + i, height) * 3 + region(wc * w + j, width)
                    }
                }
            }
        }
        var mask = [Float](repeating: 0, count: rows * cols * w * w * w * w)
        let area = w * w
        for win in 0..<(rows * cols) {
            for a in 0..<area {
                for b in 0..<area where labels[win * area + a] != labels[win * area + b] {
                    mask[(win * area + a) * area + b] = -100
                }
            }
        }
        return MLXArray(mask, [rows * cols, area, area])
    }

    /// x: [H, W, C] → [H, W, C].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x.dim(0), w = x.dim(1), c = x.dim(2), ws = swinWindow
        var hidden = layernorm_before(x)
        let padH = (ws - h % ws) % ws, padW = (ws - w % ws) % ws
        if padW > 0 { hidden = concatenated([hidden, MLXArray.zeros([h, padW, c], dtype: hidden.dtype)], axis: 1) }
        if padH > 0 { hidden = concatenated([hidden, MLXArray.zeros([padH, w + padW, c], dtype: hidden.dtype)], axis: 0) }
        let hp = h + padH, wp = w + padW
        if shift > 0 { hidden = roll(roll(hidden, shift: -shift, axis: 0), shift: -shift, axis: 1) }
        let rows = hp / ws, cols = wp / ws, windows = rows * cols
        let tokens = hidden.reshaped([rows, ws, cols, ws, c]).transposed(0, 2, 1, 3, 4).reshaped([windows, ws * ws, c])

        let a = attention.selfAttention
        let headDim = c / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([windows, ws * ws, heads, headDim]).transposed(0, 2, 1, 3) }
        var bias = take(a.relative_position_bias_table, swinRelativeIndex, axis: 0)
            .reshaped([ws * ws, ws * ws, heads]).transposed(2, 0, 1).expandedDimensions(axis: 0)  // [1, heads, 144, 144]
        if shift > 0 {
            bias = bias + SwinLayer.shiftMask(height: hp, width: wp, shift: shift).expandedDimensions(axis: 1).asType(bias.dtype)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(a.query(tokens)), keys: split(a.key(tokens)), values: split(a.value(tokens)),
            scale: pow(Float(headDim), -0.5), mask: .array(bias.asType(tokens.dtype)))
        var out = attention.output.dense(attended.transposed(0, 2, 1, 3).reshaped([windows, ws * ws, c]))
        out = out.reshaped([rows, cols, ws, ws, c]).transposed(0, 2, 1, 3, 4).reshaped([hp, wp, c])
        if shift > 0 { out = roll(roll(out, shift: shift, axis: 0), shift: shift, axis: 1) }
        if padH > 0 || padW > 0 { out = out[..<h, ..<w, 0...] }
        let residual = x + out
        return residual + output.dense(gelu(intermediate.dense(layernorm_after(residual))))
    }
}

final class PatchMerging: Module {
    var reduction: Linear, norm: LayerNorm
    init(_ dim: Int) { reduction = Linear(4 * dim, 2 * dim, bias: false); norm = LayerNorm(dimensions: 4 * dim, eps: 1e-5); super.init() }
    /// [H, W, C] → [(H+1)/2, (W+1)/2, 2C]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        let h = x.dim(0), w = x.dim(1), c = x.dim(2)
        if w % 2 == 1 { x = concatenated([x, MLXArray.zeros([h, 1, c], dtype: x.dtype)], axis: 1) }
        if h % 2 == 1 { x = concatenated([x, MLXArray.zeros([1, x.dim(1), c], dtype: x.dtype)], axis: 0) }
        let x0 = x[.stride(from: 0, by: 2), .stride(from: 0, by: 2), 0...]
        let x1 = x[.stride(from: 1, by: 2), .stride(from: 0, by: 2), 0...]
        let x2 = x[.stride(from: 0, by: 2), .stride(from: 1, by: 2), 0...]
        let x3 = x[.stride(from: 1, by: 2), .stride(from: 1, by: 2), 0...]
        return reduction(norm(concatenated([x0, x1, x2, x3], axis: -1)))
    }
}

final class SwinStage: Module {
    var blocks: [SwinLayer]
    var downsample: PatchMerging?
    init(dim: Int, depth: Int, heads: Int, downsample: Bool) {
        blocks = (0..<depth).map { SwinLayer(dim, heads: heads, shift: $0 % 2 == 0 ? 0 : swinWindow / 2) }
        self.downsample = downsample ? PatchMerging(dim) : nil
        super.init()
    }
}

final class SwinPatchEmbeddings: Module {
    var projection = Conv2d(inputChannels: 3, outputChannels: 128, kernelSize: .init(4), stride: .init(4), bias: true)
}

final class SwinEmbeddings: Module {
    var patch_embeddings = SwinPatchEmbeddings()
    var norm = LayerNorm(dimensions: 128, eps: 1e-5)
}

final class SwinEncoder: Module {
    var layers: [SwinStage]
    init(depths: [Int], heads: [Int]) {
        layers = depths.indices.map { SwinStage(dim: 128 << $0, depth: depths[$0], heads: heads[$0], downsample: $0 < depths.count - 1) }
        super.init()
    }
}

final class StageNorms: Module {
    @ModuleInfo(key: "stage2") var stage2: LayerNorm
    @ModuleInfo(key: "stage3") var stage3: LayerNorm
    @ModuleInfo(key: "stage4") var stage4: LayerNorm
    override init() {
        _stage2.wrappedValue = LayerNorm(dimensions: 256, eps: 1e-5)
        _stage3.wrappedValue = LayerNorm(dimensions: 512, eps: 1e-5)
        _stage4.wrappedValue = LayerNorm(dimensions: 1024, eps: 1e-5)
        super.init()
    }
}

final class SwinBackbone: Module {
    var embeddings = SwinEmbeddings()
    var encoder = SwinEncoder(depths: [2, 2, 18, 2], heads: [4, 8, 16, 32])
    var hidden_states_norms = StageNorms()

    /// pixels [H, W, 3] (normalized, fp16) → stage 2, 3, 4 feature maps [h, w, C].
    func callAsFunction(_ pixels: MLXArray, checkpoint: () throws -> Void) rethrows -> [MLXArray] {
        var x = pixels
        let h = x.dim(0), w = x.dim(1)
        if w % 4 != 0 { x = concatenated([x, MLXArray.zeros([h, 4 - w % 4, 3], dtype: x.dtype)], axis: 1) }
        if h % 4 != 0 { x = concatenated([x, MLXArray.zeros([4 - h % 4, x.dim(1), 3], dtype: x.dtype)], axis: 0) }
        x = embeddings.norm(embeddings.patch_embeddings.projection(x.expandedDimensions(axis: 0))[0])
        var features: [MLXArray] = []
        let norms = [nil, hidden_states_norms.stage2, hidden_states_norms.stage3, hidden_states_norms.stage4]
        for (index, stage) in encoder.layers.enumerated() {
            for block in stage.blocks {
                x = block(x)
                eval(x)
                try checkpoint()
            }
            if let norm = norms[index] { features.append(norm(x)) }
            if let down = stage.downsample { x = down(x) }
        }
        return features
    }
}

// MARK: - BERT

final class BertEmbeddings: Module {
    var word_embeddings = Embedding(embeddingCount: 30522, dimensions: 768)
    var position_embeddings = Embedding(embeddingCount: 512, dimensions: 768)
    var token_type_embeddings = Embedding(embeddingCount: 2, dimensions: 768)
    @ModuleInfo(key: "LayerNorm") var layerNorm = LayerNorm(dimensions: 768, eps: 1e-12)
}

final class BertSelf: Module { var query = Linear(768, 768), key = Linear(768, 768), value = Linear(768, 768) }
final class BertOut: Module {
    var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm = LayerNorm(dimensions: 768, eps: 1e-12)
    init(_ i: Int) { dense = Linear(i, 768); super.init() }
}
final class BertAttention: Module {
    @ModuleInfo(key: "self") var selfAttention = BertSelf()
    var output = BertOut(768)
}
final class BertLayer: Module {
    var attention = BertAttention()
    var intermediate = DenseOnly(768, 3072)
    var output = BertOut(3072)
}
final class BertEncoder: Module { var layer: [BertLayer] = (0..<12).map { _ in BertLayer() } }

final class BertModel: Module {
    var embeddings = BertEmbeddings()
    var encoder = BertEncoder()

    /// ids, positions [L]; mask [L, L] additive (0 or a large negative), float32. Returns [L, 768].
    func callAsFunction(_ ids: MLXArray, positions: MLXArray, mask: MLXArray) -> MLXArray {
        var x = embeddings.word_embeddings(ids) + embeddings.position_embeddings(positions)
            + embeddings.token_type_embeddings(MLXArray.zeros(like: ids))
        x = embeddings.layerNorm(x)
        let n = ids.dim(0), heads = 12, headDim = 64
        for layer in encoder.layer {
            let s = layer.attention.selfAttention
            func split(_ t: MLXArray) -> MLXArray { t.reshaped([n, heads, headDim]).transposed(1, 0, 2) }
            let q = split(s.query(x)), k = split(s.key(x)), v = split(s.value(x))
            let scores = matmul(q, k.transposed(0, 2, 1)).asType(.float32) / sqrt(Float(headDim)) + mask
            let context = matmul(softmax(scores, axis: -1).asType(x.dtype), v).transposed(1, 0, 2).reshaped([n, 768])
            let attended = layer.attention.output.layerNorm(layer.attention.output.dense(context) + x)
            x = layer.output.layerNorm(layer.output.dense(gelu(layer.intermediate.dense(attended))) + attended)
        }
        return x
    }
}
