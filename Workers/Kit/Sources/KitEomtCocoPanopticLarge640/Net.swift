// SPDX-License-Identifier: MIT
// EoMT (Kerssies et al., CVPR 2025; tue-mps/eomt, MIT) ported from transformers models/eomt/modeling_eomt.py:
// a plain ViT-L/16 whose last `num_blocks` layers also carry the learned queries, then a mask head and an upscaler.
// The checkpoint's attn_mask_probs are all 0, so inference uses no masked attention (the intermediate predictions
// that feed the mask are skipped, as the maker's forward does). Property keys follow the checkpoint. Channels-last.
import Foundation
import MLX
import MLXFast
import MLXNN

struct EomtConfig {
    let hidden: Int, layers: Int, heads: Int, patch: Int, imageSize: Int
    let queries: Int, registers: Int, blocks: Int, upscales: Int, eps: Float, labels: [String]

    init(_ c: [String: Any]) throws {
        func int(_ key: String) throws -> Int {
            guard let v = c[key] as? NSNumber else { throw RunnerError.failed("config.json lacks \(key).") }
            return v.intValue
        }
        hidden = try int("hidden_size"); layers = try int("num_hidden_layers"); heads = try int("num_attention_heads")
        patch = try int("patch_size"); imageSize = try int("image_size"); queries = try int("num_queries")
        registers = try int("num_register_tokens"); blocks = try int("num_blocks"); upscales = try int("num_upscale_blocks")
        eps = (c["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
        guard c["use_swiglu_ffn"] as? Bool != true, (c["mlp_ratio"] as? NSNumber)?.intValue ?? 4 == 4 else {
            throw RunnerError.failed("This runner expects the GELU MLP with ratio 4.")
        }
        guard let id2label = c["id2label"] as? [String: String] else { throw RunnerError.failed("config.json lacks id2label.") }
        labels = (0..<id2label.count).map { id2label[String($0)] ?? "class \($0)" }
    }
}

final class PatchEmbeddings: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d
    init(_ c: EomtConfig) {
        _projection.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: c.hidden, kernelSize: IntOrPair(c.patch), stride: IntOrPair(c.patch))
    }
}

final class Embeddings: Module {
    @ParameterInfo(key: "cls_token") var cls: MLXArray
    @ParameterInfo(key: "register_tokens") var registers: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patches: PatchEmbeddings
    @ModuleInfo(key: "position_embeddings") var positions: Embedding

    init(_ c: EomtConfig) {
        _cls.wrappedValue = MLXArray.zeros([1, 1, c.hidden])
        _registers.wrappedValue = MLXArray.zeros([1, c.registers, c.hidden])
        _patches.wrappedValue = PatchEmbeddings(c)
        let grid = c.imageSize / c.patch
        _positions.wrappedValue = Embedding(embeddingCount: grid * grid, dimensions: c.hidden)
    }

    /// [1, 640, 640, 3] to [1, 1 + registers + 1600, C].
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let x = patches.projection(pixels)
        let tokens = x.reshaped([1, x.dim(1) * x.dim(2), x.dim(3)]) + positions.weight.expandedDimensions(axis: 0)
        return concatenated([cls, registers, tokens], axis: 1)
    }
}

final class Attention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    let heads: Int

    init(_ c: EomtConfig) {
        heads = c.heads
        _q.wrappedValue = Linear(c.hidden, c.hidden)
        _k.wrappedValue = Linear(c.hidden, c.hidden)
        _v.wrappedValue = Linear(c.hidden, c.hidden)
        _out.wrappedValue = Linear(c.hidden, c.hidden)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), c = x.dim(2), hd = c / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([b, n, heads, hd]).transposed(0, 2, 1, 3) }
        let o = MLXFast.scaledDotProductAttention(
            queries: split(q(x)), keys: split(k(x)), values: split(v(x)), scale: 1 / Float(hd).squareRoot(), mask: .none)
        return out(o.transposed(0, 2, 1, 3).reshaped([b, n, c]))
    }
}

final class LayerScale: Module {
    @ParameterInfo(key: "lambda1") var lambda: MLXArray
    init(_ dim: Int) { _lambda.wrappedValue = MLXArray.ones([dim]) }
}

final class Mlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(_ dim: Int) {
        _fc1.wrappedValue = Linear(dim, dim * 4)
        _fc2.wrappedValue = Linear(dim * 4, dim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class Layer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: Attention
    @ModuleInfo(key: "layer_scale1") var scale1: LayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: Mlp
    @ModuleInfo(key: "layer_scale2") var scale2: LayerScale

    init(_ c: EomtConfig) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hidden, eps: c.eps)
        _attention.wrappedValue = Attention(c)
        _scale1.wrappedValue = LayerScale(c.hidden)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hidden, eps: c.eps)
        _mlp.wrappedValue = Mlp(c.hidden)
        _scale2.wrappedValue = LayerScale(c.hidden)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x)) * scale1.lambda
        return h + mlp(norm2(h)) * scale2.lambda
    }
}

/// Channel-last LayerNorm2d (eps 1e-6).
final class Norm2d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(_ dim: Int) {
        _weight.wrappedValue = MLXArray.ones([dim])
        _bias.wrappedValue = MLXArray.zeros([dim])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: 1e-6)
    }
}

final class ScaleLayer: Module {
    @ModuleInfo(key: "conv1") var conv1: ConvTransposed2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "layernorm2d") var norm: Norm2d

    init(_ dim: Int) {
        _conv1.wrappedValue = ConvTransposed2d(inputChannels: dim, outputChannels: dim, kernelSize: IntOrPair(2), stride: IntOrPair(2))
        _conv2.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim, kernelSize: IntOrPair(3), padding: IntOrPair(1), groups: dim, bias: false)
        _norm.wrappedValue = Norm2d(dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(conv2(gelu(conv1(x)))) }
}

final class ScaleBlock: Module {
    @ModuleInfo(key: "block") var block: [ScaleLayer]
    init(_ c: EomtConfig) { _block.wrappedValue = (0..<c.upscales).map { _ in ScaleLayer(c.hidden) } }
}

final class MaskHead: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "fc3") var fc3: Linear
    init(_ dim: Int) {
        _fc1.wrappedValue = Linear(dim, dim)
        _fc2.wrappedValue = Linear(dim, dim)
        _fc3.wrappedValue = Linear(dim, dim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc3(gelu(fc2(gelu(fc1(x))))) }
}

final class Eomt: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embeddings
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm
    @ModuleInfo(key: "query") var query: Embedding
    @ModuleInfo(key: "layers") var layers: [Layer]
    @ModuleInfo(key: "upscale_block") var upscale: ScaleBlock
    @ModuleInfo(key: "mask_head") var maskHead: MaskHead
    @ModuleInfo(key: "class_predictor") var classPredictor: Linear
    let c: EomtConfig

    init(_ c: EomtConfig) {
        self.c = c
        _embeddings.wrappedValue = Embeddings(c)
        _layernorm.wrappedValue = LayerNorm(dimensions: c.hidden, eps: c.eps)
        _query.wrappedValue = Embedding(embeddingCount: c.queries, dimensions: c.hidden)
        _layers.wrappedValue = (0..<c.layers).map { _ in Layer(c) }
        _upscale.wrappedValue = ScaleBlock(c)
        _maskHead.wrappedValue = MaskHead(c.hidden)
        _classPredictor.wrappedValue = Linear(c.hidden, c.labels.count + 1)
    }

    /// Checkpoint keys this runner uses, with PyTorch conv weights turned channels-last.
    static func prepare(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights where key != "attn_mask_probs" && !key.hasPrefix("criterion.") {
            if value.ndim == 4 {
                out[key] = key.hasSuffix("conv1.weight") ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)
            } else {
                out[key] = value
            }
        }
        return out
    }

    /// Mask logits [Q, 160, 160] and class logits [Q, labels + 1] for one normalized, padded image [1, 640, 640, 3].
    func callAsFunction(_ pixels: MLXArray, cancelled: () throws -> Void) rethrows -> (masks: MLXArray, classes: MLXArray) {
        var x = embeddings(pixels)
        for (i, layer) in layers.enumerated() {
            if i == c.layers - c.blocks {
                x = concatenated([query.weight.expandedDimensions(axis: 0), x], axis: 1)
            }
            x = layer(x)
            if i % 6 == 5 { eval(x); try cancelled() }
        }
        x = layernorm(x)
        let q = x[0..., 0 ..< c.queries, 0...]
        let classes = classPredictor(q)
        let grid = c.imageSize / c.patch
        var p = x[0..., (c.queries + 1 + c.registers)..., 0...].reshaped([1, grid, grid, c.hidden])
        for layer in upscale.block { p = layer(p) }
        let h = p.dim(1), w = p.dim(2)
        let masks = maskHead(q).matmul(p.reshaped([1, h * w, c.hidden]).transposed(0, 2, 1)).reshaped([c.queries, h, w])
        return (masks, classes[0])
    }
}
