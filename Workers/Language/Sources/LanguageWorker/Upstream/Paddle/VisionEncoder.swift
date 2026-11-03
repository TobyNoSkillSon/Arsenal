// SPDX-License-Identifier: MIT
// PaddleOCR-VL 1.6: PaddlePaddle's pinned SigLIP/NaViT reference (see PORTED-FROM.md).
import Foundation
import MLX
import MLXNN

final class PaddleVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int
    init(_ config: PaddleOCRVLVisionConfig) {
        heads = config.numAttentionHeads
        _query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _key.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _value.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _output.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let count = x.dim(1), dim = x.dim(2) / heads
        func project(_ layer: Linear, rotate: Bool) -> MLXArray {
            let values = layer(x).reshaped(1, count, heads, dim).transposed(0, 2, 1, 3)
            return rotate ? PaddlePositions.rotate(values, cos: cos, sin: sin) : values
        }
        let result = MLXFast.scaledDotProductAttention(
            queries: project(query, rotate: true), keys: project(key, rotate: true), values: project(value, rotate: false), scale: pow(Float(dim), -0.5), mask: .none)
        return output(result.transposed(0, 2, 1, 3).reshaped(1, count, -1))
    }
}
final class PaddleVisionMLP: Module {
    @ModuleInfo(key: "fc1") var first: Linear
    @ModuleInfo(key: "fc2") var second: Linear
    init(_ config: PaddleOCRVLVisionConfig) {
        _first.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _second.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { second(geluApproximate(first(x))) }
}
final class PaddleVisionLayer: Module {
    @ModuleInfo(key: "layer_norm1") var first: LayerNorm
    @ModuleInfo(key: "layer_norm2") var second: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: PaddleVisionAttention
    var mlp: PaddleVisionMLP
    init(_ config: PaddleOCRVLVisionConfig) {
        _first.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _second.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _attention.wrappedValue = PaddleVisionAttention(config)
        mlp = PaddleVisionMLP(config)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attention(first(x), cos: cos, sin: sin)
        return h + mlp(second(h))
    }
}
final class PaddleVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patch: Conv2d
    @ModuleInfo(key: "position_embedding") var position: Embedding
    @ModuleInfo(key: "packing_position_embedding") var packing: Embedding
    let side: Int
    init(_ config: PaddleOCRVLVisionConfig) {
        side = config.imageSize / config.patchSize
        _patch.wrappedValue = Conv2d(
            inputChannels: config.numChannels, outputChannels: config.hiddenSize, kernelSize: IntOrPair(config.patchSize), stride: IntOrPair(config.patchSize))
        _position.wrappedValue = Embedding(embeddingCount: side * side, dimensions: config.hiddenSize)
        _packing.wrappedValue = Embedding(embeddingCount: 32768, dimensions: config.hiddenSize)
    }
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let features = patch(pixels), height = features.dim(1), width = features.dim(2)
        // Exact target sizes; half-pixel centers and edge clamps match interpolate(align_corners=False).
        func axis(_ count: Int) -> (MLXArray, MLXArray, MLXArray) {
            let locations = (0..<count).map { max(0, min(Float(side - 1), (Float($0) + 0.5) * Float(side) / Float(count) - 0.5)) }
            let low = locations.map { Int32(floor($0)) }, high = low.map { min(Int32(side - 1), $0 + 1) }
            return (MLXArray(low), MLXArray(high), MLXArray(zip(locations, low).map { $0 - Float($1) }).asType(features.dtype))
        }
        let (yl, yh, yw) = axis(height), (xl, xh, xw) = axis(width)
        let table = position.weight.reshaped(side, side, -1)
        let rows = table[yl] * (1 - yw[0..., .newAxis, .newAxis]) + table[yh] * yw[0..., .newAxis, .newAxis]
        let resized = rows[0..., xl] * (1 - xw[.newAxis, 0..., .newAxis]) + rows[0..., xh] * xw[.newAxis, 0..., .newAxis]
        return (features + resized).reshaped(1, height * width, -1)
    }
}
final class PaddleVisionEncoder: Module {
    var layers: [PaddleVisionLayer]
    init(_ config: PaddleOCRVLVisionConfig) { layers = (0..<config.numHiddenLayers).map { _ in PaddleVisionLayer(config) } }
}
// The checkpoint includes the contrastive pooling head, although OCR uses last_hidden_state.
// Retain its complete learned inventory rather than silently discarding it.
final class PaddlePoolAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var weight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var bias: MLXArray
    @ModuleInfo(key: "out_proj") var output: Linear
    init(_ size: Int) {
        _weight.wrappedValue = MLX.zeros([3 * size, size]); _bias.wrappedValue = MLX.zeros([3 * size])
        _output.wrappedValue = Linear(size, size)
    }
}
final class PaddlePoolHead: Module {
    var probe: MLXArray
    var attention: PaddlePoolAttention
    var layernorm: LayerNorm
    var mlp: PaddleVisionMLP
    init(_ config: PaddleOCRVLVisionConfig) {
        probe = MLX.zeros([1, 1, config.hiddenSize]); attention = PaddlePoolAttention(config.hiddenSize)
        layernorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps); mlp = PaddleVisionMLP(config)
    }
}
final class PaddleVisionTransformer: Module {
    var embeddings: PaddleVisionEmbeddings
    var encoder: PaddleVisionEncoder
    @ModuleInfo(key: "post_layernorm") var norm: LayerNorm
    var head: PaddlePoolHead
    let heads: Int
    init(_ config: PaddleOCRVLVisionConfig) {
        embeddings = PaddleVisionEmbeddings(config); encoder = PaddleVisionEncoder(config)
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        head = PaddlePoolHead(config); heads = config.numAttentionHeads
    }
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let height = pixels.dim(1) / embeddings.patch.weight.dim(1), width = pixels.dim(2) / embeddings.patch.weight.dim(2)
        var h = embeddings(pixels)
        let dim = h.dim(2) / heads
        let angles = (0..<(height * width)).flatMap { index -> [Float] in
            let half = [index / width, index % width].flatMap { position in
                (0..<(dim / 4)).map { Float(position) / pow(10000, Float(2 * $0) / Float(dim / 2)) }
            }
            return half + half
        }
        let angle = MLXArray(angles).reshaped(1, 1, height * width, dim)
        let cosine = cos(angle).asType(h.dtype), sine = sin(angle).asType(h.dtype)
        for layer in encoder.layers { h = layer(h, cos: cosine, sin: sine) }
        return norm(h).reshaped(1, height, width, -1)
    }
}
final class NaViTVisionEncoder: Module {
    @ModuleInfo(key: "vision_model") var model: PaddleVisionTransformer
    init(config: PaddleOCRVLVisionConfig) { _model.wrappedValue = PaddleVisionTransformer(config) }
    func getImageFeatures(_ pixels: MLXArray) -> MLXArray { model(pixels) }
}
