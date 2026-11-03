// SPDX-License-Identifier: MIT
// Streaming Sortformer 4spk v2 (NVIDIA NeMo SortformerEncLabelModel) on MLX. Module structure adapted from
// Blaizzy/mlx-audio-swift Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift (MIT); checked against NeMo 3.0's
// ConformerEncoder (dw_striding, rel_pos, xscaling), TransformerEncoder (post-LN, no positional embedding) and
// SortformerModules. Module keys are NeMo's own checkpoint names, so the Get conversion only transposes convolutions.
import Foundation
import MLX
import MLXNN

struct SortformerV2Dimensions: Equatable {
    var melBins = 128
    var conformerLayers = 17
    var conformerWidth = 512
    var conformerHeads = 8
    var conformerKernel = 9
    var subsamplingChannels = 256
    var transformerLayers = 18
    var transformerWidth = 192
    var transformerHeads = 8
    var transformerInner = 768
    var speakers = 4

    init() {}
    init(_ config: [String: Any]) {
        func int(_ key: String, _ fallback: Int) -> Int { (config[key] as? NSNumber)?.intValue ?? fallback }
        melBins = int("mel_bins", 128); conformerLayers = int("conformer_layers", 17); conformerWidth = int("conformer_width", 512)
        conformerHeads = int("conformer_heads", 8); conformerKernel = int("conformer_kernel", 9); subsamplingChannels = int("subsampling_channels", 256)
        transformerLayers = int("transformer_layers", 18); transformerWidth = int("transformer_width", 192)
        transformerHeads = int("transformer_heads", 8); transformerInner = int("transformer_inner", 768); speakers = int("speakers", 4)
    }
}

/// NeMo `ConvSubsampling` (dw_striding, factor 8): conv.0 → ReLU → (dw conv.2, pw conv.3) → ReLU → (dw conv.5, pw conv.6) → ReLU → out.
final class SortformerPreEncode: Module {
    @ModuleInfo var conv: [Module]
    @ModuleInfo var out: Linear

    init(_ d: SortformerV2Dimensions) {
        let c = d.subsamplingChannels
        func strided(_ input: Int, groups: Int) -> Conv2d {
            Conv2d(inputChannels: input, outputChannels: c, kernelSize: 3, stride: 2, padding: 1, groups: groups)
        }
        _conv.wrappedValue = [
            strided(1, groups: 1), ReLU(), strided(c, groups: c), Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1), ReLU(),
            strided(c, groups: c), Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1), ReLU()
        ]
        _out.wrappedValue = Linear(c * ((d.melBins + 7) / 8), d.conformerWidth)
        super.init()
    }

    /// features [B, T, mel] → embeddings [B, ceil(T/8), width] (each stride-2 stage gives floor((L - 1) / 2) + 1).
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var h = features.expandedDimensions(axis: -1) // NHWC: [B, T, mel, 1]
        for layer in conv {
            if let c = layer as? Conv2d { h = c(h) } else { h = relu(h) }
        }
        let (b, t, f, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        return out(h.transposed(0, 1, 3, 2).reshaped(b, t, c * f))
    }
}

/// NeMo `RelPositionMultiHeadAttention` with untied biases (pos_bias_u / pos_bias_v per layer).
final class SortformerRelAttention: Module {
    let heads: Int
    let headDim: Int
    @ModuleInfo(key: "linear_q") var q: Linear
    @ModuleInfo(key: "linear_k") var k: Linear
    @ModuleInfo(key: "linear_v") var v: Linear
    @ModuleInfo(key: "linear_out") var o: Linear
    @ModuleInfo(key: "linear_pos") var pos: Linear
    @ParameterInfo(key: "pos_bias_u") var biasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var biasV: MLXArray

    init(width: Int, heads: Int) {
        self.heads = heads
        headDim = width / heads
        _q.wrappedValue = Linear(width, width)
        _k.wrappedValue = Linear(width, width)
        _v.wrappedValue = Linear(width, width)
        _o.wrappedValue = Linear(width, width)
        _pos.wrappedValue = Linear(width, width, bias: false)
        _biasU.wrappedValue = MLXArray.zeros([heads, width / heads])
        _biasV.wrappedValue = MLXArray.zeros([heads, width / heads])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let query = q(x).reshaped(b, t, heads, headDim)
        let key = k(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let value = v(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let p = pos(positions).reshaped(1, -1, heads, headDim).transposed(0, 2, 1, 3)
        let withU = (query + biasU).transposed(0, 2, 1, 3)
        let withV = (query + biasV).transposed(0, 2, 1, 3)
        let ac = matmul(withU, key.transposed(0, 1, 3, 2))
        var bd = matmul(withV, p.transposed(0, 1, 3, 2)) // [B, H, T, 2T - 1]
        // NeMo rel_shift: pad one column left, view as [.., 2T, T], drop the first row, view back.
        let positionsCount = bd.dim(3)
        bd = padded(bd, widths: [0, 0, 0, .init((1, 0))]).reshaped(b, heads, positionsCount + 1, t)
        bd = bd[0..., 0..., 1..., 0...].reshaped(b, heads, t, positionsCount)[0..., 0..., 0..., 0..<t]
        let scores = (ac + bd) * (1 / Float(headDim).squareRoot())
        let attention = softmax(scores.asType(.float32), axis: -1).asType(value.dtype)
        return o(matmul(attention, value).transposed(0, 2, 1, 3).reshaped(b, t, heads * headDim))
    }
}

final class SortformerConformerFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    init(width: Int) {
        _linear1.wrappedValue = Linear(width, width * 4)
        _linear2.wrappedValue = Linear(width * 4, width)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// NeMo `ConformerConvolution`: pointwise → GLU → depthwise (same padding) → BatchNorm (running stats) → Swish → pointwise.
final class SortformerConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "batch_norm") var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d

    init(width: Int, kernel: Int) {
        _pointwise1.wrappedValue = Conv1d(inputChannels: width, outputChannels: width * 2, kernelSize: 1)
        _depthwise.wrappedValue = Conv1d(inputChannels: width, outputChannels: width, kernelSize: kernel, padding: (kernel - 1) / 2, groups: width)
        _norm.wrappedValue = BatchNorm(featureCount: width, eps: 1e-5)
        _pointwise2.wrappedValue = Conv1d(inputChannels: width, outputChannels: width, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(pointwise1(x), parts: 2, axis: -1)
        return pointwise2(silu(norm(depthwise(parts[0] * sigmoid(parts[1])))))
    }
}

final class SortformerConformerLayer: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFF1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var ff1: SortformerConformerFeedForward
    @ModuleInfo(key: "norm_self_att") var normAttention: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: SortformerRelAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo var conv: SortformerConformerConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFF2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var ff2: SortformerConformerFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(_ d: SortformerV2Dimensions) {
        let w = d.conformerWidth
        _normFF1.wrappedValue = LayerNorm(dimensions: w, eps: 1e-5)
        _ff1.wrappedValue = SortformerConformerFeedForward(width: w)
        _normAttention.wrappedValue = LayerNorm(dimensions: w, eps: 1e-5)
        _attention.wrappedValue = SortformerRelAttention(width: w, heads: d.conformerHeads)
        _normConv.wrappedValue = LayerNorm(dimensions: w, eps: 1e-5)
        _conv.wrappedValue = SortformerConformerConvolution(width: w, kernel: d.conformerKernel)
        _normFF2.wrappedValue = LayerNorm(dimensions: w, eps: 1e-5)
        _ff2.wrappedValue = SortformerConformerFeedForward(width: w)
        _normOut.wrappedValue = LayerNorm(dimensions: w, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        // New bindings rather than `+=`, which would mutate the MLXArray shared with the caller.
        let afterFF1 = x + ff1(normFF1(x)) * 0.5
        let afterAttention = afterFF1 + attention(normAttention(afterFF1), positions: positions)
        let afterConv = afterAttention + conv(normConv(afterAttention))
        return normOut(afterConv + ff2(normFF2(afterConv)) * 0.5)
    }
}

final class SortformerConformerEncoder: Module {
    let width: Int
    @ModuleInfo(key: "pre_encode") var preEncode: SortformerPreEncode
    @ModuleInfo var layers: [SortformerConformerLayer]

    init(_ d: SortformerV2Dimensions) {
        width = d.conformerWidth
        _preEncode.wrappedValue = SortformerPreEncode(d)
        _layers.wrappedValue = (0..<d.conformerLayers).map { _ in SortformerConformerLayer(d) }
        super.init()
    }

    /// NeMo `RelPositionalEncoding` for length T: positions T-1 … -(T-1), interleaved sin/cos (computed in float32).
    static func relativePositions(length: Int, width: Int) -> MLXArray {
        let positions = MLXArray(stride(from: length - 1, through: -(length - 1), by: -1).map { Float($0) })
        let divisors = MLXArray(stride(from: 0, to: width, by: 2).map { Float(exp(Double($0) * -log(10_000.0) / Double(width))) })
        let angles = positions.expandedDimensions(axis: 1) * divisors.expandedDimensions(axis: 0)
        return stacked([sin(angles), cos(angles)], axis: -1).reshaped(1, 2 * length - 1, width)
    }

    /// Pre-encoded embeddings [B, T, width] (bypass_pre_encode) → conformer output [B, T, width]; xscale = sqrt(width).
    func encode(_ embeddings: MLXArray) -> MLXArray {
        var x = embeddings * Float(width).squareRoot()
        let positions = Self.relativePositions(length: x.dim(1), width: width).asType(x.dtype)
        for layer in layers { x = layer(x, positions: positions) }
        return x
    }
}

/// NeMo `MultiHeadAttention` (query_net/key_net/value_net/out_projection); its 1/sqrt(sqrt(d)) on q and k is the usual 1/sqrt(d).
final class SortformerTransformerAttention: Module {
    let heads: Int
    @ModuleInfo(key: "query_net") var q: Linear
    @ModuleInfo(key: "key_net") var k: Linear
    @ModuleInfo(key: "value_net") var v: Linear
    @ModuleInfo(key: "out_projection") var o: Linear
    init(width: Int, heads: Int) {
        self.heads = heads
        _q.wrappedValue = Linear(width, width)
        _k.wrappedValue = Linear(width, width)
        _v.wrappedValue = Linear(width, width)
        _o.wrappedValue = Linear(width, width)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, w) = (x.dim(0), x.dim(1), x.dim(2))
        let d = w / heads
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(b, t, heads, d).transposed(0, 2, 1, 3) }
        let out = MLXFast.scaledDotProductAttention(queries: split(q(x)), keys: split(k(x)), values: split(v(x)), scale: 1 / Float(d).squareRoot(), mask: nil)
        return o(out.transposed(0, 2, 1, 3).reshaped(b, t, w))
    }
}

final class SortformerTransformerFeedForward: Module {
    @ModuleInfo(key: "dense_in") var denseIn: Linear
    @ModuleInfo(key: "dense_out") var denseOut: Linear
    init(width: Int, inner: Int) {
        _denseIn.wrappedValue = Linear(width, inner)
        _denseOut.wrappedValue = Linear(inner, width)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { denseOut(relu(denseIn(x))) }
}

/// NeMo `TransformerEncoderBlock` with pre_ln = false: attention → add → layer_norm_1 → FFN → add → layer_norm_2.
final class SortformerTransformerLayer: Module {
    @ModuleInfo(key: "first_sub_layer") var attention: SortformerTransformerAttention
    @ModuleInfo(key: "layer_norm_1") var norm1: LayerNorm
    @ModuleInfo(key: "second_sub_layer") var feedForward: SortformerTransformerFeedForward
    @ModuleInfo(key: "layer_norm_2") var norm2: LayerNorm
    init(_ d: SortformerV2Dimensions) {
        _attention.wrappedValue = SortformerTransformerAttention(width: d.transformerWidth, heads: d.transformerHeads)
        _norm1.wrappedValue = LayerNorm(dimensions: d.transformerWidth, eps: 1e-5)
        _feedForward.wrappedValue = SortformerTransformerFeedForward(width: d.transformerWidth, inner: d.transformerInner)
        _norm2.wrappedValue = LayerNorm(dimensions: d.transformerWidth, eps: 1e-5)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm1(x + attention(x))
        return norm2(h + feedForward(h))
    }
}

final class SortformerTransformerEncoder: Module {
    @ModuleInfo var layers: [SortformerTransformerLayer]
    init(_ d: SortformerV2Dimensions) {
        _layers.wrappedValue = (0..<d.transformerLayers).map { _ in SortformerTransformerLayer(d) }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layers.reduce(x) { $1($0) } }
}

final class SortformerHead: Module {
    @ModuleInfo(key: "encoder_proj") var encoderProjection: Linear
    @ModuleInfo(key: "first_hidden_to_hidden") var hidden: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var speakers: Linear
    /// Trained but unused at inference (NeMo's `hidden_to_spks`); loaded so every checkpoint tensor has an owner.
    @ModuleInfo(key: "hidden_to_spks") var pairSpeakers: Linear
    init(_ d: SortformerV2Dimensions) {
        _encoderProjection.wrappedValue = Linear(d.conformerWidth, d.transformerWidth)
        _hidden.wrappedValue = Linear(d.transformerWidth, d.transformerWidth)
        _speakers.wrappedValue = Linear(d.transformerWidth, d.speakers)
        _pairSpeakers.wrappedValue = Linear(2 * d.transformerWidth, d.speakers)
        super.init()
    }
    /// NeMo `forward_speaker_sigmoids` (dropout inactive); sigmoid in float32.
    func probabilities(_ x: MLXArray) -> MLXArray {
        sigmoid(speakers(relu(hidden(relu(x)))).asType(.float32))
    }
}

final class SortformerV2Network: Module {
    let dimensions: SortformerV2Dimensions
    @ModuleInfo var encoder: SortformerConformerEncoder
    @ModuleInfo(key: "transformer_encoder") var transformer: SortformerTransformerEncoder
    @ModuleInfo(key: "sortformer_modules") var head: SortformerHead

    init(_ d: SortformerV2Dimensions) {
        dimensions = d
        _encoder.wrappedValue = SortformerConformerEncoder(d)
        _transformer.wrappedValue = SortformerTransformerEncoder(d)
        _head.wrappedValue = SortformerHead(d)
        super.init()
    }

    var dtype: DType { head.encoderProjection.weight.dtype }

    /// Mel features [B, T, mel] (float32) → pre-encoded embeddings [B, ceil(T/8), 512].
    func preEncode(_ features: MLXArray) -> MLXArray { encoder.preEncode(features.asType(dtype)) }

    /// NeMo `frontend_encoder(bypass_pre_encode=True)`: conformer + encoder_proj → [B, T, 192].
    func frontEnd(_ embeddings: MLXArray) -> MLXArray { head.encoderProjection(encoder.encode(embeddings.asType(dtype))) }

    /// NeMo `forward_infer` for a fully valid sequence: transformer + speaker sigmoids → [B, T, speakers] float32.
    func infer(_ projected: MLXArray) -> MLXArray { head.probabilities(transformer(projected)) }
}
