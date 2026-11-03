// SPDX-License-Identifier: MIT
// DiariZen's EEND segmentation network (BUTSpeechFIT/DiariZen, diarizen/models/eend/model_wavlm_conformer.py) in MLX's
// channel-last layouts: a structurally pruned WavLM Large (wav2vec2/components.py, layer-norm-first, gated relative
// position bias), a learned weighted sum of its 25 hidden states, a 4-layer Conformer (module/conformer.py) and a
// 16-class powerset classifier. Module keys equal the maker's state-dict names under `segmentation.`. Learned weights
// and activations are half; waveform normalisation and the 25-way weighted sum accumulate in Float32.
import Foundation
import IntegrationKit
import MLX
import MLXFast
import MLXNN

/// Network shape read from the checkpoint at Get (config.json); the pruned WavLM keeps different head and feed-forward
/// widths in every layer. Remaining head indices are not stored in the checkpoint: they come from the maker's
/// `wavlm_large_s80_md` configuration and Get checks their counts against the stored projections.
struct DiariZenDimensions: Codable, Equatable, Sendable {
    var convChannels: [Int]
    var convKernels = [10, 3, 3, 3, 3, 2, 2]
    var convStrides = [5, 2, 2, 2, 2, 2, 2]
    var embed = 1024
    var totalHeads = 16
    var remainingHeads: [[Int]]
    var feedForward: [Int]
    var buckets = 320
    var maxDistance = 800
    var positionKernel = 128
    var positionGroups = 16
    var conformerWidth = 256
    var conformerHeads = 4
    var conformerFeedForward = 1024
    var conformerLayers = 4
    var conformerKernel = 31
    var classes = 16
    var layers: Int { feedForward.count }
    var headDimension: Int { embed / totalHeads }

    /// diarizen/models/module/wavlm_config.py WAVLM_LARGE_S80_MD (DiariZen @844f5555).
    static let s80RemainingHeads: [[Int]] = [
        [1, 2, 4, 5, 6], [9, 10, 14], [0, 1, 2, 4, 5, 7], [1, 4, 7, 12, 13, 14], [0, 2, 3, 4, 13], [1, 7, 13, 14, 15], [11, 13, 15],
        [2, 3, 4, 8, 15], [2, 5, 6, 15], [], [0, 1], [1, 3, 5, 12], [], [4, 7, 11], [6, 9], [11], [], [], [14], [5, 15],
        [0, 2, 8, 11, 13, 15], [0, 1, 3, 4, 5, 6, 7, 10, 13], [0, 1, 3, 6, 7, 9, 10, 11, 12, 14], [1, 2, 3, 4, 7, 13, 14, 15]
    ]
    static let s80 = DiariZenDimensions(
        convChannels: [512, 153, 224, 255, 302, 368, 211], remainingHeads: s80RemainingHeads,
        feedForward: [1092, 925, 759, 646, 745, 615, 684, 958, 286, 294, 406, 377, 463, 542, 298, 236, 96, 104, 134, 211, 473, 1011, 1770, 1316])

    /// A tiny network of the same structure for random-weight lab runs.
    static func lab(layers: Int) -> DiariZenDimensions {
        var d = s80
        d.remainingHeads = Array(d.remainingHeads.prefix(layers))
        d.feedForward = Array(d.feedForward.prefix(layers))
        return d
    }

    /// Frames out of the convolutional feature extractor (multi_conv_num_frames, no padding).
    func frames(samples: Int) -> Int {
        zip(convKernels, convStrides).reduce(samples) { length, layer in max(0, (length - layer.0) / layer.1 + 1) }
    }
}

/// wav2vec2 ConvLayerBlock in "layer_norm" mode: Conv1d (no bias) → LayerNorm over channels → GELU.
final class DZConvLayer: Module {
    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    init(_ input: Int, _ output: Int, kernel: Int, stride: Int) {
        _conv.wrappedValue = Conv1d(inputChannels: input, outputChannels: output, kernelSize: kernel, stride: stride, bias: false)
        _layerNorm.wrappedValue = LayerNorm(dimensions: output, eps: 1e-5)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(layerNorm(conv(x))) }
}

final class DZFeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [DZConvLayer]
    /// The pruning run's saved soft mask of the last conv layer (all ones in this release), multiplied in as the maker does.
    var dummy_weight: MLXArray
    init(_ d: DiariZenDimensions) {
        var input = 1
        _convLayers.wrappedValue = d.convChannels.indices.map { i in
            defer { input = d.convChannels[i] }
            return DZConvLayer(input, d.convChannels[i], kernel: d.convKernels[i], stride: d.convStrides[i])
        }
        dummy_weight = MLXArray.ones([d.convChannels.last ?? 1])
    }
    /// [B, samples, 1] → [B, frames, channels]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in convLayers { x = layer(x) }
        return x * dummy_weight
    }
}

final class DZFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var projection: Linear
    init(_ input: Int, _ output: Int) {
        _layerNorm.wrappedValue = LayerNorm(dimensions: input, eps: 1e-5)
        _projection.wrappedValue = Linear(input, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(layerNorm(x)) }
}

/// ConvolutionalPositionalEmbedding: grouped Conv1d (kernel 128, padding 64; weight norm folded at Get), the last frame
/// removed (even kernel), GELU.
final class DZPositionalConv: Module {
    @ModuleInfo var conv: Conv1d
    init(_ d: DiariZenDimensions) {
        _conv.wrappedValue = Conv1d(
            inputChannels: d.embed, outputChannels: d.embed, kernelSize: d.positionKernel, padding: d.positionKernel / 2, groups: d.positionGroups)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x)
        if conv.weight.dim(1) % 2 == 0 { y = y[0..., 0..<(y.dim(1) - 1), 0...] }
        return gelu(y)
    }
}

/// WavLMSelfAttention over the remaining heads of one pruned layer. The gated relative position bias uses every one of
/// the 16 original heads (gates come from the layer-normed input split into 16 × 64) and then keeps the remaining heads.
final class DZAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    @ModuleInfo(key: "gru_rel_pos_linear") var gate: Linear
    var gru_rel_pos_const: MLXArray
    @ModuleInfo(key: "rel_attn_embed") var relativeEmbedding: Embedding?
    let heads: [Int]
    let totalHeads: Int
    let headDimension: Int
    init(_ d: DiariZenDimensions, heads: [Int], relative: Bool) {
        self.heads = heads; totalHeads = d.totalHeads; headDimension = d.headDimension
        let width = heads.count * d.headDimension
        _q.wrappedValue = Linear(d.embed, width)
        _k.wrappedValue = Linear(d.embed, width)
        _v.wrappedValue = Linear(d.embed, width)
        _out.wrappedValue = Linear(width, d.embed)
        _gate.wrappedValue = Linear(d.headDimension, 8)
        gru_rel_pos_const = MLXArray.ones([1, d.totalHeads, 1, 1])
        _relativeEmbedding.wrappedValue = relative ? Embedding(embeddingCount: d.buckets, dimensions: d.totalHeads) : nil
    }

    /// [16, L, L] relative position bias from the bucket table ([L, L] Int32).
    func bias(_ buckets: MLXArray) -> MLXArray? {
        guard let relativeEmbedding else { return nil }
        return relativeEmbedding(buckets).transposed(2, 0, 1)
    }

    /// x: [B, L, 1024] layer-normed input; bias: [16, L, L] → [B, L, 1024].
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.dim(0), length = x.dim(1), h = heads.count, hd = headDimension
        // Gate: sigmoid(Linear(64 → 8) summed in two groups of 4) per original head and position.
        let perHead = x.reshaped(b, length, totalHeads, hd).transposed(0, 2, 1, 3)
        let gates = sigmoid(gate(perHead).reshaped(b, totalHeads, length, 2, 4).sum(axis: -1))
        let gateA = gates[0..., 0..., 0..., 0..<1], gateB = gates[0..., 0..., 0..., 1..<2]
        let scale = gateA * (gateB * gru_rel_pos_const - 1) + 2
        let index = MLXArray(heads.map(Int32.init))
        let mask = take(scale * bias.expandedDimensions(axis: 0), index, axis: 1)
        let queries = q(x).reshaped(b, length, h, hd).transposed(0, 2, 1, 3)
        let keys = k(x).reshaped(b, length, h, hd).transposed(0, 2, 1, 3)
        let values = v(x).reshaped(b, length, h, hd).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1 / Float(hd).squareRoot(), mask: mask.asType(queries.dtype))
        return out(attended.transposed(0, 2, 1, 3).reshaped(b, length, h * hd))
    }

    /// WavLMSelfAttention._relative_positions_bucket(bidirectional=True) in torch's float32 arithmetic: 160 buckets per
    /// direction, exact below 80, logarithmic up to max_distance, truncated toward zero.
    static func buckets(length: Int, buckets: Int = 320, maxDistance: Int = 800) -> [Int32] {
        let half = buckets / 2, exact = half / 2
        let denominator = Float(log(Double(maxDistance) / Double(exact)))
        var table = [Int32](repeating: 0, count: length * length)
        for query in 0..<length {
            for key in 0..<length {
                let relative = key - query
                var bucket = relative > 0 ? half : 0
                let distance = abs(relative)
                if distance < exact {
                    bucket += distance
                } else {
                    let scaled = logf(Float(distance) / Float(exact)) / denominator * Float(half - exact)
                    bucket += min(exact + Int(scaled), half - 1)
                }
                table[query * length + key] = Int32(bucket)
            }
        }
        return table
    }
}

final class DZFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediate: Linear
    @ModuleInfo(key: "output_dense") var output: Linear
    init(_ embed: Int, _ hidden: Int) {
        _intermediate.wrappedValue = Linear(embed, hidden)
        _output.wrappedValue = Linear(hidden, embed)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(gelu(intermediate(x))) }
}

/// EncoderLayer, layer-norm first. A pruned attention leaves only the feed-forward half.
final class DZEncoderLayer: Module {
    @ModuleInfo var attention: DZAttention?
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: DZFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    init(_ d: DiariZenDimensions, index: Int) {
        let heads = d.remainingHeads[index]
        _attention.wrappedValue = heads.isEmpty ? nil : DZAttention(d, heads: heads, relative: index == 0)
        _layerNorm.wrappedValue = LayerNorm(dimensions: d.embed, eps: 1e-5)
        _feedForward.wrappedValue = DZFeedForward(d.embed, d.feedForward[index])
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: d.embed, eps: 1e-5)
    }
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        var x = x
        // mlx-swift's `+=` updates the array object in place, and the caller keeps this object as a hidden state:
        // a new array is required here (2 Nov: `+=` rewrote layer 0's stored state with layer 1's attention output).
        // swiftlint:disable:next shorthand_operator
        if let attention { x = x + attention(layerNorm(x), bias: bias) }
        return x + feedForward(finalLayerNorm(x))
    }
}

final class DZTransformer: Module {
    @ModuleInfo(key: "pos_conv_embed") var positional: DZPositionalConv
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var layers: [DZEncoderLayer]
    let dimensions: DiariZenDimensions
    init(_ d: DiariZenDimensions) {
        dimensions = d
        _positional.wrappedValue = DZPositionalConv(d)
        _layerNorm.wrappedValue = LayerNorm(dimensions: d.embed, eps: 1e-5)
        _layers.wrappedValue = (0..<d.layers).map { DZEncoderLayer(d, index: $0) }
    }
    /// get_intermediate_outputs: the preprocessed input, then every layer's output (layers + 1 states). The layers are
    /// layer-norm-first, so the maker builds this Transformer with `layer_norm_first = not layer_norm_first` (False):
    /// its `layer_norm` belongs to forward()'s end, which extract_features never reaches. It is loaded and unused.
    func states(_ x: MLXArray, buckets: MLXArray) throws -> [MLXArray] {
        var x = x + positional(x)
        var out = [x]
        guard let bias = layers.first?.attention?.bias(buckets) else {
            throw audioFailure("DiariZen's first WavLM layer must carry the relative position embedding.")
        }
        for layer in layers {
            x = layer(x, bias: bias)
            out.append(x)
        }
        return out
    }
}

final class DZEncoder: Module {
    @ModuleInfo(key: "feature_projection") var projection: DZFeatureProjection
    @ModuleInfo var transformer: DZTransformer
    init(_ d: DiariZenDimensions) {
        _projection.wrappedValue = DZFeatureProjection(d.convChannels.last ?? 1, d.embed)
        _transformer.wrappedValue = DZTransformer(d)
    }
}

final class DZWavLM: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: DZFeatureExtractor
    @ModuleInfo var encoder: DZEncoder
    init(_ d: DiariZenDimensions) {
        _featureExtractor.wrappedValue = DZFeatureExtractor(d)
        _encoder.wrappedValue = DZEncoder(d)
    }
}

/// Conformer PositionwiseFeedForward: LN → Linear → Swish → Linear, half-step residual.
final class DZConformerFeedForward: Module {
    @ModuleInfo(key: "ln_norm") var norm: LayerNorm
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear
    init(_ width: Int, _ hidden: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _w1.wrappedValue = Linear(width, hidden)
        _w2.wrappedValue = Linear(hidden, width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + 0.5 * w2(silu(w1(norm(x)))) }
}

final class DZConformerSelfAttention: Module {
    @ModuleInfo var linearQ: Linear
    @ModuleInfo var linearK: Linear
    @ModuleInfo var linearV: Linear
    @ModuleInfo var linearO: Linear
    let heads: Int
    init(_ width: Int, heads: Int) {
        self.heads = heads
        _linearQ.wrappedValue = Linear(width, width)
        _linearK.wrappedValue = Linear(width, width)
        _linearV.wrappedValue = Linear(width, width)
        _linearO.wrappedValue = Linear(width, width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), length = x.dim(1), width = x.dim(2), dk = width / heads
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(b, length, heads, dk).transposed(0, 2, 1, 3) }
        let y = MLXFast.scaledDotProductAttention(
            queries: split(linearQ(x)), keys: split(linearK(x)), values: split(linearV(x)), scale: 1 / Float(dk).squareRoot(), mask: nil)
        return linearO(y.transposed(0, 2, 1, 3).reshaped(b, length, width))
    }
}

/// ConformerMHA: LN → self-attention (no positional encoding: use_posi = false) → residual.
final class DZConformerAttention: Module {
    @ModuleInfo(key: "ln_norm") var norm: LayerNorm
    @ModuleInfo var mha: DZConformerSelfAttention
    init(_ width: Int, heads: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _mha.wrappedValue = DZConformerSelfAttention(width, heads: heads)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + mha(norm(x)) }
}

/// ConvolutionModule: LN → pointwise Conv1d (2×) → GLU → depthwise Conv1d (31, same padding) → BatchNorm → Swish →
/// pointwise Conv1d → residual.
final class DZConformerConvolution: Module {
    @ModuleInfo(key: "ln_norm") var norm: LayerNorm
    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "bn_norm") var batchNorm: C1BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d
    init(_ width: Int, kernel: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _pointwise1.wrappedValue = Conv1d(inputChannels: width, outputChannels: 2 * width, kernelSize: 1)
        _depthwise.wrappedValue = Conv1d(inputChannels: width, outputChannels: width, kernelSize: kernel, padding: (kernel - 1) / 2, groups: width)
        _batchNorm.wrappedValue = C1BatchNorm(width)
        _pointwise2.wrappedValue = Conv1d(inputChannels: width, outputChannels: width, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let doubled = pointwise1(norm(x)), width = doubled.dim(2) / 2
        let gated = doubled[0..., 0..., 0..<width] * sigmoid(doubled[0..., 0..., width...])
        return x + pointwise2(silu(batchNorm(depthwise(gated))))
    }
}

final class DZConformerBlock: Module {
    @ModuleInfo var ffn1: DZConformerFeedForward
    @ModuleInfo var mha: DZConformerAttention
    @ModuleInfo var conv: DZConformerConvolution
    @ModuleInfo var ffn2: DZConformerFeedForward
    @ModuleInfo(key: "ln_norm") var norm: LayerNorm
    init(_ d: DiariZenDimensions) {
        _ffn1.wrappedValue = DZConformerFeedForward(d.conformerWidth, d.conformerFeedForward)
        _mha.wrappedValue = DZConformerAttention(d.conformerWidth, heads: d.conformerHeads)
        _conv.wrappedValue = DZConformerConvolution(d.conformerWidth, kernel: d.conformerKernel)
        _ffn2.wrappedValue = DZConformerFeedForward(d.conformerWidth, d.conformerFeedForward)
        _norm.wrappedValue = LayerNorm(dimensions: d.conformerWidth, eps: 1e-5)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(ffn2(conv(mha(ffn1(x))))) }
}

final class DZConformer: Module {
    @ModuleInfo(key: "conformer_layer") var layers: [DZConformerBlock]
    init(_ d: DiariZenDimensions) { _layers.wrappedValue = (0..<d.conformerLayers).map { _ in DZConformerBlock(d) } }
}

/// The segmentation model: WavLM hidden states → weighted sum → Linear + LayerNorm → Conformer → powerset log-softmax.
final class DZSegmentation: Module {
    @ModuleInfo(key: "wavlm_model") var wavlm: DZWavLM
    @ModuleInfo(key: "weight_sum") var weightSum: Linear
    @ModuleInfo var proj: Linear
    @ModuleInfo var lnorm: LayerNorm
    @ModuleInfo var conformer: DZConformer
    @ModuleInfo var classifier: Linear
    let dimensions: DiariZenDimensions
    private var bucketCache: (length: Int, table: MLXArray)?
    init(_ d: DiariZenDimensions) {
        dimensions = d
        _wavlm.wrappedValue = DZWavLM(d)
        _weightSum.wrappedValue = Linear(d.layers + 1, 1, bias: false)
        _proj.wrappedValue = Linear(d.embed, d.conformerWidth)
        _lnorm.wrappedValue = LayerNorm(dimensions: d.conformerWidth, eps: 1e-5)
        _conformer.wrappedValue = DZConformer(d)
        _classifier.wrappedValue = Linear(d.conformerWidth, d.classes)
    }

    func buckets(_ length: Int) -> MLXArray {
        if let cache = bucketCache, cache.length == length { return cache.table }
        let table = MLXArray(DZAttention.buckets(length: length, buckets: dimensions.buckets, maxDistance: dimensions.maxDistance), [length, length])
        bucketCache = (length, table)
        return table
    }

    /// WavLM's normalize_waveform: layer_norm over each chunk's samples (no affine, eps 1e-5), in Float32.
    static func normalize(_ audio: MLXArray) -> MLXArray {
        let full = audio.asType(.float32)
        let mean = full.mean(axis: -1, keepDims: true), variance = full.variance(axis: -1, keepDims: true)
        return (full - mean) * rsqrt(variance + 1e-5)
    }

    struct Trace {
        var cnn: MLXArray?
        var projection: MLXArray?
        var states: [MLXArray] = []
        var weightSum: MLXArray?
        var projected: MLXArray?
        var conformer: [MLXArray] = []
    }

    /// [B, samples] waveform → [B, frames, 16] powerset log-probabilities (half). `trace` keeps the intermediates.
    func callAsFunction(_ audio: MLXArray, dtype: DType, trace: UnsafeMutablePointer<Trace>? = nil, cancellation: NativeCancellation) throws -> MLXArray {
        let x = Self.normalize(audio).asType(dtype).expandedDimensions(axis: -1)
        let cnn = wavlm.featureExtractor(x)
        let projected = wavlm.encoder.projection(cnn)
        try cancellation.check()
        let states = try wavlm.encoder.transformer.states(projected, buckets: buckets(projected.dim(1)))
        // weight_sum: Linear(25 → 1) over the stacked states, accumulated in Float32.
        let weights = weightSum.weight.asType(.float32).reshaped(-1).asArray(Float.self)
        var summed = states[0].asType(.float32) * weights[0]
        for i in 1..<states.count {
            summed = summed + states[i].asType(.float32) * weights[i] // swiftlint:disable:this shorthand_operator
        }
        let mixed = summed.asType(dtype)
        var y = lnorm(proj(mixed))
        var conformerStates: [MLXArray] = []
        for layer in conformer.layers {
            y = layer(y)
            if trace != nil { conformerStates.append(y) }
        }
        if let trace {
            trace.pointee = Trace(
                cnn: cnn, projection: projected, states: states, weightSum: mixed, projected: lnorm(proj(mixed)), conformer: conformerStates)
        }
        return logSoftmax(classifier(y).asType(.float32), axis: -1)
    }
}

/// DiariZen pipeline networks: the segmentation model, pyannote's WeSpeaker ResNet34 (Community-1's port, the same
/// architecture) and the VBx PLDA tables (byte-identical to Community-1's).
final class DiariZenNetwork: Module {
    let segmentation: DZSegmentation
    let embedding: C1Embedding
    let plda: C1PLDAWeights
    init(_ d: DiariZenDimensions, blocks: [Int] = [3, 4, 6, 3]) {
        segmentation = DZSegmentation(d)
        embedding = C1Embedding(blocks: blocks)
        plda = C1PLDAWeights()
    }
}
