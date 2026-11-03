// swiftlint:disable shorthand_operator
// SPDX-License-Identifier: MIT
// EmbeddingGemma 2's audio path: Transformers' Gemma4AudioFeatureExtractor (models/gemma4/feature_extraction_gemma4.py,
// 5.22.1: 16 kHz, semicausal 20 ms frames every 10 ms, periodic Hann, |rfft 512|, 128 HTK mel bins, log(mel + 0.001))
// and Gemma4AudioModel (modeling_gemma4.py: two stride-2 convolutions, 12 conformer layers with chunked local attention
// and relative positions, clipped linears), ported to MLX.
import Accelerate
import ArsenalWire
import Foundation
import IntegrationKit
import MLX
import MLXFast
import MLXNN

enum EG2Mel {
    static let sampleRate = 16_000, frameLength = 320, hop = 160, fftLength = 512, bins = 128
    /// The extractor's `max_length` (truncation on): 30 s of audio.
    static let maxSamples = 480_000
    static let melFloor = 0.001

    /// Frames for `samples` after the extractor pads the clip to a multiple of 128 samples.
    static func frames(samples: Int) -> Int {
        let padded = (samples + 127) / 128 * 128
        return max(0, (padded + frameLength / 2 - (frameLength + 1)) / hop + 1)
    }

    /// Frames whose whole analysis window is real audio (the extractor's mask).
    static func validFrames(samples: Int) -> Int {
        (0..<frames(samples: samples)).filter { $0 * hop + frameLength / 2 < samples }.count
    }

    /// Soft tokens after the two stride-2 convolutions: the processor's `compute_sscp` on the frame mask.
    static func tokens(samples: Int) -> Int {
        var mask = (0..<frames(samples: samples)).map { $0 * hop + frameLength / 2 < samples }
        for _ in 0..<2 { mask = Swift.stride(from: 0, to: mask.count, by: 2).map { mask[$0] } }
        return mask.filter { $0 }.count
    }

    /// `mel_filter_bank(257, 128, 0, 8000, 16000, norm=None, mel_scale="htk")`, shape [257][128].
    static func filters() -> [[Double]] {
        let freqBins = fftLength / 2 + 1
        func hzToMel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        func melToHz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let low = hzToMel(0), high = hzToMel(8000)
        let points = (0..<(bins + 2)).map { melToHz(low + (high - low) * Double($0) / Double(bins + 1)) }
        let fft = (0..<freqBins).map { Double($0) * Double(sampleRate / 2) / Double(freqBins - 1) }
        var bank = Array(repeating: Array(repeating: 0.0, count: bins), count: freqBins)
        for f in 0..<freqBins {
            for m in 0..<bins {
                let down = (fft[f] - points[m]) / (points[m + 1] - points[m])
                let up = (points[m + 2] - fft[f]) / (points[m + 2] - points[m + 1])
                bank[f][m] = max(0, min(down, up))
            }
        }
        return bank
    }

    /// Log-mel features `[frames][128]` (row-major) and the frame mask, invalid frames zeroed, as the extractor returns
    /// them for one clip. Host signal processing in float64 with Accelerate, not a model weight.
    static func features(_ pcm: [Float]) -> (values: [Float], frames: Int, mask: [Bool]) {
        let samples = min(pcm.count, maxSamples)
        let count = frames(samples: samples)
        guard count > 0 else { return ([], 0, []) }
        let padLeft = frameLength / 2
        let window = (0..<frameLength).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(frameLength)) }
        var framed = [Double](repeating: 0, count: count * frameLength)
        for t in 0..<count {
            for n in 0..<frameLength {
                let j = t * hop + n - padLeft
                framed[t * frameLength + n] = (j >= 0 && j < samples ? Double(pcm[j]) : 0) * window[n]
            }
        }
        let freqBins = fftLength / 2 + 1, columns = freqBins * 2
        var basis = [Double](repeating: 0, count: frameLength * columns)
        for n in 0..<frameLength {
            for k in 0..<freqBins {
                let angle = 2 * Double.pi * Double(n * k % fftLength) / Double(fftLength)
                basis[n * columns + k] = cos(angle); basis[n * columns + freqBins + k] = -sin(angle)
            }
        }
        var spectrum = [Double](repeating: 0, count: count * columns)
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(count), Int32(columns), Int32(frameLength), 1, framed, Int32(frameLength), basis,
            Int32(columns), 0, &spectrum, Int32(columns))
        var magnitude = [Double](repeating: 0, count: count * freqBins)
        for t in 0..<count {
            for k in 0..<freqBins {
                let re = spectrum[t * columns + k], im = spectrum[t * columns + freqBins + k]
                magnitude[t * freqBins + k] = (re * re + im * im).squareRoot()
            }
        }
        let bank = filters().flatMap { $0 }
        var mel = [Double](repeating: 0, count: count * bins)
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(count), Int32(bins), Int32(freqBins), 1, magnitude, Int32(freqBins), bank, Int32(bins), 0,
            &mel, Int32(bins))
        let mask = (0..<count).map { $0 * hop + padLeft < samples }
        var values = [Float](repeating: 0, count: count * bins)
        for t in 0..<count where mask[t] {
            for m in 0..<bins { values[t * bins + m] = Float(log(mel[t * bins + m] + melFloor)) }
        }
        return (values, count, mask)
    }
}

struct EmbeddingGemma2AudioConfiguration: Codable, Sendable {
    let hiddenSize: Int, heads: Int, layers: Int, outputDims: Int, chunk: Int, contextLeft: Int, contextRight: Int
    let logitCap: Float, invalidLogit: Float, residualWeight: Float, convKernel: Int, channels: [Int], rmsNormEps: Float, clipping: Double
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", heads = "num_attention_heads", layers = "num_hidden_layers", outputDims = "output_proj_dims"
        case chunk = "attention_chunk_size", contextLeft = "attention_context_left", contextRight = "attention_context_right"
        case logitCap = "attention_logit_cap", invalidLogit = "attention_invalid_logits_value", residualWeight = "residual_weight"
        case convKernel = "conv_kernel_size", channels = "subsampling_conv_channels", rmsNormEps = "rms_norm_eps"
        case clipping = "gradient_clipping", clipped = "use_clipped_linears", act = "hidden_act", modelType = "model_type"
    }
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize); heads = try c.decode(Int.self, forKey: .heads)
        layers = try c.decode(Int.self, forKey: .layers); outputDims = try c.decode(Int.self, forKey: .outputDims)
        chunk = try c.decode(Int.self, forKey: .chunk); contextLeft = try c.decode(Int.self, forKey: .contextLeft)
        contextRight = try c.decode(Int.self, forKey: .contextRight); logitCap = try c.decode(Float.self, forKey: .logitCap)
        invalidLogit = try c.decode(Float.self, forKey: .invalidLogit); residualWeight = try c.decode(Float.self, forKey: .residualWeight)
        convKernel = try c.decode(Int.self, forKey: .convKernel); channels = try c.decode([Int].self, forKey: .channels)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps); clipping = try c.decode(Double.self, forKey: .clipping)
        let clipped = try c.decodeIfPresent(Bool.self, forKey: .clipped) ?? false
        let act = try c.decodeIfPresent(String.self, forKey: .act) ?? "silu"
        let type = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_audio"
        guard type == "gemma4_audio", clipped, act == "silu", contextRight == 0, channels.count == 2, hiddenSize % heads == 0,
            channels[0] / 4 * channels[1] == hiddenSize
        else {
            throw DecodingError.dataCorruptedError(forKey: .modelType, in: c, debugDescription: "Unsupported EmbeddingGemma 2 audio layout.")
        }
    }
    func encode(to encoder: any Encoder) throws {}
}

/// `Gemma4ClippableLinear` with clipping: inputs and outputs clamped to the trained bounds.
final class EG2ClippedLinear: Module, UnaryLayer {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray
    @ModuleInfo(key: "input_max") var inputMax: MLXArray
    @ModuleInfo(key: "output_min") var outputMin: MLXArray
    @ModuleInfo(key: "output_max") var outputMax: MLXArray
    init(_ input: Int, _ output: Int) {
        _linear.wrappedValue = Linear(input, output, bias: false)
        _inputMin.wrappedValue = MLXArray(-Float.greatestFiniteMagnitude); _inputMax.wrappedValue = MLXArray(Float.greatestFiniteMagnitude)
        _outputMin.wrappedValue = MLXArray(-Float.greatestFiniteMagnitude); _outputMax.wrappedValue = MLXArray(Float.greatestFiniteMagnitude)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        clip(linear(clip(x, min: inputMin, max: inputMax)), min: outputMin, max: outputMax)
    }
}

final class EG2AudioFeedForward: Module {
    @ModuleInfo(key: "ffw_layer_1") var first: EG2ClippedLinear
    @ModuleInfo(key: "ffw_layer_2") var second: EG2ClippedLinear
    @ModuleInfo(key: "pre_layer_norm") var preNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_layer_norm") var postNorm: EmbeddingGemma2RMSNorm
    let scale: Float
    init(_ c: EmbeddingGemma2AudioConfiguration) {
        scale = c.residualWeight
        _first.wrappedValue = EG2ClippedLinear(c.hiddenSize, c.hiddenSize * 4)
        _second.wrappedValue = EG2ClippedLinear(c.hiddenSize * 4, c.hiddenSize)
        _preNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _postNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
    }
    func callAsFunction(_ x: MLXArray, bound: Float) -> MLXArray {
        var h = preNorm(clip(x, min: -bound, max: bound))
        h = clip(second(silu(first(h))), min: -bound, max: bound)
        return postNorm(h) * MLXArray(scale).asType(x.dtype) + x
    }
}

final class EG2AudioLightConv: Module {
    @ModuleInfo(key: "linear_start") var start: EG2ClippedLinear
    @ModuleInfo(key: "linear_end") var end: EG2ClippedLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwise: EG2DepthwiseWeight
    @ModuleInfo(key: "pre_layer_norm") var preNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "conv_norm") var convNorm: EmbeddingGemma2RMSNorm
    init(_ c: EmbeddingGemma2AudioConfiguration) {
        _start.wrappedValue = EG2ClippedLinear(c.hiddenSize, c.hiddenSize * 2)
        _end.wrappedValue = EG2ClippedLinear(c.hiddenSize, c.hiddenSize)
        _depthwise.wrappedValue = EG2DepthwiseWeight(channels: c.hiddenSize, kernel: c.convKernel)
        _preNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _convNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, bound: Float) -> MLXArray {
        let gated = start(preNorm(x))
        let half = gated.dim(-1) / 2
        var h = gated[.ellipsis, ..<half] * sigmoid(gated[.ellipsis, half...])
        h = depthwise(h)
        h = silu(convNorm(clip(h, min: -bound, max: bound)))
        return end(h) + x
    }
}

/// The causal depthwise Conv1d (`[channels, 1, kernel]`, left padding kernel - 1) as kernel shifted multiply-adds.
final class EG2DepthwiseWeight: Module {
    var weight: MLXArray
    init(channels: Int, kernel: Int) { weight = MLXArray.zeros([channels, 1, kernel]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let kernel = weight.dim(2), length = x.dim(1)
        let padded = MLX.padded(x, widths: [0, .init((kernel - 1, 0)), 0])
        var out = padded[0..., 0..<length, 0...] * weight[0..., 0, 0]
        for k in 1..<kernel { out = out + padded[0..., k..<(k + length), 0...] * weight[0..., 0, k] }
        return out
    }
}

final class EG2AudioAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: EG2ClippedLinear
    @ModuleInfo(key: "k_proj") var kProj: EG2ClippedLinear
    @ModuleInfo(key: "v_proj") var vProj: EG2ClippedLinear
    @ModuleInfo(key: "post") var post: EG2ClippedLinear
    @ModuleInfo(key: "relative_k_proj") var relativeK: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray
    let heads: Int, headDim: Int, chunk: Int, past: Int, context: Int, cap: Float, invalid: Float
    init(_ c: EmbeddingGemma2AudioConfiguration) {
        heads = c.heads; headDim = c.hiddenSize / c.heads; chunk = c.chunk; past = c.contextLeft - 1
        context = c.chunk + past + c.contextRight; cap = c.logitCap; invalid = c.invalidLogit
        _qProj.wrappedValue = EG2ClippedLinear(c.hiddenSize, c.hiddenSize)
        _kProj.wrappedValue = EG2ClippedLinear(c.hiddenSize, c.hiddenSize)
        _vProj.wrappedValue = EG2ClippedLinear(c.hiddenSize, c.hiddenSize)
        _post.wrappedValue = EG2ClippedLinear(c.hiddenSize, c.hiddenSize)
        _relativeK.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _perDimScale.wrappedValue = MLXArray.zeros([c.hiddenSize / c.heads])
    }

    /// `_rel_shift` for `[..., chunk, past + 1]` scores: key column j of query row i reads distance index j - i
    /// (0…past) and is zero elsewhere; returns `[..., chunk, context]`.
    static func relativeShift(_ rel: MLXArray, context: Int) -> MLXArray {
        let chunk = rel.dim(-2), past = rel.dim(-1) - 1
        let offsets = (0..<chunk).flatMap { i in (0..<context).map { j in Int32(min(max(j - i, 0), past)) } }
        let lead = Array(rel.shape.dropLast(2))
        let index = MLXArray(offsets, Array(repeating: 1, count: lead.count) + [chunk, context])
        let inRange = MLXArray(
            (0..<chunk).flatMap { i in (0..<context).map { j in j - i >= 0 && j - i <= past } }, Array(repeating: 1, count: lead.count) + [chunk, context])
        let gathered = takeAlong(rel, MLX.broadcast(index, to: lead + [chunk, context]), axis: -1)
        return MLX.where(inRange, gathered, MLXArray(Float(0)).asType(rel.dtype))
    }

    /// `x`: `[1, T, hidden]`; `positions`: `[past + 1, hidden]` (distances past…0); `allowed`: `[blocks, chunk, context]`.
    /// The maker computes this block in float32.
    func callAsFunction(_ x: MLXArray, positions: MLXArray, allowed: MLXArray) -> MLXArray {
        let length = x.dim(1), blocks = (length + chunk - 1) / chunk
        let qScale = Float(1 / Double(headDim).squareRoot() / log(2.0)), kScale = Float(log(1 + M_E) / log(2.0))
        var q = qProj(x).asType(.float32).reshaped(length, heads, headDim)
        q = q * qScale * softplus(perDimScale.asType(.float32))
        let k = kProj(x).asType(.float32).reshaped(length, heads, headDim) * kScale
        let v = vProj(x).asType(.float32).reshaped(length, heads, headDim)
        // Query blocks [blocks, chunk, H, D]; key/value windows of `context` starting `past` before each block.
        let queries = MLX.padded(q, widths: [.init((0, blocks * chunk - length)), 0, 0]).reshaped(blocks, chunk, heads, headDim)
        let window = MLXArray((0..<blocks).flatMap { b in (0..<context).map { Int32(b * chunk + $0) } })
        func windows(_ t: MLXArray) -> MLXArray {
            MLX.padded(t, widths: [.init((past, blocks * chunk + context - chunk - past - length)), 0, 0])
                .take(window, axis: 0).reshaped(blocks, context, heads, headDim)
        }
        let keys = windows(k), values = windows(v)
        let qh = queries.transposed(2, 0, 1, 3) // [H, blocks, chunk, D]
        let content = matmul(qh, keys.transposed(2, 0, 3, 1)) // [H, blocks, chunk, context]
        let relative = relativeK(positions).asType(.float32).reshaped(-1, heads, headDim) // [past + 1, H, D]
        let rel = matmul(qh, relative.transposed(1, 2, 0).expandedDimensions(axis: 1)) // [H, blocks, chunk, past + 1]
        var logits = content + Self.relativeShift(rel, context: context)
        logits = MLX.tanh(logits / cap) * cap
        logits = MLX.where(allowed.expandedDimensions(axis: 0), logits, MLXArray(invalid))
        let weights = softmax(logits, axis: -1)
        let out = matmul(weights, values.transposed(2, 0, 1, 3)) // [H, blocks, chunk, D]
        let merged = out.transposed(1, 2, 0, 3).reshaped(1, blocks * chunk, heads * headDim)[0..., ..<length, 0...]
        return post(merged.asType(x.dtype))
    }
}

final class EG2AudioLayer: Module {
    @ModuleInfo(key: "feed_forward1") var ff1: EG2AudioFeedForward
    @ModuleInfo(key: "feed_forward2") var ff2: EG2AudioFeedForward
    @ModuleInfo(key: "self_attn") var attention: EG2AudioAttention
    @ModuleInfo(key: "lconv1d") var lconv: EG2AudioLightConv
    @ModuleInfo(key: "norm_pre_attn") var preAttention: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "norm_post_attn") var postAttention: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "norm_out") var out: EmbeddingGemma2RMSNorm
    init(_ c: EmbeddingGemma2AudioConfiguration) {
        _ff1.wrappedValue = EG2AudioFeedForward(c); _ff2.wrappedValue = EG2AudioFeedForward(c)
        _attention.wrappedValue = EG2AudioAttention(c); _lconv.wrappedValue = EG2AudioLightConv(c)
        _preAttention.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _postAttention.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _out.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
    }
    func callAsFunction(_ x: MLXArray, positions: MLXArray, allowed: MLXArray, bound: Float) -> MLXArray {
        var h = ff1(x, bound: bound)
        let residual = h
        h = attention(preAttention(clip(h, min: -bound, max: bound)), positions: positions, allowed: allowed)
        h = postAttention(clip(h, min: -bound, max: bound)) + residual
        h = ff2(lconv(h, bound: bound), bound: bound)
        return out(clip(h, min: -bound, max: bound))
    }
}

final class EG2SubsampleLayer: Module {
    @ModuleInfo(key: "conv") var conv: EG2ConvWeight
    @ModuleInfo(key: "norm") var norm: EG2LayerNormWeight
    init(input: Int, output: Int) {
        _conv.wrappedValue = EG2ConvWeight(output: output, input: input)
        _norm.wrappedValue = EG2LayerNormWeight(output)
    }
}

/// Conv2d weight kept in the checkpoint's `[out, in, 3, 3]` layout; applied as NHWC with stride 2, padding 1.
final class EG2ConvWeight: Module {
    var weight: MLXArray
    init(output: Int, input: Int) { weight = MLXArray.zeros([output, input, 3, 3]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2d(x, weight.transposed(0, 2, 3, 1), stride: .init(2), padding: .init(1))
    }
}

final class EG2LayerNormWeight: Module {
    var weight: MLXArray
    init(_ size: Int) { weight = MLXArray.ones([size]) }
    func callAsFunction(_ x: MLXArray, eps: Float) -> MLXArray { MLXFast.layerNorm(x, weight: weight, bias: nil, eps: eps) }
}

final class EG2Subsample: Module {
    @ModuleInfo(key: "layer0") var layer0: EG2SubsampleLayer
    @ModuleInfo(key: "layer1") var layer1: EG2SubsampleLayer
    @ModuleInfo(key: "input_proj_linear") var projection: Linear
    init(_ c: EmbeddingGemma2AudioConfiguration) {
        _layer0.wrappedValue = EG2SubsampleLayer(input: 1, output: c.channels[0])
        _layer1.wrappedValue = EG2SubsampleLayer(input: c.channels[0], output: c.channels[1])
        _projection.wrappedValue = Linear(c.channels[0] / 4 * c.channels[1], c.hiddenSize, bias: false)
    }
}

/// The audio tower (`audio_tower.*`).
final class EmbeddingGemma2AudioTower: Module {
    @ModuleInfo(key: "subsample_conv_projection") var subsample: EG2Subsample
    @ModuleInfo(key: "layers") var layers: [EG2AudioLayer]
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    let config: EmbeddingGemma2AudioConfiguration

    init(_ c: EmbeddingGemma2AudioConfiguration) {
        config = c
        _subsample.wrappedValue = EG2Subsample(c)
        _layers.wrappedValue = (0..<c.layers).map { _ in EG2AudioLayer(c) }
        _outputProj.wrappedValue = Linear(c.hiddenSize, c.outputDims, bias: true)
    }

    /// Gemma4AudioRelPositionalEncoding: distances past…0 as [sin…, cos…].
    func relativePositions(dtype: DType) -> MLXArray {
        let past = config.contextLeft - 1, half = config.hiddenSize / 2
        let increment = log(10000.0) / Double(max(half - 1, 1))
        let inverse = MLXArray((0..<half).map { Float(exp(Double($0) * -increment)) })
        let distances = MLXArray((0...past).reversed().map { Float($0) }).reshaped(-1, 1)
        let scaled = distances * inverse.reshaped(1, half)
        return concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: -1).asType(dtype)
    }

    /// `[blocks, chunk, context]`: a key is allowed when it is real (inside the clip and valid) and lies 0…past - 1
    /// positions before its query (sliding window (left - 1, right 0) of the maker's mask); padded query rows see none.
    func allowedKeys(valid: [Bool]) -> MLXArray {
        let chunk = config.chunk, past = config.contextLeft - 1, context = chunk + past + config.contextRight
        let length = valid.count, blocks = (length + chunk - 1) / chunk
        var allowed = [Bool](repeating: false, count: blocks * chunk * context)
        for b in 0..<blocks {
            for i in 0..<chunk {
                let query = b * chunk + i
                for j in 0..<context {
                    let key = b * chunk - past + j, distance = query - key
                    allowed[(b * chunk + i) * context + j] = query < length && key >= 0 && key < length && valid[key] && distance >= 0 && distance < past
                }
            }
        }
        return MLXArray(allowed, [blocks, chunk, context])
    }

    /// Tower output for one clip's features (`[frames][128]` row-major, float32) and frame mask: the valid soft tokens,
    /// `[tokens, output_proj_dims]` in the weights' dtype.
    func callAsFunction(_ values: [Float], frames: Int, mask: [Bool], cancellation: NativeCancellation) throws -> MLXArray {
        let dtype = outputProj.weight.dtype
        let bound = Float(min(config.clipping, dtype == .float16 ? 65504 : Double(Float.greatestFiniteMagnitude)))
        var m = mask
        var h = MLXArray(values, [1, frames, EG2Mel.bins, 1])
        func step(_ layer: EG2SubsampleLayer, _ input: MLXArray) -> MLXArray {
            let keep = MLXArray(m.map { $0 ? Float(1) : 0 }, [1, m.count, 1, 1])
            let x = (input.asType(.float32) * keep).asType(dtype)
            m = Swift.stride(from: 0, to: m.count, by: 2).map { m[$0] }
            return relu(layer.norm(layer.conv(x), eps: config.rmsNormEps))
        }
        h = step(subsample.layer0, h)
        h = step(subsample.layer1, h)
        let length = h.dim(1)
        precondition(length == m.count)
        h = subsample.projection(h.reshaped(1, length, -1))
        let positions = relativePositions(dtype: dtype), allowed = allowedKeys(valid: m)
        for layer in layers {
            try cancellation.check()
            h = layer(h, positions: positions, allowed: allowed, bound: bound)
            eval(h)
        }
        h = outputProj(h)[0]
        let keep = m.indices.filter { m[$0] }.map(Int32.init)
        return h.take(MLXArray(keep), axis: 0)
    }
}

// swiftlint:enable shorthand_operator
