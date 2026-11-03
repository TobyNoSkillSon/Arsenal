// swiftlint:disable shorthand_operator
// SPDX-License-Identifier: MIT
// LCO-Embedding Omni's audio path: Transformers' WhisperFeatureExtractor (128 log-mel bins, n_fft 400, hop 160, 16 kHz,
// padding="max_length" to 300 s as Qwen2_5OmniProcessor requests) and Qwen2_5OmniAudioEncoder
// (transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py), ported to MLX.
import Accelerate
import Foundation
import ArsenalWire
import IntegrationKit
import MLX
import MLXNN

enum LCOMel {
    static let sampleRate = 16_000, fftSize = 400, hop = 160, bins = 128
    /// The extractor's `n_samples` (chunk_length 300 s): longer input is cut there by the maker's processor.
    static let maxSamples = 4_800_000
    static var maxFrames: Int { maxSamples / hop }

    /// Frames the processor's attention mask keeps for `samples` of audio: `attention_mask[:, ::hop]`.
    static func frames(samples: Int) -> Int { (min(samples, maxSamples) + hop - 1) / hop }

    /// `mel_filter_bank(201, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney")`, shape [201, 128].
    static func filters() -> [[Double]] {
        let freqBins = fftSize / 2 + 1
        func hzToMel(_ f: Double) -> Double {
            let logStep = 27.0 / log(6.4)
            return f >= 1000 ? 15 + log(f / 1000) * logStep : 3 * f / 200
        }
        func melToHz(_ m: Double) -> Double {
            let logStep = log(6.4) / 27.0
            return m >= 15 ? 1000 * exp(logStep * (m - 15)) : 200 * m / 3
        }
        let fft = (0..<freqBins).map { Double($0) * Double(sampleRate / 2) / Double(freqBins - 1) }
        let low = hzToMel(0), high = hzToMel(Double(sampleRate / 2))
        let points = (0..<(bins + 2)).map { melToHz(low + (high - low) * Double($0) / Double(bins + 1)) }
        var bank = Array(repeating: Array(repeating: 0.0, count: bins), count: freqBins)
        for f in 0..<freqBins {
            for m in 0..<bins {
                let down = -(points[m] - fft[f]) / (points[m + 1] - points[m])
                let up = (points[m + 2] - fft[f]) / (points[m + 2] - points[m + 1])
                bank[f][m] = max(0, min(down, up)) * 2 / (points[m + 2] - points[m])
            }
        }
        return bank
    }

    /// Log-mel features, row-major [128][frames(samples)], exactly the frames the processor's feature mask keeps.
    /// Computed as the extractor does over the zero-padded 300 s input (center reflect padding at both ends), including
    /// the global `max - 8` floor; only frames that can see real samples are evaluated, the rest of the padded clip is
    /// silence. Host signal processing with Accelerate in FP32 (not a model weight), no GPU work.
    static func features(_ pcm: [Float]) -> (values: [Float], frames: Int) {
        let samples = min(pcm.count, maxSamples)
        let kept = frames(samples: samples)
        let evaluated = min(maxFrames, kept + 3)
        let half = fftSize / 2
        // padded[i] = P[i - 200] with reflect padding on the 4,800,000-sample zero-padded signal P.
        func source(_ index: Int) -> Float {
            var j = index - half
            if j < 0 { j = -j }
            if j >= maxSamples { j = 2 * (maxSamples - 1) - j }
            return j < samples ? pcm[j] : 0
        }
        var framed = [Float](repeating: 0, count: evaluated * fftSize)
        let window = (0..<fftSize).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fftSize))) }
        for t in 0..<evaluated {
            for n in 0..<fftSize { framed[t * fftSize + n] = source(t * hop + n) * window[n] }
        }
        let freqBins = fftSize / 2 + 1, columns = freqBins * 2
        var basis = [Float](repeating: 0, count: fftSize * columns)
        for n in 0..<fftSize {
            for k in 0..<freqBins {
                let angle = 2 * Double.pi * Double(n * k % fftSize) / Double(fftSize)
                basis[n * columns + k] = Float(cos(angle)); basis[n * columns + freqBins + k] = Float(-sin(angle))
            }
        }
        var spectrum = [Float](repeating: 0, count: evaluated * columns)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(evaluated), Int32(columns), Int32(fftSize), 1, framed, Int32(fftSize), basis, Int32(columns), 0,
            &spectrum, Int32(columns))
        var power = [Float](repeating: 0, count: evaluated * freqBins)
        for t in 0..<evaluated {
            for k in 0..<freqBins {
                let re = spectrum[t * columns + k], im = spectrum[t * columns + freqBins + k]
                power[t * freqBins + k] = re * re + im * im
            }
        }
        let bank = filters().flatMap { $0.map(Float.init) }
        var mel = [Float](repeating: 0, count: evaluated * bins)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(evaluated), Int32(bins), Int32(freqBins), 1, power, Int32(freqBins), bank, Int32(bins), 0, &mel,
            Int32(bins))
        // Frames past `evaluated` see only padding: log10(1e-10) = -10, never above the real maximum.
        var peak: Float = -10
        for i in mel.indices {
            mel[i] = log10(max(mel[i], 1e-10)); peak = max(peak, mel[i])
        }
        var out = [Float](repeating: 0, count: bins * kept)
        for t in 0..<kept {
            for m in 0..<bins { out[m * kept + t] = (max(mel[t * bins + m], peak - 8) + 4) / 4 }
        }
        return (out, kept)
    }
}

final class LCOAudioAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    let heads: Int
    init(_ width: Int, heads: Int) {
        self.heads = heads
        _q.wrappedValue = Linear(width, width, bias: true); _k.wrappedValue = Linear(width, width, bias: false)
        _v.wrappedValue = Linear(width, width, bias: true); _out.wrappedValue = Linear(width, width, bias: true)
    }
    /// x [chunks, length, width]; keys beyond each chunk's valid length are masked (the maker attends per chunk).
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (c, l, w) = (x.dim(0), x.dim(1), x.dim(2)), d = w / heads
        let qs = q(x).reshaped(c, l, heads, d).transposed(0, 2, 1, 3)
        let ks = k(x).reshaped(c, l, heads, d).transposed(0, 2, 1, 3)
        let vs = v(x).reshaped(c, l, heads, d).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: qs, keys: ks, values: vs, scale: pow(Float(d), -0.5), mask: .array(mask))
        return out(o.transposed(0, 2, 1, 3).reshaped(c, l, w))
    }
}

final class LCOAudioLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LCOAudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
    init(_ c: LCOConfiguration.Audio) {
        _attention.wrappedValue = LCOAudioAttention(c.dModel, heads: c.heads)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5)
        _fc1.wrappedValue = Linear(c.dModel, c.ffn); _fc2.wrappedValue = Linear(c.ffn, c.dModel)
        _finalNorm.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x + attention(attentionNorm(x), mask: mask)
        h = h + fc2(gelu(fc1(finalNorm(h))))
        if h.dtype == .float16 {
            // The maker clamps every FP16 layer output to finfo.max - 1000.
            let bound: Float = 65504 - 1000
            h = clip(h, min: -bound, max: bound)
        }
        return h
    }
}

final class LCOAudioEncoder: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo var layers: [LCOAudioLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo var proj: Linear
    let config: LCOConfiguration.Audio
    // Leading underscore: computed, not a checkpoint parameter.
    private let _positions: MLXArray

    init(_ c: LCOConfiguration.Audio) {
        config = c
        _conv1.wrappedValue = Conv1d(inputChannels: c.melBins, outputChannels: c.dModel, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: c.dModel, outputChannels: c.dModel, kernelSize: 3, stride: 2, padding: 1)
        _layers.wrappedValue = (0..<c.layers).map { _ in LCOAudioLayer(c) }
        _lnPost.wrappedValue = LayerNorm(dimensions: c.dModel, eps: 1e-5)
        _proj.wrappedValue = Linear(c.dModel, c.outputDim)
        _positions = Self.sinusoids(length: c.maxSourcePositions, channels: c.dModel)
    }

    /// SinusoidsPositionEmbedding: [sin, cos] over channels/2 log-spaced timescales (max 10,000).
    static func sinusoids(length: Int, channels: Int) -> MLXArray {
        let increment = log(10000.0) / Double(channels / 2 - 1)
        var table = [Float](repeating: 0, count: length * channels)
        for t in 0..<length {
            for i in 0..<(channels / 2) {
                let angle = Double(t) * exp(-increment * Double(i))
                table[t * channels + i] = Float(sin(angle)); table[t * channels + channels / 2 + i] = Float(cos(angle))
            }
        }
        return MLXArray(table, [length, channels])
    }

    /// Chunk lengths of `frames` mel frames: n_window·2 each, the tail keeps the remainder (chunk_and_pad_features).
    static func chunkLengths(frames: Int, window: Int) -> [Int] {
        let size = window * 2
        let count = (frames + size - 1) / size
        return (0..<count).map { $0 == count - 1 && frames % size != 0 ? frames % size : size }
    }
    /// Tokens the encoder emits for `frames` mel frames (`_get_feat_extract_output_lengths`).
    static func outputTokens(frames: Int) -> Int { ((frames - 1) / 2 + 1 - 2) / 2 + 1 }

    /// features [128, frames] → [tokens, output_dim].
    func callAsFunction(_ features: MLXArray, cancellation: NativeCancellation) throws -> MLXArray {
        let frames = features.dim(1)
        guard frames >= 3 else { throw WorkerFailure(code: "invalid_request", message: "Audio is too short to embed (under 20 ms).") }
        let lengths = Self.chunkLengths(frames: frames, window: config.nWindow)
        let longest = lengths.max() ?? 0
        var chunks: [MLXArray] = [], start = 0
        for length in lengths {
            var chunk = features[0..., start..<(start + length)].transposed(1, 0)
            if length < longest { chunk = padded(chunk, widths: [[0, longest - length], [0, 0]]) }
            chunks.append(chunk); start += length
        }
        let dtype = conv1.weight.dtype
        var x = stacked(chunks).asType(dtype) // [chunks, longest, mel]
        let valid = MLXArray(lengths.map(Int32.init)).reshaped(-1, 1)
        let inputMask = (MLXArray(Int32(0)..<Int32(longest)).reshaped(1, -1) .< valid).expandedDimensions(axis: -1).asType(dtype)
        x = gelu(conv1(x)) * inputMask
        x = gelu(conv2(x))
        let after = x.dim(1)
        x = x + _positions[..<after].asType(dtype)
        let afterLengths = lengths.map { ($0 - 1) / 2 + 1 }
        let keyValid = MLXArray(afterLengths.map(Int32.init)).reshaped(-1, 1, 1, 1)
        let mask = MLXArray(Int32(0)..<Int32(after)).reshaped(1, 1, 1, -1) .< keyValid
        for layer in layers {
            try cancellation.check()
            x = layer(x, mask: mask)
        }
        let flat = concatenated(afterLengths.enumerated().map { x[$0.offset, ..<$0.element] }, axis: 0)
        let tokens = Self.outputTokens(frames: frames)
        let pairs = flat[..<(tokens * 2)].reshaped(tokens, 2, -1)
        let pooled = (pairs[0..., 0] + pairs[0..., 1]) / 2
        return proj(lnPost(pooled))
    }
}

// swiftlint:enable shorthand_operator
