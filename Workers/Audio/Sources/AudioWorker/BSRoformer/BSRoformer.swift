// SPDX-License-Identifier: MIT
// Band-Split RoPE Transformer (BS-RoFormer), ported from the reference used by python-audio-separator 0.45.0
// (`audio_separator/separator/uvr_lib_v5/roformer/bs_roformer.py`, after lucidrains/BS-RoFormer, MIT). Inference only:
// dropout, linear attention, sage attention and the training losses are absent from the shipped checkpoint's path.
import Foundation
import MLX
import MLXFast
import MLXNN

struct BSRoformerConfig: Equatable {
    var dim = 512
    var depth = 12
    var stereo = true
    var timeDepth = 1
    var freqDepth = 1
    var heads = 8
    var dimHead = 64
    var maskDepth = 2
    var mlpExpansion = 4
    var nFFT = 2048
    var hop = 441
    var sampleRate = 44100
    /// `inference.dim_t`: frames per chunk (chunk = hop × (dim_t − 1)).
    var dimT = 801
    var freqsPerBands: [Int] =
        Array(repeating: 2, count: 24) + Array(repeating: 4, count: 12) + Array(repeating: 12, count: 8)
        + Array(repeating: 24, count: 8) + Array(repeating: 48, count: 8) + [128, 129]
    var channels: Int { stereo ? 2 : 1 }
    /// Inputs per band: frequency bins × channels × (real, imaginary).
    var bandInputs: [Int] { freqsPerBands.map { 2 * $0 * channels } }

    /// The audio-separator YAML subset that changes the graph. Unknown model keys are refused rather than ignored.
    static func parse(_ object: [String: Any]) throws -> BSRoformerConfig {
        var config = BSRoformerConfig()
        guard let model = object["model"] as? [String: Any], let audio = object["audio"] as? [String: Any] else {
            throw audioFailure("BS-RoFormer configuration needs model and audio sections.")
        }
        func int(_ table: [String: Any], _ key: String) -> Int? { (table[key] as? NSNumber)?.intValue }
        config.dim = int(model, "dim") ?? config.dim
        config.depth = int(model, "depth") ?? config.depth
        config.stereo = (model["stereo"] as? Bool) ?? config.stereo
        config.timeDepth = int(model, "time_transformer_depth") ?? config.timeDepth
        config.freqDepth = int(model, "freq_transformer_depth") ?? config.freqDepth
        config.heads = int(model, "heads") ?? config.heads
        config.dimHead = int(model, "dim_head") ?? config.dimHead
        config.maskDepth = int(model, "mask_estimator_depth") ?? config.maskDepth
        config.mlpExpansion = int(model, "mlp_expansion_factor") ?? config.mlpExpansion
        config.nFFT = int(model, "stft_n_fft") ?? config.nFFT
        config.hop = int(model, "stft_hop_length") ?? config.hop
        if let bands = model["freqs_per_bands"] as? [NSNumber] { config.freqsPerBands = bands.map(\.intValue) }
        config.sampleRate = int(audio, "sample_rate") ?? config.sampleRate
        if let inference = object["inference"] as? [String: Any], let dimT = int(inference, "dim_t") { config.dimT = dimT }
        guard int(model, "num_stems") ?? 1 == 1, int(model, "linear_transformer_depth") ?? 0 == 0,
            (model["stft_win_length"] as? NSNumber)?.intValue ?? config.nFFT == config.nFFT, (model["stft_normalized"] as? Bool) != true,
            (model["sage_attention"] as? Bool) != true, (model["skip_connection"] as? Bool) != true,
            config.freqsPerBands.reduce(0, +) == config.nFFT / 2 + 1, config.maskDepth >= 1, config.dimT > 1,
            config.dim > 0, config.heads > 0, config.dimHead > 0, config.dimHead % 2 == 0, config.hop > 0
        else { throw audioFailure("Unsupported BS-RoFormer configuration.") }
        return config
    }
}

/// `F.normalize(x, dim=-1) * sqrt(dim) * gamma`, computed with fp32 accumulation (the residual stream's squared sum
/// exceeds FP16's range). The tiny epsilon keeps all-zero (silent) frames at zero, as normalize's clamp does.
final class RoformerRMSNorm: Module {
    @ParameterInfo var gamma: MLXArray
    init(_ dim: Int) { _gamma.wrappedValue = MLXArray.ones([dim]); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { MLXFast.rmsNorm(x, weight: gamma, eps: 1e-30) }
}

final class RoformerAttention: Module {
    @ModuleInfo var norm: RoformerRMSNorm
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "to_gates") var toGates: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    let heads: Int, dimHead: Int
    init(dim: Int, heads: Int, dimHead: Int) {
        self.heads = heads; self.dimHead = dimHead
        _norm.wrappedValue = RoformerRMSNorm(dim)
        _toQKV.wrappedValue = Linear(dim, 3 * heads * dimHead, bias: false)
        _toGates.wrappedValue = Linear(dim, heads)
        _toOut.wrappedValue = Linear(heads * dimHead, dim, bias: false)
        super.init()
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = norm(input)
        let (b, n) = (x.dim(0), x.dim(1))
        // "b n (qkv h d) -> qkv b h n d"
        let qkv = toQKV(x).reshaped(b, n, 3, heads, dimHead).transposed(2, 0, 3, 1, 4)
        // rotary_embedding_torch rotates interleaved pairs (rotate_half over "(d r)"), positions 0..<n, theta 10 000.
        let q = MLXFast.RoPE(qkv[0], dimensions: dimHead, traditional: true, base: 10_000, scale: 1, offset: 0)
        let k = MLXFast.RoPE(qkv[1], dimensions: dimHead, traditional: true, base: 10_000, scale: 1, offset: 0)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: qkv[2], scale: 1 / Float(dimHead).squareRoot(), mask: .none)
        let out = attended * sigmoid(toGates(x)).transposed(0, 2, 1).expandedDimensions(axis: -1)
        return toOut(out.transposed(0, 2, 1, 3).reshaped(b, n, heads * dimHead))
    }
}

final class RoformerFeedForward: Module {
    @ModuleInfo var norm: RoformerRMSNorm
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    init(dim: Int, mult: Int = 4) {
        _norm.wrappedValue = RoformerRMSNorm(dim)
        _projIn.wrappedValue = Linear(dim, dim * mult)
        _projOut.wrappedValue = Linear(dim * mult, dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { projOut(gelu(projIn(norm(x)))) }
}

final class RoformerLayer: Module {
    @ModuleInfo var attn: RoformerAttention
    @ModuleInfo var ff: RoformerFeedForward
    init(_ config: BSRoformerConfig) {
        _attn.wrappedValue = RoformerAttention(dim: config.dim, heads: config.heads, dimHead: config.dimHead)
        _ff.wrappedValue = RoformerFeedForward(dim: config.dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = attn(x) + x
        return ff(h) + h
    }
}

/// One axial step: a transformer over time for every band, then one over bands for every frame.
final class RoformerAxialBlock: Module {
    @ModuleInfo var time: [RoformerLayer]
    @ModuleInfo var freq: [RoformerLayer]
    init(_ config: BSRoformerConfig) {
        _time.wrappedValue = (0..<config.timeDepth).map { _ in RoformerLayer(config) }
        _freq.wrappedValue = (0..<config.freqDepth).map { _ in RoformerLayer(config) }
        super.init()
    }
    /// x: [b, t, f, d].
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (b, t, f, d) = (input.dim(0), input.dim(1), input.dim(2), input.dim(3))
        var x = input.transposed(0, 2, 1, 3).reshaped(b * f, t, d)
        for layer in time { x = layer(x) }
        x = x.reshaped(b, f, t, d).transposed(0, 2, 1, 3).reshaped(b * t, f, d)
        for layer in freq { x = layer(x) }
        return x.reshaped(b, t, f, d)
    }
}

final class RoformerBandIn: Module {
    @ModuleInfo var norm: RoformerRMSNorm
    @ModuleInfo var linear: Linear
    init(_ inputs: Int, dim: Int) {
        _norm.wrappedValue = RoformerRMSNorm(inputs)
        _linear.wrappedValue = Linear(inputs, dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(norm(x)) }
}

/// MaskEstimator's per-band MLP (Linear, Tanh, …, Linear) followed by GLU over the last axis.
final class RoformerBandOut: Module {
    @ModuleInfo var layers: [Linear]
    init(_ inputs: Int, dim: Int, hidden: Int, depth: Int) {
        let dims = [dim] + Array(repeating: hidden, count: depth - 1) + [inputs * 2]
        _layers.wrappedValue = zip(dims.dropLast(), dims.dropFirst()).map { Linear($0, $1) }
        super.init()
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        for (index, layer) in layers.enumerated() {
            x = layer(x)
            if index < layers.count - 1 { x = tanh(x) }
        }
        let half = x.dim(-1) / 2
        return x[.ellipsis, ..<half] * sigmoid(x[.ellipsis, half...])
    }
}

final class BSRoformerModel: Module {
    let config: BSRoformerConfig
    @ModuleInfo(key: "band_split") var bandSplit: [RoformerBandIn]
    @ModuleInfo var layers: [RoformerAxialBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RoformerRMSNorm
    @ModuleInfo(key: "mask_estimator") var maskEstimator: [RoformerBandOut]
    let spectral: DemucsSpectralPair

    init(_ config: BSRoformerConfig) {
        self.config = config
        spectral = DemucsSpectralPair(nFFT: config.nFFT, hopLength: config.hop)
        _bandSplit.wrappedValue = config.bandInputs.map { RoformerBandIn($0, dim: config.dim) }
        _layers.wrappedValue = (0..<config.depth).map { _ in RoformerAxialBlock(config) }
        _finalNorm.wrappedValue = RoformerRMSNorm(config.dim)
        _maskEstimator.wrappedValue = config.bandInputs.map {
            RoformerBandOut($0, dim: config.dim, hidden: config.dim * config.mlpExpansion, depth: config.maskDepth)
        }
        super.init()
    }

    /// Complex STFT [c, f, t] of a [c, samples] float32 chunk, as (real, imag) float32.
    func spectrum(_ audio: MLXArray) -> (MLXArray, MLXArray) {
        let s = spectral.stft(audio.expandedDimensions(axis: 0))
        return (s.real[0], s.imag[0])
    }

    /// Band features [1, t, bands, dim] in the weights' dtype from the complex spectrum ("b s f t c -> b t (f s c)").
    func features(real: MLXArray, imag: MLXArray, dtype: DType) -> MLXArray {
        let (c, f, t) = (real.dim(0), real.dim(1), real.dim(2))
        let packed = stacked([real, imag], axis: -1).transposed(2, 1, 0, 3).reshaped(1, t, f * c * 2).asType(dtype)
        var outputs: [MLXArray] = [], start = 0
        for (band, width) in zip(bandSplit, config.bandInputs) {
            outputs.append(band(packed[.ellipsis, start..<(start + width)]))
            start += width
        }
        return stacked(outputs, axis: -2)
    }

    /// Transformer trunk and mask estimator: [1, t, bands, dim] → mask [1, t, f·c·2] in the weights' dtype.
    func mask(_ features: MLXArray) -> MLXArray {
        var x = features
        for layer in layers { x = layer(x) }
        x = finalNorm(x)
        return concatenated(maskEstimator.enumerated().map { $1(x[0..., 0..., $0, 0...]) }, axis: -1)
    }

    /// Masked spectrum back to audio: [c, hop·(t−1)] float32.
    func render(real: MLXArray, imag: MLXArray, mask: MLXArray) -> MLXArray {
        let (c, f, t) = (real.dim(0), real.dim(1), real.dim(2))
        // mask "(f s c)" per frame → [c, f, t] real and imaginary parts.
        let m = mask.asType(.float32).reshaped(t, f, c, 2).transposed(2, 1, 0, 3)
        let (mr, mi) = (m[.ellipsis, 0], m[.ellipsis, 1])
        let outReal = real * mr - imag * mi, outImag = real * mi + imag * mr
        let length = config.hop * (t - 1)
        return spectral.istft(
            DemucsComplexSpectrogram(real: outReal.expandedDimensions(axis: 0), imag: outImag.expandedDimensions(axis: 0)), length: length)[0]
    }

    /// The target stem (vocals for the shipped checkpoint) of one [c, samples] float32 chunk.
    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let (real, imag) = spectrum(audio)
        let dtype = bandSplit[0].linear.weight.dtype
        return render(real: real, imag: imag, mask: mask(features(real: real, imag: imag, dtype: dtype)))
    }
}
