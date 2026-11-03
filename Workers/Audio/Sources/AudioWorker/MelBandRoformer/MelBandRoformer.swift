// SPDX-License-Identifier: MIT
// Mel-Band RoFormer, ported from KimberleyJensen/Mel-Band-Roformer-Vocal-Model (25f44ffb,
// `models/mel_band_roformer/mel_band_roformer.py`, after lucidrains/BS-RoFormer, MIT). Inference only. The attention,
// feed-forward, band input and band output layers are BS-RoFormer's (same code upstream); what differs is the band
// layout (overlapping mel bands from librosa's Slaney filterbank), the output norm closing every transformer, and the
// averaging of overlapping band masks.
import Foundation
import MLX
import MLXFast
import MLXNN

struct MelBandRoformerConfig: Equatable {
    var dim = 384
    var depth = 6
    var stereo = true
    var timeDepth = 1
    var freqDepth = 1
    var bands = 60
    var heads = 8
    var dimHead = 64
    var maskDepth = 2
    var mlpExpansion = 4
    var nFFT = 2048
    var hop = 441
    var sampleRate = 44100
    /// `inference.chunk_size` and `num_overlap` of the release's config.
    var chunk = 352_800
    var overlap = 2
    var channels: Int { stereo ? 2 : 1 }
    var freqs: Int { nFFT / 2 + 1 }

    /// The layer settings BS-RoFormer's shared layers read.
    var layers: BSRoformerConfig {
        var config = BSRoformerConfig()
        config.dim = dim; config.heads = heads; config.dimHead = dimHead; config.timeDepth = timeDepth; config.freqDepth = freqDepth
        config.mlpExpansion = mlpExpansion
        return config
    }
}

/// librosa 0.10+ `filters.mel(sr, n_fft, n_mels)` (Slaney scale and norm, fmin 0, fmax sr/2) reduced to membership,
/// with the maker's two forced corners: band 0 owns bin 0 and the last band owns the last bin.
struct MelBandLayout: Equatable {
    /// Band-major, ascending (frequency × channel + channel) row indices into the "(f s)" spectrum.
    let indices: [Int32]
    /// Members per band, times channels (one input row per (frequency, channel)).
    let rowsPerBand: [Int]
    /// Bands covering each frequency.
    let bandsPerFreq: [Int]

    static func hzToMel(_ hz: Double) -> Double {
        let sp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / sp, step = log(6.4) / 27
        return hz >= minLogHz ? minLogMel + log(hz / minLogHz) / step : hz / sp
    }
    static func melToHz(_ mel: Double) -> Double {
        let sp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / sp, step = log(6.4) / 27
        return mel >= minLogMel ? minLogHz * exp(step * (mel - minLogMel)) : sp * mel
    }

    /// Membership [band][freq] of the filterbank's positive weights.
    static func membership(sampleRate: Int, nFFT: Int, bands: Int) -> [[Bool]] {
        let freqs = nFFT / 2 + 1, nyquist = Double(sampleRate) / 2
        let fft = (0..<freqs).map { Double($0) * nyquist / Double(freqs - 1) }
        let low = hzToMel(0), high = hzToMel(nyquist)
        let mel = (0..<(bands + 2)).map { melToHz(low + (high - low) * Double($0) / Double(bands + 1)) }
        var member = (0..<bands).map { band in
            (0..<freqs).map { f -> Bool in
                let lower = (fft[f] - mel[band]) / (mel[band + 1] - mel[band])
                let upper = (mel[band + 2] - fft[f]) / (mel[band + 2] - mel[band + 1])
                // librosa computes in float64 and stores float32; a float64 weight below float32's smallest
                // subnormal would read as 0 there.
                return Float(max(0, min(lower, upper))) > 0
            }
        }
        member[0][0] = true
        member[bands - 1][freqs - 1] = true
        return member
    }

    init(_ config: MelBandRoformerConfig) {
        let member = Self.membership(sampleRate: config.sampleRate, nFFT: config.nFFT, bands: config.bands)
        let c = config.channels
        var indices: [Int32] = [], rows: [Int] = []
        for band in member {
            var count = 0
            for (f, on) in band.enumerated() where on {
                for s in 0..<c { indices.append(Int32(f * c + s)) }
                count += c
            }
            rows.append(count)
        }
        self.indices = indices
        rowsPerBand = rows
        bandsPerFreq = (0..<config.freqs).map { f in member.reduce(0) { $0 + ($1[f] ? 1 : 0) } }
    }

    /// Band input widths: rows × (real, imaginary).
    var bandInputs: [Int] { rowsPerBand.map { 2 * $0 } }
    var covered: Bool { bandsPerFreq.allSatisfy { $0 > 0 } }
}

/// Fixed index tensors, held outside the module tree so they are not parameters.
final class MelBandBuffers {
    let indices: MLXArray
    /// 1 / bands covering each "(f s)" row, [f·c, 1, 1] float32.
    let inverseCoverage: MLXArray
    init(_ layout: MelBandLayout, config: MelBandRoformerConfig) {
        let c = config.channels
        indices = MLXArray(layout.indices)
        inverseCoverage = MLXArray((0..<(config.freqs * c)).map { 1 / Float(max(layout.bandsPerFreq[$0 / c], 1)) }).reshaped(-1, 1, 1)
    }
}

/// A transformer stack closed by its own RMSNorm (`norm_output=True` upstream).
final class MelRoformerTransformer: Module {
    @ModuleInfo var layers: [RoformerLayer]
    @ModuleInfo var norm: RoformerRMSNorm
    init(depth: Int, config: BSRoformerConfig) {
        _layers.wrappedValue = (0..<depth).map { _ in RoformerLayer(config) }
        _norm.wrappedValue = RoformerRMSNorm(config.dim)
        super.init()
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        for layer in layers { x = layer(x) }
        return norm(x)
    }
}

final class MelRoformerAxialBlock: Module {
    @ModuleInfo var time: MelRoformerTransformer
    @ModuleInfo var freq: MelRoformerTransformer
    init(_ config: MelBandRoformerConfig) {
        _time.wrappedValue = MelRoformerTransformer(depth: config.timeDepth, config: config.layers)
        _freq.wrappedValue = MelRoformerTransformer(depth: config.freqDepth, config: config.layers)
        super.init()
    }
    /// x: [b, t, f, d].
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (b, t, f, d) = (input.dim(0), input.dim(1), input.dim(2), input.dim(3))
        var x = time(input.transposed(0, 2, 1, 3).reshaped(b * f, t, d))
        x = x.reshaped(b, f, t, d).transposed(0, 2, 1, 3).reshaped(b * t, f, d)
        return freq(x).reshaped(b, t, f, d)
    }
}

final class MelBandRoformerModel: Module {
    let config: MelBandRoformerConfig
    let layout: MelBandLayout
    @ModuleInfo(key: "band_split") var bandSplit: [RoformerBandIn]
    @ModuleInfo var layers: [MelRoformerAxialBlock]
    @ModuleInfo(key: "mask_estimator") var maskEstimator: [RoformerBandOut]
    let spectral: DemucsSpectralPair
    let buffers: MelBandBuffers

    init(_ config: MelBandRoformerConfig) {
        self.config = config
        let layout = MelBandLayout(config)
        self.layout = layout
        spectral = DemucsSpectralPair(nFFT: config.nFFT, hopLength: config.hop)
        buffers = MelBandBuffers(layout, config: config)
        _bandSplit.wrappedValue = layout.bandInputs.map { RoformerBandIn($0, dim: config.dim) }
        _layers.wrappedValue = (0..<config.depth).map { _ in MelRoformerAxialBlock(config) }
        // Upstream MLP(depth=d) has d hidden layers, so d + 1 linears.
        _maskEstimator.wrappedValue = layout.bandInputs.map {
            RoformerBandOut($0, dim: config.dim, hidden: config.dim * config.mlpExpansion, depth: config.maskDepth + 1)
        }
        super.init()
    }

    /// Complex STFT of a [c, samples] float32 chunk as "(f s)"-ordered rows: (real, imag) [f·c, t] float32.
    func spectrum(_ audio: MLXArray) -> (MLXArray, MLXArray) {
        let s = spectral.stft(audio.expandedDimensions(axis: 0))
        let (c, f, t) = (s.real.dim(1), s.real.dim(2), s.real.dim(3))
        func rows(_ x: MLXArray) -> MLXArray { x[0].transposed(1, 0, 2).reshaped(f * c, t) }
        return (rows(s.real), rows(s.imag))
    }

    /// Band features [1, t, bands, dim] in the weights' dtype ("b f t c -> b t (f c)" over the gathered rows).
    func features(real: MLXArray, imag: MLXArray, dtype: DType) -> MLXArray {
        let t = real.dim(1)
        let gathered = stacked([real.take(buffers.indices, axis: 0), imag.take(buffers.indices, axis: 0)], axis: -1) // [n, t, 2]
        let packed = gathered.transposed(1, 0, 2).reshaped(1, t, -1).asType(dtype)
        var outputs: [MLXArray] = [], start = 0
        for (band, width) in zip(bandSplit, layout.bandInputs) {
            outputs.append(band(packed[.ellipsis, start..<(start + width)]))
            start += width
        }
        return stacked(outputs, axis: -2)
    }

    /// Axial trunk: [1, t, bands, dim] → the same shape (input of the mask estimator).
    func trunk(_ features: MLXArray) -> MLXArray {
        var x = features
        for layer in layers { x = layer(x) }
        return x
    }

    /// Mask estimator: [1, t, bands, dim] → [1, t, rows·2] in the weights' dtype.
    func mask(fromTrunk x: MLXArray) -> MLXArray {
        concatenated(maskEstimator.enumerated().map { $1(x[0..., 0..., $0, 0...]) }, axis: -1)
    }

    /// Overlapping band masks averaged per "(f s)" row and applied to the spectrum, then the iSTFT: [c, hop·(t−1)].
    func render(real: MLXArray, imag: MLXArray, mask: MLXArray) -> MLXArray {
        let (rows, t) = (real.dim(0), real.dim(1)), c = config.channels
        let m = mask.asType(.float32).reshaped(t, -1, 2).transposed(1, 0, 2) // [n, t, 2]
        let averaged = MLXArray.zeros([rows, t, 2], dtype: .float32).at[buffers.indices].add(m) * buffers.inverseCoverage
        let (mr, mi) = (averaged[.ellipsis, 0], averaged[.ellipsis, 1])
        let outReal = real * mr - imag * mi, outImag = real * mi + imag * mr
        func planar(_ x: MLXArray) -> MLXArray { x.reshaped(rows / c, c, t).transposed(1, 0, 2).expandedDimensions(axis: 0) }
        return spectral.istft(DemucsComplexSpectrogram(real: planar(outReal), imag: planar(outImag)), length: config.hop * (t - 1))[0]
    }

    /// The vocals of one [c, samples] float32 chunk.
    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let (real, imag) = spectrum(audio)
        let dtype = bandSplit[0].linear.weight.dtype
        return render(real: real, imag: imag, mask: mask(fromTrunk: trunk(features(real: real, imag: imag, dtype: dtype))))
    }
}
