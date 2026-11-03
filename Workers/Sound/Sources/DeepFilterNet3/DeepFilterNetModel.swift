// SPDX-License-Identifier: MIT
// DeepFilterNet3 for Arsenal. The network layers come from Blaizzy/mlx-audio-swift @ 0dda61b29c308175e302dceef77c3bd84551e92b
// (Sources/MLXAudioSTS/Models/DeepFilterNet, MIT, licenses/mlx-audio-swift-MIT.txt). Rewritten here: loading (Arsenal's
// Get supplies the weights), libdf's STFT and feature normalisation (exact, on the GPU), the GRUs (a Metal kernel instead
// of an Accelerate CPU loop), the maker's attenuation limit and post filter, and half-precision handling. Version 1 and
// the port's streamer are not carried over.
import Foundation
import MLX
import MLXNN

public enum DeepFilterNetError: Error, LocalizedError, CustomStringConvertible {
    case missingWeightKey(String)
    case invalidInput(String)
    public var description: String {
        switch self {
        case .missingWeightKey(let key): return "DeepFilterNet3 weights lack \(key)"
        case .invalidInput(let message): return message
        }
    }
    public var errorDescription: String? { description }
}

/// Called with (name, activation) at the component boundaries the parity gate compares.
public typealias DeepFilterNetTrace = (String, MLXArray) -> Void

/// Weight-free signal processing of libdf (DeepFilterNet's Rust core): STFT, ERB and complex features with their
/// running normalisation, deep filtering, the post filter and the attenuation limit.
public struct DeepFilterNetDSP: Sendable {
    public let config: DeepFilterNetConfig
    public let window: [Float]
    public let erbWidths: [Int]
    public let alpha: Float
    public var wnorm: Float { Float(2 * config.hopSize) / Float(config.fftSize * config.fftSize) }

    public init(config: DeepFilterNetConfig = .init()) {
        self.config = config
        let half = Double(config.fftSize / 2)
        window = (0..<config.fftSize).map { i in
            let inner = sin(0.5 * Double.pi * (Double(i) + 0.5) / half)
            return Float(sin(0.5 * Double.pi * inner * inner))
        }
        erbWidths = DeepFilterNetModel.libdfErbBandWidths(
            sampleRate: config.sampleRate, fftSize: config.fftSize, nbBands: config.nbErb, minNbFreqs: max(1, config.minNbErbFreqs))
        alpha = Self.normAlpha(hopSize: config.hopSize, sampleRate: config.sampleRate)
    }

    /// libdf's norm alpha: exp(-hop/sr) rounded to the fewest decimals that keep it below 1.
    public static func normAlpha(hopSize: Int, sampleRate: Int) -> Float {
        let raw = exp(-Double(hopSize) / Double(sampleRate))
        var precision = 3.0, a = 1.0
        while a >= 1 { a = (raw * pow(10, precision)).rounded() / pow(10, precision); precision += 1 }
        return Float(a)
    }

    /// Frames the maker's enhance() analyses for `samples` input samples (the input is padded by one FFT size).
    public func frames(samples: Int) -> Int { (samples + config.fftSize) / config.hopSize }

    /// libdf analysis of `audio` padded at the end by one FFT size: [T, F] complex, scaled by wnorm.
    public func analysis(_ audio: [Float]) -> MLXArray {
        let n = config.fftSize, hop = config.hopSize, count = frames(samples: audio.count)
        let memory = n - hop
        let padded = [Float](repeating: 0, count: memory) + audio + [Float](repeating: 0, count: n + hop)
        let framed = asStrided(MLXArray(padded), [count, n], strides: [hop, 1]) * MLXArray(window)
        return MLXFFT.rfft(framed, axis: 1) * wnorm
    }

    /// libdf synthesis (unnormalised inverse FFT, window, overlap-add), trimmed by the maker's delay to `length`.
    public func synthesis(_ spectrum: MLXArray, length: Int) -> [Float] {
        let n = config.fftSize, hop = config.hopSize, count = spectrum.dim(0)
        let framesOut = MLXFFT.irfft(spectrum, n: n, axis: 1) * (Float(n) * MLXArray(window))
        let first = framesOut[0..., 0..<hop], second = framesOut[0..., hop..<n]
        let carried = concatenated([MLXArray.zeros([1, hop], type: Float.self), second[0..<(count - 1)]], axis: 0)
        let audio = (first + carried).reshaped(-1).asArray(Float.self)
        let delay = n - hop
        return Array(audio[delay..<min(audio.count, delay + length)]) + [Float](repeating: 0, count: max(0, delay + length - audio.count))
    }

    /// ERB features [T, nbErb]: band means of |X|^2 in dB, minus libdf's running mean, over 40.
    public func erbFeatures(_ spectrum: MLXArray) -> MLXArray {
        let db = erbDecibels(spectrum)
        return (db - DeepFilterNetKernels.runningMean(db, initial: erbInitial, alpha: alpha)) / 40
    }

    /// Band means of |X|^2 in dB, [T, nbErb] (libdf `erb` with its 1e-10 floor), before normalisation.
    public func erbDecibels(_ spectrum: MLXArray) -> MLXArray {
        let power = spectrum.realPart().square() + spectrum.imaginaryPart().square()
        var start = 0, bands: [MLXArray] = []
        for width in erbWidths { bands.append(power[0..., start..<(start + width)].mean(axis: 1)); start += width }
        return 10 * log10(stacked(bands, axis: 1) + 1e-10)
    }

    /// libdf's initial running states for the ERB and complex normalisations.
    public var erbInitial: [Float] { Self.linspace(-60, -90, config.nbErb) }
    public var complexInitial: [Float] { Self.linspace(0.001, 0.0001, config.nbDf) }

    /// Complex features [T, nbDf, 2]: the low bins over the square root of libdf's running mean magnitude.
    public func complexFeatures(_ spectrum: MLXArray) -> MLXArray {
        let low = spectrum[0..., 0..<config.nbDf]
        let re = low.realPart(), im = low.imaginaryPart()
        let state = DeepFilterNetKernels.runningMean(sqrt(re.square() + im.square()), initial: complexInitial, alpha: alpha)
        // On exact digital silence the state decays below the smallest normal float after about 78 s; the GPU flushes
        // it to zero and 0 / 0 turned every later frame into NaN. Clamping changes nothing for audible input (libdf
        // keeps the subnormal on the CPU), and a zero bin still gives a zero feature.
        let scale = sqrt(maximum(state, MLXArray(Float.leastNormalMagnitude)))
        return stacked([re / scale, im / scale], axis: -1)
    }

    static func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
        count > 1 ? (0..<count).map { start + Float($0) * (end - start) / Float(count - 1) } : [start]
    }

    /// The maker's deep filter (libdf `DF` op): spec [1, 1, T, F, 2], coefs [1, order, T, nbDf, 2] → spec with the
    /// low bins replaced by the complex FIR over `order` frames (`lookahead` of them ahead).
    public func deepFilter(spec: MLXArray, coefs: MLXArray) -> MLXArray {
        let t = spec.dim(2), order = config.dfOrder, lookahead = config.dfLookahead
        let low = padded(spec[0..., 0, 0..., 0..<config.nbDf, 0...], widths: [.init(0), .init((order - 1 - lookahead, lookahead)), .init(0), .init(0)])
        // Two separate arrays: MLX's += updates its left side in place, so `outI = outR` would alias them.
        var outR = MLXArray.zeros([spec.dim(0), t, config.nbDf], dtype: spec.dtype)
        var outI = MLXArray.zeros([spec.dim(0), t, config.nbDf], dtype: spec.dtype)
        for k in 0..<order {
            let window = low[0..., k..<(k + t)], coef = coefs[0..., k].asType(spec.dtype)
            let sr = window[.ellipsis, 0], si = window[.ellipsis, 1], cr = coef[.ellipsis, 0], ci = coef[.ellipsis, 1]
            outR += sr * cr - si * ci
            outI += sr * ci + si * cr
        }
        let filtered = stacked([outR, outI], axis: -1).expandedDimensions(axis: 1)
        return concatenated([filtered, spec[0..., 0..., 0..., config.nbDf..., 0...]], axis: 3)
    }

    /// The maker's post filter (DfNet.forward with mask_pf, beta 0.02), in float32: spectra [.., 2] real/imag.
    public static func postFilter(enhanced: MLXArray, noisy: MLXArray, beta: Float = 0.02) -> MLXArray {
        let eps: Float = 1e-12
        func magnitude(_ x: MLXArray) -> MLXArray { sqrt(x[.ellipsis, 0].square() + x[.ellipsis, 1].square()) }
        let mask = clip(magnitude(enhanced) / (magnitude(noisy) + eps), min: eps, max: 1)
        let maskSin = mask * maximum(sin(Float.pi * mask / 2), eps)
        let gain = (1 + beta) / (1 + beta * (mask / maskSin).square())
        return enhanced * gain.expandedDimensions(axis: -1)
    }
}

public final class DeepFilterNetModel: @unchecked Sendable {
    public let config: DeepFilterNetConfig
    public let dsp: DeepFilterNetDSP
    public let weights: [String: MLXArray]
    let erbInvFB: MLXArray
    let inferenceDType: DType
    let bnScale: [String: MLXArray]
    let bnBias: [String: MLXArray]
    let conv2dWeightsOHWI: [String: MLXArray]
    let convTransposeDenseWeights: [String: MLXArray]
    let convTransposeGroupWeights: [String: [MLXArray]]
    let gruTransposedWeights: [String: MLXArray]

    public init(config: DeepFilterNetConfig = .init(), weights: [String: MLXArray]) throws {
        guard let erbInvFB = weights["mask.erb_inv_fb"] else { throw DeepFilterNetError.missingWeightKey("mask.erb_inv_fb") }
        self.config = config
        self.dsp = DeepFilterNetDSP(config: config)
        self.weights = weights
        self.erbInvFB = erbInvFB
        self.inferenceDType = weights["enc.erb_conv0.1.weight"]?.dtype ?? .float16
        (bnScale, bnBias) = Self.buildBatchNormAffine(weights: weights)
        conv2dWeightsOHWI = Self.buildConv2dWeightCache(weights: weights)
        convTransposeDenseWeights = Self.buildDenseTransposeWeights(weights: weights, groups: max(1, config.convCh))
        convTransposeGroupWeights = Self.buildGroupedTransposeWeights(weights: weights, groups: max(1, config.convCh))
        gruTransposedWeights = Self.buildGRUTransposedWeightCache(weights: weights)
    }

    /// One GRU layer on the GPU: the input projection as one matmul, the recurrence as one Metal kernel.
    func gruLayer(_ x: MLXArray, prefix: String, layer: Int, hiddenSize: Int) throws -> MLXArray {
        guard x.dim(0) == 1 else { throw DeepFilterNetError.invalidInput("DeepFilterNet3 runs one clip at a time") }
        let wihT = try gruTransposedWeights["\(prefix).weight_ih_l\(layer)"] ?? w("\(prefix).weight_ih_l\(layer)").transposed()
        let whhT = try gruTransposedWeights["\(prefix).weight_hh_l\(layer)"] ?? w("\(prefix).weight_hh_l\(layer)").transposed()
        let t = x.dim(1)
        let gx = matmul(x.reshaped([t, x.dim(2)]), wihT) + (try w("\(prefix).bias_ih_l\(layer)"))
        let out = DeepFilterNetKernels.gru(gx: gx, whhT: whhT, bhh: try w("\(prefix).bias_hh_l\(layer)"), hidden: hiddenSize)
        return out.reshaped([1, t, hiddenSize])
    }

    /// The maker's pad_feat: shift features `lookahead` frames earlier, zero-filled at the end.
    public func applyLookahead(feature: MLXArray, lookahead: Int) -> MLXArray {
        guard lookahead > 0 else { return feature }
        // Fewer frames than the lookahead: the maker's negative pad leaves only zeros.
        guard feature.dim(2) > lookahead else { return MLXArray.zeros(feature.shape, dtype: feature.dtype) }
        let pad = MLXArray.zeros([feature.dim(0), feature.dim(1), lookahead, feature.dim(3)], dtype: feature.dtype)
        return concatenated([feature[0..., 0..., lookahead..., 0...], pad], axis: 2)
    }

    /// Cleans one mono clip at 48 kHz: the maker's enhance() with pad=True, optional attenuation limit (dB) and
    /// post filter. Returns the same number of samples.
    public func enhance(_ audio: [Float], attenuationLimit: Float? = nil, postFilter: Bool = false, trace: DeepFilterNetTrace? = nil) throws -> [Float] {
        guard !audio.isEmpty else { return [] }
        let spectrum = dsp.analysis(audio)
        let erb = dsp.erbFeatures(spectrum), cplx = dsp.complexFeatures(spectrum)
        trace?("erb_feat", erb); trace?("spec_feat", cplx)
        let spec = stacked([spectrum.realPart(), spectrum.imaginaryPart()], axis: -1).expandedDimensions(axes: [0, 1])
        var enhanced = try network(
            spec: spec, featErb: erb.expandedDimensions(axes: [0, 1]), featSpec: cplx.expandedDimensions(axes: [0, 1]), trace: trace)
        if postFilter { enhanced = DeepFilterNetDSP.postFilter(enhanced: enhanced, noisy: spec) }
        if let limit = attenuationLimit, abs(limit) > 0 {
            let floor = pow(10, -abs(limit) / 20)
            enhanced = spec * floor + enhanced * (1 - floor)
        }
        let out = enhanced[0, 0]
        return dsp.synthesis(out[.ellipsis, 0] + MLXArray(real: Float(0), imaginary: Float(1)) * out[.ellipsis, 1], length: audio.count)
    }

    /// The network on float32 spectrum [1, 1, T, F, 2] and features; convolutions and GRUs run in the weights' half
    /// dtype, the mask and deep filter apply to the float32 spectrum.
    public func network(spec: MLXArray, featErb: MLXArray, featSpec: MLXArray, trace: DeepFilterNetTrace? = nil) throws -> MLXArray {
        let out = try forward(spec: spec, featErb: featErb.asType(inferenceDType), featSpec5D: featSpec.asType(inferenceDType), trace: trace)
        trace?("spec_enhanced", out)
        return out
    }
}
