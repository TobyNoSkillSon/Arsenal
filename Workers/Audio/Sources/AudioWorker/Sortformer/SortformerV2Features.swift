// SPDX-License-Identifier: MIT
import Accelerate
import Foundation

/// NeMo 3.0 `FilterbankFeatures` as configured by diar_streaming_sortformer_4spk-v2 (`model_config.yaml`): 16 kHz,
/// preemphasis 0.97, 25 ms periodic=False Hann window centred in a 512-point FFT, 10 ms hop, centred STFT with zero
/// padding, power spectrum, 128 librosa Slaney mel filters, log(x + 2^-24), no normalisation ("NA"), no dither at
/// inference. The sequence has floor(samples / 160) frames (NeMo's `get_seq_len`; the last STFT frame is masked off).
/// Float32 on the CPU, the same as NeMo, which disables autocast here. Any frame range can be computed from the full
/// PCM, so a live session yields exactly the offline features once each frame's samples have arrived.
final class SortformerV2Features: @unchecked Sendable {
    static let sampleRate = 16_000, hop = 160, window = 400, fft = 512, mels = 128, preemphasis: Float = 0.97
    static let bins = fft / 2 + 1
    let hann: [Float]
    /// [mels, bins], row-major.
    let filterbank: [Float]
    /// The same filterbank as [bins, mels].
    private let filterbankColumns: [Float]
    private let dft: vDSP.DiscreteFourierTransform<Float>

    init() {
        let window = (0..<Self.window).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(Self.window - 1))) }
        var padded = [Float](repeating: 0, count: Self.fft)
        let left = (Self.fft - Self.window) / 2
        padded.replaceSubrange(left..<(left + Self.window), with: window)
        hann = padded
        filterbank = Self.slaneyFilterbank()
        var columns = [Float](repeating: 0, count: Self.mels * Self.bins)
        vDSP_mtrans(filterbank, 1, &columns, 1, vDSP_Length(Self.bins), vDSP_Length(Self.mels))
        filterbankColumns = columns
        // swiftlint:disable:next force_try
        dft = try! vDSP.DiscreteFourierTransform(previous: nil, count: Self.fft, direction: .forward, transformType: .complexComplex, ofType: Float.self)
    }

    /// Frames NeMo's preprocessor returns for `samples` samples.
    static func frameCount(samples: Int) -> Int { samples / hop }

    /// Highest frame index + 1 whose window lies entirely within the first `available` samples (live sessions).
    static func completeFrames(available: Int) -> Int { available < fft / 2 ? 0 : (available - fft / 2) / hop + 1 }

    /// librosa 0.10 `filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, htk=False, norm="slaney")`.
    static func slaneyFilterbank() -> [Float] {
        let fSp = 200.0 / 3, minLogHz = 1000.0, minLogMel = minLogHz / fSp, logStep = log(6.4) / 27
        func mel(_ hz: Double) -> Double { hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logStep : hz / fSp }
        func hz(_ mel: Double) -> Double { mel >= minLogMel ? minLogHz * exp(logStep * (mel - minLogMel)) : fSp * mel }
        let top = mel(Double(sampleRate) / 2)
        let edges = (0..<(mels + 2)).map { hz(top * Double($0) / Double(mels + 1)) }
        var bank = [Float](repeating: 0, count: mels * bins)
        for m in 0..<mels {
            let lower = edges[m + 1] - edges[m], upper = edges[m + 2] - edges[m + 1], norm = 2 / (edges[m + 2] - edges[m])
            for b in 0..<bins {
                let f = Double(b) * Double(sampleRate) / Double(fft)
                let weight = max(0, min((f - edges[m]) / lower, (edges[m + 2] - f) / upper))
                bank[m * bins + b] = Float(weight * norm)
            }
        }
        return bank
    }

    /// Log-mel frames `range` of the clip `pcm` (its full length so far) as [frames × 128], row-major.
    func features(_ pcm: [Float], frames range: Range<Int>) -> [Float] {
        let count = range.count
        guard count > 0 else { return [] }
        var power = [Float](repeating: 0, count: count * Self.bins)
        var real = [Float](repeating: 0, count: Self.fft), imaginary = [Float](repeating: 0, count: Self.fft)
        var outReal = [Float](repeating: 0, count: Self.fft), outImaginary = [Float](repeating: 0, count: Self.fft)
        let n = pcm.count
        pcm.withUnsafeBufferPointer { x in
            for (row, frame) in range.enumerated() {
                let start = frame * Self.hop - Self.fft / 2
                for i in 0..<Self.fft {
                    let t = start + i
                    // Preemphasis y[t] = x[t] - 0.97 x[t-1] (y[0] = x[0]); zero outside the clip (centred constant pad).
                    let value: Float = (t < 0 || t >= n) ? 0 : (t == 0 ? x[0] : x[t] - Self.preemphasis * x[t - 1])
                    real[i] = value * hann[i]
                    imaginary[i] = 0
                }
                dft.transform(inputReal: real, inputImaginary: imaginary, outputReal: &outReal, outputImaginary: &outImaginary)
                for b in 0..<Self.bins {
                    // NeMo: sqrt(re² + im²) then pow(2); the same value up to rounding.
                    power[row * Self.bins + b] = outReal[b] * outReal[b] + outImaginary[b] * outImaginary[b]
                }
            }
        }
        var mel = [Float](repeating: 0, count: count * Self.mels)
        // [count × bins] · [bins × mels]
        vDSP_mmul(power, 1, filterbankColumns, 1, &mel, 1, vDSP_Length(count), vDSP_Length(Self.mels), vDSP_Length(Self.bins))
        for i in 0..<mel.count { mel[i] = log(mel[i] + 0x1p-24) }
        return mel
    }
}
