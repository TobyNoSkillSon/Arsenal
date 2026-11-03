// SPDX-License-Identifier: MIT
// torchaudio.compliance.kaldi.fbank as configured by pyannote's WeSpeakerResNet34 (80 mel bins, 25 ms frames,
// 10 ms shift, snip_edges, no dither, hamming, power spectrum, log floor at float eps), then the per-chunk
// time-mean subtraction of BaseWeSpeakerResNet.compute_fbank. Framing runs on the host; the FFT and mel
// projection are MLX DSP intermediates in Float32; the network receives half features.
import Foundation
import MLX

enum Community1Fbank {
    static let frameLength = 400, frameShift = 160, fftSize = 512, melBins = 80

    /// Triangular HTK-mel bank (20 Hz to Nyquist) with the zero Nyquist row torchaudio pads on: [257, 80].
    static let melBank: MLXArray = {
        func mel(_ hz: Double) -> Double { 1127 * log(1 + hz / 700) }
        let low = mel(20), high = mel(8000), delta = (high - low) / Double(melBins + 1)
        var bank = [Float](repeating: 0, count: (fftSize / 2 + 1) * melBins)
        for bin in 0..<(fftSize / 2) {
            let m = mel(Double(bin) * 16_000 / Double(fftSize))
            for band in 0..<melBins {
                let left = low + Double(band) * delta, center = left + delta, right = center + delta
                let up = (m - left) / (center - left), down = (right - m) / (right - center)
                bank[bin * melBins + band] = Float(max(0, min(up, down)))
            }
        }
        return MLXArray(bank, [fftSize / 2 + 1, melBins])
    }()

    /// Symmetric Hamming window (torch.hamming_window(periodic=False)).
    static let window: [Float] = (0..<frameLength).map { Float(0.54 - 0.46 * cos(2 * Double.pi * Double($0) / Double(frameLength - 1))) }

    /// [1, frames, 80] half features for one chunk.
    static func features(_ samples: [Float]) throws -> MLXArray {
        guard samples.count >= frameLength else { throw audioFailure("Community-1 embedding needs at least 400 samples.") }
        let frames = 1 + (samples.count - frameLength) / frameShift
        var data = [Float](repeating: 0, count: frames * frameLength)
        for frame in 0..<frames {
            let start = frame * frameShift
            var mean: Float = 0
            for i in 0..<frameLength { mean += samples[start + i] * 32768 }
            mean /= Float(frameLength)
            // remove_dc_offset, then preemphasis 0.97 with the first sample replicated, then the window.
            var previous = samples[start] * 32768 - mean
            for i in 0..<frameLength {
                let value = samples[start + i] * 32768 - mean
                data[frame * frameLength + i] = (value - 0.97 * previous) * window[i]
                previous = value
            }
        }
        let framed = MLX.padded(MLXArray(data, [frames, frameLength]), widths: [0, IntOrPair((0, fftSize - frameLength))])
        let spectrum = abs(MLXFFT.rfft(framed, axis: 1))
        let energies = matmul(spectrum * spectrum, melBank)
        let logged = log(maximum(energies, Float.ulpOfOne))
        return (logged - logged.mean(axis: 0, keepDims: true)).expandedDimensions(axis: 0).asType(.float16)
    }
}
