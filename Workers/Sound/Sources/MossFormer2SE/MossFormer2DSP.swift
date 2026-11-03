// SPDX-License-Identifier: MIT
// Weight-free pre- and post-processing of MossFormer2_SE_48K, following the maker's ClearerVoice-Studio
// (clearvoice/utils/decode.py decode_one_audio_mossformer2_se_48k and utils/misc.py at 6b3774dc): Kaldi fbank
// (torchaudio.compliance.kaldi.fbank) with deltas, torch.stft/istft with center=False and a symmetric Hamming window,
// and the maker's one-pass or 4 s windowed decode. Numerical intermediates are float32; the network itself runs in
// the weights' half dtype. Written for Arsenal; the mlx-audio-swift port's DSP used an HTK mel bank instead of Kaldi's.
import Foundation
import MLX
import MLXFFT

public enum MossFormer2DSP {
    /// The maker scales [-1, 1] audio to 16-bit integer range before every transform.
    public static let maxWav: Float = 32768
    /// torch.finfo(torch.float).eps, Kaldi's log floor.
    static let logFloor: Float = 1.1920929e-07

    /// torch.hamming_window(n, periodic=False).
    public static func hamming(_ n: Int) -> [Float] {
        guard n > 1 else { return [Float](repeating: 1, count: max(n, 0)) }
        return (0..<n).map { Float(0.54 - 0.46 * cos(2 * Double.pi * Double($0) / Double(n - 1))) }
    }

    /// Kaldi mel bank (torchaudio get_mel_banks, no VTLN) as [paddedWindow/2 + 1, numMels], Nyquist row zero.
    public static func kaldiMelBank(numMels: Int, paddedWindow: Int, sampleRate: Int, lowFreq: Double = 20) -> [Float] {
        func mel(_ f: Double) -> Double { 1127 * log(1 + f / 700) }
        let bins = paddedWindow / 2, width = Double(sampleRate) / Double(paddedWindow)
        let low = mel(lowFreq), high = mel(Double(sampleRate) / 2), delta = (high - low) / Double(numMels + 1)
        var bank = [Float](repeating: 0, count: (bins + 1) * numMels)
        for m in 0..<numMels {
            let left = low + Double(m) * delta, center = left + delta, right = center + delta
            for i in 0..<bins {
                let x = mel(width * Double(i))
                let w = max(0, min((x - left) / (center - left), (right - x) / (right - center)))
                bank[i * numMels + m] = Float(w)
            }
        }
        return bank
    }

    static func frames(of x: MLXArray, window: Int, hop: Int) -> MLXArray {
        let count = 1 + (x.shape[0] - window) / hop
        return asStrided(x, [count, window], strides: [hop, 1])
    }

    /// torchaudio.compliance.kaldi.fbank with the maker's arguments (40 ms / 8 ms Hamming frames, snip edges,
    /// DC removal, 0.97 pre-emphasis, power spectrum on a 2048-point FFT, log mel energies) and dither 0.
    /// `x` [N] float32 at 16-bit scale → [frames, numMels].
    public static func fbank(_ x: MLXArray, config: MossFormer2Config) -> MLXArray {
        let win = config.winLen
        guard x.shape[0] >= win else { return MLXArray.zeros([0, config.numMels], type: Float.self) }
        var fftSize = 1
        while fftSize < win { fftSize <<= 1 }
        var f = frames(of: x.asType(.float32), window: win, hop: config.winInc)
        f -= f.mean(axis: 1, keepDims: true)
        let previous = concatenated([f[0..., 0..<1], f[0..., 0..<(win - 1)]], axis: 1)
        f = (f - config.preemphasis * previous) * MLXArray(hamming(win))
        f = padded(f, widths: [.init(0), .init((0, fftSize - win))])
        let spectrum = abs(MLXFFT.rfft(f, axis: 1)).square()
        let bank = MLXArray(kaldiMelBank(numMels: config.numMels, paddedWindow: fftSize, sampleRate: config.sampleRate)).reshaped(fftSize / 2 + 1, config.numMels)
        return log(maximum(matmul(spectrum, bank), MLXArray(logFloor)))
    }

    /// torchaudio.functional.compute_deltas (window 5, replicate padding) along time of [T, C].
    public static func deltas(_ x: MLXArray) -> MLXArray {
        let t = x.shape[0]
        guard t > 0 else { return x }
        let p = padded(x, widths: [.init((2, 2)), .init(0)], mode: .edge)
        return (2 * (p[4..<(t + 4)] - p[0..<t]) + (p[3..<(t + 3)] - p[1..<(t + 1)])) / 10
    }

    /// The network input: fbank, delta and delta-delta, [T, 180].
    public static func features(_ x: MLXArray, config: MossFormer2Config) -> MLXArray {
        let fb = fbank(x, config: config), d = deltas(fb)
        return concatenated([fb, d, deltas(d)], axis: 1)
    }

    /// torch.stft(center=False, window=hamming(winLen, periodic=False), onesided): [frames, fftLen/2 + 1] complex.
    public static func stft(_ x: MLXArray, config: MossFormer2Config) -> MLXArray {
        let f = frames(of: x.asType(.float32), window: config.winLen, hop: config.winInc) * MLXArray(hamming(config.winLen))
        return MLXFFT.rfft(f, n: config.fftLen, axis: 1)
    }

    /// torch.istft(center=False, length=`length`): overlap-add of windowed frames divided by the summed squared
    /// window; samples past the last frame are zero, as in torch.
    public static func istft(_ spectrum: MLXArray, length: Int, config: MossFormer2Config) -> [Float] {
        let window = hamming(config.winLen), hop = config.winInc, n = config.winLen
        let framesOut = MLXFFT.irfft(spectrum, n: config.fftLen, axis: 1)[0..., 0..<n].asType(.float32)
        let values = framesOut.asArray(Float.self), count = spectrum.shape[0]
        let covered = count > 0 ? (count - 1) * hop + n : 0
        var signal = [Float](repeating: 0, count: max(covered, length)), envelope = [Float](repeating: 0, count: covered)
        for frame in 0..<count {
            let base = frame * hop, row = frame * n
            for k in 0..<n {
                signal[base + k] += values[row + k] * window[k]
                envelope[base + k] += window[k] * window[k]
            }
        }
        for i in 0..<covered { signal[i] /= max(envelope[i], 1e-11) }
        return Array(signal.prefix(length))
    }

    /// One pass of the maker's decode over `segment` (16-bit scale): mask the STFT with the network's mask.
    /// `mask` maps [1, T, 180] features to a [1, T, 961] mask.
    public static func enhanceSegment(_ segment: [Float], config: MossFormer2Config, mask: (MLXArray) throws -> MLXArray) rethrows -> [Float] {
        // Shorter than one 40 ms frame: the maker cannot frame it; zero-pad to one frame and trim back.
        let x = MLXArray(segment.count >= config.winLen ? segment : segment + [Float](repeating: 0, count: config.winLen - segment.count))
        let m = try mask(features(x, config: config).expandedDimensions(axis: 0))[0].asType(.float32)
        let spectrum = stft(x, config: config)
        let frames = min(spectrum.shape[0], m.shape[0])
        let masked = spectrum[0..<frames] * m[0..<frames]
        return Array(istft(masked, length: x.shape[0], config: config).prefix(segment.count))
    }

    /// Padded length and window starts for the windowed decode. The maker's padding is kept, because it decides which
    /// windows exist; each window keeps all but `(window - stride) / 2` samples at its inner edges, the first window its
    /// start and the last window drops its final edge too. Wherever that leaves the clip's own tail uncovered (the exact
    /// fit, and the two short-clip branches where the clip ends inside the last window's dropped edge), one more stride
    /// of padding is added, so every input sample is enhanced; the maker returns silence there.
    public static func windowPlan(count t: Int, window: Int, stride: Int) -> (padded: Int, starts: [Int]) {
        var total: Int
        if t < window {
            total = window
        } else if t < window + stride {
            total = window + stride
        } else if (t - window) % stride != 0 {
            total = t + t - (t - window) / stride * stride
        } else {
            total = t
        }
        let giveUp = (window - stride) / 2
        func starts(_ total: Int) -> [Int] { Array(Swift.stride(from: 0, through: total - window, by: stride)) }
        while starts(total).last! + window - giveUp < t { total += stride }
        return (total, starts(total))
    }

    /// The maker's decode of a whole clip in [-1, 1] at 48 kHz: one pass up to `oneTimeDecodeLength` seconds, else
    /// windows of `decodeWindow` seconds at 3/4 stride, keeping each window's centre (0.5 s dropped at each inner edge).
    /// Padding and windows follow `windowPlan`.
    public static func decode(
        _ audio: [Float], config: MossFormer2Config, progress: ((Double) -> Void)? = nil,
        mask: (MLXArray) throws -> MLXArray
    ) throws -> [Float] {
        let scaled = audio.map { $0 * maxWav }
        let output: [Float]
        if audio.count <= config.sampleRate * config.oneTimeDecodeLength {
            output = try enhanceSegment(scaled, config: config, mask: mask)
            progress?(1)
        } else {
            let window = config.sampleRate * config.decodeWindow, stride = Int(Double(window) * 0.75)
            let plan = windowPlan(count: scaled.count, window: window, stride: stride)
            let input = scaled + [Float](repeating: 0, count: plan.padded - scaled.count)
            let giveUp = (window - stride) / 2, total = input.count
            var out = [Float](repeating: 0, count: total)
            for index in plan.starts {
                let segment = try enhanceSegment(Array(input[index..<(index + window)]), config: config, mask: mask)
                if index == 0 {
                    out.replaceSubrange(0..<(window - giveUp), with: segment[0..<(window - giveUp)])
                } else {
                    out.replaceSubrange((index + giveUp)..<(index + window - giveUp), with: segment[giveUp..<(window - giveUp)])
                }
                progress?(min(1, Double(index + stride) / Double(max(1, total - window + stride))))
            }
            output = out
        }
        return Array(output.prefix(audio.count)).map { $0 / maxWav }
    }
}
