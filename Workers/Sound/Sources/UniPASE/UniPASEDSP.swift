// SPDX-License-Identifier: MIT
// Weight-free pre- and post-processing of UniPASE, written for Arsenal after the maker's code (Xiaobin-Rong/unipase
// 857b60ad; MIT, with Apache-2.0 and MIT parts from PASE, ESPnet, WavLM and WavTokenizer): torchaudio's sinc resampler, the encoder's padding and packet-loss detection
// (models/wavlm/feature_extractor_plc.py), the Vocos ISTFT head (models/vocoder/vocos/head.py), the PostNet's
// STFT, band split and 8 kHz crossfade (models/postnet/tfgrid_cws_res.py), the output length rule of
// models/unipase.py, and the long-form segmentation of inference/inference_long.py. All numerical intermediates are
// float32 (resampling on the CPU through Accelerate, FFTs through MLX); the networks run in the weights' half dtype.
import Accelerate
import Foundation
import MLX
import MLXFFT

public enum UniPASEDSP {
    // MARK: Resampling

    /// Largest polyphase kernel `resample` builds, in floats (4 MB). Every pair of the family's rates and the usual
    /// recording rates fit (44.1 kHz → 16 kHz needs 76,000); a rate whose reduced ratio is large (44,099 Hz → 16 kHz
    /// would need 706 million) does not, and callers convert it another way first (`sincFits`).
    public static let maxSincKernel = 1 << 20

    /// Floats in torchaudio's kernel for `from` → `to`: new · (2·width + orig) after the gcd reduction.
    public static func sincKernelSize(from: Int, to: Int) -> Int {
        guard from != to, from > 0, to > 0 else { return 0 }
        let g = gcd(from, to), orig = from / g, new = to / g
        let width = Int((6.0 * Double(orig) / (Double(min(orig, new)) * 0.99)).rounded(.up))
        return new * (2 * width + orig)
    }

    /// Whether `resample` can convert `from` → `to` within `maxSincKernel`.
    public static func sincFits(from: Int, to: Int) -> Bool { sincKernelSize(from: from, to: to) <= maxSincKernel }

    /// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99): the same polyphase
    /// kernel, the same zero padding, and ceil(new * n / orig) output samples. Only for pairs that `sincFits`.
    public static func resample(_ x: [Float], from: Int, to: Int) -> [Float] {
        guard from != to, !x.isEmpty, from > 0, to > 0 else { return x }
        precondition(sincFits(from: from, to: to), "sinc kernel for \(from) → \(to) Hz exceeds \(maxSincKernel) floats")
        let g = gcd(from, to), orig = from / g, new = to / g, lowpass = 6.0
        let base = Double(min(orig, new)) * 0.99
        let width = Int((lowpass * Double(orig) / base).rounded(.up))
        let taps = 2 * width + orig
        // The maker builds the kernel in float32; so does this port, operation for operation.
        var kernel = [Float](repeating: 0, count: new * taps)
        let baseF = Float(base), scale = Float(base / Double(orig)), pi = Float.pi
        for j in 0..<new {
            for k in 0..<taps {
                var t = Float(-j) / Float(new) + Float(k - width) / Float(orig)
                t *= baseF
                t = min(max(t, -Float(lowpass)), Float(lowpass))
                let window = pow(cos(t * pi / Float(lowpass) / 2), 2)
                t *= pi
                kernel[j * taps + k] = (t == 0 ? 1 : sin(t) / t) * window * scale
            }
        }
        var padded = [Float](repeating: 0, count: width + x.count + width + orig)
        padded.replaceSubrange(width..<(width + x.count), with: x)
        let frames = (padded.count - taps) / orig + 1
        let target = (new * x.count + orig - 1) / orig
        var out = [Float](repeating: 0, count: frames * new)
        padded.withUnsafeBufferPointer { p in
            kernel.withUnsafeBufferPointer { kp in
                out.withUnsafeMutableBufferPointer { op in
                    DispatchQueue.concurrentPerform(iterations: min(8, max(1, frames / 4096))) { part in
                        let chunks = min(8, max(1, frames / 4096)), lo = frames * part / chunks, hi = frames * (part + 1) / chunks
                        for f in lo..<hi {
                            for j in 0..<new {
                                vDSP_dotpr(p.baseAddress! + f * orig, 1, kp.baseAddress! + j * taps, 1, op.baseAddress! + f * new + j, vDSP_Length(taps))
                            }
                        }
                    }
                }
            }
        }
        return Array(out.prefix(target))
    }

    static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

    // MARK: Encoder front

    /// WavLM_feat.pad: the 16 kHz signal becomes 320·⌊n/320⌋ + 80 samples long (zero padding, or a crop when n mod
    /// 320 > 80), so the WavLM CNN yields exactly ⌊n/320⌋ frames, one per 20 ms packet.
    public static func padForEncoder(_ x: [Float]) -> [Float] {
        let target = x.count / 320 * 320 + 80
        if x.count == target { return x }
        if x.count > target { return Array(x.prefix(target)) }
        return x + [Float](repeating: 0, count: target - x.count)
    }

    /// detect_packet_loss_indices at 16 kHz: a 20 ms packet is lost when at least 99% of its samples are below 1e-7
    /// in magnitude. One flag per packet; nil when the signal holds no whole packet (the maker returns None).
    public static func packetLossMask(_ x: [Float]) -> [Bool]? {
        let packets = x.count / 320
        guard packets > 0 else { return nil }
        return (0..<packets).map { p in
            var zeros = 0
            for i in (p * 320)..<(p * 320 + 320) where abs(x[i]) < 1e-7 { zeros += 1 }
            // The maker compares the float32 mean with 0.99: 317 of 320 passes, 316 does not.
            return Float(zeros) / 320 >= Float(0.99)
        }
    }

    /// F.layer_norm(feat, feat.shape[1:], eps=1e-6): one mean and variance over all frames and channels, no affine.
    /// `x` [1, T, D] in the half dtype; statistics in float32, one rounding back.
    public static func utteranceNorm(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let mean = f.mean(axes: [1, 2], keepDims: true), variance = (f - mean).square().mean(axes: [1, 2], keepDims: true)
        return ((f - mean) * rsqrt(variance + 1e-6)).asType(x.dtype)
    }

    // MARK: Windows and overlap-add

    /// torch.hann_window(n) (periodic).
    /// torch.hann_window(n) (periodic), in torch's float32 arithmetic: cos(k · float(2π/n)) · -0.5 + 0.5.
    public static func hann(_ n: Int) -> [Float] {
        let step = Float(2 * Double.pi / Double(n))
        return (0..<n).map { cos(Float($0) * step) * -0.5 + 0.5 }
    }

    /// Overlap-add of `frames` [T, W] at `hop`, W a multiple of hop: output length (T - 1)·hop + W.
    static func overlapAdd(_ frames: MLXArray, hop: Int) -> MLXArray {
        let t = frames.dim(0), w = frames.dim(1), r = w / hop
        let parts = frames.reshaped(t, r, hop)
        var out: MLXArray?
        for j in 0..<r {
            let piece = padded(parts[0..., j, 0...].reshaped(t * hop), widths: [.init((j * hop, (r - 1 - j) * hop))])
            out = out.map { $0 + piece } ?? piece
        }
        return out!
    }

    /// Σ window² over the frames, as overlap-add of the squared window (float32 on the CPU).
    static func envelope(window: [Float], frames: Int, hop: Int) -> [Float] {
        var env = [Float](repeating: 0, count: (frames - 1) * hop + window.count)
        for f in 0..<frames { for i in 0..<window.count { env[f * hop + i] += window[i] * window[i] } }
        return env
    }

    // MARK: Vocoder head

    /// Vocos ISTFTHead after its linear layer (models/vocoder/vocos/head.py): `hidden` [T, n_fft + 2] → magnitude
    /// exp(clip(·, -20, 5)) and phase, irfft(n_fft) × Hann, overlap-add at `hop` with "same" padding and window-envelope
    /// normalisation. Float32 throughout; returns T·hop samples.
    public static func vocosISTFT(_ hidden: MLXArray, nFFT: Int = 1280, hop: Int = 320) -> MLXArray {
        let h = hidden.asType(.float32), bins = nFFT / 2 + 1, t = h.dim(0)
        let mag = exp(clip(h[0..., 0..<bins], min: -20, max: 5)), phase = h[0..., bins..<(2 * bins)]
        let spectrum = (mag * cos(phase)).asType(.complex64) + MLXArray(real: 0, imaginary: 1) * (mag * sin(phase)).asType(.complex64)
        let window = hann(nFFT)
        let frames = MLXFFT.irfft(spectrum, n: nFFT, axis: 1) * MLXArray(window)
        let pad = (nFFT - hop) / 2
        let y = overlapAdd(frames, hop: hop)
        let env = envelope(window: window, frames: t, hop: hop)
        let length = (t - 1) * hop + nFFT - 2 * pad
        return y[pad..<(pad + length)] / MLXArray(Array(env[pad..<(pad + length)]))
    }

    // MARK: PostNet

    public static let postNFFT = 1536, postHop = 768

    /// torch.stft(x, 1536, 768, 1536, hann, center=True, pad_mode="reflect", onesided) → [T, 769] complex64.
    public static func stft(_ x: MLXArray, nFFT: Int = postNFFT, hop: Int = postHop) -> MLXArray {
        let n = x.dim(0), half = nFFT / 2
        precondition(n > half, "STFT input shorter than half a window")
        let f = x.asType(.float32)
        let left = f[1...half][.stride(by: -1)], right = f[(n - 1 - half)..<(n - 1)][.stride(by: -1)]
        let p = concatenated([left, f, right])
        let count = 1 + n / hop
        let frames = asStrided(p, [count, nFFT], strides: [hop, 1]) * MLXArray(hann(nFFT))
        return MLXFFT.rfft(frames, axis: 1)
    }

    /// torch.istft(spec, 1536, 768, 1536, hann, center=True) with no length: hop·(T - 1) samples.
    public static func istft(_ spec: MLXArray, nFFT: Int = postNFFT, hop: Int = postHop) -> MLXArray {
        let t = spec.dim(0), window = hann(nFFT), half = nFFT / 2
        let frames = MLXFFT.irfft(spec, n: nFFT, axis: 1) * MLXArray(window)
        let y = overlapAdd(frames, hop: hop), env = envelope(window: window, frames: t, hop: hop)
        let length = hop * (t - 1)
        return y[half..<(half + length)] / MLXArray(Array(env[half..<(half + length)]))
    }

    /// BandSplit: [T, 769, 2] (real, imaginary) → [T, 257, 6], three 257-bin bands sharing their edge bins.
    public static func bandSplit(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(1) == 769)
        return concatenated([x[0..., 0..<257, 0...], x[0..., 256..<513, 0...], x[0..., 512..<769, 0...]], axis: -1)
    }

    /// BandSplit.inverse: [T, 257, 6] → [T, 769, 2]; the upper bands drop their shared first bin.
    public static func bandMerge(_ x: MLXArray) -> MLXArray {
        concatenated([x[0..., 0..., 0..<2], x[0..., 1..., 2..<4], x[0..., 1..., 4..<6]], axis: 1)
    }

    /// The PostNet's crossfade weight (1 below 7.2 kHz, linear to 0 at 8 kHz): output = reference + (1 - α)·network.
    public static func crossfadeAlpha(bins: Int = 769, rate: Double = 48_000, nFFT: Int = postNFFT) -> [Float] {
        (0..<bins).map { k in
            let f = Float(Double(k) * rate / Double(nFFT))
            if f <= 8000 - 800 { return 1 }
            if f <= 8000 { return (8000 - f) / 800 }
            return 0
        }
    }

    /// TFGridNet.infer around its network, 16 kHz in, `rateOut` out: mean/std normalisation (unbiased std + 1e-12),
    /// resample to 48 kHz, STFT, band split → `network` → merge, crossfade above 7.2 kHz, iSTFT, resample to
    /// `rateOut`, trim or zero-pad to ⌊n·rateOut/16000⌋, rescale by the std (the mean is not restored, as in the maker).
    public static func postNet(_ x16: [Float], rateOut: Int, network: (MLXArray) throws -> MLXArray) rethrows -> [Float] {
        let n = x16.count
        let mean = x16.reduce(0.0) { $0 + Double($1) } / Double(n)
        let variance = x16.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) } / Double(max(n - 1, 1))
        let std = Float(variance.squareRoot()) + 1e-12
        let normalized = x16.map { ($0 - Float(mean)) / std }
        let x48 = resample(normalized, from: 16_000, to: 48_000)
        let spec = stft(MLXArray(x48))
        let ri = stacked([spec.realPart(), spec.imaginaryPart()], axis: -1) // [T, 769, 2]
        let net = bandMerge(try network(bandSplit(ri)).asType(.float32))
        let gain = (1 - MLXArray(crossfadeAlpha())).reshaped(1, 769, 1)
        let out = ri + gain * net
        let complex = out[0..., 0..., 0].asType(.complex64) + MLXArray(real: 0, imaginary: 1) * out[0..., 0..., 1].asType(.complex64)
        var y = istft(complex).asArray(Float.self)
        if rateOut != 48_000 { y = resample(y, from: 48_000, to: rateOut) }
        let target = n * rateOut / 16_000
        if y.count < target { y += [Float](repeating: 0, count: target - y.count) } else { y = Array(y.prefix(target)) }
        return y.map { $0 * std }
    }

    // MARK: Output length and long-form

    /// models/unipase.py: the output holds the input's length at the output rate, ⌊n·rateOut/rateIn⌋.
    public static func outputLength(_ n: Int, rateIn: Int, rateOut: Int) -> Int { rateIn == rateOut ? n : n * rateOut / rateIn }

    /// Trims or zero-pads to `count`.
    public static func fit(_ y: [Float], _ count: Int) -> [Float] {
        y.count >= count ? Array(y.prefix(count)) : y + [Float](repeating: 0, count: count - y.count)
    }

    /// The maker's inference scripts: rescale so the output's peak equals the input's.
    public static func matchPeak(_ y: [Float], to x: [Float]) -> [Float] {
        let peakIn = x.reduce(0) { max($0, abs($1)) }, peakOut = y.reduce(0) { max($0, abs($1)) }
        let scale = peakIn / (peakOut + 1e-8)
        return y.map { $0 * scale }
    }

    /// inference_long.py: 8 s segments at a 4 s hop. When the next segment would run past the clip by less than 2 s
    /// (at least 6 s of clip left), it is kept, cut at the clip's end; when it would run past by 2 s or more, it is
    /// dropped and the previous segment is extended to the clip's end instead. Returns [start, end) sample ranges.
    /// The maker fails for clips the loop leaves with no segment or one; this port then uses one segment, the clip.
    public static func longFormSegments(count: Int, rate: Int, segment: Int = 8, hop: Int = 4) -> [Range<Int>] {
        let seg = rate * segment, step = rate * hop
        var patches: [Range<Int>] = []
        var start = 0
        while start < count {
            let end = start + seg
            if end > count {
                if end - count < 2 * rate {
                    patches.append(start..<count)
                } else if let last = patches.popLast() {
                    patches.append(last.lowerBound..<count)
                }
                break
            }
            patches.append(start..<end)
            start += step
        }
        return patches.isEmpty ? [0..<count] : patches
    }

    /// The maker's stitch of enhanced segments: the first keeps its first 6 s, middle ones 2 s…6 s, the last from 2 s
    /// on. `rate` is the rate of the enhanced segments (the maker uses the input rate here, which is right only when
    /// the output rate equals it).
    public static func stitch(_ parts: [[Float]], rate: Int, segment: Int = 8, hop: Int = 4) -> [Float] {
        guard parts.count > 1 else { return parts.first ?? [] }
        let edge = (segment - hop) / 2 * rate, keep = (segment - (segment - hop) / 2) * rate
        var out = Array(parts[0].prefix(keep))
        for part in parts.dropFirst().dropLast() { out += part[min(edge, part.count)..<min(keep, part.count)] }
        out += parts.last!.dropFirst(edge)
        return out
    }
}
