// SPDX-License-Identifier: MIT
// Adapted from VireoCore/Loudness.swift; BS.1770 meter, downward-only playback gain.
import Accelerate
import Foundation

/// ITU-R BS.1770-4 loudness and true peak, and the level Vireo applies to speech it plays itself.
///
/// Measurement: K-weighting (high-shelf pre-filter + RLB high-pass, derived for the clip's own sample rate
/// from the analogue prototypes so 48 kHz reproduces the coefficients printed in BS.1770), mean square per
/// channel over 400 ms blocks with 75 % overlap, channel weights (L, R, C 1.0; Ls, Rs 1.41; LFE excluded),
/// absolute gate -70 LUFS, relative gate -10 LU. True peak: 4x oversampling as in BS.1770-4 Annex 2, with a
/// flatter interpolation filter than the Annex's example (see `PlaybackTruePeakFilter`); relative to the sample rate.
///
/// Playback level: one gain per clip so it lands at about `target` (-16 LUFS), reduced when needed so the
/// true peak stays at or below `ceiling` (-1 dBTP). No compression: every sample gets the same gain.
/// Files and API responses are never levelled; only audio the helper plays through the speakers.
public enum PlaybackLoudness {
    public static let target = -16.0
    public static let ceiling = -1.0
    static let absoluteGate = -70.0
    static let relativeGate = -10.0

    public struct Measurement: Sendable, Equatable {
        /// Integrated loudness in LUFS; -infinity when no block passes the gates (silence, or shorter than 400 ms).
        public var integrated: Double
        /// True peak in dBTP (4x oversampled, never below the sample peak); -infinity for digital silence.
        public var truePeak: Double
        public var seconds: Double
    }

    /// Integrated loudness and true peak of interleaved samples.
    public static func measure(_ samples: [Float], rate: Int, channels: Int = 1, weights: [Double]? = nil) -> Measurement {
        var meter = PlaybackLoudnessMeter(rate: rate, channels: channels, weights: weights)
        meter.append(samples)
        return meter.measurement
    }

    /// BS.1770 channel weights by position for 1-3 (L, R, C), 5 (L, R, C, Ls, Rs) and 6 (L, R, C, LFE, Ls, Rs) channels.
    public static func channelWeights(_ channels: Int) -> [Double] {
        switch channels {
        case 5: return [1, 1, 1, 1.41, 1.41]
        case 6: return [1, 1, 1, 0, 1.41, 1.41]
        default: return Array(repeating: 1, count: max(1, channels))
        }
    }

    /// The single gain (dB) that brings a clip of this loudness and true peak to `target`, lowered so the
    /// peak stays at or below `ceiling`, and never raised above its original level. Unmeasurable loudness gets no boost.
    public static func gainDB(loudness: Double, truePeak: Double, target: Double = target, ceiling: Double = ceiling) -> Double {
        var gain = loudness.isFinite ? min(target - loudness, 0) : 0
        if truePeak.isFinite { gain = min(gain, ceiling - truePeak) }
        return gain
    }

    public static func scaled(_ samples: [Float], db: Double) -> [Float] {
        guard db != 0 else { return samples }
        let factor = Float(pow(10, db / 20))
        return samples.map { $0 * factor }
    }

    static func db(power: Double) -> Double { power > 0 ? -0.691 + 10 * log10(power) : -.infinity }
    static func db(amplitude: Double) -> Double { amplitude > 0 ? 20 * log10(amplitude) : -.infinity }
}

/// One second-order section, a0 normalised to 1.
public struct PlaybackBiquad: Sendable, Equatable {
    public var b0, b1, b2, a1, a2: Double

    /// The K-weighting stages for `rate` (BS.1770-4, via the bilinear transform of the analogue prototypes).
    public static func kWeighting(rate: Int) -> (shelf: PlaybackBiquad, highPass: PlaybackBiquad) {
        let fs = Double(rate)
        // Stage 1: high-shelf, +4 dB above about 1.7 kHz (head acoustics).
        var f0 = 1681.974450955533, q = 0.7071752369554196
        let gain = 3.999843853973347
        var k = tan(Double.pi * f0 / fs)
        let vh = pow(10, gain / 20), vb = pow(vh, 0.4996667741545416)
        var a0 = 1 + k / q + k * k
        let shelf = PlaybackBiquad(
            b0: (vh + vb * k / q + k * k) / a0, b1: 2 * (k * k - vh) / a0, b2: (vh - vb * k / q + k * k) / a0,
            a1: 2 * (k * k - 1) / a0, a2: (1 - k / q + k * k) / a0)
        // Stage 2: RLB high-pass at about 38 Hz.
        f0 = 38.13547087602444; q = 0.5003270373238773
        k = tan(Double.pi * f0 / fs)
        a0 = 1 + k / q + k * k
        let highPass = PlaybackBiquad(b0: 1, b1: -2, b2: 1, a1: 2 * (k * k - 1) / a0, a2: (1 - k / q + k * k) / a0)
        return (shelf, highPass)
    }
}

/// Streaming BS.1770 meter: feed interleaved chunks in order; read integrated loudness and true peak at any time.
public struct PlaybackLoudnessMeter: Sendable {
    public let rate: Int
    public let channels: Int
    let weights: [Double]
    private let filter: PlaybackKWeightingFilter
    /// Per channel: the vDSP biquad delay line (2 sections) and the last `PlaybackTruePeakFilter.taps - 1` input samples.
    private var delays: [[Double]]
    private var history: [[Double]]
    /// Weighted sum of squared K-weighted samples per completed 100 ms segment, and its frame count.
    private var segments: [Double] = []
    private var segmentFrames: [Int] = []
    private var current = 0.0
    private var currentFrames = 0
    private var frames = 0
    private var peak = 0.0

    public init(rate: Int, channels: Int = 1, weights: [Double]? = nil) {
        self.rate = max(1, rate)
        self.channels = max(1, channels)
        let w = weights ?? PlaybackLoudness.channelWeights(self.channels)
        self.weights = w.count == self.channels ? w : PlaybackLoudness.channelWeights(self.channels)
        filter = PlaybackKWeightingFilter(rate: self.rate)
        delays = Array(repeating: Array(repeating: 0, count: 6), count: self.channels)
        history = Array(repeating: Array(repeating: 0, count: PlaybackTruePeakFilter.taps - 1), count: self.channels)
    }

    public var seconds: Double { Double(frames) / Double(rate) }
    public var measurement: PlaybackLoudness.Measurement { .init(integrated: integrated, truePeak: truePeak, seconds: seconds) }
    /// True peak of everything appended so far, as if the input ended here: includes the interpolated points after
    /// the last sample (the filter's lookahead, finished with the silence that follows the clip).
    public var truePeak: Double { PlaybackLoudness.db(amplitude: max(peak, tailPeak)) }

    /// The interpolated points whose filter span reaches past the last sample, with zeros after it.
    private var tailPeak: Double {
        let keep = PlaybackTruePeakFilter.taps - 1
        guard frames > 0 else { return 0 }
        var top = 0.0
        var y = [Double](repeating: 0, count: keep)
        for c in 0..<channels {
            let x = history[c] + [Double](repeating: 0, count: keep)
            for phase in PlaybackTruePeakFilter.phases {
                vDSP_convD(x, 1, phase, 1, &y, 1, vDSP_Length(keep), vDSP_Length(PlaybackTruePeakFilter.taps))
                var m = 0.0
                vDSP_maxmgvD(y, 1, &m, vDSP_Length(keep))
                top = max(top, m)
            }
        }
        return top
    }

    /// First sample of 100 ms segment `k` (exact for sample rates divisible by 10).
    private func segmentStart(_ k: Int) -> Int { k * rate / 10 }

    /// Appends interleaved samples; returns the true peak (linear: the larger of the 4x-interpolated and sample peak) in this
    /// chunk. The interpolated points near the chunk's end need later samples: they count with the next chunk (or `truePeak`).
    @discardableResult
    public mutating func append(_ samples: [Float]) -> Double {
        let n = samples.count / channels
        guard n > 0 else { return 0 }
        let count = vDSP_Length(n)
        let keep = PlaybackTruePeakFilter.taps - 1
        var power = [Double](repeating: 0, count: n)
        var x = [Double](repeating: 0, count: keep + n)
        var y = [Double](repeating: 0, count: n)
        var chunkPeak = 0.0
        for c in 0..<channels {
            x.withUnsafeMutableBufferPointer { xb in
                samples.withUnsafeBufferPointer { vDSP_vspdp($0.baseAddress! + c, vDSP_Stride(channels), xb.baseAddress! + keep, 1, count) }
            }
            x.replaceSubrange(0..<keep, with: history[c])
            var weight = weights[c]
            let setup = filter.setup
            let top: Double = x.withUnsafeBufferPointer { xb in
                y.withUnsafeMutableBufferPointer { yb in
                    power.withUnsafeMutableBufferPointer { pb in
                        let xs = xb.baseAddress!, ys = yb.baseAddress!, ps = pb.baseAddress!
                        // K-weighting: the two biquads, state carried across chunks; then weighted power.
                        delays[c].withUnsafeMutableBufferPointer { vDSP_biquadD(setup, $0.baseAddress!, xs + keep, 1, ys, 1, count) }
                        vDSP_vsqD(ys, 1, ys, 1, count)
                        vDSP_vsmaD(ys, 1, &weight, ps, 1, ps, 1, count)
                        // True peak: the sample peak and three interpolated points between samples (4x).
                        var top = 0.0
                        vDSP_maxmgvD(xs + keep, 1, &top, count)
                        for phase in PlaybackTruePeakFilter.phases {
                            vDSP_convD(xs, 1, phase, 1, ys, 1, count, vDSP_Length(PlaybackTruePeakFilter.taps))
                            var m = 0.0
                            vDSP_maxmgvD(ys, 1, &m, count)
                            top = max(top, m)
                        }
                        return top
                    }
                }
            }
            chunkPeak = max(chunkPeak, top)
            history[c] = Array(x[n...])
        }
        // Sum the weighted power into 100 ms segments.
        var i = 0
        while i < n {
            let end = min(n, i + segmentStart(segments.count + 1) - frames)
            var sum = 0.0
            power.withUnsafeBufferPointer { vDSP_sveD($0.baseAddress! + i, 1, &sum, vDSP_Length(end - i)) }
            current += sum; currentFrames += end - i; frames += end - i; i = end
            if frames == segmentStart(segments.count + 1) {
                segments.append(current); segmentFrames.append(currentFrames); current = 0; currentFrames = 0
            }
        }
        peak = max(peak, chunkPeak)
        return chunkPeak
    }

    /// BS.1770 gated integrated loudness over 400 ms blocks every 100 ms (a partial final block is not counted).
    public var integrated: Double { Self.gated(blocks(span: 4)) }

    /// A quick estimate for a first chunk of speech that may be shorter than one block: the same gates applied to
    /// 100 ms segments (the unfinished one counts from 50 ms), so leading silence does not dilute it.
    public var estimate: Double {
        var blocks = blocks(span: 1)
        if currentFrames * 20 >= rate { blocks.append(current / Double(currentFrames)) }
        return Self.gated(blocks)
    }

    /// Mean weighted power of each block of `span` consecutive segments.
    private func blocks(span: Int) -> [Double] {
        guard segments.count >= span else { return [] }
        var out: [Double] = []; out.reserveCapacity(segments.count - span + 1)
        var sum = segments[0..<span].reduce(0, +), count = segmentFrames[0..<span].reduce(0, +)
        for j in 0...(segments.count - span) {
            if j > 0 { sum += segments[j + span - 1] - segments[j - 1]; count += segmentFrames[j + span - 1] - segmentFrames[j - 1] }
            out.append(count > 0 ? max(0, sum) / Double(count) : 0)
        }
        return out
    }

    static func gated(_ blocks: [Double]) -> Double {
        let absolute = blocks.filter { PlaybackLoudness.db(power: $0) > PlaybackLoudness.absoluteGate }
        guard !absolute.isEmpty else { return -.infinity }
        let threshold = PlaybackLoudness.db(power: absolute.reduce(0, +) / Double(absolute.count)) + PlaybackLoudness.relativeGate
        let relative = absolute.filter { PlaybackLoudness.db(power: $0) > threshold }
        guard !relative.isEmpty else { return -.infinity }
        return PlaybackLoudness.db(power: relative.reduce(0, +) / Double(relative.count))
    }
}

/// The K-weighting cascade as a vDSP biquad setup (immutable once made; delay lines live with each channel).
final class PlaybackKWeightingFilter: @unchecked Sendable {
    let setup: vDSP_biquad_SetupD
    init(rate: Int) {
        let (s, h) = PlaybackBiquad.kWeighting(rate: rate)
        setup = vDSP_biquad_CreateSetupD([s.b0, s.b1, s.b2, s.a1, s.a2, h.b0, h.b1, h.b2, h.a1, h.a2], 2)!
    }
    deinit { vDSP_biquad_DestroySetupD(setup) }
}

/// 4x oversampling for true peak (BS.1770-4 Annex 2 method). The Annex's example 48-tap filter has up to
/// +/-0.24 dB passband ripple, which reads a sine at fs/6 0.21 dB high, outside EBU Tech 3341's +0.2 dB; this
/// Kaiser-windowed sinc (24 taps per phase, beta 7) stays within 0.01 dB up to 0.4 fs. Phase 0 is the sample
/// itself; the other three interpolate at 1/4, 1/2 and 3/4 of a sample, 12 samples behind the newest input.
enum PlaybackTruePeakFilter {
    static let taps = 24
    /// Coefficients for vDSP_convD: index j multiplies the sample `taps - 1 - j` steps before the newest.
    static let phases: [[Double]] = [0.25, 0.5, 0.75].map { tau in
        let half = Double(taps / 2), beta = 7.0
        func i0(_ x: Double) -> Double { // modified Bessel function of the first kind, order 0
            var sum = 1.0, term = 1.0, k = 1.0
            while term > 1e-12 * sum { term *= (x / (2 * k)) * (x / (2 * k)); sum += term; k += 1 }
            return sum
        }
        let kernel = (0..<taps).map { i -> Double in
            let t = tau - half + Double(i) // tap i steps back from the newest sample
            let sinc = t == 0 ? 1 : sin(Double.pi * t) / (Double.pi * t)
            return sinc * i0(beta * (max(0, 1 - (t / half) * (t / half))).squareRoot()) / i0(beta)
        }
        let total = kernel.reduce(0, +)
        return kernel.reversed().map { $0 / total }
    }
}
