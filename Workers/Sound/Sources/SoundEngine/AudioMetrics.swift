// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
public enum AudioMetrics {
    /// SHA-256 of the samples as 16-bit little-endian PCM (clamped, rounded): exact agreement without caring about
    /// float noise below one 16-bit step.
    public static func pcmSHA256(_ samples: [Float]) -> String {
        var bytes = [UInt8](); bytes.reserveCapacity(samples.count * 2)
        for x in samples {
            let v = Int16(max(-32768, min(32767, (Double(x.isFinite ? x : 0) * 32767).rounded())))
            let u = UInt16(bitPattern: v)
            bytes.append(UInt8(u & 0xFF)); bytes.append(UInt8(u >> 8))
        }
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }

    /// Lowercased words: letters and digits kept, apostrophes dropped ("Blair's" → "blairs"), everything else splits.
    public static func words(_ text: String) -> [String] {
        var out: [String] = [], current = ""
        for ch in text.lowercased() {
            if ch.isLetter || ch.isNumber { current.append(ch) }
            else if ch == "'" || ch == "\u{2019}" { continue }
            else if !current.isEmpty { out.append(current); current = "" }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    /// Word error rate: (substitutions + deletions + insertions) / reference words.
    public static func wordErrorRate(reference: String, hypothesis: String) -> Double {
        let r = words(reference), h = words(hypothesis)
        guard !r.isEmpty else { return h.isEmpty ? 0 : 1 }
        var previous = Array(0...h.count), row = [Int](repeating: 0, count: h.count + 1)
        for i in 1...r.count {
            row[0] = i
            for j in stride(from: 1, through: h.count, by: 1) {
                row[j] = min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (r[i - 1] == h[j - 1] ? 0 : 1))
            }
            swap(&previous, &row)
        }
        return Double(previous[h.count]) / Double(r.count)
    }

    /// Scale-invariant signal-to-noise ratio of `estimate` against `reference` (mono, same rate), in dB, over the
    /// common length, within ±`99.0`. Identical signals give +99; nothing to compare (no
    /// overlap, a non-finite sample, a silent or constant estimate or reference) gives −99, never a non-finite number.
    public static func siSNR(estimate: [Float], reference: [Float]) -> Double {
        let worst = -99.0
        let n = min(estimate.count, reference.count)
        guard n > 0 else { return worst }
        let e = estimate.prefix(n).map(Double.init), s = reference.prefix(n).map(Double.init)
        guard e.allSatisfy(\.isFinite), s.allSatisfy(\.isFinite) else { return worst }
        let me = e.reduce(0, +) / Double(n), ms = s.reduce(0, +) / Double(n)
        let ec = e.map { $0 - me }, sc = s.map { $0 - ms }
        let dot = zip(ec, sc).reduce(0) { $0 + $1.0 * $1.1 }
        let energy = sc.reduce(0) { $0 + $1 * $1 }
        let estimateEnergy = ec.reduce(0) { $0 + $1 * $1 }
        // Zero signal is not zero error: a silent or constant estimate has nothing in common with the reference.
        guard energy > 0, estimateEnergy > 0 else { return worst }
        let target = sc.map { $0 * dot / energy }
        let signal = target.reduce(0) { $0 + $1 * $1 }
        let noise = zip(ec, target).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
        guard signal > 0 else { return worst }
        guard noise > 0 else { return 99.0 }
        return max(worst, min(99.0, 10 * log10(signal / noise)))
    }

    /// Why generated audio is unusable, or nil: no samples, a non-finite sample, or silence/a constant level (no
    /// sample differs from the mean by 1/32768, below one 16-bit step).
    public static func signalProblem(_ samples: [Float]) -> String? {
        guard !samples.isEmpty else { return "no audio" }
        guard samples.allSatisfy(\.isFinite) else { return "non-finite samples in the output" }
        let mean = samples.reduce(0.0) { $0 + Double($1) } / Double(samples.count)
        let peak = samples.reduce(0.0) { max($0, abs(Double($1) - mean)) }
        return peak < 1.0 / 32768 ? "silent or constant output" : nil
    }

    /// Mono mix of interleaved samples.
    public static func mono(_ samples: [Float], channels: Int) -> [Float] {
        guard channels > 1 else { return samples }
        return stride(from: 0, to: samples.count - channels + 1, by: channels).map { i in
            (0..<channels).reduce(Float(0)) { $0 + samples[i + $1] } / Float(channels)
        }
    }

    /// Log-mel spectrogram (dB, 0.1 dB steps, floor 80 dB below the peak) of the mono mix: Hann window, `fft`-point
    /// frames every `hop` samples, `bands` triangular mel bands from 20 Hz to min(16 kHz, Nyquist).
    public static func logMel(_ samples: [Float], channels: Int = 1, sampleRate: Int, bands: Int = 32, fft: Int = 2048, hop: Int = 2048) -> LogMel {
        let x = mono(samples, channels: channels)
        let frames = x.count >= fft ? 1 + (x.count - fft) / hop : 0
        let bins = fft / 2 + 1
        let window = (0..<fft).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fft)) }
        let filters = melFilters(bands: bands, fft: fft, sampleRate: sampleRate)
        var power = [Double](repeating: 0, count: frames * bands)
        var re = [Double](repeating: 0, count: fft), im = [Double](repeating: 0, count: fft)
        for f in 0..<frames {
            for i in 0..<fft { re[i] = Double(x[f * hop + i]) * window[i]; im[i] = 0 }
            FFT.transform(&re, &im)
            for b in 0..<bands {
                var sum = 0.0
                for (k, w) in filters[b] where k < bins { sum += w * (re[k] * re[k] + im[k] * im[k]) }
                power[f * bands + b] = sum
            }
        }
        var db = power.map { 10 * log10(max($0, 1e-10)) }
        let peak = db.max() ?? 0
        db = db.map { (max($0, peak - 80) * 10).rounded() / 10 }
        return LogMel(bands: bands, frames: frames, db: db.map(Float.init))
    }

    /// Mean absolute difference in dB over the common frames; nil when the band counts differ or nothing overlaps.
    public static func melDistance(_ a: LogMel, _ b: LogMel) -> Double? {
        guard a.bands == b.bands, a.bands > 0 else { return nil }
        let frames = min(a.frames, b.frames)
        guard frames > 0 else { return nil }
        var sum = 0.0
        for i in 0..<(frames * a.bands) { sum += abs(Double(a.db[i]) - Double(b.db[i])) }
        return sum / Double(frames * a.bands)
    }

    static func melFilters(bands: Int, fft: Int, sampleRate: Int) -> [[(Int, Double)]] {
        func mel(_ hz: Double) -> Double { 2595 * log10(1 + hz / 700) }
        func hz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let low = mel(20), high = mel(min(16000, Double(sampleRate) / 2))
        let points = (0...(bands + 1)).map { hz(low + (high - low) * Double($0) / Double(bands + 1)) }
        let binHz = Double(sampleRate) / Double(fft)
        return (0..<bands).map { b in
            let (l, c, r) = (points[b], points[b + 1], points[b + 2])
            var weights: [(Int, Double)] = []
            for k in Int((l / binHz).rounded(.down))...Int((r / binHz).rounded(.up)) {
                let f = Double(k) * binHz
                let w = f <= c ? (f - l) / max(c - l, 1e-9) : (r - f) / max(r - c, 1e-9)
                if w > 0 { weights.append((k, w)) }
            }
            return weights
        }
    }
}

/// A log-mel spectrogram, frames × bands, row-major. `db` is stored as one comma-separated string (0.1 dB steps) so a
/// spectrum is one line of reference.json; an array is accepted too.
public struct LogMel: Codable, Equatable, Sendable {
    public var bands: Int
    public var frames: Int
    public var db: [Float]
    public init(bands: Int, frames: Int, db: [Float]) { self.bands = bands; self.frames = frames; self.db = db }
    enum CodingKeys: String, CodingKey { case bands, frames, db }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        bands = try c.decode(Int.self, forKey: .bands); frames = try c.decode(Int.self, forKey: .frames)
        if let text = try? c.decode(String.self, forKey: .db) {
            db = try text.split(separator: ",").map {
                guard let v = Float($0.trimmingCharacters(in: .whitespaces)) else {
                    throw DecodingError.dataCorruptedError(forKey: .db, in: c, debugDescription: "not a number: \($0)")
                }
                return v
            }
        } else { db = try c.decode([Float].self, forKey: .db) }
        guard db.count == bands * frames else { throw DecodingError.dataCorruptedError(forKey: .db, in: c, debugDescription: "db has \(db.count) values, expected \(bands * frames)") }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(bands, forKey: .bands); try c.encode(frames, forKey: .frames)
        try c.encode(db.map { String(format: "%.1f", $0) }.joined(separator: ","), forKey: .db)
    }
}

/// In-place iterative radix-2 FFT (power-of-two length).
enum FFT {
    static func transform(_ re: inout [Double], _ im: inout [Double]) {
        let n = re.count
        precondition(n > 0 && n & (n - 1) == 0, "FFT length must be a power of two")
        var j = 0
        for i in 1..<n {
            var bit = n >> 1
            while j & bit != 0 { j ^= bit; bit >>= 1 }
            j |= bit
            if i < j { re.swapAt(i, j); im.swapAt(i, j) }
        }
        var size = 2
        while size <= n {
            let angle = -2 * Double.pi / Double(size)
            let (wr, wi) = (cos(angle), sin(angle))
            for start in stride(from: 0, to: n, by: size) {
                var (cr, ci) = (1.0, 0.0)
                for k in 0..<(size / 2) {
                    let a = start + k, b = a + size / 2
                    let tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr
                    re[b] = re[a] - tr; im[b] = im[a] - ti
                    re[a] += tr; im[a] += ti
                    (cr, ci) = (cr * wr - ci * wi, cr * wi + ci * wr)
                }
            }
            size <<= 1
        }
    }
}
