import Accelerate
import Foundation

/// Objective checks for a rendered clip (no listening): length, finite samples,
/// peak/clipping, loudness, silent stretches, channel sanity.
public struct SA3AudioStats: Sendable, CustomStringConvertible {
    public var frames: Int
    public var sampleRate: Int
    public var seconds: Double { Double(frames) / Double(sampleRate) }
    public var nonFinite: Int
    public var peak: Float
    public var clippedFraction: Double       // |x| ≥ 0.999
    public var rmsDBFS: Double
    public var longestSilenceSeconds: Double // 50 ms windows below -60 dBFS
    public var silentFraction: Double
    public var channelCorrelation: Double    // L/R Pearson; 1.0 = dual mono
    public var dcOffset: Double
    /// Mean spectral flatness (geometric / arithmetic mean of the power spectrum,
    /// 2048-sample Hann frames of the mid channel, non-silent frames only).
    /// White noise ≈ 0.5–1; music and most effects well below 0.3.
    public var spectralFlatness: Double

    public var description: String {
        String(format: "%.3f s · peak %.3f · rms %.1f dBFS · clipped %.4f%% · silent %.0f%% (longest %.2f s) · L/R corr %.3f · dc %.4f · flatness %.3f · non-finite %d",
               seconds, peak, rmsDBFS, clippedFraction * 100, silentFraction * 100, longestSilenceSeconds,
               channelCorrelation, dcOffset, spectralFlatness, nonFinite)
    }

    public init(left: [Float], right: [Float], sampleRate: Int) {
        frames = left.count
        self.sampleRate = sampleRate
        var nf = 0, clipped = 0
        var pk: Float = 0, sumSq = 0.0, sum = 0.0
        var sl = 0.0, sr = 0.0, sll = 0.0, srr = 0.0, slr = 0.0
        for i in 0..<left.count {
            let l = left[i], r = right[i]
            if !l.isFinite || !r.isFinite { nf += 1; continue }
            let al = abs(l), ar = abs(r)
            pk = max(pk, max(al, ar))
            if al >= 0.999 { clipped += 1 }
            if ar >= 0.999 { clipped += 1 }
            let dl = Double(l), dr = Double(r)
            sumSq += dl * dl + dr * dr; sum += dl + dr
            sl += dl; sr += dr; sll += dl * dl; srr += dr * dr; slr += dl * dr
        }
        nonFinite = nf
        peak = pk
        let n = Double(max(1, 2 * left.count))
        clippedFraction = Double(clipped) / n
        rmsDBFS = 20 * log10(max(1e-12, (sumSq / n).squareRoot()))
        dcOffset = sum / n
        let m = Double(max(1, left.count))
        let cov = slr / m - (sl / m) * (sr / m)
        let vl = sll / m - (sl / m) * (sl / m), vr = srr / m - (sr / m) * (sr / m)
        channelCorrelation = (vl > 0 && vr > 0) ? cov / (vl * vr).squareRoot() : 1
        let win = max(1, sampleRate / 20)
        var run = 0, longest = 0, silent = 0, windows = 0
        var i = 0
        while i + win <= left.count {
            var e = 0.0
            for j in i..<(i + win) where left[j].isFinite && right[j].isFinite {
                e += Double(left[j] * left[j] + right[j] * right[j])
            }
            let db = 10 * log10(max(1e-24, e / Double(2 * win)))
            windows += 1
            if db < -60 { silent += 1; run += 1; longest = max(longest, run) } else { run = 0 }
            i += win
        }
        longestSilenceSeconds = Double(longest * win) / Double(sampleRate)
        silentFraction = windows > 0 ? Double(silent) / Double(windows) : 1
        spectralFlatness = Self.flatness(left: left, right: right)
    }

    static func flatness(left: [Float], right: [Float]) -> Double {
        let n = 2048, log2n = vDSP_Length(11)
        guard left.count >= n, let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return 0 }
        defer { vDSP_destroy_fftsetup(setup) }
        var window = [Float](repeating: 0, count: n)
        vDSP_hann_window(&window, vDSP_Length(n), Int32(vDSP_HANN_NORM))
        var total = 0.0, frames = 0
        var start = 0
        while start + n <= left.count {
            var frame = (0..<n).map { i -> Float in
                let v = 0.5 * (left[start + i] + right[start + i]); return v.isFinite ? v * window[i] : 0
            }
            var energy: Float = 0
            vDSP_svesq(frame, 1, &energy, vDSP_Length(n))
            if energy / Float(n) > 1e-8 {
                var re = [Float](repeating: 0, count: n / 2), im = re
                re.withUnsafeMutableBufferPointer { rp in
                    im.withUnsafeMutableBufferPointer { ip in
                        var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                        frame.withUnsafeBytes { raw in
                            vDSP_ctoz(raw.bindMemory(to: DSPComplex.self).baseAddress!, 2, &split, 1, vDSP_Length(n / 2))
                        }
                        vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                    }
                }
                // bins 1..n/2-1 (skip DC / packed Nyquist)
                var logSum = 0.0, sum = 0.0
                for k in 1..<(n / 2) {
                    let p = Double(re[k] * re[k] + im[k] * im[k]) + 1e-20
                    logSum += log(p); sum += p
                }
                let m = Double(n / 2 - 1)
                total += exp(logSum / m) / (sum / m)
                frames += 1
            }
            start += n * 4     // every 4th frame is plenty
        }
        return frames > 0 ? total / Double(frames) : 0
    }

    /// Failures for a render that asked for `expectedSeconds`; empty = pass.
    public func problems(expectedSeconds: Double) -> [String] {
        var p: [String] = []
        let want = Int((expectedSeconds * Double(sampleRate)).rounded())
        if frames != want { p.append("length \(frames) frames, expected \(want)") }
        if nonFinite > 0 { p.append("\(nonFinite) non-finite frames") }
        if peak < 0.01 { p.append(String(format: "near-silent (peak %.4f)", peak)) }
        if rmsDBFS < -50 { p.append(String(format: "too quiet (rms %.1f dBFS)", rmsDBFS)) }
        if clippedFraction > 0.001 { p.append(String(format: "clipping %.3f%% of samples", clippedFraction * 100)) }
        if silentFraction > 0.5 { p.append(String(format: "%.0f%% silent windows", silentFraction * 100)) }
        if abs(dcOffset) > 0.05 { p.append(String(format: "dc offset %.3f", dcOffset)) }
        if spectralFlatness > 0.5 { p.append(String(format: "noise-like spectrum (flatness %.2f)", spectralFlatness)) }
        return p
    }
}

/// 16-bit PCM WAV (stereo interleaved), peak-safe: clamps to [-1, 1] without normalising.
public func sa3WriteWAV(_ url: URL, left: [Float], right: [Float], sampleRate: Int = SA3Audio.sampleRate) throws {
    let frames = left.count
    var d = Data(capacity: 44 + frames * 4)
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + frames * 4))
    d.append(contentsOf: Array("WAVE".utf8)); d.append(contentsOf: Array("fmt ".utf8))
    u32(16); u16(1); u16(2); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 4)); u16(4); u16(16)
    d.append(contentsOf: Array("data".utf8)); u32(UInt32(frames * 4))
    var pcm = [Int16](repeating: 0, count: frames * 2)
    for i in 0..<frames {
        let l = left[i].isFinite ? left[i] : 0, r = right[i].isFinite ? right[i] : 0
        pcm[2 * i] = Int16((max(-1, min(1, l)) * 32767).rounded())
        pcm[2 * i + 1] = Int16((max(-1, min(1, r)) * 32767).rounded())
    }
    pcm.withUnsafeBytes { d.append(contentsOf: $0) }
    try d.write(to: url, options: .atomic)
}
