// SPDX-License-Identifier: MIT
// AudioSR's signal processing, matching the reference package (audiosr 0.0.7: utils.py, lowpass.py, ddpm.py
// post-processing) with SciPy 1.17 / librosa 0.9.2 semantics. Filter design, zero-phase filtering and polyphase
// resampling run on the CPU in Float64, as SciPy does.
import Foundation
import MLX
import MLXFFT

enum AudioSRDSP {
    static let rate = 48_000

    // MARK: Filter type: numpy's legacy RandomState after np.random.seed(seed), then np.random.choice(4 types)

    enum Filter: Int, CaseIterable { case butter, cheby1, ellip, bessel }

    /// numpy 2.x `RandomState(seed).choice(["butter", "cheby1", "ellip", "bessel"])`: MT19937 `init_genrand(seed)`,
    /// then a masked bounded draw (mask 3, every draw accepted), i.e. the first 32-bit output & 3.
    static func filter(seed: UInt32) -> Filter {
        var state = [UInt32](repeating: 0, count: 624)
        state[0] = seed
        for i in 1..<624 { state[i] = 1_812_433_253 &* (state[i - 1] ^ (state[i - 1] >> 30)) &+ UInt32(i) }
        // First twist, then temper output 0.
        func twisted(_ i: Int) -> UInt32 {
            let y = (state[i] & 0x8000_0000) | (state[(i + 1) % 624] & 0x7FFF_FFFF)
            return state[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) == 0 ? 0 : 0x9908_B0DF)
        }
        var y = twisted(0)
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= y >> 18
        return Filter(rawValue: Int(y & 3))!
    }

    // MARK: Low-pass design (order 8), SciPy iirfilter: prototype → lp2lp_zpk → bilinear_zpk → second-order sections

    struct Complex { var re: Double, im: Double }
    /// Analog prototypes from SciPy 1.17: buttap(8), cheb1ap(8, 0.1), ellipap(8, 0.1, 60), besselap(8, norm="phase").
    /// Upper-half-plane members; each has its conjugate.
    static func prototype(_ filter: Filter) -> (zeros: [Double], poles: [Complex], gain: Double) {
        switch filter {
        case .butter:
            return (
                [],
                [
                    (-0.19509032201612833, 0.9807852804032304), (-0.5555702330196023, 0.8314696123025452), (-0.8314696123025452, 0.5555702330196022),
                    (-0.9807852804032304, 0.19509032201612825)
                ].map(Complex.init), 1
            )
        case .cheby1:
            return (
                [],
                [
                    (-0.06398012430906706, 1.0321813618402467), (-0.1821999789824168, 0.8750411062474142), (-0.27268153850261595, 0.5846837744957937),
                    (-0.3216498056500937, 0.20531363824880286)
                ].map(Complex.init),
                0.05118908763127163
            )
        case .ellip:
            // Zeros are purely imaginary (±j·w).
            return (
                [4.5746269031186655, 1.7541181361606806, 1.309585699523888, 1.1914390300529494],
                [
                    (-0.42595527318370724, 0.3009196664683167), (-0.25985772870761314, 0.742355698261655), (-0.114085062727376, 0.9466995511242549),
                    (-0.030326559750565765, 1.0176451518280614)
                ].map(Complex.init),
                0.0009999999999999994
            )
        case .bessel:
            return (
                [],
                [
                    (-0.4621740412532124, 1.0343886811269012), (-0.71113818084854, 0.7186517314108402), (-0.8473250802359336, 0.4259017538272935),
                    (-0.9096831546652911, 0.14124379766714232)
                ].map(Complex.init), 1
            )
        }
    }

    /// Four biquads [b0, b1, b2, a0=1, a1, a2] for a digital low-pass at `wn` (fraction of Nyquist).
    static func sections(_ filter: Filter, wn: Double) -> [[Double]] {
        let (zeros, poles, gain) = prototype(filter)
        let warped = 4 * tan(Double.pi * wn / 2)
        func bilinear(_ re: Double, _ im: Double) -> Complex {
            // (4 + s) / (4 − s)
            let (nr, ni, dr, di) = (4 + re, im, 4 - re, -im)
            let d = dr * dr + di * di
            return Complex(re: (nr * dr + ni * di) / d, im: (ni * dr - nr * di) / d)
        }
        func product(_ values: [Complex]) -> Complex {
            values.reduce(Complex(re: 1, im: 0)) { Complex(re: $0.re * $1.re - $0.im * $1.im, im: $0.re * $1.im + $0.im * $1.re) }
        }
        // Full conjugate sets after lp2lp (s → s / warped).
        let analogPoles = poles.flatMap { [Complex(re: $0.re * warped, im: $0.im * warped), Complex(re: $0.re * warped, im: -$0.im * warped)] }
        let analogZeros = zeros.flatMap { [Complex(re: 0, im: $0 * warped), Complex(re: 0, im: -$0 * warped)] }
        let degree = analogPoles.count - analogZeros.count
        var k = gain * pow(warped, Double(degree))
        let numerator = product(analogZeros.map { Complex(re: 4 - $0.re, im: -$0.im) })
        let denominator = product(analogPoles.map { Complex(re: 4 - $0.re, im: -$0.im) })
        k *= (numerator.re * denominator.re + numerator.im * denominator.im) / (denominator.re * denominator.re + denominator.im * denominator.im)
        let digitalPoles = poles.map { bilinear($0.re * warped, $0.im * warped) }
        let digitalZeros: [Complex] =
            zeros.isEmpty
            ? Array(repeating: Complex(re: -1, im: 0), count: poles.count) : zeros.map { bilinear(0, $0 * warped) }
        // Pair each pole pair (closest to the unit circle first) with the nearest unused zero pair, as zpk2sos("nearest").
        var unused = Array(digitalZeros.indices)
        var result: [[Double]] = []
        for pole in digitalPoles.sorted(by: { hypot($0.re, $0.im) > hypot($1.re, $1.im) }) {
            let pick = unused.min { hypot(digitalZeros[$0].re - pole.re, digitalZeros[$0].im - pole.im) < hypot(digitalZeros[$1].re - pole.re, digitalZeros[$1].im - pole.im) }!
            unused.removeAll { $0 == pick }
            let zero = digitalZeros[pick]
            result.append([1, -2 * zero.re, zero.re * zero.re + zero.im * zero.im, 1, -2 * pole.re, pole.re * pole.re + pole.im * pole.im])
        }
        // SciPy keeps the gain in the last-designed section; any placement gives the same cascade.
        for j in 0..<3 { result[0][j] *= k }
        return result
    }

    /// SciPy sosfiltfilt (padtype "odd", default padlen, sosfilt_zi initial conditions).
    static func filtfilt(_ sos: [[Double]], _ x: [Double]) -> [Double] {
        let trailing = min(sos.filter { $0[2] == 0 }.count, sos.filter { $0[5] == 0 }.count)
        let padlen = 3 * (2 * sos.count + 1 - trailing)
        guard x.count > padlen else { return x }
        let n = x.count
        var ext = [Double](repeating: 0, count: n + 2 * padlen)
        for i in 0..<padlen { ext[i] = 2 * x[0] - x[padlen - i] }
        for i in 0..<n { ext[padlen + i] = x[i] }
        for i in 0..<padlen { ext[padlen + n + i] = 2 * x[n - 1] - x[n - 2 - i] }
        // sosfilt_zi
        var zi: [[Double]] = [], scale = 1.0
        for s in sos {
            let (b0, b1, b2, a1, a2) = (s[0], s[1], s[2], s[4], s[5])
            // Solve [[1 + a1, −1], [a2, 1]] · z = [b1 − a1·b0, b2 − a2·b0].
            let r0 = b1 - a1 * b0, r1 = b2 - a2 * b0
            let det = (1 + a1) + a2
            let z0 = (r0 + r1) / det, z1 = r1 - a2 * z0
            zi.append([scale * z0, scale * z1])
            scale *= (b0 + b1 + b2) / (1 + a1 + a2)
        }
        func run(_ input: inout [Double], first: Double) {
            for (index, s) in sos.enumerated() {
                var z0 = zi[index][0] * first, z1 = zi[index][1] * first
                let (b0, b1, b2, a1, a2) = (s[0], s[1], s[2], s[4], s[5])
                for i in input.indices {
                    let v = input[i], y = b0 * v + z0
                    z0 = b1 * v - a1 * y + z1
                    z1 = b2 * v - a2 * y
                    input[i] = y
                }
            }
        }
        run(&ext, first: ext[0])
        ext.reverse()
        run(&ext, first: ext[0])
        ext.reverse()
        return Array(ext[padlen..<(padlen + n)])
    }

    /// SciPy resample_poly(x, up, down) with its default ('kaiser', 5.0) window and zero padding.
    static func resamplePoly(_ x: [Double], up upIn: Int, down downIn: Int) -> [Double] {
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        let g = gcd(upIn, downIn), up = upIn / g, down = downIn / g
        if up == 1 && down == 1 { return x }
        let nIn = x.count
        let nOut = (nIn * up) / down + ((nIn * up) % down == 0 ? 0 : 1)
        let maxRate = max(up, down), halfLength = 10 * maxRate, taps = 2 * halfLength + 1
        let cutoff = 1 / Double(maxRate), alpha = Double(taps - 1) / 2
        var h = [Double](repeating: 0, count: taps), sum = 0.0
        let i0Beta = besselI0(5)
        for n in 0..<taps {
            let m = Double(n) - alpha
            let arg = cutoff * m
            let sinc = arg == 0 ? 1 : sin(Double.pi * arg) / (Double.pi * arg)
            let r = 2 * Double(n) / Double(taps - 1) - 1
            let window = besselI0(5 * (max(0, 1 - r * r)).squareRoot()) / i0Beta
            h[n] = cutoff * sinc * window
            sum += h[n]
        }
        for n in 0..<taps { h[n] = h[n] / sum * Double(up) }
        let prePad = down - halfLength % down
        let preRemove = (halfLength + prePad) / down
        var postPad = 0
        func outputLength(_ lengthH: Int) -> Int { ((nIn - 1) * up + lengthH - 1) / down + 1 }
        while outputLength(taps + prePad + postPad) < nOut + preRemove { postPad += 1 }
        let filter = [Double](repeating: 0, count: prePad) + h + [Double](repeating: 0, count: postPad)
        var y = [Double](repeating: 0, count: nOut)
        filter.withUnsafeBufferPointer { f in
            x.withUnsafeBufferPointer { input in
                for j in 0..<nOut {
                    let position = (j + preRemove) * down
                    // Terms i with 0 ≤ position − i·up < filter.count.
                    let iMax = min(nIn - 1, position / up)
                    let iMin = max(0, (position - f.count + up) / up)
                    guard iMin <= iMax else { continue }
                    var acc = 0.0
                    for i in iMin...iMax { acc += input[i] * f[position - i * up] }
                    y[j] = acc
                }
            }
        }
        return y
    }

    static func besselI0(_ x: Double) -> Double {
        var sum = 1.0, term = 1.0, k = 1.0
        let q = x * x / 4
        while true {
            term *= q / (k * k)
            sum += term
            if term < sum * 1e-17 { return sum }
            k += 1
        }
    }

    /// audiosr.lowpass.lowpass(data, highcut, fs=48 000, order=8, _type) → lowpass_filter: filtfilt, resample to
    /// int(hi·fs) and back, filtfilt again; length preserved.
    static func lowpass(_ x: [Double], highcut: Int, filter: Filter) -> [Double] {
        // The reference raises for a cutoff at Nyquist (SciPy refuses Wn = 1); nothing lies above it, so keep the input.
        guard highcut > 0, highcut < rate / 2 else { return x }
        let hi = Double(highcut) / Double(rate / 2)
        let sos = sections(filter, wn: hi)
        var y = filtfilt(sos, x)
        let down = Int(hi * Double(rate))
        var hard = resamplePoly(resamplePoly(y, up: down, down: rate), up: rate, down: down)
        if hard.count < y.count { hard += [Double](repeating: 0, count: y.count - hard.count) } else { hard = Array(hard.prefix(y.count)) }
        y = filtfilt(sos, hard)
        return y
    }

    /// `_find_cutoff`: the last index from the top whose cumulative energy is below `percentile` of the total.
    static func cutoff(cumulative energy: [Double], percentile: Double) -> Int {
        guard let total = energy.last else { return 0 }
        let threshold = total * percentile
        for i in 1..<max(1, energy.count) where energy[energy.count - i] < threshold { return energy.count - i }
        return 0
    }

    // MARK: Mel features (utils.mel_spectrogram_train)

    /// librosa.filters.mel(sr=48 000, n_fft=2048, n_mels=256, fmin=20, fmax=24 000): Slaney scale and area norm,
    /// as a [1025, 256] float32 matrix.
    static func melBasis(bins: Int = 256, nFFT: Int = 2048, fMin: Double = 20, fMax: Double = 24_000) -> MLXArray {
        MLXArray(melWeights(bins: bins, nFFT: nFFT, fMin: fMin, fMax: fMax), [nFFT / 2 + 1, bins])
    }
    static func melWeights(bins: Int = 256, nFFT: Int = 2048, fMin: Double = 20, fMax: Double = 24_000) -> [Float] {
        func hzToMel(_ f: Double) -> Double { f < 1000 ? f / (200.0 / 3) : 15 + log(f / 1000) / (log(6.4) / 27) }
        func melToHz(_ m: Double) -> Double { m < 15 ? m * 200.0 / 3 : 1000 * exp((log(6.4) / 27) * (m - 15)) }
        let freqs = nFFT / 2 + 1
        let fftFreqs = (0..<freqs).map { Double($0) * Double(rate) / Double(nFFT) }
        let (low, high) = (hzToMel(fMin), hzToMel(fMax))
        let points = (0..<(bins + 2)).map { melToHz(low + (high - low) * Double($0) / Double(bins + 1)) }
        var weights = [Float](repeating: 0, count: freqs * bins)
        for j in 0..<bins {
            let enorm = 2 / (points[j + 2] - points[j])
            for i in 0..<freqs {
                let lower = (fftFreqs[i] - points[j]) / (points[j + 1] - points[j])
                let upper = (points[j + 2] - fftFreqs[i]) / (points[j + 2] - points[j + 1])
                weights[i * bins + j] = Float(max(0, min(lower, upper)) * enorm)
            }
        }
        return weights
    }

    /// Log-mel [frames, 256] and magnitude [frames, 1025] of a 48 kHz waveform: reflect pad (n_fft − hop)/2, periodic
    /// Hann, center=False, hop 480, log(max(·, 1e-5)). Float32 throughout.
    static func melSpectrogram(_ wave: [Float], basis: MLXArray) -> (mel: MLXArray, magnitude: MLXArray) {
        let nFFT = 2048, hop = 480, pad = (nFFT - hop) / 2
        var padded = [Float](repeating: 0, count: wave.count + 2 * pad)
        for i in 0..<pad { padded[i] = wave[pad - i]; padded[pad + wave.count + i] = wave[wave.count - 2 - i] }
        for i in wave.indices { padded[pad + i] = wave[i] }
        let frames = (padded.count - nFFT) / hop + 1
        let signal = MLXArray(padded)
        let window = MLXArray((0..<nFFT).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(nFFT))) })
        let framed = asStrided(signal, [frames, nFFT], strides: [hop, 1]) * window
        let magnitude = abs(MLXFFT.rfft(framed, axis: 1))
        let mel = log(maximum(matmul(magnitude, basis), MLXArray(Float(1e-5))))
        return (mel, magnitude)
    }

    // MARK: librosa.stft / istft (n_fft 2048, hop 512, Hann, center with zero padding) for the low-band replacement

    static func stft(_ wave: [Float]) -> (real: MLXArray, imag: MLXArray) {
        let nFFT = 2048, hop = 512, pad = nFFT / 2
        let padded = [Float](repeating: 0, count: pad) + wave + [Float](repeating: 0, count: pad)
        let frames = 1 + (padded.count - nFFT) / hop
        let window = MLXArray((0..<nFFT).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(nFFT))) })
        let spectrum = MLXFFT.rfft(asStrided(MLXArray(padded), [frames, nFFT], strides: [hop, 1]) * window, axis: 1)
        return (spectrum.realPart(), spectrum.imaginaryPart()) // [frames, 1025]
    }

    static func istft(real: MLXArray, imag: MLXArray, length: Int) -> MLXArray {
        let nFFT = 2048, hop = 512
        let window = (0..<nFFT).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(nFFT))) }
        let frames = MLXFFT.irfft(real + imag.asImaginary(), n: nFFT, axis: 1) * MLXArray(window)
        return metalISTFTOverlapAdd(
            windowed: frames.expandedDimensions(axis: 0), windowSq: MLXArray(window.map { $0 * $0 }), numFrames: real.dim(0), nFFT: nFFT,
            hopLength: hop, targetLength: length, center: true, finalShape: [1])[0]
    }
}
