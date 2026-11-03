// SPDX-License-Identifier: MIT
import Foundation

public enum StatsQuality {
    public static func audioIdentical(reference: MeasurementOutput, candidate: MeasurementOutput) -> Bool {
        guard let rate = reference.sampleRate, rate > 0, rate == candidate.sampleRate,
            let a = reference.samples, let b = candidate.samples, !a.isEmpty, a.count == b.count,
            a.allSatisfy(\.isFinite), b.allSatisfy(\.isFinite)
        else { return false }
        return a == b
    }

    public static func agreement(reference: [Int], candidate: [Int]) throws -> Double {
        guard !reference.isEmpty, reference.count == candidate.count else { throw ManifestError.invalid("Quality item counts differ or are empty") }
        return Double(zip(reference, candidate).filter { $0 == $1 }.count) / Double(reference.count) * 100
    }
    public static func chat(reference: [[Double]], candidate: [[Double]]) throws -> (agreement: Double, kl: Double) {
        guard !reference.isEmpty, reference.count == candidate.count else { throw ManifestError.invalid("Teacher-forced step counts differ or are empty") }
        var agreed = 0, kl = 0.0
        for (baseline, trial) in zip(reference, candidate) {
            guard !baseline.isEmpty, baseline.count == trial.count, (baseline + trial).allSatisfy(\.isFinite) else {
                throw ManifestError.invalid("Chat vocabulary shapes differ or logits are invalid")
            }
            if argmax(baseline) == argmax(trial) { agreed += 1 }
            let p = logSoftmax(baseline), q = logSoftmax(trial)
            kl += zip(p, q).reduce(0) { $0 + exp($1.0) * ($1.0 - $1.1) }
        }
        return (Double(agreed) / Double(reference.count) * 100, max(0, kl / Double(reference.count)))
    }
    private static func argmax(_ x: [Double]) -> Int { x.indices.max { x[$0] < x[$1] } ?? 0 }
    private static func logSoftmax(_ x: [Double]) -> [Double] {
        let maximum = x.max() ?? 0
        let normalizer = log(x.reduce(0) { $0 + exp($1 - maximum) })
        return x.map { $0 - maximum - normalizer }
    }
    /// Pixels are normalized [0,1] RGB, identical width/height/channel count. Identity caps at 120 dB.
    public static func psnr(reference: [Double], candidate: [Double]) throws -> Double {
        guard !reference.isEmpty, reference.count == candidate.count, (reference + candidate).allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw ManifestError.invalid("Image shapes differ or pixels are invalid")
        }
        let mse = zip(reference, candidate).reduce(0) { $0 + pow($1.0 - $1.1, 2) } / Double(reference.count)
        return -10 * log10(max(1e-12, mse))
    }
    /// Deterministic Hann-window magnitude spectra, 256-sample frames, hop 128; no phase alignment.
    /// Compare the fixed clip, preserving sample rate/count. Relative L2, lower is closer.
    public static func spectralDistance(reference: [Double], candidate: [Double], frameSize: Int = 256) throws -> Double {
        guard reference.count >= frameSize, reference.count == candidate.count, frameSize >= 2,
            (reference + candidate).allSatisfy(\.isFinite)
        else { throw ManifestError.invalid("Audio sample counts differ or samples are invalid") }
        var delta = 0.0, magnitude = 0.0
        for start in stride(from: 0, through: reference.count - frameSize, by: frameSize / 2) {
            for bin in 0...frameSize / 2 {
                let a = spectrum(reference, start: start, size: frameSize, bin: bin)
                let b = spectrum(candidate, start: start, size: frameSize, bin: bin)
                delta += pow(a - b, 2); magnitude += a * a
            }
        }
        guard magnitude > 0 else {
            if delta == 0 { return 0 }
            throw ManifestError.invalid("The reference audio is silent; spectral distance is undefined")
        }
        return sqrt(delta / magnitude)
    }
    private static func spectrum(_ samples: [Double], start: Int, size: Int, bin: Int) -> Double {
        var real = 0.0, imaginary = 0.0
        for i in 0..<size {
            let hann = 0.5 * (1 - cos(2 * .pi * Double(i) / Double(size - 1)))
            let phase = 2 * .pi * Double(bin * i) / Double(size)
            real += samples[start + i] * hann * cos(phase); imaginary -= samples[start + i] * hann * sin(phase)
        }
        return hypot(real, imaginary)
    }
}
