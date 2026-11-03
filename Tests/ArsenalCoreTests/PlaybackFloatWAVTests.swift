// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// A float32 WAV laid out as libsndfile (soundfile, `subtype="FLOAT"`) writes it: IEEE-float fmt, `fact`, `PEAK`, data.
/// One `AVAudioFile.read(into:)` returns 660 of 16,000 frames short for this layout (2 Nov 2026).
private func libsndfileFloatWAV(_ samples: [Float], rate: UInt32) -> Data {
    var d = Data()
    func tag(_ s: String) { d.append(Data(s.utf8)) }
    func u32(_ n: UInt32) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
    func u16(_ n: UInt16) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
    let dataBytes = UInt32(samples.count * 4)
    tag("RIFF"); u32(4 + 24 + 12 + 24 + 8 + dataBytes); tag("WAVE")
    tag("fmt "); u32(16); u16(3); u16(1); u32(rate); u32(rate * 4); u16(4); u16(32)
    tag("fact"); u32(4); u32(UInt32(samples.count))
    let peak = samples.indices.max { abs(samples[$0]) < abs(samples[$1]) } ?? 0
    tag("PEAK"); u32(16); u32(1); u32(0x6ae8_289e); u32(abs(samples[peak]).bitPattern); u32(UInt32(peak))
    tag("data"); u32(dataBytes)
    for x in samples { u32(x.bitPattern) }
    return d
}

private func writeFloatWAV(_ samples: [Float], rate: UInt32) throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("libsndfile-f32-\(UUID().uuidString).wav")
    try libsndfileFloatWAV(samples, rate: rate).write(to: url)
    return url
}

/// The loud part is only in the last 600 frames, which a short read drops: it would measure quiet audio and return 0 dB.
@Test func playbackGainHearsTheEndOfALibsndfileFloatWAV() throws {
    let samples = (0..<16_000).map { i in Float(sin(2 * .pi * 1000 * Double(i) / 16_000) * (i >= 15_400 ? 0.9 : 0.001)) }
    let url = try writeFloatWAV(samples, rate: 16_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let m = PlaybackLoudness.measure(samples, rate: 16_000)
    let expected = min(0, PlaybackLoudness.gainDB(loudness: m.integrated, truePeak: m.truePeak))
    #expect(expected < 0)
    #expect(abs(try PlaybackQueue.measureGain(url) - expected) < 1e-9)
}
