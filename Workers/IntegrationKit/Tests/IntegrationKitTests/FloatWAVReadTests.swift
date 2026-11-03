// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import IntegrationKit

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

private let tone16k = (0..<16_000).map { Float(0.1 * sin(Double($0) / 10)) }

@Test func nativeMediaReadsEveryFrameOfALibsndfileFloatWAV() throws {
    let url = try writeFloatWAV(tone16k, rate: 16_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let audio = try NativeMedia.audio(url, sampleRate: 16_000)
    #expect(audio.samples.count == tone16k.count)
    #expect(zip(audio.samples, tone16k).allSatisfy { abs($0 - $1) < 1e-6 })
}
