// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import Testing
@testable import MusicWorker

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

/// Reference audio reaches the music models through NativeMedia (ACEStepPipeline: 48 kHz stereo), not AudioFiles,
/// so the check runs on that reader with the pipeline's own arguments (2 Nov 2026 review: it tested an unused path).
@Test func referenceAudioReadsEveryFrameOfALibsndfileFloatWAV() throws {
    let url = try writeFloatWAV(tone16k, rate: 16_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let audio = try NativeMedia.audio(url, sampleRate: 48000, channels: 2, maximumFrames: 48_000 * 600)
    #expect(audio.channels == 2 && abs(audio.samples.count / 2 - 48_000) <= 2)
    // The one channel feeds both, and the tone (RMS 0.1/√2) runs to the end: a short read would leave the tail silent.
    let left = stride(from: 0, to: audio.samples.count, by: 2).map { audio.samples[$0] }, right = stride(from: 1, to: audio.samples.count, by: 2).map { audio.samples[$0] }
    #expect(left == right)
    let tail = sqrt(left[44_000..<47_000].reduce(0) { $0 + $1 * $1 } / 3_000)
    #expect(abs(tail - 0.1 / Float(2).squareRoot()) < 0.01)
}
