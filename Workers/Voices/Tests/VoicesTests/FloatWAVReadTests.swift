// SPDX-License-Identifier: MIT
import AVFoundation
import ArsenalWire
import Foundation
import Testing
@testable import VoicesWorker

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

@Test func referenceReadsEveryFrameOfALibsndfileFloatWAV() throws {
    let url = try writeFloatWAV(tone16k, rate: 16_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let clip = try PCM.reference(url, rate: 16_000)
    #expect(clip.samples.count == tone16k.count)
    #expect(zip(clip.samples, tone16k).allSatisfy { abs($0 - $1) < 1e-6 })
}

/// Release review E (L-E5): an ALAC clip cut inside its last block reads as a clean end 4,500 frames early (0.97%, inside
/// the old 1% allowance); it is refused as cut rather than read without its tail.
@Test func referenceCutInsideItsLastBlockIsRefused() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("voices-cut-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("tone.caf"), cut = directory.appendingPathComponent("cut.caf")
    let frames = 65_536 * 7 + 4_500
    do {
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatAppleLossless, AVSampleRateKey: 16_000, AVNumberOfChannelsKey: 1, AVEncoderBitDepthHintKey: 16]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(frames)))
        buffer.frameLength = AVAudioFrameCount(frames)
        for i in 0..<frames { buffer.floatChannelData![0][i] = 0.3 * sin(Float(i) / 10) }
        try file.write(from: buffer)
    }
    #expect(try PCM.reference(url, rate: 16_000).samples.count == frames)
    let data = try Data(contentsOf: url)
    try data.prefix(data.count - 200).write(to: cut)
    #expect { try PCM.reference(cut, rate: 16_000) } throws: { ($0 as? WorkerFailure)?.code == "invalid_request" }
}
