// SPDX-License-Identifier: MIT
import ArsenalWire
import Foundation
import Testing
@testable import SoundWorker

/// A float32 WAV laid out as libsndfile (soundfile, `subtype="FLOAT"`) writes it: IEEE-float fmt, `fact`, `PEAK`, data.
func libsndfileFloatWAV(_ samples: [Float], rate: UInt32) -> Data {
    var d = Data()
    func tag(_ s: String) { d.append(Data(s.utf8)) }
    func u32(_ n: UInt32) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
    func u16(_ n: UInt16) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
    func f32(_ x: Float) { u32(x.bitPattern) }
    let dataBytes = UInt32(samples.count * 4)
    tag("RIFF"); u32(4 + 24 + 12 + 24 + 8 + dataBytes); tag("WAVE")
    tag("fmt "); u32(16); u16(3); u16(1); u32(rate); u32(rate * 4); u16(4); u16(32)
    tag("fact"); u32(4); u32(UInt32(samples.count))
    let peak = samples.indices.max { abs(samples[$0]) < abs(samples[$1]) } ?? 0
    tag("PEAK"); u32(16); u32(1); u32(0x6ae8_289e); f32(abs(samples[peak])); u32(UInt32(peak))
    tag("data"); u32(dataBytes)
    for x in samples { f32(x) }
    return d
}

/// Bug (2 Nov 2026): one `AVAudioFile.read(into:)` returned 660 of 16,000 frames short for this layout,
/// so every float WAV written by soundfile was clipped at the end.
@Test func floatWAVFromLibsndfileReadsEveryFrame() throws {
    let directory = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-sound/pcm-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    for (frames, rate) in [(16_000, UInt32(16_000)), (48_000, 48_000), (200_000, 44_100), (1_000, 16_000)] {
        let samples = (0..<frames).map { Float(0.1 * sin(Double($0) / 10)) }
        let url = directory.appendingPathComponent("f32-\(frames).wav")
        try libsndfileFloatWAV(samples, rate: rate).write(to: url)
        let clip = try PCM.input(url)
        #expect(clip.frames == frames && clip.channels == 1 && clip.sampleRate == Int(rate))
        #expect(clip.samples.count == samples.count)
        #expect(zip(clip.samples, samples).allSatisfy { $0 == $1 })
    }
}

/// Bug (2 Nov 2026 review): every refusal of the caller's audio was an EngineError, which the service returns as 500.
/// An unreadable file, one over 600 s, NaN samples and bad request options are the caller's mistakes: invalid_request (422).
@Test func badInputAudioIsTheCallersMistake() throws {
    let directory = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-sound/pcm-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    func code(_ body: () throws -> Void) -> String? {
        do { try body(); return nil } catch let failure as WorkerFailure { return failure.code } catch { return "\(type(of: error))" }
    }
    let text = directory.appendingPathComponent("text.wav")
    try Data("not audio at all".utf8).write(to: text)
    #expect(code { _ = try PCM.input(text) } == "invalid_request")
    let long = directory.appendingPathComponent("long.wav")
    try libsndfileFloatWAV([Float](repeating: 0, count: 8_000 * 601), rate: 8_000).write(to: long)
    #expect(code { _ = try PCM.input(long) } == "invalid_request")
    let nan = directory.appendingPathComponent("nan.wav")
    try libsndfileFloatWAV([0.1, .nan, 0.1, 0.2], rate: 16_000).write(to: nan)
    #expect(code { _ = try PCM.input(nan) } == "invalid_request")
    #expect(code { _ = try soundRequest(["prompt": "rain", "steps": 0], model: "stable-audio-3-small-sfx", maxSeconds: 10) } == "invalid_request")
}

/// Release review E (M-E2): the input streams through the shared decoder at the file's rate capped at 48 kHz (every
/// Sound engine's rate or below), so a high-rate multichannel file no longer reserves length × channels at its own
/// rate. 1 s of 96 kHz stereo comes back as 1 s of 48 kHz stereo with each channel kept.
@Test func highRateMultichannelInputDecodesAtFortyEightKilohertz() throws {
    let directory = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-sound/pcm-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    var d = Data()
    func u32(_ n: UInt32) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
    func u16(_ n: UInt16) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
    let frames = 96_000
    d.append(Data("RIFF".utf8)); u32(UInt32(36 + frames * 4)); d.append(Data("WAVEfmt ".utf8))
    u32(16); u16(1); u16(2); u32(96_000); u32(96_000 * 4); u16(4); u16(16); d.append(Data("data".utf8)); u32(UInt32(frames * 4))
    for _ in 0..<frames { u16(UInt16(bitPattern: 8_192)); u16(UInt16(bitPattern: -8_192)) }
    let url = directory.appendingPathComponent("96k-stereo.wav")
    try d.write(to: url)
    let clip = try PCM.input(url)
    #expect(clip.sampleRate == 48_000 && clip.channels == 2)
    #expect(abs(clip.frames - 48_000) <= 2, "\(clip.frames) frames")
    let middle = clip.samples[2_000..<94_000]
    #expect(stride(from: 0, to: middle.count, by: 2).allSatisfy { abs(middle[middle.startIndex + $0] - 0.25) < 1e-3 && abs(middle[middle.startIndex + $0 + 1] + 0.25) < 1e-3 })
}
