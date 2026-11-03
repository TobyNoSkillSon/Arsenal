// SPDX-License-Identifier: MIT
import AVFoundation
import ArsenalWire
import Foundation
import Testing
@testable import IntegrationKit

/// The shared decoder against the files the 2 Nov 2026 reviews worried about: compressed files whose length is only
/// an estimate, cut and unreadable files (a caller's mistake, so 422 rather than 503), and many-channel files.
struct NativeMediaDecodeTests {
    private let directory: URL = {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("native-media-" + UUID().uuidString)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()

    private func code(_ body: () throws -> Void) -> String? {
        do { try body(); return nil } catch let failure as WorkerFailure { return failure.code } catch { return "raw \(error)" }
    }

    /// MPEG-1 Layer III frames of silence at 44.1 kHz mono (empty side info decodes to zeros), 64 kbit/s first and
    /// 128 kbit/s after, with no Xing/LAME table: the VBR case where a reader can only estimate the length from bytes.
    private func variableBitRateMP3() -> Data {
        func frame(_ index: UInt8, _ kbps: Int) -> Data { Data([0xFF, 0xFB, index << 4, 0xC0]) + Data(count: 144 * kbps * 1000 / 44_100 - 4) }
        return (0..<20).reduce(into: Data()) { d, _ in d += frame(5, 64) } + (0..<200).reduce(into: Data()) { d, _ in d += frame(9, 128) }
    }

    /// Interleaved PCM16 WAV bytes for any channel count (AVAudioFormat has no plain target above two channels).
    private func wav(channels: Int, rate: Int, frames: Int, value: (_ frame: Int, _ channel: Int) -> Int16) -> Data {
        var d = Data()
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { d.append(contentsOf: $0) } }
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + frames * channels * 2); d.append(contentsOf: Array("WAVEfmt ".utf8))
        u32(16); u16(1); u16(channels); u32(rate); u32(rate * channels * 2); u16(channels * 2); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(frames * channels * 2)
        for f in 0..<frames { for c in 0..<channels { withUnsafeBytes(of: value(f, c).littleEndian) { d.append(contentsOf: $0) } } }
        return d
    }

    private func tone(seconds: Double, rate: Double) throws -> AVAudioPCMBuffer {
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1))
        let frames = Int(seconds * rate)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)))
        buffer.frameLength = AVAudioFrameCount(frames)
        for i in 0..<frames { buffer.floatChannelData![0][i] = 0.5 * sin(Float(i) * 2 * .pi * 440 / Float(rate)) }
        return buffer
    }

    @Test func variableBitRateMP3WithoutATableDecodesEveryFrame() throws {
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("vbr.mp3")
        try variableBitRateMP3().write(to: url)
        let audio = try NativeMedia.audio(url, sampleRate: 16_000)
        let expected = Double(220 * 1152) * 16_000 / 44_100
        #expect(abs(Double(audio.samples.count) - expected) <= 64, "\(audio.samples.count) of \(expected)")
        #expect(audio.samples.allSatisfy { abs($0) < 1e-6 })
    }

    /// AAC in an MPEG-4 file and as raw ADTS (whose length AVFoundation also derives from the stream).
    @Test(arguments: ["m4a", "aac"]) func aacDecodesItsWholeLength(_ ext: String) throws {
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.\(ext)")
        let source = try tone(seconds: 3, rate: 44_100)
        do {
            let settings: [String: Any] = [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44_100, AVNumberOfChannelsKey: 1]
            let file = try AVAudioFile(forWriting: url, settings: settings)
            try file.write(from: source)
        }
        let audio = try NativeMedia.audio(url, sampleRate: 16_000)
        // MPEG-4 records the encoder's priming and padding and trims them; raw ADTS has nowhere to record them, so it
        // decodes up to about 3,000 frames (44.1 kHz) long, as any decoder of it does. Either way the whole tone is there.
        let slack = ext == "aac" ? 1_200 : 480
        #expect(audio.samples.count >= 48_000 - 480 && audio.samples.count <= 48_000 + slack, "\(audio.samples.count) samples")
        let rms = sqrt(audio.samples[8_000..<40_000].reduce(0) { $0 + $1 * $1 } / 32_000)
        #expect(rms > 0.3)
    }

    /// A WAV whose header claims more data than the file holds decodes what is there, as before the strict check.
    @Test func cutWAVDecodesWhatItHolds() throws {
        defer { try? FileManager.default.removeItem(at: directory) }
        let full = wav(channels: 1, rate: 16_000, frames: 16_000) { f, _ in Int16(f % 100) }
        let url = directory.appendingPathComponent("cut.wav")
        try full.prefix(full.count - 3_200).write(to: url)
        let audio = try NativeMedia.audio(url, sampleRate: 16_000)
        #expect(audio.samples.count == 16_000 - 1_600)
    }

    /// A renamed text file and a garbage video are the caller's mistakes: invalid_request (422), never runner (503).
    @Test func unreadableMediaIsTheCallersMistake() async throws {
        defer { try? FileManager.default.removeItem(at: directory) }
        let audio = directory.appendingPathComponent("text.wav"), video = directory.appendingPathComponent("text.mp4")
        try Data("this is not audio".utf8).write(to: audio)
        try Data(repeating: 0x5A, count: 4_096).write(to: video)
        #expect(code { _ = try NativeMedia.audio(audio, sampleRate: 16_000) } == "invalid_request")
        #expect(code { _ = try NativeMedia.openAudio(audio) } == "invalid_request")
        var videoCode: String?
        do { _ = try await NativeVideoReader(url: video) } catch let failure as WorkerFailure { videoCode = failure.code } catch { videoCode = "raw \(error)" }
        #expect(videoCode == "invalid_request")
        #expect(code { _ = try NativeMedia.decoding("Video") { throw NSError(domain: AVFoundationErrorDomain, code: -11829) } } == "invalid_request")
        #expect(code { _ = try NativeMedia.decoding("Video") { throw WorkerFailure(code: "cancelled", message: "") } } == "cancelled")
        // Release review E (L-E8): a media services reset is the system's temporary fault, kept retryable (runner, 503).
        #expect(code { _ = try NativeMedia.decoding("Video") { throw NSError(domain: AVFoundationErrorDomain, code: -11819) } } == "runner")
    }

    /// Many channels: one channel alone, or the mean of all, streamed block by block; past the channel cap, refused.
    @Test func manyChannelFilesStreamOneChannelOrTheirMean() throws {
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("16ch.wav")
        // Channel c holds the constant 1000 × (c + 1); 3 s at 48 kHz crosses several 64k-frame blocks.
        try wav(channels: 16, rate: 48_000, frames: 144_000) { _, c in Int16(1_000 * (c + 1)) }.write(to: url)
        let third = try NativeMedia.audio(url, sampleRate: 16_000, channels: 1, mix: .channel(3))
        #expect(abs(third.samples.count - 48_000) <= 2)
        #expect(third.samples[1_000..<47_000].allSatisfy { abs($0 - 4_000 / 32_768) < 1e-3 })
        let mean = try NativeMedia.audio(url, sampleRate: 16_000, channels: 1, mix: .mean)
        #expect(mean.samples[1_000..<47_000].allSatisfy { abs($0 - 8_500 / 32_768) < 1e-3 })
        #expect(code { _ = try NativeMedia.audio(url, sampleRate: 16_000, mix: .channel(16)) } == "invalid_request")
        let wide = directory.appendingPathComponent("40ch.wav")
        try wav(channels: 40, rate: 16_000, frames: 1_600) { _, _ in 0 }.write(to: wide)
        #expect(code { _ = try NativeMedia.audio(wide, sampleRate: 16_000) } == "invalid_request")
    }

    /// Release review E (L-E6): 32 source channels are read and 33 are refused by the channel guard itself, not by
    /// AVAudioFile failing to open the file.
    @Test func thirtyTwoChannelsAreReadAndThirtyThreeRefusedByTheGuard() throws {
        defer { try? FileManager.default.removeItem(at: directory) }
        let fits = directory.appendingPathComponent("32ch.wav"), over = directory.appendingPathComponent("33ch.wav")
        try wav(channels: 32, rate: 16_000, frames: 1_600) { _, c in Int16(100 * c) }.write(to: fits)
        try wav(channels: 33, rate: 16_000, frames: 1_600) { _, _ in 0 }.write(to: over)
        #expect(try NativeMedia.audio(fits, sampleRate: 16_000, mix: .mean).samples.count == 1_600)
        #expect(try NativeMedia.openAudio(over).processingFormat.channelCount == 33)
        do {
            _ = try NativeMedia.audio(over, sampleRate: 16_000, mix: .mean)
            Issue.record("33 channels decoded")
        } catch let failure as WorkerFailure {
            #expect(failure.code == "invalid_request" && failure.message.contains("33 channels; at most 32"), "\(failure.message)")
        }
    }

    /// Release review E (L-E5): the short-end allowance is a clean end within min(1%, 4,096 frames); a read error before
    /// the length is refused however little is missing.
    @Test func shortEndsAreCleanEndsWithinTheAllowanceOnly() throws {
        let eof = NSError(domain: NSOSStatusErrorDomain, code: -39), corrupt = NSError(domain: "com.apple.coreaudio.avfaudio", code: -50)
        func passes(_ read: AVAudioFramePosition, _ length: AVAudioFramePosition, _ failure: Error?) -> Bool {
            (try? NativeMedia.checkEnd(read: read, length: length, failure: failure)) != nil
        }
        #expect(passes(16_000, 16_000, eof) && passes(16_000, 16_000, nil), "a whole read passes")
        #expect(passes(15_840, 16_000, eof) && !passes(15_839, 16_000, eof), "a short file may fall 1% short")
        let long: AVAudioFramePosition = 48_000 * 5_400
        #expect(passes(long - 4_096, long, eof) && !passes(long - 4_097, long, nil), "a 90-minute file may fall at most 4,096 frames short")
        #expect(!passes(15_999, 16_000, corrupt), "a decode error one frame before the end is refused")
        #expect(passes(16_000, 16_000, corrupt), "an error after the whole length is read is not a shortfall")
        #expect(!passes(0, 16_000, eof), "nothing read is refused")
    }

    /// Release review E (L-E5): an ALAC file cut inside its last block reads as a clean end 5,000 frames early (0.94%,
    /// inside the old 1% allowance) and is now refused rather than decoded without its tail.
    @Test func cutCompressedTailIsRefused() throws {
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.caf"), cut = directory.appendingPathComponent("cut.caf")
        let frames = 65_536 * 8 + 5_000
        do {
            let settings: [String: Any] = [AVFormatIDKey: kAudioFormatAppleLossless, AVSampleRateKey: 16_000, AVNumberOfChannelsKey: 1, AVEncoderBitDepthHintKey: 16]
            let file = try AVAudioFile(forWriting: url, settings: settings)
            try file.write(from: try tone(seconds: Double(frames) / 16_000, rate: 16_000))
        }
        #expect(try NativeMedia.audio(url, sampleRate: 16_000).samples.count == frames)
        let data = try Data(contentsOf: url)
        try data.prefix(data.count - 200).write(to: cut)
        #expect(try NativeMedia.openAudio(cut).length == AVAudioFramePosition(frames))
        #expect(code { _ = try NativeMedia.audio(cut, sampleRate: 16_000) } == "invalid_request")
    }

    /// The bound is the caller's maximum, checked before any decoding.
    @Test func longFilesAreRefusedBeforeDecoding() throws {
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("long.wav")
        try wav(channels: 2, rate: 8_000, frames: 8_000 * 11) { _, _ in 0 }.write(to: url)
        #expect(code { _ = try NativeMedia.audio(url, sampleRate: 16_000, maximumFrames: 16_000 * 10) } == "invalid_request")
        #expect(abs(try NativeMedia.audio(url, sampleRate: 16_000, maximumFrames: 16_000 * 12).samples.count - 16_000 * 11) <= 2)
    }
}
