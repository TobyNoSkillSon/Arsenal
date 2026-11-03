// SPDX-License-Identifier: MIT
import AVFoundation
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
@testable import RestorationWorker
import Testing

@Suite(.serialized)
struct AudioMuxTests {
    private var repo: URL { (0..<5).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() } }

    /// Six 16x16 frames at 24 fps (0.25 s), video only, as the restoration writers produce.
    private func silentVideo(at url: URL) throws {
        let writer = try NativeVideoWriter(url: url, width: 16, height: 16)
        for index in 0..<6 {
            let frame = MLXArray.zeros([1, 16, 16, 3], dtype: .float16) + Float(index) / 6
            try writer.append(MediaBridge.buffer(frame), time: CMTime(value: Int64(index), timescale: 24), cancellation: NativeCancellation())
        }
        try MediaBridge.wait { try await writer.finish() }
    }

    /// One second of a mono tone: AAC in an M4A, or linear PCM in a WAV (a codec MP4 passthrough cannot carry).
    private func tone(at url: URL, pcm: Bool = false) throws {
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 48000))
        buffer.frameLength = 48000
        let samples = try #require(buffer.floatChannelData)[0]
        for index in 0..<48000 { samples[index] = sin(Float(index) * 2 * Float.pi * 440 / 48000) * 0.1 }
        do {
            let file = try AVAudioFile(
                forWriting: url,
                settings: pcm
                    ? [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48000, AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16]
                    : [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 48000, AVNumberOfChannelsKey: 1])
            try file.write(from: buffer)
        }
    }

    private func trackCounts(_ url: URL) throws -> (video: Int, audio: Int) {
        try MediaBridge.wait { () async throws -> (video: Int, audio: Int) in
            let asset = AVURLAsset(url: url)
            let video = try await asset.loadTracks(withMediaType: .video).count
            let audio = try await asset.loadTracks(withMediaType: .audio).count
            return (video, audio)
        }
    }

    @Test func sourceAudioIsKeptAndSilentSourcesLeaveVideoUnchanged() throws {
        NativeRuntime.prepare()
        let root = repo.appendingPathComponent("lab/g4/audio-mux/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let video = root.appendingPathComponent("rendered.mp4"), silent = root.appendingPathComponent("silent.mp4"), audio = root.appendingPathComponent("tone.m4a")
        try silentVideo(at: video); try FileManager.default.copyItem(at: video, to: silent); try tone(at: audio)

        // A source without audio leaves the rendered file byte-identical.
        let before = try Data(contentsOf: video)
        #expect(try !MediaBridge.keepAudio(from: silent, video: video, origin: 0, cancellation: NativeCancellation()))
        #expect(try Data(contentsOf: video) == before)

        // Audio outside the rendered span is not kept.
        #expect(try !MediaBridge.keepAudio(from: audio, video: video, origin: 5, cancellation: NativeCancellation()))

        // Audio inside the span is muxed in place; the video track and its duration survive, and no temporary file remains.
        let rendered = try MediaBridge.wait { try await AVURLAsset(url: video).load(.duration).seconds }
        #expect(rendered > 0.15)
        #expect(try MediaBridge.keepAudio(from: audio, video: video, origin: 0, cancellation: NativeCancellation()))
        let counts = try trackCounts(video)
        #expect(counts.video == 1)
        #expect(counts.audio == 1)
        let duration = try MediaBridge.wait { try await AVURLAsset(url: video).load(.duration).seconds }
        #expect(abs(duration - rendered) < 0.05)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).sorted() == ["rendered.mp4", "silent.mp4", "tone.m4a"])

        // A cancelled request stops before touching the file.
        let cancelled = NativeCancellation(); cancelled.cancel()
        #expect(throws: WorkerFailure.self) { try MediaBridge.keepAudio(from: audio, video: silent, origin: 0, cancellation: cancelled) }
    }

    /// The re-encode fallback (taken when passthrough cannot put the audio codec into MP4) keeps the same assets alive.
    @Test func pcmSourceAudioIsKeptThroughTheReencodeFallback() throws {
        NativeRuntime.prepare()
        let root = repo.appendingPathComponent("lab/g4/audio-mux/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let video = root.appendingPathComponent("rendered.mp4"), audio = root.appendingPathComponent("tone.wav")
        try silentVideo(at: video); try tone(at: audio, pcm: true)
        #expect(try MediaBridge.keepAudio(from: audio, video: video, origin: 0, cancellation: NativeCancellation()))
        let counts = try trackCounts(video)
        #expect(counts.video == 1 && counts.audio == 1)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).sorted() == ["rendered.mp4", "tone.wav"])
    }
}
