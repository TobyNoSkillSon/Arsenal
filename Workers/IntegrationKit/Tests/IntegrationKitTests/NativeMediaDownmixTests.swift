// SPDX-License-Identifier: MIT
import AVFoundation
import Foundation
import Testing
@testable import IntegrationKit

struct NativeMediaDownmixTests {
    private func rms(_ samples: ArraySlice<Float>) -> Float {
        sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(samples.count))
    }

    /// One speaker per channel: the left channel speaks in the first second, the right channel in the second. A mono
    /// decode must keep both; a converter that keeps only the first channel leaves the second second silent.
    @Test func stereoToMonoKeepsEveryChannel() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("downmix-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48_000, channels: 2, interleaved: false))
        let frames = 2 * 48_000
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)))
        buffer.frameLength = AVAudioFrameCount(frames)
        let left = try #require(buffer.floatChannelData?[0]), right = try #require(buffer.floatChannelData?[1])
        for i in 0..<frames {
            let tone = sin(Float(i) * 2 * .pi * 440 / 48_000) * 0.5
            left[i] = i < 48_000 ? tone : 0
            right[i] = i < 48_000 ? 0 : tone
        }
        try AVAudioFile(forWriting: url, settings: format.settings).write(from: buffer)

        let audio = try NativeMedia.audio(url, sampleRate: 16_000, channels: 1)
        #expect(audio.channels == 1)
        #expect(audio.samples.count >= 31_000)
        // Windows well inside each second, clear of the resampler's edges.
        let firstSpeaker = rms(audio.samples[2_000..<14_000]), secondSpeaker = rms(audio.samples[18_000..<30_000])
        #expect(firstSpeaker > 0.1, "left channel missing from the mono decode")
        #expect(secondSpeaker > 0.1, "right channel missing from the mono decode")
    }
}
