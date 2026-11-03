// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit

/// Audio with 3–8 channels as librosa.load(sr, mono=True) sees it. AVAudioFormat has no plain 3–8 channel PCM target
/// (it returns nil without a channel layout), so the shared reader averages the channels at the file's own rate, then
/// resamples the mono signal; both steps are linear, so the order does not change the mean. It streams in 64k-frame
/// blocks, accepts a compressed file's estimated length, and reports a corrupt file as invalid_request (422).
/// EmbeddingGemma 2 and LCO-Embedding share it.
enum SurroundAudio {
    static let block = 65_536

    static func mono(_ url: URL, sampleRate: Int, maximumSamples: Int) throws -> [Float] {
        try NativeMedia.audio(url, sampleRate: sampleRate, channels: 1, mix: .mean, maximumFrames: maximumSamples).samples
    }
}
