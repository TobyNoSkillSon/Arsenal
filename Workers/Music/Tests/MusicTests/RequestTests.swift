// SPDX-License-Identifier: MIT
import Testing
import ArsenalWire
@testable import MusicWorker

/// Long songs are sized against the per-request artifact limit before any generation starts.
struct RequestTests {
    @Test func outputBoundUsesTheEngineMaximumAndTheArtifactLimit() throws {
        let one = try MusicRequests.decode(["prompt": "piano", "duration": 600], maxSeconds: 600)
        // 600 s x 48 kHz x 2 channels x 2 bytes = 115,200,000 bytes, plus 1% and 64 KiB.
        #expect(MusicRequests.outputBound(one, format: .wav, maxSeconds: 600) == 115_200_000 + 1_152_000 + 65_536)
        let open = try MusicRequests.decode(["prompt": "piano"], maxSeconds: 360)
        #expect(MusicRequests.outputBound(open, format: .wav, maxSeconds: 360) == 69_120_000 + 691_200 + 65_536)
        // The largest request the schemas allow (16 songs, 600 s, 32-bit WAV) fits the 4 GiB per-request limit.
        let largest = try MusicRequests.decode(["prompt": "piano", "duration": 600, "batch_size": 16, "format": "wav32"], maxSeconds: 600)
        #expect(MusicRequests.outputBound(largest, format: .wav32, maxSeconds: 600) <= MusicRequests.artifactLimit)
        #expect(MusicRequests.artifactLimit == 4 * 1024 * 1024 * 1024)
    }

    @Test func unknownFormatsFailBeforeGeneration() {
        #expect(throws: WorkerFailure.self) { try MusicRequests.decode(["prompt": "piano", "format": "mp3"], maxSeconds: 600) }
    }

    @Test func progressNamesTheStageAndSong() {
        guard case .json(let event) = MusicSupport.progress(SongProgress(stage: .decoding, fraction: 0.404, item: 1), batch: 2) else {
            Issue.record("progress is a JSON event"); return
        }
        #expect(event["label"]?.string == "Song 2 of 2 \u{00b7} Decoding audio 40%")
        #expect(event["progress"]?.number == 0.404 && event["step"] == nil)
    }
}
