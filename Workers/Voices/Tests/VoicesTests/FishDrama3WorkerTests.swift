// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import Testing
@testable import VoicesWorker

/// The voices worker's guards for Drama 3: its sampler controls only, cloning with a transcript, and no presets.
@Suite struct FishDrama3WorkerTests {
    let id = "fish-drama-3"

    @Test func samplingControlsAreTheReferenceOnes() throws {
        try SpeechOperation.check(["op": "speak", "max_tokens": 96, "temperature": 0.7, "top_p": 0.7, "top_k": 30], modelID: id)
        let refused: [JSONObject] = [["repetition_penalty": 1.05], ["guidance": 3], ["top_k": 0], ["max_tokens": 4097], ["temperature": 2.5], ["top_p": 0]]
        #expect(refused.count == 6)
        for value in refused { #expect(throws: (any Error).self) { try SpeechOperation.check(value, modelID: id) } }
    }

    @Test func cloningNeedsTheClipAndItsTranscript() throws {
        try SpeechOperation.check(["op": "clone_voice", "audio": ["path": "/tmp/a.wav"], "transcript": "Hello.", "name": "Mine"], modelID: id)
        #expect(throws: (any Error).self) { try SpeechOperation.check(["op": "clone_voice", "audio": ["path": "/tmp/a.wav"], "name": "Mine"], modelID: id) }
        #expect(throws: (any Error).self) { try SpeechOperation.check(["op": "design_voice", "description": "warm"], modelID: id) }
        try SpeechOperation.check(["op": "speak", "reference": ["path": "/tmp/a.wav", "text": "Hello."]], modelID: id)
    }
}
