// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalWire
@testable import VoicesWorker

@Test func requestAdmissionAndCancellationStayBounded() throws {
    let requests = Requests()
    let active = try #require(requests.admit(7))
    #expect(requests.admit(8) == nil)
    #expect(!requests.cancel(8))
    #expect(!active.isCancelled)
    #expect(requests.cancel(7))
    #expect(active.isCancelled)
    requests.release()
    let next = try #require(requests.admit(8))
    #expect(!next.isCancelled)
    requests.stop()
    #expect(next.isCancelled)
}

@Test func speechSamplingContractRejectsInvalidOrIgnoredControls() throws {
    for id in ["qwen3-tts-base", "qwen3-tts-voicedesign", "breeze-tts-2"] {
        try SpeechOperation.check(["op": "speak", "max_tokens": 64, "temperature": 0.7, "top_p": 0.8, "top_k": 50, "repetition_penalty": 1.05], modelID: id)
        for value: JSONObject in [["max_tokens": 1.5], ["temperature": "0.7"], ["top_p": 0], ["max_tokens": 8193], ["top_k": true], ["repetition_penalty": 0]] {
            #expect(throws: (any Error).self) { try SpeechOperation.check(value, modelID: id) }
        }
    }
    try SpeechOperation.check(["guidance": 3], modelID: "breeze-tts-2")
    #expect(throws: (any Error).self) { try SpeechOperation.check(["max_tokens": 751], modelID: "breeze-tts-2") }
    for id in ["kokoro-82m", "pocket-tts", "qwen3-tts-base"] {
        #expect(throws: (any Error).self) { try SpeechOperation.check(["guidance": 3], modelID: id) }
    }
}

@Test func savedDesignKeepsSamplingIdentity() throws {
    let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-saved-sampling-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: support) }
    let store = SavedVoices(model: "breeze-tts-2", support: support)
    let design: JSONObject = ["seed": 42, "model_revision": "revision", "preview_text": "Hello.", "generation": ["guidance": 3, "temperature": 0.7]]
    let saved = try store.save(name: "Narrator", description: "Calm narrator", design: design)
    let request = try store.resolve(["voice": .string(saved.id), "text": "Hello.", "guidance": 10], modelRevision: "revision")
    #expect(request["guidance"] == 3 && request["temperature"] == 0.7 && request["seed"] == 42)
}

@Test func savedDesignNeverRestoresPreviewOutputLimit() throws {
    let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-saved-limit-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: support) }
    let store = SavedVoices(model: "breeze-tts-2", support: support)
    let design: JSONObject = ["seed": 42, "generation": ["max_tokens": 64, "temperature": 0.7]]
    let saved = try store.save(name: "Narrator", description: "Calm narrator", design: design)
    #expect(saved.generation?["max_tokens"] == nil)
    #expect(try store.resolve(["voice": .string(saved.id), "max_tokens": 750])["max_tokens"] == 750)
    // Already-saved voices from the previous build may still contain the preview limit.
    var legacy = saved; legacy.generation?["max_tokens"] = 64
    try JSONEncoder().encode(legacy).write(to: store.directory.appendingPathComponent(saved.id + ".json"))
    #expect(try store.resolve(["voice": .string(saved.id), "max_tokens": 750])["max_tokens"] == 750)
    #expect(try store.resolve(["voice": .string(saved.id)])["max_tokens"] == nil)
}

@Test func pocketCloneAdmitsAudioWithoutTranscript() throws {
    try SpeechOperation.check(["op": "clone_voice", "audio": ["path": "/tmp/ref.wav"], "name": "Narrator"], modelID: "pocket-tts")
    #expect(throws: (any Error).self) {
        try SpeechOperation.check(["op": "clone_voice", "audio": ["path": "/tmp/ref.wav"], "name": "Narrator"], modelID: "qwen3-tts-base")
    }
}
