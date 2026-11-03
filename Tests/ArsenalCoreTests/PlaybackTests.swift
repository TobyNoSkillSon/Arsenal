// SPDX-License-Identifier: MIT
import AVFoundation
import Foundation
import Testing
@testable import ArsenalCore

@Test func playbackLoudnessDownwardOnly() {
    let rate = 24000
    let loud = (0..<rate).map { Float(sin(2 * .pi * 1000 * Double($0) / Double(rate)) * 0.8) }
    let m = PlaybackLoudness.measure(loud, rate: rate)
    let gain = PlaybackLoudness.gainDB(loudness: m.integrated, truePeak: m.truePeak)
    #expect(gain < 0 && abs(m.integrated + gain + 16) < 0.1)
    let quiet = PlaybackLoudness.measure(loud.map { $0 * 0.001 }, rate: rate)
    #expect(PlaybackLoudness.gainDB(loudness: quiet.integrated, truePeak: quiet.truePeak) == 0)
    #expect(PlaybackLoudness.gainDB(loudness: -.infinity, truePeak: -.infinity) == 0)
}

@MainActor @Test func playbackMutedQueueStopAndMetadataWithoutSound() async throws {
    let path = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-playback-" + UUID().uuidString + ".wav")
    defer { try? FileManager.default.removeItem(at: path) }
    let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1))
    let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 24000)); buffer.frameLength = 24000
    for index in 0..<24000 { buffer.floatChannelData![0][index] = Float(sin(2 * .pi * 1000 * Double(index) / 24000) * 0.8) }
    do { let file = try AVAudioFile(forWriting: path, settings: format.settings); try file.write(from: buffer) }
    let original = try Data(contentsOf: path), queue = PlaybackQueue(muted: false, outputEnabled: false)
    let item = try #require(try await queue.enqueue(path: path, requester: "Test Agent", model: "kokoro-82m", voice: "af_heart"))
    #expect(item.gainDB <= 0 && item.requester == "Test Agent")
    #expect(queue.snapshot().current?.voice == "af_heart" && !queue.snapshot().muted && !queue.snapshot().playing)
    _ = try await queue.enqueue(path: path, requester: "Other", model: "kokoro-82m")
    #expect(queue.snapshot().queued == 1)
    try queue.setMuted(true)
    #expect(try await queue.enqueue(path: path, requester: "Silent", model: "kokoro-82m") == nil)
    #expect(queue.snapshot().queued == 1)
    queue.stop(); #expect(queue.snapshot().current == nil && queue.snapshot().queued == 0)
    #expect(try Data(contentsOf: path) == original)
}

private actor PlaybackMeterGate {
    var started = false
    var continuation: CheckedContinuation<Double, Never>?
    func measure() async -> Double { started = true; return await withCheckedContinuation { continuation = $0 } }
    func release() { continuation?.resume(returning: -3); continuation = nil }
}
@MainActor @Test func playbackStopDuringMeteringDoesNotQueueLater() async throws {
    let gate = PlaybackMeterGate()
    let queue = PlaybackQueue(muted: false, outputEnabled: false, meter: { _ in await gate.measure() })
    let task = Task { try await queue.enqueue(path: URL(fileURLWithPath: "/unused.wav"), requester: "Test", model: "Test") }
    while !(await gate.started) { await Task.yield() }
    queue.stop(); await gate.release()
    #expect(try await task.value == nil)
    #expect(queue.snapshot().current == nil)
}

@MainActor @Test func muteSurvivesRelaunchBeforeSpeechIsAccepted() async throws {
    let base = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-mute-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: base) }
    let preference = base.appendingPathComponent("playback-muted.json")
    let first = PlaybackQueue(preference: preference, outputEnabled: false)
    #expect(!first.muted)
    try first.setMuted(true)
    let relaunched = PlaybackQueue(preference: preference, outputEnabled: false)
    #expect(relaunched.muted)
    // A nonexistent file proves the muted queue never meters or opens the audio device.
    #expect(try await relaunched.enqueue(path: base.appendingPathComponent("missing.wav"), requester: "MCP", model: "test") == nil)
    try relaunched.setMuted(false)
    #expect(!PlaybackQueue(preference: preference, outputEnabled: false).muted)
    #expect(PlaybackQueue(preference: preference, forceMuted: true, outputEnabled: false).muted)
    try Data("invalid".utf8).write(to: preference)
    #expect(PlaybackQueue(preference: preference, outputEnabled: false).muted)
}
