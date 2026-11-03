// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private actor VoiceSettingsWorkerSpy: WorkerArtifact {
    private(set) var calls = 0
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws { calls += 1 }
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        calls += 1
        throw WorkerFailure(code: "test", message: "Settings must never start a worker.")
    }
}

@Test func savedVoiceFingerprintChangesWithoutStartingAWorkerOrEmittingRefreshEvents() async throws {
    let spy = VoiceSettingsWorkerSpy()
    let fixture = try await makeLocalCatalogFixture(
        kinds: ["voices"], artifacts: spy,
        extraTable: [
            "setting": .array([
                .table([
                    "id": .string("voice"), "title": .string("Voice"), "type": .string("voice"), "default": .string("af_heart"),
                    "applies": .string("request"), "maps_to": .string("request.voice")
                ])
            ])
        ])
    defer { fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent()) }
    let events = await fixture.service.events()
    let received = VoiceInventoryEvents()
    // Subscription sends one initial catalog snapshot; only subsequent events could trigger a refresh loop.
    let watcher = Task { for await _ in events.dropFirst() { await received.increment() } }
    defer { watcher.cancel() }
    func fingerprint() async throws -> String {
        try #require(try await fixture.service.settings(id: "fixture-voices").first { $0.id == "voice" }?.declaration["saved_voices_sha256"]?.string)
    }
    let initial = try await fingerprint()
    #expect(initial.count == 64)
    for _ in 0..<20 { #expect(try await fingerprint() == initial) }
    let directory = fixture.paths.support.appendingPathComponent("voices/fixture-voices")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let file = directory.appendingPathComponent("voice-" + UUID().uuidString + ".json")
    try WireJSON.encode(["id": .string(file.deletingPathExtension().lastPathComponent), "name": "Narrator"]).write(to: file)
    let added = try await fingerprint()
    #expect(added != initial)
    for _ in 0..<20 { #expect(try await fingerprint() == added) }
    try WireJSON.encode(["id": .string(file.deletingPathExtension().lastPathComponent), "name": "Renamed narrator"]).write(to: file)
    #expect(try await fingerprint() != added)
    try FileManager.default.removeItem(at: file)
    #expect(try await fingerprint() == initial)
    try await Task.sleep(for: .milliseconds(50))
    let workerCalls = await spy.calls, eventCount = await received.count
    #expect(workerCalls == 0 && eventCount == 0)
    #expect(await fixture.service.rows().allSatisfy { $0.residentDType == nil })
    await fixture.service.shutdown()
}

private actor VoiceInventoryEvents {
    private(set) var count = 0
    func increment() { count += 1 }
}
