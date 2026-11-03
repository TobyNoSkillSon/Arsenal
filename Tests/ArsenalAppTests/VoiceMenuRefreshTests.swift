// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct VoiceMenuRefreshTests {
    @Test func refreshingTheMenuDoesNotLoadAnUnloadedVoiceForItsInventory() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-unloaded-voice-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        var row = ModelRow(id: "voice-fixture", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])
        row.tools = ["voice"]
        let service = StubModelsService(rows: [row], supportDirectory: support)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start(); defer { api.stop() }
        let client = ArsenalClient(port: try #require(api.port))
        let center = ToolCenter(client: { client }, playback: FixturePlayback())
        await center.refreshVoices([row], service: service)
        #expect(await service.rows().first?.state == .ready)
        #expect(center.voices.isEmpty && center.errors.isEmpty)
    }

    @Test func cachedPresetsStillReadAnAPIChangedVoiceSelection() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-voice-menu-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let row = ModelRow(id: "kokoro-82m", name: "Kokoro", kind: "voices", states: ["bf16": .init(state: .loaded)])
        let metadata: [String: JSONObject] = [row.id: ["setting": [["id": "voice", "type": "voice", "default": "af_heart", "applies": "request"]]]]
        let service = StubModelsService(rows: [row], metadata: metadata, supportDirectory: support)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        let client = ArsenalClient(port: try #require(api.port))
        let center = ToolCenter(client: { client }, playback: FixturePlayback())
        let rows = await service.rows()
        await center.refreshVoices(rows, service: service)
        #expect(center.currentVoice[row.id] == "af_heart" && center.voices[row.id]?.first?.id == "af_heart")
        let cached = center.voices[row.id]
        try await client.setSettings(row.id, ["voice": "bm_lewis"])
        await center.refreshVoices(rows, service: service)
        #expect(center.currentVoice[row.id] == "bm_lewis" && center.voices[row.id] == cached && center.errors[row.id] == nil)
        try await client.setSettings(row.id, ["voice": "af_heart"])
        await center.refreshVoices(rows, service: service)
        #expect(center.currentVoice[row.id] == "af_heart")
    }

    @Test func voiceInventoryInvalidatesOnceAndStatusRefreshesShareTheRequest() async throws {
        let service = RefreshReadService()
        await service.setVoiceDelay(.milliseconds(150))
        let center = ToolCenter(client: { nil }, playback: FixturePlayback())
        var row = ModelRow(id: "fixture", name: "Voice", kind: "voices", states: ["bf16": .init(state: .loaded)]); row.tools = ["voice"]
        let rows = [row]
        let first = Task { await center.refreshVoices(rows, service: service) }
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while await service.counts[.voices] != 1 {
            guard ContinuousClock.now < deadline else { throw ToolError("Voice read did not start") }
            try await Task.sleep(for: .milliseconds(5))
        }
        // A worker status event starts a second menu refresh before the first voices request completes.
        await center.refreshVoices(rows, service: service); await first.value
        #expect(await service.counts[.voices] == 1 && center.voices[row.id]?.map(\.id) == ["af_heart"])
        for _ in 0..<10 { await center.refreshVoices(rows, service: service) }
        #expect(await service.counts[.voices] == 1)
        await service.setInventory("added", voice: "voice-saved", saved: true)
        await center.refreshVoices(rows, service: service)
        #expect(await service.counts[.voices] == 2 && center.currentVoice[row.id] == "voice-saved")
        #expect(center.voices[row.id]?.map(\.id) == ["af_heart", "voice-saved"])
        for _ in 0..<10 { await center.refreshVoices(rows, service: service) }
        #expect(await service.counts[.voices] == 2)
        await service.setInventory("empty", voice: "af_heart", saved: false)
        await center.refreshVoices(rows, service: service)
        #expect(await service.counts[.voices] == 3 && center.currentVoice[row.id] == "af_heart" && center.voices[row.id]?.count == 1)
        for _ in 0..<10 { await center.refreshVoices(rows, service: service) }
        #expect(await service.counts[.voices] == 3 && center.errors[row.id] == nil)
    }

}
