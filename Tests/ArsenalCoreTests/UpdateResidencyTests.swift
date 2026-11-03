// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private func reopenForUpdate(_ f: LocalCatalogFixture) async throws -> RealModelsService {
    try await RealModelsService.open(
        catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey, source: f.source, kinds: KindSpec.defaults,
        paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client, minuteSeconds: 1000)
}
@Test func v3UpdateRestoresDtypeModeResidencyAndKeepHot() async throws {
    // This tests update handoff, not idle expiry. The fixture's usual 150 ms residency can expire
    // the voice worker while the image worker starts on a busy Mac.
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices", "images"], minuteSeconds: 1000); defer { f.stop() }
    for id in ["fixture-chat", "fixture-voices", "fixture-images"] { try await f.service.get(id: id, precision: "bf16") }
    try await f.service.get(id: "fixture-chat", precision: "int4")
    try await f.service.load(id: "fixture-chat", precision: "int4", fast: false)
    _ = try await f.service.run(id: "fixture-voices", op: "speak", params: ["text": "hello", "stream": false])
    _ = try await f.service.run(id: "fixture-images", op: "generate", params: ["prompt": "hello"])
    try await f.service.setKeepHot(id: "fixture-voices", on: true)
    let modes = await f.service.loadedPreferences.mapValues(\.fast)
    let ids = try await f.service.beginUpdate()
    #expect(ids == ["fixture-chat", "fixture-images", "fixture-voices"])
    let file = f.paths.support.appendingPathComponent("state/update-residency.json")
    let record = try WireJSON.decode(Data(contentsOf: file))
    #expect(record["fixture-chat"]?["dtype"] == "int4" && record["fixture-chat"]?["fast"] == true)
    #expect(record["fixture-voices"]?["residency"] == "on_demand" && record["fixture-voices"]?["keep_hot"] == true)
    await f.service.shutdown()
    let restoredAt = Date().timeIntervalSince1970
    let restored = try await reopenForUpdate(f)
    await restored.restoreAfterUpdate(ids: ids)
    #expect(await restored.residents["fixture-chat"]?.residency == .manual)
    #expect(await restored.residents["fixture-voices"]?.residency == .onDemand)
    #expect(await restored.residents["fixture-voices"]?.keepHot == true)
    #expect(await restored.residents["fixture-images"]?.residency == .onDemand)
    #expect(await restored.loadedPreferences.mapValues(\.precision) == ["fixture-chat": "int4", "fixture-voices": "bf16", "fixture-images": "bf16"])
    #expect(await restored.loadedPreferences.mapValues(\.fast) == modes)
    #expect(await restored.residents["fixture-chat"]?.keepHot == false)
    #expect(await restored.residents["fixture-images"]?.keepHot == false)
    for id in ids { #expect(await (restored.residents[id]?.lastUsed ?? 0) >= restoredAt) }
    #expect(!FileManager.default.fileExists(atPath: file.path))
    await restored.shutdown()
}
@Test func v3UpdateRestoreFailureIsOnRowAndDoesNotThrow() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"]); defer { f.stop() }
    for id in ["fixture-chat", "fixture-voices"] {
        try await f.service.get(id: id, precision: "bf16"); try await f.service.load(id: id, precision: "bf16", fast: false)
    }
    let ids = try await f.service.beginUpdate(), file = f.paths.support.appendingPathComponent("state/update-residency.json")
    var record = try WireJSON.decode(Data(contentsOf: file)), failed = try #require(record["fixture-chat"]?.object)
    failed["dtype"] = "int4"; record["fixture-chat"] = .object(failed); try WireJSON.encode(record).write(to: file)
    await f.service.shutdown()
    let restored = try await reopenForUpdate(f)
    await restored.restoreAfterUpdate(ids: ids)
    #expect(await restored.rows().first { $0.id == "fixture-chat" }?.states["int4"]?.phase == "update_restore_failed")
    #expect(await restored.rows().first { $0.id == "fixture-chat" }?.state == .failed)
    #expect(await restored.rows().first { $0.id == "fixture-voices" }?.residentDType == "bf16")
    #expect(FileManager.default.fileExists(atPath: file.path))
    try await restored.get(id: "fixture-chat", precision: "int4")
    await restored.restoreAfterUpdate(ids: ids)
    #expect(await restored.rows().allSatisfy { $0.residentDType != nil })
    #expect(!FileManager.default.fileExists(atPath: file.path))
    await restored.shutdown()
}
@Test func v3CancelledOrRefusedUpdateClearsGateAndHandoffState() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16"); try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false)
    let file = f.paths.support.appendingPathComponent("state/update-residency.json")
    _ = try await f.service.beginUpdate(); await f.service.cancelUpdate()
    #expect(!FileManager.default.fileExists(atPath: file.path))
    try FileManager.default.createDirectory(at: file, withIntermediateDirectories: true)
    await #expect(throws: (any Error).self) { _ = try await f.service.beginUpdate() }
    #expect(await !f.service.updating)
    _ = try await f.service.run(id: "fixture-chat", op: "chat", params: ["prompt": "still running", "stream": false])
    try FileManager.default.removeItem(at: file)
    await f.service.shutdown()
}

@Test func v3UpdateRestoreReportsFullMemoryWithoutQueueing() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"]); defer { f.stop() }
    for id in ["fixture-chat", "fixture-voices"] {
        try await f.service.get(id: id, precision: "bf16"); try await f.service.load(id: id, precision: "bf16", fast: false)
    }
    try await f.service.setMemorySettings(.init(whenFull: .queue))
    let ids = try await f.service.beginUpdate(); await f.service.shutdown()
    let file = f.paths.support.appendingPathComponent("restore-memory.json")
    try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 1290))]).write(to: file)
    let restored = try await RealModelsService.open(
        catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey, source: f.source, kinds: KindSpec.defaults,
        paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client, minuteSeconds: 1000,
        memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": file.path], totalMB: 128_000))
    await restored.setLoadPeak("fixture-chat", gb: 1.28)
    await restored.setLoadPeak("fixture-voices", gb: 1.28)
    let restore = Task { await restored.restoreAfterUpdate(ids: ids) }
    let deadline = Task {
        try await Task.sleep(for: .seconds(1)); restore.cancel()
    }
    await restore.value; deadline.cancel()
    let rows = await restored.rows()
    #expect(rows.first { $0.id == "fixture-chat" }?.residentDType == "bf16")
    #expect(rows.first { $0.id == "fixture-voices" }?.states["bf16"]?.error?.contains("This model needs about 1.19 GB at BF16") == true)
    #expect(await restored.memorySettings().whenFull == .queue)
    #expect(FileManager.default.fileExists(atPath: f.paths.support.appendingPathComponent("state/update-residency.json").path))
    await restored.shutdown()
}

@Test func v3UpdateRestoreMissingSnapshotShowsRowFailureAndContinues() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"]); defer { f.stop() }
    for id in ["fixture-chat", "fixture-voices"] {
        try await f.service.get(id: id, precision: "bf16"); try await f.service.load(id: id, precision: "bf16", fast: false)
    }
    let ids = try await f.service.beginUpdate(); await f.service.shutdown()
    try FileManager.default.removeItem(at: f.paths.directory("weights/snapshots").appendingPathComponent("fixture-chat"))
    let restored = try await reopenForUpdate(f)
    await restored.restoreAfterUpdate(ids: ids)
    let rows = await restored.rows()
    #expect(rows.first { $0.id == "fixture-chat" }?.states["bf16"]?.phase == "update_restore_failed")
    #expect(rows.first { $0.id == "fixture-chat" }?.states["bf16"]?.error?.contains("\(MCPNames.prefix)download") == true)
    #expect(rows.first { $0.id == "fixture-voices" }?.residentDType == "bf16")
    #expect(FileManager.default.fileExists(atPath: f.paths.support.appendingPathComponent("state/update-residency.json").path))
    await restored.shutdown()
}

private extension RealModelsService {
    func v3DisableFast(_ id: String) { modelRows[id]?.fastExact = false; modelRows[id]?.hasFastKernels = false }
}
@Test func v3UpdatePrunesPermanentFailuresButKeepsRetryableFailures() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices", "images"]); defer { f.stop() }
    for id in ["fixture-chat", "fixture-voices", "fixture-images"] {
        try await f.service.get(id: id, precision: "bf16"); try await f.service.load(id: id, precision: "bf16", fast: false)
    }
    let ids = try await f.service.beginUpdate(), file = f.paths.support.appendingPathComponent("state/update-residency.json")
    var record = try WireJSON.decode(Data(contentsOf: file)), voice = try #require(record["fixture-voices"]?.object)
    voice["fast"] = true; record["fixture-voices"] = .object(voice); try WireJSON.encode(record).write(to: file)
    await f.service.shutdown()
    try FileManager.default.removeItem(at: f.paths.directory("weights/snapshots").appendingPathComponent("fixture-images"))
    let restored = try await reopenForUpdate(f)
    try await restored.remove(id: "fixture-chat", deleteWeights: true)
    await restored.v3DisableFast("fixture-voices")
    await restored.restoreAfterUpdate(ids: ids)
    #expect(try Set(WireJSON.decode(Data(contentsOf: file)).keys) == Set(["fixture-voices", "fixture-images"]))
    let errors = await restored.modelRows.mapValues { $0.states["bf16"]?.error }
    // Off the table and no longer downloaded: nothing to restore, so it is skipped without an error.
    #expect((errors["fixture-chat"] ?? nil) == nil)
    #expect(await restored.residents["fixture-voices"] != nil)
    #expect(await restored.modelRows["fixture-voices"]?.fast == true)
    #expect(errors["fixture-images"]??.contains("\(MCPNames.prefix)download") == true)
    try await restored.get(id: "fixture-images", precision: "bf16")
    await restored.restoreAfterUpdate(ids: ids)
    #expect(await restored.modelRows["fixture-voices"]?.states["bf16"]?.error == errors["fixture-voices"]!)
    #expect(await restored.residents["fixture-images"] != nil)
    #expect(!FileManager.default.fileExists(atPath: file.path))
    await restored.shutdown()
}

/// A downloaded model that is off the table (an agent's, or one the user removed from the table) restores through the
/// off-table path, with no error on its row.
@Test func v3UpdateRestoresOffTableResidentWithoutError() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"]); defer { f.stop() }
    for id in ["fixture-chat", "fixture-voices"] {
        try await f.service.get(id: id, precision: "bf16"); try await f.service.load(id: id, precision: "bf16", fast: false)
    }
    let ids = try await f.service.beginUpdate(), file = f.paths.support.appendingPathComponent("state/update-residency.json")
    await f.service.shutdown()
    let restored = try await reopenForUpdate(f)
    try await restored.remove(id: "fixture-chat", deleteWeights: false)
    await restored.restoreAfterUpdate(ids: ids)
    #expect(await restored.residents["fixture-chat"]?.residency == .manual)
    #expect(await restored.modelRows["fixture-chat"]?.states["bf16"]?.error == nil)
    #expect(await restored.modelRows["fixture-chat"]?.states["bf16"]?.phase != "update_restore_failed")
    #expect(await restored.residents["fixture-voices"] != nil)
    #expect(!FileManager.default.fileExists(atPath: file.path))
    await restored.shutdown()
}
