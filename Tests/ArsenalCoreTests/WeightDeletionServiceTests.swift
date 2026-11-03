// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The service's side of per-precision deletion, through the real Get and Load of the fixture catalog: the on-disk listing,
/// the table's ready state after a corrupt file, the refusal while a precision is loaded, repeated deletes, and whole-model
/// deletion. The store's file logic is covered in WeightDeletionTests.
@Test func weightDeletionServiceNeverReportsCorruptWeightsReadyAndRepeatsAreNoOps() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let service = f.service
    let id = "fixture-chat"
    try await service.get(id: id, precision: "bf16")
    #expect(try #require(await service.rows().first { $0.id == id }).states["bf16"]?.state == .ready)
    let listed = try await service.onDiskModels()
    let model = try #require(listed.first { $0.id == id })
    #expect(model.offered && model.precisions.map(\.dtype) == ["bf16"] && model.precisions[0].bytes > 3 && model.precisions[0].inUse == false)

    // Loaded: the precision is in use, and its deletion refuses until the model is unloaded.
    try await service.load(id: id, precision: "bf16", fast: false)
    let loaded = try await service.onDiskModels()
    #expect(loaded.first { $0.id == id }?.precisions.first?.inUse == true)
    do {
        _ = try await service.deletePrecision(id: id, precision: "bf16")
        Issue.record("a loaded precision was deleted")
    } catch {
        #expect(error.localizedDescription.contains("Unload"), "message was: \(error.localizedDescription)")
    }
    try await service.unload(id: id)

    // The blob changes under the snapshot: the next refresh must not report the precision ready.
    let blob = try await service.fetcher.store.blobURL(sha256: FileDigest.sha256(Data([12, 34, 56])))
    try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: blob.path)
    try Data([99, 98, 97]).write(to: blob)
    await service.refreshInstalled(forceDownloadSizes: true)
    // Not ready weights, so it is read from Get more models.
    let corrupt = try #require(await catalogRow(service, id))
    #expect(try #require(corrupt.states["bf16"]).state != .ready)

    // The corrupt bytes are still on disk and listed; deleting them frees them once, and a repeat frees nothing.
    let freed = try await service.deletePrecision(id: id, precision: "bf16")
    #expect(freed >= 3)
    #expect(try await service.deletePrecision(id: id, precision: "bf16") == 0)
    let after = try await service.onDiskModels()
    #expect(after.contains { $0.id == id } == false)
    await service.shutdown()
}

/// A converted 16-bit copy is the source of the quantized copies built from it: they verify through it, so deleting it would
/// strand them. The rule names them (the service refuses the delete and the table's trash says why). A plain 16-bit copy has none.
@Test func aConvertedSixteenBitCopyNamesTheQuantizedCopiesBuiltFromIt() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let service = f.service
    var plain = try await service.manifest("fixture-chat")
    plain.derive = nil
    #expect(RealModelsService.dependents(of: "bf16", in: plain, onDisk: ["bf16", "int4"]).isEmpty)
    var converted = plain
    converted.derive = ["bf16": "cast:fixture-chat"]
    #expect(converted.basePrecision == "bf16")
    #expect(RealModelsService.dependents(of: "bf16", in: converted, onDisk: ["bf16", "int4"]) == ["int4"])
    #expect(RealModelsService.dependents(of: "bf16", in: converted, onDisk: ["bf16"]).isEmpty, "nothing built from it is on disk")
    #expect(RealModelsService.dependents(of: "int4", in: converted, onDisk: ["bf16", "int4"]).isEmpty, "only the 16-bit copy has dependents")
    await service.shutdown()
}

/// The table holds only models with weights on this Mac, so deleting them all takes the model off the table;
/// it stays in Get more models, recorded, to download again.
@Test func weightDeletionServiceDeleteAllClearsWeightsAndTakesTheModelOffTheTable() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let service = f.service
    try await service.get(id: "fixture-chat", precision: "bf16")
    #expect(try await service.orphanBytes() == 0)
    #expect(try await service.deleteOrphans() == 0)
    let freed = try await service.deleteAllWeights(id: "fixture-chat")
    #expect(freed > 3)
    let remaining = try await service.onDiskModels()
    #expect(remaining.isEmpty)
    #expect(await service.rows().contains { $0.id == "fixture-chat" } == false)
    #expect(await service.catalog().first { $0.id == "fixture-chat" }?.added == false)
    let removed = try #require(await catalogRow(service, "fixture-chat"))
    #expect(try #require(removed.states["bf16"]).state != .ready)
    await service.shutdown()
}

/// A loaded model is refused by both delete routes, idle or not: its weights stay, and it stays in the table, until the user
/// unloads it. Nothing is unloaded behind the user's back.
@Test func weightDeletionServiceRefusesToDeleteALoadedModel() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let service = f.service
    try await service.get(id: "fixture-chat", precision: "bf16")
    try await service.load(id: "fixture-chat", precision: "bf16", fast: false)
    do {
        try await service.remove(id: "fixture-chat", deleteWeights: true)
        Issue.record("removing a loaded model's weights must be refused")
    } catch let error as WorkerFailure {
        #expect(error.code == "busy" && error.message.hasPrefix("Unload "), "\(error)")
    }
    do {
        _ = try await service.deleteAllWeights(id: "fixture-chat")
        Issue.record("deleting a loaded model's weights must be refused")
    } catch let error as WorkerFailure {
        #expect(error.code == "busy", "\(error)")
    }
    #expect(try await service.onDiskModels().first { $0.id == "fixture-chat" }?.precisions.first?.inUse == true)
    #expect(await service.rows().contains { $0.id == "fixture-chat" })
    await service.shutdown()
}

/// Whole-model deletion removes every weight, revision and cache, and leaves the model folder: the signed definition and its
/// measurement results are not weights and cannot be rebuilt.
@Test func weightDeletionServiceDeleteAllKeepsTheModelFolder() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let service = f.service
    try await service.get(id: "fixture-chat", precision: "bf16")
    let results = f.paths.directory("models").appendingPathComponent("fixture-chat/results/local")
    try FileManager.default.createDirectory(at: results, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: results.appendingPathComponent("measured.json"))
    #expect(try await service.deleteAllWeights(id: "fixture-chat") > 3)
    #expect(try await service.onDiskModels().isEmpty)
    #expect(FileManager.default.fileExists(atPath: results.appendingPathComponent("measured.json").path))
    await service.shutdown()
}

/// Item D: a precision delete that starts while a revision swap runs must re-check for a Get after the swap, so a Get that
/// starts during the wait refuses the delete.
@Test func deletingAPrecisionIsRefusedWhenAGetStartsDuringARevisionSwap() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let service = f.service
    try await service.get(id: "fixture-chat", precision: "bf16")
    await service.startRevisionSwap("fixture-chat", for: .milliseconds(400))
    let deletion = Task { try await service.deletePrecision(id: "fixture-chat", precision: "bf16") }
    // Wait (bounded) until the deletion is parked behind the swap.
    for _ in 0..<400 {
        if await service.waitingRevisionRuns["fixture-chat", default: 0] > 0 { break }
        try await Task.sleep(for: .milliseconds(5))
    }
    #expect(await service.waitingRevisionRuns["fixture-chat", default: 0] > 0, "the deletion never waited for the swap")
    // A Get for the model starts during the wait and is still running when the swap finishes.
    _ = await service.holdGetRunning("fixture-chat")
    do {
        _ = try await deletion.value
        Issue.record("bf16 was deleted while a Get was running")
    } catch {
        #expect(error.localizedDescription.contains("Finish Get or Load"), "message was: \(error.localizedDescription)")
    }
    await service.finishHeldGet("fixture-chat")
    // The refused precision is still on disk and still listed.
    #expect(try await service.onDiskModels().first { $0.id == "fixture-chat" }?.precisions.map(\.dtype) == ["bf16"])
    await service.shutdown()
}

private extension RealModelsService {
    /// A Get registered as running for `id` until `finishHeldGet`: the state a real Get holds for its whole download. A real
    /// Get on localhost finishes before a test can race it, so the test holds the registration itself.
    func holdGetRunning(_ id: String) -> Task<Void, any Error> {
        let running = Task { try await Task.sleep(for: .seconds(60)) }
        gets[id] = running
        return running
    }
    func finishHeldGet(_ id: String) { gets[id]?.cancel(); gets[id] = nil }
    /// A revision swap in progress for `id`, finished `delay` later the way the swap does: admission waits, then it clears.
    func startRevisionSwap(_ id: String, for delay: Duration) {
        reloading.insert(id)
        swapTasks[id] = Task {
            try? await Task.sleep(for: delay)
            reloading.remove(id); swapTasks[id] = nil; signalMemoryChange()
        }
    }
}
