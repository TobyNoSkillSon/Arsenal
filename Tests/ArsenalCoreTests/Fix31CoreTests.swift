// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private actor Fix31Gate {
    private var waiter: CheckedContinuation<Void, Never>?
    private var released = false
    func wait() async { if !released { await withCheckedContinuation { waiter = $0 } } }
    func release() { released = true; waiter?.resume(); waiter = nil }
}
private extension RealModelsService {
    func fixtureSwap(_ id: String, gate: Fix31Gate) {
        reloading.insert(id)
        swapTasks[id] = Task {
            defer { swapTasks[id] = nil; reloading.remove(id); signalMemoryChange() }
            await gate.wait()
            guard !Task.isCancelled else { return }
            try? await reserveMemory(id: id, precision: "bf16", estimateMB: 1)
        }
    }
    func fixtureRevisionGate(_ local: LocalModelFolder, previous: String, results: [ResultsFile]) throws {
        config.localCodeConsent[local.manifest.id] = previous
        resultCache[local.manifest.id] = results
        try requireSwapEvidence(local)
    }
    func fixtureChecks(_ local: LocalModelFolder, results: [ResultsFile]) -> ModelChecksStatus? {
        let id = local.manifest.id
        localModels[id] = local; resultCache[id] = results
        modelRows[id] = ModelRow(id: id, name: id, kind: local.manifest.kind)
        updateChecks(local.manifest)
        return modelRows[id]?.checks
    }
}
@Suite(.serialized) struct Fix31CoreTests {
    @Test func updateWaitsForAnExistingSwapAndAllowsItsReservation() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let gate = Fix31Gate()
        await f.service.fixtureSwap("fixture-voices", gate: gate)
        let updating = Task { try await f.service.beginUpdate() }
        for _ in 0..<100 {
            if await f.service.updating { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        let updateStarted = await f.service.updating, swapPresent = await f.service.swapTasks["fixture-voices"] != nil
        #expect(updateStarted && swapPresent)
        await gate.release()
        _ = try await updating.value
        #expect(await f.service.loadReservations["fixture-voices"] != nil)
        #expect(await f.service.swapTasks.isEmpty)
        await f.service.cancelUpdate(); await f.service.shutdown()
    }
    @Test(arguments: [false, true]) func removeAndDeleteCancelAndAwaitSwap(delete: Bool) async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        try await f.service.get(id: "fixture-voices", precision: "bf16")
        let gate = Fix31Gate()
        await f.service.fixtureSwap("fixture-voices", gate: gate)
        let removal = Task {
            if delete { try await f.service.delete(id: "fixture-voices") } else { try await f.service.remove(id: "fixture-voices") }
        }
        try await Task.sleep(for: .milliseconds(30))
        #expect(await f.service.reloading.contains("fixture-voices"))
        await gate.release(); try await removal.value
        let swapsEmpty = await f.service.swapTasks.isEmpty, residentsEmpty = await f.service.residents.isEmpty
        #expect(swapsEmpty && residentsEmpty)
        #expect(await f.service.rejectedRevisionDigests.isEmpty)
        // Delete takes the weights and so the row off the table; Remove alone leaves the weights, which keep it there.
        if delete {
            #expect(await catalogRow(f.service, "fixture-voices")?.states["bf16"]?.state == .notInstalled)
            #expect(await f.service.rows().isEmpty)
        } else {
            #expect(await f.service.rows().map(\.id) == ["fixture-voices"])
        }
        await f.service.shutdown()
    }
    @Test func compatibleCheckWinsOverNewerStaleCheckAndPathOrder() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = f.paths.support.appendingPathComponent("local-models/fixture-layers")
        try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("Tests/Fixtures/DropIn/fixture-layers"), to: folder)
        let evidence = folder.appendingPathComponent("results/retained.txt")
        try FileManager.default.createDirectory(at: evidence.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("fixed fixture evidence".utf8).write(to: evidence)
        let local = try LocalModelFolders.discoverSnapshot(folder)
        var current = ResultsFile(
            model: local.manifest.id, kind: local.manifest.kind, chip: "fixture", osBuild: "test", appBuild: "test", runner: "swift",
            workerDigest: "fixture", precision: "bf16", mode: "exact", suite: .init(id: "fixture", sha256: "fixture"),
            profile: "quick", stub: false, gatePassed: true, conditions: .init(locked: false, idle: [], run: []), metrics: [:], cases: [])
        current.checks = .init(
            protocolStatus: .pass, implementationFingerprint: try ImplementationEvidence.fingerprint(local),
            folderDigest: local.contentDigest, fixtureIdentity: ImplementationEvidence.fixtures(local), evidence: ["results/retained.txt"])
        current.checks?.evidenceHashes = ["results/retained.txt": try FileDigest.sha256(evidence)]
        current.date = "2026-10-08T10:00:00Z"
        var stale = current; stale.date = "2026-10-08T11:00:00Z"; stale.checks?.implementationFingerprint = "stale"
        var older = current; older.date = "2026-10-08T09:00:00Z"; older.checks?.protocolStatus = .fail
        let selected = await f.service.fixtureChecks(local, results: [stale, older, current])
        #expect(selected?.protocolStatus == .pass && selected?.stale == false)
        await f.service.shutdown()
    }
    @Test(
        arguments: ["parity", "speed", "task", "protocol", "not-run"],
        [
            "2026-10-08T10:00:00.500Z", "2026-10-08T09:00:01-01:00", "invalid-date"
        ])
    func approvalRejectsTheNewestCompatibleFailure(status: String, date: String) async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = f.paths.support.appendingPathComponent("local-models/fixture-layers")
        try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("Tests/Fixtures/DropIn/fixture-layers"), to: folder)
        let old = try LocalModelFolders.discoverSnapshot(folder)
        _ = try LocalCodeSnapshot.install(old, support: f.paths.support)
        let runner = folder.appendingPathComponent("runners/swift/Sources/FixtureLayers/main.swift")
        try (String(contentsOf: runner, encoding: .utf8) + "\n# revised implementation\n").write(to: runner, atomically: true, encoding: .utf8)
        let evidence = folder.appendingPathComponent("results/retained.txt")
        try FileManager.default.createDirectory(at: evidence.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("fixed fixture evidence".utf8).write(to: evidence)
        let local = try LocalModelFolders.discoverSnapshot(folder)
        var passing = ResultsFile(
            model: local.manifest.id, kind: local.manifest.kind, chip: "fixture", osBuild: "test", appBuild: "test", runner: "swift",
            workerDigest: "fixture", precision: "bf16", mode: "exact", suite: .init(id: "fixture", sha256: "fixture"),
            profile: "quick", stub: false, gatePassed: true, conditions: .init(locked: false, idle: [], run: []), metrics: ["latency_s": 1], cases: [])
        passing.checks = .init(
            protocolStatus: .pass, parity: .pass, task: .pass, reference: ["independent": true],
            implementationFingerprint: try ImplementationEvidence.fingerprint(local), folderDigest: local.contentDigest,
            fixtureIdentity: ImplementationEvidence.fixtures(local), evidence: ["results/retained.txt"])
        passing.checks?.speed = .pass; passing.checks?.evidenceHashes = ["results/retained.txt": try FileDigest.sha256(evidence)]
        passing.date = "2026-10-08T10:00:00Z"
        try await f.service.fixtureRevisionGate(local, previous: old.contentDigest, results: [passing])
        var failing = passing; failing.date = date
        switch status {
        case "parity": failing.checks?.parity = .fail
        case "speed": failing.checks?.speed = .fail
        case "protocol": failing.checks?.protocolStatus = .fail
        case "not-run": failing.checks?.task = .notRun
        default: failing.checks?.task = .fail
        }
        // The cache's path order deliberately puts the older pass first.
        await #expect(throws: WorkerFailure.self) {
            try await f.service.fixtureRevisionGate(local, previous: old.contentDigest, results: [passing, failing])
        }
        let selected = await f.service.fixtureChecks(local, results: [passing, failing])
        var expected = failing.checks; expected?.stale = failing.recordedAt == nil
        #expect(selected == expected)
        var stale = failing; stale.date = "2026-10-08T12:00:00Z"; stale.checks?.implementationFingerprint = "stale"
        try await f.service.fixtureRevisionGate(local, previous: old.contentDigest, results: [stale, passing])
        await f.service.shutdown()
    }

}
