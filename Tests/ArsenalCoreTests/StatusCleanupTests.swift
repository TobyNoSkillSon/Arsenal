// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private actor StatusQueryGate {
    private var waiter: CheckedContinuation<Void, Never>?
    private var released = false
    private(set) var entered = false
    private(set) var queries = 0
    private(set) var cancelled = false
    func recordCancellation(_ value: Bool) { cancelled = value }
    func wait() async {
        entered = true
        if !released { await withCheckedContinuation { waiter = $0 } }
    }
    func query() -> Bool { queries += 1; return true }
    func release() { released = true; waiter?.resume(); waiter = nil }
}
private enum StatusOwner: CaseIterable { case swap, revision, load, get, unload }
private extension RealModelsService {
    func fixtureResident(_ id: String, lastUsed: Double = 10) {
        residents[id] = .init(id: id, residency: .manual, lastUsed: lastUsed)
        modelRows[id]?.states["bf16"] = .init(state: .loaded)
    }
    func fixtureOwner(_ id: String, _ owner: StatusOwner, gate: StatusQueryGate) {
        switch owner {
        case .swap: swapTasks[id] = Task { await gate.wait() }
        case .revision: reloading.insert(id)
        case .load: loadTasks[id] = Task { await gate.wait() }
        case .get: gets[id] = Task { await gate.wait() }
        case .unload: beginUnload(id)
        }
    }
    func clearFixtureOwners(_ id: String) {
        swapTasks[id] = nil; reloading.remove(id); loadTasks[id] = nil; gets[id] = nil; unloading[id] = nil
    }
    func fixtureIntentionalSwapUnload(_ id: String, completion: StatusQueryGate) -> Task<Void, Never> {
        reloading.insert(id)
        let result = Task {
            defer { reloading.remove(id); swapTasks[id] = nil; signalMemoryChange() }
            try? await unloadResident(id: id)
            await completion.recordCancellation(Task.isCancelled)
        }
        swapTasks[id] = result
        return result
    }
}
private struct StatusGapArtifact: WorkerArtifact {
    let gate: URL
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/status_cleanup_worker.py")
        return WorkerLaunch(
            executable: URL(fileURLWithPath: "/opt/homebrew/bin/python3"), arguments: ["-I", "-B", script.path, gate.path], kind: "chat", runner: "native",
            environment: ["ARSENAL_SUPPORT_DIR": paths.support.path], log: paths.support.appendingPathComponent("status-gap.log"))
    }
}
@Suite(.serialized) struct StatusCleanupTests {
    private func waitFor(_ predicate: () async -> Bool) async throws {
        for _ in 0..<200 {
            if await predicate() { return }
            try await Task.sleep(for: .milliseconds(2))
        }
        throw WorkerFailure(code: "fixture_timeout", message: "Status cleanup fixture did not reach its gate.")
    }
    @Test(arguments: [false, true]) func statusCannotCancelSwapInSupervisorDetachGap(queryStartedFirst: Bool) async throws {
        let gateRoot = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-status-gap-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: gateRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: gateRoot) }
        let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: StatusGapArtifact(gate: gateRoot))
        defer { f.stop(); try? FileManager.default.removeItem(at: f.paths.support.deletingLastPathComponent()) }
        let id = "fixture-chat", queryGate = StatusQueryGate(), completion = StatusQueryGate()
        try await f.service.get(id: id, precision: "bf16"); try await f.service.load(id: id)
        let cleanup: Task<Void, Never>?
        if queryStartedFirst {
            cleanup = Task {
                await f.service.refreshExitedResidents { model in
                    await queryGate.wait()
                    return await f.service.supervisor.hasExited(id: model)
                }
            }
            try await waitFor { await queryGate.entered }
        } else {
            cleanup = nil
        }
        let swap = await f.service.fixtureIntentionalSwapUnload(id, completion: completion)
        do {
            try await waitFor { FileManager.default.fileExists(atPath: gateRoot.appendingPathComponent("entered").path) }
            // Exact production gap: Supervisor detached the worker, service still has the resident.
            #expect(await f.service.supervisor.hasExited(id: id))
            #expect(await f.service.residents[id] != nil)
            if let cleanup { await queryGate.release(); await cleanup.value }
            _ = await f.service.rows()
            #expect(await f.service.swapTasks[id] != nil)
            #expect(await f.service.residents[id] != nil)
            #expect(await f.service.unloading[id] == 1)
        } catch {
            try? Data().write(to: gateRoot.appendingPathComponent("release"))
            await queryGate.release(); _ = await swap.value; await cleanup?.value; await f.service.shutdown(); throw error
        }
        try Data().write(to: gateRoot.appendingPathComponent("release"))
        await swap.value
        #expect(await !completion.cancelled) // Public Unload would cancel the swap task.
        #expect(await f.service.residents[id] == nil)
        #expect(await f.service.unloading[id] == nil)
        await f.service.shutdown()
    }

    @Test(arguments: StatusOwner.allCases, [false, true])
    fileprivate func statusRespectsEveryLifecycleOwnerAndRechecksAfterExitQuery(owner: StatusOwner, ownerStartsDuringQuery: Bool) async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["chat"])
        defer { f.stop(); try? FileManager.default.removeItem(at: f.paths.support.deletingLastPathComponent()) }
        let id = "fixture-chat", ownerGate = StatusQueryGate(), queryGate = StatusQueryGate()
        await f.service.fixtureResident(id)
        if !ownerStartsDuringQuery { await f.service.fixtureOwner(id, owner, gate: ownerGate) }
        let cleanup = Task {
            await f.service.refreshExitedResidents { _ in
                if ownerStartsDuringQuery { await queryGate.wait() }
                return await queryGate.query()
            }
        }
        if ownerStartsDuringQuery {
            try await waitFor { await queryGate.entered }
            await f.service.fixtureOwner(id, owner, gate: ownerGate)
            await queryGate.release()
        }
        await cleanup.value
        #expect(await f.service.residents[id] != nil)
        #expect(await queryGate.queries == (ownerStartsDuringQuery ? 1 : 0))
        await ownerGate.release(); await f.service.clearFixtureOwners(id)
        _ = await f.service.rows()
        #expect(await f.service.residents[id] == nil) // A genuine unowned exit is still cleaned up.
        #expect(await f.service.modelRows[id]?.states["bf16"]?.state == .notInstalled)
        await f.service.shutdown()
    }

    @Test func completedReplacementInvalidatesAnAwaitedExitObservation() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["chat"])
        defer { f.stop(); try? FileManager.default.removeItem(at: f.paths.support.deletingLastPathComponent()) }
        let id = "fixture-chat", gate = StatusQueryGate()
        await f.service.fixtureResident(id)
        let cleanup = Task {
            await f.service.refreshExitedResidents { _ in
                await gate.wait(); return true
            }
        }
        try await waitFor { await gate.entered }
        await f.service.fixtureResident(id, lastUsed: 11)
        await gate.release(); await cleanup.value
        #expect(await f.service.residents[id]?.lastUsed == 11)
        await f.service.shutdown()
    }
}
