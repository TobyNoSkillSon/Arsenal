// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private actor V3RunGate {
    var entered = false
    var releaseWaiter: CheckedContinuation<Void, Never>?
    var enteredWaiter: CheckedContinuation<Void, Never>?
    func pause() async {
        guard !entered else { return }
        entered = true; enteredWaiter?.resume(); enteredWaiter = nil
        await withCheckedContinuation { releaseWaiter = $0 }
    }
    func waitEntered() async { if !entered { await withCheckedContinuation { enteredWaiter = $0 } } }
    func release() { releaseWaiter?.resume(); releaseWaiter = nil }
}
private func v3MemoryService(
    _ fixture: LocalCatalogFixture, availableMB: Double = 1290, minuteSeconds: Double = 1000, totalMB: Double? = nil, recheckSeconds: Double = 30,
    artifacts: any WorkerArtifact = StubWorkerArtifact()
)
    async throws -> RealModelsService
{
    await fixture.service.shutdown()
    let file = fixture.paths.support.appendingPathComponent("memory-fixture.json")
    try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: availableMB, totalMB: totalMB ?? 128_000))]).write(to: file)
    return try await RealModelsService.open(
        catalogData: fixture.catalogData, signature: fixture.signature, publicKey: fixture.publicKey, source: fixture.source,
        kinds: KindSpec.defaults, paths: fixture.paths, artifacts: artifacts, client: fixture.client, minuteSeconds: minuteSeconds,
        memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": file.path], totalMB: totalMB ?? 128_000), memoryRecheckSeconds: recheckSeconds)
}
private func v3WaitQueued(_ service: RealModelsService, id: String) async throws {
    let limit = Date().addingTimeInterval(5)
    while Date() < limit {
        if await service.rows().first(where: { $0.id == id })?.states.values.contains(where: { $0.phase == "queued" }) == true { return }
        try await Task.sleep(for: .milliseconds(10))
    }
    throw WorkerFailure(code: "test", message: "Request did not enter the memory queue.")
}

@Test func v3DemandLoadedRequestExpiresAfterKeepHotWindow() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    let service = try await v3MemoryService(f, minuteSeconds: 0.002)
    do {
        try await service.get(id: "fixture-voices", precision: "bf16")
        _ = try await service.run(id: "fixture-voices", op: "speak", params: ["text": "hello", "stream": false])
        let events = await service.events()
        // Wait for the intentional unload itself; rows() no longer completes it during the detach gap.
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                if await service.residents["fixture-voices"] == nil { return }
                for await event in events {
                    if case .rowsChanged = event, await service.residents["fixture-voices"] == nil { return }
                }
                try Task.checkCancellation()
                throw WorkerFailure(code: "test", message: "Service stopped before idle unload completed.")
            }
            group.addTask {
                try await Task.sleep(for: .seconds(3))
                throw WorkerFailure(code: "test", message: "Idle unload did not complete.")
            }
            defer { group.cancelAll() }
            try await group.next()
        }
        #expect(await service.rows().first?.residentDType == nil)
        await service.shutdown()
    } catch { await service.shutdown(); throw error }
}

@Test func v3MemoryPolicyDoesNotEvictProtectedModels() {
    let manual = ResidentModel(id: "manual", residency: .manual, memoryMB: 2000)
    let hot = ResidentModel(id: "hot", residency: .onDemand, memoryMB: 2000, keepHot: true)
    let busy = ResidentModel(id: "busy", residency: .onDemand, memoryMB: 2000, busy: 1)
    #expect(ResidencyPolicy.admission(id: "new", estimateMB: 500, availableMB: 0, residents: [manual, hot, busy], allowSwap: false) == .refuse(needMB: 1012, freeMB: 0))
    #expect(ResidencyPolicy.pressureVictims([manual, hot, busy], order: []) == [])
}

@Test func v3BeginUpdateDrainsGeneratingAndBlocksNewRequests() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    let gate = V3RunGate()
    let run = Task { try await f.service.run(id: "fixture-chat", op: "chat", params: ["prompt": "hello", "stream": true], sink: { _ in await gate.pause() }) }
    await gate.waitEntered()
    let update = Task { try await f.service.beginUpdate() }
    // Actor entry gives the update gate a deterministic opportunity before the next public request.
    for _ in 0..<100 where await !f.service.updating { await Task.yield() }
    await #expect(throws: WorkerFailure.self) { _ = try await f.service.run(id: "fixture-chat", op: "chat", params: ["prompt": "new"]) }
    await gate.release(); _ = try await run.value
    #expect(try await update.value == ["fixture-chat"])
    await f.service.cancelUpdate()
    _ = try await f.service.run(id: "fixture-chat", op: "chat", params: ["prompt": "new", "stream": false])
    await f.service.shutdown()
}

private actor V3HandoffArtifact: WorkerArtifact {
    let gate: V3RunGate
    let pauseGet: Bool
    var enabled: Bool
    init(gate: V3RunGate, pauseGet: Bool) { self.gate = gate; self.pauseGet = pauseGet; enabled = pauseGet }
    func enable() { enabled = true }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {
        if enabled && pauseGet { await gate.pause() }
    }
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        if enabled && !pauseGet { await gate.pause() }
        return try await StubWorkerArtifact().launch(model: model, runner: runner, directory: directory, paths: paths)
    }
}
@Test(arguments: [false, true]) func v3BeginUpdateDrainsAdmittedLoadAndGet(pauseGet: Bool) async throws {
    let gate = V3RunGate()
    let artifact = V3HandoffArtifact(gate: gate, pauseGet: pauseGet)
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: artifact); defer { f.stop() }
    if !pauseGet { try await f.service.get(id: "fixture-chat", precision: "bf16") }
    await artifact.enable()
    let operation = Task {
        if pauseGet { try await f.service.get(id: "fixture-chat", precision: "bf16") } else { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    }
    await gate.waitEntered()
    let update = Task { try await f.service.beginUpdate() }
    for _ in 0..<100 where await !f.service.updating { await Task.yield() }
    await #expect(throws: WorkerFailure.self) { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    await #expect(throws: WorkerFailure.self) { try await f.service.get(id: "fixture-chat", precision: "bf16") }
    await gate.release()
    if pauseGet {
        try await operation.value
    } else {
        do { try await operation.value; Issue.record("Load crossed the drained handoff") } catch let error as WorkerFailure { #expect(error.code == "updating") }
    }
    #expect(try await update.value == [])
    await f.service.cancelUpdate()
    try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false)
    await f.service.shutdown()
}

private extension RealModelsService {
    func v3CodeBarrier(_ barrier: @escaping @Sendable () async -> Void) { localCodeCheckBarrier = barrier }
    func v3FailOtherDtype(_ id: String) { modelRows[id]?.states["int8"] = .init(state: .failed, error: "other dtype error") }
}

@Test(arguments: [false, true]) func v3UpdateRejectsUnregisteredWorkPausedAtCodeCheck(get: Bool) async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    let gate = V3RunGate()
    await f.service.v3CodeBarrier { await gate.pause() }
    let operation = Task {
        if get { try await f.service.get(id: "fixture-chat", precision: "bf16") } else { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    }
    await gate.waitEntered()
    #expect(try await f.service.beginUpdate() == [])
    await gate.release()
    do { try await operation.value; Issue.record("Unregistered work crossed update") } catch let error as WorkerFailure { #expect(error.code == "updating") }
    #expect(await f.service.loadTasks.isEmpty)
    #expect(await f.service.gets.isEmpty)
    #expect(await f.service.runTasks.isEmpty)
    await f.service.cancelUpdate(); await f.service.shutdown()
}

private actor V3FailingLoadArtifact: WorkerArtifact {
    let gate: V3RunGate
    var enabled = false
    var attempts = 0
    init(_ gate: V3RunGate) { self.gate = gate }
    func enable() { enabled = true }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        if enabled {
            attempts += 1; await gate.pause()
            throw WorkerFailure(code: "fixture_failure", message: "Shared load failed once.")
        }
        return try await StubWorkerArtifact().launch(model: model, runner: runner, directory: directory, paths: paths)
    }
}
@Test func v3SharedFailedLoadReturnsSameErrorWithoutRetry() async throws {
    let gate = V3RunGate(), artifact = V3FailingLoadArtifact(gate)
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: artifact); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16"); await artifact.enable()
    let first = Task { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    await gate.waitEntered()
    let second = Task { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    for _ in 0..<1000 where await f.service.memoryWaiters.isEmpty { await Task.yield() }
    #expect(await !f.service.memoryWaiters.isEmpty)
    await gate.release()
    for task in [first, second] {
        do { try await task.value; Issue.record("Failed load succeeded") } catch let error as WorkerFailure {
            #expect(error.code == "fixture_failure" && error.message == "Shared load failed once.")
        }
    }
    #expect(await artifact.attempts == 1); await f.service.shutdown()
}

@Test func v3PersistAndKeepHotNotifyMemoryWaiters() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    for on in [true, false] {
        let waiting = Task { try await f.service.waitForMemoryChange() }
        for _ in 0..<1000 where await f.service.memoryWaiters.isEmpty { await Task.yield() }
        try await f.service.setKeepHot(id: "fixture-chat", on: on); try await waiting.value
    }
    let waiting = Task { try await f.service.waitForMemoryChange() }
    for _ in 0..<1000 where await f.service.memoryWaiters.isEmpty { await Task.yield() }
    try await f.service.setToolsShown(id: "fixture-chat", on: false); try await waiting.value
    await f.service.shutdown()
}

@Test func v3CancelGetPreservesOtherDtypeErrors() async throws {
    let gate = V3RunGate(), artifact = V3HandoffArtifact(gate: gate, pauseGet: true)
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: artifact); defer { f.stop() }
    await f.service.v3FailOtherDtype("fixture-chat")
    let get = Task { try await f.service.get(id: "fixture-chat", precision: "int4") }
    await gate.waitEntered()
    let cancelled = Task { await f.service.cancelGet(id: "fixture-chat") }
    await Task.yield(); await gate.release(); await cancelled.value
    _ = try? await get.value
    let row = try #require(await f.service.rows().first)
    #expect(row.states["int4"]?.error == nil)
    #expect(row.states["int8"]?.error == "other dtype error")
    await f.service.shutdown()
}

private actor V3CancelledLoadArtifact: WorkerArtifact {
    let gate = V3RunGate()
    var enabled = false, attempts = 0
    func enable() { enabled = true }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        if enabled {
            attempts += 1
            if attempts == 1 { await gate.pause(); try Task.checkCancellation() }
        }
        return try await StubWorkerArtifact().launch(model: model, runner: runner, directory: directory, paths: paths)
    }
}
@Test func v3SharedLoadRetriesOwnerCancellationButNotWaiterCancellation() async throws {
    let artifact = V3CancelledLoadArtifact()
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: artifact); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16"); await artifact.enable()
    let owner = Task { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    await artifact.gate.waitEntered()
    let cancelled = Task { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    let independent = Task { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    for _ in 0..<1000 where await f.service.memoryWaiters.count < 2 { await Task.yield() }
    #expect(await f.service.memoryWaiters.count == 2)
    cancelled.cancel(); await #expect(throws: CancellationError.self) { try await cancelled.value }
    owner.cancel(); await artifact.gate.release()
    await #expect(throws: CancellationError.self) { try await owner.value }
    try await independent.value
    #expect(await artifact.attempts == 2)
    #expect(await f.service.rows().first?.residentDType == "bf16")
    await f.service.shutdown()
}

@Test(arguments: [false, true]) func v3ImmediateRefusalDoesNotPartiallyEvict(waitForMemory: Bool) async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices", "images"]); defer { f.stop() }
    let service = try await v3MemoryService(f, availableMB: 2000)
    for id in ["fixture-chat", "fixture-voices", "fixture-images"] { try await service.get(id: id, precision: "bf16") }
    _ = try await service.run(id: "fixture-voices", op: "speak", params: ["text": "idle", "stream": false])
    let gate = V3RunGate()
    let running = Task { try await service.run(id: "fixture-chat", op: "chat", params: ["prompt": "busy", "stream": true], sink: { _ in await gate.pause() }) }
    await gate.waitEntered()
    try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 1295))]).write(to: f.paths.support.appendingPathComponent("memory-fixture.json"))
    try await service.setMemorySettings(.init(whenFull: waitForMemory ? .refuse : .queue))
    await service.setLoadPeak("fixture-images", gb: 1.5)
    do {
        try await service.ensureLoaded(id: "fixture-images", precision: "bf16", fast: false, residency: .manual, waitForMemory: waitForMemory)
        Issue.record("Insufficient memory admitted")
    } catch let error as WorkerFailure { #expect(error.code == "insufficient_memory") }
    #expect(await Set(service.residents.keys) == Set(["fixture-chat", "fixture-voices"]))
    #expect(await service.memoryWaiters.isEmpty)
    await gate.release(); _ = try await running.value; await service.shutdown()
}

@Test func v3CancelUpdateWhileDrainingDoesNotWriteHandoff() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    let gate = V3RunGate()
    let running = Task { try await f.service.run(id: "fixture-chat", op: "chat", params: ["prompt": "busy", "stream": true], sink: { _ in await gate.pause() }) }
    await gate.waitEntered()
    let update = Task { try await f.service.beginUpdate() }
    for _ in 0..<1000 where await !f.service.updating { await Task.yield() }
    #expect(await f.service.updating)
    await f.service.cancelUpdate(); await gate.release(); _ = try await running.value
    do { _ = try await update.value; Issue.record("Cancelled update completed") } catch let error as WorkerFailure { #expect(error.code == "update_cancelled") }
    #expect(!FileManager.default.fileExists(atPath: f.paths.support.appendingPathComponent("state/update-residency.json").path))
    #expect(await !f.service.updating)
    try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false)
    await f.service.shutdown()
}

@Test func v3QueuedWakeupsPublishRowOnlyOnFirstTransition() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let events = await f.service.events(), gate = V3RunGate()
    let monitor = Task {
        var count = 0
        for await event in events {
            if case .rowsChanged = event { count += 1; await gate.pause() }
        }
        return count
    }
    await f.service.markMemoryQueued(id: "fixture-chat", precision: "bf16"); await gate.waitEntered()
    await f.service.markMemoryQueued(id: "fixture-chat", precision: "bf16"); await gate.release()
    try await Task.sleep(for: .milliseconds(50)); monitor.cancel()
    #expect(await monitor.value == 1)
    await f.service.shutdown()
}

@Test func v3MemoryErrorNamesNeedFreeAndGeneratingModelWithoutJargon() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"]); defer { f.stop() }
    let service = try await v3MemoryService(f)
    try await service.get(id: "fixture-chat", precision: "bf16")
    let gate = V3RunGate()
    let running = Task { try await service.run(id: "fixture-chat", op: "chat", params: ["prompt": "busy", "stream": true], sink: { _ in await gate.pause() }) }
    await gate.waitEntered()
    let error = await service.insufficientMemory(id: "fixture-voices", need: 15600, available: 4100, waitForMemory: true)
    #expect(error.message.hasPrefix("Not enough memory for "))
    #expect(error.message.contains("needs 14.5 GB, 3.82 GB free while Qwen3.8"))
    #expect(error.message.contains("is generating); choose Memory ▸ When a model doesn't fit ▸ Wait in queue, or unload a model."))
    #expect(!error.message.contains("headroom"))
    await gate.release(); _ = try await running.value; await service.shutdown()
}

/// Launch pauses on a gate, then starts the stub worker (or fails): a Load held after it registers and before `.loading`.
private actor V3PausingLaunchArtifact: WorkerArtifact {
    let gate: V3RunGate
    var enabled = false, fails = false
    init(_ gate: V3RunGate) { self.gate = gate }
    func enable(failing: Bool = false) { enabled = true; fails = failing }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        if enabled {
            await gate.pause()
            if fails { throw WorkerFailure(code: "fixture_failure", message: "Launch failed.") }
        }
        return try await StubWorkerArtifact().launch(model: model, runner: runner, directory: directory, paths: paths)
    }
}
/// What a subscriber that re-reads the service on every rows-changed event sees (the Models table does exactly this):
/// each event's `refreshRefusal()` answer, in order.
private actor V3RefusalWatcher {
    private(set) var readings: [String?] = []
    private var task: Task<Void, Never>?
    func start(_ service: RealModelsService) async {
        let events = await service.events()
        task = Task { for await event in events { if case .rowsChanged = event { await self.record(await service.refreshRefusal()) } } }
    }
    func record(_ reading: String?) { readings.append(reading) }
    func stop() { task?.cancel() }
    var count: Int { readings.count }
    var last: String?? { readings.last }
    /// A reading after the first `count` ones said Refresh is allowed, and it is the latest.
    func endedAfter(_ count: Int) -> Bool { readings.count > count && readings.last == .some(nil) }
}
private func v3Wait(_ what: String, _ condition: () async -> Bool) async throws {
    let limit = Date().addingTimeInterval(5)
    while !(await condition()) {
        if Date() > limit { throw WorkerFailure(code: "test", message: "Timed out waiting for \(what).") }
        try await Task.sleep(for: .milliseconds(5))
    }
}
enum V3LoadOutcome: String, CaseIterable, Sendable { case success, failure, cancellation }
/// Refresh is refused from the moment a Load registers, while it still verifies files and prepares its runner (the row
/// does not show Loading yet). A subscriber learns it from the registration's announcement, and learns that it ended
/// from a later announcement, whether the Load succeeds, fails or is cancelled.
@Test(arguments: V3LoadOutcome.allCases)
func refreshRefusalCoversALoadBeforeItShowsLoading(_ outcome: V3LoadOutcome) async throws {
    let gate = V3RunGate(), artifact = V3PausingLaunchArtifact(gate)
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: artifact); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16"); await artifact.enable(failing: outcome == .failure)
    #expect(await f.service.refreshRefusal() == nil)
    let watcher = V3RefusalWatcher(); await watcher.start(f.service); defer { Task { await watcher.stop() } }
    let load = Task { try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false) }
    await gate.waitEntered()
    // In the gap: no Loading state yet, but the service refuses Refresh, and its announcement told the subscriber.
    let row = try #require(await f.service.rows().first { $0.id == "fixture-chat" })
    #expect(row.states["bf16"]?.state != .loading)
    #expect(await f.service.refreshRefusal() == RealModelsService.refreshBusyMessage)
    try await v3Wait("the registration's announcement") { await watcher.last == .some(RealModelsService.refreshBusyMessage) }
    await #expect(throws: WorkerFailure.self) { try await f.service.refresh() }
    let seenWhileBusy = await watcher.count
    if outcome == .cancellation { load.cancel() }
    await gate.release()
    switch outcome {
    case .success: try await load.value
    case .failure: await #expect(throws: WorkerFailure.self) { try await load.value }
    case .cancellation: await #expect(throws: CancellationError.self) { try await load.value }
    }
    // Ended: the service no longer refuses, and a later announcement brought the subscriber that answer.
    #expect(await f.service.refreshRefusal() == nil)
    try await v3Wait("the closing announcement") { await watcher.endedAfter(seenWhileBusy) }
    try await f.service.refresh()
    await f.service.shutdown()
}

private extension RealModelsService {
    /// The loaded model's revision as recovery keeps it, its worker started through a shell that waits `delay` seconds
    /// first (a recovery Load held inside its worker start, after it registered).
    func v3CaptureRevision(_ id: String, delay: Int) throws -> ResidentRevision {
        guard let manifest = residentManifests[id], let operations = residentOperations[id], let settings = residentSettings[id],
            var launch = residentLaunches[id], let load = residentLoads[id], let preference = loadedPreferences[id], let resident = residents[id]
        else { throw WorkerFailure(code: "test", message: "\(id) is not resident.") }
        launch.arguments = ["-c", "sleep \(delay); exec \"$0\" \"$@\"", launch.executable.path] + launch.arguments
        launch.executable = URL(fileURLWithPath: "/bin/sh")
        return ResidentRevision(
            manifest: manifest, operations: operations, settings: settings, launch: launch, load: load, preference: preference, resident: resident,
            digest: "previous", menus: [])
    }
    /// A local folder whose newer contents were rejected, with `revision` as its previous one: the next Load restores it.
    func v3PlantRecovery(_ id: String, _ revision: ResidentRevision) throws {
        dormantRevisions[id] = revision
        localModels[id] = LocalModelFolder(manifest: try manifest(id), directory: paths.support, fileDigests: [:], contentDigest: "rejected")
        rejectedRevisionDigests[id] = "rejected"; config.rejectedRevisions[id] = "rejected"
    }
    var v3RecoveryReady: Bool { residents.isEmpty && recoveryRevision("fixture-voices") != nil }
    var v3Resident: Bool { residents["fixture-voices"] != nil }
}
/// The recovery Load (a rejected revision's previous one) registers like any Load: held while its worker starts,
/// Refresh is refused and the subscriber was told so; when it ends, a later announcement says Refresh is allowed.
@Test func aRecoveryLoadAnnouncesItsRegistrationAndEnd() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    let service = try await v3MemoryService(f, availableMB: 100_000)
    try await service.get(id: "fixture-voices", precision: "bf16")
    try await service.load(id: "fixture-voices", precision: "bf16", fast: false)
    let revision = try await service.v3CaptureRevision("fixture-voices", delay: 3)
    try await service.unload(id: "fixture-voices")
    try await service.v3PlantRecovery("fixture-voices", revision)
    #expect(await service.v3RecoveryReady)
    let watcher = V3RefusalWatcher(); await watcher.start(service); defer { Task { await watcher.stop() } }
    let recovery = Task { try await service.load(id: "fixture-voices", precision: nil, fast: nil) }
    // Held in the worker start: no resident yet, Refresh refused, and the registration's announcement carried it.
    try await v3Wait("the registration's announcement") { await watcher.last == .some(RealModelsService.refreshBusyMessage) }
    #expect(await service.refreshRefusal() == RealModelsService.refreshBusyMessage)
    #expect(await !service.v3Resident)
    await #expect(throws: WorkerFailure.self) { try await service.refresh() }
    let seenWhileBusy = await watcher.count
    try await recovery.value
    #expect(await service.v3Resident)
    #expect(await service.refreshRefusal() == nil)
    try await v3Wait("the closing announcement") { await watcher.endedAfter(seenWhileBusy) }
    await service.shutdown()
}

/// Previous verified revisions must not bypass the new admission rule.
@Test func recoveryUsesItsCapturedEvidenceAndResidentLowerBound() async throws {
    for (oldPeak, residentMB, candidate, expected) in [(20.4 as Double?, 1000.0, 2.0, 20.4), (2.0, 1000.0, 40.0, 2.0), (nil, 20400.0, 2.0, 20.4)] {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let service = try await v3MemoryService(f, availableMB: 11200)
        try await service.get(id: "fixture-voices", precision: "bf16")
        try await service.load(id: "fixture-voices", precision: nil, fast: nil)
        var old = try await service.v3CaptureRevision("fixture-voices", delay: 0)
        old.resident.memoryMB = residentMB
        if let oldPeak {
            var row = ModelRow(id: "fixture-voices", name: "Old revision", kind: "voices")
            var stats = PrecisionStatSet(kind: .voices, dtype: "bf16", runner: "native")
            stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: oldPeak)); row.precisionStats["bf16"] = stats
            old.memoryRequirement = LoadMemoryRequirement(row: row, precision: "bf16", forecast: nil)
        }
        try await service.unload(id: "fixture-voices")
        try await service.v3PlantRecovery("fixture-voices", old)
        await service.setLoadPeak("fixture-voices", gb: candidate)
        #expect(await service.supervisor.residentIDs().isEmpty)
        if expected > 11.2 {
            do {
                try await service.load(id: "fixture-voices", precision: nil, fast: nil)
                Issue.record("Recovery used the candidate's requirement")
            } catch let failure as WorkerFailure {
                #expect(failure.details["needed_gb"]?.number == expected && failure.details["free_gb"] == 11.2)
            }
            #expect(await service.supervisor.residentIDs().isEmpty) // No recovery process started before approval.
            #expect(await service.v3RecoveryReady)
            try await service.load(id: "fixture-voices", precision: nil, fast: nil, force: true)
        } else {
            try await service.load(id: "fixture-voices", precision: nil, fast: nil)
        }
        #expect(await service.v3Resident)
        #expect(await service.forceLoadedResidents.contains("fixture-voices") == (expected > 11.2))
        try await service.saveRecoveryRevision("fixture-voices", old)
        let saved = try JSONDecoder().decode(SavedRevision.self, from: Data(contentsOf: f.paths.support.appendingPathComponent("state/revisions/fixture-voices.json")))
        #expect(saved.memoryRequirement == old.memoryRequirement)
        await service.shutdown()
    }
}

@Test func joiningAForcedLoadDoesNotDemandASecondModelsMemory() async throws {
    let gate = V3RunGate(), artifact = V3PausingLaunchArtifact(gate)
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: artifact); defer { f.stop() }
    let service = try await v3MemoryService(f, availableMB: 2000, artifacts: artifact)
    try await service.get(id: "fixture-chat", precision: "bf16")
    await service.setLoadPeak("fixture-chat", gb: 1)
    await artifact.enable(failing: false)
    let owner = Task { try await service.load(id: "fixture-chat", precision: nil, fast: nil, force: true) }
    await gate.waitEntered()
    try WireJSON.encode(["available_mb": .number(macAvailableMB(roomMB: 0))]).write(to: f.paths.support.appendingPathComponent("memory-fixture.json"))
    #expect(await service.runRefusal(id: "fixture-chat", op: "chat") == nil)
    let joined = Task { try await service.load(id: "fixture-chat", precision: nil, fast: nil) }
    try await v3Wait("the existing load's waiter") { await !service.memoryWaiters.isEmpty }
    await gate.release()
    try await owner.value; try await joined.value
    #expect(await service.rows().first?.residentDType == "bf16")
    await service.shutdown()
}

@Test(arguments: [8.0, 20.0, 22.0]) func reloadCreditsOnlyItsOutgoingResident(_ replacementGB: Double) async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    let service = try await v3MemoryService(f, availableMB: 21000)
    try await service.get(id: "fixture-voices", precision: "bf16")
    try await service.load(id: "fixture-voices", precision: nil, fast: nil)
    await service.status("fixture-voices", WorkerMemory(footprintMB: 20000))
    await service.setLoadPeak("fixture-voices", gb: replacementGB)
    #expect(await service.memory().availableMB == macAvailableMB(roomMB: 1000)) // Raw no-swap figure; admission subtracts the line.
    if replacementGB > 21 {
        do { try await service.applySettings(id: "fixture-voices"); Issue.record("Oversized replacement admitted") } catch let error as WorkerFailure {
            #expect(error.details["needed_gb"] == .number(replacementGB) && error.details["free_gb"] == 21)
            #expect(error.details["released_gb"] == 20)
            #expect(error.message.contains("19.6 GB fits under that limit after unloading the current model"))
        }
        #expect(await service.residents["fixture-voices"]?.memoryMB == 20000)
        #expect(await service.supervisor.residentIDs() == ["fixture-voices"])
    } else {
        try await service.applySettings(id: "fixture-voices")
        #expect(await service.residents["fixture-voices"]?.memoryMB != 20000) // The replacement, not the old resident.
    }
    #expect(await service.loadReservations.isEmpty)
    await service.shutdown()
}

private actor TwoLoadGate {
    var entered = 0
    var waiting: [CheckedContinuation<Void, Never>] = []
    func pause() async { entered += 1; await withCheckedContinuation { waiting.append($0) } }
    func release() { let all = waiting; waiting = []; for waiter in all { waiter.resume() } }
}
private actor TwoLoadArtifact: WorkerArtifact {
    let gate: TwoLoadGate
    var enabled = false
    init(_ gate: TwoLoadGate) { self.gate = gate }
    func enable() { enabled = true }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        var launch = try await StubWorkerArtifact().launch(model: model, runner: runner, directory: directory, paths: paths)
        if enabled {
            await gate.pause()
            launch.arguments = ["-c", "sleep 0.2; exec \"$0\" \"$@\"", launch.executable.path] + launch.arguments
            launch.executable = URL(fileURLWithPath: "/bin/sh")
        }
        return launch
    }
}
@Test func concurrentLoadsReserveBeforeEitherWorkerMaterializes() async throws {
    let gate = TwoLoadGate(), artifact = TwoLoadArtifact(gate)
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"], artifacts: artifact); defer { f.stop() }
    let service = try await v3MemoryService(f, availableMB: 1500, artifacts: artifact)
    for id in ["fixture-chat", "fixture-voices"] { try await service.get(id: id, precision: "bf16") }
    // Get refreshes table evidence for every row; plant both requirements only after those refreshes.
    for id in ["fixture-chat", "fixture-voices"] { await service.setLoadPeak(id, gb: 1) }
    await artifact.enable()
    let tasks = ["fixture-chat", "fixture-voices"].map { id in
        Task { () -> Bool in
            do { try await service.load(id: id, precision: nil, fast: nil); return true } catch let error as WorkerFailure {
                #expect(error.code == "insufficient_memory" && error.details["needed_gb"] == 1); return false
            } catch { Issue.record(error); return false }
        }
    }
    try await v3Wait("both advisory checks") { await gate.entered == 2 }
    await gate.release()
    var admitted = 0; for task in tasks { admitted += await task.value ? 1 : 0 }
    #expect(admitted == 1)
    #expect(await service.supervisor.residentIDs().count == 1)
    #expect(await service.loadReservations.isEmpty)
    await service.shutdown()
}

@Test func forcedOnDemandResidentSurvivesWorkerObservedSwapGrowthAndPressure() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices", "images"]); defer { f.stop() }
    let service = try await v3MemoryService(f, availableMB: 2000)
    for id in ["fixture-chat", "fixture-voices", "fixture-images"] { try await service.get(id: id, precision: "bf16") }
    try await service.load(id: "fixture-chat", precision: nil, fast: nil)
    await service.setLoadPeak("fixture-voices", gb: 3)
    _ = try await service.run(id: "fixture-voices", op: "speak", params: ["text": "Hello", "stream": false], force: true)
    try await service.ensureLoaded(id: "fixture-images", precision: nil, fast: nil, residency: .onDemand)
    let before = try await service.supervisor.request(id: "fixture-voices", op: "test_memory_counters", params: ["swap_used_bytes": 0])
    let swap: JSONValue = 8_000_000_000
    let grown = try await service.supervisor.request(id: "fixture-voices", op: "test_memory_counters", params: ["swap_used_bytes": swap])
    #expect(grown["swap_used_bytes"] == swap && grown["loaded"] == true)
    await service.shedPressure()
    let after = try await service.supervisor.request(id: "fixture-voices", op: "status")
    #expect(after["worker_pid"] == before["worker_pid"] && after["swap_used_bytes"] == swap && after["loaded"] == true)
    #expect(await service.supervisor.residentIDs() == ["fixture-chat", "fixture-voices"])
    #expect(await service.forceLoadedResidents == ["fixture-voices"])
    #expect(await service.residencySettings().allowSwap == false && !LoadMemoryContext.force)
    // A real allocation error remains a reported request failure, with the app/service and healthy worker alive.
    do {
        _ = try await service.run(id: "fixture-voices", op: "speak", params: ["text": "__fixture_allocation_failure__", "stream": false])
        Issue.record("The allocation failure was lost")
    } catch let failure as WorkerFailure {
        #expect(failure.code == "out_of_memory" && failure.message == "Injected worker allocation failure.")
    }
    let alive = try await service.supervisor.request(id: "fixture-voices", op: "ping")
    #expect(alive["loaded"] == true && alive["worker_pid"] == before["worker_pid"])
    try await service.unload(id: "fixture-voices")
    #expect(await service.forceLoadedResidents.isEmpty)
    // Consent belongs to the old resident: a new ordinary load gets ordinary pressure/idle policy again.
    await service.setLoadPeak("fixture-voices", gb: 0.1)
    try await service.ensureLoaded(id: "fixture-voices", precision: nil, fast: nil, residency: .onDemand)
    await service.shedPressure()
    #expect(await service.supervisor.residentIDs() == ["fixture-chat"])
    await service.shutdown()
}

@Test func pressureDoesNotCancelAnInFlightForcedReplacement() async throws {
    let gate = V3RunGate(), artifact = V3PausingLaunchArtifact(gate)
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"], artifacts: artifact); defer { f.stop() }
    let service = try await v3MemoryService(f, availableMB: 2000, artifacts: artifact)
    for id in ["fixture-chat", "fixture-voices"] { try await service.get(id: id, precision: "bf16") }
    try await service.load(id: "fixture-chat", precision: nil, fast: nil)
    try await service.ensureLoaded(id: "fixture-voices", precision: nil, fast: nil, residency: .onDemand)
    await artifact.enable()
    let replacement = Task { try await LoadMemoryContext.$force.withValue(true) { try await service.applySettings(id: "fixture-voices") } }
    await gate.waitEntered()
    // If pressure incorrectly cancels that Load, release the fixture so a regression fails rather than hangs.
    let escape = Task {
        try? await Task.sleep(for: .seconds(2)); await gate.release()
    }
    await service.shedPressure()
    escape.cancel(); await gate.release()
    try await replacement.value
    #expect(await service.forceLoadedResidents == ["fixture-voices"])
    #expect(await service.supervisor.residentIDs() == ["fixture-chat", "fixture-voices"])
    await service.shutdown()
}

private actor V3FailNextLoadArtifact: WorkerArtifact {
    var failNext = false
    func failNextLoad() { failNext = true }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        var launch = try await StubWorkerArtifact().launch(model: model, runner: runner, directory: directory, paths: paths)
        if failNext { failNext = false; launch.arguments.append("--fail-load") }
        return launch
    }
}
extension RealModelsService {
    func v3EnableRevisionCapture(_ id: String) { residentLocalDigests[id] = "previous" }
}
@Test func failedReplacementRestoresTheForcedResidentWithoutAuthorizingLaterLoads() async throws {
    let artifact = V3FailNextLoadArtifact()
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"], artifacts: artifact); defer { f.stop() }
    let service = try await v3MemoryService(f, availableMB: 2000, artifacts: artifact)
    for id in ["fixture-chat", "fixture-voices"] { try await service.get(id: id, precision: "bf16") }
    try await service.load(id: "fixture-chat", precision: nil, fast: nil)
    await service.setLoadPeak("fixture-voices", gb: 3)
    _ = try await service.run(id: "fixture-voices", op: "speak", params: ["text": "Hello", "stream": false], force: true)
    await service.v3EnableRevisionCapture("fixture-voices")
    let old = try #require(await service.captureRevision("fixture-voices"))
    #expect(old.memoryForceApproved)
    // The candidate fits; its allocation fails. Recovery needs the old 3 GB consent against the same 2 GB budget.
    await service.setLoadPeak("fixture-voices", gb: 0.1)
    await artifact.failNextLoad()
    do { try await service.applySettings(id: "fixture-voices"); Issue.record("The failed candidate loaded") } catch let error as WorkerFailure {
        #expect(error.code == "out_of_memory")
    }
    #expect(await service.forceLoadedResidents == ["fixture-voices"])
    #expect(await service.residentLocalDigests["fixture-voices"] == "previous")
    await service.shedPressure()
    #expect(await service.supervisor.residentIDs() == ["fixture-chat", "fixture-voices"])
    // Save/reopen or a later explicit Load is not an automatic rollback of that interrupted resident.
    try await service.saveRecoveryRevision("fixture-voices", old)
    let saved = try WireJSON.decode(Data(contentsOf: f.paths.support.appendingPathComponent("state/revisions/fixture-voices.json")))
    #expect(saved["memoryForceApproved"] == nil)
    try await service.unload(id: "fixture-voices")
    try await service.v3PlantRecovery("fixture-voices", old)
    do { try await service.load(id: "fixture-voices", precision: nil, fast: nil); Issue.record("Old consent escaped into a new Load") } catch let error as WorkerFailure {
        #expect(error.code == "insufficient_memory" && error.details["needed_gb"] == 3)
    }
    let remaining = await service.supervisor.residentIDs(), approvals = await service.forceLoadedResidents
    #expect(remaining == ["fixture-chat"] && approvals.isEmpty)
    await service.shutdown()
}

/// VM run, 2 Nov: a kit-kind model (Chronos-Bolt) showed its folder quick check's 0.24 GB in the table but "Memory
/// requirement unknown" in the menu and its window. Load admission now reads the same folder figure.
@Test func folderQuickCheckPeakGivesTheLoadRequirement() {
    var row = ModelRow(id: "chronos-bolt-small", name: "Chronos-Bolt Small", kind: "local.forecast", runner: "kit")
    row.precisions = ["fp16"]; row.selectedPrecision = "fp16"
    row.folderFigures["fp16"] = .init(
        model: row.id, kind: row.kind, revision: "r", precision: "fp16", runner: "kit",
        peakMemoryGB: StatValue(state: .quickCheck, value: 0.24), secondsPerRequest: StatValue())
    let need = LoadMemoryRequirement(row: row, precision: "fp16", forecast: nil)
    #expect(need.neededGB == 0.24 && need.basis == .unmeasuredPeak)
    #expect(LoadMemoryRequirement(row: row, precision: "int8", forecast: nil).basis == .unknown)
}
