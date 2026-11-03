// SPDX-License-Identifier: MIT
// Regression cases reproduced by both independent v2.1 reviewers.
import Foundation
import Testing
import ArsenalWire
@testable import ArsenalCore

private func swiftRunner(_ name: String, fastExact: Bool = true) -> TOMLValue {
    .table([
        "name": .string(name), "language": .string("swift"), "worker": .string("stub"), "protocol": .integer(1), "chips": .array([.string("*")]), "min_macos": .string("26.0"),
        "fast_exact": .bool(fastExact)
    ])
}
@Test(arguments: [true, false], [nil, false] as [Bool?]) func v21FixFastAfterRunnerRoundTrip(fast: Bool, tableFast: Bool?) async throws {
    let table: [String: TOMLValue] = ["runner": .array([swiftRunner("native"), swiftRunner("alt", fastExact: false)]), "selection": .table(["default": .string("native")])]
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: table); defer { f.stop() }
    let s = f.service, id = "fixture-voices"
    try await s.get(id: id, precision: "bf16"); try await s.load(id: id, precision: "bf16", fast: fast)
    let before = try #require(await s.rows().first)
    try await s.setRunner(id: id, runner: "alt")
    let mid = try #require(await s.rows().first)
    let queued = try PendingBenchmarkQueue(support: f.paths.support).read().filter { $0.runner == "alt" && $0.state == "pending" }
    #expect(queued.count == 3 && queued.allSatisfy { $0.mode == "exact" })
    // Get/Keep Hot must not overwrite a preference merely because this runner cannot honor it.
    try await s.get(id: id, precision: "bf16")
    try await s.setKeepHot(id: id, on: true)
    try await s.setRunner(id: id, runner: nil)
    let roundTrip = try #require(await s.rows().first)
    #expect(roundTrip.fast && !roundTrip.needsReload)
    try await s.setRunner(id: id, runner: "alt")
    // RowPresentation explicitly sends false for a runner without Fast.
    try await s.load(id: id, precision: "bf16", fast: tableFast)
    #expect(await s.rows().first?.fast == false)
    #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).selections[id]?.fast == true)
    try await s.setRunner(id: id, runner: nil)
    let after = try #require(await s.rows().first)
    print(
        "PROBE P2 before fast=\(before.fast) reload=\(before.needsReload) | alt fast=\(mid.fast) reload=\(mid.needsReload) | back fast=\(after.fast) reload=\(after.needsReload) runner=\(after.runner) resident=\(String(describing: after.residentRunner))"
    )
    #expect(after.fast && after.needsReload)
    try await s.load(id: id, precision: "bf16", fast: nil)
    let reloaded = await s.rows().first
    #expect(reloaded?.fast == true && reloaded?.needsReload == false)
    await s.shutdown()
    let reopened = try await f.reopen(threshold: 8_000_000_000)
    #expect(await reopened.rows().first?.fast == true)
    await reopened.shutdown()
}

/// The accepted default remains unpinned even when selected explicitly through API/CLI.
@Test func v21FixDefaultByNameClearsOverride() async throws {
    let table: [String: TOMLValue] = ["runner": .array([swiftRunner("native"), swiftRunner("alt")]), "selection": .table(["default": .string("native")])]
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: table); defer { f.stop() }
    let s = f.service, id = "fixture-voices"
    try await s.setRunner(id: id, runner: "alt")
    try await s.setRunner(id: id, runner: "native")
    let configuration = try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json"))
    #expect(configuration.runnerOverrides[id] == nil)
    print("PROBE P4 override stored=\(String(describing: configuration.runnerOverrides[id])) rowOverride=\(String(describing: await s.rows().first?.runnerOverride))")
    await s.shutdown()
}

private actor PreparationCheckingArtifact: WorkerArtifact {
    var prepared = Set<String>(); var launches: [String] = []
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws { prepared.insert(runner.name) }
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        launches.append(runner.name)
        if !prepared.contains(runner.name) { throw WorkerFailure(code: "worker_missing", message: "Get did not prepare conversion runner") }
        return stubLaunch(model.kind)
    }
}
@Test func v21FixConversionUsesSelectedRunner() async throws {
    let py: TOMLValue = .table([
        "name": .string("alternate"), "language": .string("swift"), "worker": .string("stub"), "protocol": .integer(1), "chips": .array([.string("*")]),
        "min_macos": .string("26.0"), "fast_exact": .bool(false)
    ])
    let artifact = PreparationCheckingArtifact()
    let f = try await makeLocalCatalogFixture(
        kinds: ["voices"], artifacts: artifact, extraTable: ["runner": .array([py, swiftRunner("native")]), "selection": .table(["default": .string("alternate")])])
    defer { f.stop() }
    let s = f.service
    try await s.setRunner(id: "fixture-voices", runner: "native")
    do { try await s.get(id: "fixture-voices", precision: "int4"); print("RV21 conversion succeeded") } catch {
        print("RV21 conversion failed: \(error)"); Issue.record("Native override Get must not depend on an unprepared default runner: \(error)")
    }
    #expect(await artifact.prepared == ["native"])
    #expect(await artifact.launches.allSatisfy { $0 == "native" })
    await s.shutdown()
    // The override is also selected at a later service open, not frozen into a default converter map.
    let reopened = try await RealModelsService.open(
        catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey, source: f.source, kinds: KindSpec.defaults, paths: f.paths, artifacts: artifact,
        client: f.client)
    try await reopened.get(id: "fixture-voices", precision: "int4")
    #expect(await artifact.launches.allSatisfy { $0 == "native" })
    await reopened.shutdown()
}

private actor SlowPrepareArtifact: WorkerArtifact {
    var preparing = false
    var cancelled = false
    var blocks = true
    func setBlocking(_ value: Bool) { blocks = value; preparing = false; cancelled = false }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {
        guard blocks else { return }
        preparing = true
        do { try await Task.sleep(for: .seconds(30)) } catch { cancelled = true; throw error }
    }
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch { stubLaunch(model.kind) }
}
@Test func v21FixRemoveCancelsGetBeforeDroppingRow() async throws {
    for deleteWeights in [false, true] {
        let artifact = SlowPrepareArtifact()
        let f = try await makeLocalCatalogFixture(kinds: ["voices"], artifacts: artifact); defer { f.stop() }
        let s = f.service, id = "fixture-voices"
        let get = Task { try await s.get(id: id, precision: "bf16") }
        for _ in 0..<200 where await !artifact.preparing { try await Task.sleep(for: .milliseconds(5)) }
        #expect(await artifact.preparing)
        try await s.remove(id: id, deleteWeights: deleteWeights)
        #expect(await artifact.cancelled)
        #expect(await s.rows().isEmpty)
        await #expect(throws: CancellationError.self) { try await get.value }
        await s.shutdown()
    }
}

private actor ActiveRequestGate {
    private var entered = false
    private var release: CheckedContinuation<Void, Never>?
    func pause() async {
        entered = true
        await withCheckedContinuation { release = $0 }
    }
    func isEntered() -> Bool { entered }
    func resume() { release?.resume(); release = nil }
}

@Test(arguments: [false, true]) func v21FixRefusedRemoveKeepsGetRunning(deleteWeights: Bool) async throws {
    let artifact = SlowPrepareArtifact()
    await artifact.setBlocking(false)
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: artifact); defer { f.stop() }
    let s = f.service, id = "fixture-chat"
    try await s.get(id: id, precision: "bf16"); try await s.load(id: id)
    let gate = ActiveRequestGate()
    let run = Task { try await s.run(id: id, op: "chat", params: ["stream": true], sink: { _ in await gate.pause() }) }
    for _ in 0..<200 where await !gate.isEntered() { try await Task.sleep(for: .milliseconds(5)) }
    #expect(await gate.isEntered())
    await artifact.setBlocking(true)
    let get = Task { try await s.get(id: id, precision: "int4") }
    for _ in 0..<200 where await !artifact.preparing { try await Task.sleep(for: .milliseconds(5)) }
    #expect(await artifact.preparing)
    do { try await s.remove(id: id, deleteWeights: deleteWeights); Issue.record("Busy resident must refuse Remove") } catch let error as WorkerFailure {
        #expect(error.code == "busy")
    }
    #expect(await artifact.cancelled == false)
    #expect(await s.rows().first?.states["int4"]?.state == .downloading)
    #expect(await s.rows().first?.residentDType == "bf16")
    await gate.resume(); _ = try await run.value
    await s.cancelGet(id: id); _ = try? await get.value
    await s.shutdown()
}
