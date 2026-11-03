// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private actor GetCommitGate: WorkerArtifact {
    private var target: String?
    private var entered: CheckedContinuation<Void, Never>?
    private var release: CheckedContinuation<Void, Never>?
    private var didEnter = false
    func arm(_ dtype: String) { target = dtype; didEnter = false }
    func waitUntilCommitted() async { if didEnter { return }; await withCheckedContinuation { entered = $0 } }
    func failed() { didEnter = true; entered?.resume(); entered = nil }
    func resume() { release?.resume(); release = nil }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        if let target,
            FileManager.default.fileExists(
                atPath: paths.directory("weights").appendingPathComponent("snapshots/\(model.id)/\(model.weights.revision)/\(target)/arsenal-snapshot.json").path)
        {
            self.target = nil; didEnter = true; entered?.resume(); entered = nil
            await withCheckedContinuation { release = $0 }
        }
        var launch = stubLaunch(model.kind); launch.environment["ARSENAL_SUPPORT_DIR"] = paths.support.path; return launch
    }
}
@Test func v2GetAnotherDtypeKeepsWorkerUntilExplicitReloadAndPublishesReadyLast() async throws {
    let gate = GetCommitGate()
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: gate); defer { f.stop() }
    let s = f.service, id = "fixture-chat"
    try await s.get(id: id, precision: "bf16"); try await s.load(id: id, precision: "bf16", fast: false)
    #expect(try await s.run(id: id, op: "test_config")["dtype"] == "bf16")
    await gate.arm("int4")
    let get = Task { do { try await s.get(id: id, precision: "int4") } catch { await gate.failed(); throw error } }
    await gate.waitUntilCommitted()
    // The target snapshot exists, but Get's final admission/configuration work has not finished.
    let during = try #require(await s.rows().first)
    #expect(during.selectedPrecision == "int4" && during.state != .ready)
    #expect(during.states["int4"]?.state != .ready && during.states["bf16"]?.state == .loaded)
    #expect(try await s.run(id: id, op: "test_config")["dtype"] == "bf16")
    #expect(await s.memory().workersMB == 16)
    await gate.resume(); try await get.value
    let ready = try #require(await s.rows().first)
    #expect(ready.states["int4"]?.state == .ready && ready.states["bf16"]?.state == .loaded && ready.needsReload)
    #expect(try await s.run(id: id, op: "test_config")["dtype"] == "bf16")
    // Ready is actionable immediately: no remaining Get task can reject this user Load/Reload.
    try await s.applySettings(id: id)
    #expect(try await s.run(id: id, op: "test_config")["dtype"] == "int4")
    #expect(await s.rows().first?.needsReload == false)
    #expect(await s.rows().first?.states["bf16"]?.state == .ready)
    await s.shutdown()
}
@Test func v2StubGetAnotherDtypeIsPendingNotAnImplicitReload() async throws {
    let row = ModelRow(id: "voice", name: "Voice", kind: "voices", precisions: ["bf16", "int4"])
    let s = StubModelsService(rows: [row])
    try await s.get(id: row.id, precision: "bf16"); try await s.load(id: row.id)
    try await s.get(id: row.id, precision: "int4")
    #expect(await s.rows().first?.needsReload == true); #expect(await s.rows().first?.states["bf16"]?.state == .loaded)
    _ = try await s.run(id: row.id, op: "voices")
    #expect(await s.rows().first?.states["bf16"]?.state == .loaded)
    try await s.applySettings(id: row.id)
    #expect(await s.rows().first?.needsReload == false); #expect(await s.rows().first?.states["int4"]?.state == .loaded)
}
