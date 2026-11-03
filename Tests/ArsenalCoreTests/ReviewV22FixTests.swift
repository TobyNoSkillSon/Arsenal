// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func reviewV22ImplicitAffineEstimatesIncludeScales() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let model = try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: root.appendingPathComponent("models/qwen3.8-27b/model.toml"), encoding: .utf8))
    #expect(model.formats.first { $0.id == "int4" }?.bitsPerWeight == 4.5)
    #expect(model.formats.first { $0.id == "int8" }?.bitsPerWeight == 8.5)
}

@Test func reviewV22MatchingIsWidestFirstAfterDefault() {
    var entry = CatalogEntry(row: ModelRow(id: "m", name: "M", kind: "chat", precisions: ["bf16", "int4", "wide"], selectedPrecision: "bf16"))
    entry.formats = [
        .init(id: "bf16", bitsPerWeight: 16, source: .vendor, runners: []), .init(id: "int4", bitsPerWeight: 4, source: .derived, runners: []),
        .init(id: "wide", bitsPerWeight: 4.25, source: .vendor, runners: [])
    ]
    entry.ramMBByDtype = ["bf16": 58000, "int4": 15000, "wide": 14000]
    var query = CatalogQuery(); query.minRAMMB = 8000; query.maxRAMMB = 16000
    #expect(CatalogEntry.query([entry], query).first?.matchingDtype == "wide")
}

@Test func reviewV22FailedLocalForecastMustNotDownload() async throws {
    let donorRoot = URL(fileURLWithPath: "/tmp/arsenal-mb-v22-fix-l/donor-\(UUID().uuidString)")
    let donor = donorRoot.appendingPathComponent("test/fixture/base.bin")
    try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data([99, 99, 99]).write(to: donor)
    var config = ArsenalConfiguration(); config.get = ["localSources": [.string(donorRoot.path)]]
    let fixture = try await makeLocalCatalogFixture(kinds: ["voices"], configuration: config); defer { fixture.stop() }
    let plan = try await fixture.service.getPlan(id: "fixture-voices", dtype: "bf16")
    #expect(plan.totalDownloadBytes == 0 && plan.items.first?.localCopyBytes == 3)
    do {
        try await fixture.service.get(id: "fixture-voices", precision: "bf16", approvedPlan: plan)
        Issue.record("Unapproved fallback downloaded")
    } catch let changed as GetPlanChanged { #expect(changed.revisedPlan.totalDownloadBytes == 3) }
    #expect(!FileManager.default.fileExists(atPath: fixture.paths.directory("weights/downloads").path))
    await fixture.service.cancelGet(id: "fixture-voices")
    #expect(await fixture.service.rows().first?.states["bf16"]?.error == nil)
    let revised = try await fixture.service.getPlan(id: "fixture-voices", dtype: "bf16")
    #expect(revised.totalDownloadBytes == 3 && revised.items.first?.networkContent?[fixture.weightHash] == 3)
    await #expect(throws: GetPlanChanged.self) { try await fixture.service.get(id: "fixture-voices", precision: "bf16", approvedPlan: plan) }
    try await fixture.service.get(id: "fixture-voices", precision: "bf16", approvedPlan: revised)
    #expect(await fixture.service.rows().first?.state == .ready)
    await fixture.service.shutdown()
}

@Test func reviewV22ExternalConfigEditSurvivesAutomaticSave() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { fixture.stop() }
    let file = fixture.paths.support.appendingPathComponent("config.json")
    var config = try ArsenalConfiguration.read(file)
    config.runnerOverrides["fixture-voices"] = "agent-edited"
    try config.write(file)
    try await fixture.service.setKeepHot(id: "fixture-voices", on: true)
    let saved = try ArsenalConfiguration.read(file)
    #expect(saved.runnerOverrides["fixture-voices"] == "agent-edited")
    #expect(saved.selections["fixture-voices"]?.keepHot == true)
    #expect(await catalogRow(fixture.service, "fixture-voices")?.keepHot == true)
    await fixture.service.shutdown()
}

private actor ReviewV22PausedSource: CatalogSource {
    let files: [String: Data]
    var paused = false
    var blocked = false
    var releaseWaiter: CheckedContinuation<Void, Never>?
    var blockedWaiter: CheckedContinuation<Void, Never>?
    init(_ files: [String: Data]) { self.files = files }
    func pause() { paused = true }
    func waitBlocked() async { if !blocked { await withCheckedContinuation { blockedWaiter = $0 } } }
    func release() { paused = false; releaseWaiter?.resume(); releaseWaiter = nil }
    func read(path: String, commit: String) async throws -> Data {
        if paused {
            blocked = true; blockedWaiter?.resume(); blockedWaiter = nil
            await withCheckedContinuation { releaseWaiter = $0 }
        }
        guard let value = files[path] else { throw ManifestError.invalid("missing paused-source fixture") }
        return value
    }
}

@Test func reviewV22ResidentRunsAndConcurrentConfigChangesSurviveRefresh() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }; await f.service.shutdown()
    let source = ReviewV22PausedSource(f.source.files)
    let service = try await RealModelsService.open(
        catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey, source: source,
        kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    let id = "fixture-voices"
    try await service.get(id: id, precision: "bf16"); try await service.load(id: id, precision: "bf16", fast: false)
    await source.pause()
    let refreshing = Task { try await service.refresh() }
    await source.waitBlocked()
    let result = try? await service.run(id: id, op: "test_config")
    try await service.setKeepHot(id: id, on: true)
    let file = f.paths.support.appendingPathComponent("config.json")
    var config = try ArsenalConfiguration.read(file); config.residency.manualIdleMinutes = 5; try config.write(file)
    await source.release(); try await refreshing.value
    #expect(result?["dtype"] == "bf16")
    #expect(await service.residencySettings().manualIdleMinutes == 5)
    #expect(await service.rows().first?.keepHot == true)
    await service.shutdown()
}

@Test func reviewV22ConsentBindsIdentityFilesAndKnownByteLimits() {
    let weights = GetPlanItem(id: "weights", name: "Weights", kind: "weights", downloadBytes: 3, networkContent: ["hash": 3])
    let plan = GetPlan(id: "m", dtype: "bf16", runner: "native", items: [weights], identity: "revision")
    #expect(plan.covers(plan))
    var more = plan; more.items[0].downloadBytes = 4; #expect(!plan.covers(more))
    more = plan; more.items[0].networkContent = ["other-hash": 3]; #expect(!plan.covers(more))
    more = plan; more.identity = "another-revision"; #expect(!plan.covers(more))
    more = plan; more.items.append(.init(id: "assets", name: "Voice assets", kind: "asset", downloadBytes: nil)); #expect(!more.covers(more))
}

@Test func reviewV22FitsUsesDecimalMBAndHeadroom() {
    // Under the Mac-wide line (100/128 of RAM) with the headroom, on any Mac: 16, 32 and 128 GB alike.
    for gib: UInt64 in [16, 32, 128] {
        let bytes = gib * 1_073_741_824
        let line = MacMemoryGuard.safeInUseMB(totalMB: Double(bytes) / 1e6)
        #expect(CatalogMatch.fits(ramMB: line - 512, physicalBytes: bytes))
        #expect(!CatalogMatch.fits(ramMB: line - 511, physicalBytes: bytes))
    }
    #expect(CatalogMatch.fits(ramMB: 26000, physicalBytes: 32 * 1_073_741_824))
    #expect(!CatalogMatch.fits(ramMB: 33000, physicalBytes: 32 * 1_073_741_824))
}
