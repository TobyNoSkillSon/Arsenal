// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func precisionPreferencePersistsRealServiceWithoutFetchingOrLoading() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat"])
    defer { fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent()) }
    let before = try #require(await catalogRow(fixture.service))
    try await fixture.service.setPrecision(id: before.id, precision: "int4")
    let after = try #require(await catalogRow(fixture.service))
    #expect(after.selectedPrecision == "int4" && after.states == before.states && after.residentDType == nil)
    // Nothing was fetched, so the model is still not on the table.
    #expect(await fixture.service.rows().isEmpty)
    let saved = try ArsenalConfiguration.read(fixture.paths.support.appendingPathComponent("config.json"))
    #expect(saved.selections[before.id]?.precision == "int4" && saved.selections[before.id]?.explicitSelection == true)
    #expect(await fixture.service.catalog(query: .init()).first?.row.selectedPrecision == "int4")
    await fixture.service.shutdown()
    let reopened = try await fixture.reopen(threshold: 8_000_000_000)
    #expect(await catalogRow(reopened)?.selectedPrecision == "int4")
    #expect(await catalogRow(reopened)?.residentDType == nil)
    await reopened.shutdown()
}

@Test func precisionPreferenceRefusesCurrentFormatSupportDespiteOldPassingEvidence() async throws {
    let formats: [TOMLValue] = [
        .table(["id": .string("int4"), "bits_per_weight": .integer(4), "source": .string("derived"), "storage": .string("affine"), "runners": .array([])])
    ]
    let old = v3MeasuredResult(chip: HostChip.current, dtype: "int4")
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], extraTable: ["format": .array(formats)], extraFiles: ["results/old.json": try JSONEncoder().encode(old)])
    defer { fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent()) }
    let before = try Data(contentsOf: fixture.paths.support.appendingPathComponent("config.json"))
    do {
        try await fixture.service.setPrecision(id: "fixture-chat", precision: "int4")
        Issue.record("Unsupported precision was accepted")
    } catch let failure as WorkerFailure {
        #expect(failure.code == "unsupported_format" && failure.message == "No runner in this Arsenal yet.")
    }
    do {
        try await fixture.service.setPrecision(id: "fixture-chat", precision: "fp32")
        Issue.record("Unlisted precision was accepted")
    } catch let failure as WorkerFailure { #expect(failure.code == "precision") }
    #expect(try Data(contentsOf: fixture.paths.support.appendingPathComponent("config.json")) == before)
    #expect(await catalogRow(fixture.service)?.selectedPrecision == "bf16")
    await fixture.service.shutdown()
}

@Test func precisionPreferenceAcceptsWarnedInt4AndPreservesOtherSavedPreferences() async throws {
    let support = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".build/precision-core-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: support) }
    var config = ArsenalConfiguration()
    var preference = ModelPreferences(precision: "bf16", fast: false, keepHot: true)
    preference.toolsShown = false; config.selections["model"] = preference
    try config.write(support.appendingPathComponent("config.json"))
    var row = ModelRow(id: "model", name: "Model", kind: "chat", precisions: ["bf16", "int8", "int4"])
    row.precisionStats["int4"] = PrecisionStatSet(kind: .chat, dtype: "int4", runner: "native", unavailableReason: "Quality gate failed")
    row.precisionStats["int8"] = PrecisionStatSet(kind: .chat, dtype: "int8", runner: "old-runner", unavailableReason: "Old runner failed")
    let service = StubModelsService(rows: [row], supportDirectory: support)
    try await service.setPrecision(id: row.id, precision: "int4")
    #expect(await catalogRow(service)?.precisionQualityWarnings["int4"] == PrecisionStatSet.int4QualityWarning)
    let saved = try #require(ArsenalConfiguration.read(support.appendingPathComponent("config.json")).selections[row.id])
    #expect(saved.precision == "int4" && saved.explicitSelection && !saved.fast && saved.keepHot && !saved.toolsShown)
    try await service.get(id: row.id, precision: saved.precision)
    try await service.load(id: row.id)
    #expect(await service.rows().first?.residentDType == "int4")
}

/// Pauses after performLoad has captured its dtype, using only the CPU protocol fixture.
private actor PrecisionSwapArtifact: WorkerArtifact {
    private var armed = false
    private var gate: CheckedContinuation<Void, Never>?
    var holding: Bool { gate != nil }
    func arm() { armed = true }
    func release() { gate?.resume(); gate = nil }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        if armed { armed = false; await withCheckedContinuation { gate = $0 } }
        return try await StubWorkerArtifact().launch(model: model, runner: runner, directory: directory, paths: paths)
    }
}
private extension RealModelsService {
    /// Same swapTasks/reloading admission and production loader used by swapRevision.
    /// Revision evidence admission is covered separately by Fix31CoreTests/DropInFlowTests.
    func startPrecisionFixtureSwap(_ id: String) throws -> Task<Void, any Error> {
        let model = try manifest(id), dtype = modelRows[id]?.selectedPrecision ?? model.basePrecision
        reloading.insert(id)
        let load = Task { try await performLoad(model, precision: dtype, fast: false, residency: .manual, waitForMemory: true) }
        swapTasks[id] = Task {
            _ = try? await load.value
            reloading.remove(id); swapTasks[id] = nil; updatePending(id); signalMemoryChange(); changed()
        }
        return load
    }
}

@Test func precisionPreferenceSurvivesSwapCompletingAtCapturedOldDtype() async throws {
    let artifact = PrecisionSwapArtifact()
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: artifact)
    defer { fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent()) }
    let service = fixture.service, id = "fixture-chat"
    try await service.get(id: id, precision: "bf16")
    try await service.load(id: id, precision: "bf16", fast: false)
    await artifact.arm()
    let swap = try await service.startPrecisionFixtureSwap(id)
    let deadline = ContinuousClock.now + .seconds(5)
    while !(await artifact.holding), ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
    try #require(await artifact.holding)
    let reloading = await service.reloading.contains(id), swapping = await service.swapTasks[id] != nil
    #expect(reloading && swapping)
    try await service.setPrecision(id: id, precision: "int4")
    let configFile = fixture.paths.support.appendingPathComponent("config.json")
    let savedDuring = try Data(contentsOf: configFile)
    #expect(await service.rows().first?.selectedPrecision == "int4")
    await artifact.release(); try await swap.value
    let row = try #require(await service.rows().first)
    #expect(row.selectedPrecision == "int4" && row.residentDType == "bf16" && row.needsReload)
    #expect(try Data(contentsOf: configFile) == savedDuring)
    #expect(try ArsenalConfiguration.read(configFile).selections[id]?.precision == "int4")
    // A later unrelated config write and a new service must keep the explicit choice too.
    try await service.setKeepHot(id: id, on: true)
    #expect(try ArsenalConfiguration.read(configFile).selections[id]?.precision == "int4")
    await service.shutdown()
    let reopened = try await fixture.reopen(threshold: 8_000_000_000)
    #expect(await reopened.rows().first?.selectedPrecision == "int4")
    await reopened.shutdown()
}

@Test func internalReloadDoesNotCreateAnExplicitPrecisionPreference() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat"])
    defer { fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent()) }
    let service = fixture.service, id = "fixture-chat"
    try await service.get(id: id, precision: "bf16")
    try await service.setPrecision(id: id, precision: "int4")
    // A captured old internal load has no authority to replace the new choice.
    let model = try await service.manifest(id)
    let file = fixture.paths.support.appendingPathComponent("config.json"), before = try Data(contentsOf: file)
    try await service.performLoad(model, precision: "bf16", fast: false, residency: .onDemand, waitForMemory: true)
    #expect(try Data(contentsOf: file) == before)
    #expect(await service.rows().first?.selectedPrecision == "int4")
    #expect(await service.rows().first?.residentDType == "bf16")
    // Explicit Load with a dtype is a writer even if the requested resident is already loaded.
    try await service.load(id: id, precision: "bf16", fast: false)
    #expect(try ArsenalConfiguration.read(file).selections[id]?.precision == "bf16")
    #expect(await service.rows().first?.selectedPrecision == "bf16")
    await service.shutdown()
}

// Drama 3's 16-bit release has no runner here; the row must start on the vendor int4, or the first Get/Load fails.
@Test func unrunnableBaseFormatDefaultsToFirstRunnableFormat() async throws {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let drama = try ModelManifest.load(shippedOrRetiredFolder(repo, "fish-drama-3"), kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    #expect(drama.basePrecision == "bf16" && drama.precisions == ["bf16", "int4"])
    #expect(drama.defaultPrecision == "int4" && drama.runnablePrecision("bf16") == "int4" && drama.runnablePrecision("int4") == "int4")

    let formats: [TOMLValue] = [
        .table(["id": .string("bf16"), "bits_per_weight": .integer(16), "source": .string("vendor"), "storage": .string("bf16"), "runners": .array([])])
    ]
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], extraTable: ["format": .array(formats)])
    defer { fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent()) }
    let row = try #require(await catalogRow(fixture.service))
    #expect(row.precisions == ["bf16", "int8", "int4"])
    #expect(row.selectedPrecision == "int4")
    #expect(row.dtypeUnavailableReason("bf16") == "No runner in this Arsenal yet." && row.dtypeUnavailableReason("int8") == nil)
    #expect(await fixture.service.catalog(query: .init()).first?.row.selectedPrecision == "int4")
    await fixture.service.shutdown()
}
