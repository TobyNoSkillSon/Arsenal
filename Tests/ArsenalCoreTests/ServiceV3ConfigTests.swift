// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func v3ConfigMergePreservesUntouchedNestedAndUnknownKeys() throws {
    let file = URL(fileURLWithPath: "/tmp/arsenal-v3-config-\(UUID().uuidString)/config.json")
    var baseline = ArsenalConfiguration(); baseline.selections["m"] = .init(precision: "bf16")
    try baseline.write(file)
    var disk = try WireJSON.decode(Data(contentsOf: file))
    var selections = disk["selections"]?.object ?? [:]
    var selection = selections["m"]?.object ?? [:]
    selection["fast"] = false; selection["agent"] = "kept"
    selections["m"] = .object(selection); disk["selections"] = .object(selections)
    disk["appearance"] = "light"; disk["future"] = ["value": 42]
    try WireJSON.encode(disk).write(to: file)
    var candidate = baseline; candidate.selections["m"]?.keepHot = true
    let merged = try candidate.writeMerging(file, baseline: baseline)
    #expect(merged.selections["m"]?.precision == "bf16")
    #expect(merged.selections["m"]?.fast == true && merged.selections["m"]?.keepHot == true)
    let saved = try WireJSON.decode(Data(contentsOf: file))
    #expect(saved["future"]?["value"] == 42 && saved["selections"]?["m"]?["agent"] == "kept")
    #expect(saved["appearance"] == nil)
}

@Test func v3ExternalEditsDoNotBlockGetLoadOrOnDemandRun() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices", "chat"]); defer { f.stop() }
    let file = f.paths.support.appendingPathComponent("config.json")
    func edit() throws {
        var disk = try WireJSON.decode(Data(contentsOf: file)); disk["future"] = "kept"; disk["appearance"] = "light"
        try WireJSON.encode(disk).write(to: file, options: .atomic)
    }
    try edit(); try await f.service.get(id: "fixture-voices", precision: "bf16")
    try edit(); try await f.service.load(id: "fixture-voices", precision: "bf16", fast: false)
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    try edit(); _ = try await f.service.run(id: "fixture-chat", op: "chat", params: ["prompt": "hello", "stream": false])
    #expect(await f.service.rows().allSatisfy { $0.residentDType == "bf16" })
    #expect(try WireJSON.decode(Data(contentsOf: file))["future"] == "kept")
    await f.service.shutdown()
}

@Test func v3RefusedSaveLeavesKeepHotAndResidencyUnchanged() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    let file = f.paths.support.appendingPathComponent("config.json")
    try Data("broken json".utf8).write(to: file)
    await #expect(throws: (any Error).self) { try await f.service.setKeepHot(id: "fixture-voices", on: true) }
    await #expect(throws: (any Error).self) { try await f.service.setResidencySettings(.init(onDemandIdleMinutes: 5)) }
    #expect(await catalogRow(f.service, "fixture-voices")?.keepHot == false)
    #expect(await f.service.residencySettings().onDemandIdleMinutes == 15)
    #expect(try Data(contentsOf: file) == Data("broken json".utf8))
    await f.service.shutdown()
}

@Test func v3InvalidExternalConfigDoesNotUndoSuccessfulLoad() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    try await f.service.get(id: "fixture-voices", precision: "bf16")
    try Data("broken json".utf8).write(to: f.paths.support.appendingPathComponent("config.json"))
    try await f.service.load(id: "fixture-voices", precision: "bf16", fast: false)
    #expect(await f.service.rows().first?.residentDType == "bf16")
    await f.service.shutdown()
}

@Test func v3GetRejectsUnknownSizeOnceWithActionableError() throws {
    let plan = GetPlan(id: "m", dtype: "bf16", runner: "native", items: [.init(id: "weights", name: "Model assets", kind: "weights", downloadBytes: nil)])
    do { try plan.requireKnownDownloadSize(); Issue.record("Unpinned size accepted") } catch let failure as WorkerFailure {
        #expect(failure.code == "download_size" && failure.message.contains("Model assets") && failure.message.contains("Refresh"))
    }
}

@Test func v3ConfigListsMergeOutsideMembershipAndAppChanges() throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-v3-lists-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("config.json")
    var baseline = ArsenalConfiguration(); baseline.table = ["a", "b", "c"]; baseline.kindOrder = ["chat", "voices", "images"]
    try baseline.write(file)
    var disk = try WireJSON.decode(Data(contentsOf: file))
    disk["table"] = ["a", "b", "outside"]; disk["kindOrder"] = ["outside", "chat", "voices", "images"]
    try WireJSON.encode(disk).write(to: file)
    var candidate = baseline; candidate.table = ["b", "c", "app"]; candidate.kindOrder = ["voices", "chat", "images"]
    let merged = try candidate.writeMerging(file, baseline: baseline)
    #expect(merged.table == ["b", "outside", "app"])
    #expect(merged.kindOrder == ["voices", "chat", "images"])
}

@Test func v3OutsideToolsEditInvalidatesCatalogCacheOnSave() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    // Downloaded, so the table shows it too.
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    // Added models start out of the menu; an outside edit turns Show in menu on.
    #expect(await f.service.catalog(query: .init()).first?.toolsShown == false)
    let file = f.paths.support.appendingPathComponent("config.json")
    var disk = try WireJSON.decode(Data(contentsOf: file)), preferences = disk["selections"]?.object ?? [:]
    var selection = preferences["fixture-chat"]?.object ?? [:]; selection["toolsShown"] = true
    preferences["fixture-chat"] = .object(selection); disk["selections"] = .object(preferences)
    try WireJSON.encode(disk).write(to: file)
    try await f.service.setKeepHot(id: "fixture-chat", on: true)
    #expect(await f.service.rows().first?.toolsShown == true)
    #expect(await f.service.catalog(query: .init()).first?.toolsShown == true)
    await f.service.shutdown()
}

@Test func v3UnreadableConfigUsesPlainMessageAndPreservesState() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let file = f.paths.support.appendingPathComponent("config.json")
    try Data("broken json".utf8).write(to: file)
    do { try await f.service.setKeepHot(id: "fixture-chat", on: true); Issue.record("Corrupt config saved") } catch let error as WorkerFailure {
        #expect(error.message == "Arsenal could not read config.json. Repair the file and try again.")
    }
    #expect(await catalogRow(f.service, "fixture-chat")?.keepHot == false)
    do { _ = try ArsenalConfiguration.read(file); Issue.record("Corrupt config read") } catch let error as WorkerFailure { #expect(error.code == "config") }
    await f.service.shutdown()
}

private actor V3CorruptOnEvidenceSource: CatalogSource {
    let source: FixtureCatalogSource
    let file: URL
    init(source: FixtureCatalogSource, file: URL) { self.source = source; self.file = file }
    func read(path: String, commit: String) async throws -> Data { try await source.read(path: path, commit: commit) }
    func readAvailable(paths: [String], commit: String) async -> [String: Data] {
        try? Data("broken json".utf8).write(to: file)
        return await source.readAvailable(paths: paths, commit: commit)
    }
}
@Test func v3FailedMigrationSaveKeepsInstalledModelsVisible() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16"); await f.service.shutdown()
    let file = f.paths.support.appendingPathComponent("config.json")
    var legacy = try WireJSON.decode(Data(contentsOf: file)); legacy["table"] = nil; legacy["migrations"] = [:]
    let bytes = try WireJSON.encode(legacy); try bytes.write(to: file)
    let source = V3CorruptOnEvidenceSource(source: f.source, file: file)
    let service = try await RealModelsService.open(
        catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey, source: source,
        kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    #expect(await service.rows().map(\.id) == ["fixture-chat"])
    #expect(await service.config.table == ["fixture-chat"])
    await #expect(throws: WorkerFailure.self) { try await service.setKeepHot(id: "fixture-chat", on: true) }
    #expect(await service.config.table == ["fixture-chat"])
    #expect(await service.rows().first?.keepHot == false)
    try bytes.write(to: file)
    try await service.setKeepHot(id: "fixture-chat", on: true)
    let saved = try ArsenalConfiguration.read(file)
    #expect(saved.table == ["fixture-chat"] && saved.migrations["tableV2"] == true)
    await service.shutdown()
}

@Test func retiredAppearanceMigratesSilentlyWithoutLosingOtherConfiguration() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-appearance-migration-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("config.json")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    for legacy: JSONValue in ["light", "future-auto", 42, ["unknown": true]] {
        try ArsenalConfiguration().write(file)
        var disk = try WireJSON.decode(Data(contentsOf: file))
        disk["appearance"] = legacy; disk["future"] = ["kept": true]
        try WireJSON.encode(disk).write(to: file)
        let baseline = try ArsenalConfiguration.read(file)
        #expect(baseline.warnings.isEmpty)
        var candidate = baseline; candidate.residency.onDemandIdleMinutes = 5
        _ = try candidate.writeMerging(file, baseline: baseline)
        var saved = try WireJSON.decode(Data(contentsOf: file))
        #expect(saved["appearance"] == nil && saved["future"]?["kept"] == true)
        saved["appearance"] = legacy; try WireJSON.encode(saved).write(to: file)
        try candidate.write(file)
        #expect(try WireJSON.decode(Data(contentsOf: file))["appearance"] == nil)
    }
}
