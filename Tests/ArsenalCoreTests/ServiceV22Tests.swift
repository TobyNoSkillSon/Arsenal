// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import ArsenalCore

@Test func v22DtypeAndSizeConstraintsMustMatchTheSameDtype() {
    let row = ModelRow(id: "large", name: "Large", kind: "voices", precisions: ["bf16", "int4"], selectedPrecision: "bf16")
    let entry = CatalogEntry(row: row, sizesByDtype: ["bf16": 12_000_000_000, "int4": 3_000_000_000])
    var query = CatalogQuery(); query.dtypes = ["bf16"]; query.maxSizeBytes = 8_000_000_000
    #expect(CatalogEntry.query([entry], query).isEmpty)
    query.dtypes = ["int4"]
    #expect(CatalogEntry.query([entry], query).first?.row.selectedPrecision == "int4")
}

@Test func v22RangesAndRAMAreConjoinedWithinOneVariantAndSortUsesIt() throws {
    var entry = CatalogEntry(
        row: ModelRow(id: "m", name: "Model", kind: "voices", precisions: ["bf16", "int8", "int4"], selectedPrecision: "bf16"),
        sizesByDtype: ["bf16": 12_000_000_000, "int8": 6_000_000_000, "int4": 3_000_000_000])
    entry.ramMBByDtype = ["bf16": 14000, "int8": 8000, "int4": 4000]
    entry.fitsRAMByDtype = ["bf16": false, "int8": true, "int4": true]
    var query = CatalogQuery(); query.minSizeBytes = 4_000_000_000; query.maxSizeBytes = 8_000_000_000; query.maxRAMMB = 6000
    #expect(CatalogEntry.query([entry], query).isEmpty)
    query.maxRAMMB = 9000; query.fitsRAM = true
    let match = try #require(CatalogEntry.query([entry], query).first)
    #expect(match.matchingDtype == "int8" && match.row.selectedPrecision == "int8" && match.row.sizeBytes == 6_000_000_000 && match.row.peakRAMMB == 8000)
    #expect(entry.row.selectedPrecision == "bf16")
    #expect(try JSONDecoder().decode(CatalogEntry.self, from: JSONEncoder().encode(match)) == match)
    let small = CatalogEntry(row: ModelRow(id: "s", name: "Small", kind: "voices", precisions: ["int4"], selectedPrecision: "int4"), sizesByDtype: ["int4": 1_000_000_000])
    query = CatalogQuery(sort: .size); query.dtypes = ["int4"]
    #expect(CatalogEntry.query([entry, small], query).map(\.id) == ["s", "m"])
}

private func v22Table(_ text: String) throws -> [String: TOMLValue] { var parser = TOMLParser(text); return try parser.parse() }

@Test func v22GeneralFormatsGetDerivedVendorAndRunnerNativeThroughRealStore() async throws {
    let hash = FileDigest.sha256(Data([12, 34, 56])), revision = String(repeating: "a", count: 40)
    let table = try v22Table(
        """
        [derive]
        exl3 = "opaque-fixture-recipe"
        [[format]]
        id = "exl3"
        storage = "exl3"
        method = "trellis"
        bits_per_weight = 3.25
        source = "derived"
        runners = ["native"]
        size_bytes = 2
        [[format]]
        id = "oQ4e"
        storage = "affine"
        method = "oq"
        profile = { group_size = 64, default_bits = 4, module_bits = { lm_head = 8 } }
        bits_per_weight = 4.5
        source = "runner-native"
        runners = ["native"]
        [[format]]
        id = "mxfp4"
        storage = "mxfp4"
        bits_per_weight = 4
        source = "vendor"
        runners = ["native"]
        [vendor.mxfp4]
        repo = "test/fixture"
        revision = "\(revision)"
        dtype = "mxfp4"
        files = ["base.bin"]
        sha256 = { "base.bin" = "\(hash)" }
        bytes = { "base.bin" = 3 }
        """)
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: table); defer { f.stop() }
    let id = "fixture-voices"
    let entry = try #require(await f.service.catalog().first)
    #expect(Set(entry.dtypes) == Set(["bf16", "exl3", "oQ4e", "mxfp4"]))
    #expect(entry.sizesByDtype["exl3"] == 2 && entry.sizesByDtype["mxfp4"] == 3)
    #expect(entry.formats.first { $0.id == "oQ4e" }?.profile?["module_bits"]?["lm_head"] == 8)
    for dtype in ["exl3", "oQ4e", "mxfp4"] {
        try await f.service.get(id: id, precision: dtype)
        try await f.service.load(id: id, precision: dtype, fast: false)
        #expect(try await f.service.run(id: id, op: "test_config")["dtype"] == .string(dtype))
        try await f.service.unload(id: id)
    }
    await f.service.shutdown()
}

@Test func v22UnsupportedFormatRemainsFilterableButGetAndLoadRefuse() async throws {
    let table = try v22Table(
        """
        [[format]]
        id = "future-q3"
        storage = "exl3"
        bits_per_weight = 3
        source = "runner-native"
        runners = []
        """)
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: table); defer { f.stop() }
    var query = CatalogQuery(); query.dtypes = ["future-q3"]
    let entry = try #require(await f.service.catalog(query: query).first)
    #expect(entry.matchingDtype == "future-q3" && entry.row.verification == "No runner in this Arsenal yet.")
    let row = try #require(await catalogRow(f.service, "fixture-voices"))
    #expect(row.dtypeUnavailableReason("future-q3") == "No runner in this Arsenal yet.")
    #expect(row.dtypeUnavailableReason("bf16") == nil)
    #expect(try JSONDecoder().decode(ModelRow.self, from: JSONEncoder().encode(row)) == row)
    await #expect(throws: WorkerFailure.self) { _ = try await f.service.getPlan(id: "fixture-voices", dtype: "future-q3") }
    await #expect(throws: WorkerFailure.self) { try await f.service.get(id: "fixture-voices", precision: "future-q3") }
    await #expect(throws: WorkerFailure.self) { try await f.service.load(id: "fixture-voices", precision: "future-q3", fast: false) }
    #expect(await f.service.memory().workersMB == 0)
    await f.service.shutdown()
}

@Test func v22GetPlanForecastDoesNotFetchAndReflectsInstalledWeights() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    let plan = try await f.service.getPlan(id: "fixture-voices", dtype: "int4")
    #expect(plan.totalDownloadBytes == 3 && plan.requiresDownload && plan.items.first?.kind == "weights")
    #expect(plan.items.contains { $0.kind == "engine" && $0.downloadBytes == 0 })
    #expect(try JSONDecoder().decode(GetPlan.self, from: JSONEncoder().encode(plan)) == plan)
    // The forecast fetched nothing: no weights, so the model is not on the table yet.
    #expect(await catalogRow(f.service, "fixture-voices")?.state == .notInstalled)
    #expect(await f.service.rows().isEmpty)
    try await f.service.get(id: "fixture-voices", precision: "int4")
    #expect(try await f.service.getPlan(id: "fixture-voices", dtype: "int4").totalDownloadBytes == 0)
    await f.service.shutdown()
}

@Test func v22RefreshReReadsConfigAndKeepsResidentMenuAndWorker() async throws {
    let table = try v22Table(
        """
        [[setting]]
        id = "context"
        title = "Context"
        type = "choice"
        default = 16
        choices = [16, 32]
        applies = "load"
        maps_to = "config.context_limit"
        [[menu.section]]
        title = "Reader"
        items = [{id="read", title="Read clipboard", op="speak", input="clipboard"}]
        """)
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: table); defer { f.stop() }
    #expect(await f.service.menuSections().isEmpty)
    try await f.service.get(id: "fixture-voices", precision: "bf16")
    try await f.service.load(id: "fixture-voices", precision: "bf16", fast: false)
    let sections = await f.service.menuSections()
    #expect(sections.first?.title == "Reader" && sections.first?.items.first?.op == "speak")
    let before = try await f.service.run(id: "fixture-voices", op: "test_config")
    let file = f.paths.support.appendingPathComponent("config.json")
    var config = try ArsenalConfiguration.read(file); config.settings["fixture-voices"] = ["context": 32]
    try config.write(file)
    try await f.service.refresh()
    #expect(await f.service.rows().first?.needsReload == true)
    #expect(await f.service.menuSections() == sections)
    let after = try await f.service.run(id: "fixture-voices", op: "test_config")
    #expect(after["config"] == before["config"] && after["dtype"] == before["dtype"])
    #expect(await f.service.rows().first?.states["bf16"]?.state == .loaded)
    try await f.service.unload(id: "fixture-voices")
    #expect(await f.service.menuSections().isEmpty)
    await f.service.shutdown()
}

@Test func v22RefreshDiscoversDroppedLocalFolderAndRejectsInvalidConfigWithoutLosingResident() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    try await f.service.get(id: "fixture-voices", precision: "bf16")
    try await f.service.load(id: "fixture-voices", precision: "bf16", fast: false)
    let source = f.paths.support.deletingLastPathComponent().appendingPathComponent("folders/fixture-voices")
    let local = f.paths.support.appendingPathComponent("local-models/dropped-voice")
    try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: source, to: local)
    let manifest = local.appendingPathComponent("model.toml")
    let text = try String(contentsOf: manifest, encoding: .utf8).replacingOccurrences(of: "fixture-voices", with: "dropped-voice")
    try text.write(to: manifest, atomically: true, encoding: .utf8)
    try await f.service.refresh()
    #expect(await f.service.catalog().contains { $0.id == "dropped-voice" && $0.local })
    let file = f.paths.support.appendingPathComponent("config.json"), saved = try Data(contentsOf: file)
    try Data("invalid json".utf8).write(to: file)
    await #expect(throws: (any Error).self) { try await f.service.refresh() }
    #expect(await f.service.rows().first?.states["bf16"]?.state == .loaded)
    #expect(try await f.service.run(id: "fixture-voices", op: "test_config")["dtype"] == "bf16")
    try saved.write(to: file)
    await f.service.shutdown()
}

private actor V22MutableSource: CatalogSource {
    var files: [String: Data]
    init(_ files: [String: Data]) { self.files = files }
    func read(path: String, commit: String) throws -> Data {
        guard let data = files[path] else { throw ManifestError.invalid("missing fixture file") }; return data
    }
    func replace(path: String, data: Data) { files[path] = data }
}

@Test func v22RefreshReadsNewSignedCatalogWithoutReplacingResidentAndRejectsBadSignature() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }; await f.service.shutdown()
    let key = Curve25519.Signing.PrivateKey(), source = V22MutableSource(f.source.files)
    let s = try await RealModelsService.open(
        catalogData: f.catalogData, signature: key.signature(for: f.catalogData), publicKey: key.publicKey.rawRepresentation, source: source, kinds: KindSpec.defaults,
        paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    let id = "fixture-voices"
    try await s.get(id: id, precision: "bf16"); try await s.load(id: id, precision: "bf16", fast: false)
    var doc = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData)
    doc.sequence += 1; doc.sourceCommit = String(repeating: "b", count: 40); doc.models[0].manifest.name = "Updated voice"
    let path = "models/\(id)/model.toml"
    let original = try #require(f.source.files[path])
    let updated = Data(String(decoding: original, as: UTF8.self).replacingOccurrences(of: "Fixture voices", with: "Updated voice").utf8)
    await source.replace(path: path, data: updated)
    let index = try #require(doc.models[0].files.firstIndex { $0.path == "model.toml" })
    doc.models[0].files[index] = CatalogFile(path: "model.toml", sha256: FileDigest.sha256(updated), bytes: Int64(updated.count))
    let data = try doc.encoded(), directory = f.paths.directory("catalog")
    try data.write(to: directory.appendingPathComponent("catalog.json")); try key.signature(for: data).write(to: directory.appendingPathComponent("catalog.json.sig"))
    try await s.refresh()
    #expect(await s.rows().first?.name == "Updated voice")
    #expect(await s.rows().first?.states["bf16"]?.state == .loaded)
    #expect(await s.rows().first?.needsReload == true)
    #expect(try await s.run(id: id, op: "test_config")["dtype"] == "bf16")
    try Data(repeating: 0, count: 64).write(to: directory.appendingPathComponent("catalog.json.sig"))
    await #expect(throws: (any Error).self) { try await s.refresh() }
    #expect(await s.rows().first?.states["bf16"]?.state == .loaded)
    await s.shutdown()
}

@Test func v22RefreshRemovingTableMembershipDoesNotHideOrInterruptResident() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    try await f.service.get(id: "fixture-voices", precision: "bf16")
    try await f.service.load(id: "fixture-voices", precision: "bf16", fast: false)
    let file = f.paths.support.appendingPathComponent("config.json")
    var config = try ArsenalConfiguration.read(file); config.table = []; try config.write(file)
    try await f.service.refresh()
    #expect(await f.service.rows().first?.residentDType == "bf16")
    #expect(try await f.service.run(id: "fixture-voices", op: "test_config")["dtype"] == "bf16")
    try await f.service.unload(id: "fixture-voices")
    // Its weights are still on this Mac, so it stays on the table though the file edit dropped its record.
    #expect(await f.service.rows().map { [$0.id: $0.onTable] } == [["fixture-voices": true]])
    await f.service.shutdown()
}

/// The table holds exactly the models with weights on this Mac or a Get under way. Removing a downloaded
/// model without deleting its weights leaves it on the table (no weights hide in the catalogue), where it loads and runs;
/// one that isn't downloaded is not on the table and is still refused; deleting the weights takes it off.
@Test func downloadedModelsStayOnTheTableUntilTheirWeightsAreDeleted() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices", "images"]); defer { f.stop() }
    try await f.service.get(id: "fixture-voices", precision: "bf16")
    try await f.service.remove(id: "fixture-voices", deleteWeights: false)
    #expect(await f.service.rows().map(\.id) == ["fixture-voices"])
    #expect(try await f.service.run(id: "fixture-voices", op: "test_config")["dtype"] == "bf16")
    let running = try #require(await f.service.rows().first { $0.id == "fixture-voices" })
    #expect(running.onTable && running.residentDType == "bf16")
    #expect(await f.service.catalog().first { $0.id == "fixture-voices" }?.added == true)
    try await f.service.unload(id: "fixture-voices")
    try await f.service.load(id: "fixture-voices", precision: "bf16", fast: false)
    try await f.service.unload(id: "fixture-voices")
    #expect(await f.service.rows().map(\.id) == ["fixture-voices"])
    // Not downloaded: not on the table, and refused as before.
    #expect(await f.service.catalog().first { $0.id == "fixture-images" }?.added == false)
    try await f.service.remove(id: "fixture-images", deleteWeights: false)
    await #expect(throws: WorkerFailure.self) { try await f.service.load(id: "fixture-images", precision: "bf16", fast: false) }
    try await f.service.add(id: "fixture-voices")
    #expect(await f.service.rows().map { [$0.id: $0.onTable] } == [["fixture-voices": true]])
    // Deleting its weights takes it off the table.
    try await f.service.remove(id: "fixture-voices", deleteWeights: true)
    #expect(await f.service.rows().isEmpty)
    #expect(await f.service.catalog().first { $0.id == "fixture-voices" }?.added == false)
    await f.service.shutdown()
}

/// Core review, 2 Nov: Get more models showed SeedVR2 needing 9.2 GB (weights + 768 MB) against the test night's
/// recorded 38.7 GB peak, so "fits this Mac" and find's RAM need misled smaller Macs.
@Test func catalogMemoryNeedUsesTheRecordedPeakBeforeTheWeightSize() {
    var row = ModelRow(id: "seedvr2-3b", name: "SeedVR2 3B", kind: "restoration", runner: "native")
    row.precisions = ["fp16"]; row.selectedPrecision = "fp16"
    var stats = PrecisionStatSet(kind: .restoration, dtype: "fp16", runner: "native")
    stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: 38.654705664))
    row.precisionStats["fp16"] = stats
    let size: Int64 = 8_440_000_000
    #expect(abs(CatalogEntry.ramMB(row: row, precision: "fp16", forecast: nil, measuredMB: nil, declaredMB: nil, sizeBytes: size) - 38_654.7) < 1)
    // Nothing recorded: the declared figure, then weights + 768 MB.
    row.precisionStats = [:]
    #expect(CatalogEntry.ramMB(row: row, precision: "fp16", forecast: nil, measuredMB: nil, declaredMB: 12_000, sizeBytes: size) == 12_000)
    #expect(CatalogEntry.ramMB(row: row, precision: "fp16", forecast: nil, measuredMB: nil, declaredMB: nil, sizeBytes: size) == 8_440 + 768)
}
