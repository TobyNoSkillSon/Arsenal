// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private func runnerTable(_ name: String, language: String = "swift", chips: [String] = ["*"], os: String = "26.0") -> TOMLValue {
    .table([
        "name": .string(name), "language": .string(language), "worker": .string("stub"), "protocol": .integer(1), "chips": .array(chips.map(TOMLValue.string)),
        "min_macos": .string(os), "fast_exact": .bool(true)
    ])
}
private let v21Runners: [String: TOMLValue] = [
    "runner": .array([runnerTable("native"), runnerTable("custom", language: "rust"), runnerTable("foreign", chips: ["imaginary"]), runnerTable("future", os: "99.0")]),
    "selection": .table(["default": .string("custom"), "chips": .table([RunnerSelection.family(HostChip.current): .string("native")])])
]

@Test func v21RunnerChoicePersistsAndReloadsWithoutReplacingResidentOnRun() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: v21Runners); defer { f.stop() }
    let s = f.service, id = "fixture-voices"
    #expect(await s.catalog().first?.runners.filter(\.acceptedDefault).map(\.name) == ["native"])
    #expect(await s.catalog().first?.runners.filter(\.eligible).map(\.name) == ["native", "custom"])
    try await s.get(id: id, precision: "bf16"); try await s.load(id: id)
    try await s.setRunner(id: id, runner: "custom")
    let pending = try #require(await s.rows().first)
    #expect(pending.needsReload && pending.runner == "custom" && pending.runnerOverride == "custom" && pending.residentRunner == "native")
    _ = try await s.run(id: id, op: "voices")
    let afterRun = await s.rows().first
    #expect(afterRun?.needsReload == true && afterRun?.residentRunner == "native")
    try await s.load(id: id)
    let afterLoad = await s.rows().first
    #expect(afterLoad?.residentRunner == "custom" && afterLoad?.needsReload == false)
    try await s.setKeepHot(id: id, on: true)
    await s.shutdown()
    let reopened = try await f.reopen(threshold: 8_000_000_000)
    let restored = await reopened.rows().first
    #expect(restored?.runner == "custom" && restored?.keepHot == true)
    try await reopened.setRunner(id: id, runner: nil)
    let reset = await reopened.rows().first
    #expect(reset?.runner == "native" && reset?.runnerOverride == nil)
    for name in ["foreign", "future", "missing", ""] {
        await #expect(throws: WorkerFailure.self) { try await reopened.setRunner(id: id, runner: name) }
        #expect(await reopened.rows().first?.runner == "native")
    }
    let configuration = try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json"))
    #expect(configuration.runnerOverrides[id] == nil)
    await reopened.shutdown()
}

private actor EvidenceSource: CatalogSource {
    var files: [String: Data], evidenceReads = 0
    init(files: [String: Data]) { self.files = files }
    func read(path: String, commit: String) async throws -> Data {
        if path.contains("/log/") || path.contains("/results/") { evidenceReads += 1 }
        guard let data = files[path] else { throw ManifestError.invalid("Missing fixture file") }; return data
    }
}
@Test func v21DetailsCarryVerifiedHistoryResultsAndSettingsWithoutRenderReads() async throws {
    let result = ResultsFile(
        model: "fixture-voices", kind: "voices", chip: HostChip.current, osBuild: "fixture", appBuild: "fixture", runner: "custom", workerDigest: String(repeating: "a", count: 64),
        precision: "bf16", mode: "fast", suite: .init(id: "fixture", sha256: String(repeating: "b", count: 64)), profile: "quick", stub: true, gatePassed: true,
        conditions: .init(locked: false, idle: [], run: [], note: "fixture; not a measurement"), metrics: ["rtf": 0.5], cases: [])
    var otherChip = result; otherChip.chip = "imaginary"
    var table = v21Runners
    table["description"] = .string("A longer folder description.")
    table["links"] = .table(["paper": .string("https://example.com/paper")])
    table["setting"] = .array([
        .table([
            "id": .string("speed"), "title": .string("Speed"), "type": .string("number"), "default": .integer(1), "applies": .string("request"), "maps_to": .string("request.speed")
        ])
    ])
    let files = [
        "results/\(RunnerSelection.family(HostChip.current)).json": try JSONEncoder().encode(result), "results/other.json": try JSONEncoder().encode(otherChip),
        "log/002-custom.md": Data("+++\nrunner = \"custom\"\nverdict = \"kept\"\n+++\n# Custom runner proof\n".utf8),
        "log/003-native.md": Data("+++\nrunner = \"native\"\nverdict = \"rejected\"\n+++\n# Native experiment\n".utf8)
    ]
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: table, extraFiles: files); defer { f.stop() }; await f.service.shutdown()
    let source = EvidenceSource(files: f.source.files)
    let s = try await RealModelsService.open(
        catalogData: f.catalogData, signature: f.signature, publicKey: f.publicKey, source: source, kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(),
        client: f.client)
    let reads = await source.evidenceReads
    for _ in 0..<3 {
        let entry = try #require(await s.catalog().first)
        #expect(entry.description == "A longer folder description." && entry.publisher == "Arsenal" && entry.released == 2026)
        #expect(entry.links["paper"] == "https://example.com/paper" && entry.links["huggingface"] == "https://huggingface.co/test/fixture")
        #expect(entry.licenceURL == "https://example.com/licence" && entry.languages == ["en"] && entry.sizesByDtype["bf16"] == 3)
        #expect(entry.settingsSummary.first?.id == "speed" && entry.settingsSummary.first?.value == 1)
        let custom = try #require(entry.runners.first { $0.name == "custom" })
        #expect(custom.language == "rust" && custom.results.count == 1 && custom.results.first?["runner"] == "custom")
        #expect(custom.history?.title == "Custom runner proof" && custom.history?.verdict == "kept" && custom.history?.path == "log/002-custom.md")
        #expect(try JSONDecoder().decode(CatalogEntry.self, from: JSONEncoder().encode(entry)) == entry)
    }
    let finalReads = await source.evidenceReads
    #expect(reads > 0 && finalReads == reads)
    await s.shutdown()
}
@Test func v21LegacyConfigurationAndCatalogDecodeStayCompatible() throws {
    let config = try JSONDecoder().decode(ArsenalConfiguration.self, from: Data("{}".utf8))
    #expect(config.runnerOverrides.isEmpty)
    let row = ModelRow(id: "legacy", name: "Legacy", kind: "voices", summary: "Existing summary")
    let entry = try JSONDecoder().decode(CatalogEntry.self, from: JSONEncoder().encode(row))
    #expect(entry.description == row.summary && entry.runners.isEmpty && entry.settingsSummary.isEmpty)
}

@Test func v21HistorySupportsPortableYAMLAndRunnerSpecificAttribution() {
    let files: [String: Data] = [
        "log/001-other.md": Data("+++\nrunner = \"other\"\nverdict = \"rejected\"\n+++\n# Other runner\n".utf8),
        "log/002-custom.md": Data("---\nrunner: custom\ntitle: 'A useful title'\nverdict: kept\n---\nBody\n".utf8)
    ]
    #expect(CatalogHistory.latest(files, runner: "custom") == .init(path: "log/002-custom.md", title: "A useful title", verdict: "kept"))
    #expect(CatalogHistory.latest(files, runner: "other")?.title == "Other runner")
    #expect(CatalogHistory.latest(files, runner: "missing") == nil)
}
@Test func v21RunnerInvalidPersistedOverrideFallsBack() async throws {
    var configuration = ArsenalConfiguration(); configuration.runnerOverrides["fixture-voices"] = "foreign"
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], configuration: configuration, extraTable: v21Runners); defer { f.stop() }
    let row = try #require(await catalogRow(f.service, "fixture-voices"))
    #expect(row.runner == "native" && row.runnerOverride == nil)
    await f.service.shutdown()
}

private struct RejectCustomArtifact: WorkerArtifact {
    func validateEngine(model: ModelManifest, runner: RunnerSpec) throws {
        if runner.name == "custom" { throw WorkerFailure(code: "worker_missing", message: "Fixture engine unavailable") }
    }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch { stubLaunch(model.kind) }
}
@Test func v21RunnerMissingEngineRefusesMutation() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], artifacts: RejectCustomArtifact(), extraTable: v21Runners); defer { f.stop() }
    await #expect(throws: WorkerFailure.self) { try await f.service.setRunner(id: "fixture-voices", runner: "custom") }
    #expect(await catalogRow(f.service, "fixture-voices")?.runner == "native")
    #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).runnerOverrides.isEmpty)
    await f.service.shutdown()
}
