// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Kit folders whose own kind declares no table figures (local.document, local.forecast, …) once never showed speed or
/// peak memory, and a figure file added to an installed folder would change the digest its code approval is bound to.
/// Arsenal now ships their quick checks in Resources/figures (bound to id, kind, weights revision, precision and runner)
/// and reads the folder's own quick-check records; the installed folder is never written.
struct FolderReferenceFiguresTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var shipped: FolderReferenceFigures { FolderReferenceFigures.load(resources: repo.appendingPathComponent("Resources")) }

    private func shippedManifest(_ id: String) throws -> ModelManifest {
        try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: repo.appendingPathComponent("models/\(id)/model.toml"), encoding: .utf8))
    }
    private func stat(_ value: Double, date: String) -> StatValue {
        StatValue(state: .quickCheck, value: value, provenance: StatProvenance(chip: "m5-max", date: date, method: "fixture", gpuCores: 40))
    }

    /// The 23 kit-kind models of the 1 Nov test night, each bound to its kit folder's own pins.
    @Test func shippedFiguresCoverTheKitKindModelsAndMatchTheirFolders() throws {
        let entries = shipped.entries
        #expect(entries.count == 23)
        #expect(Set(entries.map(\.model)).count == 23)
        for entry in entries {
            #expect(entry.kind.hasPrefix("local."), "\(entry.model)")
            #expect(entry.peakMemoryGB.state == .quickCheck && (entry.peakMemoryGB.value ?? 0) > 0, "\(entry.model)")
            #expect(entry.secondsPerRequest.state == .quickCheck && (entry.secondsPerRequest.value ?? 0) > 0, "\(entry.model)")
            #expect(entry.peakMemoryGB.provenance?.chip == "m5-max" && entry.peakMemoryGB.provenance?.gpuCores == 40, "\(entry.model)")
            // Every entry cites a file that exists (the kit folder's record, or the measurements file).
            let cited = try #require(entry.peakMemoryGB.provenance?.source?.split(separator: ";").first.map(String.init))
            let citedPath = cited.hasPrefix("~/") ? NSString(string: cited).expandingTildeInPath : repo.appendingPathComponent(cited).path
            #expect(FileManager.default.fileExists(atPath: citedPath), "\(entry.model): \(cited)")
            let folder = repo.appendingPathComponent("models/" + entry.model)
            guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("model.toml").path) else { continue }
            let manifest = try shippedManifest(entry.model)
            #expect(manifest.kind == entry.kind && manifest.weights.revision == entry.revision && manifest.precisions.contains(entry.precision), "\(entry.model)")
            #expect(entry.runner == (manifest.selection?.defaultRunner ?? manifest.runners.first?.name), "\(entry.model)")
            #expect(shipped.figures(manifest, runner: entry.runner)[entry.precision] == entry, "\(entry.model)")
            // The record it cites is in the kit folder.
            let source = try #require(entry.peakMemoryGB.provenance?.source?.split(separator: ";").first.map(String.init))
            #expect(FileManager.default.fileExists(atPath: repo.appendingPathComponent(source).path), "\(source)")
        }
    }

    /// Other weights, another runner or another kind get none of the shipped figures.
    @Test func figuresBindToWeightsRunnerAndKind() throws {
        var manifest = try shippedManifest("chronos-2")
        #expect(shipped.figures(manifest, runner: "kit")["fp16"]?.secondsPerRequest.value == 0.073)
        #expect(shipped.figures(manifest, runner: "native").isEmpty)
        var kind = manifest; kind.kind = "local.other"
        #expect(shipped.figures(kind, runner: "kit").isEmpty)
        manifest.weights.revision = String(repeating: "0", count: 40)
        #expect(shipped.figures(manifest, runner: "kit").isEmpty)
    }

    /// A folder's own quick-check record (kit/new-model.py) stands in; per precision the newest figure wins.
    @Test func folderRecordStandsInAndNewestWins() throws {
        let manifest = try shippedManifest("glm-ocr")
        let path = "results/quick-checks/kit-2026-10-31.json"
        let data = try Data(contentsOf: repo.appendingPathComponent("models/glm-ocr/" + path))
        let kind = try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).first { $0.id == manifest.kind }
        #expect(kind != nil)
        let records = FolderReferenceFigures.records([path: data, "results/other.json": Data("{}".utf8)], model: manifest, runner: "kit", kind: kind)
        #expect(records.count == 1)
        let record = try #require(records.first)
        #expect(record.precision == "fp16" && record.secondsPerRequest.value == 0.432)
        #expect(abs((record.peakMemoryGB.value ?? 0) - 4162.4 * 1_048_576 / 1e9) < 1e-9)
        #expect(record.secondsPerRequest.basis == "One warm Read request from the folder's own quick check.")
        #expect(record.peakMemoryGB.provenance?.source == path + "; fp16")
        // Without Arsenal's own figures the record stands; with them, the newer (1 Nov) wins.
        #expect(FolderReferenceFigures().figures(manifest, runner: "kit", records: records)["fp16"] == record)
        let newest = try #require(shipped.figures(manifest, runner: "kit", records: records)["fp16"])
        #expect(newest.peakMemoryGB.provenance?.run == "test-night-2026-11-01-fp16")
        // A record for other weights is ignored.
        var other = manifest; other.weights.revision = String(repeating: "1", count: 40)
        #expect(FolderReferenceFigures.records([path: data], model: other, runner: "kit", kind: kind).isEmpty)
    }

    /// The timed operation follows the importer's rule: primary, then default, then other declared, then by name.
    @Test func timedOperationFollowsTheImportersRule() throws {
        let op = { (name: String, isDefault: Bool) in
            OpDeclaration(name: name, title: name, input: "ops/\(name).input.json", output: "json", stream: false, defaultOp: isDefault)
        }
        var kind = KindSpec(id: "local.x", title: "X", order: 0, columns: [], operations: [op("b", false), op("c", true)])
        #expect(FolderReferenceFigures.timedOperation(completed: ["a", "b", "c"], kind: kind) == "c")
        #expect(FolderReferenceFigures.timedOperation(completed: ["a", "b"], kind: kind) == "b")
        #expect(FolderReferenceFigures.timedOperation(completed: ["z", "a"], kind: kind) == "a")
        kind.primaryOperation = "b"
        #expect(FolderReferenceFigures.timedOperation(completed: ["a", "b", "c"], kind: kind) == "b")
        #expect(FolderReferenceFigures.timedOperation(completed: [], kind: kind) == nil)
    }

    /// An entry needs a quick-check or measured state, a finite value and a dated provenance; otherwise it is no figure.
    @Test func invalidFiguresAreDropped() throws {
        let ok = stat(1.5, date: "2026-11-01T00:00:00Z")
        func entry(_ peak: StatValue, _ request: StatValue) -> FolderReferenceFigures.Entry {
            .init(model: "m", kind: "local.x", revision: "r", precision: "fp16", runner: "kit", peakMemoryGB: peak, secondsPerRequest: request)
        }
        let figures = FolderReferenceFigures(entries: [
            entry(StatValue(state: .derived, value: 1, provenance: ok.provenance), StatValue()),
            entry(stat(.infinity, date: "2026-11-01T00:00:00Z"), stat(1, date: "yesterday")),
            entry(ok, StatValue(state: .quickCheck, value: 0.1))
        ])
        #expect(figures.entries.count == 1)
        #expect(figures.entries.first?.peakMemoryGB == ok && figures.entries.first?.secondsPerRequest == StatValue())
        #expect(throws: (any Error).self) { _ = try FolderReferenceFigures.decode(Data(#"{"schema":"other","entries":[]}"#.utf8)) }
        #expect(FolderReferenceFigures.load(resources: URL(fileURLWithPath: "/nonexistent")).entries.isEmpty)
    }
}

/// An installed kit folder gets its figures from Arsenal's Resources and its own quick-check record without a byte of the
/// folder changing, so its code approval digest stays the one the owner approved.
@Test func installedKitFolderGetsShippedFiguresWithoutChangingItsDigest() async throws {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], addToTable: false); defer { f.stop() }
    let local = f.paths.support.appendingPathComponent("local-models/fixture-layers")
    try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: repo.appendingPathComponent("Tests/Fixtures/DropIn/fixture-layers"), to: local)
    let revision = String(repeating: "a", count: 40)
    // The folder's own record, as kit/new-model.py writes it (part of the approved bytes).
    let record: JSONObject = [
        "schema": "quick-check-1", "model": "fixture-layers", "revision": .string(revision), "chip": "m5-max", "gpu_cores": 40, "run": "kit-2026-10-30",
        "method": "fixture", "rows": [["precision": "bf16", "date": "2026-10-30T10:00:00Z", "peak_footprint_mb": 1000, "ops": ["separate": ["seconds_warm": 0.5, "status": 200]]]]
    ]
    try FileManager.default.createDirectory(at: local.appendingPathComponent("results/quick-checks"), withIntermediateDirectories: true)
    try WireJSON.encode(record).write(to: local.appendingPathComponent("results/quick-checks/kit-2026-10-30.json"))
    let before = try LocalModelFolders.discoverSnapshot(local).contentDigest
    let row = { () async throws -> ModelRow in
        let entries = await f.service.catalog()
        return try #require(entries.first { $0.id == "fixture-layers" }?.row, "\(entries.map { "\($0.id): \($0.row.verification)" })")
    }
    // Before Arsenal's own figures are read: the folder's record.
    #expect(try await row().folderFigures["bf16"]?.secondsPerRequest.value == 0.5)
    #expect(try await row().folderFigures["bf16"]?.secondsPerRequest.basis == "One warm Separate layers request from the folder's own quick check.")
    // Arsenal's Resources carry a newer quick check of the same pins.
    let resources = f.paths.support.deletingLastPathComponent().appendingPathComponent("resources-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: resources) }
    let provenance = StatProvenance(chip: "m5-max", date: "2026-11-01T02:00:00Z", method: "fixture night", gpuCores: 40)
    let shipped: JSONObject = [
        "schema": "folder-figures-1",
        "entries": [
            [
                "model": "fixture-layers", "kind": "local.layers", "revision": .string(revision), "precision": "bf16", "runner": "swift",
                "peakMemoryGB": .object(try WireJSON.decode(JSONEncoder().encode(StatValue(state: .quickCheck, value: 2.5, provenance: provenance)))),
                "secondsPerRequest": .object(try WireJSON.decode(JSONEncoder().encode(StatValue(state: .quickCheck, value: 0.25, provenance: provenance))))
            ]
        ]
    ]
    try FileManager.default.createDirectory(at: resources.appendingPathComponent("figures"), withIntermediateDirectories: true)
    try WireJSON.encode(shipped).write(to: resources.appendingPathComponent(FolderReferenceFigures.path))
    await f.service.setBundledResources(resources)
    let after = try await row()
    #expect(after.local && after.contentDigest == before)
    #expect(after.folderFigures["bf16"]?.secondsPerRequest.value == 0.25 && after.folderFigures["bf16"]?.peakMemoryGB.value == 2.5)
    #expect(try LocalModelFolders.discoverSnapshot(local).contentDigest == before)
    await f.service.shutdown()
}
