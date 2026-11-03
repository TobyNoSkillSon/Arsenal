// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// Test night 1 Nov, findings 1 and 2: the 23 kit models whose own kind declares no table columns showed a triangle for
/// speed and peak memory, and the kit kind local.document made a second, description-less "Documents" group beside the
/// bundled one (TeleOCR).
@MainActor @Suite(.menuState, .serialized) final class FolderKindFigureTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-folder-figures-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    private var shipped: FolderReferenceFigures { FolderReferenceFigures.load(resources: repo.appendingPathComponent("Resources")) }
    private func manifest(_ id: String) throws -> ModelManifest {
        try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: repo.appendingPathComponent("models/\(id)/model.toml"), encoding: .utf8))
    }
    /// A kit model's row as the service builds it: signed, kit runner, no declared figures, and the figures Arsenal
    /// ships for it.
    private func row(_ id: String, figures: Bool = true) throws -> ModelRow {
        let m = try manifest(id)
        var row = ModelRow(
            id: id, name: m.name, kind: m.kind, params: m.params, precisions: m.precisions, selectedPrecision: m.defaultPrecision, fast: false, runner: "kit",
            sizeBytes: 1_000_000, summary: m.summary, verification: "Not tried on this Mac yet", fastExact: false)
        for precision in m.precisions { row.states[precision] = PrecisionState(state: .ready) }
        if figures { row.folderFigures = shipped.figures(m, runner: "kit") }
        return row
    }
    /// The kit kinds ship with the signed kinds (Resources/kinds).
    private func kinds(_ models: [String]) throws -> [KindSpec] {
        let kinds = try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds"))
        for id in models {
            let kind = try manifest(id).kind
            #expect(kinds.contains { $0.id == kind }, "\(id)")
        }
        return kinds
    }

    @Test func kitKindRowShowsShippedQuickChecksAsTheOperationKindsDo() throws {
        let before = try row("chronos-2", figures: false)
        #expect(TableFigures.figure(before, precision: "fp16", column: .memory, host: "M5 Max").missing)
        #expect(TableFigures.figure(before, precision: "fp16", column: .speed, host: "M5 Max").missing)

        let chronos = try row("chronos-2")
        let memory = TableFigures.figure(chronos, precision: "fp16", column: .memory, host: "M5 Max")
        // 389.1 MiB (0.38 GiB) as Activity Monitor's GB.
        #expect(memory.value == "0.38 GB" && memory.reference == nil)
        #expect(memory.help.hasPrefix("fp16 · 0.38 GB at most"))
        #expect(memory.help.contains("Quick check on M5 Max · 40 GPU cores · 1 Nov 2026") && memory.help.contains(TableFigures.quickCheckHelp))
        let speed = TableFigures.figure(chronos, precision: "fp16", column: .speed, host: "M5 Max")
        #expect(speed.value == "73 ms" && speed.reference == nil && speed.delta == nil)
        #expect(speed.help.hasPrefix("fp16 · 73 ms per request") && speed.help.contains("One warm Forecast request"))
        // Another Mac: the speed is the M5 Max reference; peak memory stands as it is.
        let elsewhere = TableFigures.figure(chronos, precision: "fp16", column: .speed, host: "M3 Pro")
        #expect(elsewhere.value == "73 ms" && elsewhere.reference == "M5 Max" && elsewhere.help.contains("Not measured on your Mac; measured on M5 Max."))
        #expect(TableFigures.figure(chronos, precision: "fp16", column: .memory, host: "M3 Pro").reference == nil)
        #expect(RowPresentation.hasFigures(chronos))
        // The group's speed heading names what it shows.
        let forecasts = try #require(try kinds(["chronos-2"]).first { $0.id == "local.forecast" })
        #expect(TableStats.help(kind: forecasts, column: "speed") == "Time for one request of the model's own task. Lower is faster.")
    }

    /// Every one of the 23 models gets peak memory; speed wherever its group measures one request's time.
    @Test func everyShippedKitModelGetsPeakMemoryAndRequestTime() throws {
        for entry in shipped.entries {
            guard FileManager.default.fileExists(atPath: repo.appendingPathComponent("models/\(entry.model)/model.toml").path) else { continue }
            let r = try row(entry.model)
            #expect(!TableFigures.figure(r, precision: entry.precision, column: .memory, host: "M5 Max").missing, "\(entry.model)")
            #expect(!TableFigures.figure(r, precision: entry.precision, column: .speed, host: "M5 Max").missing, "\(entry.model)")
        }
    }

    /// The kit Documents kind joins the bundled Documents group: one group, its description, and a speed column whose unit
    /// (answer tokens per second) one request's time cannot give, so the kit readers' speed stays a triangle that says so.
    @Test func kitDocumentsJoinTheBundledDocumentsGroup() throws {
        let kinds = try kinds(["glm-ocr", "chronos-2"])
        #expect(KindPresentation.joined("local.document", kinds: kinds) == "documents")
        #expect(KindPresentation.joined("local.forecast", kinds: kinds) == nil)
        #expect(KindPresentation.joined("chat", kinds: kinds) == nil)
        // A folder kind declaring its own figure columns keeps its group: its units are not the group's.
        var declaring = try #require(kinds.first { $0.id == "local.document" })
        declaring.columns = [KindColumn(id: "speed", title: "Speed", unit: "s")]
        #expect(KindPresentation.joined("local.document", kinds: kinds.filter { $0.id != "local.document" } + [declaring]) == nil)
        #expect(KindPresentation.id(kind: "local.document", tasks: ["image-to-text"], kinds: kinds) == "documents")

        var teleocr = ModelRow(id: "teleocr", name: "TeleOCR", kind: "chat", precisions: ["fp16"], selectedPrecision: "fp16")
        teleocr.states["fp16"] = PrecisionState(state: .ready)
        let glm = try row("glm-ocr"), manga = try row("manga-ocr-base"), chronos = try row("chronos-2")
        let rows = [teleocr, glm, manga, chronos]
        var shown = ["teleocr": KindPresentation.id(kind: "chat", tasks: ["read_document"], kinds: kinds)]
        for r in [glm, manga, chronos] {
            let id = KindPresentation.id(kind: r.kind, tasks: [], kinds: kinds)
            if id != r.kind { shown[r.id] = id }
        }
        let c = TableController(service: FixtureService(rows: rows, supportDirectory: support), chip: "M5 Max")
        c.apply(rows: rows, kinds: kinds, catalog: CatalogStatus(source: "local", pinnedCommit: "f600b7d4c1e2", verified: true))
        c.setShownKinds(shown)
        let titles = c.groups.map(\.kind.title)
        #expect(titles.filter { $0 == "Documents" }.count == 1, "\(titles)")
        let documents = try #require(c.groups.first { $0.kind.title == "Documents" })
        #expect(documents.kind.id == "documents" && documents.rows.map(\.id) == ["teleocr", "glm-ocr", "manga-ocr-base"])
        #expect(GroupHeader.titleHelp(documents.kind).hasPrefix("Read pages, tables and handwriting into text"))
        #expect(c.groups.contains { $0.kind.id == "local.forecast" && $0.rows.map(\.id) == ["chronos-2"] })

        let speed = TableFigures.figure(glm, precision: "fp16", column: .speed, host: "M5 Max", shown: c.shownKinds["glm-ocr"])
        #expect(speed.missing && speed.help == "fp16 · Speed in Documents is answer tokens per second, which one request's time cannot give.")
        let memory = TableFigures.figure(glm, precision: "fp16", column: .memory, host: "M5 Max", shown: c.shownKinds["glm-ocr"])
        #expect(memory.value == "5.88 GB")
    }
}
