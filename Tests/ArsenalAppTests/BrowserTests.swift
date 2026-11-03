// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// The model browser (v21-ui): sidebar filters, sorting, search, count line, Add/Remove, expansion, column geometry and
/// the expanded row's details, on catalog entries built the way the service does.
@MainActor @Suite(.menuState, .serialized) final class BrowserTests {
    static let kinds = [
        KindSpec(
            id: "voices", title: "Voices", order: 0, columns: [.init(id: "speed", title: "Speed"), .init(id: "first_audio", title: "First audio")], icon: "waveform",
            titleSingular: "Voice", facets: ModelFacets(modalitiesIn: ["text"], modalitiesOut: ["audio"])),
        KindSpec(id: "decisions", title: "Decisions", order: 1, columns: [], icon: "checkmark.circle", titleSingular: "Decision"),
        KindSpec(id: "chat", title: "Chat", order: 3, columns: [], icon: "text.bubble", titleSingular: "Chat", facets: ModelFacets(modalitiesIn: ["text"], modalitiesOut: ["text"]))
    ]
    static func entry(
        _ id: String, _ name: String, kind: String, params: String, bytes: Int64, licence: String = "Apache-2.0", commercial: Bool = true,
        languages: [String] = ["en"], suggested: Bool = false, downloads: Double = 0, ram: Double = 1000, added: Bool = false, released: Int? = nil,
        badge: ModelBadge = .unverified, verification: String = "Not tried on this Mac yet", installed: Bool = false
    ) -> CatalogEntry {
        var row = ModelRow(
            id: id, name: name, kind: kind, params: params, licence: licence, precisions: ["bf16", "int8"], selectedPrecision: "bf16",
            badge: badge, peakRAMMB: ram, downloadBytes: ["bf16": bytes, "int8": bytes], verification: verification, released: released, commercial: commercial)
        if installed { row.states["bf16"] = .init(state: .ready) }
        let metadata: JSONObject = [
            "languages": .array(languages.map(JSONValue.string)), "suggested": .bool(suggested),
            "popularity": ["hf_downloads": .number(downloads)], "derive": ["int8": "mlx-affine-g64"], "weights": ["dtype": "bf16"]
        ]
        return CatalogEntry(row: row, metadata: metadata, sizesByDtype: ["bf16": bytes, "int8": Int64(Double(bytes) * 8.5 / 16)], added: added)
    }
    static let entries: [CatalogEntry] = [
        entry("von", "Von 1.2", kind: "decisions", params: "395M", bytes: 1_600_000_000, downloads: 50, released: 2025),
        entry(
            "qwen", "Qwen3.8 27B", kind: "chat", params: "27B", bytes: 55_600_000_000, languages: ["en", "zh", "pl", "de"], suggested: true, downloads: 900, ram: 58_000,
            added: true, released: 2026),
        entry("kokoro", "Kokoro 82M", kind: "voices", params: "82M", bytes: 342_000_000, suggested: true, downloads: 5000, ram: 800, released: 2024, verification: "quick check"),
        entry(
            "breeze", "Breeze TTS 2", kind: "voices", params: "3B", bytes: 1_900_000_000, licence: "LicenseRef-Breeze", commercial: false, languages: ["en", "zh"], downloads: 10,
            ram: 3000, installed: true),
        entry("giant", "Giant 400B", kind: "chat", params: "400B", bytes: 800_000_000_000, ram: 800_000)
    ]
    private func controller() -> BrowserController {
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        c.apply(entries: Self.entries, kinds: Self.kinds)
        return c
    }
    private func ids(_ c: BrowserController) -> [String] { c.shown.map(\.id) }

    /// Arsenal recommends no model: no flag column, and the list opens most downloaded first.
    @Test func defaultIsMostDownloadedWithNoRecommendation() {
        let c = controller()
        #expect(c.state.sort == BrowserSort(key: .downloads, ascending: true))
        #expect(!BrowserColumn.allCases.map(\.rawValue).contains("flag"))
        #expect(ids(c) == ["kokoro", "qwen", "von", "breeze", "giant"])
        #expect(c.count == "5 models \u{00b7} 1 added")
        #expect(c.summary == "All kinds \u{00b7} Most downloaded")
    }

    @Test func sortKeysHeaderClicksAndThePopUpShareOneState() {
        var s = BrowserSort.default
        s.click(.download); #expect(s == BrowserSort(key: .download, ascending: true))
        s.click(.download); #expect(s == BrowserSort(key: .download, ascending: false))
        s.click(.ram); #expect(s == BrowserSort(key: .ram, ascending: true))
        #expect(SortKey.of(.model) == .name && SortKey.downloads.column == nil && SortKey.of(.chevron) == nil)
        let c = controller()
        c.state.sort = BrowserSort(key: .downloads); #expect(ids(c).first == "kokoro")
        c.state.sort = BrowserSort(key: .newest); #expect(ids(c).prefix(2) == ["qwen", "von"])
        // Status ranks the states that change what you do; none of these has one (a quick check is not a status).
        c.state.sort = BrowserSort(key: .status); #expect(Set(ids(c)) == Set(c.items.map(\.id)) && c.items.allSatisfy { $0.status == .none })
        c.state.sort = BrowserSort(key: .licence); #expect(ids(c).last == "breeze")
        c.state.choose(kind: "chat"); c.state.sort.click(.params)
        #expect(ids(c) == ["qwen", "giant"])
        #expect(c.count == "2 of 5 models \u{00b7} 1 added")
        #expect(c.summary == "Chat \u{00b7} Params")
    }

    @Test func modalityAndFacetsAreOrInsideAndAcross() {
        let c = controller()
        c.state.toggle(.params, "< 1B"); c.state.toggle(.params, "1\u{2013}4B")
        #expect(Set(ids(c)) == ["von", "kokoro", "breeze"])
        c.state.toggle(.licence, BrowserFacet.commercialValue)
        #expect(Set(ids(c)) == ["von", "kokoro"])
        c.state.clearFacet(.licence)
        c.state.toggle(output: .audio)
        #expect(Set(ids(c)) == ["kokoro", "breeze"]) // the voices kind's facets
        c.state.toggle(.library, BrowserFacet.onThisMac)
        #expect(ids(c) == ["breeze"])
        #expect(c.summary.hasPrefix("All kinds \u{00b7} Audio out"))
        #expect(c.available(.audio, input: false) && !c.available(.video, input: true))
        c.state.clearFilters()
        #expect(ids(c).count == 5 && !c.state.filtering)
        let licences = BrowserFacet.licence.values(c.items)
        #expect(licences.first == BrowserFacet.commercialValue && licences.last == BrowserFacet.nonCommercialValue && !licences.contains("Breeze"))
        c.state.toggle(.licence, BrowserFacet.nonCommercialValue)
        #expect(ids(c) == ["breeze"])
    }

    @Test func fitsAndSearch() {
        let c = controller()
        c.state.fits = true
        #expect(!ids(c).contains("giant"))
        let giant = c.items.first { $0.id == "giant" }!
        #expect(giant.ramHelp(physicalBytes: 128 * 1_073_741_824)?.hasSuffix("Needs about 746 GB; Arsenal keeps this 128 GB Mac under 100 GB.") == true)
        c.state = BrowserState(); c.state.search = "chinese"
        #expect(Set(ids(c)) == ["qwen", "breeze"])
        c.state.search = "klingon"
        #expect(ids(c).isEmpty && c.noResults == "No models match")
        c.state.choose(kind: "voices")
        #expect(c.noResults == "No speech match")
    }

    @Test func cellsReadAsSpecified() {
        let c = controller()
        let qwen = c.items.first { $0.id == "qwen" }!, breeze = c.items.first { $0.id == "breeze" }!, kokoro = c.items.first { $0.id == "kokoro" }!
        // Languages in words, the kind by its group's name, the licence by name with its orange explained.
        #expect(qwen.languagesText == "4 languages" && qwen.kindTitle == "Chat" && qwen.dtypeText == "bf16")
        #expect(breeze.downloadCheck && breeze.downloadHelp == "On this Mac")
        #expect(breeze.licenceText == "Breeze" && !breeze.commercial && breeze.licenceHelp == "Orange: commercial use is limited.\nBreeze\nNon-commercial use only.")
        #expect(kokoro.accessibilityLabel == "Kokoro 82M, Speech, 82M, 342 MB")
        #expect(kokoro.help.contains("Quick check: "))
    }

    /// The table holds exactly the models with weights on this Mac or a download under way: Add downloads, Remove deletes
    /// the weights after one confirmation, and a failed Get (no weights) is removed without asking.
    @Test func addDownloadsAndRemoveDeletesInPlaceThroughTheService() async throws {
        let service = FixtureService(rows: [], catalog: Self.entries.map(\.row))
        await service.stub.setGetScript { id, _, _ in
            if id == "kokoro" { throw WorkerFailure(code: "network", message: "The connection stalled.") }
        }
        let c = BrowserController(service: service)
        await c.refresh()
        c.state.sort = BrowserSort(key: .name)
        let before = ids(c)
        // Breeze's weights are on this Mac, so it is on the table already.
        #expect(c.addedCount == 1)
        c.add("von")
        #expect(c.shown.first { $0.id == "von" }?.added == true)
        try await waitUntil {
            let row = await service.rows().first { $0.id == "von" }
            return row?.state == .ready
        }
        #expect(await service.stub.getCalls.map { $0.hasPrefix("von ") } == [true])
        await c.refresh()
        #expect(ids(c) == before && c.addedCount == 2)
        // With weights: one confirmation, then Remove deletes them and the model leaves the table.
        var asked = 0
        c.confirmRemove = { _ in
            asked += 1; return .deleteWeights
        }
        c.remove("von")
        #expect(c.shown.first { $0.id == "von" }?.added == false)
        try await waitUntil { await !service.rows().contains { $0.id == "von" } }
        #expect(asked == 1 && ids(c) == before)
        #expect(await service.calls.contains("remove von delete"))
        // A failed Get left no weights: its row stays on the table until Remove, which takes it off at once, no alert.
        c.add("kokoro")
        try await waitUntil { c.lastError != nil }
        #expect(c.lastError?.hasPrefix("Kokoro 82M: ") == true)
        try await waitUntil { await service.rows().first { $0.id == "kokoro" }?.state == .failed }
        await c.refresh()
        #expect(c.shown.first { $0.id == "kokoro" }?.added == true)
        c.remove("kokoro")
        #expect(c.shown.first { $0.id == "kokoro" }?.added == false)
        try await waitUntil { await service.calls.contains("remove kokoro keep") }
        #expect(asked == 1)
    }

    @Test func removeWithWeightsAsksAndCancelKeepsTheRow() async throws {
        var row = Self.entries[3].row; row.states["bf16"] = .init(state: .ready)
        let service = FixtureService(rows: [row])
        let c = BrowserController(service: service)
        await c.refresh()
        var asked: [String] = []
        c.confirmRemove = { r in
            asked.append(r.id); return .cancel
        }
        c.remove("breeze")
        #expect(asked == ["breeze"] && c.items.first?.added == true)
        c.confirmRemove = { _ in .deleteWeights }
        c.remove("breeze")
        try await waitUntil { await service.calls.contains("remove breeze delete") }
    }

    @Test func removeGoesThroughTheTablesDeleteWhenWired() async throws {
        var row = Self.entries[3].row; row.states["bf16"] = .init(state: .ready)
        let service = FixtureService(rows: [row])
        let c = BrowserController(service: service)
        await c.refresh()
        var routed: [String] = [], asked = 0
        c.removeModel = { routed.append($0.id) }
        c.confirmRemove = { _ in
            asked += 1; return .deleteWeights
        }
        c.remove("breeze")
        // The table's trash reads the files on disk and asks with every precision's size; the browser's own alert is not used.
        #expect(routed == ["breeze"] && asked == 0)
        #expect(await !service.calls.contains("remove breeze delete"))
    }

    @Test func peakMemoryFromAnotherChipStillShowsAsMeasured() {
        let r = ModelRow(id: "k", name: "K", kind: "local.audio", badge: .otherChip, metrics: ["peak_ram": "0.4 GB"])
        let f = TableFigures.local(r, precision: "fp16", column: .memory)
        #expect(f.value == "0.4 GB")
    }

    @Test func expansionTogglesBySessionAndDimsOnlyVisibleOnes() {
        let c = controller()
        c.toggleExpanded("kokoro"); c.toggleExpanded("qwen")
        #expect(c.visibleExpanded == ["kokoro", "qwen"])
        c.state.choose(kind: "voices")
        #expect(c.visibleExpanded == ["kokoro"] && c.expanded.contains("qwen"))
        c.toggleExpanded("kokoro")
        #expect(c.visibleExpanded.isEmpty)
    }

    // MARK: Column geometry

    @Test func columnsFillProportionallyClampAndHideInOrder() {
        let wide = BrowserColumns.solve(width: 1203)
        #expect(wide.hidden.isEmpty)
        for c in BrowserColumns.elastic { let s = BrowserColumns.specs[c]!; #expect(wide.widths[c]! >= s.min - 0.01 && wide.widths[c]! <= s.max + 0.01) }
        #expect(wide.widths[.dtype]! >= 118)
        let used = BrowserColumns.elastic.map { wide.widths[$0]! }.reduce(0, +)
        #expect(abs(used + wide.flex - wide.elasticWidth) < 1)
        // Peak memory and Precision (bolt and the loaded format's gold) need more room than RAM and dtype did.
        let narrow = BrowserColumns.solve(width: 980)
        #expect(narrow.hidden == [.kind, .languages])
        let narrower = BrowserColumns.solve(width: 760)
        #expect(narrower.hidden.isSuperset(of: [.kind, .languages, .dtype]))
        let huge = BrowserColumns.solve(width: 3000)
        #expect(huge.flex > 0 && huge.widths[.model] == 420)
    }

    @Test func aUserResizeKeepsItsShareAndTheOthersRatios() {
        // Wide enough that Licence can reach 200 with every other column at or above its minimum.
        let layout = BrowserColumns.solve(width: 1300)
        let w = BrowserColumns.resized(.licence, to: 200, in: layout, weights: BrowserColumns.defaultWeights)
        #expect(abs(BrowserColumns.elastic.map { w[$0]! }.reduce(0, +) - 1) < 0.0001)
        let after = BrowserColumns.solve(width: 1300, weights: w)
        #expect(abs(after.widths[.licence]! - 200) < 1.5)
        // Ratio of two untouched columns is unchanged.
        #expect(abs(w[.model]! / w[.status]! - BrowserColumns.specs[.model]!.weight / BrowserColumns.specs[.status]!.weight) < 0.0001)
        let defaults = UserDefaults(suiteName: "arsenal-tests-\(UUID().uuidString)")!
        BrowserColumns.save(w, defaults)
        #expect(abs(BrowserColumns.savedWeights(defaults)[.licence]! - w[.licence]!) < 0.0001)
    }

    // MARK: Expanded details

    private static func kokoroEntry(runners: [CatalogRunner], resident: String? = nil, ready: Bool = false) -> CatalogEntry {
        var row = ModelRow(
            id: "kokoro-82m", name: "Kokoro 82M", kind: "voices", params: "82M", licence: "Apache-2.0", precisions: ["fp16", "int8"], selectedPrecision: "fp16",
            runner: runners.first(where: \.selected)?.name ?? "native", downloadBytes: ["fp16": 342_000_000, "int8": 342_000_000], publisher: "Hexgrad", released: 2025)
        row.defaultRunner = "native"; row.residentRunner = resident
        if ready { row.states["fp16"] = .init(state: resident == nil ? .ready : .loaded) }
        let metadata: JSONObject = [
            "derive": ["fp16": "cast:kokoro", "int8": "mlx-affine-g64:kokoro-82m"], "weights": ["dtype": "fp32", "repo": "mlx-community/Kokoro-82M-bf16"],
            "languages": ["en"], "licence": ["url": "https://example.com/l"]
        ]
        var e = CatalogEntry(row: row, metadata: metadata, sizesByDtype: ["fp16": 171_000_000, "int8": 91_000_000], added: true)
        e.runners = runners
        return e
    }
    private static let result: JSONObject = {
        let r = ResultsFile(
            model: "kokoro-82m", kind: "voices", chip: "m5-max", osBuild: "x", appBuild: "x", runner: "native", workerDigest: String(repeating: "a", count: 64),
            precision: "fp16", mode: "fast", suite: .init(id: "s", sha256: String(repeating: "b", count: 64)), profile: "quick", stub: false, gatePassed: false,
            conditions: .init(locked: false, idle: [], run: [], note: "Quick check under gpulock"),
            metrics: ["speed_x_realtime": 46.83, "first_audio_s": 0.088, "peak_ram_mb": 824], cases: [])
        // swiftlint:disable:next force_try
        var o = try! WireJSON.decode(JSONEncoder().encode(r)); o["date"] = "2026-10-02T09:10:19Z"; return o
    }()

    @Test func sizesNeverListTheSourceDtypeAndNameTheDownload() {
        let e = Self.kokoroEntry(runners: [])
        let sizes = ModelDetail.sizes(e)
        // 8-bit is not offered: the table of sizes lists the 16-bit copy only.
        #expect(sizes.map(\.dtype) == ["fp16"])
        #expect(sizes[0] == .init(dtype: "fp16", size: "~171 MB", note: "from a 342 MB download", label: "16-bit"))
        // fp16 here: the one listed copy is on this Mac.
        var here = e; here.row.states["fp16"] = .init(state: .ready)
        #expect(ModelDetail.sizes(here).map(\.note) == ["on this Mac"])
        #expect(!ModelDetail.sizes(e).contains { $0.dtype == "int8" })
        // A published 16-bit source is listed as itself.
        let qwen = ModelDetail.sizes(Self.entries[1])
        #expect(qwen.map(\.dtype) == ["bf16"] && qwen.first == .init(dtype: "bf16", size: "55.6 GB", note: "download", label: "16-bit"))
    }

    /// Core review 8: Get more models' Add and Remove are buttons to accessibility, named with their model. (The tool
    /// windows' SwiftUI slider and model pop-up are not reachable offscreen; their fixes are checked by hand.)
    @Test func addAndRemoveAreAccessibleButtons() {
        var e = Self.kokoroEntry(runners: []); e.added = true
        let cell = ActionCell()
        cell.configure(item: BrowserItem(e, kind: Self.kinds[0]), palette: .dark) {}
        #expect(cell.button.isAccessibilityElement() && cell.button.accessibilityRole() == .button)
        #expect(cell.button.accessibilityLabel() == "Remove \(e.name)" && cell.button.accessibilityTitle() == "Remove")
        #expect(cell.accessibilityChildren()?.contains { ($0 as? NSObject) === cell.button } ?? cell.subviews.contains(cell.button))
    }

    /// Core review 7: a published 16-bit copy whose download is larger (an FP32 source cast to FP16 as it arrives: RIFE
    /// downloads 23 MB and keeps 11 MB) names both, in the sizes and in the download's hover.
    @Test func aDownloadLargerThanItsCopyNamesBoth() {
        var e = Self.entries[1]
        e.sizesByDtype["bf16"] = 11_000_000; e.row.downloadBytes["bf16"] = 23_000_000
        // A 16-bit source whose plan is larger (a separate encoder) is not a cast: one figure, as before.
        #expect(ModelDetail.sizes(e).first == .init(dtype: "bf16", size: "11 MB", note: "download", label: "16-bit"))
        #expect(BrowserItem(e, kind: Self.kinds[0]).downloadParts.first == "bf16 11 MB")
        e.sourceDtype = "fp32"
        #expect(ModelDetail.sizes(e).first == .init(dtype: "bf16", size: "11 MB", note: "from a 23 MB download", label: "16-bit"))
        #expect(BrowserItem(e, kind: Self.kinds[0]).downloadParts.first == "bf16 11 MB from a 23 MB download")
    }

    @Test func oneRunnerIsStaticTextWithItsRecordedNumbers() {
        let native = CatalogRunner(
            name: "native", language: "swift", engine: "voices", acceptedDefault: true, selected: true, results: [Self.result],
            history: RunnerHistory(path: "log/005.md", title: "005-v2-real-cycles", verdict: "kept"))
        let item = BrowserItem(Self.kokoroEntry(runners: [native]), kind: Self.kinds[0])
        let d = ModelDetail(item, chip: "M5 Max")
        #expect(!d.choosable && d.note == nil && d.runners.count == 1)
        let r = d.runners[0]
        #expect(r.engine == "Swift \u{00b7} MLX" && r.isDefault && r.figures.map(\.1) == ["46.8×", "88 ms", "0.82 GB"])
        // The intake history is not a line on the card (the user's polish list): the runner carries none.
        // One precision vocabulary: the condition line says "16-bit", as the Sizes line and the Models table do.
        #expect(r.condition == "16-bit \u{00b7} Fast \u{00b7} quick check \u{00b7} 2 Oct")
        #expect(!r.figuresHelp.contains("fp16") && r.figuresHelp.hasPrefix("16-bit Fast"))
        #expect(d.about.map(\.key) == ["Publisher", "Released", "Licence", "Languages"] && d.links.map(\.title) == ["Hugging Face", "Licence"])
        let unmeasured = ModelDetail.runner(
            CatalogRunner(name: "native", language: "swift", engine: "chat", acceptedDefault: true, selected: true), row: item.entry.row, columns: item.columns, chip: "M5 Max")
        #expect(unmeasured.condition == "Not measured on M5 Max yet" && unmeasured.figures.isEmpty)
    }

    @Test func nativeRunnerChangesRequireReloadAndKeepEligibilityReasons() {
        let runners = [
            CatalogRunner(name: "native", language: "swift", engine: "voices", acceptedDefault: true),
            CatalogRunner(name: "mlx-audio", language: "swift", engine: "voices", selected: true),
            CatalogRunner(name: "m1-only", language: "swift", engine: "voices", eligible: false, ineligibleReason: "Built for M1")
        ]
        let loaded = ModelDetail(BrowserItem(Self.kokoroEntry(runners: runners, resident: "native", ready: true), kind: Self.kinds[0]))
        #expect(loaded.choosable && loaded.runners[1].eligible)
        #expect(loaded.note == "Loaded with native until Reload")
        #expect(loaded.runners[2].reason == "Built for M1" && !loaded.runners[2].eligible)
    }

    @Test func choosingTheDefaultRunnerClearsTheOverride() async throws {
        let service = FixtureService(
            rows: [Self.kokoroEntry(runners: []).row],
            metadata: [
                "kokoro-82m": [
                    "runner": .array([
                        ["name": "native", "language": "swift"], ["name": "mlx-audio", "language": "swift"]
                    ])
                ]
            ])
        let c = BrowserController(service: service)
        await c.refresh()
        c.chooseRunner("kokoro-82m", "mlx-audio")
        try await waitUntil { await service.calls.contains("runner kokoro-82m mlx-audio") }
        c.chooseRunner("kokoro-82m", "native")
        try await waitUntil { await service.calls.contains("runner kokoro-82m default") }
    }
}
