// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

/// v2.1 review fixes (app): browser errors are visible, a busy row's Remove refuses with a reason, the sidebar's content
/// follows its width, and the settings window's action matches the row's (never a "Reload" that is a Get).
@MainActor @Suite(.menuState, .serialized) final class V21FixHTests {
    private static func entry(_ id: String, runners: [CatalogRunner] = [], state: ModelState? = nil) -> CatalogEntry {
        var row = ModelRow(
            id: id, name: "Model \(id)", kind: "voices", params: "82M", licence: "Apache-2.0", precisions: ["fp16", "int8"], selectedPrecision: "fp16",
            runner: "native", downloadBytes: ["fp16": 342_000_000, "int8": 342_000_000], publisher: "Hexgrad", released: 2025)
        row.defaultRunner = "native"
        if let state { row.states["fp16"] = .init(state: state) }
        var e = CatalogEntry(row: row, metadata: ["languages": ["en"], "weights": ["dtype": "fp16"], "derive": ["int8": "mlx-affine-g64"]], added: true)
        e.runners = runners
        return e
    }

    // MARK: Sidebar width (O-S3)

    private func sidebarWidth(at proposal: CGFloat) -> CGFloat {
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        c.apply(entries: BrowserTests.entries, kinds: BrowserTests.kinds)
        let host = NSHostingController(rootView: BrowserSidebar(controller: c, search: NSSearchField(), band: 28))
        return host.sizeThatFits(in: CGSize(width: proposal, height: 900)).width
    }
    @Test func theSidebarFillsEveryWidthItCanBeDraggedTo() {
        for w: CGFloat in [208, 236, 300] { #expect(abs(sidebarWidth(at: w) - w) < 0.5, "sidebar at \(w)") }
    }

    /// Found while rendering: a layout pass before the window has a size (width 0) took itself for a narrowing and
    /// saved the sidebar closed.
    @Test func aLayoutBeforeTheWindowHasASizeKeepsTheSidebarOpen() {
        let defaults = UserDefaults.standard, saved = defaults.object(forKey: BrowserRootView.openKey)
        defer { if let saved { defaults.set(saved, forKey: BrowserRootView.openKey) } else { defaults.removeObject(forKey: BrowserRootView.openKey) } }
        defaults.set(true, forKey: BrowserRootView.openKey)
        let c = BrowserController(service: FixtureService(rows: [])), list = BrowserList()
        list.controller = c
        let root = BrowserRootView(controller: c, list: list, sidebarSearch: NSSearchField(), toolbarSearch: NSSearchField())
        root.frame = .zero; root.layout()
        #expect(root.sidebarOpen && defaults.bool(forKey: BrowserRootView.openKey))
        // A real narrowing below sidebar + 700 still collapses it.
        root.frame = NSRect(x: 0, y: 0, width: 800, height: 600); root.layout()
        #expect(!root.sidebarOpen)
    }

    // MARK: Busy Remove (O-S2, UI half) and visible errors (A-S4, O-S1)

    @Test func removeOfABusyRowRefusesWithItsReasonAndKeepsTheRow() async throws {
        for busy in [ModelState.downloading, .deriving, .loading] {
            let service = FixtureService(rows: [Self.entry("k", state: busy).row])
            let c = BrowserController(service: service)
            await c.refresh()
            var asked = 0
            c.confirmRemove = { _ in
                asked += 1; return .deleteWeights
            }
            c.remove("k")
            #expect(asked == 0 && c.items.first?.added == true && c.shown.first?.added == true)
            #expect(c.lastError?.hasPrefix("Model k: ") == true && c.footer.error)
            try await Task.sleep(nanoseconds: 50_000_000)
            #expect(await !service.calls.contains { $0.hasPrefix("remove") })
        }
    }

    @Test func aRefusedRunnerChoiceIsShownInTheCountLineUntilTheNextAction() async throws {
        let service = FixtureService(rows: [Self.entry("k").row], metadata: ["k": ["runner": .array([["name": "native", "language": "swift"]])]])
        let c = BrowserController(service: service)
        await c.refresh()
        #expect(!c.footer.error && c.footer.text == c.count)
        c.chooseRunner("k", "nowhere")
        try await waitUntil { c.lastError != nil }
        #expect(c.lastError == "Model k: That runner does not support this Mac.")
        #expect(c.footer == .init(text: "Model k: That runner does not support this Mac.", error: true))
        // The toolbar (sidebar collapsed) shows it in place of the filter summary.
        #expect(c.toolbarSummary == c.footer)
        // The next action clears it; dismissing does too.
        c.chooseRunner("k", "native")
        #expect(c.lastError == nil)
        c.lastError = "x"; c.dismissError()
        #expect(c.lastError == nil && c.toolbarSummary.text == c.summary)
    }

    // MARK: The settings window's action is the row's action (A-S5)

    private func row(state: ModelState, download: Int64, resident: String?, runner: String) -> ModelRow {
        var r = ModelRow(
            id: "k", name: "Kokoro 82M", kind: "voices", precisions: ["fp16", "int8"], selectedPrecision: "fp16", runner: runner,
            downloadBytes: ["fp16": download], runnerLanguage: "swift")
        r.states["fp16"] = .init(state: state); r.defaultRunner = "native"; r.residentRunner = resident
        if resident != nil, resident != runner { r.needsReload = true }
        return r
    }
    @Test func theSettingsFooterNamesTheSameActionAsTheRow() {
        // Missing native weights: the row and settings window both show the same Get size.
        let get = row(state: .notInstalled, download: 256_000_000, resident: "native", runner: "mlx-audio")
        let p = RowPresentation(get)
        let f = SettingsPane.footerAction(get, presentation: p, changes: ["runner mlx-audio"])
        #expect(p.action == .get && f.title == "Get \u{00b7} 256 MB" && f.title == p.actionTitle)
        #expect(f.help == ModelHelp.action(get, presentation: p, changes: ["runner mlx-audio"]) && !f.help.contains("Reload applies"))
        // Weights ready: a changed native runner reloads with the pending choices.
        let reload = row(state: .loaded, download: 0, resident: "native", runner: "mlx-audio")
        let r = SettingsPane.footerAction(reload, presentation: RowPresentation(reload), changes: ["runner mlx-audio"])
        #expect(r.title == "Reload" && r.help == ModelHelp.reloadHelp(["runner mlx-audio"]))
    }

    // MARK: Live resize with expanded rows (O-note 8)

    private static let result: JSONObject = {
        let r = ResultsFile(
            model: "k", kind: "voices", chip: "m5-max", osBuild: "x", appBuild: "x", runner: "native", workerDigest: String(repeating: "a", count: 64),
            precision: "fp16", mode: "fast", suite: .init(id: "s", sha256: String(repeating: "b", count: 64)), profile: "quick", stub: false, gatePassed: false,
            conditions: .init(locked: false, idle: [], run: [], note: "Quick check under gpulock"),
            metrics: ["speed_x_realtime": 46.83, "first_audio_s": 0.088, "peak_ram_mb": 824], cases: [])
        // swiftlint:disable:next force_try
        var o = try! WireJSON.decode(JSONEncoder().encode(r)); o["date"] = "2026-10-02T09:10:19Z"; return o
    }()
    private func expandedList(_ n: Int) -> (BrowserController, BrowserList) {
        let runners = [
            CatalogRunner(name: "native", language: "swift", engine: "voices", acceptedDefault: true, selected: true, results: [Self.result, Self.result]),
            CatalogRunner(name: "mlx-audio", language: "swift", engine: "voices", results: [Self.result])
        ]
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        c.apply(
            entries: (0..<n).map { i in
                var e = Self.entry("m\(i)", runners: runners)
                e.description = String(repeating: "A small voice model that reads English text aloud. ", count: 1 + i % 4)
                return e
            }, kinds: BrowserTests.kinds)
        let list = BrowserList(); list.controller = c
        list.show(c.shown)
        list.toggleAll()
        return (c, list)
    }
    private func resize(_ list: BrowserList, _ w: CGFloat) {
        list.scrollView.frame = NSRect(x: 0, y: 0, width: w, height: 800); list.table.frame.size.width = w
        list.applyLayout()
        for i in 0..<list.lines.count { _ = list.tableView(list.table, heightOfRow: i) }
    }

    @Test func detailHeightsMatchAFreshMeasureAcrossArrangements() {
        let (c, list) = expandedList(4)
        for w: CGFloat in [1400, 1000, 700, 1400] {
            resize(list, w)
            let width = max(300, list.table.bounds.width - list.detailLeading - 24)
            for item in c.shown {
                let view = ModelDetailView(detail: ModelDetail(item, chip: "M5 Max"), palette: Appearance.shared.palette, width: width, licenceAmber: false) { _ in }
                let fresh = ceil(NSHostingController(rootView: view).sizeThatFits(in: CGSize(width: width, height: 4000)).height) + 6
                #expect(list.detailHeight(item.id) == fresh, "\(item.id) at \(w)")
            }
        }
        // About/Sizes stack at 1000, but the runner list may determine the same total height.
        resize(list, 1400); let wide = list.detailHeight("m0")
        resize(list, 1000); #expect(list.detailHeight("m0") >= wide)
        #expect(ModelDetailView.stacked(max(300, list.table.bounds.width - list.detailLeading - 24)))
    }

    @Test func aDetailCellFollowsTheListWidthByConstraint() throws {
        let (_, list) = expandedList(1)
        resize(list, 1400)
        let cell = try #require(list.tableView(list.table, viewFor: nil, row: 1))
        for w: CGFloat in [1400, 900] {
            cell.frame = NSRect(x: 0, y: 0, width: w, height: 200); cell.layoutSubtreeIfNeeded()
            #expect(abs(cell.subviews[0].frame.width - (w - list.detailLeading - 24)) < 0.5)
        }
    }

    @Test func liveResizeWithFifteenExpandedRowsReMeasuresOnlyAtArrangementChanges() {
        let (_, list) = expandedList(15)
        resize(list, 1400)
        let clock = ContinuousClock()
        var frames: [Duration] = []
        // Within one arrangement (side by side, 1100–1400 pt): no SwiftUI measuring at all.
        for step in 0..<40 { frames.append(clock.measure { resize(list, CGFloat(1400 - step * 7)) }) }
        frames.sort()
        #expect(frames[20] < .milliseconds(5), "median \(frames[20])")
    }
}
