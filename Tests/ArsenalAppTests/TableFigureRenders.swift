// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// Review renders for the unified Models table (the same columns for every kind, Vella's colours, a yellow
/// triangle for a missing figure, another chip's figure as its reference): the table at real width, 2× pixels, dark and
/// light, first frame. Scenes: on this Mac (M5 Max) and on another Mac (M3 Pro), every kind, loaded rows, a pending
/// Reload, disabled tiers, int4 quality triangles (one on a chosen int4), missing figures, deltas (round 2: Vella's
/// precision control). Run with ARSENAL_TABLE_FIGURE_RENDERS=<dir>.
@MainActor @Suite(.menuState, .serialized) final class TableFigureRenders {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-table-figures-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_TABLE_FIGURE_RENDERS"] != nil))
    func renderFigures() async throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_TABLE_FIGURE_RENDERS"]!)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { Appearance.shared.set(.dark); NSApp.appearance = nil }
        func controller(_ name: String, chip: String, rows: [ModelRow] = Fixtures.figureRows()) -> TableController {
            let c = TableController(service: FixtureService(rows: rows, supportDirectory: support.appendingPathComponent(name)), chip: chip)
            c.apply(rows: rows, kinds: Fixtures.figureKinds, catalog: CatalogStatus(source: "local", pinnedCommit: "f600b7d4c1e2", verified: true))
            return c
        }
        let here = controller("here", chip: "M5 Max")
        let other = controller("other", chip: "M3 Pro")
        // A table as a user has it: six models in five kinds.
        let realistic = controller("realistic", chip: "M5 Max", rows: Fixtures.realisticRows())
        var tips = ""
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            try await Task.sleep(for: .milliseconds(100))
            for (name, c) in [("m5max", here), ("m3pro", other)] {
                await QA.renderTable(c, to: directory.appendingPathComponent("figures-\(name)-\(mode.rawValue).png"))
            }
            await QA.renderTable(realistic, to: directory.appendingPathComponent("realistic-m5max-\(mode.rawValue).png"))
        }
        for (name, c) in [("m5max", here), ("m3pro", other)] { tips += "# \(name)\n\n" + RenderDelegate.tooltips(c) + "\n" }
        try tips.write(to: directory.appendingPathComponent("figures-tooltips.txt"), atomically: true, encoding: .utf8)
        try "table width \(ModelTable.width(here)) pt\n".write(to: directory.appendingPathComponent("figures-width.txt"), atomically: true, encoding: .utf8)
        try "table height: figures \(ModelTable.height(here)) pt (scrolls: \(ModelTable.scrolls(here))), realistic \(ModelTable.height(realistic)) pt\n"
            .write(to: directory.appendingPathComponent("figures-height.txt"), atomically: true, encoding: .utf8)
        here.stop(); other.stop(); realistic.stop()
        try await renderGetMoreModels(directory)
    }

    /// Get more models with the same rows, Laya and Kokoro expanded: the runner panel shows the table's two figures.
    private func renderGetMoreModels(_ directory: URL) async throws {
        let defaults = UserDefaults.standard
        let oldSidebar = defaults.object(forKey: BrowserRootView.openKey), oldFrame = defaults.object(forKey: "NSWindow Frame GetMoreModels")
        defer { defaults.set(oldSidebar, forKey: BrowserRootView.openKey); defaults.set(oldFrame, forKey: "NSWindow Frame GetMoreModels") }
        let rows = Fixtures.figureRows()
        let entries = rows.enumerated().map { index, row in
            let metadata: JSONObject = ["languages": .array([.string("en")]), "popularity": ["hf_downloads": .number(Double(1000 - index * 100))]]
            // Each format's weights: the 16-bit download, and int8/int4 at their bits per weight plus the scales.
            let full = Double(row.sizeBytes ?? 0)
            let sizes = Dictionary(
                uniqueKeysWithValues: row.precisions.map { d in
                    (d, Int64(full * ((PrecisionColumn.bits(d) ?? 16) >= 12 ? 1 : (PrecisionColumn.bits(d)! + 0.5) / 16)))
                })
            return CatalogEntry(row: row, metadata: metadata, sizesByDtype: sizes, added: !row.states.isEmpty)
        }
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            let browser = BrowserController(service: FixtureService(rows: rows, supportDirectory: support.appendingPathComponent("browser")), chip: "M5 Max")
            browser.apply(entries: entries, kinds: Fixtures.figureKinds)
            browser.expanded = ["laya-multilingual", "kokoro-82m"]
            let window = BrowserWindowController(controller: browser)
            guard let w = window.window else { continue }
            window.applyAppearance()
            w.setFrame(NSRect(x: -9000, y: -9000, width: 1300, height: 760), display: false)
            window.list.expansionChanged()
            w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
            try await Task.sleep(for: .milliseconds(700))
            if let rep = WindowSnapshot.capture(w) {
                try rep.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("getmore-\(mode.rawValue).png"))
            }
            w.orderOut(nil); w.close()
        }
    }
}
