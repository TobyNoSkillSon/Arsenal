// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// Review renders for the calmer Models table: the table at real width, 2x pixels, dark and light, on a stand-in for the
/// menu material and over a busy blurred wallpaper. States: loaded with a red pending row; a model loading (Refresh off);
/// a refused Refresh in the footer; a pending load setting; the empty table.
/// Run with ARSENAL_TABLE_DESIGN_RENDERS=<dir>.
@MainActor @Suite(.menuState, .serialized) final class TableDesignRenders {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-table-design-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_TABLE_DESIGN_RENDERS"] != nil))
    func renderTableDesign() async throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_TABLE_DESIGN_RENDERS"]!)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { Appearance.shared.set(.dark); NSApp.appearance = nil }

        let qwenID = "qwen3.8-27b"
        func controller(_ rows: [ModelRow], _ name: String, metadata: [String: JSONObject] = [:]) async -> TableController {
            let c = TableController(
                service: FixtureService(rows: rows, metadata: metadata, supportDirectory: support.appendingPathComponent(name)), chip: "M5 Max")
            await c.refresh()
            return c
        }
        // Loaded at int4, its 16-bit cell clicked (the red pending row).
        let loaded = await controller(Fixtures.rows(.loaded), "loaded")
        if let qwen = loaded.rows.first(where: { $0.id == qwenID }) { loaded.select(qwen, column: "16") }
        // A model loading: Refresh is off with its reason.
        var loadingRows = Fixtures.rows(.loaded)
        if let i = loadingRows.firstIndex(where: { $0.id == "flux2-klein-4b" }) { loadingRows[i].states["bf16"] = PrecisionState(state: .loading) }
        let loading = await controller(loadingRows, "loading")
        // A refused Refresh in the footer, while the Get that caused it still runs (FLUX downloading): Refresh is off too.
        var gettingRows = Fixtures.rows(.loaded)
        if let i = gettingRows.firstIndex(where: { $0.id == "flux2-klein-4b" }) {
            gettingRows[i].states["bf16"] = PrecisionState(state: .downloading, progress: 0.41, bytes: 6_550_000_000, totalBytes: 15_980_140_843)
        }
        let refused = await controller(gettingRows, "refused")
        refused.refreshNotice = "Finish Get or Load before refreshing models."
        // A loaded chat model with folder settings, one load setting changed (pending, Reload).
        let settings = await controller(Fixtures.rows(.loaded), "settings", metadata: [qwenID: PolishTableTests.settings])
        if let qwen = settings.rows.first(where: { $0.id == qwenID }) {
            settings.set(qwen, "context", .number(131_072))
            try await waitUntil { settings.settings[qwenID]?.first { $0.id == "context" }?.value.number == 131_072 }
            await settings.refresh()
        }
        let empty = await controller([], "empty")
        // Empty, with a problem in the footer (the table widens for the line).
        let emptyError = await controller([], "empty-error")
        emptyError.lastError = "Catalog: signature did not verify"

        let shots: [(String, TableController)] = [
            ("loaded", loaded), ("loading", loading), ("refused", refused), ("settings", settings), ("empty", empty), ("empty-error", emptyError)
        ]
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            try await Task.sleep(for: .milliseconds(100))
            for (name, c) in shots {
                for wallpaper in [false, true] {
                    let suffix = mode.rawValue + (wallpaper ? "-wallpaper" : "")
                    await QA.renderTable(c, to: directory.appendingPathComponent("table-\(name)-\(suffix).png"), wallpaper: wallpaper)
                }
            }
        }
        // Every tooltip the table shows, as text (a PNG cannot hover).
        let tips = [("loaded", loaded), ("loading", loading)].map { "## \($0.0)\n\n" + RenderDelegate.tooltips($0.1) }.joined(separator: "\n")
        try tips.write(to: directory.appendingPathComponent("table-tooltips.txt"), atomically: true, encoding: .utf8)
        for c in [loaded, loading, refused, settings, empty, emptyError] { c.stop() }
        let widths = "table width \(ModelTable.width(loaded)) pt; empty \(ModelTable.width(empty)) pt\n"
        try widths.write(to: directory.appendingPathComponent("table-width.txt"), atomically: true, encoding: .utf8)
    }
}
