// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import SwiftUI
import Testing

@testable import Arsenal

/// The design review of the unified Models table (15 Oct, round 1): each finding fixed, held by a test.
@MainActor @Suite struct TableReviewRound2Tests {
    let rows = Fixtures.figureRows()
    func row(_ id: String) throws -> ModelRow { try #require(rows.first { $0.id == id }) }

    /// 1. A pending Reload keeps the loaded row: no red, the chosen precision's gold on the status line and the button,
    /// the loaded precision ringed.
    @Test func pendingReloadKeepsTheLoadedRow() throws {
        // Loaded at fp16; the int4 cell chosen (int8 has no cell).
        var kokoro = try row("kokoro-82m")
        kokoro.selectedPrecision = "int4"
        let p = RowPresentation(kokoro)
        #expect(p.pending && p.hot && p.action == .reload)
        #expect(p.status == StatusLine(text: "4-bit on next load", tone: .pending, help: "16-bit stays loaded until Reload."))
        #expect(ModelTable.tone(.pending, hot: true, ArsenalPalette.menu(dark: true)) == ArsenalPalette.gold.swiftUI)
    }

    /// 2. A folder kind's figures carry its own words and the service's attribution, never the host chip by assumption
    /// or another kind's unit text; its heading names its declared unit.
    @Test func folderKindFiguresAreAttributedLikeTheRest() throws {
        let video = try row("my-video-5b")
        let speed = TableFigures.figure(video, precision: "bf16", column: .speed, host: "M3 Pro")
        #expect(speed.value == "38 s")
        #expect(!speed.help.contains("speech") && !speed.help.contains("M3 Pro") && speed.help.contains("720p, 5 s clip"))
        var elsewhere = video; elsewhere.badge = .otherChip
        let other = TableFigures.figure(elsewhere, precision: "bf16", column: .speed, host: "M3 Pro")
        #expect(other.reference == "Other chip" && other.help.contains("Not measured on your Mac"))
        #expect(TableFigures.figure(elsewhere, precision: "bf16", column: .memory, host: "M3 Pro").reference == nil)
        let heading = TableStats.help(kind: Fixtures.videoKind, column: FigureColumn.speed.rawValue)
        #expect(heading.hasPrefix("Seconds per run") && heading.hasSuffix("Lower is faster."))
        #expect(ColumnHelp.known["speed"] == nil)
    }

    /// 3. The status hover never contradicts the figures, and the Local line is said once.
    @Test func statusHoverSplitsQualityFromFigures() throws {
        // Loaded and chosen at its 16-bit copy (the int8 copy is no longer offered).
        var laya = try row("laya-multilingual")
        laya.states["int8"] = PrecisionState(state: .ready)
        laya.states["fp16"] = PrecisionState(state: .loaded)
        laya.selectedPrecision = "fp16"
        let status = RowPresentation(laya).status
        #expect(!status.help.contains("measured figures") && status.help.contains("fp16 has no quality check yet.") && status.help.contains("int4 has no quality check yet."))
        #expect(RowPresentation(try row("stable-audio-3-medium")).status.help.hasSuffix("No figures yet."))
        let local = RowPresentation(try row("my-chat-4b")).status
        #expect(local.text == "Local" && !local.help.hasPrefix("Local"))
    }

    /// 4. On the light menu's solid blue the delta's words are white and an arrow carries the tone.
    @Test func lightLoadedRowDeltasAreWhiteWithAnArrow() {
        #expect(ModelTable.deltaArrow(hot: true, ArsenalPalette.menu(dark: false)))
        #expect(!ModelTable.deltaArrow(hot: true, ArsenalPalette.menu(dark: true)))
        #expect(!ModelTable.deltaArrow(hot: false, ArsenalPalette.menu(dark: false)))
    }

    /// 5, 6. Get more models: the loaded state (flame, blue, gold format, bolt), the table's names and one source
    /// for peak memory (the triangle where the table has none).
    @Test func getMoreModelsUsesTheTablesStateNamesAndFigures() throws {
        let laya = BrowserItem(CatalogEntry(row: try row("laya-multilingual"), metadata: [:], added: true), kind: nil)
        // Still resident at int8, so it stays lit as loaded; its figure is the 16-bit copy's, the copy offered in place of
        // the 8-bit one.
        #expect(laya.loadedDtype == "int8" && laya.fastKernels && laya.ramText == "1.18 GB")
        let reuse = BrowserItem(CatalogEntry(row: try row("re-use"), metadata: [:], added: false), kind: nil)
        #expect(reuse.loadedDtype == nil && reuse.peak == nil && reuse.ramText.isEmpty)
        #expect(BrowserColumn.ram.title == "Peak memory" && BrowserColumn.dtype.title == "Precision")
        #expect(SortKey.ram.title == "Peak memory" && BrowserFacet.dtype.name == "Precision" && BrowserFacet.ram.name == "Peak memory")
        // The saved identifiers stay.
        #expect(BrowserColumn.ram.rawValue == "RAM" && SortKey.dtype.rawValue == "dtype")
    }

    /// 11. The runner card shows the precision as a format (gold when loaded), apart from the runner's "default".
    @Test func runnerCardMarksTheLoadedFormat() throws {
        let item = BrowserItem(CatalogEntry(row: try row("laya-multilingual"), metadata: [:], added: true), kind: nil)
        let runner = try #require(ModelDetail(item, chip: "M5 Max").runners.first)
        #expect(runner.precision == "int8" && runner.precisionLoaded)
        let kokoro = BrowserItem(CatalogEntry(row: try row("kokoro-82m"), metadata: [:], added: true), kind: nil)
        #expect(ModelDetail(kokoro, chip: "M5 Max").runners.first?.precisionLoaded == false)
    }

    /// 8. The Precision heading is centred on the cells, not on the bolt slot.
    @Test func precisionHeadingCentresOnTheCells() {
        // Measured on the laid-out table in TableVellaTests.precisionHeadingSitsOverTheCells.
        let parts = PrecisionLines.boltWidth + PrecisionLines.boltGap + PrecisionLines.pillWidth + PrecisionLines.trashGap + PrecisionLines.trashWidth
        #expect(abs(parts - ModelTable.W.precision) < 0.5)
    }

    /// 12. Refresh has a tooltip.
    @Test func refreshHasATooltip() {
        #expect(!RefreshControl.help.isEmpty)
        let host = NSHostingView(rootView: RefreshControl(enabled: true, click: {}).frame(width: 20, height: 20))
        host.frame = NSRect(x: 0, y: 0, width: 20, height: 20); host.layoutSubtreeIfNeeded()
        func find(_ v: NSView) -> RefreshControlView? { (v as? RefreshControlView) ?? v.subviews.lazy.compactMap(find).first }
        #expect(find(host)?.toolTip == RefreshControl.help)
    }
}
