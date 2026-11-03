// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import Testing

@testable import Arsenal

/// "we should have like information right under it … so that it's completely clear what model does …
/// right now we have like a quick check, which doesn't tell us anything". The line under a model's name in the Models
/// table and Get more models says what it does; a state that changes what the row can do keeps the line while it
/// applies; the verification it replaced stays in that line's hover.
@MainActor struct PurposeLineTests {
    private func row(_ change: (inout ModelRow) -> Void = { _ in }) -> ModelRow {
        var r = ModelRow(id: "flux2-klein-4b", name: "FLUX.2 Klein 4B", kind: "images", precisions: ["bf16", "int8", "int4"], selectedPrecision: "bf16")
        for p in r.precisions { r.states[p] = PrecisionState(state: .ready) }
        r.purpose = "Image generation"
        change(&r)
        return r
    }

    @Test func atRestTheLineSaysWhatTheModelDoesAndItsHoverKeepsTheVerification() {
        let unverified = RowPresentation(row(), chip: "M5 Max")
        #expect(unverified.line.text == "Image generation" && unverified.line.tone == .plain)
        #expect(unverified.status.text == "Unverified")
        #expect(unverified.line.help.hasPrefix("Unverified\n"))
        let quick = RowPresentation(row { $0.verification = "quick check" }, chip: "M5 Max")
        #expect(quick.line.text == "Image generation" && quick.line.help.hasPrefix("Quick check\n"))
        let local = RowPresentation(row { $0.local = true }, chip: "M5 Max")
        #expect(local.line.text == "Image generation" && local.line.help.hasPrefix("Local\n"))
        // Loaded at the chosen precision: still the purpose.
        let loaded = RowPresentation(row { $0.states["bf16"] = PrecisionState(state: .loaded) }, chip: "M5 Max")
        #expect(loaded.line.text == "Image generation")
        #expect(ModelTableRow.hasStatusLine(row(), unverified))
    }

    @Test func statesThatChangeWhatTheRowCanDoKeepTheLineWhileTheyApply() {
        let downloading = RowPresentation(row { $0.states["bf16"] = PrecisionState(state: .downloading, progress: 0.4) }, chip: nil)
        #expect(downloading.line.text.hasPrefix("Downloading"))
        let deriving = RowPresentation(
            row {
                $0.states["int8"] = PrecisionState(state: .deriving); $0.selectedPrecision = "int8"
            }, chip: nil)
        #expect(deriving.line.text == "Making 8-bit\u{2026}")
        let loading = RowPresentation(row { $0.states["bf16"] = PrecisionState(state: .loading) }, chip: nil)
        #expect(loading.line.text == "Loading 16-bit\u{2026}")
        let failed = RowPresentation(row { $0.states["bf16"] = PrecisionState(state: .failed, error: "Hash mismatch") }, chip: nil)
        #expect(failed.line.text == "Hash mismatch" && failed.line.tone == .error)
        let update = RowPresentation(row { $0.states["bf16"] = PrecisionState(state: .updateAvailable) }, chip: nil)
        // An update waiting for its Get is the button's to say ("Update · 3 MB"), as before: the hover keeps it.
        #expect(update.line.text == "Image generation" && update.line.help.hasPrefix("Update available\n"))
        let next = RowPresentation(
            row {
                $0.states["bf16"] = PrecisionState(state: .loaded); $0.selectedPrecision = "int4"
            }, chip: nil)
        #expect(next.line.text == "4-bit on next load" && next.line.tone == .pending)
        let leaving = RowPresentation(
            row {
                $0.states["bf16"] = PrecisionState(state: .ready, phase: "catalogRemoved"); $0.verification = "Removed from the catalog"
            }, chip: nil)
        #expect(leaving.line.text == "Removed from the catalog")
        // Without a purpose (an old service), the line is the state as before.
        let none = RowPresentation(row { $0.purpose = "" }, chip: nil)
        #expect(none.line == none.status)
    }

    @Test func getMoreModelsShowsAndSearchesThePurpose() {
        let entry = CatalogEntry(row: row(), metadata: [:], added: false)
        let item = BrowserItem(entry, kind: Fixtures.figureKinds.first { $0.id == "images" })
        #expect(item.purpose == "Image generation")
        #expect(item.searchText.contains("image generation"))
        #expect(BrowserList.rowHeight >= 40)
        // Status at rest: blank; how far it was verified is in the name's hover.
        #expect(item.status == .none && item.statusHelp.isEmpty && item.help.contains("Unverified: not tried on this Mac yet."))
        let quick = BrowserItem(CatalogEntry(row: row { $0.verification = "quick check" }, added: false), kind: nil)
        #expect(quick.status == .none && quick.help.contains("Quick check: a quick check ran"))
        // States that change what you do keep the cell.
        #expect(BrowserItem(CatalogEntry(row: row { $0.local = true }, added: true), kind: nil).status == .local)
        let update = BrowserItem(CatalogEntry(row: row { $0.states["bf16"] = PrecisionState(state: .updateAvailable) }, added: true), kind: nil)
        #expect(update.status == .update && update.statusHelp.contains("Update in the Models table"))
        let leaving = BrowserItem(CatalogEntry(row: row { $0.states["bf16"] = PrecisionState(state: .ready, phase: "catalogRemoved") }, added: true), kind: nil)
        #expect(leaving.status == .leaving)
        #expect(BrowserItem.Status.allCases.map(\.rawValue).allSatisfy { !["Measured", "Quick check", "Unverified"].contains($0) })
    }
}
