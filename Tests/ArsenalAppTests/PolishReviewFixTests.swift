// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

/// Never shown: retain AppKit's existing cells to catch state updates a full reload would hide.
@MainActor @Suite(.menuState, .serialized) final class PolishReviewFixTests {
    private func items(_ ids: [String]) -> [BrowserItem] {
        ids.map { BrowserItem(CatalogEntry(row: ModelRow(id: $0, name: $0, kind: "voices")), kind: nil) }
    }
    private func window(_ list: BrowserList) -> NSWindow {
        _ = NSApplication.shared
        let w = NSWindow(
            contentRect: NSRect(x: -9000, y: -9000, width: 1200, height: 600),
            styleMask: .borderless, backing: .buffered, defer: false)
        w.isReleasedWhenClosed = false
        w.contentView = list.scrollView
        return w
    }
    private func selectedID(_ list: BrowserList) -> String? {
        let row = list.table.selectedRow
        guard row >= 0, row < list.lines.count, case .item(let id) = list.lines[row] else { return nil }
        return id
    }
    private func chevron(_ list: BrowserList, _ id: String) -> ChevronCell? {
        guard let row = list.lines.firstIndex(of: .item(id)) else { return nil }
        return list.table.view(atColumn: list.table.column(withIdentifier: .init("chevron")), row: row, makeIfNecessary: true) as? ChevronCell
    }

    @Test func existingChevronFollowsIndividualAllAndKeyboardExpansion() throws {
        let c = BrowserController(service: FixtureService(rows: []))
        let list = BrowserList(); list.controller = c
        let w = window(list); defer { w.close() }
        let rows = items(["a", "b", "c"])
        c.apply(entries: rows.map(\.entry), kinds: [])
        list.show(rows); w.contentView?.layoutSubtreeIfNeeded()
        let before = try #require(chevron(list, "a"))
        #expect(before.toolTip == "Expand")
        // Astra's exact probe: change the controller, without selecting the row or showing a window.
        c.expanded = ["a"]; list.expansionChanged(); w.contentView?.layoutSubtreeIfNeeded()
        #expect(list.lines.count == 4)
        #expect(chevron(list, "a")?.toolTip == "Collapse")
        #expect(chevron(list, "a")?.subviews.first?.accessibilityLabel() == "Collapse")
        list.show(rows) // An unchanged refresh must not be needed to repair the disclosure.
        #expect(chevron(list, "a")?.toolTip == "Collapse")
        list.table.selectRowIndexes([0], byExtendingSelection: false)
        list.toggleSelected(open: false)
        #expect(chevron(list, "a")?.toolTip == "Expand")
        list.toggleAll()
        for id in ["a", "b", "c"] { #expect(chevron(list, id)?.toolTip == "Collapse") }
        list.toggleAll()
        for id in ["a", "b", "c"] { #expect(chevron(list, id)?.toolTip == "Expand") }
        list.toggleSelected(open: true); list.escape()
        #expect(c.expanded.isEmpty && chevron(list, "a")?.toolTip == "Expand")
    }

    @Test func selectionFollowsIdentityAcrossSortInsertRemoveAndExpansion() {
        let c = BrowserController(service: FixtureService(rows: []))
        let list = BrowserList(); list.controller = c
        let w = window(list); defer { w.close() }
        list.show(items(["a", "b", "c"])); w.contentView?.layoutSubtreeIfNeeded()
        list.table.selectRowIndexes([0], byExtendingSelection: false)
        list.show(items(["c", "b", "a"]))
        #expect(selectedID(list) == "a")
        list.show(items(["a", "b", "c"]))
        #expect(selectedID(list) == "a")
        list.show(items(["z", "a", "b", "c"]))
        #expect(selectedID(list) == "a")
        list.show(items(["a", "b", "c"]))
        #expect(selectedID(list) == "a")
        c.expanded = ["b"]; list.expansionChanged()
        list.show(items(["c", "b", "a"]))
        #expect(selectedID(list) == "a")
        var refreshed = items(["c", "b", "a"])
        var entry = refreshed[2].entry; entry.added = true
        refreshed[2] = BrowserItem(entry, kind: nil)
        list.show(refreshed)
        #expect(selectedID(list) == "a")
        list.show(items(["c", "b"]))
        #expect(selectedID(list) == nil && list.table.selectedRow == -1)
        list.show(items(["c", "b", "a"]))
        #expect(selectedID(list) == nil)
    }

    @Test func rowKeepsItsControllerUntilTheRowIsReleased() throws {
        var controller: TableController? = TableController(service: FixtureService(rows: []))
        weak var retained: TableController?
        retained = controller
        let model = ModelRow(id: "a", name: "A", kind: "voices")
        var row: ModelTableRow? = ModelTableRow(
            row: model, columns: [], slots: 0, p: RowPresentation(model), chip: nil,
            hasSettings: false, hovered: false, changes: [],
            loadChoices: [], pal: .dark, controller: try #require(controller))
        controller = nil
        // Do not dereference an unowned reference when this fails on the old implementation.
        #expect(retained != nil)
        if retained != nil { #expect(row == row); _ = row?.body }
        row = nil
        #expect(retained == nil)
    }

    @Test func startupMenuCanQuitBeforeServiceVerificationFinishes() throws {
        let menu = LaunchDelegate.startupMenu()
        #expect(menu.items.first?.title == "Verifying model catalog\u{2026}")
        #expect(menu.items.first?.isEnabled == false)
        let quit = try #require(menu.items.last)
        #expect(quit.title == "Quit Arsenal" && quit.isEnabled && quit.keyEquivalent == "q")
        #expect(quit.action == #selector(NSApplication.terminate(_:)))
        #expect(quit.target === NSApplication.shared)
    }
}
