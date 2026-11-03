// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal
import ArsenalCore

/// v2.2 items 1–2 on a never-shown browser window (no screen, cursor or focus is touched). Each tap asks the window's
/// frame view which view a click at that point reaches (AppKit's own hit-testing, as for a real click), then hands that
/// view the mouse-down and mouse-up as NSWindow would. (`NSWindow.sendEvent` drops clicks on table rows of a window that
/// was never ordered in, so dispatching them ourselves keeps the test offscreen.)
@MainActor @Suite(.menuState, .serialized) final class V22ClickTests {
    private func browser() throws -> (BrowserController, BrowserWindowController, NSWindow, BrowserRootView) {
        _ = NSApplication.shared
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        c.apply(entries: BrowserTests.entries, kinds: BrowserTests.kinds)
        UserDefaults.standard.set(true, forKey: BrowserRootView.openKey)
        let wc = BrowserWindowController(controller: c)
        let w = try #require(wc.window)
        w.setFrame(NSRect(x: -9000, y: -9000, width: 1440, height: 820), display: false)
        let root = try #require(w.contentView as? BrowserRootView)
        wc.list.show(c.shown)
        settle(w, wc.list)
        return (c, wc, w, root)
    }
    private func settle(_ w: NSWindow, _ list: BrowserList) {
        w.contentView?.layoutSubtreeIfNeeded(); w.displayIfNeeded()
        for r in 0..<list.lines.count { _ = list.table.rowView(atRow: r, makeIfNecessary: true) }
    }
    /// A tap: mouse-down then mouse-up at one point, handed to the view AppKit's hit-test picks there. An NSButton runs a
    /// nested tracking loop on mouse-down, which inside a test would also run the test runner's own queued work, so it
    /// is pressed through its own click path instead (the hit-test already proved the click reaches it).
    private func tap(_ w: NSWindow, _ p: NSPoint, count: Int = 1, flags: NSEvent.ModifierFlags = []) {
        func make(_ type: NSEvent.EventType) -> NSEvent {
            NSEvent.mouseEvent(
                with: type, location: p, modifierFlags: flags, timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: w.windowNumber,
                context: nil, eventNumber: 0, clickCount: count, pressure: type == .leftMouseDown ? 1 : 0)!
        }
        guard let view = hit(w, p) else { Issue.record("nothing at \(p)"); return }
        if let button = view as? NSButton { button.performClick(nil); return }
        view.mouseDown(with: make(.leftMouseDown))
        view.mouseUp(with: make(.leftMouseUp))
    }
    private func hit(_ w: NSWindow, _ p: NSPoint) -> NSView? {
        guard let frame = w.contentView?.superview else { return nil }
        return frame.hitTest(frame.convert(p, from: nil))
    }
    private func point(_ list: BrowserList, _ column: BrowserColumn, row: Int, dx: CGFloat? = nil) -> NSPoint {
        let r = list.table.frameOfCell(atColumn: list.table.column(withIdentifier: .init(column.rawValue)), row: row)
        return list.table.convert(NSPoint(x: dx.map { r.minX + $0 } ?? r.midX, y: r.midY), to: nil)
    }
    private func id(_ list: BrowserList, _ row: Int) -> String { list.lines[row].id }

    @Test func aTapOnTheChevronOrAnywhereOnTheRowExpandsAndCollapsesAtOnce() throws {
        let (c, wc, w, _) = try browser(); defer { w.close() }
        let list = wc.list
        let first = id(list, 0)
        // The chevron glyph's centre (BrowserColumns.pad + 6 into its cell): before the fix the table took this click.
        tap(w, point(list, .chevron, row: 0, dx: BrowserColumns.pad + 6))
        #expect(c.expanded == [first])
        #expect(list.lines[1] == .detail(first))
        tap(w, point(list, .chevron, row: 0, dx: BrowserColumns.pad + 6))
        #expect(c.expanded.isEmpty)
        // Any other cell of the row: name, a number, the licence, the empty flex gap.
        settle(w, list)
        let second = id(list, 1)
        for column in [BrowserColumn.model, .download, .licence, .flex] {
            tap(w, point(list, column, row: 1))
            #expect(c.expanded == [second], "\(column)")
            settle(w, list)
            tap(w, point(list, column, row: 1))
            #expect(c.expanded.isEmpty, "\(column)")
            settle(w, list)
        }
        #expect(list.table.selectedRow == 1)
    }

    @Test func aDoubleClickLeavesTheRowOpenAndOptionOnTheChevronOpensEveryRow() throws {
        let (c, wc, w, _) = try browser(); defer { w.close() }
        let list = wc.list
        let first = id(list, 0)
        tap(w, point(list, .model, row: 0))
        tap(w, point(list, .model, row: 0), count: 2)
        #expect(c.expanded == [first])
        c.expanded = []; list.expansionChanged(); settle(w, list)
        tap(w, point(list, .chevron, row: 0, dx: BrowserColumns.pad + 6), flags: .option)
        #expect(c.expanded == Set(c.shown.map(\.id)))
    }

    /// rv22-opus B1: the pill must be the view a real click reaches. NSWindow sends the mouse-up to whichever view took the
    /// mouse-down; when the table claimed the pill's point, the pill never saw its mouse-up and Add/Remove did nothing.
    /// The tap goes only to the hit-tested view (no event is handed to the button by name).
    @Test func addAndRemoveReceiveARealClickAndDoNotExpand() throws {
        let (c, wc, w, _) = try browser(); defer { w.close() }
        let list = wc.list
        let row = try #require(list.lines.firstIndex { line in c.shown.contains { $0.id == line.id && !$0.added } })
        let itemID = id(list, row)
        let p = point(list, .action, row: row, dx: BrowserColumns.fixed[.action]! - BrowserColumns.pad - 35)
        let button = try #require(list.table.control(at: list.table.convert(p, from: nil), row: row) as? PillButton)
        #expect(hit(w, p) === button)
        tap(w, p)
        #expect(c.expanded.isEmpty)
        #expect(c.items.first { $0.id == itemID }?.added == true)
        settle(w, list)
        let again = try #require(list.lines.firstIndex { $0.id == itemID })
        let q = point(list, .action, row: again, dx: BrowserColumns.fixed[.action]! - BrowserColumns.pad - 35)
        #expect((hit(w, q) as? PillButton)?.caption == "Remove")
        tap(w, q)
        #expect(c.expanded.isEmpty)
        #expect(c.items.first { $0.id == itemID }?.added == false)
        // The rest of the row still belongs to the table: a tap beside the pill expands.
        settle(w, list)
        let r = try #require(list.lines.firstIndex { $0.id == itemID })
        #expect(hit(w, point(list, .flex, row: r)) === list.table)
    }

    @Test func theSidebarButtonHidesAndShowsTheSidebar() throws {
        let (_, _, w, root) = try browser(); defer { w.close() }
        func centre() -> NSPoint { root.convert(NSPoint(x: root.sidebarButton.frame.midX, y: root.sidebarButton.frame.midY), to: nil) }
        #expect(root.sidebarOpen)
        #expect(hit(w, centre()) === root.sidebarButton)
        #expect(root.sidebarButton.toolTip == "Hide sidebar")
        tap(w, centre())
        #expect(!root.sidebarOpen)
        root.layoutSubtreeIfNeeded()
        #expect(hit(w, centre()) === root.sidebarButton)
        #expect(root.sidebarButton.toolTip == "Show sidebar")
        tap(w, centre())
        #expect(root.sidebarOpen)
    }
}
