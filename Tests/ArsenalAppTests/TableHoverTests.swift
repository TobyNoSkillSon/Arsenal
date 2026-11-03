// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import SwiftUI
import Testing

@testable import Arsenal

/// (3.4.2): "there is no information on hover." Every hover in the Models table, on the real event path:
/// the real menu item view in AppKit's own item viewer at its real size (SettingsCardHost), and at each cell's centre
/// the view AppKit's hit test reaches must be the view that owns that cell's tooltip and report its text there. AppKit's
/// tooltip manager shows a tooltip only for the view under the pointer, so a tooltip view that the hosting view's own
/// content covers never shows, and a disabled segment or control shows none.
@MainActor @Suite(.menuState, .serialized) final class TableHoverTests {
    /// Figure fixture on another chip (reference figures), with Stable Audio busy making int8 (a locked row).
    private func open() async throws -> SettingsCardHost {
        var rows = Fixtures.figureRows()
        let audio = try #require(rows.firstIndex { $0.id == "stable-audio-3-medium" })
        rows[audio].states["int8"] = PrecisionState(state: .deriving)
        let controller = TableController(service: FixtureService(rows: rows, kinds: Fixtures.figureKinds), chip: "M3 Pro")
        controller.apply(rows: rows, kinds: Fixtures.figureKinds)
        // The store's on-disk listing: the Precision lines.
        await controller.refresh()
        let host = SettingsCardHost(controller)
        await host.settle()
        return host
    }
    private func views<T: NSView>(_ root: NSView, _: T.Type) -> [T] {
        (root as? T).map { [$0] } ?? root.subviews.flatMap { views($0, T.self) }
    }
    private func centre(_ view: NSView, _ rect: NSRect? = nil) -> NSPoint {
        let r = rect ?? view.bounds
        return view.convert(NSPoint(x: r.midX, y: r.midY), to: nil)
    }

    /// Every Precision line (chosen, loaded, flagged int4, a leftover 8-bit, locked) and every trash shows its own hover,
    /// where AppKit's hit test reaches it.
    @Test func everyPrecisionLineShowsItsHover() async throws {
        let host = try await open(); defer { host.close() }
        let window = try #require(host.window), c = host.controller
        let order = c.groups.flatMap(\.rows)
        let columns = views(host.view, PrecisionLinesView.self).sorted { $0.convert($0.bounds, to: nil).maxY > $1.convert($1.bounds, to: nil).maxY }
        #expect(columns.count == order.count)
        var offered: [String] = []
        for (row, view) in zip(order, columns) {
            let lines = c.precisionLines(row)
            #expect(view.lines == lines, "\(row.id)")
            for (i, line) in lines.enumerated() {
                let pill = Hover.offered(window, at: centre(view, view.pillRect(i)))
                #expect(pill == line.help, "\(row.id) \(line.dtype)")
                #expect(Hover.offered(window, at: centre(view, view.trashRect(i))) == line.copy.help, "\(row.id) \(line.dtype) trash")
                if let pill { offered.append(pill) }
            }
        }
        #expect(offered.contains { $0.hasPrefix(PrecisionStatSet.int4QualityWarning) }, "a flagged int4")
        #expect(offered.contains { $0.hasSuffix("Locked while int8 is made on this Mac; you can choose again when it finishes.") }, "a locked row")
        #expect(offered.contains { $0.hasSuffix(PrecisionLine.retiredLine) }, "a leftover 8-bit copy")
    }

    /// A click on a locked row's line, or on a leftover 8-bit line, changes nothing.
    @Test func lockedAndRetiredLinesRefuseClicks() async throws {
        let host = try await open(); defer { host.close() }
        let window = try #require(host.window), c = host.controller
        let order = c.groups.flatMap(\.rows).map(\.id)
        let columns = views(host.view, PrecisionLinesView.self).sorted { $0.convert($0.bounds, to: nil).maxY > $1.convert($1.bounds, to: nil).maxY }
        func click(_ id: String, dtype: String) throws {
            let view = columns[try #require(order.firstIndex(of: id))]
            let i = try #require(view.lines.firstIndex { $0.dtype == dtype })
            let point = centre(view, view.pillRect(i))
            let root = try #require(window.contentView?.superview)
            let hit = try #require(root.hitTest(root.convert(point, from: nil)))
            #expect(hit === view)
            let event = try #require(
                NSEvent.mouseEvent(
                    with: .leftMouseDown, location: point, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                    windowNumber: window.windowNumber, context: nil, eventNumber: 1, clickCount: 1, pressure: 1))
            hit.mouseDown(with: event)
        }
        try click("stable-audio-3-medium", dtype: "int4")
        try click("kokoro-82m", dtype: "int8")
        #expect(c.previewPrecision.isEmpty)
    }

    /// Every other hover in the table: figures (missing-figure triangles and other-chip references), names, status lines,
    /// headings, group headers and action buttons. Each one is the view AppKit's hit test reaches at its centre.
    @Test func everyCellHoverIsReachable() async throws {
        let host = try await open(); defer { host.close() }
        let window = try #require(host.window), c = host.controller
        var offered: [String] = []
        for tip in views(host.view, TooltipCell.TooltipView.self) {
            guard let text = tip.toolTip else { continue }
            let got = Hover.offered(window, at: centre(tip))
            #expect(got == text, "\(text.prefix(40)) is covered: \(got?.prefix(40) ?? "nothing")")
            if let got { offered.append(got) }
        }
        for action in views(host.view, RowActionView.self) {
            let got = Hover.offered(window, at: centre(action, action.buttonRect))
            #expect(got != nil && got == action.toolTipText(at: NSPoint(x: action.buttonRect.midX, y: action.buttonRect.midY)))
        }
        for header in views(host.view, GroupHeaderView.self) {
            for (rect, text) in header.tips { #expect(Hover.offered(window, at: centre(header, rect)) == text, "\(header.title)") }
        }
        let audio = try #require(c.rows.first { $0.id == "stable-audio-3-medium" })
        let missing = TableFigures.figure(audio, precision: c.presentation(audio).selected, column: .memory, host: c.chip)
        #expect(missing.missing && offered.contains(missing.help), "the missing-figure triangle")
        let laya = try #require(c.rows.first { $0.id == "laya-multilingual" })
        let reference = TableFigures.figure(laya, precision: c.presentation(laya).selected, column: .speed, host: c.chip)
        #expect(reference.reference != nil && offered.contains(reference.help), "the chip-reference figure")
        let kokoro = try #require(c.rows.first { $0.id == "kokoro-82m" })
        #expect(offered.contains(c.presentation(kokoro).line.help), "Kokoro's line under its name")
        #expect(offered.contains(TierControl.headerHelp), "the Precision heading")
        let bolts = views(host.view, PrecisionLinesView.self).compactMap { v in v.boltRect.map { Hover.offered(window, at: centre(v, $0)) } }
        #expect(bolts == [TierControl.boltHelp], "the bolt, on the one row with custom kernels")
    }
}

/// Review r1 (M1): a precision cell that cannot be chosen now (greyed, or every cell of a locked row) is disabled where
/// accessibility and AppKit look (the exposed cell and its segment elements), and no native path (accessibility press,
/// keyboard, mouse) changes the selection or reports a choice, while its hover stays offered.
extension TableHoverTests {
    /// Nested, so it runs serially with the table's hover tests (their menu settling pumps the run loop).
    @MainActor @Suite final class PrecisionAccessTests {
        final class Calls { var chosen: [String] = [] }
        static let reason = "Not offered. This model has no 4-bit format."

        private func make(enabled: Bool, _ calls: Calls) throws -> (NSWindow, TierSegments.Control) {
            let view = TierControl(
                labels: ["bf16", "int4"], selected: "16", enabled: enabled, unavailable: ["4": Self.reason], help: { "help \($0)" },
                onSelect: { calls.chosen.append($0) })
            let host = NSHostingView(rootView: view)
            host.frame = NSRect(x: 0, y: 0, width: TierControl.width, height: TierControl.height)
            // Offscreen and never ordered in.
            let window = NSWindow(
                contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: host.frame.size), styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false; window.contentView = host; host.layoutSubtreeIfNeeded()
            func find(_ v: NSView) -> TierSegments.Control? { (v as? TierSegments.Control) ?? v.subviews.lazy.compactMap(find).first }
            return (window, try #require(find(host)))
        }
        private func elements(_ control: NSView) throws -> (NSSegmentedCell, [AnyObject]) {
            let cell = try #require(NSAccessibility.unignoredDescendant(of: control) as? NSSegmentedCell)
            return (cell, (cell.accessibilityChildren() ?? []).map { $0 as AnyObject })
        }
        @Test func aGreyedCellIsDisabledForAccessibilityAndEveryActivation() throws {
            let calls = Calls()
            let (window, control) = try make(enabled: true, calls); defer { window.close() }
            let (cell, segments) = try elements(control)
            #expect(cell.isAccessibilityEnabled() && segments.count == 2)
            #expect(segments.map { $0.isAccessibilityEnabled?() ?? true } == [true, false])
            _ = segments[1].accessibilityPerformPress?()
            #expect(control.selectedSegment == 0 && calls.chosen.isEmpty, "a press on the greyed int4 changed nothing")
            // An available cell is chosen through the same action path (an accessibility press on an enabled segment runs
            // AppKit's click flash in its own event loop, which ends a full offscreen test run, so it is not pressed here).
            control.selectedSegment = 0
            _ = control.sendAction(control.action, to: control.target)
            #expect(calls.chosen == ["16"])
            // Keyboard: AppKit's keyboard selection moves `selectedSegment` and sends the action, the path every native
            // activation ends in. Driven here directly: real key events start AppKit's own event loop (a segment's click
            // flash), which ended the test process offscreen. The greyed cell is put back and nothing is reported.
            control.selectedSegment = 1
            _ = control.sendAction(control.action, to: control.target)
            // Put back to the selection the app holds (this fixture keeps 16; the press on int4 only reported it).
            #expect(control.selectedSegment == 0 && !calls.chosen.contains("4"))
            // Mouse: a click on the greyed cell changes nothing.
            let before = control.selectedSegment, count = calls.chosen.count
            let point = control.convert(NSPoint(x: 1.5 * (TierControl.Natural.cellWidth - 1), y: control.bounds.midY), to: nil)
            control.mouseDown(
                with: try #require(
                    NSEvent.mouseEvent(
                        with: .leftMouseDown, location: point, modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber, context: nil, eventNumber: 1,
                        clickCount: 1, pressure: 1)))
            #expect(control.selectedSegment == before && calls.chosen.count == count)
            // Its hover is still offered, from the control's own registered region.
            #expect(Hover.offered(window, at: point) == Self.reason)
        }

        @Test func aLockedRowIsDisabledForAccessibilityAndEveryActivation() throws {
            let calls = Calls()
            let (window, control) = try make(enabled: false, calls); defer { window.close() }
            let (cell, segments) = try elements(control)
            #expect(!cell.isAccessibilityEnabled() && segments.count == 2)
            #expect(segments.allSatisfy { !($0.isAccessibilityEnabled?() ?? true) })
            for segment in segments { _ = segment.accessibilityPerformPress?() }
            #expect(control.selectedSegment == 0 && calls.chosen.isEmpty)
            // Keyboard path (see the greyed-cell test): every cell is put back while the row is locked.
            for index in 0..<2 {
                control.selectedSegment = index
                _ = control.sendAction(control.action, to: control.target)
                #expect(control.selectedSegment == 0 && calls.chosen.isEmpty, "segment \(index)")
            }
            // The 16-bit cell (segment 0) is the chosen one; its hover names the lock.
            let point = control.convert(NSPoint(x: 0.5 * (TierControl.Natural.cellWidth - 1), y: control.bounds.midY), to: nil)
            control.mouseDown(
                with: try #require(
                    NSEvent.mouseEvent(
                        with: .leftMouseDown, location: point, modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber, context: nil, eventNumber: 1,
                        clickCount: 1, pressure: 1)))
            #expect(control.selectedSegment == 0 && calls.chosen.isEmpty)
            // Its hover is offered with the lock line.
            #expect(Hover.offered(window, at: point) == TierControl.tooltip("help 16", enabled: false))
        }

        /// The regions AppKit was given: one per drawn cell, exactly the cell, whatever the cell's state (deleting the
        /// registration or registering an empty or wrong rect fails here and in TableHoverTests).
        @Test func eachCellHasExactlyItsOwnRegisteredTooltipRegion() throws {
            for enabled in [true, false] {
                let (window, control) = try make(enabled: enabled, Calls()); defer { window.close() }
                let pitch = TierControl.Natural.cellWidth - 1
                let rects = control.toolTipRegions.values.map(\.rect).sorted { $0.minX < $1.minX }
                #expect(rects == (0..<2).map { NSRect(x: CGFloat($0) * pitch, y: 0, width: pitch, height: control.bounds.height) })
            }
        }
    }
}
