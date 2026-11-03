// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import SwiftUI
import Testing

@testable import Arsenal

/// Review r1 (S2): the hover surfaces now lie over the table's cells, over locked card controls, and the settings card no
/// longer has an AppKit click shield. Clicks, drags and scrolls through them, on the real menu item view (AppKit's own
/// item viewer, offscreen, never shown) through AppKit's hit test: enabled controls still work, covered and blank regions
/// reach nothing beneath, and the menu sees no action, no cancelTracking, no structural change, no move and no growth.
/// Offscreen fixtures are not AppKit's live menu tracking; that difference stays unverified here.
@MainActor @Suite(.menuState, .serialized) final class HoverClickTests {
    private let support = URL(fileURLWithPath: "/private/tmp/arsenal-hover-clicks/\(UUID())")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// What the menu and table look like from outside, before and after (geometry and state).
    private struct Snapshot: Equatable {
        var frame: NSRect, natural: NSSize, height: CGFloat
        var selected: [String], previews: [String: String], open: String?
    }
    private func snapshot(_ open: SettingsCardTests.Open) -> Snapshot {
        let t = open.app.table
        return Snapshot(
            frame: open.host.view.frame, natural: open.host.view.naturalSize, height: ModelTable.height(t), selected: t.rows.map(\.selectedPrecision),
            previews: t.previewPrecision, open: t.settingsCard.openID)
    }
    private func event(_ type: NSEvent.EventType, _ window: NSWindow, at point: NSPoint, number: Int) throws -> NSEvent {
        try #require(
            NSEvent.mouseEvent(
                with: type, location: point, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
                context: nil, eventNumber: number, clickCount: 1, pressure: type == .leftMouseUp ? 0 : 1))
    }
    private static var clicks = 1000
    /// A click (or a drag, with `drag`) at a window point, delivered to the view AppKit's hit test picks, as the menu's
    /// tracking delivers it: the mouse-down, any drags, the mouse-up. A control that tracks the mouse itself (an NSButton)
    /// is completed as a finished click instead (see below). Returns the view the click reached.
    @discardableResult private func click(_ open: SettingsCardTests.Open, at point: NSPoint, drag: CGFloat = 0) throws -> NSView {
        let window = try #require(open.host.window)
        let root = try #require(window.contentView?.superview)
        let target = try #require(root.hitTest(root.convert(point, from: nil)))
        Self.clicks += 1
        let down = try event(.leftMouseDown, window, at: point, number: Self.clicks)
        let end = NSPoint(x: point.x + drag, y: point.y)
        let up = try event(.leftMouseUp, window, at: end, number: Self.clicks)
        let previous = open.host.view.currentEvent
        open.host.view.currentEvent = { down }
        defer { open.host.view.currentEvent = previous }
        if let button = target as? NSButton {
            // An NSButton tracks the mouse in AppKit's own event loop (and `performClick` flashes in one), which ends the
            // test process offscreen: the click reached it through the hit test, and what a completed click does follows
            // (the next state, then the action).
            button.setNextState()
            _ = button.sendAction(button.action, to: button.target)
        } else {
            target.mouseDown(with: down)
            if drag != 0 { target.mouseDragged(with: try event(.leftMouseDragged, window, at: end, number: Self.clicks)) }
            target.mouseUp(with: up)
        }
        return target
    }
    private func centre(_ view: NSView) -> NSPoint { view.convert(NSPoint(x: view.bounds.midX, y: view.bounds.midY), to: nil) }

    /// Every hover cell of the table, clicked and dragged through its hover view: nothing happens, in the table or the menu.
    @Test func clicksAndDragsOnTableHoverCellsDoNothing() async throws {
        let ids = SettingsCardTests.catalogTables.first ?? []
        let open = try await SettingsCardTests.open(ids, support: support.appendingPathComponent("table"))
        defer { open.close() }
        let calls = SettingsCardTests.MenuCalls(root: open.app.menu)
        defer { calls.stop() }
        let before = snapshot(open)
        var reached = 0
        let cancellations = try await SettingsCardTests.countingCancellations {
            for tip in Hover.views(open.host.view).compactMap({ $0 as? TooltipCell.TooltipView }) where tip.toolTip != nil {
                let target = try click(open, at: centre(tip))
                #expect(target === tip, "the hover view takes the click: \(tip.toolTip?.prefix(30) ?? "")")
                try click(open, at: centre(tip), drag: 12)
                reached += 1
            }
            await open.host.settle()
        }
        #expect(reached > 20)
        #expect(snapshot(open) == before, "nothing in the table changed")
        #expect(calls.actions == 0 && calls.structural == 0 && cancellations == 0)
    }

    /// The settings card: an enabled switch toggles through a real click; a blank part of the card and a locked control
    /// (its hover shield) reach nothing, neither in the card nor in the table under it.
    @Test func cardClicksReachItsControlsAndNothingBeneath() async throws {
        let id = "laya-english"
        for state in [ModelState.ready, .downloading] {
            let open = try await SettingsCardTests.open([id, "kokoro-82m", "flux2-klein-4b"], support: support.appendingPathComponent("\(state)"), state: state)
            defer { open.close() }
            try open.host.press(try #require(open.host.gear(id)))
            await open.host.settle()
            #expect(open.app.table.settingsCard.openID == id)
            let calls = SettingsCardTests.MenuCalls(root: open.app.menu)
            defer { calls.stop() }
            let toggle = try #require(open.host.views(MenuSwitch.self).first { $0.isEnabled }, "an enabled switch (Keep loaded)")
            let locked = open.host.views(NSControl.self).filter { !$0.isEnabled && ($0 is MenuSwitch || $0 is MenuSegments || $0 is NSSlider || $0 is MenuField) }
            #expect(state == .ready ? locked.isEmpty : !locked.isEmpty, "load-time rows lock while a Get runs")
            let card = open.host.frame
            let hosting = open.host.view.hosting
            let cancellations = try await SettingsCardTests.countingCancellations {
                // An enabled switch toggles through a click that AppKit's hit test sends to it.
                let was = toggle.state
                #expect(try click(open, at: centre(toggle)) === toggle && toggle.state != was, "the enabled switch toggles")
                // A locked control: its hover shield (with the lock line) takes the click; the control does not change.
                for control in locked {
                    let value = control.objectValue as AnyObject?
                    let target = try click(open, at: centre(control))
                    #expect(target is TooltipCell.TooltipView && target.toolTip?.contains("Locked") == true, "\(type(of: control))")
                    #expect((control.objectValue as AnyObject?)?.isEqual(value) ?? (value == nil))
                }
                await open.host.settle()
                // A blank point of the card (its title line, right of the name) over the table: nothing under it is reached.
                let blank = hosting.convert(NSPoint(x: card.maxX - 12, y: card.minY + SettingsCard.headerHeight / 2), to: nil)
                let beforeBlank = snapshot(open)
                let reached = try click(open, at: blank)
                #expect(!(reached is PrecisionLinesView || reached is GearView || reached is RowActionView || reached is TooltipCell.TooltipView))
                await open.host.settle()
                #expect(snapshot(open) == beforeBlank && open.app.table.settingsCard.openID == id, "a blank click keeps the card and the table")
            }
            #expect(calls.actions == 0 && calls.structural == 0 && cancellations == 0, "\(state)")
        }
    }

    /// A scrolled table: the wheel over a hover view goes where it goes over the cell without one, the table's scroll view.
    /// Synthetic wheel events move no SwiftUI scroll view offscreen, not even sent to the scroll view directly (pixel,
    /// line and phased events tried, window ordered far off every screen), so the scroll itself is unverified here; what
    /// is checked is the event path: the hit view at a hover cell passes the wheel up its responder chain unchanged (it
    /// does not implement it), and that chain reaches the table's scroll view.
    @Test func aScrolledTablePassesTheWheelThroughItsHoverViews() async throws {
        let saved = ModelTable.maxHeight
        defer { ModelTable.maxHeight = saved }
        ModelTable.maxHeight = 320
        let open = try await SettingsCardTests.open(SettingsCardTests.catalogTables.first ?? [], support: support.appendingPathComponent("scroll"))
        defer { open.close() }
        #expect(ModelTable.scrolls(open.app.table))
        let window = try #require(open.host.window), root = try #require(window.contentView?.superview)
        let selector = #selector(NSView.scrollWheel(with:))
        #expect(class_getMethodImplementation(TooltipCell.TooltipView.self, selector) == class_getMethodImplementation(NSView.self, selector))
        // Hover views inside the scroll view whose centre is in its visible part.
        let tips = Hover.views(open.host.view).compactMap { $0 as? TooltipCell.TooltipView }.filter { tip in
            guard let clip = tip.enclosingScrollView?.contentView, tip.toolTip != nil else { return false }
            let r = tip.convert(tip.bounds, to: clip)
            return clip.bounds.contains(NSPoint(x: r.midX, y: r.midY))
        }
        #expect(tips.count > 5)
        for tip in tips {
            let target = try #require(root.hitTest(root.convert(centre(tip), from: nil)))
            #expect(target === tip)
            var responder: NSResponder? = target.nextResponder
            while let r = responder, r !== tip.enclosingScrollView { responder = r.nextResponder }
            #expect(responder === tip.enclosingScrollView, "the wheel reaches the table's scroll view")
        }
    }
}
