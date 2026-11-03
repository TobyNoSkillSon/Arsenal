// SPDX-License-Identifier: MIT
import Foundation
import Testing
import AppKit
import SwiftUI
@testable import Arsenal
import ArsenalCore

/// The table in the real menu item, on the first layout pass, in every render state: exactly as wide as declared, every
/// control inside the item view, one action cell per row and one header per group. The group header's own controls
/// (arrows, drop slots) are exercised by calling the view's handlers directly; nothing touches the screen.
@MainActor @Suite(.menuState, .serialized) final class TableLayoutTests {
    private func all<T: NSView>(_ type: T.Type, in view: NSView) -> [T] {
        var found: [T] = []
        func walk(_ v: NSView) { if let t = v as? T { found.append(t) }; v.subviews.forEach(walk) }
        walk(view); return found
    }

    private func laidOut(_ c: TableController) throws -> (NSMenu, NSView, NSWindow) {
        let root = ModelsMenu(controller: c).modelItem()
        let menu = try #require(root.submenu)
        let view = try #require(menu.items.first?.view)
        let window = NSWindow(contentRect: view.frame, styleMask: .borderless, backing: .buffered, defer: false)
        window.contentView = view
        view.layoutSubtreeIfNeeded()
        return (menu, view, window)
    }

    @Test func everyRenderStateFitsTheDeclaredSize() throws {
        for shot in RenderDelegate.shots() {
            let c = RenderDelegate.controller(shot)
            let free = NSHostingView(rootView: ModelTable(controller: c))
            #expect(free.fittingSize.width == ModelTable.width(c), "\(shot.name): fitting width")
            #expect(free.fittingSize.height == ModelTable.height(c), "\(shot.name): fitting height")
            let (menu, view, _) = try laidOut(c)
            // The submenu holds the table and the activity panel beside it.
            #expect(view.frame.width == ModelsMenuView.width(c), Comment(rawValue: shot.name))
            #expect(menu.minimumWidth >= ModelsMenuView.width(c), Comment(rawValue: shot.name))
            let actions = all(RowActionView.self, in: view)
            #expect(actions.count == c.rows.count, "\(shot.name): one action per model")
            #expect(all(GroupHeaderView.self, in: view).count == c.groups.count, "\(shot.name): one header per group")
            let limit = view.bounds.maxX - ModelTable.trailingMargin
            for action in actions {
                for rect in [action.buttonRect, action.trashRect, action.bounds] {
                    #expect(action.convert(rect, to: view).maxX <= limit + 0.5, Comment(rawValue: shot.name))
                }
            }
            for control in all(NSSegmentedControl.self, in: view) {
                #expect(view.bounds.contains(control.convert(control.bounds, to: view)), "\(shot.name): \(type(of: control)) outside")
            }
        }
    }

    @Test func theTableIsSaneWithTheV1Catalog() {
        let c = RenderDelegate.controller(.init(name: "x", rows: Fixtures.rows(.loaded)))
        #expect(c.groups.count == 3)
        #expect(c.rows.count == 3)
        #expect(ModelTable.width(c) < 1200)
        #expect(ModelTable.height(c) < 860)
    }

    @Test func aTallTableScrollsWithinTheMaximumHeight() {
        let saved = ModelTable.maxHeight
        defer { ModelTable.maxHeight = saved }
        ModelTable.maxHeight = 200
        let c = RenderDelegate.controller(.init(name: "x", rows: Fixtures.rows(.loaded)))
        #expect(ModelTable.scrolls(c))
        #expect(ModelTable.height(c) == 200)
        let free = NSHostingView(rootView: ModelTable(controller: c))
        #expect(free.fittingSize.height == 200)
    }

    @Test func rowsDoNotMoveWhenAPrecisionChanges() throws {
        let c = RenderDelegate.controller(.init(name: "x", rows: Fixtures.rows(.loaded)))
        let (_, view, _) = try laidOut(c)
        let before = all(RowActionView.self, in: view).map { $0.convert($0.bounds, to: view) }
        for row in c.rows where !c.presentation(row).locked { c.select(row, column: "4") }
        view.layoutSubtreeIfNeeded()
        #expect(all(RowActionView.self, in: view).map { $0.convert($0.bounds, to: view) } == before)
    }

    @Test func theArrowsMoveTheGroupAndTheDragPicksTheSlotUnderThePointer() throws {
        let c = RenderDelegate.controller(.init(name: "x", rows: Fixtures.rows(.loaded)))
        let (_, view, window) = try laidOut(c)
        let headers = GroupHeaderView.headers(in: view)
        #expect(headers.map(\.kindID) == ["voices", "images", "chat"])
        // Down arrow on Voices.
        let voices = headers[0]
        let point = voices.convert(NSPoint(x: voices.downRect.midX, y: voices.downRect.midY), to: nil)
        let event = try #require(
            NSEvent.mouseEvent(
                with: .leftMouseDown, location: point, modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber, context: nil,
                eventNumber: 0, clickCount: 1, pressure: 1))
        voices.mouseDown(with: event)
        #expect(c.groups.map(\.kind.id) == ["images", "voices", "chat"])
        view.layoutSubtreeIfNeeded()
        // A pointer on the Chat header's upper half drops before Chat; below the last group's middle, after it.
        let after = GroupHeaderView.headers(in: view)
        let chat = after[2]
        let onChat = chat.convert(NSPoint(x: 10, y: 2), to: nil)
        #expect(chat.dropSlot(atWindowPoint: onChat) == 2)
        let bottom = view.convert(NSPoint(x: 10, y: view.isFlipped ? view.bounds.maxY - 40 : 40), to: nil)
        #expect(chat.dropSlot(atWindowPoint: bottom) == 3)
        let top = after[0].convert(NSPoint(x: 10, y: 1), to: nil)
        #expect(chat.dropSlot(atWindowPoint: top) == 0)
        // The drop itself: what mouse-up does with the slot.
        chat.onDrop?(0)
        #expect(c.groups.map(\.kind.id) == ["chat", "images", "voices"])
        #expect(c.drag == nil)
    }

    @Test func headerTooltipsNameTheColumnsAndTheMove() throws {
        let c = RenderDelegate.controller(.init(name: "x", rows: Fixtures.rows(.loaded)))
        let (_, view, _) = try laidOut(c)
        let voices = GroupHeaderView.headers(in: view)[0]
        #expect(voices.toolTipText(at: NSPoint(x: 20, y: 10)) == "Text to speech\nDrag to reorder")
        #expect(all(TooltipCell.TooltipView.self, in: view).contains { $0.toolTip == TableStats.headerHelp("memory") })
    }

    @Test func theSmokeCheckPasses() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-app-smoke-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: []), paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        // No status item in tests (it would appear in the menu bar); everything else as in `Arsenal --smoke`.
        let line = SmokeDelegate.check(app)
        #expect(line.text.contains("0 groups, 0 rows, 0 action cells, 1 pinned Get more models row"), Comment(rawValue: line.text))
        #expect(!line.ok) // No status item: smoke must retain the failure result in tests.
    }
}
