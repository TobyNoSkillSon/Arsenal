// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import SwiftUI
import Testing

@testable import Arsenal

/// 3.2.1 (the Models table "being cut off" and "that black is different than the background … make the
/// background the same as a menu"): the Models and Activity submenus hold their hosted view alone, like Vella's, and
/// neither view paints a fill of its own over the menu's material. Refresh lives in the table's header.
@MainActor @Suite(.menuState, .serialized) final class MenuFitTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-fit-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    private func app(_ scene: Fixtures.Scene = .loaded) async -> AppDelegate {
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(scene), supportDirectory: support), paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        return app
    }

    @Test func modelsAndActivitySubmenusHoldTheirHostedViewAlone() async throws {
        let app = await app()
        // A refused Refresh used to add an imaged notice item below the table; it is the footer line now.
        app.modelsMenu.updateRefresh(enabled: true, notice: "Finish Get or Load before refreshing models.")
        app.rebuildMenu()
        defer { app.table.stop(); app.tools.closeAll() }
        let models = try #require(app.menu.items.first { $0.title == "Models" }?.submenu)
        let activity = try #require(app.menu.items.first { $0.title == "Activity" }?.submenu)
        for menu in [models, activity] {
            #expect(menu.items.count == 1)
            #expect(menu.items.allSatisfy { $0.image == nil && !$0.isSeparatorItem })
            #expect(menu.items.first?.view != nil)
        }
        #expect(models.items.first?.view is MenuTableHostingView)
        #expect(activity.items.first?.view is ActivityHostingView)
        // The item view is exactly the table's width, and the menu is no narrower.
        let view = try #require(models.items.first?.view)
        #expect(view.frame.width == ModelTable.width(app.table) && models.minimumWidth == view.frame.width)
        #expect(models.size.width == view.frame.width)
    }

    /// Rendered over a transparent window: every corner of each hosted view stays fully transparent (the 3.2.0 table and
    /// panel filled their whole frame with the palette's canvas), in the dark and the light appearance.
    @Test func hostedViewsDrawNoBackground() async throws {
        let app = await app()
        defer { app.table.stop(); app.tools.closeAll(); Appearance.shared.set(.dark) }
        let models = try #require(app.modelsMenu.modelItem().submenu?.items.first?.view)
        let activity = try #require(app.activityMenu.item().submenu?.items.first?.view)
        app.activityMenu.resize()
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode)
            for view in [models, activity] {
                #expect(!view.isOpaque)
                if let layer = view.layer { #expect((layer.backgroundColor?.alpha ?? 0) == 0) }
                let window = NSWindow(contentRect: view.frame.offsetBy(dx: -9000, dy: -9000), styleMask: .borderless, backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false; window.backgroundColor = .clear; window.isOpaque = false
                window.contentView = view; window.appearance = Appearance.shared.nsAppearance
                view.layoutSubtreeIfNeeded()
                let rep = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
                view.cacheDisplay(in: view.bounds, to: rep)
                let w = rep.pixelsWide, h = rep.pixelsHigh
                for (x, y) in [(1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2)] {
                    let alpha = rep.colorAt(x: x, y: y)?.alphaComponent ?? 1
                    #expect(alpha == 0, "\(type(of: view)) \(mode) pixel (\(x), \(y)) alpha \(alpha)")
                }
                window.contentView = nil; window.close()
            }
        }
    }

    @Test func headerRefreshCallsRefreshModelsAndRespectsItsEnabledState() async throws {
        let controller = TableController(service: FixtureService(rows: Fixtures.rows(.loaded)))
        controller.apply(rows: Fixtures.rows(.loaded), kinds: Fixtures.kinds)
        let menu = ModelsMenu(controller: controller)
        var calls = 0
        menu.refreshModels = { calls += 1 }
        let view = try #require(menu.modelItem().submenu?.items.first?.view)
        let window = NSWindow(contentRect: view.frame.offsetBy(dx: -9000, dy: -9000), styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.contentView = view
        defer { window.contentView = nil; window.close() }
        view.layoutSubtreeIfNeeded()
        let refresh = try #require(MenuTests.find(RefreshControlView.self, in: view))
        #expect(refresh.toolTip == RefreshControl.help && refresh.isAccessibilityEnabled())
        // An icon at the top right: in the header, above the first group, centred over the rows' trash glyphs, inside the
        // table's width (nothing past the right edge).
        let frame = refresh.convert(refresh.bounds, to: view)
        let header = try #require(MenuTests.find(GroupHeaderView.self, in: view))
        let headerFrame = header.convert(header.bounds, to: view)
        #expect(view.isFlipped ? frame.maxY <= headerFrame.minY : frame.minY >= headerFrame.maxY)
        let action = try #require(MenuTests.find(RowActionView.self, in: view))
        let trash = action.convert(action.trashRect, to: view)
        #expect(abs(frame.midX - trash.midX) < 0.5 && frame.width == trash.width)
        #expect(frame.maxX <= view.bounds.maxX - ModelTable.trailingMargin + 0.5)
        #expect(MenuTests.find(PinnedRowView.self, in: view).map { $0.convert($0.bounds, to: view).width } == ModelTable.W.row(slots: ModelTable.slots(controller.kinds)))
        #expect(refresh.accessibilityLabel() == "Refresh models")

        // The real event path: a mouse down on the control.
        let down = try #require(
            NSEvent.mouseEvent(
                with: .leftMouseDown, location: refresh.convert(NSPoint(x: refresh.bounds.midX, y: refresh.bounds.midY), to: nil), modifierFlags: [],
                timestamp: 0, windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1))
        let frameView = try #require(window.contentView?.superview)
        let hit = try #require(frameView.hitTest(frameView.convert(down.locationInWindow, from: nil)))
        #expect(hit === refresh, "hit \(type(of: hit))")
        hit.mouseDown(with: down)
        #expect(calls == 1)
        #expect(refresh.accessibilityPerformPress() && calls == 2)

        // Off while a Refresh runs; a refusal shows in the footer and the table grows to show it.
        let before = view.frame.height
        menu.updateRefresh(enabled: false, notice: nil)
        try await waitUntil { !refresh.enabled }
        refresh.mouseDown(with: down)
        #expect(!refresh.accessibilityPerformPress() && !refresh.isAccessibilityEnabled() && calls == 2)
        #expect(refresh.toolTip == "Checking for new models\u{2026}")
        // A refusal that is not "busy" keeps its footer line while Refresh is allowed.
        menu.updateRefresh(enabled: true, notice: "The catalog could not be read.")
        try await waitUntil { refresh.enabled && view.frame.height == ModelTable.height(controller) }
        #expect(ModelTable.footer(controller)?.text == "The catalog could not be read.")
        #expect(view.frame.height > before && view.frame.height == ModelTable.height(controller))
        refresh.mouseDown(with: down)
        #expect(calls == 3)

        // Off, with its reason on hover, while a Get or Load runs (the service would refuse it).
        var rows = Fixtures.rows(.loaded)
        rows[0].states[rows[0].precisions[0]] = PrecisionState(state: .loading)
        controller.apply(rows: rows, kinds: Fixtures.kinds)
        try await waitUntil { !refresh.enabled }
        #expect(refresh.toolTip == RealModelsService.refreshBusyMessage && !refresh.isAccessibilityEnabled())
        refresh.mouseDown(with: down)
        #expect(calls == 3)
        controller.apply(rows: Fixtures.rows(.loaded), kinds: Fixtures.kinds)
        try await waitUntil { refresh.enabled }
        #expect(refresh.toolTip == RefreshControl.help)
    }

    /// The most recent problem wins the footer line: a refusal replaces an older Load error; a later error replaces the
    /// refusal, and the menu's routine re-statement of the same refusal does not clear it.
    @Test func aRefusedRefreshReplacesAnOlderErrorInTheFooter() {
        // While a Get runs (the refusal's own situation: FLUX downloading).
        var rows = Fixtures.rows(.loaded)
        if let i = rows.firstIndex(where: { $0.id == "flux2-klein-4b" }) { rows[i].states["bf16"] = PrecisionState(state: .downloading, progress: 0.4) }
        let controller = TableController(service: FixtureService(rows: rows))
        controller.apply(rows: rows, kinds: Fixtures.kinds)
        let menu = ModelsMenu(controller: controller)
        _ = menu.modelItem()
        controller.lastError = "Load failed: out of memory."
        #expect(ModelTable.footer(controller)?.text == "Load failed: out of memory.")
        menu.updateRefresh(enabled: false, notice: nil)
        menu.updateRefresh(enabled: true, notice: "Finish Get or Load before refreshing models.")
        #expect(ModelTable.footer(controller)?.text == "Finish Get or Load before refreshing models.")
        controller.lastError = "Get failed: disk full."
        menu.updateRefresh(enabled: true, notice: "Finish Get or Load before refreshing models.")
        #expect(ModelTable.footer(controller)?.text == "Get failed: disk full.")
        // The Get ends: the busy refusal is stale and goes; a real problem stays.
        controller.lastError = nil
        #expect(ModelTable.footer(controller)?.text == "Finish Get or Load before refreshing models.")
        controller.apply(rows: Fixtures.rows(.loaded), kinds: Fixtures.kinds)
        #expect(controller.refreshAllowed && ModelTable.footer(controller) == nil)
        menu.updateRefresh(enabled: true, notice: "The catalog could not be read.")
        #expect(ModelTable.footer(controller)?.text == "The catalog could not be read.")
    }

    /// The settings switch's state reads in a window without the active appearance (a menu-bar app is not active while
    /// its menu is open; an NSSwitch draws "on" as a grey track there): the user's accent colour when on, a faint neutral
    /// track off. The accent is compared with the same colour drawn through the same offscreen path, whatever it is.
    @Test func settingsSwitchDrawsItsStateInAnInactiveWindow() throws {
        func pixel(_ view: NSView, at point: NSPoint) throws -> NSColor {
            let window = NSWindow(contentRect: NSRect(x: -9000, y: -9000, width: 44, height: 20), styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false; window.contentView = view
            defer { window.contentView = nil; window.close() }
            #expect(!window.isKeyWindow)
            let rep = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
            view.cacheDisplay(in: view.bounds, to: rep)
            let scale = CGFloat(rep.pixelsWide) / view.bounds.width
            return try #require(rep.colorAt(x: Int(point.x * scale), y: Int(point.y * scale))?.usingColorSpace(.sRGB))
        }
        func track(_ on: Bool) throws -> NSColor {
            let toggle = MenuSwitch(); toggle.state = on ? .on : .off
            // A track point away from the knob on either side.
            let t = toggle.track
            return try pixel(toggle, at: NSPoint(x: on ? t.minX + 4 : t.maxX - 4, y: t.midY))
        }
        final class Swatch: NSView {
            override func draw(_ dirtyRect: NSRect) { NSColor.controlAccentColor.setFill(); bounds.fill() }
        }
        let accent = try pixel(Swatch(frame: NSRect(origin: .zero, size: SettingControl.Switch.size)), at: NSPoint(x: 10, y: 10))
        let on = try track(true), off = try track(false)
        let distance = abs(on.redComponent - accent.redComponent) + abs(on.greenComponent - accent.greenComponent) + abs(on.blueComponent - accent.blueComponent)
        #expect(on.alphaComponent > 0.95 && distance < 0.06, "on \(on), accent \(accent)")
        #expect(off.alphaComponent < 0.5, "off \(off)")
        // Accessibility exposes the cell: a checkbox with the switch subrole and the setting's name.
        let toggle = MenuSwitch(); toggle.label = "Show Chat in menu"
        let exposed = try #require(NSAccessibility.unignoredDescendant(of: toggle) as? NSCell)
        #expect(exposed === toggle.cell && exposed.accessibilityLabel() == "Show Chat in menu")
        #expect(exposed.accessibilityRole() == .checkBox && exposed.accessibilitySubrole() == .switch)
    }

    /// Precision labels (Vella's TierControl): dark on the chosen cell's gold, the text colour on the others, greyed where
    /// a cell cannot be chosen; the track is neutral until a cell is chosen and lighter on the loaded row. Locked, the
    /// control draws at half strength except the chosen cell's fill (Vella 2.1.0), so the label colours stay as they are.
    @Test func precisionLabelsRankTheSelectedSegmentFirstEvenWhenLocked() {
        let control = TierSegments.Control()
        control.segmentCount = 3; control.selectedSegment = 0; control.unavailableSegments = [2]
        let c = TierSegments.Control.colors(hot: false, dark: true)
        #expect(control.labelColor(0, colors: c) == TierControl.ink && control.labelColor(1, colors: c) == c.label && control.labelColor(2, colors: c) == c.off)
        control.locked = true
        #expect(control.labelColor(0, colors: c) == TierControl.ink && control.labelColor(1, colors: c) == c.label && control.labelColor(2, colors: c) == c.off)
        #expect(c.selection == TierControl.gold && ArsenalPalette.gold == TierControl.gold)
        let hot = TierSegments.Control.colors(hot: true, dark: true)
        #expect(hot.selection == TierControl.gold && hot.selectedLabel == TierControl.ink && hot.trackFill != c.trackFill)
        #expect(TierSegments.Control.colors(hot: true, dark: false) == hot)
        let light = TierSegments.Control.colors(hot: false, dark: false)
        #expect(light.selection == TierControl.gold && light.label != c.label && light.trackFill != c.trackFill)
    }

    /// The switch's track ends at its frame's trailing edge, where the pop-ups, values and hairlines end.
    @Test func switchTrackIsFlushTrailing() {
        let toggle = MenuSwitch()
        #expect(abs(toggle.track.maxX - toggle.bounds.maxX) <= 0.5 && toggle.bounds.size == SettingControl.Switch.size)
    }

    /// The palette for content on the menu's material: lines, fills and the unavailable text step are overlays (they read
    /// on any material); nothing is an opaque slab.
    @Test func menuPaletteDrawsOverlaysOnTheMaterial() {
        for dark in [true, false] {
            let p = ArsenalPalette.menu(dark: dark)
            for color in [p.hairline, p.raisedBorder, p.field, p.hover, p.selFill, p.t4, p.raised] { #expect(color.alphaComponent < 0.5) }
            #expect(ActivityColours.other(p).alphaComponent > 3 * p.field.alphaComponent)
        }
    }

    /// Review renders at real width (2x pixels) on a stand-in for the menu material, and over a busy blurred wallpaper
    /// under the translucent material: the table loaded (hot rows, a red pending row), with a refused Refresh while a Get
    /// runs, and the Activity panel busy.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MENU_FIT_RENDERS"] != nil))
    func renderMenuFit() async throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_MENU_FIT_RENDERS"]!)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var rows = Fixtures.rows(.loaded)
        if let i = rows.firstIndex(where: { $0.id == "flux2-klein-4b" }) { rows[i].states["bf16"] = .init(state: .loaded) }
        let app = AppDelegate(service: FixtureService(rows: rows, supportDirectory: support), paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        // The refusal's own situation: a Get in progress (FLUX downloading) when Refresh was clicked.
        var getting = Fixtures.rows(.loaded)
        if let i = getting.firstIndex(where: { $0.id == "flux2-klein-4b" }) {
            getting[i].states["bf16"] = PrecisionState(state: .downloading, progress: 0.41, bytes: 6_550_000_000, totalBytes: 15_980_140_843)
        }
        // Through the service like the real table, so its rows carry settings (gears).
        let refused = TableController(service: FixtureService(rows: getting, supportDirectory: support.appendingPathComponent("refused")))
        await refused.refresh()
        let refusedMenu = ModelsMenu(controller: refused)
        _ = refusedMenu.modelItem()
        refusedMenu.updateRefresh(enabled: true, notice: "Finish Get or Load before refreshing models.")
        defer { app.table.stop(); refused.stop(); app.tools.closeAll(); Appearance.shared.set(.dark); NSApp.appearance = nil }
        let c = app.table
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            try await Task.sleep(for: .milliseconds(100))
            // Kokoro loaded with another precision clicked: the red pending row.
            if let kokoro = c.rows.first(where: { $0.id == "kokoro-82m" }) { c.select(kokoro, column: "8") }
            for wallpaper in [false, true] {
                let suffix = mode.rawValue + (wallpaper ? "-wallpaper" : "")
                await QA.renderTable(c, to: directory.appendingPathComponent("menu-fit-table-\(suffix).png"), wallpaper: wallpaper)
                app.activityMenu.activity.preview(V3RenderDelegate.busyFeed())
                await QA.renderActivity(app.activityMenu.activity, to: directory.appendingPathComponent("menu-fit-activity-\(suffix).png"), wallpaper: wallpaper)
            }
            await QA.renderTable(refused, to: directory.appendingPathComponent("menu-fit-table-refused-\(mode.rawValue).png"))
        }
    }
}
