// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MenuGeometryTests {
    /// At natural width the wrapper must draw exactly what a plain host draws, with no transform.
    @Test func naturalAllocationIsPixelIdenticalToPlainHost() throws {
        _ = NSApplication.shared
        let controller = RenderDelegate.controller(.init(name: "pixels", rows: Fixtures.rows(.loaded)))
        func compare<Content: View>(_ content: Content, size: NSSize, tracksContentSize: Bool = false) throws {
            let plain = NSHostingView(rootView: content)
            plain.sizingOptions = []
            let fitted = MenuContentView(rootView: content, tracksContentSize: tracksContentSize)
            fitted.setNaturalSize(size)
            func pixels(_ view: NSView) throws -> Data {
                let window = NSWindow(
                    contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: size),
                    styleMask: .borderless, backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false
                window.appearance = NSAppearance(named: .darkAqua)
                window.contentView = view
                defer { window.contentView = nil; window.close() }
                view.frame = NSRect(origin: .zero, size: size)
                view.layoutSubtreeIfNeeded()
                let rep = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
                view.cacheDisplay(in: view.bounds, to: rep)
                return Data(bytes: try #require(rep.bitmapData), count: rep.bytesPerRow * rep.pixelsHigh)
            }
            let baseline = try pixels(plain), candidate = try pixels(fitted)
            #expect(Set(baseline).count > 8, "The baseline must contain rendered content")
            #expect(fitted.contentScale == 1)
            #expect(candidate == baseline, "Natural allocation must preserve every pixel")
        }
        try compare(ModelsMenuView(controller: controller), size: NSSize(width: ModelTable.width(controller), height: ModelTable.height(controller)), tracksContentSize: true)
        let empty = TableController(service: FixtureService(rows: [], kinds: Fixtures.kinds))
        empty.apply(rows: [], kinds: Fixtures.kinds)
        try compare(ModelsMenuView(controller: empty), size: NSSize(width: ModelTable.width(empty), height: ModelTable.height(empty)), tracksContentSize: true)
        let activity = ActivityMenu(controller: controller)
        let panel = ActivityPanel(model: activity.activity)
        try compare(panel, size: NSHostingView(rootView: panel).fittingSize)
    }

    @Test func leftSubmenuLogsOriginalOverflowEveryFitAndParentWindowOverlap() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-left-menu-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support)
        let parentWindow = NSWindow(
            contentRect: NSRect(x: -8000, y: -9000, width: 240, height: 100), styleMask: .borderless, backing: .buffered, defer: false)
        let childWindow = VisibilityFixtureWindow(
            contentRect: NSRect(x: -8300, y: -9000, width: 300, height: 100), styleMask: .borderless, backing: .buffered, defer: false)
        parentWindow.isReleasedWhenClosed = false; childWindow.isReleasedWhenClosed = false
        let parent = GeometryFixtureMenu(), child = GeometryFixtureMenu()
        parent.fixtureWindow = parentWindow; child.fixtureWindow = childWindow
        parent.fixtureFrame = parentWindow.frame; child.fixtureFrame = childWindow.frame
        let item = NSMenuItem(); item.submenu = child; parent.addItem(item)
        let host = MenuContentView(rootView: Color.red.frame(width: 316, height: 100))
        host.setNaturalSize(NSSize(width: 316, height: 100))
        let container = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: 100))
        childWindow.contentView = container; container.addSubview(host)
        defer { childWindow.contentView = nil; childWindow.close(); parentWindow.close() }
        host.onOpenLayout = { view in log.record(menu: child, name: .submenu, view: view, content: view.hosting, scale: view.contentScale) }
        func entries() throws -> [[String: Any]] {
            try String(contentsOf: log.url, encoding: .utf8).split(separator: "\n").map {
                try #require(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
            }
        }
        // The actual open/visibility callback: no window is ordered and no host input is sent.
        childWindow.reportVisible = true
        NotificationCenter.default.post(name: NSWindow.didChangeOcclusionStateNotification, object: childWindow)
        for _ in 0..<5 { await Task.yield() }
        await log.flush()
        let first = try #require(entries().last)
        #expect(first["side"] as? String == "left")
        #expect(first["hostOutsideWindow"] as? Bool == true)
        #expect(first["outsideWindow"] as? Bool == false)
        #expect(try #require(first["scale"] as? Double) < 1)
        #expect(first["overlapsParentWindow"] as? Bool == false)
        let count = try entries().count
        host.needsLayout = true; host.layoutSubtreeIfNeeded()
        await log.flush()
        #expect(try entries().count == count + 1)
        parentWindow.setFrameOrigin(NSPoint(x: childWindow.frame.maxX - 16, y: childWindow.frame.minY))
        parent.fixtureFrame = parentWindow.frame
        host.needsLayout = true; host.layoutSubtreeIfNeeded()
        await log.flush()
        let overlap = try #require(entries().last)
        #expect(overlap["overlapsParentWindow"] as? Bool == true)
        #expect((overlap["parentWindowOverlap"] as? [Double])?[2] == 16)
        #expect(overlap["parentWindow"] as? [Double] == MenuDiagnostics.rect(parentWindow.frame).map(Double.init))
    }

    /// Exercise the actual NSMenuItemViewer without tracking/opening a menu. An ordinary host-only test misses this.
    @Test func bothHostedSubmenusFitNarrowNativeViewersAndWindowInsets() throws {
        _ = NSApplication.shared
        let controller = RenderDelegate.controller(.init(name: "fit", rows: Fixtures.rows(.loaded)))
        let empty = TableController(service: FixtureService(rows: [], kinds: Fixtures.kinds))
        empty.apply(rows: [], kinds: Fixtures.kinds)
        let models = ModelsMenu(controller: controller)
        let emptyModels = ModelsMenu(controller: empty)
        let activity = ActivityMenu(controller: controller)
        let menus = try [models.modelItem(), emptyModels.modelItem(), activity.item()].map { try #require($0.submenu) }
        print("menu fit widths: populated \(ModelTable.width(controller)) pt, empty \(ModelTable.width(empty)) pt")
        for (index, menu) in menus.enumerated() {
            let item = try #require(menu.items.first)
            let wrapper = try #require(item.view)
            _ = menu.size
            try #require(item.responds(to: NSSelectorFromString("_menuItemViewer")))
            let viewer = try #require(item.value(forKey: "_menuItemViewer") as? NSView)
            let natural = wrapper.frame.size
            let window = NSWindow(
                contentRect: NSRect(x: -9000, y: -9000, width: natural.width, height: natural.height),
                styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            let material = NSView(frame: NSRect(origin: .zero, size: natural))
            window.contentView = material
            material.addSubview(viewer)
            defer { viewer.removeFromSuperview(); window.contentView = nil; window.close() }
            for lostWidth: CGFloat in [0, 10, 16, 40, natural.width / 2] {
                viewer.frame = NSRect(x: 8, y: 0, width: natural.width - lostWidth - 8, height: natural.height)
                wrapper.needsLayout = true; wrapper.layoutSubtreeIfNeeded()
                let content = try #require(wrapper.subviews.first)
                let actual = content.convert(content.bounds, to: material)
                #expect(!MenuDiagnostics.overflows(actual, container: material.bounds))
                #expect(!MenuDiagnostics.overflows(content.convert(content.bounds, to: viewer), container: viewer.bounds))
                #expect(content.frame.width <= viewer.bounds.width + 0.5)
                if wrapper is MenuTableHostingView {
                    func actions(_ v: NSView) -> [RowActionView] {
                        (v as? RowActionView).map { [$0] } ?? v.subviews.flatMap(actions)
                    }
                    let controls = actions(content)
                    #expect(controls.count == (index == 0 ? controller.rows.count : 0))
                    for action in controls {
                        let point = action.convert(NSPoint(x: action.buttonRect.midX, y: action.buttonRect.midY), to: material)
                        let hit = material.hitTest(point)
                        #expect(hit === action || hit?.isDescendant(of: action) == true)
                        #expect(!MenuDiagnostics.overflows(action.convert(action.bounds, to: material), container: material.bounds))
                    }
                }
            }
        }
    }

    @Test func legacyFixedHostOverflowsTheNativeViewerByTheLostAllocation() throws {
        let controller = RenderDelegate.controller(.init(name: "legacy-fit", rows: Fixtures.rows(.loaded)))
        let host = NSHostingView(rootView: ModelsMenuView(controller: controller))
        let size = NSSize(width: ModelsMenuView.width(controller), height: ModelsMenuView.height(controller))
        host.frame = NSRect(origin: .zero, size: size)
        let menu = NSMenu(); let item = NSMenuItem(); item.view = host; menu.addItem(item)
        _ = menu.size
        try #require(item.responds(to: NSSelectorFromString("_menuItemViewer")))
        let viewer = try #require(item.value(forKey: "_menuItemViewer") as? NSView)
        let window = NSWindow(
            contentRect: NSRect(x: -9000, y: -9000, width: size.width - 40, height: size.height),
            styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.contentView = viewer
        defer { window.contentView = nil; window.close() }
        viewer.setFrameSize(NSSize(width: size.width - 40, height: size.height))
        host.layoutSubtreeIfNeeded()
        #expect(host.frame.width == size.width)
        #expect(MenuDiagnostics.overflows(host.convert(host.bounds, to: viewer), container: viewer.bounds))
        func actions(_ v: NSView) -> [RowActionView] {
            (v as? RowActionView).map { [$0] } ?? v.subviews.flatMap(actions)
        }
        let cells = actions(host)
        #expect(cells.count == controller.rows.count)
        // The action column runs past the viewer (its buttons are centred in their slot, so the narrow ones may not).
        #expect(cells.contains { MenuDiagnostics.overflows($0.convert($0.bounds, to: viewer), container: viewer.bounds) })
        print("legacy menu allocation: host \(size.width) pt, native viewer \(viewer.bounds.width) pt; action cells overflow")
    }

    @Test func geometryRecordsRealWindowCoordinatesClippingAndBoundedWholeLines() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-geometry-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support)
        let menu = NSMenu()
        let window = NSWindow(
            contentRect: NSRect(x: -9000, y: -9000, width: 200, height: 100),
            styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let parent = NSView(frame: NSRect(x: 0, y: 0, width: 200, height: 100))
        window.contentView = parent
        let content = NSView(frame: NSRect(x: 16, y: 0, width: 200, height: 100)); parent.addSubview(content)
        defer { window.contentView = nil; window.close() }
        log.record(menu: menu, name: .submenu, view: content, content: content)
        await log.flush()
        var lines = try String(contentsOf: log.url, encoding: .utf8).split(separator: "\n")
        let entry = try #require(JSONSerialization.jsonObject(with: Data(lines[0].utf8)) as? [String: Any])
        print("menu geometry fixture sample: \(lines[0])")
        #expect(entry["outsideWindow"] as? Bool == true)
        #expect(entry["content"] as? [Double] == [16, 0, 200, 100])
        content.setFrameSize(NSSize(width: 184, height: 100))
        log.record(menu: menu, name: .submenu, view: content, content: content)
        for _ in 0..<1000 { try log.append(["menu": "Fixture", "padding": String(repeating: "x", count: 128)]) }
        await log.flush()
        let data = try Data(contentsOf: log.url)
        #expect(data.count <= MenuDiagnostics.maximumBytes)
        lines = String(decoding: data, as: UTF8.self).split(separator: "\n")
        for line in lines { #expect(throws: Never.self) { try JSONSerialization.jsonObject(with: Data(line.utf8)) } }
        #expect(MenuDiagnostics.side(child: NSRect(x: 0, y: 0, width: 100, height: 100), parent: NSRect(x: 100, y: 0, width: 80, height: 100)) == "left")
        #expect(MenuDiagnostics.side(child: .zero, parent: .zero) == "unavailable")
    }

    @Test func openingAndVisibilityEventsRecordWithoutAnIdleTimerOrAVisibleTestWindow() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-event-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support)
        let menu = NSMenu(); let root = NSMenuItem(title: "Memory", action: nil, keyEquivalent: "")
        root.identifier = MenuDiagnosticName.memory.itemIdentifier
        root.submenu = NSMenu(); menu.addItem(root)
        log.observeSubmenus(in: menu)
        let sub = try #require(root.submenu)
        #expect((sub.delegate as? MenuTrackingDelegate)?.recorder === log.trackingRecorder)
        sub.delegate?.menuWillOpen?(sub)
        for _ in 0..<5 { await Task.yield() }
        await log.flush()
        let text = try String(contentsOf: log.url, encoding: .utf8)
        #expect(text.contains("Memory") && text.contains("unavailable"))
        let host = MenuContentView(rootView: Color.clear)
        host.setNaturalSize(NSSize(width: 200, height: 100))
        let window = VisibilityFixtureWindow(
            contentRect: NSRect(x: -9000, y: -9000, width: 200, height: 100),
            styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        var recordings = 0
        host.onOpenLayout = { _ in recordings += 1 }
        window.contentView = host
        defer { window.contentView = nil; window.close() }
        for _ in 0..<5 { await Task.yield() }
        #expect(recordings == 0)
        window.reportVisible = true // Simulates notification state; the window is never ordered on screen.
        NotificationCenter.default.post(name: NSWindow.didChangeOcclusionStateNotification, object: window)
        for _ in 0..<5 { await Task.yield() }
        #expect(recordings == 1)
    }

    /// AppKit's effective image, not image assignment or preferred policy alone. No global defaults are mutated.
    @Test func imagePolicyIsVisibleAndAppKitResolvesBothVisibilityValues() throws {
        let app = NSApplication.shared
        _ = app
        let menu = LaunchDelegate.startupMenu()
        let item = try #require(menu.items.last)
        #expect(item.image != nil)
        if #available(macOS 27.0, *) {
            #expect(item.menuImagePreference == 1)
            for raw in [2, 1, 2, 1] {
                item.setValue(raw, forKey: "preferredImageVisibility")
                #expect(item.applicableMenuImage as? Bool == (raw == 1))
                item.showMenuImage()
                #expect(item.menuImagePreference == 1)
            }
        }
    }
}

// AppKit's accessibility overrides are nonisolated; this fixture is used only by the serialized main-actor suite.
@MainActor private final class GeometryFixtureMenu: NSMenu {
    nonisolated(unsafe) var fixtureWindow: NSWindow?
    nonisolated(unsafe) var fixtureFrame = NSRect.zero
    override func accessibilityWindow() -> Any? { fixtureWindow }
    override func accessibilityFrame() -> NSRect { fixtureFrame }
}

@MainActor private final class VisibilityFixtureWindow: NSWindow {
    var reportVisible = false
    override var isVisible: Bool { reportVisible }
}
