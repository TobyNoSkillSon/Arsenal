// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

/// A model's settings card for tests and renders: the Models menu's real hosted table in AppKit's own item viewer, in an
/// unordered borderless window far off every screen, sized as the menu window would be; the card opens through the gear's
/// real path (`TableController.openSettings`) and is drawn by the table itself. Nothing is ordered on screen, activated,
/// focused or typed outside this process's windows.
@MainActor final class SettingsCardHost {
    let controller: TableController
    let models: ModelsMenu
    let menu: NSMenu
    let view: MenuTableHostingView
    private(set) var window: NSWindow?

    /// The host each controller's card is attached to, for tests that close it by controller.
    private static var live: [ObjectIdentifier: SettingsCardHost] = [:]
    /// The controller's host, made on first use.
    static func of(_ controller: TableController) -> SettingsCardHost { live[ObjectIdentifier(controller)] ?? SettingsCardHost(controller) }
    /// Closes the controller's card and releases its fixture window.
    static func close(_ controller: TableController) {
        if let host = live[ObjectIdentifier(controller)] { host.close() } else { controller.settingsCard.close() }
    }

    init(_ controller: TableController, models: ModelsMenu? = nil) {
        self.controller = controller
        let models = models ?? ModelsMenu(controller: controller)
        self.models = models
        // The Models item always holds one submenu with the hosted table (ModelsMenu.modelItem).
        guard let menu = models.modelItem().submenu, let view = menu.items.first?.view as? MenuTableHostingView else {
            preconditionFailure("the Models item holds the hosted table")
        }
        self.menu = menu
        self.view = view
        Self.live[ObjectIdentifier(controller)] = self
        attach()
    }
    var card: SettingsCard { controller.settingsCard }
    var state: SettingsState { controller.settingsCard.state }
    var isOpen: Bool { controller.settingsCard.openID != nil }
    /// The open card's frame in the table (top-left origin), zero when none is open.
    var frame: CGRect { card.frame(controller) ?? .zero }
    var height: CGFloat { frame.height }

    /// Opens the row's card as its gear does and lets it settle.
    func open(_ row: ModelRow, from anchor: NSView? = nil) async {
        controller.openSettings(row, from: anchor ?? gear(row.id))
        await settle()
    }
    func gear(_ id: String) -> GearView? { controller.settingsCard.gear(id) }

    /// The table's item view in AppKit's own item viewer, in an unordered window far off every screen.
    func attach() {
        _ = menu.size
        let item = menu.items.first!
        let viewer = (item.value(forKey: "_menuItemViewer") as? NSView) ?? view
        if window == nil {
            let window = NSWindow(
                contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: view.naturalSize), styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.backgroundColor = .clear; window.isOpaque = false
            self.window = window
        }
        window?.appearance = Appearance.shared.nsAppearance
        if window?.contentView !== viewer { window?.contentView = viewer }
        allocate()
    }
    /// Mirrors the menu window's allocation: the item view's natural size.
    func allocate() {
        guard let window, let viewer = window.contentView else { return }
        models.resize()
        let size = view.naturalSize
        if window.contentLayoutRect.size != size { window.setContentSize(size) }
        if viewer.frame.size != size { viewer.frame = NSRect(origin: .zero, size: size) }
        if viewer !== view, view.frame.size != size { view.frame = NSRect(origin: .zero, size: size) }
        view.needsLayout = true; view.layoutSubtreeIfNeeded()
    }

    /// SwiftUI updates, the tracking-mode run-loop callbacks and layout, as in menu tracking.
    func settle() async {
        for _ in 0..<16 {
            await Task.yield()
            Self.pump()
            allocate()
        }
        window?.displayIfNeeded()
    }
    /// One short turn of the main run loop (tracking-mode callbacks included), outside the async context.
    static func pump() { RunLoop.main.run(until: Date(timeIntervalSinceNow: 1.0 / 240)) }

    /// Views of a type inside the open card (by frame: the card's controls are the views whose frames lie in it).
    func views<T: NSView>(_ type: T.Type) -> [T] {
        let card = frame
        return Self.views(type, view).filter { v in
            let r = v.convert(v.bounds, to: view.hosting)
            let s = max(view.contentScale, 0.01)
            return !card.isEmpty && card.insetBy(dx: -0.5, dy: -0.5).contains(CGRect(x: r.minX / s, y: r.minY / s, width: r.width / s, height: r.height / s))
        }
    }
    static func views<T: NSView>(_ type: T.Type, _ root: NSView) -> [T] {
        (root as? T).map { [$0] } ?? root.subviews.flatMap { views(type, $0) }
    }

    /// Scrolls a view below or above the table's fold into sight, as a user scrolls a long table before clicking its
    /// gear, and lets the table settle. A view already in sight stays where it is.
    func reveal(_ view: NSView) async {
        view.scrollToVisible(view.bounds)
        await settle()
    }
    /// Scrolls a long table to a row, as a user does before using it: a scrolling table builds only the rows in sight
    /// (ModelTable's lazy stack), so a row below the fold has no views until then. Its place comes from the table's
    /// metrics; a table that does not scroll is left alone.
    func reveal(row id: String) async {
        // The table's own scroll view: the one holding the rows (an open card can hold one of its own).
        guard ModelTable.scrolls(controller), let rect = ModelTable.gearRect(controller, id: id),
            let scroll = Self.views(NSScrollView.self, view).first(where: { !Self.views(RowActionView.self, $0).isEmpty })
        else { return }
        let chrome = ModelTable.verticalPadding + ModelTable.headerHeight + 1 + 2 * ModelTable.dividerPadding
        let clip = scroll.contentView
        let top = max(0, rect.minY - chrome - (clip.bounds.height - rect.height) / 2)
        let limit = max(0, (scroll.documentView?.frame.height ?? 0) - clip.bounds.height)
        clip.scroll(to: NSPoint(x: clip.bounds.minX, y: min(top, limit)))
        scroll.reflectScrolledClipView(clip)
        await settle()
    }
    /// The mouse event a click at the view's centre makes, in the fixture window.
    func event(_ type: NSEvent.EventType, at view: NSView, number: Int) throws -> NSEvent {
        let window = try #require(self.window)
        let location = view.convert(NSPoint(x: view.bounds.midX, y: view.bounds.midY), to: nil)
        return try #require(
            NSEvent.mouseEvent(
                with: type, location: location, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
                context: nil, eventNumber: number, clickCount: 1, pressure: 1))
    }
    /// A click through AppKit's hit test at the view's centre, as the menu's tracking delivers it: the table host's hit
    /// test sees the mouse-down (it closes the card for a click elsewhere), the view it finds gets the mouseDown and then
    /// the mouseUp. Returns the view the click reached.
    private static var clicks = 100
    @discardableResult func press(_ view: NSView) throws -> NSView {
        let window = try #require(self.window)
        Self.clicks += 1
        let down = try event(.leftMouseDown, at: view, number: Self.clicks)
        let root = try #require(window.contentView?.superview)
        let previous = self.view.currentEvent
        self.view.currentEvent = { down }
        defer { self.view.currentEvent = previous }
        let target = try #require(root.hitTest(root.convert(down.locationInWindow, from: nil)))
        #expect(target === view || target.isDescendant(of: view), "the click reaches \(type(of: target)), not \(type(of: view))")
        target.mouseDown(with: down)
        target.mouseUp(with: try event(.leftMouseUp, at: view, number: Self.clicks))
        return target
    }

    /// The table's pixels (the card included) after it settled.
    func render() async throws -> Data {
        await settle()
        let rep = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: rep)
        return Data(bytes: try #require(rep.bitmapData), count: rep.bytesPerRow * rep.pixelsHigh)
    }

    /// Closes the card and releases the fixture window.
    func close() {
        controller.settingsCard.close()
        if Self.live[ObjectIdentifier(controller)] === self { Self.live[ObjectIdentifier(controller)] = nil }
        window?.contentView = nil; window?.close(); window = nil
    }
}
