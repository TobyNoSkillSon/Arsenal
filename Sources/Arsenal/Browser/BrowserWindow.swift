// SPDX-License-Identifier: MIT
import AppKit
import Combine
import ArsenalCore
import SwiftUI

// "Get more models": a normal, resizable, non-modal window with no visible title bar. The
// column header sits in the title bar's band, level with the traffic lights; filters live in a collapsible sidebar;
// nothing reflows when a filter changes. One instance; reopening brings it front and keeps filters and expanded rows
// for the session. Frame, sidebar width/visibility and column weights persist.

@MainActor final class BrowserWindowController: NSWindowController, NSWindowDelegate, NSSearchFieldDelegate {
    let controller: BrowserController
    let list = BrowserList()
    let sidebarSearch = NSSearchField(), toolbarSearch = NSSearchField()
    private var root: BrowserRootView!
    private var observers: Set<AnyCancellable> = []
    static let title = "Get more models"
    static let defaultSize = NSSize(width: 1440, height: 820), minSize = NSSize(width: 900, height: 520)

    init(controller: BrowserController) {
        self.controller = controller
        let window = BrowserWindow(
            contentRect: NSRect(origin: .zero, size: Self.defaultSize),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        window.title = Self.title
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.minSize = Self.minSize
        window.isReleasedWhenClosed = false
        super.init(window: window)
        window.delegate = self
        for field in [sidebarSearch, toolbarSearch] { field.placeholderString = "Search"; field.delegate = self; field.sendsSearchStringImmediately = true }
        list.controller = controller
        list.focusSearch = { [weak self] in self?.focusSearch() }
        list.closeWindow = { [weak self] in self?.close() }
        root = BrowserRootView(controller: controller, list: list, sidebarSearch: sidebarSearch, toolbarSearch: toolbarSearch)
        window.contentView = root
        window.onCommand = { [weak self] key, control in self?.command(key, control: control) ?? false }
        window.center()
        window.setFrameAutosaveName("GetMoreModels")
        controller.$shown.sink { [weak self] rows in
            self?.list.show(rows); self?.root.updateEmpty(rows.isEmpty)
        }.store(in: &observers)
        controller.$state.map(\.sort).removeDuplicates().sink { [weak self] sort in self?.list.indicate(sort) }.store(in: &observers)
        controller.$expanded.removeDuplicates().dropFirst().sink { [weak self] _ in DispatchQueue.main.async { self?.list.expansionChanged() } }.store(in: &observers)
        controller.$state.map(\.search).removeDuplicates().sink { [weak self] text in
            for field in [self?.sidebarSearch, self?.toolbarSearch].compactMap({ $0 }) where field.stringValue != text { field.stringValue = text }
        }.store(in: &observers)
        Appearance.shared.$mode.sink { [weak self] _ in DispatchQueue.main.async { self?.applyAppearance() } }.store(in: &observers)
        applyAppearance()
    }
    required init?(coder: NSCoder) { nil }

    /// Brings the window front (a menu-bar app has no other window to lean on); focus in the search field.
    func show() {
        controller.start()
        showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
        focusSearch()
    }
    func windowWillClose(_ notification: Notification) { controller.stop() }
    func focusSearch() { window?.makeFirstResponder(root.sidebarOpen ? sidebarSearch : toolbarSearch) }

    func applyAppearance() {
        guard let window else { return }
        let a = Appearance.shared, p = a.palette
        window.appearance = a.nsAppearance
        window.isOpaque = !p.translucent
        window.backgroundColor = p.translucent ? .clear : p.canvas
        root.applyPalette()
        list.redraw()
    }

    /// ⌘F search, ⌘W close, ⌃⌘S the sidebar, ⌘1 All and ⌘2… the kinds in order.
    private func command(_ key: String, control: Bool) -> Bool {
        if control { if key == "s" { root.toggleSidebar(); return true }; return false }
        switch key {
        case "f": focusSearch(); return true
        case "w": close(); return true
        default:
            guard let n = Int(key), n >= 1 else { return false }
            if n == 1 { controller.state.kind = nil; return true }
            let kinds = controller.kindRows
            guard n - 2 < kinds.count else { return false }
            controller.state.kind = kinds[n - 2].kind.id
            return true
        }
    }

    // MARK: Search: live, ↓ moves to the list, Esc clears then closes

    func controlTextDidChange(_ obj: Notification) {
        guard let field = obj.object as? NSSearchField else { return }
        controller.state.search = field.stringValue
    }
    func control(_ control: NSControl, textView: NSTextView, doCommandBy selector: Selector) -> Bool {
        switch selector {
        case #selector(NSResponder.moveDown(_:)):
            list.focusFirstRow(); return true
        case #selector(NSResponder.cancelOperation(_:)):
            if controller.state.search.isEmpty { close() } else { controller.state.search = "" }
            return true
        default: return false
        }
    }
}

/// Command keys reach the window even though a menu-bar app has no main menu to carry them.
final class BrowserWindow: NSWindow {
    var onCommand: (String, Bool) -> Bool = { _, _ in false }
    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if let key = event.charactersIgnoringModifiers?.lowercased() {
            if flags == .command, onCommand(key, false) { return true }
            if flags == [.command, .control] || flags == [.command, .option], onCommand(key == "ß" ? "s" : key, true) { return true }
        }
        return super.performKeyEquivalent(with: event)
    }
}

/// A plain colour fill (palette surfaces over the vibrant views in translucent mode).
final class FillView: NSView {
    var color: NSColor = .clear { didSet { layer?.backgroundColor = color.cgColor } }
    override init(frame: NSRect) { super.init(frame: frame); wantsLayer = true; layer?.backgroundColor = color.cgColor }
    required init?(coder: NSCoder) { nil }
    override func viewDidChangeEffectiveAppearance() { layer?.backgroundColor = color.cgColor }
    override var mouseDownCanMoveWindow: Bool { true }
}

/// The sidebar's trailing edge: drag to resize (208–300 pt).
final class DividerHandle: NSView {
    var dragged: (CGFloat) -> Void = { _ in }
    override func resetCursorRects() { addCursorRect(bounds, cursor: .resizeLeftRight) }
    override func mouseDragged(with event: NSEvent) { dragged(superview?.convert(event.locationInWindow, from: nil).x ?? 0) }
}

/// The window's content, laid out by hand: sidebar | list (header in the top band), or, collapsed, a slim toolbar in
/// the band with the header below it.
@MainActor final class BrowserRootView: NSView {
    let controller: BrowserController
    let list: BrowserList
    private let sidebarEffect = NSVisualEffectView(), contentEffect = NSVisualEffectView()
    private let sidebarFill = FillView(), contentFill = FillView(), divider = FillView(), toolbarRule = FillView()
    private let handle = DividerHandle()
    /// Hide/show sidebar: one AppKit button above both SwiftUI hosts. The sidebar's SwiftUI ScrollView backs itself with
    /// an NSScrollView that extends under the band and took every click meant for a SwiftUI button there (v2.2 item 1).
    let sidebarButton = NSButton()
    static let buttonSize: CGFloat = 28
    private var sidebarHost: NSHostingView<BrowserSidebar>!
    private var toolbarHost: NSHostingView<CollapsedToolbar>!
    private var emptyHost: NSHostingView<NoResultsView>!
    private let sidebarSearch: NSSearchField, toolbarSearch: NSSearchField
    static let minContent: CGFloat = 700
    static let openKey = "BrowserSidebarOpen", widthKey = "BrowserSidebarWidth"
    private(set) var sidebarOpen = UserDefaults.standard.object(forKey: openKey) as? Bool ?? true
    private var sidebarWidth = min(
        BrowserSidebar.maxWidth, max(BrowserSidebar.minWidth, CGFloat(UserDefaults.standard.object(forKey: widthKey) as? Double ?? Double(BrowserSidebar.width))))
    /// What the hosted SwiftUI roots were last built with: layout runs on every frame of a live resize, and rebuilding
    /// the sidebar's root each time re-renders all of it.
    private var sidebarBand: CGFloat?, toolbarLeading: CGFloat?
    override var isFlipped: Bool { true }

    init(controller: BrowserController, list: BrowserList, sidebarSearch: NSSearchField, toolbarSearch: NSSearchField) {
        self.controller = controller; self.list = list; self.sidebarSearch = sidebarSearch; self.toolbarSearch = toolbarSearch
        super.init(frame: NSRect(origin: .zero, size: BrowserWindowController.defaultSize))
        sidebarEffect.material = .sidebar; contentEffect.material = .underWindowBackground
        for e in [sidebarEffect, contentEffect] { e.blendingMode = .behindWindow; e.state = .active }
        sidebarHost = NSHostingView(rootView: BrowserSidebar(controller: controller, search: sidebarSearch, band: 28))
        toolbarHost = NSHostingView(rootView: CollapsedToolbar(controller: controller, search: toolbarSearch, leading: 80, expand: { [weak self] in self?.toggleSidebar() }))
        emptyHost = NSHostingView(rootView: NoResultsView(controller: controller))
        for v: NSView in [sidebarEffect, contentEffect, sidebarFill, contentFill, list.scrollView, toolbarHost, toolbarRule, emptyHost, sidebarHost, divider, handle, sidebarButton]
        {
            addSubview(v)
        }
        sidebarButton.isBordered = false
        sidebarButton.imagePosition = .imageOnly
        sidebarButton.image = Symbol.image("sidebar.left", size: 14)
        sidebarButton.target = self
        sidebarButton.action = #selector(sidebarButtonClicked)
        handle.dragged = { [weak self] x in
            guard let self else { return }
            self.sidebarWidth = min(BrowserSidebar.maxWidth, max(BrowserSidebar.minWidth, x)); UserDefaults.standard.set(Double(self.sidebarWidth), forKey: Self.widthKey)
            self.needsLayout = true
        }
        emptyHost.isHidden = true
    }
    required init?(coder: NSCoder) { nil }

    /// The title bar's height: the band the traffic lights are centred in.
    var band: CGFloat {
        guard let window else { return 28 }
        return max(28, window.frame.height - window.contentLayoutRect.height)
    }
    /// Where the collapsed toolbar's content starts: after the traffic lights.
    private var trafficLightsEnd: CGFloat {
        guard let zoom = window?.standardWindowButton(.zoomButton), let superview = zoom.superview else { return 80 }
        return convert(superview.convert(zoom.frame, to: nil), from: nil).maxX + 14
    }

    @objc private func sidebarButtonClicked() { toggleSidebar() }
    func toggleSidebar() {
        sidebarOpen.toggle()
        if sidebarOpen, bounds.width - sidebarWidth < Self.minContent { sidebarOpen = false }
        UserDefaults.standard.set(sidebarOpen, forKey: Self.openKey)
        needsLayout = true
        window?.makeFirstResponder(sidebarOpen ? sidebarSearch : toolbarSearch)
    }

    func applyPalette() {
        let p = Appearance.shared.palette
        sidebarEffect.isHidden = !p.translucent; contentEffect.isHidden = !p.translucent
        sidebarFill.color = p.sidebar; contentFill.color = p.canvas
        divider.color = p.hairline; toolbarRule.color = p.hairline
        sidebarButton.contentTintColor = p.t2
        appearance = Appearance.shared.nsAppearance
        needsLayout = true
    }
    func updateEmpty(_ empty: Bool) { emptyHost.isHidden = !(empty && !controller.items.isEmpty) }

    override func layout() {
        super.layout()
        let width = bounds.width, height = bounds.height, band = self.band
        // Narrowing the window below sidebar + 700 collapses the sidebar; widening does not reopen it. A pass before the
        // window has a size (width 0, seen when a second window is made) is not a narrowing.
        if sidebarOpen, width > 0, width - sidebarWidth < Self.minContent { sidebarOpen = false; UserDefaults.standard.set(false, forKey: Self.openKey) }
        if sidebarBand != band {
            sidebarBand = band
            sidebarHost.rootView = BrowserSidebar(controller: controller, search: sidebarSearch, band: band)
        }
        for v: NSView in [sidebarHost, sidebarEffect, sidebarFill, divider, handle] { v.isHidden = !sidebarOpen }
        sidebarEffect.isHidden = !sidebarOpen || !Appearance.shared.palette.translucent
        toolbarHost.isHidden = sidebarOpen; toolbarRule.isHidden = sidebarOpen
        let x0: CGFloat
        if sidebarOpen {
            let sw = sidebarWidth
            for v: NSView in [sidebarEffect, sidebarFill, sidebarHost] { v.frame = NSRect(x: 0, y: 0, width: sw, height: height) }
            divider.frame = NSRect(x: sw, y: 0, width: 1, height: height)
            handle.frame = NSRect(x: sw - 3, y: band, width: 7, height: height - band)
            x0 = sw + 1
            list.setHeaderHeight(band)
            list.scrollView.frame = NSRect(x: x0, y: 0, width: width - x0, height: height)
            sidebarButton.frame = NSRect(x: sw - 12 - Self.buttonSize, y: (band - Self.buttonSize) / 2, width: Self.buttonSize, height: Self.buttonSize)
        } else {
            x0 = 0
            sidebarButton.frame = NSRect(x: trafficLightsEnd - 7, y: (band - Self.buttonSize) / 2, width: Self.buttonSize, height: Self.buttonSize)
            let leading = trafficLightsEnd - 7 + Self.buttonSize + 5
            if toolbarLeading != leading {
                toolbarLeading = leading
                toolbarHost.rootView = CollapsedToolbar(controller: controller, search: toolbarSearch, leading: leading, expand: { [weak self] in self?.toggleSidebar() })
            }
            toolbarHost.frame = NSRect(x: 0, y: 0, width: width, height: band)
            toolbarRule.frame = NSRect(x: 0, y: band, width: width, height: 1)
            list.setHeaderHeight(30)
            list.scrollView.frame = NSRect(x: 0, y: band + 1, width: width, height: height - band - 1)
        }
        for v: NSView in [contentEffect, contentFill] { v.frame = NSRect(x: x0, y: 0, width: width - x0, height: height) }
        contentEffect.isHidden = !Appearance.shared.palette.translucent
        emptyHost.frame = NSRect(x: x0, y: band + 40, width: width - x0, height: 80)
        sidebarButton.toolTip = sidebarOpen ? "Hide sidebar" : "Show sidebar"
        sidebarButton.setAccessibilityLabel(sidebarButton.toolTip)
        list.applyLayout()
    }
}

/// The collapsed sidebar's toolbar in the band, after the root view's show-sidebar button: search, the filter summary,
/// the count.
struct CollapsedToolbar: View {
    @ObservedObject var controller: BrowserController
    @ObservedObject var appearance = Appearance.shared
    let search: NSSearchField
    let leading: CGFloat
    let expand: () -> Void
    private var p: ArsenalPalette { appearance.palette }
    var body: some View {
        HStack(spacing: 12) {
            SearchFieldHost(field: search, palette: p).frame(width: 220, height: 24)
            let summary = controller.toolbarSummary
            if summary.error {
                // A refused action replaces the summary until the next action or a click on it.
                Button {
                    controller.dismissError()
                } label: {
                    Text(summary.text).font(.system(size: 12)).foregroundStyle(p.red.swiftUI).lineLimit(1).truncationMode(.middle)
                }
                .buttonStyle(.plain).help((summary.help ?? summary.text) + "\nClick to dismiss.")
            } else {
                Button(action: expand) { Text(summary.text).font(.system(size: 12)).foregroundStyle(p.t3.swiftUI).lineLimit(1) }.buttonStyle(.plain)
            }
            Spacer(minLength: 8)
            Text(controller.count).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t3.swiftUI).lineLimit(1)
        }
        .padding(.leading, leading).padding(.trailing, 16)
        .frame(maxHeight: .infinity)
        .contentShape(Rectangle()).gesture(WindowDragGesture())
        .ignoresSafeArea()
        .environment(\.colorScheme, appearance.colorScheme)
    }
}

struct NoResultsView: View {
    @ObservedObject var controller: BrowserController
    @ObservedObject var appearance = Appearance.shared
    var body: some View {
        VStack(spacing: 6) {
            Text(controller.noResults).font(.system(size: 13)).foregroundStyle(appearance.palette.t2.swiftUI)
            if controller.state.filtering {
                Button("Clear filters") { controller.state.clearFilters() }.buttonStyle(.plain)
                    .font(.system(size: 12)).foregroundStyle(appearance.palette.accentText.swiftUI)
            }
        }.frame(maxWidth: .infinity)
    }
}
