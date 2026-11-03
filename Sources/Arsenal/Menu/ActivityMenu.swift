// SPDX-License-Identifier: MIT
import AppKit
import Combine
import ArsenalCore
import SwiftUI

/// The Activity… item: its submenu holds the activity panel alone, at a fixed width where nothing is cut, as tall as
/// what it shows. The panel subscribes while this submenu is open and stops when it closes.
@MainActor final class ActivityMenu: NSObject, NSMenuDelegate {
    let activity = ActivityModel()
    private weak var panelMenu: NSMenu?
    private weak var panelView: ActivityHostingView?
    var diagnostics: MenuDiagnostics?
    private var naming: AnyCancellable?
    private var sizing: AnyCancellable?
    private var appearanceObservation: AnyCancellable?

    init(controller: TableController) {
        super.init()
        appearanceObservation = Appearance.shared.$mode.removeDuplicates().sink { [weak self] _ in
            DispatchQueue.main.async { self?.refreshAppearance() }
        }
        naming = controller.$rows.sink { [weak self] rows in
            self?.activity.rows = Dictionary(rows.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        }
        // A card or one of its lines arriving or leaving changes the panel's height: the item view follows, the width
        // never changes.
        sizing = activity.$feed.map { [activity] in activity.layout($0) }.removeDuplicates().sink { [weak self] _ in
            DispatchQueue.main.async { self?.resize() }
        }
    }

    func menuWillOpen(_ menu: NSMenu) {
        activity.show()
        refreshAppearance()
        resize()
        panelView?.recordAfterLayout()
    }
    func menuDidClose(_ menu: NSMenu) { activity.hide() }

    private func refreshAppearance() {
        panelMenu?.appearance = Appearance.shared.nsAppearance
        panelView?.appearance = Appearance.shared.nsAppearance
    }

    /// The panel's natural height for what it lists now.
    func resize() {
        guard let view = panelView else { return }
        let size = NSSize(width: ActivityPanel.width, height: Self.height(view))
        view.setNaturalSize(size)
        view.recordAfterLayout()
    }
    static func height(_ view: NSView) -> CGFloat {
        view.layoutSubtreeIfNeeded()
        return min(ActivityPanel.maxHeight, max(ActivityPanel.minHeight, ((view as? ActivityHostingView)?.contentFittingSize().height ?? view.fittingSize.height).rounded(.up)))
    }

    /// The Activity item, made once (the status menu is rebuilt on every change; the panel's view is kept).
    private var root: NSMenuItem?
    func item() -> NSMenuItem {
        if let root { return root }
        let root = NSMenuItem(title: "Activity", action: nil, keyEquivalent: "")
        self.root = root
        root.image = NSImage(systemSymbolName: "gauge.with.dots.needle.50percent", accessibilityDescription: nil)
        root.showMenuImage()
        root.toolTip = "What is running now, for whom, and how busy the GPU is."
        let menu = NSMenu(); menu.autoenablesItems = false; menu.delegate = self; panelMenu = menu
        let item = NSMenuItem()
        let view = ActivityHostingView(rootView: ActivityPanel(model: activity))
        view.setNaturalSize(NSSize(width: ActivityPanel.width, height: ActivityPanel.minHeight))
        view.onOpenLayout = { [weak self, weak menu] view in
            guard let menu else { return }
            self?.diagnostics?.record(menu: menu, name: .activity, view: view, content: view.hosting, scale: view.contentScale)
        }
        menu.minimumWidth = ActivityPanel.width
        item.view = view; menu.addItem(item); root.submenu = menu
        panelView = view
        refreshAppearance()
        // The empty panel's text can exceed minHeight at this menu scale. Its first allocation must fit the same
        // content as subsequent opens, rather than expanding after the first frame's asynchronous sizing callback.
        resize()
        return root
    }
}

/// A transparent, vibrant host for the panel inside the menu's own material.
final class ActivityHostingView: MenuContentView<ActivityPanel> {
    override var allowsVibrancy: Bool { true }
}
