// SPDX-License-Identifier: MIT
import AppKit

/// Retain surviving items and submenus. In particular never detach Models/Activity while AppKit
/// is tracking them: an empty native menu gets a provisional 10×10 viewer.
///
/// An open menu never changes its structure (13 Oct: Show in menu removed a main-menu item while Models was open, and
/// AppKit closed Models): with `structure` false, survivors update in place and nothing is inserted or removed. The
/// result says whether the menu now holds `desired` in full; the caller applies the rest once every menu has closed.
@MainActor enum MenuReconciliation {
    @discardableResult static func apply(_ desired: [NSMenuItem], to menu: NSMenu, structure: Bool = true, nested: Bool = false) -> Bool {
        let old = keyed(menu.items, nested: nested), next = keyed(desired, nested: nested)
        let wanted = Set(next.map(\.0)), had = Set(old.map(\.0))
        var complete = true
        if !structure, wanted != had {
            complete = false
        } else {
            // Remove only genuinely absent entries; surviving entries never see a remove/add cycle.
            for (key, item) in old where !wanted.contains(key) { menu.removeItem(item) }
        }
        let retained = Dictionary(uniqueKeysWithValues: old)
        // Keep the live relative order of survivors even if a catalog refresh reorders its snapshot.
        // AppKit has no attached-item move operation: remove/insert can tear down a tracked submenu.
        for (index, entry) in next.enumerated() {
            let (key, snapshot) = entry
            if let item = retained[key] {
                if item !== snapshot, !update(item, from: snapshot, structure: structure) { complete = false }
                continue
            }
            guard structure else { continue }
            // New entries precede their next surviving neighbour. Missing entries are the only inserts.
            let anchor = next.dropFirst(index + 1).compactMap { retained[$0.0] }.first
            let position = anchor.map { menu.index(of: $0) } ?? menu.numberOfItems
            menu.insertItem(snapshot, at: position)
        }
        return complete
    }

    private static func keyed(_ items: [NSMenuItem], nested: Bool) -> [(String, NSMenuItem)] {
        func base(_ item: NSMenuItem) -> String {
            if let pair = item.representedObject as? [String] { return "pair:" + pair.joined(separator: "/") }
            let selector = item.action.map(NSStringFromSelector) ?? ""
            if let id = item.representedObject as? String { return selector + ":" + id }
            if item.isSectionHeader { return "header:" + item.title }
            // AppKit gives every submenu's item the same submenu action. In the main menu a root (Settings, Recent) is
            // keyed by its identifier or title, or a Recent that appears would be matched to the old Settings by
            // position; inside a submenu an unidentified root keeps its place (Recent's results, newest first, are
            // retitled in place: reconciliation never reorders survivors).
            if item.submenu != nil, item.action == nil || item.action == #selector(NSMenu.submenuAction(_:)) {
                if let id = item.identifier?.rawValue { return "submenu:" + id }
                if !nested { return "submenu:" + item.title }
            }
            if !selector.isEmpty {
                return selector + ":" + String(describing: item.representedObject ?? "") + ":" + String(item.tag)
            }
            return "title:" + item.title
        }
        var counts: [String: Int] = [:]
        return items.enumerated().map { index, item in
            // A separator belongs to the following section, not its shifting numeric index.
            let key = item.isSeparatorItem ? "separator:" + (items.dropFirst(index + 1).first(where: { !$0.isSeparatorItem }).map(base) ?? "end") : base(item)
            let count = counts[key, default: 0]; counts[key] = count + 1
            return (key + ":" + String(count), item)
        }
    }

    /// Returns whether the item's submenu now holds the snapshot's in full.
    private static func update(_ item: NSMenuItem, from snapshot: NSMenuItem, structure: Bool) -> Bool {
        if item.title != snapshot.title {
            item.title = snapshot.title
            (item.view as? ProminentMenuItemView)?.refit()
        }
        if item.subtitle != snapshot.subtitle { item.subtitle = snapshot.subtitle }
        if item.isEnabled != snapshot.isEnabled { item.isEnabled = snapshot.isEnabled }
        if item.isHidden != snapshot.isHidden { item.isHidden = snapshot.isHidden }
        if item.state != snapshot.state { item.state = snapshot.state }
        if item.toolTip != snapshot.toolTip { item.toolTip = snapshot.toolTip }
        if item.attributedTitle != snapshot.attributedTitle { item.attributedTitle = snapshot.attributedTitle }
        // A task's trailing note ("42 %", "Failed") changes in place as a download moves.
        if item.badge?.stringValue != snapshot.badge?.stringValue { item.badge = snapshot.badge }
        if item.image != snapshot.image { item.image = snapshot.image; item.showMenuImage() }
        item.target = snapshot.target; item.action = snapshot.action
        item.representedObject = snapshot.representedObject
        if let current = item.submenu, let next = snapshot.submenu {
            // A hosted submenu owns its view and delegate; retain it without any mutation.
            if !current.items.contains(where: { $0.view is MenuTableHostingView || $0.view is ActivityHostingView }) {
                let items = next.items; next.removeAllItems()
                return apply(items, to: current, structure: structure, nested: true)
            }
        } else if item.submenu !== snapshot.submenu {
            // Attaching or detaching a submenu is structure too.
            guard structure else { return false }
            item.submenu = snapshot.submenu
        }
        return true
    }
}
