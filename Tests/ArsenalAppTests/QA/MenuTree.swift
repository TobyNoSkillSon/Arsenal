// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal

// The main menu by task keeps the app's own switches under Settings and the results under Recent: tests reach an item
// wherever it lives.
extension NSMenu {
    /// Every item of this menu and of its submenus, depth first (hosted views' menus included).
    var everyItem: [NSMenuItem] { items.flatMap { [$0] + ($0.submenu?.everyItem ?? []) } }
    /// The first item titled `title` here or in any submenu.
    func item(titled title: String) -> NSMenuItem? { everyItem.first { $0.title == title } }
    /// The main menu's Settings submenu (Keep Loaded, Memory, the agents' voice, Mute Agent Speech, Launch at Login,
    /// Show Models Folder).
    var settings: NSMenu? { items.first { $0.title == "Settings" }?.submenu }
    /// The main menu's Models submenu (the hosted table).
    var models: NSMenu? { items.first { $0.title == "Models" }?.submenu }
    /// The main menu's Recent submenu, when there are results.
    var recent: NSMenu? { items.first { $0.title == "Recent" }?.submenu }
}
