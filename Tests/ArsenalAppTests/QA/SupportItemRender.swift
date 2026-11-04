// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal

/// `ARSENAL_SUPPORT_RENDER=<dir>`: Support the Developer beside About and Quit as the status menu draws them, at 2× pixels,
/// dark and light, at rest and under the pointer (support-<appearance>-<state>.png). No window: the root view carries the
/// appearance and is drawn into a bitmap, so nothing appears on screen. The menu's material is a flat stand-in, so only
/// the row's own drawing is judged here.
@MainActor @Suite(.menuState, .serialized) struct SupportItemRender {
    nonisolated static var directory: URL? { ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_RENDER"].map { URL(fileURLWithPath: $0) } }

    @Test(.enabled(if: SupportItemRender.directory != nil)) func supportRow() async throws {
        _ = NSApplication.shared
        let directory = try #require(Self.directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (name, appearance, fill) in [
            ("dark", NSAppearance.Name.darkAqua, NSColor(calibratedWhite: 0.16, alpha: 1)), ("light", .aqua, NSColor(calibratedWhite: 0.96, alpha: 1))
        ] {
            for lit in [false, true] {
                let menu = NSMenu()
                let items = [("About Arsenal", "info.circle"), (AppDelegate.supportTitle, AppDelegate.supportSymbol), ("Quit Arsenal", "power")].map { title, icon in
                    let item = NSMenuItem(title: title, action: nil, keyEquivalent: ""); item.image = Symbol.menu(icon); menu.addItem(item); return item
                }
                let width: CGFloat = 250
                let root = Backdrop(frame: NSRect(x: 0, y: 0, width: width, height: 24 * 3 + 12))
                root.fill = fill; root.appearance = NSAppearance(named: appearance); root.autoresizesSubviews = false
                for (i, item) in items.enumerated() {
                    let view = ProminentMenuItemView(item: item)
                    if i == 1 { view.resting = true; view.forcesHighlight = lit; view.rendersSelectionFill = lit }
                    let row = NSView(frame: NSRect(x: 0, y: root.frame.height - 6 - CGFloat(i + 1) * 24, width: width, height: ProminentMenuItemView.height))
                    row.autoresizesSubviews = false
                    view.frame = row.bounds
                    row.addSubview(view)
                    root.addSubview(row)
                }
                root.layoutSubtreeIfNeeded()
                let rep = NSBitmapImageRep(
                    bitmapDataPlanes: nil, pixelsWide: Int(root.bounds.width * 2), pixelsHigh: Int(root.bounds.height * 2), bitsPerSample: 8, samplesPerPixel: 4,
                    hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
                rep.size = root.bounds.size
                root.cacheDisplay(in: root.bounds, to: rep)
                try rep.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("support-\(name)-\(lit ? "hover" : "rest").png"))
            }
        }
    }
}

/// The flat stand-in for the menu's material, drawn by the view itself so a windowless bitmap render includes it.
private final class Backdrop: NSView {
    var fill = NSColor.black
    override func draw(_ dirtyRect: NSRect) { fill.setFill(); bounds.fill() }
}
