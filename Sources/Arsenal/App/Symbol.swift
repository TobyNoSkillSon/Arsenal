// SPDX-License-Identifier: MIT
import AppKit

/// SF Symbols as the app's AppKit views draw them: looked up and configured once per name, size, weight and colour,
/// then reused (the table's cells redraw on every hover and progress tick, the browser's on every filter).
@MainActor enum Symbol {
    private struct Key: Hashable { var name: String; var size: CGFloat; var weight: NSFont.Weight; var color: NSColor?; var knockout = false }
    private static var cache: [Key: NSImage] = [:]

    /// A status-menu item's symbol, one image per name: the menu is rebuilt on every table refresh, and reconciliation
    /// leaves an item alone only when its image is the same object (a new image makes AppKit re-measure the item).
    private static var menuImages: [String: NSImage] = [:]
    static func menu(_ name: String) -> NSImage? {
        if let image = menuImages[name] { return image }
        let image = NSImage(systemSymbolName: name, accessibilityDescription: nil)
        menuImages[name] = image
        return image
    }

    /// `color` nil: a template image (an NSImageView tints it).
    static func image(_ name: String, size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor? = nil) -> NSImage? {
        let key = Key(name: name, size: size, weight: weight, color: color)
        if let image = cache[key] { return image }
        var config = NSImage.SymbolConfiguration(pointSize: size, weight: weight)
        if let color { config = config.applying(.init(paletteColors: [color])) }
        guard let image = NSImage(systemSymbolName: name, accessibilityDescription: nil)?.withSymbolConfiguration(config) else { return nil }
        cache[key] = image
        return image
    }

    /// A filled symbol in one colour with its inner glyph knocked out (the X of `xmark.circle.fill` stays clear): the
    /// template image tinted, as a monochrome symbol draws. A one-colour palette would paint every layer, X included.
    static func knockout(_ name: String, size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor) -> NSImage? {
        let key = Key(name: name, size: size, weight: weight, color: color, knockout: true)
        if let image = cache[key] { return image }
        guard let template = image(name, size: size, weight: weight) else { return nil }
        let tinted = NSImage(size: template.size, flipped: false) { rect in
            template.draw(in: rect)
            color.set(); rect.fill(using: .sourceAtop)
            return true
        }
        cache[key] = tinted
        return tinted
    }

    /// Draws a symbol centred on `rect` (flipped views included); `knockout` keeps a filled symbol's inner glyph clear.
    static func draw(_ name: String, centredIn rect: NSRect, size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor, knockout: Bool) {
        guard knockout, let image = Self.knockout(name, size: size, weight: weight, color: color) else {
            return draw(name, centredIn: rect, size: size, weight: weight, color: color)
        }
        let s = image.size
        image.draw(
            in: NSRect(x: rect.midX - s.width / 2, y: rect.midY - s.height / 2, width: s.width, height: s.height),
            from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
    }

    /// Draws a symbol centred on `rect` (flipped views included).
    static func draw(_ name: String, centredIn rect: NSRect, size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor) {
        guard let image = image(name, size: size, weight: weight, color: color) else { return }
        let s = image.size
        image.draw(
            in: NSRect(x: rect.midX - s.width / 2, y: rect.midY - s.height / 2, width: s.width, height: s.height),
            from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
    }
}
