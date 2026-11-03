// Ported from Projects/Vella Sources/Vella/QA/MenuMock.swift @ f4c5c34 (Vella; relicensed MIT by its author, 2 Oct 2026); draws subtitles as Arsenal uses them.
import AppKit
@testable import Arsenal

// MenuMock: generic drawing for documentation renders only.

/// Draws the actual NSMenuItems for documentation in the current system palette. A real NSMenu cannot be rendered offscreen.
final class MenuMock: NSView {
    let items: [NSMenuItem]
    init(items: [NSMenuItem], width: CGFloat) {
        self.items = items.filter { !$0.isHidden }
        // Wide enough for the longest title (menus size to their content).
        let font = NSFont.systemFont(ofSize: 13)
        let longest = self.items.map { ($0.title as NSString).size(withAttributes: [.font: font]).width + (MenuMock.keyEquivalentText($0).isEmpty ? 0 : 50) }.max() ?? 0
        let width = max(width, ceil(longest) + 80)
        let height = self.items.reduce(CGFloat(12)) { $0 + ($1.isSeparatorItem ? 11 : $1.isSectionHeader ? 22 : MenuMock.hasSubtitle($1) ? 40 : 26) }
        super.init(frame: NSRect(x: 0, y: 0, width: width, height: height))
    }
    required init?(coder: NSCoder) { nil }

    /// A stand-in for the native menu material behind menu-hosted content (a real menu window cannot be drawn offscreen).
    /// Dark: sampled from the installed 3.2.0 main menu in a screenshot of 10 Oct 2026 (#252525 to #2A2A2A).
    /// Light: an estimate of the light material over a light desktop.
    static func material(dark: Bool) -> NSColor { ArsenalPalette.rgb(dark ? 0x282828 : 0xF0F0F0) }

    /// The translucent material over a picture: `picture` (a blurred wallpaper) under the material colour at `opacity`
    /// (an approximation of how much the real menu material lets through).
    static func applyMaterial(to layer: CALayer?, dark: Bool, picture: NSImage?, opacity: CGFloat = 0.72) {
        guard let layer else { return }
        guard let picture else { layer.backgroundColor = material(dark: dark).cgColor; return }
        layer.contents = picture; layer.contentsGravity = .resizeAspectFill
        let tint = CALayer(); tint.frame = layer.bounds; tint.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        tint.backgroundColor = material(dark: dark).withAlphaComponent(opacity).cgColor
        layer.addSublayer(tint)
    }
    /// A busy, colourful wallpaper, blurred as the material blurs what is behind it.
    static func wallpaper(size: NSSize) -> NSImage {
        let picture = NSImage(contentsOfFile: "/System/Library/Desktop Pictures/.thumbnails/Big Sur Graphic.heic") ?? V21RenderDelegate.darkPicture()
        return V21RenderDelegate.blurred(picture, size: size)
    }

    /// A behind-window NSVisualEffectView draws a flat grey offscreen (the blur is the window server's). Before a capture,
    /// each one inside `container` gets the same stand-in as the menu around it: the wallpaper's matching part under the
    /// material colour, or the plain material colour.
    static func standInMaterials(in container: NSView, dark: Bool, picture: NSImage?, opacity: CGFloat = 0.72) {
        func walk(_ v: NSView) {
            if let effect = v as? NSVisualEffectView, effect.blendingMode == .behindWindow, let layer = effect.layer {
                layer.sublayers?.filter { $0.name == "stand-in" }.forEach { $0.removeFromSuperlayer() }
                let standIn = CALayer(); standIn.name = "stand-in"; standIn.frame = layer.bounds
                if let picture {
                    let r = effect.convert(effect.bounds, to: container), b = container.bounds
                    standIn.contents = picture
                    standIn.contentsRect = CGRect(x: r.minX / b.width, y: r.minY / b.height, width: r.width / b.width, height: r.height / b.height)
                    let tint = CALayer(); tint.frame = standIn.bounds
                    tint.backgroundColor = material(dark: dark).withAlphaComponent(opacity).cgColor
                    standIn.addSublayer(tint)
                } else {
                    standIn.backgroundColor = material(dark: dark).cgColor
                }
                layer.addSublayer(standIn)
            }
            v.subviews.forEach(walk)
        }
        walk(container)
    }

    /// Draws `view` offscreen to `url` (PNG), then calls `done`; `prepare` runs just before the bitmap is taken.
    static func capture(_ view: NSView, to url: URL, prepare: (() -> Void)? = nil, done: @escaping () -> Void) {
        // Born off every screen, so ordering it in never shows a frame at the screen's corner first.
        let window = NSWindow(
            contentRect: NSRect(origin: NSPoint(x: -5000, y: -5000), size: view.frame.size), styleMask: .borderless, backing: .buffered, defer: false)
        window.backgroundColor = .clear; window.contentView = view; window.appearance = Appearance.shared.nsAppearance
        window.setFrameOrigin(NSPoint(x: -5000, y: -5000)); window.orderFrontRegardless()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            view.layoutSubtreeIfNeeded(); prepare?()
            if let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
                view.cacheDisplay(in: view.bounds, to: rep)
                try? rep.representation(using: .png, properties: [:])?.write(to: url)
            }
            window.orderOut(nil); done()
        }
    }
    static func render(_ items: [NSMenuItem], width: CGFloat, to url: URL, done: @escaping () -> Void) {
        capture(MenuMock(items: items, width: width), to: url, done: done)
    }
    /// `<prefix><name>.png` for each named submenu of `menu`.
    static func renderSubmenus(of menu: NSMenu, titles: [(String, String)], into directory: URL, prefix: String = "", done: @escaping () -> Void) {
        let subs = titles.compactMap { title, name in menu.items.first(where: { $0.title == title })?.submenu.map { ($0.items, name) } }
        func next(_ index: Int) {
            guard index < subs.count else { done(); return }
            render(subs[index].0, width: 300, to: directory.appendingPathComponent("\(prefix)\(subs[index].1).png")) { next(index + 1) }
        }
        next(0)
    }
    static func hasSubtitle(_ item: NSMenuItem) -> Bool { !(item.subtitle ?? "").isEmpty }

    static func keyEquivalentText(_ item: NSMenuItem) -> String {
        guard !item.keyEquivalent.isEmpty else { return "" }
        let m = item.keyEquivalentModifierMask
        var text = ""
        if m.contains(.control) { text += "⌃" }
        if m.contains(.option) { text += "⌥" }
        if m.contains(.shift) { text += "⇧" }
        if m.contains(.command) { text += "⌘" }
        return text + item.keyEquivalent.uppercased()
    }

    override func draw(_ dirtyRect: NSRect) {
        let dark = Appearance.shared.palette.dark
        let textColor: NSColor = dark ? .white : .black
        let panel = NSBezierPath(roundedRect: bounds.insetBy(dx: 1, dy: 1), xRadius: 10, yRadius: 10)
        (dark ? NSColor(calibratedRed: 0.14, green: 0.14, blue: 0.15, alpha: 1) : NSColor(calibratedWhite: 0.97, alpha: 1)).setFill(); panel.fill()
        textColor.withAlphaComponent(0.12).setStroke(); panel.lineWidth = 1; panel.stroke()
        var y = bounds.height - 6
        let attrs: (NSColor, CGFloat) -> [NSAttributedString.Key: Any] = { c, size in [.font: NSFont.systemFont(ofSize: size), .foregroundColor: c] }
        for item in items {
            if item.isSectionHeader {
                y -= 22
                NSAttributedString(
                    string: item.title,
                    attributes: [
                        .font: NSFont.systemFont(ofSize: 11, weight: .semibold),
                        .foregroundColor: textColor.withAlphaComponent(0.5)
                    ]
                ).draw(at: NSPoint(x: 14, y: y + 4))
                continue
            }
            if item.isSeparatorItem {
                y -= 5.5
                textColor.withAlphaComponent(0.14).setFill(); NSBezierPath(rect: NSRect(x: 14, y: y, width: bounds.width - 28, height: 1)).fill()
                y -= 5.5; continue
            }
            let lift: CGFloat = Self.hasSubtitle(item) ? 14 : 0
            y -= 26 + lift
            // Rows drawn by their own view (Copy Last, Support) rest unlit, as native rows do.
            let color: NSColor = item.isEnabled ? textColor : textColor.withAlphaComponent(0.4)
            let title =
                item.attributedTitle.map {
                    NSAttributedString(string: $0.string, attributes: attrs(($0.attribute(.foregroundColor, at: 0, effectiveRange: nil) as? NSColor) ?? color, 13))
                }
                ?? NSAttributedString(string: item.title, attributes: attrs(color, 13))
            var x: CGFloat = 14
            if item.state == .on { NSAttributedString(string: "✓", attributes: attrs(color, 12)).draw(at: NSPoint(x: 8, y: y + 5)) }
            if items.contains(where: { $0.state == .on }) { x += 8 }
            if let image = item.image, item.applicableMenuImage as? Bool == true, !image.isTemplate {
                image.draw(in: NSRect(x: x, y: y + 6 + lift / 2, width: 14, height: 14), from: .zero, operation: .sourceOver, fraction: item.isEnabled ? 1 : 0.4)
                x += 22
            } else if let image = item.image, item.applicableMenuImage as? Bool == true, let tinted = image.copy() as? NSImage { // NSImage.copy() is an NSImage
                tinted.isTemplate = false
                let r = NSRect(x: x, y: y + 6 + lift / 2, width: 14, height: 14)
                NSGraphicsContext.saveGraphicsState()
                tinted.lockFocus(); color.set(); NSRect(origin: .zero, size: tinted.size).fill(using: .sourceAtop); tinted.unlockFocus()
                tinted.draw(in: r, from: .zero, operation: .sourceOver, fraction: item.isEnabled ? 1 : 0.4)
                NSGraphicsContext.restoreGraphicsState()
                x += 22
            } else if items.contains(where: { ($0.applicableMenuImage as? Bool == true) }) {
                x += 22
            }
            title.draw(at: NSPoint(x: x, y: y + 5 + lift))
            if let subtitle = item.subtitle, !subtitle.isEmpty {
                NSAttributedString(string: subtitle, attributes: attrs(textColor.withAlphaComponent(0.5), 11)).draw(at: NSPoint(x: x, y: y + 4))
            }
            if item.submenu != nil { NSAttributedString(string: "›", attributes: attrs(color, 15)).draw(at: NSPoint(x: bounds.width - 22, y: y + 4)) }
            // A trailing badge (a task's "16 GB download"), as AppKit draws it: secondary text before the arrow or the edge.
            if let badge = item.badge?.stringValue, !badge.isEmpty {
                let text = NSAttributedString(string: badge, attributes: attrs(textColor.withAlphaComponent(0.5), 12))
                let right = bounds.width - (item.submenu != nil ? 30 : 16)
                text.draw(at: NSPoint(x: right - text.size().width, y: y + 6 + lift))
            }
            let key = Self.keyEquivalentText(item)
            if !key.isEmpty {
                let text = NSAttributedString(string: key, attributes: attrs(textColor.withAlphaComponent(0.5), 13))
                text.draw(at: NSPoint(x: bounds.width - 16 - text.size().width, y: y + 5))
            }
        }
    }
}
