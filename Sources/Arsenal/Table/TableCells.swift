// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// The cells of the menu-hosted table that react to the pointer: the gear (a row's settings card), the pinned "Get more
// models" row and the Refresh glyph at the top right. Hover uses RowAction's mechanism (an `.activeAlways` tracking area, redrawn at once), the
// click is a plain mouseDown and the tooltip is the view's own `toolTip`, all of which work during menu tracking.

/// The gear in a row's settings column: `gearshape` 13 pt; rest tertiary, hovered secondary; while its card is open,
/// `gearshape.fill` in the label colour (its row takes a neutral fill: ModelTableRow). Hover only shades it. A click opens
/// the model's settings card beside it inside the table (SettingsCard); clicked again, it closes the card. It acts on the
/// mouse-down and takes the mouse-up itself, as the row's Load button and the precision segments do, so neither reaches
/// the enclosing menu item: the menu stays open and nothing about it changes. No tooltip: the symbol is the standard one,
/// and a tooltip under the pointer covered the row's action beside it.
struct GearCell: NSViewRepresentable {
    /// Its accessibility name.
    static let help = "Settings"
    let id: String
    let settings: SettingsCard
    var active = false
    /// On the loaded row's colour: drawn in white.
    var inverted = false
    let click: (NSView) -> Void
    func makeNSView(context: Context) -> GearView { GearView() }
    func updateNSView(_ v: GearView, context: Context) {
        v.click = click
        v.inverted = inverted
        settings.register(v, for: id)
        if active { v.active = true }
    }
}

final class GearView: NSView {
    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    var click: (NSView) -> Void = { _ in }
    /// Its row's card is open.
    var active = false { didSet { if active != oldValue { needsDisplay = true; if window != nil { display() } } } }
    var inverted = false { didSet { if inverted != oldValue { needsDisplay = true } } }
    private var hovered = false
    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }
    override var acceptsFirstResponder: Bool { false }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas where area.owner === self { removeTrackingArea(area) }
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect], owner: self, userInfo: nil))
    }
    override func mouseEntered(with event: NSEvent) { hovered = true; needsDisplay = true; if window != nil { display() } }
    override func mouseExited(with event: NSEvent) { hovered = false; needsDisplay = true; if window != nil { display() } }
    /// RowActionView's path: act on the press, then redraw the host in the menu's tracking mode.
    override func mouseDown(with event: NSEvent) {
        click(self)
        HostRefresh.after(self)
    }
    /// The release belongs to the press: it goes nowhere else (not up the responder chain to the menu item).
    override func mouseUp(with event: NSEvent) {}
    override func rightMouseDown(with event: NSEvent) {}
    var color: NSColor {
        if inverted { return ArsenalPalette.hotText.withAlphaComponent(active || hovered ? 1 : 0.92) }
        let p = ArsenalPalette.onMenu
        return active ? p.t1 : hovered ? p.t2 : p.t3
    }
    var symbol: String { active ? "gearshape.fill" : "gearshape" }
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif
        Symbol.draw(symbol, centredIn: bounds, size: TableMetrics.font(13), color: color)
    }
    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .button }
    override func accessibilityLabel() -> String? { GearCell.help }
    override func isAccessibilityExpanded() -> Bool { active }
    /// The card opens beside the gear inside the same table: nothing ends the menu this gear is in.
    override func accessibilityPerformPress() -> Bool {
        click(self)
        HostRefresh.after(self)
        return true
    }
}

/// The last row of the table, always: `plus.circle` in the flame slot and "Get more models", both secondary; under
/// the pointer the progress wash and the primary colour. The whole row is the target.
struct PinnedRow: NSViewRepresentable {
    static let title = "Get more models"
    /// None: the row's words say it.
    static let help: String? = nil
    /// Render harness: draw as under the pointer.
    var hovered = false
    let click: () -> Void
    func makeNSView(context: Context) -> PinnedRowView { PinnedRowView() }
    func updateNSView(_ v: PinnedRowView, context: Context) { v.click = click; v.forcedHover = hovered; v.toolTip = Self.help; v.needsDisplay = true }
}

final class PinnedRowView: NSView {
    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    var click: () -> Void = {}
    var forcedHover = false
    var inverted = false { didSet { if inverted != oldValue { needsDisplay = true } } }
    private var hovered = false
    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas where area.owner === self { removeTrackingArea(area) }
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect], owner: self, userInfo: nil))
    }
    override func mouseEntered(with event: NSEvent) { hovered = true; needsDisplay = true; if window != nil { display() } }
    override func mouseExited(with event: NSEvent) { hovered = false; needsDisplay = true; if window != nil { display() } }
    override func mouseDown(with event: NSEvent) { click() }
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif

        let on = hovered || forcedHover
        if on {
            ArsenalPalette.onMenu.selFill.setFill()
            NSBezierPath(roundedRect: bounds, xRadius: ModelTable.rowCorner, yRadius: ModelTable.rowCorner).fill()
        }
        let color: NSColor = on ? ArsenalPalette.onMenu.t1 : ArsenalPalette.onMenu.t2
        let iconX = ModelTable.W.rowPadding, iconW = ModelTable.W.icon
        Symbol.draw("plus.circle", centredIn: NSRect(x: iconX, y: 0, width: iconW, height: bounds.height), size: ModelTable.W.flameSize, color: color)
        let text = NSAttributedString(string: PinnedRow.title, attributes: [.font: NSFont.systemFont(ofSize: ModelTable.valueSize), .foregroundColor: color])
        let h = text.size().height
        text.draw(at: NSPoint(x: iconX + iconW + ModelTable.W.iconGap, y: bounds.midY - h / 2))
    }
    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .button }
    override func accessibilityLabel() -> String? { PinnedRow.title }
    override func accessibilityHelp() -> String? { PinnedRow.help }
    override func accessibilityPerformPress() -> Bool { click(); return true }
}

/// Refresh models: an `arrow.clockwise` glyph at the table's top right, over the rows' trash/cancel glyphs. Secondary at
/// rest; the label colour on a faint rounded wash under the pointer; dimmed and inert while a Refresh, Get or Load runs,
/// its tooltip saying why. A refused Refresh says why in the table's footer line.
struct RefreshControl: NSViewRepresentable {
    /// Its accessibility name.
    static let title = "Refresh models"
    static let help = "Check for new models"
    let enabled: Bool
    /// The tooltip: `help`, or why it is off.
    var help: String = RefreshControl.help
    /// Render harness: draw as under the pointer.
    var hovered = false
    let click: () -> Void
    func makeNSView(context: Context) -> RefreshControlView { RefreshControlView() }
    func updateNSView(_ v: RefreshControlView, context: Context) {
        v.enabled = enabled; v.click = click; v.forcedHover = hovered; v.toolTip = help; v.needsDisplay = true
    }
}

final class RefreshControlView: NSView {
    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    var enabled = true, forcedHover = false
    var click: () -> Void = {}
    var inverted = false { didSet { if inverted != oldValue { needsDisplay = true } } }
    private var hovered = false
    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas where area.owner === self { removeTrackingArea(area) }
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect], owner: self, userInfo: nil))
    }
    override func mouseEntered(with event: NSEvent) { hovered = true; needsDisplay = true; if window != nil { display() } }
    override func mouseExited(with event: NSEvent) { hovered = false; needsDisplay = true; if window != nil { display() } }
    override func mouseDown(with event: NSEvent) { _ = accessibilityPerformPress() }

    var color: NSColor {
        let p = ArsenalPalette.onMenu
        return !enabled ? p.t4 : (hovered || forcedHover) ? p.t1 : p.t2
    }
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif

        if enabled, hovered || forcedHover {
            ArsenalPalette.onMenu.overlay(0.10).setFill()
            NSBezierPath(roundedRect: bounds, xRadius: RowAction.cornerRadius, yRadius: RowAction.cornerRadius).fill()
        }
        Symbol.draw("arrow.clockwise", centredIn: bounds, size: TableMetrics.font(13), weight: .medium, color: color)
    }
    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .button }
    override func accessibilityLabel() -> String? { RefreshControl.title }
    override func accessibilityHelp() -> String? { toolTip }
    override func isAccessibilityEnabled() -> Bool { enabled }
    override func accessibilityPerformPress() -> Bool {
        guard enabled else { return false }
        click(); HostRefresh.after(self)
        return true
    }
}
