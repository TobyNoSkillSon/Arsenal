// Adapted from Projects/Vella Sources/Vella/ModelsTable/RowAction.swift @ f4c5c34 (Vella; relicensed MIT by its author, 2 Oct 2026): adds the
// download's cancel glyph (always shown while a Get runs) beside the progress text.
import AppKit
import SwiftUI

// The Models table's action cell: a normal small bordered button at the row end, always visible, one word (Get, Load,
// Unload), fixed size, at the weight of the table's numbers; delete is a trash glyph beside it under the pointer; while a
// download runs, a cancel glyph sits in the same place and is always visible. AppKit and SwiftUI only, no app types.
//
// - Reload on a pending row (loaded, something else chosen) is the row's prominent button: filled in the gold of the
//   chosen precision with a dark title; the row keeps its loaded blue. While busy (a download's percentage, a load's
//   "…") the button shows that text and takes no click.
// - Hover inside an NSMenu (for the trash glyph only): menus track the mouse in their own run-loop mode and their
//   windows are never key, so SwiftUI's `.onHover` is not relied on. An NSTrackingArea with `.activeAlways` (active in
//   any window, key or not) and `.inVisibleRect` delivers mouseEntered/mouseExited to this view during menu tracking;
//   the view redraws itself at once (`display()`), not on SwiftUI's schedule. Menus cannot host context menus.
// - A click on the trash glyph removes the row from the table (or, while downloading, on the cancel glyph cancels); anywhere else performs.
// - Tooltips are AppKit tooltip rects (SwiftUI `.help` never shows in a menu): the action's text on the cell, the
//   delete text on the trash glyph.
// - The geometry is explicit (`width` × `height`, `buttonRect`, `trashRect`), so the first frame of a menu is right.

struct RowAction: View {
    nonisolated static let width: CGFloat = TableMetrics.pt(130)
    nonisolated static let height: CGFloat = TableMetrics.pt(28)
    /// The button's slot: wide enough for the longest real title, "Get · 16.0 GB" (the size shows before a download
    /// starts), at Vella's font and padding; a small control's height.
    nonisolated static let buttonWidth: CGFloat = TableMetrics.pt(98), buttonHeight: CGFloat = TableMetrics.pt(22)
    /// Vella's button (pt(64)) for one-word titles (Load, Unload, Reload); a longer one widens to its text plus Vella's
    /// padding, up to the slot. The button is centred in the slot, so the column and the trash glyph never move.
    nonisolated static let minimumButtonWidth: CGFloat = TableMetrics.pt(64), titlePadding: CGFloat = TableMetrics.pt(11)
    @MainActor static func buttonWidth(_ shown: String, font: NSFont) -> CGFloat {
        let text = (shown as NSString).size(withAttributes: [.font: font]).width
        return min(buttonWidth, max(minimumButtonWidth, ((text + 2 * titlePadding) * 2).rounded(.up) / 2))
    }
    /// The title as drawn: one that does not fit the slot drops its " · size" part (the hover says the size).
    @MainActor static func shownTitle(_ title: String, font: NSFont) -> String {
        let text = (title as NSString).size(withAttributes: [.font: font]).width
        guard text + 2 * TableMetrics.pt(4) > buttonWidth, let cut = title.range(of: " \u{00b7} ") else { return title }
        return String(title[..<cut.lowerBound])
    }
    /// The trash glyph's square, its gap after the button, and the button's corner radius and leading inset.
    nonisolated static let trashSize: CGFloat = TableMetrics.pt(22), trashGap: CGFloat = TableMetrics.pt(8)
    nonisolated static let cornerRadius: CGFloat = TableMetrics.pt(5), inset: CGFloat = TableMetrics.pt(2)
    /// The trash glyph's tooltip when it can act (Add's inverse, for a row without weights on this Mac).
    static let deleteHelp = "Remove from the table"
    /// Cancel's tooltip (a Get resumes from the partial files).
    static let cancelHelp = "Stop the download; Get resumes it later"
    /// The countdown's tooltip: a click on it stops waiting for a gated model.
    static let stopWaitingHelp = "Stop waiting and take it off the table"
    /// The automatic retry's interval: the ring is full at this many seconds.
    static let countdownSeconds: Double = 30

    /// The button's title: Get, Load, Unload or Reload.
    let title: String
    /// A download's percentage or a load's "…": shown instead of the title; no click.
    var busyText: String?
    let enabled: Bool
    let deletable: Bool
    /// Drawn on the loaded row's colour: white title, white fill and border (Vella's Unload).
    var hot = false
    /// Reload pending: the gold filled button.
    var prominent = false
    /// Render harness: draw the pointer over the cell (the trash glyph shows).
    var hovered = false
    let help: String
    /// A download runs: the cancel glyph shows and a click on it calls `onCancel`.
    var cancellable = false
    let onPerform: () -> Void
    let onDelete: () -> Void
    var onCancel: () -> Void = {}
    /// The trash's tooltip when it can act: the model's delete says what it removes.
    var trashHelp: String = RowAction.deleteHelp
    /// The reason the trash is off (its delete is refused now). Off, the glyph is drawn dimmed, says why on hover, and a click does nothing.
    var trashBlocked: String?
    /// A gated Get's automatic retry: a ring with the seconds left sits in the trash glyph's place (the only thing that
    /// redraws at rest, once a second); a click on the button retries now, on the ring stops waiting (`onCancel`).
    var countdown: Date?

    var body: some View {
        ActionRepresentable(spec: self).frame(width: Self.width, height: Self.height)
    }
}

private struct ActionRepresentable: NSViewRepresentable {
    let spec: RowAction
    func makeNSView(context: Context) -> RowActionView { let view = RowActionView(); view.spec = spec; view.forcedHover = spec.hovered; return view }
    func updateNSView(_ view: RowActionView, context: Context) { view.spec = spec; view.forcedHover = spec.hovered; view.needsDisplay = true }
    func sizeThatFits(_ proposal: ProposedViewSize, nsView: RowActionView, context: Context) -> CGSize? {
        CGSize(width: RowAction.width, height: RowAction.height)
    }
}

final class RowActionView: NSView, NSViewToolTipOwner, ToolTipRegions {
    /// The tooltip regions this view registered with AppKit (tag → rect and user data), for the hover tests.
    private(set) var toolTipRegions: [NSView.ToolTipTag: ToolTipRegion] = [:]
    override func addToolTip(_ rect: NSRect, owner: Any, userData data: UnsafeMutableRawPointer?) -> NSView.ToolTipTag {
        let tag = super.addToolTip(rect, owner: owner, userData: data)
        toolTipRegions[tag] = ToolTipRegion(rect: rect, userData: data)
        return tag
    }
    override func removeAllToolTips() { super.removeAllToolTips(); toolTipRegions = [:] }
    override func removeToolTip(_ tag: NSView.ToolTipTag) { super.removeToolTip(tag); toolTipRegions[tag] = nil }

    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    var spec: RowAction? { didSet { refreshToolTips(); scheduleTick() } }
    /// The countdown's clock (renders pin it).
    static var now: () -> Date = { Date() }
    private var tick: Timer?
    /// Whole seconds left on the countdown, or nil without one.
    var secondsLeft: Int? { spec?.countdown.map { max(0, Int(($0.timeIntervalSince(Self.now())).rounded(.up))) } }
    /// One redraw at each second's change, only while a countdown shows in a window; none at rest otherwise.
    private func scheduleTick() {
        tick?.invalidate(); tick = nil
        guard let end = spec?.countdown, window != nil else { return }
        let left = end.timeIntervalSince(Self.now())
        guard left > 0 else { return }
        let wait = left - (left.rounded(.up) - 1) + 0.01
        let timer = Timer(timeInterval: max(0.05, wait), repeats: false) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.needsDisplay = true; if self.window != nil { self.display() }
                self.scheduleTick()
            }
        }
        RunLoop.main.add(timer, forMode: .common); tick = timer
    }
    override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); scheduleTick() }
    var forcedHover = false
    private(set) var hovered = false
    static let titleFont = NSFont.systemFont(ofSize: TableMetrics.font(12))
    static let prominentFont = NSFont.systemFont(ofSize: TableMetrics.font(12), weight: .semibold)
    static let busyFont = NSFont.monospacedDigitSystemFont(ofSize: TableMetrics.font(12), weight: .regular)

    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }
    override var acceptsFirstResponder: Bool { false }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    /// The drawn text and its font.
    var shown: (text: String, font: NSFont) {
        guard let spec else { return ("", Self.titleFont) }
        if let busy = spec.busyText { return (busy, Self.busyFont) }
        let font = spec.prominent ? Self.prominentFont : Self.titleFont
        return (RowAction.shownTitle(spec.title, font: font), font)
    }
    var buttonRect: NSRect {
        let width = RowAction.buttonWidth(shown.text, font: shown.font)
        return NSRect(
            x: RowAction.inset + (RowAction.buttonWidth - width) / 2, y: (bounds.height - RowAction.buttonHeight) / 2, width: width,
            height: RowAction.buttonHeight)
    }
    var trashRect: NSRect {
        NSRect(
            x: RowAction.buttonWidth + RowAction.trashGap, y: (bounds.height - RowAction.trashSize) / 2, width: RowAction.trashSize,
            height: RowAction.trashSize)
    }
    private var showsHover: Bool { hovered || forcedHover }

    // MARK: Hover (see the note above)
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas where area.owner === self { removeTrackingArea(area) }
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect], owner: self, userInfo: nil))
    }
    override func mouseEntered(with event: NSEvent) { setHovered(true) }
    override func mouseExited(with event: NSEvent) { setHovered(false) }
    func setHovered(_ value: Bool) {
        guard hovered != value else { return }
        hovered = value
        needsDisplay = true
        if window != nil { display() }
    }

    // MARK: Click: trash deletes, anywhere else performs
    override func mouseDown(with event: NSEvent) {
        guard let spec else { return }
        let point = convert(event.locationInWindow, from: nil)
        let glyph = trashRect.insetBy(dx: -TableMetrics.pt(2), dy: -TableMetrics.pt(4))
        if spec.busyText != nil {
            guard spec.cancellable, glyph.contains(point) else { return }
            spec.onCancel()
            HostRefresh.after(self)
            return
        }
        if spec.countdown != nil, glyph.contains(point) {
            spec.onCancel()
        } else if spec.deletable, glyph.contains(point) {
            // A refused delete does nothing; its reason is the tooltip.
            if spec.trashBlocked == nil { spec.onDelete() }
        } else if spec.enabled {
            spec.onPerform()
        } else {
            return
        }
        HostRefresh.after(self)
    }

    // MARK: Tooltips
    private func refreshToolTips() {
        removeAllToolTips()
        addToolTip(bounds, owner: self, userData: nil)
    }
    override func setFrameSize(_ newSize: NSSize) { super.setFrameSize(newSize); refreshToolTips() }
    func view(_ view: NSView, stringForToolTip tag: NSView.ToolTipTag, point: NSPoint, userData data: UnsafeMutableRawPointer?) -> String {
        guard let spec else { return "" }
        if spec.busyText != nil { return spec.cancellable && trashRect.contains(point) ? RowAction.cancelHelp : spec.help }
        if spec.countdown != nil { return trashRect.contains(point) ? RowAction.stopWaitingHelp : spec.help }
        return spec.deletable && trashRect.contains(point) ? (spec.trashBlocked ?? spec.trashHelp) : spec.help
    }
    /// The text a tooltip shows at a point (tests and renders).
    func toolTipText(at point: NSPoint) -> String { view(self, stringForToolTip: 0, point: point, userData: nil) }

    // MARK: Drawing
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif

        guard let spec else { return }
        let pal = ArsenalPalette.onMenu
        let white = ArsenalPalette.hotText
        let text: NSColor = spec.hot ? white : pal.t1
        let quiet: NSColor = spec.hot ? white.withAlphaComponent(0.8) : pal.t2
        func overlay(_ alpha: CGFloat) -> NSColor { spec.hot ? white.withAlphaComponent(alpha * 1.5) : pal.overlay(alpha) }
        let button = buttonRect
        let shape = NSBezierPath(roundedRect: button.insetBy(dx: 0.5, dy: 0.5), xRadius: RowAction.cornerRadius, yRadius: RowAction.cornerRadius)
        let context = NSGraphicsContext.current?.cgContext
        if spec.busyText != nil {
            overlay(0.18).setStroke(); shape.lineWidth = 1; shape.stroke()
            drawCentred(shown.text, in: button, font: Self.busyFont, color: quiet)
        } else if spec.prominent {
            context?.setAlpha(spec.enabled ? 1 : 0.4)
            ArsenalPalette.gold.setFill(); shape.fill()
            ArsenalPalette.ink.withAlphaComponent(0.18).setStroke(); shape.lineWidth = 1; shape.stroke()
            drawCentred(shown.text, in: button, font: Self.prominentFont, color: ArsenalPalette.ink)
            context?.setAlpha(1)
        } else {
            // A small bordered button: a faint fill and a hairline border, lighter on the loaded row.
            context?.setAlpha(spec.enabled ? 1 : 0.4)
            overlay(0.10).setFill(); shape.fill()
            overlay(0.22).setStroke(); shape.lineWidth = 1; shape.stroke()
            drawCentred(shown.text, in: button, font: Self.titleFont, color: text)
            context?.setAlpha(1)
        }
        if spec.busyText != nil, spec.cancellable {
            Symbol.draw("xmark.circle.fill", centredIn: trashRect, size: TableMetrics.font(14), color: quiet, knockout: true)
        } else if let left = secondsLeft {
            drawCountdown(left, total: RowAction.countdownSeconds, text: text)
        } else if showsHover && spec.deletable && spec.busyText == nil {
            Symbol.draw("trash", centredIn: trashRect, size: TableMetrics.font(13), color: spec.trashBlocked == nil ? quiet : quiet.withAlphaComponent(0.35))
        }
    }
    /// The ring: a faint track, the time left as an arc from twelve o'clock, the seconds in the middle at full contrast.
    private func drawCountdown(_ left: Int, total: Double, text: NSColor) {
        guard let spec else { return }
        let ring = trashRect.insetBy(dx: TableMetrics.pt(1.5), dy: TableMetrics.pt(1.5))
        let track = NSBezierPath(ovalIn: ring); track.lineWidth = TableMetrics.pt(1.5)
        (spec.hot ? ArsenalPalette.hotText.withAlphaComponent(0.3) : ArsenalPalette.onMenu.overlay(0.22)).setStroke(); track.stroke()
        let fraction = min(1, Double(left) / total)
        if fraction > 0 {
            let arc = NSBezierPath()
            let centre = NSPoint(x: ring.midX, y: ring.midY), radius = ring.width / 2
            // Flipped view: clockwise on screen from twelve o'clock.
            arc.appendArc(withCenter: centre, radius: radius, startAngle: -90, endAngle: -90 + 360 * fraction, clockwise: false)
            arc.lineWidth = TableMetrics.pt(1.5); arc.lineCapStyle = .round
            (spec.hot ? ArsenalPalette.hotText : ArsenalPalette.onMenu.amber).setStroke(); arc.stroke()
        }
        drawCentred("\(left)", in: trashRect, font: Self.countdownFont, color: text)
    }
    static let countdownFont = NSFont.monospacedDigitSystemFont(ofSize: TableMetrics.font(9.5), weight: .medium)
    private func drawCentred(_ string: String, in rect: NSRect, font: NSFont, color: NSColor) {
        let s = NSAttributedString(string: string, attributes: [.font: font, .foregroundColor: color])
        let size = s.size()
        s.draw(at: NSPoint(x: rect.midX - size.width / 2, y: rect.midY - size.height / 2))
    }

    // Accessibility: a button named by its action.
    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .button }
    override func accessibilityLabel() -> String? {
        guard let spec else { return nil }
        if spec.busyText == nil, let left = secondsLeft { return "\(spec.title), retries in \(left) seconds" }
        return spec.busyText ?? spec.title
    }
    override func accessibilityHelp() -> String? { spec?.help }
    override func isAccessibilityEnabled() -> Bool { spec.map { $0.enabled && $0.busyText == nil } ?? false }
    override func accessibilityPerformPress() -> Bool {
        guard let spec, spec.enabled, spec.busyText == nil else { return false }
        spec.onPerform(); return true
    }
    /// Delete, for VoiceOver (the trash glyph shows only under the pointer).
    override func accessibilityCustomActions() -> [NSAccessibilityCustomAction]? {
        guard let spec else { return nil }
        if spec.busyText == nil, spec.countdown != nil {
            return [
                NSAccessibilityCustomAction(name: RowAction.stopWaitingHelp) {
                    spec.onCancel(); return true
                }
            ]
        }
        if spec.busyText != nil, spec.cancellable {
            return [
                NSAccessibilityCustomAction(name: RowAction.cancelHelp) {
                    spec.onCancel(); return true
                }
            ]
        }
        guard spec.deletable, spec.busyText == nil, spec.trashBlocked == nil else { return nil }
        return [
            NSAccessibilityCustomAction(name: spec.trashHelp) {
                spec.onDelete(); return true
            }
        ]
    }
}
