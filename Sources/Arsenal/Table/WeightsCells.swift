// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// The Precision column: one line per precision on disk, 16-bit on top, then 8-bit, then 4-bit, each
// "16-bit   846 MB" with its trash on the same line; a precision that is not on disk has no line. The line is the choice:
// a click chooses that precision (gold, as Vella's chosen cell); on a loaded row the precision that runs carries the white
// edge. One AppKit view draws the column and takes its clicks (a plain mouseDown, as the gear and RowAction do), so it
// works while the menu tracks the mouse; its hovers are its own tooltip regions, which show inside an NSMenu.

/// The column's geometry, at the table's own type size (the row's name and figures).
enum PrecisionLines {
    private static func pt(_ base: CGFloat) -> CGFloat { TableMetrics.pt(base) }
    /// The precision label, and the size in tabular figures so sizes line up digit for digit.
    static var labelFont: NSFont { .systemFont(ofSize: ModelTable.valueSize, weight: .medium) }
    static var sizeFont: NSFont { .monospacedDigitSystemFont(ofSize: ModelTable.valueSize, weight: .regular) }
    /// One line, and the pill drawn inside it.
    static let lineHeight: CGFloat = pt(22)
    static let pillInset: CGFloat = 1, pillRadius: CGFloat = pt(6)
    /// The bolt's slot (custom kernels) before the lines, kept on every row so the lines line up.
    static let boltWidth: CGFloat = pt(16), boltGap: CGFloat = pt(2)
    /// Inside the pill: padding, the label's column (the widest label and int4's triangle), the gap, the size's column.
    static let padding: CGFloat = pt(8), gap: CGFloat = pt(10)
    static let triangleSize: CGFloat = TableMetrics.font(10), triangleGap: CGFloat = pt(3)
    static let labelWidth: CGFloat = ["16-bit", "8-bit", "4-bit"].map { width($0, labelFont) }.max()! + triangleGap + triangleSize + 1
    static let sizeWidth: CGFloat = ["888 MB", "88.8 GB", "888.8 GB"].map { width($0, sizeFont) }.max()!
    static let pillWidth: CGFloat = (2 * padding + labelWidth + gap + sizeWidth).rounded(.up)
    /// The trash after the pill.
    static let trashGap: CGFloat = pt(2), trashWidth: CGFloat = pt(22)
    static let trashSize: CGFloat = TableMetrics.font(12)
    /// The column's width.
    static let width: CGFloat = boltWidth + boltGap + pillWidth + trashGap + trashWidth
    /// The lines' block height.
    static func height(lines: Int) -> CGFloat { CGFloat(lines) * lineHeight }

    static func width(_ text: String, _ font: NSFont) -> CGFloat { ceil((text as NSString).size(withAttributes: [.font: font]).width) }
}

/// The SwiftUI cell for one row's Precision column.
struct PrecisionLinesCell: NSViewRepresentable {
    let lines: [PrecisionLine]
    var hot = false
    /// The row is downloading, making or loading a precision: nothing can be chosen and the lines dim (each line's hover
    /// says why: `PrecisionLine.help`).
    var locked = false
    var bolt = false
    let onSelect: (String) -> Void
    let onDelete: (String) -> Void

    func makeNSView(context: Context) -> PrecisionLinesView { update(PrecisionLinesView()) }
    func updateNSView(_ view: PrecisionLinesView, context: Context) { _ = update(view) }
    func sizeThatFits(_ proposal: ProposedViewSize, nsView: PrecisionLinesView, context: Context) -> CGSize? {
        CGSize(width: PrecisionLines.width, height: proposal.height ?? PrecisionLines.height(lines: lines.count))
    }
    private func update(_ view: PrecisionLinesView) -> PrecisionLinesView {
        view.lines = lines; view.hot = hot; view.locked = locked; view.bolt = bolt
        view.onSelect = onSelect; view.onDelete = onDelete
        return view
    }
}

final class PrecisionLinesView: NSView, NSViewToolTipOwner, ToolTipRegions {
    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    var lines: [PrecisionLine] = [] { didSet { if lines != oldValue { changed() } } }
    var hot = false { didSet { if hot != oldValue { needsDisplay = true } } }
    var locked = false { didSet { if locked != oldValue { changed() } } }
    var bolt = false { didSet { if bolt != oldValue { changed() } } }
    var onSelect: (String) -> Void = { _ in }
    var onDelete: (String) -> Void = { _ in }

    /// What the pointer is over: a line's pill or its trash.
    enum Part: Equatable { case pill(Int), trash(Int), bolt }
    private(set) var hovered: Part? { didSet { if hovered != oldValue { needsDisplay = true } } }

    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }
    override var acceptsFirstResponder: Bool { false }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override var intrinsicContentSize: NSSize { NSSize(width: PrecisionLines.width, height: PrecisionLines.height(lines: lines.count)) }

    private func changed() {
        needsDisplay = true
        refreshToolTips()
        accessibilityParts = nil
    }
    override func setFrameSize(_ newSize: NSSize) {
        let old = frame.size
        super.setFrameSize(newSize)
        // Only a real change moves the regions: re-registering on every layout pass would hide a tooltip that is showing.
        if newSize != old { refreshToolTips() }
    }

    // MARK: Geometry

    /// The lines' block, centred on the row (the other columns centre their content on it too).
    var blockTop: CGFloat { ((bounds.height - PrecisionLines.height(lines: lines.count)) / 2).rounded() }
    func lineRect(_ i: Int) -> NSRect {
        NSRect(x: 0, y: blockTop + CGFloat(i) * PrecisionLines.lineHeight, width: bounds.width, height: PrecisionLines.lineHeight)
    }
    func pillRect(_ i: Int) -> NSRect {
        let line = lineRect(i)
        return NSRect(x: PrecisionLines.boltWidth + PrecisionLines.boltGap, y: line.minY, width: PrecisionLines.pillWidth, height: line.height)
            .insetBy(dx: 0, dy: PrecisionLines.pillInset)
    }
    func trashRect(_ i: Int) -> NSRect {
        let line = lineRect(i)
        return NSRect(
            x: PrecisionLines.boltWidth + PrecisionLines.boltGap + PrecisionLines.pillWidth + PrecisionLines.trashGap, y: line.minY,
            width: PrecisionLines.trashWidth, height: line.height)
    }
    /// The bolt sits on the chosen line (the precision that will run), else on the first.
    var boltRect: NSRect? {
        guard bolt, !lines.isEmpty else { return nil }
        let line = lineRect(lines.firstIndex(where: \.selected) ?? 0)
        return NSRect(x: 0, y: line.minY, width: PrecisionLines.boltWidth, height: line.height)
    }
    func part(at point: NSPoint) -> Part? {
        for i in lines.indices {
            if pillRect(i).contains(point) { return .pill(i) }
            if trashRect(i).contains(point) { return .trash(i) }
        }
        if let boltRect, boltRect.contains(point) { return .bolt }
        return nil
    }

    // MARK: Hover and clicks

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas where area.owner === self { removeTrackingArea(area) }
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseEnteredAndExited, .mouseMoved, .activeAlways, .inVisibleRect], owner: self, userInfo: nil))
    }
    override func mouseEntered(with event: NSEvent) { track(event) }
    override func mouseMoved(with event: NSEvent) { track(event) }
    override func mouseExited(with event: NSEvent) { hovered = nil; if window != nil { displayIfNeeded() } }
    private func track(_ event: NSEvent) {
        hovered = part(at: convert(event.locationInWindow, from: nil))
        if window != nil { displayIfNeeded() }
    }
    /// Acts on the press and takes the release, so neither reaches the menu item (as GearView does).
    override func mouseDown(with event: NSEvent) {
        guard let part = part(at: convert(event.locationInWindow, from: nil)) else { return }
        press(part)
    }
    override func mouseUp(with event: NSEvent) {}
    override func rightMouseDown(with event: NSEvent) {}
    /// A pill chooses its precision when it can be chosen; a trash deletes its copy when its delete can run.
    @discardableResult func press(_ part: Part) -> Bool {
        switch part {
        case .pill(let i):
            // The chosen line is already chosen: no write, no refresh.
            guard lines.indices.contains(i), lines[i].choosable, !locked, !lines[i].selected, let column = lines[i].column else { return false }
            onSelect(column)
        case .trash(let i):
            guard lines.indices.contains(i), lines[i].copy.enabled else { return false }
            onDelete(lines[i].dtype)
        case .bolt: return false
        }
        HostRefresh.after(self)
        return true
    }

    // MARK: Tooltips

    /// The tooltip regions this view registered with AppKit (tag → rect and user data), for the hover tests.
    private(set) var toolTipRegions: [NSView.ToolTipTag: ToolTipRegion] = [:]
    override func addToolTip(_ rect: NSRect, owner: Any, userData data: UnsafeMutableRawPointer?) -> NSView.ToolTipTag {
        let tag = super.addToolTip(rect, owner: owner, userData: data)
        toolTipRegions[tag] = ToolTipRegion(rect: rect, userData: data)
        return tag
    }
    override func removeAllToolTips() { super.removeAllToolTips(); toolTipRegions = [:] }
    override func removeToolTip(_ tag: NSView.ToolTipTag) { super.removeToolTip(tag); toolTipRegions[tag] = nil }
    func refreshToolTips() {
        removeAllToolTips()
        for i in lines.indices {
            addToolTip(pillRect(i), owner: self, userData: nil)
            addToolTip(trashRect(i), owner: self, userData: nil)
        }
        if let boltRect { addToolTip(boltRect, owner: self, userData: nil) }
    }
    /// The hover of a part: a line's own (its warning first, what it is, and the lock line while the row is busy), its
    /// trash's, or the bolt's.
    func tooltip(_ part: Part) -> String {
        switch part {
        case .pill(let i): return lines[i].help
        case .trash(let i): return lines[i].copy.help
        case .bolt: return TierControl.boltHelp
        }
    }
    func view(_ view: NSView, stringForToolTip tag: NSView.ToolTipTag, point: NSPoint, userData data: UnsafeMutableRawPointer?) -> String {
        part(at: point).map(tooltip) ?? ""
    }

    // MARK: Drawing

    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif
        guard !lines.isEmpty else { return }
        let c = TierSegments.Control.colors(hot: hot, dark: ArsenalPalette.onMenu.dark)
        let context = NSGraphicsContext.current?.cgContext
        // A busy row: the lines at half strength; the gold and the loaded edge stay opaque (gold at half strength over the
        // loaded blue turns muddy), as in the precision cells.
        let dimmed: CGFloat = locked ? 0.5 : 1
        context?.setAlpha(dimmed)
        let choice = PrecisionLine.showsChoice(lines)
        for (i, line) in lines.enumerated() {
            let pill = pillRect(i)
            let shape = NSBezierPath(roundedRect: pill, xRadius: PrecisionLines.pillRadius, yRadius: PrecisionLines.pillRadius)
            let selected = choice && line.selected
            if !choice || !line.offered {
                // Nothing to choose, or a precision not offered: the line is plain text.
            } else if selected {
                context?.setAlpha(1)
                c.selection.setFill(); shape.fill()
                context?.setAlpha(dimmed)
            } else {
                // A precision you can choose sits on the Get button's own faint track; under the pointer a little lighter.
                let fill = hovered == .pill(i) ? c.trackFill.withAlphaComponent(min(1, c.trackFill.alphaComponent * 1.8)) : c.trackFill
                fill.setFill(); shape.fill()
                c.trackStroke.setStroke(); shape.lineWidth = 1
                NSBezierPath(roundedRect: pill.insetBy(dx: 0.5, dy: 0.5), xRadius: PrecisionLines.pillRadius - 0.5, yRadius: PrecisionLines.pillRadius - 0.5).stroke()
            }
            let ink = selected ? c.selectedLabel : c.label
            let label = line.copy.label as NSString
            let labelAttributes: [NSAttributedString.Key: Any] = [.font: PrecisionLines.labelFont, .foregroundColor: ink]
            let labelSize = label.size(withAttributes: labelAttributes)
            let x = pill.minX + PrecisionLines.padding
            label.draw(at: NSPoint(x: x, y: pill.midY - labelSize.height / 2), withAttributes: labelAttributes)
            if line.warned, let triangle = TierSegments.Control.triangle(onGold: selected) {
                let t = triangle.size
                triangle.draw(
                    in: NSRect(x: x + labelSize.width + PrecisionLines.triangleGap, y: pill.midY - t.height / 2, width: t.width, height: t.height),
                    from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
            }
            let size = line.copy.size as NSString
            let sizeAttributes: [NSAttributedString.Key: Any] = [.font: PrecisionLines.sizeFont, .foregroundColor: ink]
            let sizeSize = size.size(withAttributes: sizeAttributes)
            size.draw(at: NSPoint(x: pill.maxX - PrecisionLines.padding - sizeSize.width, y: pill.midY - sizeSize.height / 2), withAttributes: sizeAttributes)
            // The precision that runs: a white edge and a faint light just outside it.
            if choice, line.loaded {
                context?.setAlpha(1)
                let edge = NSBezierPath(roundedRect: pill.insetBy(dx: 0.75, dy: 0.75), xRadius: PrecisionLines.pillRadius - 0.75, yRadius: PrecisionLines.pillRadius - 0.75)
                edge.lineWidth = 1.5
                TierControl.loadedEdge.setStroke(); edge.stroke()
                context?.setAlpha(dimmed)
            }
            Symbol.draw("trash", centredIn: trashRect(i), size: PrecisionLines.trashSize, color: trashColor(line, hovered: hovered == .trash(i), colors: c))
        }
        if let boltRect {
            context?.setAlpha(1)
            Symbol.draw("bolt.fill", centredIn: boltRect, size: TableMetrics.font(12), weight: .semibold, color: TierControl.boltColor(hot: hot))
        }
    }
    /// The trash: the label's colour, a little quieter at rest, full under the pointer; faint while its delete is refused.
    func trashColor(_ line: PrecisionLine, hovered: Bool, colors c: TierSegments.Control.Colors) -> NSColor {
        guard line.copy.enabled else { return c.off }
        return hovered ? c.label.withAlphaComponent(1) : c.label.withAlphaComponent(0.7)
    }

    // MARK: Accessibility

    private var accessibilityParts: [Element]?
    /// Each line's pill and trash as buttons.
    final class Element: NSAccessibilityElement {
        weak var owner: PrecisionLinesView?
        var part: Part = .bolt
        override func accessibilityPerformPress() -> Bool { owner?.press(part) ?? false }
        override func isAccessibilityEnabled() -> Bool {
            guard let owner else { return false }
            switch part {
            case .pill(let i): return owner.lines.indices.contains(i) && owner.lines[i].choosable && !owner.locked
            case .trash(let i): return owner.lines.indices.contains(i) && owner.lines[i].copy.enabled
            case .bolt: return false
            }
        }
        /// In screen coordinates, converted through the owner's own (flipped) space, so no flip is done by hand.
        override func accessibilityFrame() -> NSRect {
            guard let owner, let window = owner.window else { return .zero }
            let rect: NSRect
            switch part {
            case .pill(let i): rect = owner.pillRect(i)
            case .trash(let i): rect = owner.trashRect(i)
            case .bolt: rect = owner.boltRect ?? .zero
            }
            return window.convertToScreen(owner.convert(rect, to: nil))
        }
    }
    override func isAccessibilityElement() -> Bool { false }
    override func accessibilityRole() -> NSAccessibility.Role? { .group }
    override func accessibilityLabel() -> String? { TierControl.title }
    override func accessibilityChildren() -> [Any]? {
        if let accessibilityParts { return accessibilityParts }
        var parts: [Element] = []
        for (i, line) in lines.enumerated() {
            let pill = Element()
            pill.owner = self; pill.part = .pill(i)
            pill.setAccessibilityParent(self); pill.setAccessibilityRole(.button)
            pill.setAccessibilityLabel("\(line.copy.label), \(line.copy.size)")
            pill.setAccessibilityValue(line.selected ? "chosen" : "")
            pill.setAccessibilityHelp(line.help)
            let trash = Element()
            trash.owner = self; trash.part = .trash(i)
            trash.setAccessibilityParent(self); trash.setAccessibilityRole(.button)
            trash.setAccessibilityLabel("Delete the \(line.copy.label) copy")
            trash.setAccessibilityHelp(line.copy.help)
            parts += [pill, trash]
        }
        accessibilityParts = parts
        return parts
    }
}
