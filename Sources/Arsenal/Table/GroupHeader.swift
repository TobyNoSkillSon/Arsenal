// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// A group's header row as an AppKit view behind the SwiftUI labels: it moves the group as a whole.
//
// - Drag: a mouse-down runs AppKit's own tracking loop (`nextEvent(matching:)`, as NSSlider does), which works inside
//   an NSMenu where SwiftUI gestures and drag-and-drop do not. While dragging, the controller's `drag` names the group
//   and the slot it would land in (the table draws an accent line there and dims the group); mouse-up moves it.
// - Hover: an `.activeAlways` tracking area (Vella's RowAction mechanism, which works during menu tracking) shows two
//   arrows at the row end; a click on one moves the group one place. They are the plain-click equivalent of the drag.
// - Tooltips: AppKit tooltip rects (SwiftUI `.help` never shows in a menu): the title, each figure heading, the arrows.
// - VoiceOver: "Move up" / "Move down" custom actions.

struct GroupHeader: NSViewRepresentable {
    let kindID: String
    let title: String
    let index: Int
    let count: Int
    /// Height of the group (header + rows), for the drop slot under the pointer.
    let extent: CGFloat
    /// Tooltip rects in the header's coordinates (top-left origin).
    let tips: [(CGRect, String)]
    /// Render harness: draw as under the pointer.
    var hovered = false
    let controller: TableController

    static func titleHelp(_ kind: KindSpec) -> String {
        let what: [String: String] = [
            "voices": "Text to speech", "decisions": "Typed yes/no, choice and score answers", "images": "Images from text prompts",
            "chat": "Text chat and instructions", "sound": "Sound effects and music from text",
            "enhancement": "Speech cleanup: noise and echo removed", "documents": "Read pages, tables and handwriting into text",
            "translation": "Translate text", "extraction": "Pull named fields out of text and documents",
            "audio": "Transcribe speech and tell who spoke when", "separation": "Pull voices, instruments or a described sound out of a mix",
            "embeddings": "Rank passages and pictures by how well they match", "perception": "Find and select objects, and see depth, in pictures",
            "restoration": "Upscale pictures and video, and smooth video"
        ]
        if kind.id == "local" { return "Unreadable local folders stay after the model groups until their kind can be read." }
        return (what[kind.id].map { $0 + "\n" } ?? "") + "Drag to reorder"
    }

    func makeNSView(context: Context) -> GroupHeaderView { let v = GroupHeaderView(); update(v); return v }
    func updateNSView(_ v: GroupHeaderView, context: Context) { update(v) }
    private func update(_ v: GroupHeaderView) {
        v.kindID = kindID; v.title = title; v.index = index; v.count = count; v.extent = extent; v.forcedHover = hovered
        v.movable = controller.kinds.contains { $0.id == KindPresentation.parent(kindID) }
        v.tips = tips
        v.onMove = { [weak controller] offset in controller?.moveGroup(kindID, by: offset) }
        v.onDrag = { [weak controller] slot in controller?.drag = slot.map { (kindID, $0) } }
        v.onDrop = { [weak controller] slot in
            controller?.drag = nil; controller?.moveGroup(kindID, toSlot: slot)
        }
        v.refreshToolTips()
        v.needsDisplay = true
    }
}

final class GroupHeaderView: NSView, NSViewToolTipOwner, ToolTipRegions {
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
    var movable = true
    var kindID = ""
    var title = ""
    var index = 0
    var count = 1
    var extent: CGFloat = 0
    var tips: [(CGRect, String)] = []
    var forcedHover = false
    var onMove: ((Int) -> Void)?
    var onDrag: ((Int?) -> Void)?
    var onDrop: ((Int) -> Void)?
    private(set) var hovered = false
    /// Below this distance a press is a click, not a drag.
    static let dragThreshold: CGFloat = 4

    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }
    override var acceptsFirstResponder: Bool { false }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    static let arrowSize: CGFloat = TableMetrics.pt(20)
    /// The two arrows, right-aligned in the action column.
    var upRect: NSRect {
        NSRect(
            x: bounds.maxX - ModelTable.W.rowPadding - 2 * Self.arrowSize - TableMetrics.pt(4), y: (bounds.height - Self.arrowSize) / 2,
            width: Self.arrowSize, height: Self.arrowSize)
    }
    var downRect: NSRect { upRect.offsetBy(dx: Self.arrowSize + TableMetrics.pt(4), dy: 0) }
    private var showsHover: Bool { hovered || forcedHover }

    // MARK: Hover
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
    override func resetCursorRects() { addCursorRect(bounds, cursor: movable ? .openHand : .arrow) }

    // MARK: Click and drag
    override func mouseDown(with event: NSEvent) {
        guard movable else { return }
        let point = convert(event.locationInWindow, from: nil)
        if index > 0, upRect.insetBy(dx: -2, dy: -4).contains(point) { onMove?(-1); HostRefresh.after(self); return }
        if index < count - 1, downRect.insetBy(dx: -2, dy: -4).contains(point) { onMove?(1); HostRefresh.after(self); return }
        guard let window else { return }
        let start = event.locationInWindow
        var slot: Int?
        while let next = window.nextEvent(matching: [.leftMouseDragged, .leftMouseUp]) {
            if next.type == .leftMouseUp {
                if slot != nil { slot = dropSlot(atWindowPoint: next.locationInWindow) }
                break
            }
            let p = next.locationInWindow
            if slot == nil, hypot(p.x - start.x, p.y - start.y) < Self.dragThreshold { continue }
            let s = dropSlot(atWindowPoint: p)
            if s != slot { slot = s; onDrag?(s); refreshHost() }
        }
        guard let slot else { return }
        onDrop?(slot)
        HostRefresh.after(self)
    }

    /// Redraw the table at once: inside the tracking loop the run loop only turns for the events we ask for.
    private func refreshHost() {
        var host: NSView = self
        while let parent = host.superview { host = parent }
        host.layoutSubtreeIfNeeded()
        host.displayIfNeeded()
    }

    /// The slot (0 = before the first group … count = after the last) for a pointer: the first group whose middle lies
    /// below the pointer, from every group header in the same table.
    func dropSlot(atWindowPoint point: NSPoint) -> Int {
        var root: NSView = self
        while let parent = root.superview { root = parent }
        let headers = Self.headers(in: root)
        let y = root.convert(point, from: nil).y
        let flipped = root.isFlipped
        let spans: [(top: CGFloat, extent: CGFloat)] = headers.map { h in
            let r = h.convert(h.bounds, to: root)
            return (flipped ? r.minY : root.bounds.height - r.maxY, h.extent)
        }
        let pointerY = flipped ? y : root.bounds.height - y
        return Self.slot(pointerY: pointerY, spans: spans)
    }
    /// Pure slot rule (unit-tested): spans in table order, y growing downwards.
    static func slot(pointerY: CGFloat, spans: [(top: CGFloat, extent: CGFloat)]) -> Int {
        for (i, s) in spans.enumerated() where pointerY < s.top + s.extent / 2 { return i }
        return spans.count
    }
    static func headers(in view: NSView) -> [GroupHeaderView] {
        var found: [GroupHeaderView] = []
        func walk(_ v: NSView) { if let h = v as? GroupHeaderView { found.append(h) }; v.subviews.forEach(walk) }
        walk(view)
        return found.sorted { $0.index < $1.index }
    }

    // MARK: Tooltips
    func refreshToolTips() {
        removeAllToolTips()
        for (i, tip) in tips.enumerated() { addToolTip(tip.0, owner: self, userData: UnsafeMutableRawPointer(bitPattern: i + 1)) }
        addToolTip(upRect, owner: self, userData: UnsafeMutableRawPointer(bitPattern: 1000))
        addToolTip(downRect, owner: self, userData: UnsafeMutableRawPointer(bitPattern: 1001))
    }
    override func setFrameSize(_ newSize: NSSize) { super.setFrameSize(newSize); refreshToolTips() }
    func view(_ view: NSView, stringForToolTip tag: NSView.ToolTipTag, point: NSPoint, userData data: UnsafeMutableRawPointer?) -> String {
        toolTipText(at: point)
    }
    func toolTipText(at point: NSPoint) -> String {
        if !movable { return tips.first { $0.0.contains(point) }?.1 ?? "" }
        if upRect.contains(point) { return showsHover && index > 0 ? "Move up" : "" }
        if downRect.contains(point) { return showsHover && index < count - 1 ? "Move down" : "" }
        return tips.first { $0.0.contains(point) }?.1 ?? ""
    }

    // MARK: Drawing: a faint band and the arrows under the pointer; nothing otherwise.
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif
        guard showsHover, movable else { return }
        ArsenalPalette.onMenu.overlay(0.06).setFill()
        NSBezierPath(roundedRect: bounds, xRadius: ModelTable.rowCorner, yRadius: ModelTable.rowCorner).fill()
        drawArrow("chevron.up", in: upRect, enabled: index > 0)
        drawArrow("chevron.down", in: downRect, enabled: index < count - 1)
    }
    private func drawArrow(_ name: String, in rect: NSRect, enabled: Bool) {
        Symbol.draw(name, centredIn: rect, size: TableMetrics.font(11), weight: .semibold, color: enabled ? ArsenalPalette.onMenu.t2 : ArsenalPalette.onMenu.t4)
    }

    // MARK: Accessibility
    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .group }
    override func accessibilityLabel() -> String? { "\(title) group" }
    override func accessibilityCustomActions() -> [NSAccessibilityCustomAction]? {
        guard movable else { return [] }
        var actions: [NSAccessibilityCustomAction] = []
        if index > 0 {
            actions.append(
                NSAccessibilityCustomAction(name: "Move up") { [weak self] in
                    guard let self else { return false }
                    onMove?(-1); HostRefresh.after(self); return true
                })
        }
        if index < count - 1 {
            actions.append(
                NSAccessibilityCustomAction(name: "Move down") { [weak self] in
                    guard let self else { return false }
                    onMove?(1); HostRefresh.after(self); return true
                })
        }
        return actions
    }
}
