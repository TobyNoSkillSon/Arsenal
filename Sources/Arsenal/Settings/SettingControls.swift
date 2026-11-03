// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// The settings card's controls as AppKit views (a drawn switch, segments, NSSlider, NSTextField, the foot's wide button),
// drawn in the menu palette on the menu's own material: fields are overlays of the material
// (field fill, raised border, corner 6), a chosen segment is the browser's pill (accent tint and outline). They take a
// plain mouseDown and draw their own state, which both work while AppKit tracks the menu. Each carries its `toolTip`.

enum SettingControl {
    /// The palette every control draws with: the menu's (overlays on the menu's material), as the Models table's cells.
    @MainActor static var pal: ArsenalPalette { ArsenalPalette.onMenu }
    static let corner: CGFloat = 6

    /// Target/action for a control, with the newest closure.
    @MainActor final class Target: NSObject, NSTextFieldDelegate {
        var action: (NSControl) -> Void = { _ in }
        /// A number field's arrow keys: the text one step up (+1) or down (-1), nil to leave it.
        var step: ((Int, String) -> String?)?
        @objc func fire(_ sender: NSControl) { action(sender) }
        /// Editing: the field editor draws no box of its own (the field's layer is the box), and the border takes the
        /// accent so the field being typed in is plain to see.
        func controlTextDidBeginEditing(_ notification: Notification) {
            (notification.userInfo?["NSFieldEditor"] as? NSTextView)?.drawsBackground = false
            (notification.object as? MenuField)?.editing = true
        }
        func controlTextDidEndEditing(_ notification: Notification) {
            (notification.object as? MenuField)?.editing = false
            SettingsDrafts.drop(self)
        }
        /// Each keystroke keeps the text as a draft, committed if the menu closes before the field ends editing.
        func controlTextDidChange(_ notification: Notification) {
            guard let field = notification.object as? NSTextField else { return }
            let text = (notification.userInfo?["NSFieldEditor"] as? NSTextView)?.string ?? field.stringValue
            SettingsDrafts.hold(self) { [weak self, weak field] in
                guard let self, let field else { return }
                field.stringValue = text
                self.action(field)
            }
        }
        func control(_ control: NSControl, textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            // Return commits the field and keeps it focused: in a menu it must not reach the menu, which would close.
            if selector == #selector(NSResponder.insertNewline(_:)) {
                (control as? NSTextField)?.stringValue = textView.string
                SettingsDrafts.drop(self)
                action(control)
                return true
            }
            guard let step, selector == #selector(NSResponder.moveUp(_:)) || selector == #selector(NSResponder.moveDown(_:)) else { return false }
            guard let text = step(selector == #selector(NSResponder.moveUp(_:)) ? 1 : -1, textView.string) else { return true }
            textView.string = text; textView.selectAll(nil)
            (control as? NSTextField)?.stringValue = text
            action(control)
            return true
        }
    }

    struct Switch: NSViewRepresentable {
        var label = ""
        let on: Bool; let enabled: Bool; let tooltip: String; let change: (Bool) -> Void
        func makeCoordinator() -> Target { Target() }
        func makeNSView(context: Context) -> MenuSwitch {
            let v = MenuSwitch(); v.target = context.coordinator; v.action = #selector(Target.fire(_:)); return v
        }
        /// The switch's frame (the size a `.small` NSSwitch draws).
        static let size = CGSize(width: 44, height: 20)
        func sizeThatFits(_ proposal: ProposedViewSize, nsView v: MenuSwitch, context: Context) -> CGSize? { Self.size }
        func updateNSView(_ v: MenuSwitch, context: Context) {
            context.coordinator.action = { change(($0 as? MenuSwitch)?.state == .on) }
            v.state = on ? .on : .off; v.isEnabled = enabled; v.toolTip = tooltip; v.label = label
            v.needsDisplay = true
        }
    }

    /// Drawn segments (MenuSegments): text or an SF Symbol per segment, each with its own tooltip.
    struct Segments: NSViewRepresentable {
        var label = ""
        let labels: [String]
        var icons: [String?] = []
        var tips: [String] = []
        let selected: Int?; let enabled: Bool; let tooltip: String; let change: (Int) -> Void
        func makeCoordinator() -> Target { Target() }
        func makeNSView(context: Context) -> MenuSegments {
            let v = MenuSegments(); v.trackingMode = .selectOne; v.target = context.coordinator; v.action = #selector(Target.fire(_:))
            return v
        }
        /// The row decides the width; the segments' own titles never widen it.
        func sizeThatFits(_ proposal: ProposedViewSize, nsView: MenuSegments, context: Context) -> CGSize? {
            CGSize(width: proposal.width ?? CGFloat(labels.count) * 48, height: proposal.height ?? SettingsPane.M.rowHeight)
        }
        func updateNSView(_ v: MenuSegments, context: Context) {
            context.coordinator.action = { if let s = $0 as? NSSegmentedControl, s.selectedSegment >= 0 { change(s.selectedSegment) } }
            v.update(labels: labels, icons: icons, tips: tips.isEmpty ? labels : tips)
            v.selectedSegment = selected ?? -1; v.isEnabled = enabled; v.toolTip = tooltip; v.setAccessibilityLabel(label)
            v.needsDisplay = true
        }
    }

    struct Slider: NSViewRepresentable {
        var label = ""
        let value: Double; let min: Double; let max: Double; let step: Double?; let enabled: Bool; let tooltip: String; let change: (Double) -> Void
        func makeCoordinator() -> Target { Target() }
        func makeNSView(context: Context) -> NSSlider {
            #if DEBUG
                let v = MenuSlider(value: value, minValue: min, maxValue: max, target: context.coordinator, action: #selector(Target.fire(_:)))
            #else
                let v = NSSlider(value: value, minValue: min, maxValue: max, target: context.coordinator, action: #selector(Target.fire(_:)))
            #endif
            v.controlSize = .small; v.isContinuous = true; return v
        }
        func updateNSView(_ v: NSSlider, context: Context) {
            context.coordinator.action = { change($0.doubleValue) }
            v.minValue = min; v.maxValue = max
            if v.doubleValue != value { v.doubleValue = value }
            v.isEnabled = enabled; v.toolTip = tooltip; v.setAccessibilityLabel(label)
        }
    }

    /// One-line text; commits on Enter or when it loses focus; an invalid number reverts (the caller ignores it and
    /// the next update puts the stored value back).
    struct Field: NSViewRepresentable {
        var label = ""
        let text: String; var placeholder = ""; let width: CGFloat; let alignment: NSTextAlignment; let enabled: Bool; let tooltip: String
        var step: ((Int, String) -> String?)?
        let commit: (String) -> Void
        func makeCoordinator() -> Target { Target() }
        /// The row decides the size: a text field's own height would otherwise win over its frame.
        func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSTextField, context: Context) -> CGSize? {
            CGSize(width: proposal.width ?? width, height: proposal.height ?? SettingsPane.M.fieldHeight)
        }
        func makeNSView(context: Context) -> NSTextField {
            let v = MenuField(string: text)
            v.cell = InsetFieldCell(textCell: text)
            // The image window's field: the layer draws the fill and the border, edge to edge; the cell only insets the text.
            v.controlSize = .small; v.isBezeled = false; v.drawsBackground = false; v.focusRingType = .none
            v.wantsLayer = true; v.layer?.cornerRadius = SettingControl.corner; v.layer?.borderWidth = 1
            v.lineBreakMode = .byTruncatingTail; v.usesSingleLineMode = true
            v.font = alignment == .right ? .monospacedDigitSystemFont(ofSize: SettingsPane.M.controlSize, weight: .regular) : .systemFont(ofSize: SettingsPane.M.controlSize)
            v.cell?.sendsActionOnEndEditing = true
            v.target = context.coordinator; v.action = #selector(Target.fire(_:)); v.delegate = context.coordinator
            return v
        }
        func updateNSView(_ v: NSTextField, context: Context) {
            context.coordinator.action = { commit($0.stringValue) }; context.coordinator.step = step
            (v as? MenuField)?.paint()
            if v.currentEditor() == nil, v.stringValue != text { v.stringValue = text }
            // The placeholder sits where a value would: "Random" right-aligned like a pinned seed.
            let align = NSMutableParagraphStyle(); align.alignment = alignment
            v.placeholderAttributedString = NSAttributedString(
                string: placeholder,
                attributes: [.font: v.font ?? NSFont.systemFont(ofSize: SettingsPane.M.controlSize), .foregroundColor: SettingControl.pal.t3, .paragraphStyle: align])
            v.textColor = SettingControl.pal.t1
            v.alignment = alignment; v.isEnabled = enabled; v.toolTip = tooltip; v.setAccessibilityLabel(label)
        }
    }

    /// The form's foot: the row's own action (Reload, or Get when an environment is missing) as the table's own small
    /// bordered button, at the trailing edge.
    struct Action: NSViewRepresentable {
        let title: String; let enabled: Bool; let tooltip: String; let click: () -> Void
        func makeNSView(context: Context) -> ActionButtonView { ActionButtonView() }
        func updateNSView(_ v: ActionButtonView, context: Context) {
            v.title = title; v.enabled = enabled; v.toolTip = tooltip; v.click = click; v.needsDisplay = true
        }
        func sizeThatFits(_ proposal: ProposedViewSize, nsView: ActionButtonView, context: Context) -> CGSize? {
            CGSize(width: RowAction.buttonWidth, height: proposal.height ?? SettingsPane.M.rowHeight)
        }
    }

    /// A small symbol button (voice preview, random seed, choose a file, restore defaults).
    struct Glyph: NSViewRepresentable {
        let symbol: String; let tooltip: String; var label: String?; var enabled = true
        /// Drawn against the frame's trailing edge (a glyph that ends a row) instead of centred.
        var trailing = false
        let click: () -> Void
        func makeNSView(context: Context) -> GlyphButton { GlyphButton() }
        func updateNSView(_ v: GlyphButton, context: Context) {
            v.symbol = symbol; v.toolTip = tooltip; v.label = label; v.enabled = enabled; v.trailing = trailing; v.click = click; v.needsDisplay = true
        }
    }
}

/// A settings field whose frame is its drawn edge (no alignment insets), so it lines up with the column.
final class MenuField: NSTextField {
    #if DEBUG
        private(set) var drawPasses = 0
        override func draw(_ dirtyRect: NSRect) { drawPasses += 1; super.draw(dirtyRect) }
    #endif
    override var alignmentRectInsets: NSEdgeInsets { NSEdgeInsetsZero }
    var editing = false { didSet { paint() } }
    /// The box: the field fill and the raised border, the accent while typing.
    func paint() {
        let p = SettingControl.pal
        layer?.backgroundColor = p.field.cgColor
        layer?.borderColor = (editing ? p.accent.withAlphaComponent(0.8) : p.raisedBorder).cgColor
    }
    override func becomeFirstResponder() -> Bool {
        let ok = super.becomeFirstResponder()
        if ok { (currentEditor() as? NSTextView)?.drawsBackground = false; editing = true }
        return ok
    }
}

/// The settings fields' cell: text inset from the field's edges and centred vertically, drawing and editing alike.
final class InsetFieldCell: NSTextFieldCell {
    let inset: CGFloat = 8
    override init(textCell string: String) {
        super.init(textCell: string); isEditable = true; isSelectable = true; isScrollable = true; wraps = false; usesSingleLineMode = true
    }
    required init(coder: NSCoder) { super.init(coder: coder) }
    private func inner(_ rect: NSRect) -> NSRect {
        let r = rect.insetBy(dx: inset, dy: 0), h = cellSize(forBounds: r).height
        return NSRect(x: r.minX, y: r.minY + max(0, (r.height - h) / 2), width: r.width, height: min(h, r.height))
    }
    override func drawingRect(forBounds rect: NSRect) -> NSRect { inner(super.drawingRect(forBounds: rect)) }
    override func edit(withFrame rect: NSRect, in controlView: NSView, editor textObj: NSText, delegate: Any?, event: NSEvent?) {
        super.edit(withFrame: inner(rect), in: controlView, editor: textObj, delegate: delegate, event: event)
    }
    override func select(withFrame rect: NSRect, in controlView: NSView, editor textObj: NSText, delegate: Any?, start selStart: Int, length selLength: Int) {
        super.select(withFrame: inner(rect), in: controlView, editor: textObj, delegate: delegate, start: selStart, length: selLength)
    }
}

@MainActor enum SettingsDrafts {
    private static var pending: [ObjectIdentifier: () -> Void] = [:]
    static func hold(_ owner: AnyObject, commit: @escaping () -> Void) { pending[ObjectIdentifier(owner)] = commit }
    static func drop(_ owner: AnyObject) { pending[ObjectIdentifier(owner)] = nil }
    static var count: Int { pending.count }
    static func commitAll() {
        let all = pending
        pending = [:]
        for commit in all.values { commit() }
    }
}

/// The table's action button (RowActionView's bordered button: a faint fill, a hairline border, the title in the table's
/// font), centred vertically in its row.
final class ActionButtonView: NSView {
    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    var title = ""; var enabled = true; var click: () -> Void = {}
    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func mouseDown(with event: NSEvent) { _ = accessibilityPerformPress() }
    var buttonRect: NSRect {
        NSRect(x: bounds.maxX - RowAction.buttonWidth, y: ((bounds.height - RowAction.buttonHeight) / 2).rounded(), width: RowAction.buttonWidth, height: RowAction.buttonHeight)
    }
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif
        let pal = ArsenalPalette.onMenu
        let button = buttonRect
        let shape = NSBezierPath(roundedRect: button.insetBy(dx: 0.5, dy: 0.5), xRadius: RowAction.cornerRadius, yRadius: RowAction.cornerRadius)
        NSGraphicsContext.current?.cgContext.setAlpha(enabled ? 1 : 0.4)
        pal.overlay(0.10).setFill(); shape.fill()
        pal.overlay(0.22).setStroke(); shape.lineWidth = 1; shape.stroke()
        let text = NSAttributedString(string: title, attributes: [.font: RowActionView.titleFont, .foregroundColor: pal.t1])
        let size = text.size()
        text.draw(at: NSPoint(x: button.midX - size.width / 2, y: button.midY - size.height / 2))
    }
    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .button }
    override func accessibilityLabel() -> String? { title }
    override func isAccessibilityEnabled() -> Bool { enabled }
    override func accessibilityPerformPress() -> Bool { guard enabled else { return false }; click(); return true }
}

final class GlyphButton: NSView {
    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    var symbol = "speaker.wave.2"; var label: String?; var enabled = true; var trailing = false; var click: () -> Void = {}
    override var isFlipped: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func mouseDown(with event: NSEvent) { guard enabled else { return }; click() }
    static let size: CGFloat = 13
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif
        // Off, the glyph takes the unavailable colour itself: an image drawn at full fraction ignores the context's alpha.
        let color = enabled ? SettingControl.pal.t2 : SettingControl.pal.t4
        guard trailing, let width = Symbol.image(symbol, size: Self.size, color: color)?.size.width else {
            return Symbol.draw(symbol, centredIn: bounds, size: Self.size, color: color)
        }
        // The symbol's image carries about a point of empty margin: drawn that much further out, its ink ends on the edge.
        let x = bounds.maxX - width + 1
        Symbol.draw(symbol, centredIn: NSRect(x: x, y: bounds.minY, width: width, height: bounds.height), size: Self.size, color: color)
    }
    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .button }
    override func accessibilityLabel() -> String? { label ?? (symbol == "speaker.wave.2" ? "Preview voice" : "Stop preview") }
    override func isAccessibilityEnabled() -> Bool { enabled }
    override func accessibilityPerformPress() -> Bool { guard enabled else { return false }; click(); return true }
}

/// Segments drawn as the browser's pills sit in a field: the chosen cell tinted and outlined with the accent, its
/// label in the accent text colour. A cell shows its SF Symbol when it has one, else its text; every cell has its
/// tooltip. None selected (values that match no preset) leaves every cell plain. Clicks select directly.
final class MenuSegments: NSSegmentedControl {
    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    private(set) var icons: [String?] = []
    override var isFlipped: Bool { true }
    override var intrinsicContentSize: NSSize { NSSize(width: NSView.noIntrinsicMetric, height: SettingsPane.M.fieldHeight) }
    override var alignmentRectInsets: NSEdgeInsets { NSEdgeInsetsZero }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    func update(labels: [String], icons: [String?], tips: [String]) {
        if segmentCount != labels.count { segmentCount = labels.count }
        self.icons = labels.indices.map { icons.indices.contains($0) ? icons[$0] : nil }
        for (i, l) in labels.enumerated() {
            if label(forSegment: i) != l { setLabel(l, forSegment: i) }
            setToolTip(tips.indices.contains(i) ? tips[i] : l, forSegment: i)
        }
        layoutSegments()
    }
    override func setFrameSize(_ newSize: NSSize) { super.setFrameSize(newSize); layoutSegments() }
    private func layoutSegments() {
        guard segmentCount > 0, bounds.width > 0 else { return }
        let w = floor(bounds.width / CGFloat(segmentCount)) - 1
        for i in 0..<segmentCount where width(forSegment: i) != w { setWidth(w, forSegment: i) }
    }
    func cell(_ i: Int) -> NSRect {
        let pitch = bounds.width / CGFloat(max(segmentCount, 1))
        return NSRect(x: CGFloat(i) * pitch, y: 0, width: pitch, height: bounds.height)
    }
    override func mouseDown(with event: NSEvent) {
        guard isEnabled else { return }
        let point = convert(event.locationInWindow, from: nil)
        guard let index = (0..<segmentCount).first(where: { cell($0).contains(point) }) else { return }
        selectedSegment = index
        sendAction(action, to: target)
        needsDisplay = true
    }
    /// The segments' track inside the row: as tall as the fields beside it.
    var track: NSRect {
        let h = min(bounds.height, SettingsPane.M.fieldHeight)
        return NSRect(x: 0.5, y: ((bounds.height - h) / 2).rounded() + 0.5, width: bounds.width - 1, height: h - 1)
    }
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif
        let p = SettingControl.pal
        NSGraphicsContext.current?.cgContext.setAlpha(isEnabled ? 1 : 0.45)
        let t = track, radius = SettingControl.corner
        let field = NSBezierPath(roundedRect: t, xRadius: radius, yRadius: radius)
        p.field.setFill(); field.fill()
        p.raisedBorder.setStroke(); field.lineWidth = 1; field.stroke()
        for i in 0..<segmentCount {
            let c = cell(i).intersection(t)
            let selected = i == selectedSegment
            if selected {
                let shape = NSBezierPath(roundedRect: c.insetBy(dx: 1.5, dy: 1.5), xRadius: radius - 1.5, yRadius: radius - 1.5)
                p.accent.withAlphaComponent(0.2).setFill(); shape.fill()
                p.accent.withAlphaComponent(0.55).setStroke(); shape.lineWidth = 1; shape.stroke()
            } else if i > 0, i - 1 != selectedSegment {
                p.raisedBorder.setFill()
                NSRect(x: c.minX - 0.5, y: c.minY + 6, width: 1, height: c.height - 12).fill()
            }
            let color = selected ? p.accentText : p.t2
            if let icon = icons[i], Symbol.image(icon, size: 13, color: color) != nil {
                // An image ignores the context's alpha: a locked control's icons are dimmed in their colour.
                Symbol.draw(icon, centredIn: c, size: 13, weight: selected ? .medium : .regular, color: isEnabled ? color : color.withAlphaComponent(0.45))
            } else {
                let font = NSFont.systemFont(ofSize: SettingsPane.M.controlSize, weight: .medium)
                let style = NSMutableParagraphStyle(); style.alignment = .center; style.lineBreakMode = .byTruncatingTail
                let text = NSAttributedString(string: label(forSegment: i) ?? "", attributes: [.font: font, .foregroundColor: color, .paragraphStyle: style])
                let h = text.size().height
                text.draw(in: NSRect(x: c.minX + 3, y: c.midY - h / 2, width: c.width - 6, height: h))
            }
        }
    }
}

/// An on/off switch that draws its own state: accent track and knob right when on, a neutral track and knob left when
/// off, whether or not its window is key (NSSwitch greys its "on" track in an inactive window, which reads as off). It
/// is a checkbox button underneath (click tracking, `state`, action, accessibility press); its name and switch subrole
/// are set on the cell, the element accessibility exposes.
final class MenuSwitch: NSButton {
    #if DEBUG
        private(set) var drawPasses = 0
    #endif
    convenience init() {
        self.init(frame: NSRect(origin: .zero, size: SettingControl.Switch.size))
        setButtonType(.switch); title = ""; imagePosition = .imageOnly; isBordered = false
        exposed?.setAccessibilitySubrole(.switch)
    }
    /// What accessibility exposes: the button's cell (a control with a cell is itself ignored), a checkbox with the
    /// switch subrole, as NSSwitch reports.
    var exposed: NSCell? { cell }
    /// The setting's name, on the exposed element.
    var label: String = "" { didSet { exposed?.setAccessibilityLabel(label) } }
    override var wantsUpdateLayer: Bool { false }
    override var isFlipped: Bool { true }
    override var intrinsicContentSize: NSSize { SettingControl.Switch.size }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    /// The track: the native small switch's proportions, flush with the frame's trailing edge, so a switch placed at the
    /// control column's trailing edge ends where the pop-ups, values and hairlines end. The whole frame is the hit area.
    var track: NSRect {
        let h = bounds.height - 4, w = min(bounds.width - 2, (h * 1.75).rounded())
        return NSRect(x: bounds.maxX - w - 0.5, y: (bounds.height - h) / 2, width: w, height: h)
    }
    var knob: NSRect {
        let t = track, d = t.height - 4
        return NSRect(x: state == .on ? t.maxX - 2 - d : t.minX + 2, y: t.minY + 2, width: d, height: d)
    }
    override func draw(_ dirtyRect: NSRect) {
        #if DEBUG
            drawPasses += 1
        #endif
        let p = SettingControl.pal
        NSGraphicsContext.current?.cgContext.setAlpha(isEnabled ? 1 : 0.4)
        let t = track
        (state == .on ? NSColor.controlAccentColor : p.overlay(p.dark ? 0.22 : 0.4)).setFill()
        NSBezierPath(roundedRect: t, xRadius: t.height / 2, yRadius: t.height / 2).fill()
        let k = NSBezierPath(ovalIn: knob)
        NSColor.white.setFill(); k.fill()
        if !p.dark { NSColor.black.withAlphaComponent(0.12).setStroke(); k.lineWidth = 0.5; k.stroke() }
    }
}

#if DEBUG
    // Instrument native settings drawing without changing release controls or their event paths.
    final class MenuSlider: NSSlider {
        private(set) var drawPasses = 0
        override func draw(_ dirtyRect: NSRect) { drawPasses += 1; super.draw(dirtyRect) }
    }
#endif
