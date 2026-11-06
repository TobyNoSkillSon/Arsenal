// Ported from Projects/Vella Sources/Vella/ModelsTable/TooltipCell.swift @ f4c5c34 (Vella; relicensed MIT by its author, 2 Oct 2026),
// plus `cellTooltip`: see the note below the original one.
import AppKit
import SwiftUI

// Hover text for SwiftUI cells hosted inside an NSMenu (a table in an NSHostingView set as an NSMenuItem's view).
// Shared verbatim by Verdict, Vella and Vireo: AppKit and SwiftUI only, no app types.
//
// SwiftUI's `.help` never shows in that setting: while NSMenu tracks the mouse, only AppKit's tooltip manager runs,
// and it looks for an NSView with a `toolTip` under the pointer (the Bits segmented control, an NSSegmentedControl,
// is why its segment tooltips work). `.appKitTooltip(text)` puts a transparent NSView carrying that `toolTip` behind
// the cell, sized to the cell's frame; it draws nothing.
//
// Use it on non-interactive cells only (text, numbers, icons). AppKit hit-tests NSView subviews before the hosting
// view's own SwiftUI content, so a click on the cell reaches the tooltip view first; it forwards mouse events to the
// hosting view, but a SwiftUI Button or segmented control must not sit over it. Nested cells work: a cell's tooltip
// view is added before its content's, so an inner cell's own tooltip wins inside its rect.

// Hover help: on this macOS the hosting view hit-tests its own
// SwiftUI content first, so a tooltip view behind a cell's text or icon is never the view under the pointer, and AppKit's
// tooltip manager, which shows only the hit view's tooltip, shows nothing (TableHoverTests: 31 table hovers unreachable).
// `.cellTooltip(text)` puts the same view OVER the cell instead, where the hit test reaches it; it still forwards mouse
// events, so it is for non-interactive cells only, and cells carrying one must not nest (the outer would cover the inner).

/// One tooltip region a view registered with AppKit (`addToolTip(_:owner:userData:)`).
struct ToolTipRegion: Equatable {
    var rect: NSRect
    var userData: UnsafeMutableRawPointer?
}
/// A view that registers tooltip regions with an owner and keeps the list, so the hover tests check what AppKit was
/// actually given (each region's tag, rect and user data) rather than only what the owner would answer.
@MainActor protocol ToolTipRegions: NSView {
    var toolTipRegions: [NSView.ToolTipTag: ToolTipRegion] { get }
}

/// A transparent, hit-testable NSView whose only job is to carry `toolTip` for AppKit's tooltip manager.
struct TooltipCell: NSViewRepresentable {
    /// nil or empty: no tooltip.
    var text: String?

    final class TooltipView: NSView {
        override var isOpaque: Bool { false }
        override var acceptsFirstResponder: Bool { false }
        override func draw(_ dirtyRect: NSRect) {}
        // Not a control: a click lands where it would have without the tooltip view.
        override func mouseDown(with event: NSEvent) { nextResponder?.mouseDown(with: event) }
        override func mouseUp(with event: NSEvent) { nextResponder?.mouseUp(with: event) }
        override func mouseDragged(with event: NSEvent) { nextResponder?.mouseDragged(with: event) }
        override func rightMouseDown(with event: NSEvent) { nextResponder?.rightMouseDown(with: event) }
    }

    func makeNSView(context: Context) -> TooltipView {
        let view = TooltipView()
        view.toolTip = Self.trimmed(text)
        return view
    }
    func updateNSView(_ view: TooltipView, context: Context) {
        let next = Self.trimmed(text)
        if view.toolTip != next { view.toolTip = next }
    }
    /// The cell's whole frame, whatever SwiftUI proposes (an NSView has no intrinsic size of its own).
    func sizeThatFits(_ proposal: ProposedViewSize, nsView: TooltipView, context: Context) -> CGSize? {
        proposal.replacingUnspecifiedDimensions(by: .zero)
    }
    static func trimmed(_ text: String?) -> String? {
        guard let text = text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return nil }
        return text
    }
}

extension View {
    /// AppKit hover text for a non-interactive cell in a menu-hosted SwiftUI view (see the note above). nil or an
    /// empty string means no tooltip.
    func appKitTooltip(_ text: String?) -> some View {
        background(TooltipCell(text: text))
    }
    /// AppKit hover text over a non-interactive cell in a menu-hosted SwiftUI view, where AppKit's hit test reaches it
    /// (the Models table; see the Arsenal note above). nil or an empty string means no tooltip.
    func cellTooltip(_ text: String?) -> some View {
        overlay(TooltipCell(text: text))
    }
    /// `cellTooltip` for a cell of a row that has `rowTooltips()`: the same TooltipView over the cell, but as an AppKit
    /// subview of the row's one tooltip view instead of a SwiftUI platform view of its own. A row of the Models table has
    /// five; SwiftUI walks every platform view each time the table moves into a new menu window (every Models open), so
    /// a long table opened slowly (6 Nov 2026).
    func rowTooltip(_ text: String?) -> some View {
        anchorPreference(key: RowTooltips.Key.self, value: .bounds) { [RowTooltips.Anchored(text: text, bounds: $0)] }
    }
    /// Carries the `rowTooltip` hovers inside this view.
    func rowTooltips() -> some View {
        overlayPreferenceValue(RowTooltips.Key.self) { tips in
            GeometryReader { proxy in
                RowTooltips(tips: tips.map { RowTooltips.Tip(text: TooltipCell.trimmed($0.text), frame: proxy[$0.bounds]) }, size: proxy.size)
            }
        }
        // Used here: the row's hovers go no further up the table.
        .preference(key: RowTooltips.Key.self, value: [])
    }
}

/// One view carrying the hovers of a row's cells (`rowTooltip`): a TooltipCell.TooltipView per cell, at the cell's frame,
/// as its AppKit subviews. Its own hit test reaches only those, so the controls under it get their clicks.
struct RowTooltips: NSViewRepresentable {
    struct Anchored { var text: String?; var bounds: Anchor<CGRect> }
    struct Key: PreferenceKey {
        static let defaultValue: [Anchored] = []
        static func reduce(value: inout [Anchored], nextValue: () -> [Anchored]) { value.append(contentsOf: nextValue()) }
    }
    struct Tip: Equatable { var text: String?; var frame: CGRect }
    var tips: [Tip]
    /// The row's size in the coordinates of `tips`' frames.
    var size: CGSize

    final class RowTooltipsView: NSView {
        override var isFlipped: Bool { true }
        override var isOpaque: Bool { false }
        override var acceptsFirstResponder: Bool { false }
        override func draw(_ dirtyRect: NSRect) {}
        /// Only a cell's tooltip view, never this view itself: elsewhere the row's own content is under the pointer.
        override func hitTest(_ point: NSPoint) -> NSView? {
            let hit = super.hitTest(point)
            return hit === self ? nil : hit
        }
        override func setFrameSize(_ newSize: NSSize) { super.setFrameSize(newSize); needsLayout = true }
        override func setBoundsSize(_ newSize: NSSize) { super.setBoundsSize(newSize); needsLayout = true }
        var tips: [Tip] = [] { didSet { if tips != oldValue { needsLayout = true } } }
        var size: CGSize = .zero { didSet { if size != oldValue { needsLayout = true } } }
        /// Each cell's view at its frame, scaled with the row when the menu fits a shrunk table (`FittedMenuContent`).
        override func layout() {
            super.layout()
            let shown = tips.filter { $0.text != nil && !$0.frame.isEmpty }
            while subviews.count > shown.count { subviews.last?.removeFromSuperview() }
            while subviews.count < shown.count { addSubview(TooltipCell.TooltipView()) }
            let sx = size.width > 0 ? bounds.width / size.width : 1, sy = size.height > 0 ? bounds.height / size.height : 1
            for (tip, cell) in zip(shown, subviews) {
                let f = tip.frame
                let frame = CGRect(x: f.minX * sx, y: f.minY * sy, width: f.width * sx, height: f.height * sy)
                if cell.frame != frame { cell.frame = frame }
                if cell.toolTip != tip.text { cell.toolTip = tip.text }
            }
        }
    }

    func makeNSView(context: Context) -> RowTooltipsView { RowTooltipsView() }
    func updateNSView(_ view: RowTooltipsView, context: Context) {
        view.tips = tips
        view.size = size
        view.layoutSubtreeIfNeeded()
    }
    func sizeThatFits(_ proposal: ProposedViewSize, nsView: RowTooltipsView, context: Context) -> CGSize? {
        proposal.replacingUnspecifiedDimensions(by: .zero)
    }
}
