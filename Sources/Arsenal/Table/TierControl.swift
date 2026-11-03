// Shared from Projects/Vella Sources/Vella/ModelsTable/TierControl.swift @ 796781e (Vella release/2.1.0, the one-row
// control in a 15 Oct screenshot; relicensed MIT by its author, 2 Oct 2026). Sizes, colours, drawing, the in-use
// interlock and the hovers are Vella's; diff this file against Vella's to see the Arsenal differences, each
// marked "Arsenal:" below:
// - No recommendation (Arsenal shows none, AGENTS.md "The Models table"): the chosen cell is always Vella's gold, its
//   chosen-and-recommended state; Vella's blue "chosen, not recommended" fill and gold "recommended" ring are not used.
// - The bolt shows only where the model has custom kernels for this chip (`bolt`); its slot stays on every row so the
//   cells line up. In Vella every model runs the optimized path, so Vella always shows it.
// - `loaded`: a loaded row with another cell chosen (Reload pending) keeps the white loaded edge on the cell that runs
//   now, without fill; the gold moves to the chosen cell. The white edge always marks what runs.
// - `warnings`: int4's quality warning (a triangle, its words only on hover, "clear we are talking only
//   about int4"), drawn where Vella draws its no-matrix-hardware triangle: Vella's orange as a filled triangle with an ink
//   "!" (the table's missing-figure triangle is the same shape in yellow), so it reads on light and dark tracks; on the
//   gold cell it is ink.
// - `lockedHelp`: the interlock line says what is busy (which precision downloads, is made or loads).
// - `removed`: a cell whose weights Arsenal removed to save space (its 16-bit copy, once a smaller version was made and
//   loaded) carries a download arrow where the warning triangle sits, in the label's colour; its hover says what
//   happened and what Get downloads. It stays choosable: it is neither greyed (not offered) nor plain (never got).
// - Light menus: Arsenal follows the system appearance, so the track, labels and dividers have light-menu values.
// - Hovers (15 Oct, 3.4.2): every cell's hover is one of the control's own tooltip regions, independent of whether the
//   cell can be chosen (AppKit's per-segment tooltips did not show a greyed cell's reason). Cells that cannot be chosen
//   are disabled in the cell, for accessibility and every native activation. The bolt's hover sits over it
//   (`cellTooltip`), where AppKit's hit test reaches it.
// - Clicks: `Control.mouseDown` selects the cell under the pointer from the drawn geometry (84e17703: the stock tracking
//   loop disagreed with the hand-drawn cells and never returns inside a tracked menu).
// - Swift 6 isolation annotations; `TierSegments` is internal so tests can reach the control.
import AppKit
import SwiftUI

/// The table's size (Vella's TableMetrics; About 10 % smaller in both directions, proportions kept): one
/// factor for the table and its controls. Lengths round to the half point; font and symbol sizes are exact.
enum TableMetrics {
    nonisolated static let scale: CGFloat = 0.9
    /// A length (width, height, spacing, padding, radius): `base × scale` to the nearest half point.
    nonisolated static func pt(_ base: CGFloat) -> CGFloat { (base * scale * 2).rounded() / 2 }
    /// A font or symbol point size: `base × scale`.
    nonisolated static func font(_ base: CGFloat) -> CGFloat { base * scale }
}

/// The Models table's Precision control: one row of two cells after a bolt slot, e.g. `⚡ [bf16][int4]` (only
/// 16-bit and the int4 that makes sense; int8 is not offered).
/// AppKit and SwiftUI only, no app types; plain values in, one callback out.
///
/// - The row always shows both columns (16, 4), equal width, labelled with the dtype that runs (`labels`, e.g.
///   bf16, fp16, int4). A cell that cannot be chosen is greyed in place, never hidden: the row never shifts.
///   `unavailable` maps it to its one-line reason, which is its tooltip; a click on it does nothing.
/// - Exactly one cell is selected. Clicking a cell reports it; what that means (Get, Load or Reload at that precision) is
///   the app's business.
/// - Cell states: not chosen, the Get button's neutral background; chosen, gold; loaded, that same fill edged in white on
///   the row's blue (Arsenal: with another cell chosen, the loaded cell keeps the white edge alone).
/// - `warnings`: cells with a warning carry the orange triangle (ink "!") after their label; the warning is the first line of
///   their tooltip.
/// - `enabled == false` is the in-use interlock (downloading, making, loading): the row is disabled and every segment's
///   tooltip gains `inUseHelp`.
/// - Each cell's hover is a tooltip region of the segmented control itself (Arsenal; see `Control`), whatever the cell's
///   state. It works inside an NSMenu, where SwiftUI `.help` never shows (TooltipCell.swift).
/// - The geometry is explicit (a regular segmented control, `segmentHeight` high, `cellWidth` per column): a hosted
///   NSControl can take the environment's control size until its first update, so nothing here asks the control for its
///   size and the FIRST layout of a menu is already right.
/// - Scale: AppKit draws a regular segment's bezel 24 pt high whatever its frame, so the segmented control keeps that
///   natural geometry (`Natural`) as its own bounds and its frame is the scaled size. Drawing and hit testing scale with
///   it (a control's bounds are its coordinate system).
struct TierControl: View {
    /// The column header and its tooltip.
    static let title = "Precision"
    static let headerHelp =
        "How the model\u{2019}s weights are stored. 16-bit (fp16 or bf16) is the model as published; int4 is a smaller version made on this Mac that uses less memory and can lose some quality. Speed and peak memory follow the precision you choose."
    /// Arsenal: the bolt marks custom kernels for this chip, not a path every model runs.
    static let boltHelp = "Runs with custom kernels for this Mac's chip"
    /// The interlock line when the app does not say more (Arsenal: the row downloads, makes or loads a precision; the
    /// table passes the exact one as `lockedHelp`).
    static let inUseHelp = "Locked while this model downloads, converts or loads."
    /// Columns, highest precision first: 16 and 4 bits (the cells' identity; `labels` names them). Since 21 Oct, only
    /// 16-bit; int8 is not offered for any model.
    nonisolated static let columns = ["16", "4"]
    /// A segmented control's own coordinate space (its bounds): a regular NSSegmentedControl as AppKit draws it, 24 pt
    /// high, 52 pt per column (a four-letter dtype and the warning triangle), labels at 13 pt medium. Its frame is this
    /// times `TableMetrics.scale`.
    enum Natural {
        nonisolated static let segmentHeight: CGFloat = 24
        nonisolated static let cellWidth: CGFloat = 52
        nonisolated static func segmentsWidth(_ count: Int) -> CGFloat { count == 0 ? 0 : CGFloat(count) * (cellWidth - 1) - 1 }
        static var font: NSFont { .systemFont(ofSize: 13, weight: .medium) }
        /// The warning triangle's point size and its gap after the label.
        nonisolated static let triangleSize: CGFloat = 10, triangleGap: CGFloat = 3
        /// Arsenal: the removed mark is a size smaller and closer, so a 16-bit label keeps its margin in the cell.
        nonisolated static let removedSize: CGFloat = 9, removedGap: CGFloat = 2
    }
    /// The row's height on screen (the regular 24 pt, scaled); the control's own height.
    nonisolated static let segmentHeight: CGFloat = TableMetrics.pt(Natural.segmentHeight)
    nonisolated static let height: CGFloat = segmentHeight
    /// Width of one column on screen. Every cell is this wide.
    nonisolated static let cellWidth: CGFloat = TableMetrics.pt(Natural.cellWidth)
    /// The bolt's slot and the gap after it.
    nonisolated static let iconWidth: CGFloat = TableMetrics.pt(24), iconGap: CGFloat = TableMetrics.pt(6)
    /// Width of the segments on screen: each segment is `cellWidth - 2` wide plus a 1 pt divider, less the outer one.
    nonisolated static func segmentsWidth(_ count: Int) -> CGFloat { count == 0 ? 0 : CGFloat(count) * (cellWidth - 1) - 1 }
    nonisolated static let segments: CGFloat = segmentsWidth(columns.count)
    /// Bolt plus three columns.
    nonisolated static let width: CGFloat = iconWidth + iconGap + CGFloat(columns.count) * cellWidth
    /// The segment labels in the control's own (natural) space; on screen 13 × scale.
    static var font: NSFont { Natural.font }

    /// The label of each column, in `columns` order: the dtype that runs (`bf16`, `int4`); bare bits by default.
    var labels: [String] = columns
    let selected: String?
    let enabled: Bool
    /// Cells that cannot be chosen, each with its one-line reason (shown as its tooltip): greyed in place, never
    /// selectable. A cell becomes selectable as soon as the app stops listing it.
    var unavailable: [String: String] = [:]
    /// The model is loaded: the row sits on the loaded (blue) background.
    var hot = false
    /// Arsenal: the loaded cell when another one is chosen (Reload pending); nil when the chosen cell is the loaded one.
    var loaded: String?
    /// Arsenal: the model has custom kernels for this chip, so the bolt shows.
    var bolt = false
    /// Cells with a warning → the warning (the triangle; the tooltip's first line). Arsenal: int4's quality warning.
    var warnings: [String: String] = [:]
    /// Arsenal: cells whose weights were removed to save space (the download arrow); a warning's triangle wins the slot.
    var removed: Set<String> = []
    /// Arsenal: the interlock line for this row, saying what is busy (`inUseHelp` when the app has nothing more exact).
    var lockedHelp: String = TierControl.inUseHelp
    /// Tooltip per available cell.
    let help: (String) -> String
    let onSelect: (String) -> Void

    /// A segment's tooltip as shown: the app's text, plus the interlock line while in use.
    static func tooltip(_ text: String, enabled: Bool, locked: String = inUseHelp) -> String { enabled ? text : text + "\n" + locked }
    /// An available cell's whole tooltip: its warning first, then the app's text, then the interlock line.
    static func cellTooltip(warning: String?, help: String, enabled: Bool, locked: String = inUseHelp) -> String {
        tooltip([warning, help].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: "\n"), enabled: enabled, locked: locked)
    }

    var body: some View {
        HStack(spacing: Self.iconGap) {
            Group {
                if bolt {
                    Self.bolt(hot: hot).cellTooltip(Self.boltHelp).accessibilityLabel("Custom kernels")
                } else {
                    Color.clear
                }
            }.frame(width: Self.iconWidth, height: Self.segmentHeight)
            TierSegments(
                tiers: Self.columns, labels: labels, selected: selected, enabled: enabled, hot: hot, loaded: loaded,
                unavailable: Set(unavailable.keys), warned: Set(warnings.keys).subtracting(unavailable.keys),
                removed: removed.subtracting(unavailable.keys).subtracting(warnings.keys),
                help: { tier in unavailable[tier] ?? Self.cellTooltip(warning: warnings[tier], help: help(tier), enabled: enabled, locked: lockedHelp) },
                onSelect: { tier in if unavailable[tier] == nil { onSelect(tier) } }
            )
            .frame(width: Self.segments, height: Self.segmentHeight)
        }.frame(width: Self.width, height: Self.height, alignment: .leading)
            .fixedSize()
    }

    /// The bolt's colour (on Vella: "lightning should be like yellow. So that it's fast"): Vella's lemon
    /// yellow, 54° 100 % 63 % (#FFE840), 7:1 on the loaded blue, 13:1 on the dark menu. Arsenal: on the light menu that
    /// yellow is about 1.2:1, so there it takes Vella's knob amber (#B8800A, 3.3:1 on white); on the light menu's solid blue
    /// loaded row the lemon yellow again.
    nonisolated static let boltTint = NSColor(srgbRed: 255 / 255, green: 232 / 255, blue: 64 / 255, alpha: 1)
    nonisolated static let boltTintLight = NSColor(srgbRed: 184 / 255, green: 128 / 255, blue: 10 / 255, alpha: 1)
    @MainActor static func boltColor(hot: Bool) -> NSColor { hot || ArsenalPalette.onMenu.dark ? boltTint : boltTintLight }
    /// The chosen cell (Vella's recommended gold): a warm gold, ≥ 7:1 on the dark menu and ≥ 4.5:1 on the loaded row's blue.
    nonisolated static let gold = NSColor(srgbRed: 236 / 255, green: 190 / 255, blue: 72 / 255, alpha: 1)
    /// The loaded cell's edge and the faint light around it.
    nonisolated static let loadedEdge = NSColor.white
    nonisolated static let loadedLight = NSColor.white.withAlphaComponent(0.35)
    /// The warning triangle's colour, Vella's; on the gold cell, the label's ink.
    nonisolated static let warningColor = NSColor.systemOrange
    /// Dark labels on the gold.
    nonisolated static let ink = NSColor(white: 0.1, alpha: 1)

    @MainActor @ViewBuilder static func bolt(hot: Bool = false) -> some View {
        Image(systemName: "bolt.fill").font(.system(size: TableMetrics.font(15), weight: .semibold)).foregroundStyle(Color(nsColor: boltColor(hot: hot)))
    }
}

/// After a click inside a menu, redraw the hosting view on the next run-loop pass in the menu's tracking mode as well:
/// a SwiftUI table hosted in an NSMenuItem must show the new state while the menu stays open, whatever run-loop mode
/// SwiftUI's own update is scheduled in. Shared with RowAction.swift and the table's cells.
@MainActor enum HostRefresh {
    static func after(_ view: NSView) {
        var root: NSView = view
        while let parent = root.superview { root = parent }
        let host = root
        // The main run loop runs this block on the main thread.
        RunLoop.main.perform(inModes: [.common, .eventTracking, .default]) {
            MainActor.assumeIsolated {
                host.needsLayout = true
                host.layoutSubtreeIfNeeded()
                host.needsDisplay = true
                host.displayIfNeeded()
            }
        }
    }
}

/// The row's segments: a regular NSSegmentedControl with per-segment tooltips, drawn by hand (`Control.draw`).
struct TierSegments: NSViewRepresentable {
    let tiers: [String]
    let labels: [String]
    let selected: String?
    let enabled: Bool
    let hot: Bool
    var loaded: String?
    var unavailable: Set<String> = []
    var warned: Set<String> = []
    var removed: Set<String> = []
    let help: (String) -> String
    let onSelect: (String) -> Void

    final class Coordinator: NSObject {
        var parent: TierSegments
        init(_ parent: TierSegments) { self.parent = parent }
        @MainActor @objc func changed(_ sender: NSSegmentedControl) {
            let index = sender.selectedSegment
            guard parent.tiers.indices.contains(index) else { return }
            // Every native path (mouse, keyboard, accessibility press) ends here: a greyed cell or a locked row puts the
            // selection back and reports nothing.
            guard parent.enabled, !parent.unavailable.contains(parent.tiers[index]) else {
                sender.selectedSegment = parent.selected.flatMap { parent.tiers.firstIndex(of: $0) } ?? -1
                return
            }
            parent.onSelect(parent.tiers[index])
            HostRefresh.after(sender)
        }
    }
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    /// Always regular and always one segment high, whatever size the hosting environment pushes onto it. Its bounds are
    /// the natural regular geometry (`TierControl.Natural`) whatever its frame: everything below draws in that space.
    final class Control: NSSegmentedControl, NSViewToolTipOwner, ToolTipRegions {
        #if DEBUG
            private(set) var drawPasses = 0
        #endif
        override var isFlipped: Bool { true }
        override var alignmentRectInsets: NSEdgeInsets { NSEdgeInsetsZero }
        /// A greyed cell's reason and a locked row's line
        /// never showed through AppKit's per-segment tooltips. Every cell's hover is now one of this control's own
        /// tooltip regions (`segmentTips`, `toolTipRegions`), which do not depend on whether the cell can be chosen. The
        /// control stays enabled for those regions; the cells that cannot be chosen now (greyed, or all of a locked
        /// row's) are disabled in the cell itself, which accessibility exposes and every native activation respects, and
        /// `Coordinator.changed` refuses them as well.
        var unavailableSegments: Set<Int> = [] { didSet { if unavailableSegments != oldValue { needsDisplay = true } } }
        var locked = false { didSet { if locked != oldValue { needsDisplay = true } } }
        var segmentTips: [String] = [] { didSet { if segmentTips != oldValue { refreshToolTips() } } }
        override func toolTip(forSegment segment: Int) -> String? { segmentTips.indices.contains(segment) ? segmentTips[segment] : nil }
        override func isAccessibilityEnabled() -> Bool { !locked }
        override static var cellClass: AnyClass? {
            get { Cell.self }
            set { _ = newValue }
        }
        /// The cell accessibility exposes: not enabled while the row is locked (its segments are disabled one by one).
        final class Cell: NSSegmentedCell {
            override func isAccessibilityEnabled() -> Bool { !((controlView as? Control)?.locked ?? false) }
        }
        /// The tooltip regions this view registered with AppKit (tag → rect and user data), for the hover tests.
        private(set) var toolTipRegions: [NSView.ToolTipTag: ToolTipRegion] = [:]
        override func addToolTip(_ rect: NSRect, owner: Any, userData data: UnsafeMutableRawPointer?) -> NSView.ToolTipTag {
            let tag = super.addToolTip(rect, owner: owner, userData: data)
            toolTipRegions[tag] = ToolTipRegion(rect: rect, userData: data)
            return tag
        }
        override func removeAllToolTips() { super.removeAllToolTips(); toolTipRegions = [:] }
        override func removeToolTip(_ tag: NSView.ToolTipTag) { super.removeToolTip(tag); toolTipRegions[tag] = nil }
        /// Arsenal: the cell under the pointer, from the drawn geometry, selected on the press (see the file header).
        override func mouseDown(with event: NSEvent) {
            guard !locked else { return }
            let point = convert(event.locationInWindow, from: nil)
            guard let index = segment(at: point), isEnabled(forSegment: index) else { return }
            selectedSegment = index
            sendAction(action, to: target)
            needsDisplay = true
        }
        /// One tooltip rect per drawn cell, owned by this control (the mechanism RowActionView and GroupHeaderView use).
        func refreshToolTips() {
            removeAllToolTips()
            let pitch = TierControl.Natural.cellWidth - 1
            for i in 0..<segmentCount {
                addToolTip(NSRect(x: CGFloat(i) * pitch, y: 0, width: pitch, height: bounds.height), owner: self, userData: nil)
            }
        }
        func view(_ view: NSView, stringForToolTip tag: NSView.ToolTipTag, point: NSPoint, userData data: UnsafeMutableRawPointer?) -> String {
            segment(at: point).flatMap { toolTip(forSegment: $0) } ?? ""
        }
        /// The segment drawn at a point in the control's own (natural) coordinates.
        func segment(at point: NSPoint) -> Int? {
            let index = Int(point.x / (TierControl.Natural.cellWidth - 1))
            return bounds.contains(point) && (0..<segmentCount).contains(index) ? index : nil
        }
        override var controlSize: NSControl.ControlSize {
            get { super.controlSize }
            set { _ = newValue; super.controlSize = .regular } // any requested size stays regular
        }
        override var intrinsicContentSize: NSSize {
            NSSize(width: TierControl.segmentsWidth(segmentCount), height: TierControl.segmentHeight)
        }
        override func setFrameSize(_ newSize: NSSize) {
            super.setFrameSize(newSize)
            naturalBounds()
            refreshToolTips()
        }
        /// Bounds = the natural size, so a frame of `TableMetrics.scale` of it scales the drawing and its hit testing.
        func naturalBounds() {
            let natural = NSSize(width: TierControl.Natural.segmentsWidth(segmentCount), height: TierControl.Natural.segmentHeight)
            if frame.width > 0, frame.height > 0, natural.width > 0, bounds.size != natural { setBoundsSize(natural) }
        }
        /// The row is loaded (blue background): a lighter track, the selected cell's fill edged in white.
        var hot = false { didSet { if hot != oldValue { needsDisplay = true } } }
        /// Arsenal: the loaded segment when another one is selected: the white edge alone.
        var loadedSegment = -1 { didSet { if loadedSegment != oldValue { needsDisplay = true } } }
        var warnedSegments: Set<Int> = [] { didSet { if warnedSegments != oldValue { needsDisplay = true } } }
        /// Arsenal: segments whose weights were removed to save space (TierControl `removed`).
        var removedSegments: Set<Int> = [] { didSet { if removedSegments != oldValue { needsDisplay = true } } }

        /// The colours of one state, from Vella's drawing (dark menu and the loaded row; Arsenal: the light menu).
        struct Colors: Equatable {
            var trackFill: NSColor, trackStroke: NSColor, divider: NSColor
            var label: NSColor, selectedLabel: NSColor, off: NSColor
            var selection: NSColor
        }
        /// The track is the Get/Load button's own background (RowAction): a faint fill and a hairline, lighter on the loaded
        /// row. No colour until a cell is chosen (on Vella: "something that we haven't chose is just like the
        /// default color").
        static func colors(hot: Bool, dark: Bool) -> Colors {
            if hot || dark {
                return Colors(
                    trackFill: NSColor.white.withAlphaComponent(hot ? 0.20 : 0.10), trackStroke: NSColor.white.withAlphaComponent(hot ? 0.35 : 0.22),
                    divider: NSColor.white.withAlphaComponent(0.18), label: NSColor.white.withAlphaComponent(0.92), selectedLabel: TierControl.ink,
                    off: NSColor.white.withAlphaComponent(hot ? 0.45 : 0.3), selection: TierControl.gold)
            }
            let ink = NSColor(white: 0.07, alpha: 1)
            return Colors(
                trackFill: NSColor.black.withAlphaComponent(0.05), trackStroke: NSColor.black.withAlphaComponent(0.16),
                divider: NSColor.black.withAlphaComponent(0.16), label: ink.withAlphaComponent(0.92), selectedLabel: TierControl.ink,
                off: ink.withAlphaComponent(0.34), selection: TierControl.gold)
        }
        /// A segment's label colour: dark on the gold, greyed when it cannot be chosen, else the text colour.
        func labelColor(_ i: Int, colors c: Colors) -> NSColor {
            unavailableSegments.contains(i) ? c.off : i == selectedSegment ? c.selectedLabel : c.label
        }
        /// The warning triangle for a segment, built like the table's missing-figure triangle (an ink "!" on a filled
        /// triangle) so it reads on every track, light or dark, loaded or not: orange where Vella draws its own warning
        /// orange, which keeps it apart from the yellow missing-figure mark; on the gold cell an ink triangle with the gold
        /// showing through its "!".
        static let orangeTriangle: NSImage? = NSImage(systemSymbolName: "exclamationmark.triangle.fill", accessibilityDescription: "int4 quality warning")?
            .withSymbolConfiguration(
                NSImage.SymbolConfiguration(pointSize: TierControl.Natural.triangleSize, weight: .semibold)
                    .applying(NSImage.SymbolConfiguration(paletteColors: [TierControl.ink, TierControl.warningColor])))
        static func triangle(onGold: Bool) -> NSImage? {
            onGold
                ? Symbol.knockout("exclamationmark.triangle.fill", size: TierControl.Natural.triangleSize, weight: .semibold, color: TierControl.ink)
                : orangeTriangle
        }
        /// Arsenal: the removed-to-save-space mark, a download arrow in the label's own colour (it is a state, not a
        /// warning), at the triangle's size.
        static func removedMark(_ color: NSColor) -> NSImage? {
            Symbol.image("arrow.down.circle", size: TierControl.Natural.removedSize, weight: .semibold, color: color)
        }

        override func draw(_ dirtyRect: NSRect) {
            #if DEBUG
                drawPasses += 1
            #endif
            // In use (disabled): the track and labels at half strength; the chosen cell's fill and loaded edge stay
            // opaque, since gold at half strength over the loaded row's blue turns muddy.
            let dimmed: CGFloat = locked ? 0.5 : 1
            let context = NSGraphicsContext.current?.cgContext
            context?.setAlpha(dimmed)
            let c = Self.colors(hot: hot, dark: ArsenalPalette.onMenu.dark)
            let pitch = TierControl.Natural.cellWidth - 1
            let track = NSBezierPath(roundedRect: bounds.insetBy(dx: 0.5, dy: 0.5), xRadius: 6, yRadius: 6)
            c.trackFill.setFill(); track.fill()
            c.trackStroke.setStroke(); track.lineWidth = 1; track.stroke()
            for i in 0..<segmentCount {
                let cell = NSRect(x: CGFloat(i) * pitch, y: 0, width: pitch, height: bounds.height)
                let selected = i == selectedSegment
                let available = !unavailableSegments.contains(i)
                let fill = cell.insetBy(dx: 1.5, dy: 2)
                if selected {
                    // Chosen: gold; loaded: the same fill with a white edge (on Vella: "once we actually
                    // load it, it should be white").
                    context?.setAlpha(1)
                    c.selection.setFill()
                    NSBezierPath(roundedRect: fill, xRadius: 5, yRadius: 5).fill()
                    context?.setAlpha(dimmed)
                } else if i > 0, i - 1 != selectedSegment {
                    c.divider.setFill()
                    NSRect(x: cell.minX - 0.5, y: cell.minY + 6, width: 1, height: cell.height - 12).fill()
                }
                let text = label(forSegment: i) ?? ""
                let attributes: [NSAttributedString.Key: Any] = [.font: font ?? TierControl.font, .foregroundColor: labelColor(i, colors: c)]
                let size = (text as NSString).size(withAttributes: attributes)
                let warned = available && warnedSegments.contains(i)
                let warn = warned ? Self.triangle(onGold: selected) : available && removedSegments.contains(i) ? Self.removedMark(labelColor(i, colors: c)) : nil
                let gap = warned ? TierControl.Natural.triangleGap : TierControl.Natural.removedGap
                let extra = warn.map { $0.size.width + gap } ?? 0
                let x = cell.midX - (size.width + extra) / 2
                (text as NSString).draw(at: NSPoint(x: x, y: cell.midY - size.height / 2), withAttributes: attributes)
                if let warn {
                    let w = warn.size
                    warn.draw(
                        in: NSRect(x: x + size.width + gap, y: cell.midY - w.height / 2, width: w.width, height: w.height),
                        from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
                }
                // Loaded: a white edge on the cell that runs and a faint light just outside it, so the loaded cell reads
                // as lit, not as another fill.
                if hot, selected ? loadedSegment < 0 : i == loadedSegment {
                    context?.setAlpha(1)
                    defer { context?.setAlpha(dimmed) }
                    let edge = NSBezierPath(roundedRect: fill, xRadius: 5, yRadius: 5)
                    edge.lineWidth = 1.5
                    TierControl.loadedEdge.setStroke()
                    edge.stroke()
                    let light = NSBezierPath(roundedRect: fill.insetBy(dx: -1.25, dy: -1.25), xRadius: 6.25, yRadius: 6.25)
                    light.lineWidth = 1
                    TierControl.loadedLight.setStroke()
                    light.stroke()
                }
            }
        }
    }

    func makeNSView(context: Context) -> NSSegmentedControl {
        let control = Control()
        control.controlSize = .regular
        control.trackingMode = .selectOne
        control.target = context.coordinator
        control.action = #selector(Coordinator.changed(_:))
        control.setContentHuggingPriority(.defaultLow, for: .horizontal)
        update(control)
        return control
    }
    func updateNSView(_ control: NSSegmentedControl, context: Context) {
        context.coordinator.parent = self
        update(control)
    }
    func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSSegmentedControl, context: Context) -> CGSize? {
        CGSize(width: TierControl.segmentsWidth(tiers.count), height: TierControl.segmentHeight)
    }

    private func update(_ control: NSSegmentedControl) {
        if control.segmentCount != tiers.count { control.segmentCount = tiers.count }
        control.controlSize = .regular // SwiftUI may push its environment size onto hosted controls
        control.font = TierControl.font
        for (i, tier) in tiers.enumerated() {
            control.setLabel(labels.indices.contains(i) ? labels[i] : tier, forSegment: i)
            control.setWidth(TierControl.Natural.cellWidth - 2, forSegment: i)
            // A cell is enabled only while it can be chosen; its hover does not depend on it (see Control).
            control.setEnabled(enabled && !unavailable.contains(tier), forSegment: i)
        }
        (control as? Control)?.naturalBounds()
        control.selectedSegment = selected.flatMap { tiers.firstIndex(of: $0) } ?? -1
        control.isEnabled = true
        if let control = control as? Control {
            control.unavailableSegments = Set(unavailable.compactMap { tiers.firstIndex(of: $0) })
            control.locked = !enabled
            control.segmentTips = tiers.map(help)
            control.hot = hot
            control.loadedSegment = loaded.flatMap { tiers.firstIndex(of: $0) } ?? -1
            control.warnedSegments = Set(warned.compactMap { tiers.firstIndex(of: $0) })
            control.removedSegments = Set(removed.compactMap { tiers.firstIndex(of: $0) })
        }
        control.needsDisplay = true
    }
}
