// Layout adapted from Projects/Vella Sources/Vella/ModelsTable/ModelTable.swift @ f4c5c34 (Vella; relicensed MIT by its author, 2 Oct 2026): the same
// visual language (flame and blue loaded row with white text, blue precision cells turning gold when loaded, green/red
// deltas, one-word action, AppKit tooltips), one table for every kind of model, grouped by kind, with the same columns
// for every kind: Model · Params · Precision · Speed · Peak memory · gear · action.
import AppKit
import ArsenalCore
import SwiftUI

private func pt(_ base: CGFloat) -> CGFloat { TableMetrics.pt(base) }
private func size(_ base: CGFloat) -> CGFloat { TableMetrics.font(base) }

struct ModelTable: View {
    /// Column widths; `spacing` between columns, `rowPadding` inside a row on each side.
    enum W {
        /// The longest catalog name (Stable Audio 3 Small SFX) and the download line ("Downloading · 6.6 of 16.0 GB")
        /// after the flame, with a few points to spare; a longer local name truncates and its tooltip names it.
        static let model: CGFloat = pt(190), params: CGFloat = pt(52)
        /// The Precision column: a line per precision on disk (WeightsCells.swift).
        static let precision: CGFloat = PrecisionLines.width
        /// One figure column (Speed, Peak memory).
        static let metric: CGFloat = pt(104)
        static let action: CGFloat = RowAction.width
        /// The settings gear's column, between Peak memory and the action.
        static let gear: CGFloat = pt(24)
        static let spacing: CGFloat = pt(6), rowPadding: CGFloat = pt(8)
        static let icon: CGFloat = pt(24), flameSize: CGFloat = size(17)
        static let iconGap: CGFloat = pt(6), lineGap: CGFloat = pt(1)
        /// Everything left of the figure slots, and the row's width for a number of slots.
        static let fixed: [CGFloat] = [model, params, precision]
        static func row(slots: Int) -> CGFloat {
            let columns = fixed + Array(repeating: metric, count: slots) + [gear, action]
            return columns.reduce(0, +) + CGFloat(columns.count - 1) * spacing + 2 * rowPadding
        }
        /// x of a figure slot's leading edge inside a row.
        static func slotX(_ slot: Int) -> CGFloat {
            rowPadding + fixed.reduce(0, +) + CGFloat(fixed.count) * spacing + CGFloat(slot) * (metric + spacing)
        }
    }
    static let leadingPadding: CGFloat = pt(6), trailingPadding: CGFloat = pt(2)
    /// Figure columns: Speed and Peak memory, for every kind.
    static func slots(_ kinds: [KindSpec]) -> Int { TableStats.columns.count }
    /// Whole points: the menu item view rounds its frame, so the table's own content is exactly that wide.
    static func width(slots: Int) -> CGFloat { (leadingPadding + W.row(slots: slots) + trailingPadding).rounded(.up) }
    /// The trailing padding plus the rounding.
    static func trailing(slots: Int) -> CGFloat { width(slots: slots) - leadingPadding - W.row(slots: slots) }
    @MainActor static func width(_ c: TableController) -> CGFloat { (leadingPadding + rowWidth(c) + trailingPadding).rounded(.up) }
    @MainActor static func trailing(_ c: TableController) -> CGFloat { width(c) - leadingPadding - rowWidth(c) }
    /// The rows' width: the whole grid; an empty table is as wide as its Get more models row and the Refresh glyph (no
    /// figure headings over nothing), widened only for a footer line, up to the grid's width.
    @MainActor static func rowWidth(_ c: TableController) -> CGFloat {
        let grid = W.row(slots: slots(c.kinds))
        guard c.rows.isEmpty else { return grid }
        let line = footer(c).map { textWidth($0.text, size: footerSize) + 2 * W.rowPadding } ?? 0
        return min(grid, max(emptyRowWidth, line).rounded(.up))
    }
    static func textWidth(_ text: String, size: CGFloat) -> CGFloat {
        (text as NSString).size(withAttributes: [.font: NSFont.systemFont(ofSize: size)]).width.rounded(.up)
    }
    /// The Refresh glyph's distance from the row end: over the rows' trash/cancel glyph.
    static let refreshTrailing = RowAction.width - RowAction.buttonWidth - RowAction.trashGap - RowAction.trashSize
    static let emptyRowWidth: CGFloat =
        W.rowPadding + W.icon + W.iconGap + textWidth(PinnedRow.title, size: valueSize) + pt(24) + RowAction.trashSize + refreshTrailing + W.rowPadding
    static let trailingMargin: CGFloat = W.rowPadding + trailingPadding

    /// Vella's row (its ModelTable.rowHeight: the Exact/Fast switch's 43 pt with air around it). Arsenal has no switch and
    /// keeps the height, so controls and two-line figures sit with Vella's air. A table holds the models a user added
    /// (A user has 5–6); a long one scrolls past `maxHeight`.
    static let rowHeight: CGFloat = pt(43) + pt(6)
    static let rowGap: CGFloat = pt(2)
    /// Air above and below the Precision column's lines when they make the row taller than `rowHeight`.
    static let linesPadding: CGFloat = pt(5)
    /// A row's height with `lines` precisions on disk: the Models row, grown to fit its lines (one line per
    /// precision, nothing overlapping).
    static func rowSpan(lines: Int) -> CGFloat { max(rowHeight, PrecisionLines.height(lines: lines) + 2 * linesPadding) }
    /// Vella's heading, section label and footer heights, and the air around the hairline dividers.
    static let headerHeight: CGFloat = pt(26), groupHeaderHeight: CGFloat = pt(24), footerHeight: CGFloat = pt(28)
    static let verticalPadding: CGFloat = pt(6), dividerPadding: CGFloat = pt(4), rowCorner: CGFloat = pt(5)
    /// The rule between groups, with Vella's air above and below it (pt(8), pt(2)) but a 1 pt hairline instead of Vella's
    /// pt(3) rule: Vella's thick rule separates its two groups, while a user's table here has a group per kind, mostly of
    /// one row, and five thick rules in a six-model table read as stripes (realistic-m5max render, 15 Oct). The air and
    /// the group label carry the separation; the hairline is the thinnest line that still marks it.
    static let groupRule: CGFloat = pt(1), groupRuleAbove: CGFloat = pt(8), groupRuleBelow: CGFloat = pt(2)
    static let groupRuleHeight: CGFloat = groupRuleAbove + groupRule + groupRuleBelow
    /// The pinned Get more models row: below a group rule when groups exist, alone in an empty table.
    /// The area under the groups: its rule, the unlisted-weights row when shown, and Get more models.
    static func pinnedHeight(groups: Int, orphan: Bool = false) -> CGFloat {
        (groups > 0 ? groupRuleHeight : 0) + (orphan ? rowHeight + rowGap : 0) + rowHeight + rowGap
    }
    /// Above this height the groups scroll inside the table (header and any footer stay); set from the screen at launch.
    @MainActor static var maxHeight: CGFloat = 860

    /// A group's height from its rows' heights (each row's span is `rowSpan(copies:)`).
    static func groupHeight(heights: [CGFloat], first: Bool) -> CGFloat {
        (first ? 0 : groupRuleHeight) + groupHeaderHeight + heights.reduce(0) { $0 + $1 + rowGap }
    }
    static func groupHeight(rows: Int, first: Bool) -> CGFloat {
        groupHeight(heights: Array(repeating: rowHeight, count: rows), first: first)
    }
    static func groupsHeight(_ groups: [[CGFloat]]) -> CGFloat {
        groups.enumerated().reduce(0) { $0 + groupHeight(heights: $1.element, first: $1.offset == 0) }
    }
    static func groupsHeight(_ rowCounts: [Int]) -> CGFloat { groupsHeight(rowCounts.map { Array(repeating: rowHeight, count: $0) }) }
    /// Header, its divider and the paddings; a footer (only shown for a problem) adds its divider and line.
    static func chromeHeight(footer: Bool = false) -> CGFloat {
        2 * verticalPadding + headerHeight + 1 + 2 * dividerPadding + (footer ? 1 + 2 * dividerPadding + footerHeight : 0)
    }
    static func contentHeight(groups: [[CGFloat]], orphan: Bool = false) -> CGFloat {
        groupsHeight(groups) + pinnedHeight(groups: groups.count, orphan: orphan)
    }
    static func contentHeight(_ rowCounts: [Int]) -> CGFloat { contentHeight(groups: rowCounts.map { Array(repeating: rowHeight, count: $0) }) }
    static func height(groups: [[CGFloat]], footer: Bool = false, orphan: Bool = false) -> CGFloat {
        min(maxHeight, chromeHeight(footer: footer) + contentHeight(groups: groups, orphan: orphan))
    }
    static func height(_ rowCounts: [Int], footer: Bool = false) -> CGFloat {
        height(groups: rowCounts.map { Array(repeating: rowHeight, count: $0) }, footer: footer)
    }
    /// Every row's span in every group, in the order the table shows them.
    @MainActor static func spans(_ c: TableController) -> [[CGFloat]] {
        c.groups.map { group in group.rows.map { rowSpan(lines: c.weights($0).copies.count) } }
    }
    /// The unlisted-weights row shows while any weights are unlisted.
    @MainActor static func orphanShown(_ c: TableController) -> Bool { c.orphanBytes > 0 }
    /// The table's height: its rows and footer only (a model's settings card floats over it and never adds to it).
    @MainActor static func height(_ c: TableController) -> CGFloat {
        height(groups: spans(c), footer: footer(c) != nil, orphan: orphanShown(c))
    }
    @MainActor static func scrolls(_ c: TableController) -> Bool {
        chromeHeight(footer: footer(c) != nil) + contentHeight(groups: spans(c), orphan: orphanShown(c)) > maxHeight
    }
    /// The line under the table, only when something is wrong: a refused action, a refused Refresh, or a catalog that is
    /// not verified. A verified catalog needs no line (v2.2 item 3).
    @MainActor static func footer(_ c: TableController) -> (text: String, help: String)? {
        // A raw system message never shows: the controller words its own (its hover keeps the raw text); a line set
        // elsewhere with a raw message in it is reworded here, with that message as its hover.
        if let error = c.lastError {
            if let help = c.lastErrorHelp, help != error { return (error, help) }
            return ErrorWording.reword(error) ?? (error, error)
        }
        if let notice = c.memoryNotice { return (notice, notice) }
        if let notice = c.refreshNotice, !c.refreshNoticeIsStale { return (notice, notice) }
        if let catalog = c.catalog, !catalog.verified || catalog.error != nil { return (FooterText.catalog(catalog), FooterText.catalogHelp(catalog)) }
        return nil
    }

    static let valueSize = NSFont.systemFontSize, deltaSize = size(10.5), headingSize = size(12), footerSize = size(10)
    static let valueFont = Font.system(size: valueSize).monospacedDigit()
    static let deltaFont = Font.system(size: deltaSize).monospacedDigit()

    @ObservedObject var controller: TableController
    @ObservedObject var appearance = Appearance.shared
    private var pal: ArsenalPalette { appearance.menuPalette }

    static let paramsHeaderHelp = "Parameters"

    var body: some View {
        let slots = Self.slots(controller.kinds)
        let rowWidth = Self.rowWidth(controller)
        VStack(alignment: .leading, spacing: 0) {
            header.frame(width: rowWidth, height: Self.headerHeight)
            Rectangle().fill(pal.hairline.swiftUI).frame(height: 1).padding(.vertical, Self.dividerPadding)
            if Self.scrolls(controller) {
                // Lazy: only the groups in sight are built. Every Models open moves the table into a new menu window, and
                // SwiftUI then lays out and walks (for accessibility focus) every row it holds: with 85 models that was
                // 0.4-1 s on each pass of the pointer over Models (6 Nov 2026).
                ScrollView(.vertical) { LazyVStack(alignment: .leading, spacing: 0) { groupList(slots: slots, width: rowWidth) } }
                    .frame(width: rowWidth, height: Self.height(controller) - Self.chromeHeight(footer: Self.footer(controller) != nil))
            } else {
                groups(slots: slots, width: rowWidth)
            }
            Spacer(minLength: 0)
            if let line = Self.footer(controller) {
                Rectangle().fill(pal.hairline.swiftUI).frame(height: 1).padding(.vertical, Self.dividerPadding)
                footer(line).frame(width: rowWidth, height: Self.footerHeight)
            }
        }.padding(.vertical, Self.verticalPadding).padding(.leading, Self.leadingPadding).padding(.trailing, Self.trailing(controller))
            .fixedSize(horizontal: true, vertical: false)
            .frame(height: Self.height(controller), alignment: .top)
            .foregroundStyle(pal.t1.swiftUI)
            // A model's settings card floats over the table beside its gear; an overlay takes no part in the layout.
            .overlay(alignment: .topLeading) { SettingsCardView(controller: controller, state: controller.settingsCard.state) }
            .environment(\.colorScheme, appearance.colorScheme)
    }

    /// Where a row's gear sits in the table (top-left origin), from the metrics that lay the rows out; nil for a row the
    /// table does not show. A scrolled table moves it; the card then reads the gear's own view (SettingsCard.gearRect).
    @MainActor static func gearRect(_ c: TableController, id: String) -> CGRect? {
        var y = verticalPadding + headerHeight + 1 + 2 * dividerPadding
        for (index, group) in c.groups.enumerated() {
            let top = y + (index == 0 ? 0 : groupRuleHeight) + groupHeaderHeight
            let heights = group.rows.map { rowSpan(lines: c.weights($0).copies.count) }
            if let i = group.rows.firstIndex(where: { $0.id == id }) {
                let x = leadingPadding + W.slotX(slots(c.kinds))
                let above = heights.prefix(i).reduce(0) { $0 + $1 + rowGap }
                return CGRect(x: x, y: top + above, width: W.gear, height: heights[i])
            }
            y += groupHeight(heights: heights, first: index == 0)
        }
        return nil
    }

    private func groups(slots: Int, width: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 0) { groupList(slots: slots, width: width) }
    }

    /// The groups, then the rule, the unlisted-weights row and Get more models, for a plain or a lazy stack.
    @ViewBuilder private func groupList(slots: Int, width: CGFloat) -> some View {
        let groups = controller.groups
        ForEach(Array(groups.enumerated()), id: \.element.kind.id) { index, group in
            VStack(alignment: .leading, spacing: 0) {
                if index > 0 {
                    Rectangle().fill(pal.raisedBorder.swiftUI).frame(width: width, height: Self.groupRule)
                        .padding(.top, Self.groupRuleAbove).padding(.bottom, Self.groupRuleBelow)
                        .accessibilityHidden(true)
                }
                let spans = group.rows.map { Self.rowSpan(lines: controller.weights($0).copies.count) }
                groupHeader(group.kind, index: index, count: groups.count, extent: Self.groupHeight(heights: spans, first: true), slots: slots, width: width)
                let columns = TableStats.columns(group.kind)
                ForEach(group.rows) { row in
                    let weights = controller.weights(row)
                    let lines = PrecisionLine.make(row, presentation: controller.presentation(row), weights: weights)
                    ModelTableRow(
                        row: row, columns: columns, slots: slots, p: controller.presentation(row), chip: controller.chip,
                        hasSettings: controller.settings[row.id]?.isEmpty == false && row.kind != Unlisted.kind,
                        hovered: controller.renderHover == row.id, changes: controller.pendingChanges(row),
                        loadChoices: controller.loadChoices(row), settingsOpen: controller.settingsRow == row.id, pal: pal, controller: controller,
                        weights: weights, lines: lines, height: Self.rowSpan(lines: lines.count)
                    ).equatable().padding(.bottom, Self.rowGap)
                }
            }
            .opacity(controller.drag?.group == group.kind.id ? 0.55 : 1)
            .overlay(alignment: .top) { if controller.drag.map({ $0.slot == index && $0.group != group.kind.id }) == true { dropLine(width) } }
            .overlay(alignment: .bottom) {
                if index == groups.count - 1, controller.drag.map({ $0.slot == groups.count && $0.group != group.kind.id }) == true { dropLine(width) }
            }
        }
        if !groups.isEmpty {
            Rectangle().fill(pal.raisedBorder.swiftUI).frame(width: width, height: Self.groupRule)
                .padding(.top, Self.groupRuleAbove).padding(.bottom, Self.groupRuleBelow).accessibilityHidden(true)
        }
        // Weights no model refers to: one row, deleted in one click.
        if Self.orphanShown(controller) {
            OrphanRow(bytes: controller.orphanBytes, busy: controller.removing.contains(WeightsPresentation.orphanKey), controller: controller, pal: pal)
                .frame(width: width, height: Self.rowHeight).padding(.bottom, Self.rowGap)
        }
        // The last row: Get more models.
        PinnedRow(hovered: controller.renderHoverPinned, click: { controller.openBrowser() })
            .frame(width: width, height: Self.rowHeight).padding(.bottom, Self.rowGap)
    }

    /// Where a dragged group would land.
    private func dropLine(_ width: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 1).fill(Color.accentColor).frame(width: width, height: pt(2)).allowsHitTesting(false)
    }

    /// Column labels shared by every group (each group's header carries its kind's units as hovers); an empty table has
    /// none, only Refresh.
    @ViewBuilder private var header: some View {
        if controller.rows.isEmpty {
            HStack(spacing: 0) {
                Spacer(minLength: 0)
                refresh.padding(.trailing, Self.refreshTrailing)
            }.padding(.horizontal, W.rowPadding)
        } else {
            columnHeader
        }
    }
    /// Top right, over the rows' trash/cancel glyphs: check for new models.
    private var refresh: some View {
        RefreshControl(enabled: controller.refreshAllowed, help: controller.refreshHelp, click: { controller.refreshModels() })
            .frame(width: RowAction.trashSize, height: RowAction.trashSize)
    }
    private var columnHeader: some View {
        HStack(spacing: W.spacing) {
            heading("Model", W.model, .leading, help: nil)
            heading("Params", W.params, .center, help: Self.paramsHeaderHelp)
            // Centred on the lines' pills, not on the bolt's slot before them or the trash after them.
            heading(TierControl.title, PrecisionLines.pillWidth, .center, help: TierControl.headerHelp)
                .padding(.leading, PrecisionLines.boltWidth + PrecisionLines.boltGap)
                .padding(.trailing, PrecisionLines.trashGap + PrecisionLines.trashWidth)
            let slots = Self.slots(controller.kinds)
            ForEach(TableStats.columns, id: \.id) { column in
                heading(column.title, W.metric, .center, help: TableStats.headerHelp(column.id))
            }
            HStack(spacing: 0) {
                Spacer(minLength: 0)
                refresh.padding(.trailing, Self.refreshTrailing)
            }.frame(width: W.gear + W.action + W.spacing, height: Self.headerHeight)
        }.padding(.horizontal, W.rowPadding)
    }

    private func heading(_ text: String, _ width: CGFloat, _ alignment: Alignment, help: String?) -> some View {
        Text(text).font(.system(size: Self.headingSize, weight: .medium)).foregroundStyle(pal.t2.swiftUI)
            .frame(width: width, alignment: alignment).cellTooltip(help)
    }

    /// The group's header: its title and its own figure columns. An AppKit view behind it takes the drag (move the
    /// group as a whole), the hover arrows and the tooltips (GroupHeader.swift).
    private func groupHeader(_ kind: KindSpec, index: Int, count: Int, extent: CGFloat, slots: Int, width: CGFloat) -> some View {
        // Weights the catalog does not list have no figures: their group names no figure columns.
        let columns = kind.id == Unlisted.kind ? [] : TableStats.columns(kind)
        let first = slots - columns.count
        let tips =
            [(CGRect(x: 0, y: 0, width: W.slotX(0) - W.spacing, height: Self.groupHeaderHeight), GroupHeader.titleHelp(kind))]
            + columns.enumerated().map { i, c in
                (
                    CGRect(x: W.slotX(first + i), y: 0, width: W.metric, height: Self.groupHeaderHeight),
                    TableStats.help(kind: kind, column: c.id)
                )
            }
        return ZStack(alignment: .topLeading) {
            GroupHeader(
                kindID: kind.id, title: kind.title, index: index, count: count, extent: extent,
                tips: tips, hovered: controller.renderHoverGroup == kind.id, controller: controller
            ).frame(width: width, height: Self.groupHeaderHeight)
            HStack(spacing: W.spacing) {
                Text(kind.title).font(.system(size: Self.headingSize, weight: .semibold)).foregroundStyle(pal.t2.swiftUI)
                    .frame(width: W.slotX(first) - W.rowPadding - W.spacing, alignment: .leading)
                Spacer(minLength: 0)
            }.padding(.horizontal, W.rowPadding).frame(width: width, height: Self.groupHeaderHeight, alignment: .center).allowsHitTesting(false)
        }.frame(width: width, height: Self.groupHeaderHeight)
    }

    /// The status line under the name; on the loaded row's blue, Vella's lighter green and red and white for the rest.
    static func tone(_ tone: StatusLine.Tone, hot: Bool, _ pal: ArsenalPalette) -> Color {
        switch tone {
        case .measured: return (hot ? ArsenalPalette.hotGreen : pal.green).swiftUI
        case .estimate: return (hot ? ArsenalPalette.gold : pal.amber).swiftUI
        case .error: return (hot ? ArsenalPalette.hotRed : pal.red).swiftUI
        // The chosen precision waiting for Reload: the gold of its cell; on the light menu's solid blue the gold is too
        // pale for small text, so white (the gold cell and the gold Reload carry the colour there).
        case .pending: return (hot ? (pal.dark ? ArsenalPalette.gold : ArsenalPalette.hotText) : pal.amber).swiftUI
        case .busy, .quiet, .plain: return hot ? ArsenalPalette.hotText.withAlphaComponent(0.8).swiftUI : (tone == .quiet ? pal.t3 : pal.t2).swiftUI
        }
    }
    /// A delta under a figure: green faster, red slower, quiet when the same (Vella's tones, lighter on the loaded row).
    /// On the light menu's solid blue loaded row the light green and red fall under 4.5:1, so the words go white and an
    /// arrow in the tone carries the colour (`deltaArrow`).
    static func tone(_ tone: TableFigure.Delta.Tone, hot: Bool, _ pal: ArsenalPalette) -> Color {
        switch tone {
        case .better: return (hot ? ArsenalPalette.hotGreen : pal.green).swiftUI
        case .worse: return (hot ? ArsenalPalette.hotRed : pal.red).swiftUI
        case .same: return hot ? ArsenalPalette.hotText.withAlphaComponent(0.8).swiftUI : pal.t2.swiftUI
        }
    }

    static func deltaArrow(hot: Bool, _ pal: ArsenalPalette) -> Bool { hot && !pal.dark }

    /// Left: the last error, else the catalog's state. The rows carry their own progress.
    private func footer(_ line: (text: String, help: String)) -> some View {
        HStack {
            Text(line.text).font(.system(size: Self.footerSize)).foregroundStyle((line.text == controller.memoryNotice ? pal.t2 : pal.red).swiftUI).lineLimit(1).cellTooltip(
                line.help)
            Spacer(minLength: 0)
        }.padding(.horizontal, W.rowPadding)
    }
}

/// One model's row. Equatable on what it shows, so a refresh that changed one row (a download's progress) re-renders
/// that row only; the controller is there for the actions, not compared.
struct ModelTableRow: View, Equatable {
    typealias W = ModelTable.W
    let row: ModelRow
    let columns: [KindColumn]
    let slots: Int
    let p: RowPresentation
    let chip: String?
    let hasSettings: Bool
    let hovered: Bool
    /// What a pending row's Reload applies, and load-time settings away from their defaults (the action's tooltip).
    let changes: [String]
    let loadChoices: [String]
    /// This row's settings card is open beside it: the row takes a neutral fill (colour only; nothing moves).
    var settingsOpen = false
    let pal: ArsenalPalette
    let controller: TableController
    /// The row's weights on this Mac: its Precision lines' copies, and the model's Delete.
    var weights: WeightsPresentation = .empty
    /// The Precision column: a line per precision on disk.
    var lines: [PrecisionLine] = []
    /// The row's height, grown to fit its lines (`ModelTable.rowSpan`).
    var height: CGFloat = ModelTable.rowHeight

    nonisolated static func == (a: Self, b: Self) -> Bool {
        a.row == b.row && a.columns == b.columns && a.slots == b.slots && a.p == b.p && a.chip == b.chip && a.hasSettings == b.hasSettings
            && a.hovered == b.hovered && a.changes == b.changes && a.loadChoices == b.loadChoices && a.settingsOpen == b.settingsOpen
            && a.pal == b.pal && a.controller === b.controller && a.weights == b.weights && a.lines == b.lines && a.height == b.height
    }

    /// The purpose (a plain line) and a failure's reason wrap onto a second line; any other state's line stays one.
    static func purposeLines(_ line: StatusLine) -> Int { line.tone == .plain || line.tone == .error ? 2 : 1 }
    /// The row shows a second line under the name: what the model does, or a state while it applies (RowPresentation.line).
    static func hasStatusLine(_ row: ModelRow, _ p: RowPresentation) -> Bool { !p.line.text.isEmpty }
    /// How far the name's centre sits above the row's centre: half the status line and its gap when there is one.
    static func nameRaise(_ row: ModelRow, _ p: RowPresentation) -> CGFloat {
        guard hasStatusLine(row, p) else { return 0 }
        let name = NSFont.systemFont(ofSize: ModelTable.valueSize), status = NSFont.systemFont(ofSize: ModelTable.deltaSize, weight: .medium)
        func line(_ f: NSFont) -> CGFloat { ceil(f.ascender - f.descender + f.leading) }
        return ((line(name) + W.lineGap + line(status)) / 2 - line(name) / 2)
    }

    var body: some View {
        // A pending change keeps the loaded row's blue: the loaded precision keeps a ring, the chosen one is gold, the
        // status line says it applies on next load and Reload is the row's prominent button.
        let hot = p.hot
        let inverted = hot
        let controller = controller, row = row
        HStack(spacing: W.spacing) {
            // Each line carries its own hover over it (TooltipCell's `rowTooltip`): the name and flame the model's, the
            // status line its own. Not nested, so neither covers the other.
            let modelHelp = ModelHelp.text(row, presentation: p, name: Self.truncates(row.name))
            HStack(spacing: W.iconGap) {
                Image(systemName: "flame.fill").font(.system(size: W.flameSize)).foregroundStyle(Color.orange)
                    .frame(width: W.icon).opacity(hot ? 1 : 0).rowTooltip(modelHelp)
                VStack(alignment: .leading, spacing: W.lineGap) {
                    Text(row.name).font(.system(size: ModelTable.valueSize)).lineLimit(1)
                        .frame(maxWidth: .infinity, alignment: .leading).rowTooltip(modelHelp)
                    if Self.hasStatusLine(row, p) {
                        // What the model does and why a Get failed take two lines when they need them, so neither is
                        // cut mid-word; a busy state (getting, loading) keeps one.
                        Text(p.line.text).font(.system(size: ModelTable.deltaSize, weight: .medium)).lineLimit(Self.purposeLines(p.line))
                            .fixedSize(horizontal: false, vertical: true)
                            .foregroundStyle(ModelTable.tone(p.line.tone, hot: inverted, pal))
                            .frame(maxWidth: .infinity, alignment: .leading).rowTooltip(p.line.help)
                    }
                }
            }.frame(width: W.model, alignment: .leading)
            Text(row.params.isEmpty ? MetricText.missing : ParamsText.display(row.params, name: row.name)).frame(width: W.params)
            // One line per precision on disk, 16-bit on top; a click on a line chooses it, its trash deletes that copy.
            PrecisionLinesCell(
                lines: lines, hot: hot, locked: p.locked, bolt: row.hasFastKernels,
                onSelect: { controller.select(row, column: $0) }, onDelete: { controller.deleteCopy(row, $0) }
            ).frame(width: W.precision, height: height)
            ForEach(TableFigures.columns, id: \.self) { column in
                if row.kind == Unlisted.kind {
                    // Weights the catalog does not list have no figures to show.
                    Color.clear.frame(width: W.metric, height: height)
                } else {
                    figure(
                        TableFigures.figure(
                            row, precision: p.selected, column: column, host: chip, modePreviewChanged: p.fast != (row.fastAvailable && row.fast),
                            shown: controller.shownKinds[row.id]),
                        hot: inverted)
                }
            }
            Group {
                if hasSettings {
                    GearCell(id: row.id, settings: controller.settingsCard, active: settingsOpen, inverted: inverted) { controller.openSettings(row, from: $0) }
                } else {
                    Color.clear
                }
            }.frame(width: W.gear, height: height)
            // Remove (a catalog-removed row) deletes the weights with it, so it is off while that delete is refused.
            let removeRefused = p.action == .remove && weights.hasWeights && weights.deleteBlocked != nil
            RowAction(
                title: p.actionTitle, busyText: p.busyText,
                enabled: controller.admitting && !removeRefused
                    && (p.action == .remove || p.action == .unload || ModelHelp.unavailable(row)[PrecisionColumn.of(p.selected, formats: row.formats) ?? ""] == nil),
                deletable: p.action != .remove && (weights.hasWeights || !p.locked), hot: inverted,
                prominent: p.action == .reload && p.pending && p.busyText == nil, hovered: hovered,
                help: removeRefused ? weights.deleteBlocked ?? "" : ModelHelp.action(row, presentation: p, changes: changes, loadChoices: loadChoices),
                cancellable: p.cancellable,
                onPerform: { controller.perform(row) }, onDelete: { controller.remove(row) }, onCancel: { controller.cancelGet(row) },
                trashHelp: weights.hasWeights
                    ? "Delete its weights (\(formatBytes(weights.bytes))) and remove it from the table" : RowAction.deleteHelp,
                trashBlocked: weights.hasWeights ? weights.deleteBlocked : nil, countdown: p.countdown)
        }.font(ModelTable.valueFont)
            .padding(.horizontal, W.rowPadding).frame(height: height)
            .background(
                hot ? pal.hotRow.swiftUI : settingsOpen ? pal.overlay(0.08).swiftUI : .clear,
                in: RoundedRectangle(cornerRadius: ModelTable.rowCorner)
            )
            .background {
                if let value = p.progress {
                    GeometryReader { geometry in
                        RoundedRectangle(cornerRadius: ModelTable.rowCorner)
                            .fill(Color(nsColor: .selectedContentBackgroundColor).opacity(0.35))
                            .frame(width: geometry.size.width * min(1, max(0, value)))
                            // Each progress report eases the fill forward instead of stepping it.
                            .animation(.easeOut(duration: 0.4), value: value)
                    }.allowsHitTesting(false)
                }
            }
            .overlay {
                // Downloading, making or loading: a soft band sweeps over the row, the loaded blue included (Core
                // Animation; nothing draws at rest).
                if p.busyText != nil { BusySheen(hot: hot, cornerRadius: ModelTable.rowCorner).allowsHitTesting(false) }
            }
            .foregroundStyle((inverted ? ArsenalPalette.hotText : pal.t1).swiftUI)
            .contentShape(Rectangle())
            // The cells' hovers, on one AppKit view for the whole row.
            .rowTooltips()
    }

    /// The name does not fit its line: the Model tooltip names it in full.
    static func truncates(_ name: String) -> Bool {
        (name as NSString).size(withAttributes: [.font: NSFont.systemFont(ofSize: ModelTable.valueSize)]).width > W.model - W.icon - W.iconGap
    }

    static let triangleSize = ModelTable.valueSize
    /// A figure (`hot`: drawn on the loaded row's colour): the value at full contrast, with the chip it was measured on, or its change against the 16-bit format,
    /// beneath; a missing figure is the yellow triangle alone. The hover says how the figure was made, or why it is missing.
    private func figure(_ f: TableFigure, hot: Bool) -> some View {
        VStack(spacing: W.lineGap) {
            if let value = f.value {
                Text(value).lineLimit(1).minimumScaleFactor(0.85)
                if let reference = f.reference {
                    HStack(spacing: TableMetrics.pt(3)) {
                        Self.triangle(size: ModelTable.deltaSize * 0.9)
                        Text(reference).font(ModelTable.deltaFont).lineLimit(1)
                            .foregroundStyle(hot ? ArsenalPalette.hotText.withAlphaComponent(0.8).swiftUI : pal.t2.swiftUI)
                    }
                } else if let delta = f.delta {
                    if ModelTable.deltaArrow(hot: hot, pal), delta.tone != .same {
                        HStack(spacing: TableMetrics.pt(2)) {
                            Image(systemName: delta.tone == .better ? "arrow.up" : "arrow.down").font(.system(size: ModelTable.deltaSize * 0.9, weight: .bold))
                                .foregroundStyle(ModelTable.tone(delta.tone, hot: true, pal))
                            Text(delta.text).font(ModelTable.deltaFont).lineLimit(1).fixedSize().foregroundStyle(ArsenalPalette.hotText.swiftUI)
                        }
                    } else {
                        Text(delta.text).font(ModelTable.deltaFont).lineLimit(1).fixedSize().foregroundStyle(ModelTable.tone(delta.tone, hot: hot, pal))
                    }
                }
            } else {
                Self.triangle(size: Self.triangleSize)
            }
        }.frame(width: W.metric, height: height)
            .rowTooltip(f.help)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(([f.value ?? "Not available"] + [f.reference, f.delta?.text].compactMap { $0 }).joined(separator: ", "))
            .accessibilityHint(f.help)
    }
    /// The missing-figure mark: a yellow triangle with a dark exclamation mark, the same on every row.
    static func triangle(size: CGFloat) -> some View {
        Image(systemName: "exclamationmark.triangle.fill").font(.system(size: size))
            .symbolRenderingMode(.palette).foregroundStyle(Color(nsColor: ArsenalPalette.ink), Color(nsColor: .systemYellow))
    }
}

/// "Unlisted weights": weights on disk that no model refers to. Its one button deletes them; the size is its line.
struct OrphanRow: View {
    typealias W = ModelTable.W
    static let title = "Unlisted weights"
    let bytes: Int64
    let busy: Bool
    let controller: TableController
    let pal: ArsenalPalette

    var body: some View {
        HStack(spacing: W.spacing) {
            VStack(alignment: .leading, spacing: W.lineGap) {
                Text(Self.title).font(.system(size: ModelTable.valueSize)).lineLimit(1)
                Text(formatBytes(bytes)).font(.system(size: ModelTable.deltaSize, weight: .medium)).foregroundStyle(pal.t2.swiftUI)
            }.frame(width: W.model, alignment: .leading)
            Color.clear.frame(width: W.params, height: ModelTable.rowHeight)
            Color.clear.frame(width: W.precision, height: ModelTable.rowHeight)
            ForEach(TableFigures.columns, id: \.self) { _ in Color.clear.frame(width: W.metric, height: ModelTable.rowHeight) }
            Color.clear.frame(width: W.gear, height: ModelTable.rowHeight)
            RowAction(
                title: "Delete", busyText: busy ? "\u{2026}" : nil, enabled: controller.admitting && !busy, deletable: false,
                help: "Delete the weights no model uses (\(formatBytes(bytes))).", onPerform: { controller.deleteOrphans() }, onDelete: {})
        }.font(ModelTable.valueFont)
            .padding(.horizontal, W.rowPadding).frame(height: ModelTable.rowHeight)
            .foregroundStyle(pal.t1.swiftUI)
            .contentShape(Rectangle())
    }
}

enum FooterText {
    static func catalog(_ c: CatalogStatus) -> String {
        if let error = c.error { return "Catalog: \(error)" }
        let commit = c.pinnedCommit.map { " \u{00b7} " + String($0.prefix(7)) } ?? ""
        return "Catalog: \(c.source)\(commit)" + (c.verified ? " \u{00b7} verified" : " \u{00b7} not verified")
    }
    static func catalogHelp(_ c: CatalogStatus) -> String {
        c.verified
            ? "The model list is signed, and each model's files are pinned to a reviewed commit and checked by hash before use."
            : "The model list could not be verified; Get is refused until it is."
    }
}
