// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// The browser's list: an NSTableView whose columns BrowserColumns sizes on every width
// change (user drags update the weights), Finder's header sorting with the header in the window's top band (a drag on
// it moves the window), 30 pt rows with a hover and a quiet selection, and expanded rows: the row plus a full-width
// detail row (a non-floating group row) drawn together as one raised card. A loaded model's row takes the Models table's
// loaded blue with white text, its flame, and its loaded format in the table's gold.

@MainActor final class BrowserList: NSObject, NSTableViewDataSource, NSTableViewDelegate {
    enum Line: Equatable {
        case item(String), detail(String)
        var id: String {
            switch self {
            case .item(let id), .detail(let id): return id
            }
        }
    }
    weak var controller: BrowserController?
    var focusSearch: () -> Void = {}
    var closeWindow: () -> Void = {}
    let table = BrowserTableView()
    let scrollView = NSScrollView()
    let header = BrowserHeaderView()
    private(set) var rows: [BrowserItem] = []
    private var byID: [String: BrowserItem] = [:]
    private(set) var lines: [Line] = []
    private var palette: ArsenalPalette { Appearance.shared.palette }
    private(set) var weights = BrowserColumns.savedWeights()
    private(set) var layout = BrowserColumns.solve(width: 1200)
    private var applying = false
    /// Per expanded model: its details (rebuilt only when its item changes) and their height for one layout key.
    private var details: [String: (item: BrowserItem, detail: ModelDetail)] = [:]
    private var detailHeights: [String: (key: ModelDetailView.LayoutKey, height: CGFloat, detail: ModelDetail)] = [:]
    /// The detail rows on screen, to give them the new arrangement when the layout key changes.
    private let detailCells = NSMapTable<NSString, DetailCell>.strongToWeakObjects()
    private var detailKey: ModelDetailView.LayoutKey?
    /// One off-screen host measures every detail height (a new host per row costs more than the layout itself).
    private var measurer: NSHostingController<ModelDetailView>?
    /// Two lines: the name and, under it, what the model does.
    static let rowHeight: CGFloat = 40
    /// Space under an expanded row's details, inside its card.
    static let detailGap: CGFloat = 6

    override init() {
        super.init()
        table.rowHeight = Self.rowHeight
        table.intercellSpacing = NSSize(width: BrowserColumns.gap, height: 0)
        table.style = .plain
        table.usesAlternatingRowBackgroundColors = false
        table.allowsTypeSelect = false
        table.allowsColumnReordering = false
        table.allowsColumnResizing = true
        table.columnAutoresizingStyle = .noColumnAutoresizing
        table.backgroundColor = .clear
        table.gridStyleMask = []
        table.floatsGroupRows = false
        table.selectionHighlightStyle = .regular
        for column in BrowserColumn.allCases {
            let c = NSTableColumn(identifier: .init(column.rawValue))
            let elastic = BrowserColumns.specs[column]
            c.width = BrowserColumns.fixed[column] ?? elastic?.min ?? 0
            c.minWidth = elastic?.min ?? BrowserColumns.fixed[column] ?? 0
            c.maxWidth = elastic?.max ?? BrowserColumns.fixed[column] ?? 10_000
            c.resizingMask = elastic != nil ? .userResizingMask : []
            c.headerCell = HeaderCell(column: column)
            table.addTableColumn(c)
        }
        table.headerView = header
        header.list = self
        table.dataSource = self; table.delegate = self
        table.list = self
        scrollView.documentView = table
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = false
        scrollView.automaticallyAdjustsContentInsets = false
        scrollView.contentInsets = NSEdgeInsets()
        scrollView.contentView.postsFrameChangedNotifications = true
        NotificationCenter.default.addObserver(self, selector: #selector(widthChanged), name: NSView.frameDidChangeNotification, object: scrollView.contentView)
        let menu = NSMenu(); menu.addItem(withTitle: "Reset Column Widths", action: #selector(resetWidths), keyEquivalent: "").target = self
        header.menu = menu
        indicate(.default)
    }

    // MARK: Data

    /// New rows from the controller (a filter, the sort, or a data refresh). Only what differs is touched: rows that
    /// left are removed, rows that came are inserted, rows whose item changed are redrawn. A full reload re-creates every
    /// cell, which a search keystroke or a table Get's progress cannot afford.
    func show(_ rows: [BrowserItem]) {
        let previous = byID
        self.rows = rows
        byID = Dictionary(uniqueKeysWithValues: rows.map { ($0.id, $0) })
        details = details.filter { byID[$0.key] != nil }; detailHeights = detailHeights.filter { byID[$0.key] != nil }
        let changed = Set(rows.compactMap { item in previous[item.id].map { $0 == item ? nil : item.id } ?? nil })
        update(changed: changed)
    }
    /// The expanded set changed (a chevron, Space, ←/→, Esc): the detail rows slide in or out.
    func expansionChanged() { update(animated: true) }
    /// Every cell again, for a new appearance (the cells hold palette colours).
    func redraw() {
        lines = makeLines()
        table.reloadData()
        updateRowStyles()
    }

    private func makeLines() -> [Line] {
        let expanded = controller?.expanded ?? []
        return rows.flatMap { item -> [Line] in expanded.contains(item.id) ? [.item(item.id), .detail(item.id)] : [.item(item.id)] }
    }
    private func update(changed: Set<String> = [], animated: Bool = false) {
        let old = lines, new = makeLines()
        let selectedID = table.selectedRow >= 0 && table.selectedRow < old.count ? old[table.selectedRow].id : nil
        let oldExpanded = Set(
            old.compactMap { line -> String? in
                if case .detail(let id) = line { return id }; return nil
            })
        let newExpanded = Set(
            new.compactMap { line -> String? in
                if case .detail(let id) = line { return id }; return nil
            })
        let disclosureChanged = oldExpanded.symmetricDifference(newExpanded)
        lines = new
        guard !old.isEmpty else { table.reloadData(); updateRowStyles(); return }
        var removed = IndexSet(), inserted = IndexSet()
        for change in new.difference(from: old) {
            switch change {
            case .remove(let offset, _, _): removed.insert(offset)
            case .insert(let offset, _, _): inserted.insert(offset)
            }
        }
        // Filters and sorting are immediate; an expanding row slides, as an outline view's disclosure does.
        let motion = animated && !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        if !removed.isEmpty || !inserted.isEmpty {
            table.beginUpdates()
            if !removed.isEmpty { table.removeRows(at: removed, withAnimation: motion ? [.effectFade, .slideUp] : []) }
            if !inserted.isEmpty { table.insertRows(at: inserted, withAnimation: motion ? [.effectFade, .slideDown] : []) }
            table.endUpdates()
        }
        // A move is remove + insert to AppKit. Preserve the model, not the old row number or its fallback selection.
        if let selectedID {
            if let index = new.firstIndex(of: .item(selectedID)) {
                if table.selectedRow != index { table.selectRowIndexes([index], byExtendingSelection: false) }
            } else {
                table.deselectAll(nil)
            }
        }
        if !changed.isEmpty {
            var items = IndexSet(), details = IndexSet()
            for (i, line) in new.enumerated() where changed.contains(line.id) && !inserted.contains(i) {
                if case .detail = line { details.insert(i) } else { items.insert(i) }
            }
            if !items.isEmpty { table.reloadData(forRowIndexes: items, columnIndexes: IndexSet(integersIn: 0..<table.numberOfColumns)) }
            for i in details {
                if let cell = table.view(atColumn: 0, row: i, makeIfNecessary: false) as? DetailCell { cell.reload() }
            }
            if !details.isEmpty { table.noteHeightOfRows(withIndexesChanged: details) }
        }
        // Detail insertion leaves the existing item's cells intact; only its disclosure needs reconfiguration.
        let chevrons = IndexSet(
            new.indices.filter { index in
                if case .item(let id) = new[index] { return disclosureChanged.contains(id) && !inserted.contains(index) && !changed.contains(id) }
                return false
            })
        if !chevrons.isEmpty {
            table.reloadData(forRowIndexes: chevrons, columnIndexes: IndexSet(integer: table.column(withIdentifier: .init("chevron"))))
        }
        updateRowStyles(animated: motion)
    }
    private func item(_ id: String) -> BrowserItem? { byID[id] }

    /// Finder marks the sorted column.
    func indicate(_ sort: BrowserSort) {
        for c in table.tableColumns {
            guard let cell = c.headerCell as? HeaderCell else { continue }
            cell.sorted = sort.key.column == cell.column ? sort.ascending : nil
        }
        header.needsDisplay = true
    }
    func setHeaderHeight(_ h: CGFloat) {
        guard header.frame.height != h else { return }
        header.frame.size.height = h
        scrollView.tile()
    }

    // MARK: Columns

    @objc private func widthChanged() { applyLayout() }
    @objc private func resetWidths() { weights = BrowserColumns.defaultWeights; BrowserColumns.save(weights); applyLayout() }
    func applyLayout() {
        let width = scrollView.contentView.bounds.width
        guard width > 0 else { return }
        layout = BrowserColumns.solve(width: width, weights: weights)
        applying = true
        for c in table.tableColumns {
            guard let column = BrowserColumn(rawValue: c.identifier.rawValue) else { continue }
            let hidden = layout.hidden.contains(column)
            if c.isHidden != hidden { c.isHidden = hidden }
            let w = column == .flex ? layout.flex : (layout.widths[column] ?? 0).rounded(.down)
            if column == .flex { c.minWidth = 0; c.maxWidth = 100_000 }
            if abs(c.width - w) > 0.5 { c.width = w }
        }
        applying = false
        // Detail rows follow the width by their constraints; their height and arrangement change only with the layout key
        // (side by side or stacked, the description's wrap), so a live resize re-measures nothing between those points.
        let key = ModelDetailView.layoutKey(detailWidth)
        guard key != detailKey else { return }
        detailKey = key
        for case let cell as DetailCell in detailCells.objectEnumerator()?.allObjects ?? [] { cell.width = detailWidth }
        let rows = IndexSet(
            lines.indices.filter {
                if case .detail = lines[$0] { return true }; return false
            })
        if !rows.isEmpty { table.noteHeightOfRows(withIndexesChanged: rows) }
    }
    func tableViewColumnDidResize(_ notification: Notification) {
        guard !applying, let c = notification.userInfo?["NSTableColumn"] as? NSTableColumn, let column = BrowserColumn(rawValue: c.identifier.rawValue),
            BrowserColumns.specs[column] != nil
        else { return }
        weights = BrowserColumns.resized(column, to: c.width, in: layout, weights: weights)
        BrowserColumns.save(weights)
        applyLayout()
    }

    // MARK: Actions

    func focusFirstRow() {
        table.window?.makeFirstResponder(table)
        if !lines.isEmpty, table.selectedRow < 0 { table.selectRowIndexes([0], byExtendingSelection: false); table.scrollRowToVisible(0) }
    }
    private var selectedItem: BrowserItem? {
        guard table.selectedRow >= 0, table.selectedRow < lines.count, case .item(let id) = lines[table.selectedRow] else { return nil }
        return item(id)
    }
    func addSelected() { if let i = selectedItem, !i.added { controller?.add(i.id) } }
    func toggleSelected(open: Bool? = nil) {
        guard let i = selectedItem else { return }
        let isOpen = controller?.expanded.contains(i.id) ?? false
        if open == nil || open != isOpen { toggle(i.id) }
    }
    func toggle(_ id: String) {
        controller?.toggleExpanded(id)
        expansionChanged()
        if let i = lines.firstIndex(of: .item(id)) { table.selectRowIndexes([i], byExtendingSelection: false) }
    }
    /// ⌥-click on a chevron: every visible row opens, or all close.
    func toggleAll() {
        guard let c = controller else { return }
        if c.visibleExpanded.isEmpty { c.expanded.formUnion(rows.map(\.id)) } else { c.expanded.subtract(rows.map(\.id)) }
        expansionChanged()
    }
    /// Esc: collapse all, then clear the selection, then close.
    func escape() {
        if let c = controller, !c.visibleExpanded.isEmpty {
            c.expanded.subtract(rows.map(\.id)); expansionChanged()
        } else if table.selectedRow >= 0 {
            table.deselectAll(nil)
        } else {
            closeWindow()
        }
    }
    /// Whether a click on row `r` is the row's own (expand/collapse) rather than a control's or a detail's.
    func isItemRow(_ r: Int) -> Bool {
        guard r >= 0, r < lines.count, case .item = lines[r] else { return false }
        return true
    }
    /// A plain click (a trackpad tap included) anywhere on a row expands or collapses it at once, on mouse-down; ⌥ on the
    /// chevron opens or closes every visible row. The second click of a double-click is absorbed, so the old habit of
    /// double-clicking leaves the row open.
    func click(row r: Int, column: Int, event: NSEvent) {
        guard isItemRow(r), case .item(let id) = lines[r] else { return }
        table.window?.makeFirstResponder(table)
        guard event.clickCount < 2 else { return }
        let onChevron = column >= 0 && table.tableColumns[column].identifier.rawValue == BrowserColumn.chevron.rawValue
        if onChevron, event.modifierFlags.contains(.option) { toggleAll() } else { toggle(id) }
    }

    // MARK: Hover and dim

    var hoveredRow = -1 { didSet { if hoveredRow != oldValue { updateRowStyles() } } }
    /// `animated`: the dim follows an expanding row at the same pace; hover is always immediate.
    func updateRowStyles(animated: Bool = false) {
        let open = controller?.visibleExpanded ?? []
        NSAnimationContext.runAnimationGroup { context in
            context.allowsImplicitAnimation = animated
            if !animated { context.duration = 0 }
            for r in 0..<lines.count {
                guard let view = table.rowView(atRow: r, makeIfNecessary: false) as? BrowserRowView else { continue }
                style(view, row: r, open: open, animated: animated)
            }
        }
    }
    private func style(_ view: BrowserRowView, row r: Int, open: Set<String>, animated: Bool = false) {
        let line = lines[r]
        view.palette = palette
        view.part = {
            switch line {
            case .detail: return .cardBottom
            case .item(let id): return open.contains(id) ? .cardTop : .plain
            }
        }()
        view.hovered = r == hoveredRow
        if case .item(let id) = line { view.setAccessibilityLabel(item(id)?.accessibilityLabel) }
        // Every row at full contrast, whatever is expanded (figures are never greyed).
        if case .item(let id) = line { view.loaded = item(id)?.loadedDtype != nil } else { view.loaded = false }
        if case .item(let id) = line, let i = item(id) { view.added = i.added && i.loadedDtype == nil } else { view.added = false }
        view.needsDisplay = true
    }

    // MARK: Data source / delegate

    func numberOfRows(in tableView: NSTableView) -> Int { lines.count }
    func tableView(_ tableView: NSTableView, isGroupRow row: Int) -> Bool { if case .detail = lines[row] { return true }; return false }
    func tableView(_ tableView: NSTableView, shouldSelectRow row: Int) -> Bool { if case .item = lines[row] { return true }; return false }
    func tableView(_ tableView: NSTableView, didClick tableColumn: NSTableColumn) {
        guard let c = BrowserColumn(rawValue: tableColumn.identifier.rawValue) else { return }
        controller?.state.sort.click(c)
    }
    func tableView(_ tableView: NSTableView, heightOfRow row: Int) -> CGFloat {
        guard case .detail(let id) = lines[row] else { return Self.rowHeight }
        return detailHeight(id)
    }
    func tableView(_ tableView: NSTableView, rowViewForRow row: Int) -> NSTableRowView? {
        let v = BrowserRowView()
        style(v, row: row, open: controller?.visibleExpanded ?? [])
        return v
    }
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        switch lines[row] {
        case .detail(let id):
            guard tableColumn == nil, let item = item(id) else { return nil }
            return detailView(item)
        case .item(let id):
            guard let raw = tableColumn?.identifier.rawValue, let column = BrowserColumn(rawValue: raw), let item = item(id) else { return nil }
            return cell(item, column)
        }
    }

    // MARK: Cells (reused through the table's queue: a row scrolled or filtered back in costs no new views)

    private func reuse<V: NSView>(_ id: String, _ make: () -> V) -> V {
        if let v = table.makeView(withIdentifier: .init(id), owner: nil) as? V { return v }
        let v = make(); v.identifier = .init(id); return v
    }
    private func cell(_ item: BrowserItem, _ column: BrowserColumn) -> NSView? {
        let hot = item.loadedDtype != nil
        let p = hot ? palette.onLoadedRow : palette, physical = controller?.physicalBytes ?? 0
        switch column {
        case .chevron:
            let open = controller?.expanded.contains(item.id) ?? false
            let cell = reuse("chevron") { ChevronCell() }
            cell.configure(open: open, color: open ? p.t2 : p.t3) { [weak self] option in
                if option { self?.toggleAll() } else { self?.toggle(item.id) }
            }
            return cell
        case .icon:
            if let loaded = item.loadedDtype { return glyph("flame.fill", size: 13, color: .systemOrange, help: "Loaded at \(loaded)") }
            return glyph(item.icon, size: 13, color: p.t2, help: item.kindTitle)
        case .model:
            let cell = reuse("name") { NameCell() }
            cell.configure(name: item.name, purpose: item.purpose, palette: p, help: item.help)
            return cell
        case .kind: return wrapped(item.kindTitle, color: p.t2, help: nil)
        case .params: return label(item.params.isEmpty ? "\u{2014}" : item.params, color: p.t1, alignment: .right, digits: true)
        case .download:
            if item.downloadCheck { return glyph("checkmark", size: 11, color: p.t3, help: item.downloadHelp, alignment: .alignRight) }
            return label(item.downloadText, color: item.downloadBytes > 0 ? p.t1 : p.t2, help: item.downloadHelp, alignment: .right, digits: true)
        case .ram:
            // The Models table's figure, or its yellow triangle: one source for both windows.
            guard item.peak != nil else {
                let cell = reuse("triangle") { Cell(NSImageView()) }
                cell.imageView?.imageAlignment = .alignRight
                cell.imageView?.image = Self.triangle
                cell.imageView?.toolTip = item.ramHelp(physicalBytes: physical)
                cell.imageView?.setAccessibilityLabel("Not available")
                return cell
            }
            return label(item.ramText, color: item.ramExceeds(physicalBytes: physical) ? p.red : p.t1, help: item.ramHelp(physicalBytes: physical), alignment: .right, digits: true)
        case .dtype:
            let cell = reuse("formats") { FormatsCell() }
            cell.configure(item, palette: p, hot: hot)
            return cell
        case .licence: return wrapped(item.licenceText, color: item.commercial ? p.t1 : p.amber, help: item.licenceHelp)
        case .languages: return wrapped(item.languagesText, color: p.t2, help: item.languagesHelp)
        case .status:
            let color: NSColor = [.leaving: p.t2, .failedChecks: p.red, .update: p.accentText, .local: p.amber, .none: p.t3][item.status]!
            return wrapped(item.status.rawValue, color: color, help: item.statusHelp.isEmpty ? nil : item.statusHelp)
        case .flex: return nil
        case .action:
            let cell = reuse("action") { ActionCell() }
            cell.configure(item: item, palette: p, busy: item.added ? BrowserController.busyText(item.entry.row) : nil) { [weak self] in
                if item.added { self?.controller?.remove(item.id) } else { self?.controller?.add(item.id) }
            }
            return cell
        }
    }
    /// Words wrapped at a word onto a second line rather than cut mid-word (a kind, a licence, the languages, a status).
    private func wrapped(_ string: String, color: NSColor, help: String?) -> NSView {
        let cell = reuse("wrapped") { WrapCell() }
        cell.field.stringValue = string
        cell.field.textColor = color
        cell.field.toolTip = help
        cell.needsLayout = true
        return cell
    }
    private func label(
        _ string: String, color: NSColor, weight: NSFont.Weight = .regular, help: String? = nil, alignment: NSTextAlignment = .left, digits: Bool = false, inset: CGFloat = 0
    ) -> NSView {
        let cell = reuse("label-\(Int(inset))") { Cell(NSTextField(labelWithString: ""), leading: inset) }
        guard let field = cell.textField else { return cell }
        field.stringValue = string
        field.font = digits ? .monospacedDigitSystemFont(ofSize: 13, weight: weight) : .systemFont(ofSize: 13, weight: weight)
        field.textColor = color; field.alignment = alignment; field.lineBreakMode = .byTruncatingTail
        field.toolTip = help ?? (string.count > 14 ? string : nil)
        cell.needsLayout = true
        return cell
    }
    /// The Models table's missing-figure mark: a yellow triangle with a dark exclamation mark.
    static let triangle: NSImage? = NSImage(systemSymbolName: "exclamationmark.triangle.fill", accessibilityDescription: "Not available")?
        .withSymbolConfiguration(NSImage.SymbolConfiguration(pointSize: 12, weight: .regular).applying(.init(paletteColors: [ArsenalPalette.ink, .systemYellow])))
    private func glyph(_ symbol: String?, size: CGFloat, color: NSColor, help: String?, alignment: NSImageAlignment = .alignCenter) -> NSView {
        let cell = reuse("glyph") { Cell(NSImageView()) }
        guard let view = cell.imageView else { return cell }
        view.imageAlignment = alignment
        view.image = symbol.flatMap { Symbol.image($0, size: size) }
        view.contentTintColor = color; view.toolTip = help; view.setAccessibilityLabel(help)
        return cell
    }

    // MARK: Detail rows

    /// Detail content starts under the Model column.
    var detailLeading: CGFloat { (BrowserColumns.fixed[.chevron]! + BrowserColumns.fixed[.icon]!) + 2 * BrowserColumns.gap }
    private var detailWidth: CGFloat { max(300, table.bounds.width - detailLeading - 24) }
    private func detailView(_ item: BrowserItem) -> NSView {
        let id = item.id
        // Looked up at each render: a data refresh hands the cell its item's newest details.
        let cell = DetailCell(leading: detailLeading, trailing: 24, width: detailWidth) { [weak self] width in
            guard let self, let item = self.item(id) else { return nil }
            return self.content(self.modelDetail(item), item, width: width)
        }
        detailCells.setObject(cell, forKey: id as NSString)
        return cell
    }
    private func content(_ detail: ModelDetail, _ item: BrowserItem, width: CGFloat) -> ModelDetailView {
        ModelDetailView(
            detail: detail, palette: palette, width: width, licenceAmber: !item.commercial,
            choose: { [weak self] name in self?.controller?.chooseRunner(item.id, name) },
            act: { [weak self] action in self?.controller?.act(item.id, action) },
            readme: { [weak self] in await self?.controller?.readme(item.id) })
    }
    /// The item's details, built once per item revision (they decode results and format dates).
    private func modelDetail(_ item: BrowserItem) -> ModelDetail {
        if let cached = details[item.id], cached.item == item { return cached.detail }
        let detail = ModelDetail(item, chip: controller?.chip)
        details[item.id] = (item, detail)
        return detail
    }
    func detailHeight(_ id: String) -> CGFloat {
        guard let item = item(id) else { return 0 }
        let detail = modelDetail(item), width = detailWidth, key = ModelDetailView.layoutKey(width)
        if let cached = detailHeights[id], cached.key == key, cached.detail == detail { return cached.height }
        let view = content(detail, item, width: width)
        if let measurer { measurer.rootView = view } else { measurer = NSHostingController(rootView: view) }
        let height = ceil(measurer!.sizeThatFits(in: CGSize(width: width, height: 4000)).height) + Self.detailGap
        detailHeights[id] = (key, height, detail)
        return height
    }
}

/// An expanded row's detail cell: the hosted details span the row from under the Model column to the list's padding, so
/// they follow a resize by constraint; `width` re-renders them only when it changes their arrangement.
final class DetailCell: NSTableCellView {
    private let host: NSHostingView<ModelDetailView>?
    private let make: (CGFloat) -> ModelDetailView?
    private var key: ModelDetailView.LayoutKey
    var width: CGFloat {
        didSet {
            let k = ModelDetailView.layoutKey(width)
            guard k != key, let view = make(width) else { return }
            key = k; host?.rootView = view
        }
    }
    /// Its item changed: the same arrangement, new values.
    func reload() { if let view = make(width) { host?.rootView = view } }
    init(leading: CGFloat, trailing: CGFloat, width: CGFloat, make: @escaping (CGFloat) -> ModelDetailView?) {
        self.make = make; self.width = width; key = ModelDetailView.layoutKey(width)
        host = make(width).map { NSHostingView(rootView: $0) }
        super.init(frame: .zero)
        guard let host else { return }
        // The row's height is the measured one and its width the list's: the host adds no size constraints of its own,
        // which would hold it at the old arrangement's minimum width until `width` catches up.
        host.sizingOptions = []
        host.translatesAutoresizingMaskIntoConstraints = false
        addSubview(host)
        NSLayoutConstraint.activate([
            host.leadingAnchor.constraint(equalTo: leadingAnchor, constant: leading),
            host.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -trailing),
            host.topAnchor.constraint(equalTo: topAnchor), host.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -BrowserList.detailGap)
        ])
    }
    required init?(coder: NSCoder) { nil }
}

/// A cell view centring its one subview vertically.
/// Cells are laid out by frame, not constraints: a search keystroke or a filter brings dozens of rows in and out, and
/// their constraints were most of that cost.
final class Cell: NSTableCellView {
    private let content: NSView
    private let leading: CGFloat
    init(_ content: NSView, leading: CGFloat = 0) {
        self.content = content; self.leading = leading
        super.init(frame: .zero)
        addSubview(content)
        if let field = content as? NSTextField { textField = field }
        if let image = content as? NSImageView { imageView = image }
    }
    required init?(coder: NSCoder) { nil }
    /// The content across the cell from `leading`, centred vertically: a label at its own height, an image the cell's.
    override func layout() {
        super.layout()
        let height = (content as? NSTextField).map { ceil($0.cell?.cellSize.height ?? $0.intrinsicContentSize.height) } ?? bounds.height
        content.frame = NSRect(x: leading, y: ((bounds.height - height) / 2).rounded(), width: max(0, bounds.width - leading), height: height)
    }
    /// The selection is a quiet fill, not the system blue: text keeps its colour.
    override var backgroundStyle: NSView.BackgroundStyle {
        get { .normal }
        set {}
    }
}

/// A label of up to two lines, wrapped at word boundaries (a licence's name).
final class WrapCell: NSTableCellView {
    let field = NSTextField(wrappingLabelWithString: "")
    init() {
        super.init(frame: .zero)
        field.font = .systemFont(ofSize: 13)
        field.maximumNumberOfLines = 2
        field.lineBreakMode = .byWordWrapping
        field.cell?.truncatesLastVisibleLine = true
        field.isSelectable = false
        addSubview(field)
        textField = field
    }
    required init?(coder: NSCoder) { nil }
    /// Its wrapped height (two lines at most), centred.
    override func layout() {
        super.layout()
        let size = field.cell?.cellSize(forBounds: NSRect(x: 0, y: 0, width: bounds.width, height: bounds.height)) ?? .zero
        let height = min(bounds.height, ceil(size.height))
        field.frame = NSRect(x: 0, y: ((bounds.height - height) / 2).rounded(), width: bounds.width, height: height)
    }
    override var backgroundStyle: NSView.BackgroundStyle {
        get { .normal }
        set {}
    }
}

/// The Model column: the name, and under it what the model does in the table's secondary text (as the Models table
/// has it). Both lines carry the model's hover.
final class NameCell: NSTableCellView {
    let name = NSTextField(labelWithString: ""), purpose = NSTextField(labelWithString: "")
    init() {
        super.init(frame: .zero)
        addSubview(name); addSubview(purpose)
        for field in [name, purpose] { field.lineBreakMode = .byTruncatingTail }
        name.font = .systemFont(ofSize: 13, weight: .medium)
        purpose.font = .systemFont(ofSize: 11, weight: .medium)
        textField = name
    }
    required init?(coder: NSCoder) { nil }
    /// The name over what it does, 1 pt apart, centred as a pair (the name alone when there is no purpose line).
    override func layout() {
        super.layout()
        let h1 = ceil(name.cell?.cellSize.height ?? 16), h2 = purpose.isHidden ? 0 : ceil(purpose.cell?.cellSize.height ?? 14)
        let total = h1 + (purpose.isHidden ? 0 : 1 + h2)
        // Flipped or not, the name sits above: NSTableCellView is not flipped, so y grows upwards.
        let bottom = ((bounds.height - total) / 2).rounded()
        purpose.frame = NSRect(x: 0, y: bottom, width: bounds.width, height: h2)
        name.frame = NSRect(x: 0, y: bottom + total - h1, width: bounds.width, height: h1)
    }
    func configure(name text: String, purpose words: String, palette p: ArsenalPalette, help: String) {
        name.stringValue = text; name.textColor = p.t1; name.toolTip = help
        purpose.stringValue = words; purpose.textColor = p.t2; purpose.toolTip = words.isEmpty ? nil : help
        if purpose.isHidden != words.isEmpty { purpose.isHidden = words.isEmpty; needsLayout = true }
    }
    override var backgroundStyle: NSView.BackgroundStyle {
        get { .normal }
        set {}
    }
}

/// The disclosure chevron (`chevron.right`, rotated when open). Clicks reach the table (BrowserTableView.hitTest), which
/// treats the whole row as the target.
final class ChevronCell: NSTableCellView {
    private var action: (Bool) -> Void = { _ in }
    private let image = NSImageView()
    init() {
        super.init(frame: .zero)
        addSubview(image)
        setAccessibilityRole(.disclosureTriangle)
    }
    required init?(coder: NSCoder) { nil }
    override func layout() {
        super.layout()
        let size = image.intrinsicContentSize
        image.frame = NSRect(
            x: (BrowserColumns.pad + 6 - size.width / 2).rounded(), y: ((bounds.height - size.height) / 2).rounded(), width: size.width, height: size.height)
    }
    func configure(open: Bool, color: NSColor, action: @escaping (Bool) -> Void) {
        self.action = action
        image.image = Symbol.image(open ? "chevron.down" : "chevron.right", size: 9, weight: .bold)
        image.setAccessibilityLabel(open ? "Collapse" : "Expand")
        image.contentTintColor = color
        toolTip = open ? "Collapse" : "Expand"
    }
    /// VoiceOver's press on the disclosure triangle.
    override func accessibilityPerformPress() -> Bool { action(false); return true }
}

/// Add (neutral fill) or Remove (quiet outline), right-aligned with the list's padding.
final class ActionCell: NSTableCellView {
    let button = PillButton()
    init() {
        super.init(frame: .zero)
        addSubview(button)
    }
    required init?(coder: NSCoder) { nil }
    override func layout() {
        super.layout()
        button.frame = NSRect(x: bounds.width - BrowserColumns.pad - 70, y: ((bounds.height - 22) / 2).rounded(), width: 70, height: 22)
    }
    /// `busy`: why Remove refuses now (a Get or Load runs); its tooltip says so before the click does.
    func configure(item: BrowserItem, palette: ArsenalPalette, busy: String? = nil, action: @escaping () -> Void) {
        button.caption = item.added ? "Remove" : "Add"; button.filled = !item.added; button.palette = palette; button.handler = action
        button.toolTip = busy.map { "Not now: " + $0 } ?? (item.added ? "Remove from the table" : "Add to the table")
        button.setAccessibilityLabel("\(item.added ? "Remove" : "Add") \(item.name)")
        button.needsDisplay = true
    }
}

/// A small drawn button: Add = field fill + border, Remove = border only; hover fills with the selection grey.
final class PillButton: NSControl {
    var caption = "" { didSet { needsDisplay = true; hovering = false } }
    var filled = true, palette = ArsenalPalette.dark, handler: () -> Void = {}
    private var hovering = false { didSet { needsDisplay = true } }
    private var pressed = false { didSet { needsDisplay = true } }
    init() {
        super.init(frame: .zero)
        setAccessibilityRole(.button)
    }
    required init?(coder: NSCoder) { nil }
    // A drawn control has no cell for accessibility to read: it is a button element itself, titled by its caption.
    override func isAccessibilityElement() -> Bool { true }
    override func accessibilityRole() -> NSAccessibility.Role? { .button }
    override func accessibilityTitle() -> String? { caption }
    override func updateTrackingAreas() {
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: bounds, options: [.mouseEnteredAndExited, .activeInKeyWindow], owner: self))
    }
    override func mouseEntered(with event: NSEvent) { hovering = true }
    override func mouseExited(with event: NSEvent) { hovering = false }
    override func mouseDown(with event: NSEvent) { pressed = true }
    override func mouseUp(with event: NSEvent) {
        pressed = false
        if bounds.contains(convert(event.locationInWindow, from: nil)) { handler() }
    }
    override func accessibilityPerformPress() -> Bool { handler(); return true }
    override func draw(_ dirtyRect: NSRect) {
        let rect = bounds.insetBy(dx: 0.5, dy: 0.5), path = NSBezierPath(roundedRect: rect, xRadius: 6, yRadius: 6)
        (hovering || pressed ? palette.selFill : filled ? palette.field : .clear).setFill(); path.fill()
        palette.raisedBorder.setStroke(); path.lineWidth = 1; path.stroke()
        let text = NSAttributedString(
            string: caption, attributes: [.font: NSFont.systemFont(ofSize: 12, weight: .medium), .foregroundColor: filled || hovering ? palette.t1 : palette.t2])
        let s = text.size()
        text.draw(at: NSPoint(x: bounds.midX - s.width / 2, y: bounds.midY - s.height / 2))
    }
}

/// Row backgrounds: hover and selection fills, and the expanded card (top part on the row, bottom part on its detail).
final class BrowserRowView: NSTableRowView {
    enum Part { case plain, cardTop, cardBottom }
    var part = Part.plain
    var hovered = false
    /// A loaded model's row: the Models table's loaded blue (the card's top part too).
    var loaded = false
    /// A model already on the table and not loaded: a light-green wash (the card's top part too), under hover and selection.
    var added = false
    var palette = ArsenalPalette.dark
    static let inset: CGFloat = 8, corner: CGFloat = 9
    override var isEmphasized: Bool {
        get { false }
        set {}
    }
    override func drawBackground(in dirtyRect: NSRect) {
        let r = bounds.insetBy(dx: Self.inset, dy: 0)
        switch part {
        case .plain:
            if loaded { palette.hotRow.setFill(); NSBezierPath(roundedRect: r, xRadius: Self.corner, yRadius: Self.corner).fill() }
            if hovered && !isSelected { palette.hover.setFill(); NSBezierPath(roundedRect: r, xRadius: Self.corner, yRadius: Self.corner).fill() }
            if added && !isSelected { palette.addedRow.setFill(); NSBezierPath(roundedRect: r, xRadius: Self.corner, yRadius: Self.corner).fill() }
        case .cardTop, .cardBottom:
            let top = part == .cardTop
            let card =
                top
                ? NSRect(x: r.minX, y: r.minY, width: r.width, height: r.height + Self.corner)
                : NSRect(x: r.minX, y: r.minY - Self.corner, width: r.width, height: r.height + Self.corner - 6)
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath(rect: bounds).setClip()
            let path = NSBezierPath(roundedRect: card.insetBy(dx: 0.5, dy: 0.5), xRadius: Self.corner, yRadius: Self.corner)
            palette.raised.setFill(); path.fill()
            if top, loaded { palette.hotRow.setFill(); path.fill() }
            if top, added { palette.addedRow.setFill(); path.fill() }
            palette.raisedBorder.setStroke(); path.lineWidth = 1; path.stroke()
            NSGraphicsContext.restoreGraphicsState()
        }
    }
    override func drawSelection(in dirtyRect: NSRect) {
        guard part == .plain else { return }
        let shape = NSBezierPath(roundedRect: bounds.insetBy(dx: Self.inset, dy: 0), xRadius: Self.corner, yRadius: Self.corner)
        palette.selFill.setFill(); shape.fill()
        if added { palette.addedRow.setFill(); shape.fill() }
    }
    override func drawSeparator(in dirtyRect: NSRect) {}
}

/// The header in the window's top band: titles only (no header chrome), a hairline under them. A click sorts; a drag
/// that does not start on a column divider moves the window. Rows scroll under it, so it carries the window's own
/// ground: the canvas fill over the same behind-window material as the content (22 Oct: the header was transparent and
/// a scrolled 'Pocket TTS' row read through the column titles). The ground and titles are subviews that take no clicks.
final class BrowserHeaderView: NSTableHeaderView {
    weak var list: BrowserList?
    private let backdrop = PassThroughEffectView()
    private let titles = HeaderTitlesView()
    override init(frame: NSRect) {
        super.init(frame: frame)
        backdrop.material = .underWindowBackground; backdrop.blendingMode = .behindWindow; backdrop.state = .active
        for v: NSView in [backdrop, titles] {
            v.frame = bounds; v.autoresizingMask = [.width, .height]
            addSubview(v)
        }
        titles.header = self
    }
    required init?(coder: NSCoder) { nil }
    override var isOpaque: Bool { false }
    override func draw(_ dirtyRect: NSRect) {}
    // AppKit invalidates the header both ways (a sort sets the property, a column resize a rect); the titles follow.
    override var needsDisplay: Bool {
        didSet { if needsDisplay { titles.needsDisplay = true } }
    }
    override func setNeedsDisplay(_ invalidRect: NSRect) {
        super.setNeedsDisplay(invalidRect)
        titles.setNeedsDisplay(invalidRect)
    }
    override func viewWillDraw() {
        backdrop.isHidden = !Appearance.shared.palette.translucent
        super.viewWillDraw()
    }
    /// Tests: how many times the titles have drawn.
    var titleDraws: Int { titles.draws }
    /// The ground, the visible columns' titles and the hairline, in the header's coordinates.
    fileprivate func drawTitles() {
        let p = Appearance.shared.palette
        p.canvas.setFill(); bounds.fill()
        guard let table = tableView else { return }
        for (i, c) in table.tableColumns.enumerated() where !c.isHidden {
            (c.headerCell as? HeaderCell)?.drawInterior(withFrame: headerRect(ofColumn: i), in: self)
        }
        p.hairline.setFill()
        NSRect(x: 0, y: isFlipped ? bounds.maxY - 1 : 0, width: bounds.width, height: 1).fill()
    }
    override func mouseDown(with event: NSEvent) {
        guard let table = tableView, let window else { return }
        let start = convert(event.locationInWindow, from: nil)
        // A column divider: the table's own resize tracking.
        let nearEdge = table.tableColumns.indices.contains { i in
            !table.tableColumns[i].isHidden && table.tableColumns[i].resizingMask.contains(.userResizingMask) && abs(headerRect(ofColumn: i).maxX - start.x) < 5
        }
        if nearEdge { super.mouseDown(with: event); return }
        while let next = window.nextEvent(matching: [.leftMouseDragged, .leftMouseUp]) {
            if next.type == .leftMouseUp {
                let column = self.column(at: start)
                if column >= 0 { table.delegate?.tableView?(table, didClick: table.tableColumns[column]) }
                return
            }
            let p = convert(next.locationInWindow, from: nil)
            if hypot(p.x - start.x, p.y - start.y) > 3 { window.performDrag(with: event); return }
        }
    }
}

/// The header's drawing surface above its backdrop; clicks and drags fall through to the header.
final class HeaderTitlesView: NSView {
    weak var header: BrowserHeaderView?
    override var isFlipped: Bool { true }
    override var isOpaque: Bool { false }
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
    private(set) var draws = 0
    override func draw(_ dirtyRect: NSRect) { draws += 1; header?.drawTitles() }
}

final class PassThroughEffectView: NSVisualEffectView {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

/// Header title: 12 pt medium; the sorted column a chevron after its title.
final class HeaderCell: NSTableHeaderCell {
    let column: BrowserColumn
    var sorted: Bool?
    init(column: BrowserColumn) {
        self.column = column
        super.init(textCell: "")
        let hidden: Set<BrowserColumn> = [.chevron, .icon, .flex, .action]
        stringValue = hidden.contains(column) ? "" : column.title
    }
    required init(coder: NSCoder) { fatalError() }
    private var right: Bool { [.params, .download, .ram].contains(column) }
    private func symbol(_ name: String, _ size: CGFloat, _ weight: NSFont.Weight, _ color: NSColor) -> NSImage? {
        Symbol.image(name, size: size, weight: weight, color: color)
    }
    override func drawInterior(withFrame frame: NSRect, in view: NSView) {
        let color = Appearance.shared.palette.t2
        func draw(_ image: NSImage, at x: CGFloat) {
            let s = image.size
            image.draw(
                in: NSRect(x: x, y: frame.midY - s.height / 2, width: s.width, height: s.height), from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil
            )
        }
        let chevron = sorted.flatMap { symbol($0 ? "chevron.up" : "chevron.down", 7, .bold, color) }
        guard !stringValue.isEmpty else { return }
        let text = NSAttributedString(string: stringValue, attributes: [.font: NSFont.systemFont(ofSize: 12, weight: .medium), .foregroundColor: color])
        let size = text.size(), extra = chevron.map { $0.size.width + 3 } ?? 0
        let inset: CGFloat = column == .dtype ? 8 : 0
        let x = right ? frame.maxX - BrowserColumns.gap - extra - size.width : frame.minX + inset
        text.draw(at: NSPoint(x: x, y: frame.midY - size.height / 2))
        if let chevron { draw(chevron, at: x + size.width + 3) }
    }
}

final class BrowserTableView: NSTableView, NSViewToolTipOwner, ToolTipRegions {
    /// The tooltip regions this view registered with AppKit (tag → rect and user data), for the hover tests.
    private(set) var toolTipRegions: [NSView.ToolTipTag: ToolTipRegion] = [:]
    override func addToolTip(_ rect: NSRect, owner: Any, userData data: UnsafeMutableRawPointer?) -> NSView.ToolTipTag {
        let tag = super.addToolTip(rect, owner: owner, userData: data)
        toolTipRegions[tag] = ToolTipRegion(rect: rect, userData: data)
        return tag
    }
    override func removeAllToolTips() { super.removeAllToolTips(); toolTipRegions = [:] }
    override func removeToolTip(_ tag: NSView.ToolTipTag) { super.removeToolTip(tag); toolTipRegions[tag] = nil }

    weak var list: BrowserList?

    // MARK: Hovers
    // The table answers the hit test over an item row's labels, glyphs and cells (below), so their own tooltips never
    // showed: AppKit's tooltip manager shows only the hit view's (HoverReachTests). The table carries them instead: one
    // tooltip rect per visible cell that has a tooltip, re-registered when the cells change, and the text is the
    // innermost such cell's under the pointer.
    private var cellTips: [(rect: NSRect, text: String)] = []
    /// Every visible item-row view with a tooltip, in table coordinates.
    func cellToolTips() -> [(rect: NSRect, text: String)] {
        var tips: [(rect: NSRect, text: String)] = []
        let rows = self.rows(in: visibleRect)
        guard rows.length > 0 else { return [] }
        func walk(_ view: NSView) {
            guard !view.isHidden else { return }
            if let text = view.toolTip, !text.isEmpty { tips.append((view.convert(view.bounds, to: self), text)) }
            view.subviews.forEach(walk)
        }
        for r in rows.location..<(rows.location + rows.length) where list?.isItemRow(r) == true {
            if let rowView = rowView(atRow: r, makeIfNecessary: false) { walk(rowView) }
        }
        return tips
    }
    override func viewWillDraw() {
        super.viewWillDraw()
        let tips = cellToolTips()
        guard tips.map(\.text) != cellTips.map(\.text) || tips.map(\.rect) != cellTips.map(\.rect) else { return }
        cellTips = tips
        removeAllToolTips()
        for tip in tips { addToolTip(tip.rect, owner: self, userData: nil) }
    }
    func view(_ view: NSView, stringForToolTip tag: NSView.ToolTipTag, point: NSPoint, userData data: UnsafeMutableRawPointer?) -> String {
        cellToolTips().filter { $0.rect.contains(point) }.min { $0.rect.width * $0.rect.height < $1.rect.width * $1.rect.height }?.text ?? ""
    }

    override func keyDown(with event: NSEvent) {
        switch event.keyCode {
        case 36, 76: list?.addSelected() // Return, Enter
        case 49: list?.toggleSelected() // Space
        case 124: list?.toggleSelected(open: true) // →
        case 123: list?.toggleSelected(open: false) // ←
        case 53: list?.escape()
        default:
            if event.charactersIgnoringModifiers == "/" { list?.focusSearch() } else { super.keyDown(with: event) }
        }
    }
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.filter { $0.owner === self }.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect], owner: self))
    }
    override func mouseMoved(with event: NSEvent) { list?.hoveredRow = row(at: convert(event.locationInWindow, from: nil)) }
    override func mouseExited(with event: NSEvent) { list?.hoveredRow = -1 }
    /// The empty canvas below the last row moves the window.
    override var mouseDownCanMoveWindow: Bool { false }
    /// An item row is one click target. NSTableView hit-tests to itself and forwards a mouse-down only to controls (the
    /// Add/Remove button), so the chevron's cell view never saw a click: a single click just selected the row and only a
    /// double-click (the old doubleAction) opened it. Text cells were inconsistent too; here every point of a row except
    /// its controls lands on the table's mouse-down. A control keeps its own hit: NSWindow sends the mouse-up to the view
    /// that took the mouse-down, and the Add/Remove pill acts on its own mouse-up, so a table that claimed the
    /// mouse-down would swallow the click. (NSTableView's own hit-test can answer the table even over the pill, so the
    /// control is returned explicitly.)
    override func hitTest(_ point: NSPoint) -> NSView? {
        let hit = super.hitTest(point)
        guard let hit, let list else { return hit }
        let p = convert(point, from: superview), r = row(at: p)
        guard list.isItemRow(r) else { return hit }
        return control(at: p, row: r) ?? self
    }
    /// The enabled control under `point` (table coordinates) in row `r`, not counting labels and glyphs: it keeps its
    /// own click.
    func control(at point: NSPoint, row r: Int) -> NSControl? {
        guard let control = rowView(atRow: r, makeIfNecessary: false)?.hitTest(point) as? NSControl, control.isEnabled,
            !(control is NSTextField), !(control is NSImageView)
        else { return nil }
        return control
    }
    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil), r = row(at: point)
        if r < 0 { window?.performDrag(with: event); return }
        guard let list, list.isItemRow(r), !event.modifierFlags.contains(.control), control(at: point, row: r) == nil else {
            super.mouseDown(with: event); return
        }
        list.click(row: r, column: column(at: point), event: event)
    }
}

/// The Precision cell: the formats widest first, as the Models table labels them and with its marks: the chosen one in
/// the table's gold, the loaded one edged in white, int4's orange triangle where its quality check did not pass or has
/// not run, and the yellow bolt before them where the model has custom kernels for this chip. A Precision filter that
/// matched one format sets it in the primary colour and the others quiet.
final class FormatsCell: NSTableCellView {
    private var item: BrowserItem?
    private var palette = ArsenalPalette.dark
    private var hot = false
    static let font = NSFont.systemFont(ofSize: 13), matchedFont = NSFont.systemFont(ofSize: 13, weight: .semibold)
    static let leading: CGFloat = 8, boltWidth: CGFloat = 14, pill: CGFloat = 5
    override var isFlipped: Bool { true }
    override init(frame: NSRect) { super.init(frame: frame); clipsToBounds = true }
    required init?(coder: NSCoder) { nil }
    func configure(_ item: BrowserItem, palette: ArsenalPalette, hot: Bool) {
        self.item = item; self.palette = palette; self.hot = hot
        let state =
            item.matchedDtype.map { "Shown at \($0): Download and Peak memory are for this format" }
            ?? [item.chosenDtype.map { "Chosen in the table: \($0)" }, item.loadedDtype.map { "Loaded at \($0)" }].compactMap { $0 }
            .joined(separator: "\n")
        let tip = [item.int4Warning, state.isEmpty ? nil : state].compactMap { $0 }.joined(separator: "\n")
        toolTip = tip.isEmpty ? nil : tip
        setAccessibilityLabel(item.dtypeText)
        needsDisplay = true
    }
    override func draw(_ dirtyRect: NSRect) {
        guard let item else { return }
        var x = Self.leading
        if item.fastKernels {
            Symbol.draw(
                "bolt.fill", centredIn: NSRect(x: x, y: 0, width: Self.boltWidth, height: bounds.height), size: 12, weight: .semibold,
                color: hot ? TierControl.boltTint : palette.bolt)
            x += Self.boltWidth + 4
        }
        for (i, d) in item.dtypes.enumerated() {
            if i > 0 { x += draw(" \u{00b7} ", at: x, font: Self.font, color: palette.t3) }
            let matched = item.matchedDtype.map { $0 == d }
            let font = matched == true ? Self.matchedFont : Self.font
            let chosen = d == item.chosenDtype, loaded = d == item.loadedDtype
            let warn = d == "int4" && item.int4Warning != nil ? TierSegments.Control.triangle(onGold: chosen) : nil
            let size = (d as NSString).size(withAttributes: [.font: font])
            let extra = warn.map { $0.size.width + 3 } ?? 0
            if chosen || loaded {
                let pill = NSRect(x: x, y: (bounds.height - size.height) / 2 - 1, width: size.width + extra + 2 * Self.pill, height: size.height + 2)
                let shape = NSBezierPath(roundedRect: pill.insetBy(dx: 0.75, dy: 0.75), xRadius: 4, yRadius: 4)
                if chosen { ArsenalPalette.gold.setFill(); NSBezierPath(roundedRect: pill, xRadius: 4, yRadius: 4).fill() }
                // Loaded: the table's white edge (on the gold, or alone when another precision is chosen).
                if loaded { (chosen || hot ? NSColor.white : palette.t2).setStroke(); shape.lineWidth = 1.5; shape.stroke() }
                _ = draw(d, at: x + Self.pill, font: font, color: chosen ? ArsenalPalette.ink : hot ? ArsenalPalette.hotText : palette.t1)
                if let warn { drawTriangle(warn, at: x + Self.pill + size.width + 3) }
                x += pill.width
            } else {
                x += draw(d, at: x, font: font, color: matched == false ? palette.t3 : matched == true ? palette.t1 : palette.t2)
                if let warn { drawTriangle(warn, at: x + 3); x += extra }
            }
            if x > bounds.width { break }
        }
    }
    private func drawTriangle(_ image: NSImage, at x: CGFloat) {
        let s = image.size
        image.draw(
            in: NSRect(x: x, y: (bounds.height - s.height) / 2, width: s.width, height: s.height), from: .zero, operation: .sourceOver, fraction: 1,
            respectFlipped: true, hints: nil)
    }
    private func draw(_ text: String, at x: CGFloat, font: NSFont, color: NSColor) -> CGFloat {
        let s = NSAttributedString(string: text, attributes: [.font: font, .foregroundColor: color])
        let size = s.size()
        s.draw(at: NSPoint(x: x, y: (bounds.height - size.height) / 2))
        return size.width
    }
}
