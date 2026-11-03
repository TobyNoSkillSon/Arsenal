// SPDX-License-Identifier: MIT
import Combine
import Foundation
import ArsenalCore

/// The browser's state: every model the service's catalog lists, the kinds, the filters, the shown order and the
/// expanded rows. Add, Remove and the runner choice go through `ModelsService`. While the window is open it follows the
/// service's events (no polling); closed, it holds nothing live.
@MainActor final class BrowserController: ObservableObject {
    let service: any ModelsService
    @Published private(set) var items: [BrowserItem] = []
    @Published private(set) var kinds: [KindSpec] = []
    @Published var state = BrowserState() { didSet { if state != oldValue { reorder() } } }
    /// The shown rows, in order. Recomputed when filters, search or sort change; a data refresh (an Add, a Remove, a
    /// runner choice, a Get in the table) updates the rows in place, so nothing moves under the pointer.
    @Published private(set) var shown: [BrowserItem] = []
    /// Expanded rows by model id (kept for the session; a filtered-out one comes back expanded).
    @Published var expanded: Set<String> = []
    /// Folded/open facet sections (session only).
    @Published var openSections: Set<BrowserFacet> = Set(BrowserFacet.allCases.filter(\.openByDefault))
    /// Folded kind list and modality grid; kept across launches.
    @Published private(set) var foldedGroups: Set<SidebarGroup> = Set(
        (UserDefaults.standard.stringArray(forKey: BrowserController.foldedKey) ?? []).compactMap(SidebarGroup.init(rawValue:)))
    static let foldedKey = "BrowserFoldedGroups"
    func toggleFold(_ g: SidebarGroup) {
        if foldedGroups.contains(g) { foldedGroups.remove(g) } else { foldedGroups.insert(g) }
        UserDefaults.standard.set(foldedGroups.map(\.rawValue).sorted(), forKey: Self.foldedKey)
    }
    /// A refused Add, Remove or runner choice, prefixed with the model's name. Shown in red in the sidebar's count line
    /// (or the collapsed toolbar) until the next action or a click on it.
    @Published var lastError: String?
    let physicalBytes: UInt64
    /// This Mac's memory as it is named (36 GB); Fits itself compares bytes under the Mac-wide line (`CatalogMatch.fits`).
    let physicalGB: Double
    /// Fits' hover: the line in this Mac's own numbers.
    var fitsHelp: String {
        String(
            format: "Models that run within the %.0f GB Arsenal uses on this %.0f GB Mac with nothing else running, at a precision that fits",
            (physicalGB * MacMemoryGuard.safeInUseFraction).rounded(.down),
            physicalGB)
    }
    let chip: String?
    /// Asks before Remove deletes a model's weights from this Mac: `.deleteWeights` or `.cancel` (a model on the table
    /// always has weights, so Remove cannot keep them).
    var confirmRemove: (ModelRow) -> RemoveChoice = { _ in .deleteWeights }
    /// Add's download: the table's own Get at that precision (download plan, consent, code approval). Returns why nothing
    /// started, or nil. Unset (tests), Add asks the service directly.
    var getModel: ((ModelRow, String) async -> String?)?
    /// Remove's delete: the table's own trash (it reads the files on disk, refuses a loaded model before asking and names
    /// the size of every precision). Unset (tests), Remove asks `confirmRemove` and calls the service.
    var removeModel: ((ModelRow) -> Void)?
    /// Opens a model's tool window (the app's ToolCenter).
    var openTool: (ToolKind, ModelRow) -> Void = { _, _ in }
    func act(_ id: String, _ action: ModelDetail.Action) {
        guard let row = items.first(where: { $0.id == id })?.entry.row else { return }
        switch action {
        case .openTool(let tool, _): if let kind = ToolKind(rawValue: tool) { openTool(kind, row) }
        case .docs: break // the detail view shows it (`readme`)
        case .rollback:
            Task {
                do {
                    try await MemoryLoadPresenter.perform { force in
                        try await LoadMemoryContext.$force.withValue(force) { try await self.service.rollbackRevision(id: id) }
                    }
                } catch is CancellationError {
                } catch { lastError = "\(row.name): \(TableController.message(error))" }
            }
        }
    }
    /// A model's README for the detail's Docs popover; nil when it cannot be read.
    func readme(_ id: String) async -> String? { try? await service.modelREADME(id: id) }
    private var events: Task<Void, Never>?
    /// Items as built from their entries, reused while the entry is unchanged.
    private var built: [String: (entry: CatalogEntry, item: BrowserItem)] = [:]
    private var builtKinds: [KindSpec] = []
    /// One refresh at a time, with one more for events that arrived during it (a Get reports every chunk).
    private var refresher: Task<Void, Never>?
    private var refreshWanted = false
    private var generation = 0

    init(service: any ModelsService, physicalBytes: UInt64 = ProcessInfo.processInfo.physicalMemory, chip: String? = HostChip.current) {
        self.service = service; self.chip = chip
        self.physicalBytes = physicalBytes
        physicalGB = (Double(physicalBytes) / 1_073_741_824).rounded()
    }

    func refresh() async {
        async let e = service.catalog(query: CatalogQuery())
        async let k = service.kinds()
        let (entries, kinds) = await (e, k)
        apply(entries: entries, kinds: kinds)
    }
    /// Tests and renders set the data directly.
    func apply(entries: [CatalogEntry], kinds: [KindSpec]) {
        // Only what changed is published: the sidebar and the list redraw on every publish, and most refreshes (another
        // model's download progress) change nothing here.
        let serviceKinds = kinds, kinds = KindPresentation.expand(kinds)
        if self.kinds != kinds { self.kinds = kinds }
        let byID = Dictionary(uniqueKeysWithValues: kinds.map { ($0.id, $0) })
        let first = items.isEmpty
        // An item is rebuilt only when its entry changed (a Get's progress changes one row): building one reads its
        // figures, sizes and hovers, which for every model on every refresh cost more than the redraw.
        if kinds != builtKinds { built = [:]; builtKinds = kinds }
        let items = entries.map { e -> BrowserItem in
            if let hit = built[e.id], hit.entry == e { return hit.item }
            let shown = KindPresentation.id(kind: e.kind, tasks: e.tasks, kinds: serviceKinds)
            let item = BrowserItem(e, kind: byID[shown] ?? byID[e.kind], shown: shown)
            built[e.id] = (e, item)
            return item
        }
        if built.count > entries.count {
            let ids = Set(entries.map(\.id))
            built = built.filter { ids.contains($0.key) }
        }
        guard items != self.items || first else { return }
        self.items = items
        if first { reorder() } else { updateInPlace() }
    }
    private func reorder() { let next = state.visible(items, physicalBytes: physicalBytes); if next != shown { shown = next } }
    private func updateInPlace() {
        let byID = Dictionary(uniqueKeysWithValues: items.map { ($0.id, $0) })
        // Visible rows come back projected onto the dtype their filter matched; a shown row that no longer matches
        // stays, at its default dtype.
        let projected = Dictionary(uniqueKeysWithValues: state.visible(items, physicalBytes: physicalBytes).map { ($0.id, $0) })
        let visible = Set(projected.keys)
        var next = shown.compactMap { projected[$0.id] ?? byID[$0.id] }
        // A model that newly matches (a local folder dropped in) goes last; nothing already shown moves.
        let present = Set(next.map(\.id))
        next += items.filter { visible.contains($0.id) && !present.contains($0.id) }.map { projected[$0.id] ?? $0 }
        if next != shown { shown = next }
    }

    func start() {
        guard events == nil else { return }
        events = Task { [weak self] in
            guard let self else { return }
            let stream = await self.service.events()
            self.scheduleRefresh()
            for await event in stream {
                if Task.isCancelled { break }
                if case .rowsChanged = event { self.scheduleRefresh() }
            }
        }
    }
    func stop() {
        events?.cancel(); events = nil; lastError = nil
        refresher?.cancel(); refresher = nil; refreshWanted = false; generation += 1
    }
    /// A burst of events (a download's progress) is one refresh, plus one trailing refresh for what arrived during it.
    private func scheduleRefresh() {
        refreshWanted = true
        guard refresher == nil else { return }
        let generation = generation
        refresher = Task { [weak self] in
            while let self, generation == self.generation, self.refreshWanted, !Task.isCancelled {
                self.refreshWanted = false
                await self.refresh()
                // AsyncStream hands a buffered burst over several turns: let it arrive before deciding to refresh again.
                try? await Task.sleep(for: .milliseconds(10))
            }
            if let self, generation == self.generation { self.refresher = nil }
        }
    }
    /// Tests: until the scheduled refreshes have run.
    func waitUntilSettled() async {
        while let refresher { await refresher.value }
    }

    /// The kind list: kinds the catalog has models of, in the service's (the table's) order, with their totals.
    var kindRows: [(kind: KindSpec, count: Int)] {
        kinds.compactMap { k in
            let n = items.filter { $0.kind == k.id }.count; return n > 0 ? (k, n) : nil
        }
    }
    var addedCount: Int { items.filter(\.added).count }
    var count: String { BrowserState.count(shown: shown.count, total: items.count, added: addedCount, filtered: state.filtering) }
    var kindTitle: String? { state.kind.flatMap { k in kinds.first { $0.id == k }?.title } }
    var summary: String { state.summary(kindTitle: kindTitle) }
    var noResults: String { state.noResults(kindTitle: kindTitle) }
    /// The sidebar's count line: the count, or the last refusal in red.
    /// `help`: the raw message behind an error line's plain words (`ErrorWording`), for its hover.
    struct Line: Equatable { var text: String; var error: Bool; var help: String? }
    var footer: Line { lastError.map(Self.errorLine) ?? Line(text: count, error: false) }
    /// A refusal as it shows: a raw system message in plain words, the raw text on hover.
    static func errorLine(_ error: String) -> Line {
        let worded = ErrorWording.reword(error)
        return Line(text: worded?.text ?? error, error: true, help: worded?.help)
    }
    /// The collapsed toolbar's middle text: the filter summary, or the last refusal in red.
    var toolbarSummary: Line { lastError.map(Self.errorLine) ?? Line(text: summary, error: false) }
    func dismissError() { lastError = nil }
    /// Modalities no model has on that side (drawn disabled so the grid keeps its shape).
    func available(_ m: Modality, input: Bool) -> Bool { items.contains { (input ? $0.modalitiesIn : $0.modalitiesOut).contains(m.rawValue) } }

    func detail(_ id: String) -> ModelDetail? { items.first { $0.id == id }.map { ModelDetail($0, chip: chip) } }
    func toggleExpanded(_ id: String) { if expanded.contains(id) { expanded.remove(id) } else { expanded.insert(id) } }
    /// Expanded rows the list shows now: the dim applies only while one of them is visible.
    var visibleExpanded: Set<String> { expanded.intersection(shown.map(\.id)) }

    /// Add: downloads the model at its shown precision; it joins the table as the download starts (the table holds exactly
    /// the models with weights on this Mac or a download under way). The row flips to Remove at once and stays where it is.
    func add(_ id: String) {
        guard let i = items.firstIndex(where: { $0.id == id }), !items[i].added else { return }
        let item = items[i], precision = items[i].chosenDtype ?? items[i].defaultDtype
        lastError = nil
        setAdded(id, true)
        Task {
            let failure: String?
            if let getModel {
                failure = await getModel(item.entry.row, precision)
            } else {
                do { try await service.get(id: id, precision: precision); failure = nil } catch is HFGateError {
                    failure = nil // The table's row waits for access and counts down; the pages are open.
                } catch { failure = TableController.message(error) }
            }
            if let failure { lastError = "\(item.name): \(failure)" }
            await refresh()
        }
    }
    /// Remove, as the table's trash: deletes the model's weights after one confirmation, which takes it off the table.
    /// While a Get or Load runs it refuses, as the table hides its trash then: removing would leave the download running
    /// with no row to cancel it from.
    func remove(_ id: String) {
        guard let i = items.firstIndex(where: { $0.id == id }), items[i].added else { return }
        let row = items[i].entry.row
        lastError = nil
        if let busy = Self.busyText(row) { lastError = "\(row.name): \(busy)"; return }
        if let removeModel { removeModel(row); return }
        // A failed Get left no weights: Remove only clears the row.
        let hasWeights = row.precisions.contains { [.ready, .loaded, .updateAvailable].contains(row.states[$0]?.state) } || !row.withdrawnTiers.isEmpty
        if hasWeights, confirmRemove(row) != .deleteWeights { return }
        setAdded(id, false)
        Task {
            do { try await service.remove(id: id, deleteWeights: hasWeights) } catch { lastError = "\(row.name): \(TableController.message(error))" }
            await refresh()
        }
    }
    /// Why a row cannot be removed now (a Get or Load of some dtype is under way), or nil.
    static func busyText(_ row: ModelRow) -> String? {
        let states = row.precisions.compactMap { row.states[$0]?.state }
        if states.contains(where: { $0 == .downloading || $0 == .deriving }) { return "a Get is running; cancel it in the table or wait." }
        if states.contains(.loading) { return "still loading; wait, then Remove." }
        return nil
    }
    private func setAdded(_ id: String, _ added: Bool) {
        if let i = items.firstIndex(where: { $0.id == id }) { items[i].added = added; items[i].entry.added = added }
        if let j = shown.firstIndex(where: { $0.id == id }) { shown[j].added = added; shown[j].entry.added = added }
    }
    /// The runner choice: the accepted default clears the override.
    func chooseRunner(_ id: String, _ name: String) {
        guard let item = items.first(where: { $0.id == id }) else { return }
        lastError = nil
        if let choice = ModelDetail(item).runners.first(where: { $0.name == name }), !choice.eligible {
            lastError = "\(item.name): \(choice.reason ?? "That runner does not support this Mac.")"
            return
        }
        let fallback = item.entry.row.defaultRunner ?? item.entry.runners.first(where: \.acceptedDefault)?.name
        Task {
            do { try await service.setRunner(id: id, runner: name == fallback ? nil : name) } catch { lastError = "\(item.name): \(TableController.message(error))" }
            await refresh()
        }
    }
}
