// SPDX-License-Identifier: MIT
import AppKit
import Combine
import ArsenalCore

/// The table's rows, group order and in-flight preferences. Every action goes through `ModelsService`; the service owns persistence.
@MainActor final class TableController: ObservableObject {
    /// A model's declared operation names as the tool windows know them (Show in menu names its capability windows).
    var declaredOperations: (String) -> Set<String>? = { _ in nil }
    let service: any ModelsService
    @Published private(set) var rows: [ModelRow] = []
    @Published private(set) var kinds: [KindSpec] = []
    /// Every row the service holds, on the table or not: the menu's memory lines and Unload All read it from here.
    private(set) var everyRow: [ModelRow] = []
    /// The catalog's rows at the last full refresh: an installed catalog model off the table is listed from these.
    private var catalogRows: [ModelRow] = []
    /// The rows the menu builds tools and status from: the service's own rows (the table and loaded models), not the
    /// installed catalog models the table lists off the table, so listing them never adds menu items.
    var menuRows: [ModelRow] {
        let service = Set(everyRow.map(\.id))
        return rows.filter { service.contains($0.id) || $0.kind == Unlisted.kind }
    }
    /// Remembered positions survive absent kinds. Only the initial seed and user saves write this list.
    @Published private var userOrder: [String] = []
    var groupOrder: [String] { Self.order(kinds: kinds, saved: userOrder).map(\.id) }
    private var refreshSequence: UInt64 = 0
    private var appliedRefreshSequence: UInt64 = 0
    private var orderLoaded = false
    private var confirmedOrder: [String] = []
    private var orderWrites: [[String]] = []
    private var orderWriter: Task<Void, Never>?
    @Published private(set) var catalog: CatalogStatus?
    @Published var lastError: String?
    /// The hover of `lastError` when it was reworded: the raw message behind its plain words (`ErrorWording`).
    private(set) var lastErrorRaw: (text: String, raw: String)?
    /// The footer's hover for `lastError`: the raw message when the line rewords one, else the line itself.
    var lastErrorHelp: String? { lastError.map { text in lastErrorRaw?.text == text ? lastErrorRaw!.raw : text } }
    /// Optimistic selection while the precision preference is being saved.
    @Published private(set) var previewPrecision: [String: String] = [:]
    private var precisionWrites: [(id: String, name: String, dtype: String)] = []
    private var precisionWriter: Task<Void, Never>?
    /// While a group header is dragged: the group and the slot (0…groups) it would drop into.
    @Published var drag: (group: String, slot: Int)?
    /// Render harness: the model whose action cell is drawn under the pointer, and the group header drawn hovered.
    @Published var memoryNotice: String?
    @Published var renderHover: String?
    @Published var renderHoverGroup: String?
    @Published var renderHoverPinned = false
    /// This Mac's chip ("M5 Max"), for the badges' tooltips.
    let chip: String?
    /// Each row's settings as its folder declares them, with current values and `pending` (read with the rows).
    @Published var settings: [String: [SettingState]] = [:]
    /// Values a click set and showed at once whose write (and its own refresh) has not finished, by model and key: a
    /// refresh that read the service before the write landed must not put the old value back for a moment.
    private var pendingSettings: [String: [String: (write: UUID, value: JSONValue)]] = [:]
    /// Each table model's declared runners (from the catalog; read at start, on catalog changes and for new rows).
    @Published private(set) var runners: [String: [CatalogRunner]] = [:]
    /// The group each row shows in when it is not its kind's own (KindPresentation: a chat-kind model that reads
    /// documents, translates or extracts), from the catalog's tasks.
    @Published private(set) var shownKinds: [String: String] = [:]
    /// The settings' Runner row: one available runner is named; several eligible runners can be chosen.
    nonisolated static let runnerKey = "__runner"
    /// The settings' "Show in menu" switch (C1 `toolsShown`), present for a model whose folder declares tools.
    static let toolsKey = "__tools"
    /// The settings' first row, on every model: Keep loaded (`keepHot`, the service's per-model preference).
    nonisolated static let keepKey = "__keep"
    /// A row's settings card, beside its gear inside the table (its gear opens it).
    let settingsCard = SettingsCard()
    /// The row whose settings card is open now (its row in the table shows it).
    @Published private(set) var settingsRow: String?
    /// Each row's settings panel from its folder (`settings.toml`); a row without one gets the generic list.
    @Published var panels: [String: SettingsPanel] = [:]
    /// Voices a loaded model lists (its `voices` op), for voice settings.
    @Published private(set) var voices: [String: [VoiceChoice]] = [:]
    @Published private(set) var voiceErrors: [String: String] = [:]
    @Published private(set) var loadingVoices: Set<String> = []
    /// The voice being previewed (speaker glyph), if any, with the model speaking it: two models can share a voice id.
    struct VoicePreview: Equatable { var model: String; var voice: String }
    @Published private(set) var previewing: VoicePreview?
    /// Each model's last failed action from its settings (a refused setting, a failed Reload), for its settings menu;
    /// cleared when that model's next action starts or its settings close. The table's footer shows `lastError`.
    @Published var rowErrors: [String: String] = [:]
    /// Asks how to remove a row whose weights are on this Mac (the menu closes and shows an alert). The table no longer
    /// asks: its Delete removes every weight (`confirmDeleteWeights`). Kept for the menu's own harness.
    var confirmRemove: (ModelRow) -> RemoveChoice = { _ in .cancel }
    /// Asks before a model's whole weights go: the size that will be freed is the second argument. True = delete.
    var confirmDeleteWeights: (ModelRow, Int64) -> Bool = { _, _ in false }
    /// Weights on disk by model id, read with the rows (`onDiskModels`); empty until the first read.
    @Published private(set) var onDisk: [String: OnDiskModel] = [:]
    /// Each model's size per precision, as the catalog and Arsenal's MCP report it (`CatalogEntry.sizesByDtype`), read
    /// with the rows: the size each line of the Precision column shows.
    @Published private(set) var catalogSizes: [String: [String: Int64]] = [:]
    /// Bytes of weights no model refers to (the "Unlisted weights" row shows when this is above zero).
    @Published private(set) var orphanBytes: Int64 = 0
    /// Deletes in flight: a model's whole weights (its id), one copy (`WeightsPresentation.copyKey`), or the unlisted
    /// weights (`WeightsPresentation.orphanKey`). Their controls are off until each ends; a second click does nothing.
    @Published private(set) var removing: Set<String> = []
    /// Get, Load, Reload and Unload in flight, by model id: the row's controls wait.
    @Published private(set) var working: Set<String> = []
    /// Picks a file for a path setting (an open panel; set by the app); nil when cancelled.
    var pickFile: (ModelRow, SettingState) -> String? = { _, _ in nil }
    /// Asks once before a local model's unreviewed code first runs; true = approved.
    var confirmCode: (ModelRow) -> Bool = { _ in false }
    /// Opens the model browser (the pinned Get more models row).
    var openBrowser: () -> Void = {}
    /// Re-reads model folders, local models and config (the header's Refresh); off while one runs or Arsenal quits.
    var refreshModels: () -> Void = {}
    @Published var refreshEnabled = true
    /// The service's own answer to "would Refresh be refused now?" (a Get or Load registered, including a Load still
    /// verifying files before its row shows Loading), read with every full refresh.
    @Published private(set) var serviceRefusal: String?
    /// Why Refresh is off, or nil: off whenever the service would refuse it, and while any row shows a Get or Load.
    var refreshBlock: String? {
        if !refreshEnabled { return admitting ? "Checking for new models\u{2026}" : "Arsenal is quitting" }
        let busy: Set<ModelState> = [.downloading, .deriving, .loading]
        if serviceRefusal != nil || rows.contains(where: { row in row.precisions.contains { busy.contains(row.states[$0]?.state ?? .notInstalled) } }) {
            return RealModelsService.refreshBusyMessage
        }
        return nil
    }
    /// A refused Refresh's notice that only said "busy": stale once Refresh is allowed again, so the footer drops it.
    var refreshNoticeIsStale: Bool { refreshNotice == RealModelsService.refreshBusyMessage && refreshAllowed }
    var refreshAllowed: Bool { refreshBlock == nil }
    var refreshHelp: String { refreshBlock ?? RefreshControl.help }
    /// Why the last Refresh was refused; the table's footer line shows it until the next Refresh.
    @Published var refreshNotice: String?
    /// True when this model's sample starts (nil cancels only its matching preview).
    var previewVoice: (ModelRow, String?) -> Bool = { _, _ in false }
    /// Asks before a Get that downloads more than the weights (assets or an engine) or a large one
    /// (at least `confirmGetBytes`); true = go ahead. The plan is the service's (`getPlan`).
    var confirmGet: (ModelRow, GetPlan) -> Bool = { _, _ in true }
    /// A plan with a part of unknown size bounds nothing (the service would refuse every Get under it): one plain
    /// error instead of a Get question, and no Get.
    var unknownGetSize: (ModelRow, GetPlan) -> Void = { _, _ in }
    nonisolated static let confirmGetBytes: Int64 = 2_000_000_000
    /// The plan's lines worth naming: what is downloaded, or of unknown size; a bundled part (0 bytes) is not a download.
    nonisolated static func downloads(_ plan: GetPlan) -> [GetPlanItem] { plan.items.filter { $0.downloadBytes != 0 } }
    /// Asked when something besides the weights must be downloaded, or the whole is large (an unknown size never gets
    /// this far: `approve`).
    nonisolated static func asks(_ plan: GetPlan) -> Bool {
        guard plan.requiresDownload else { return false }
        return downloads(plan).contains { $0.kind != "weights" } || (plan.totalDownloadBytes ?? .max) >= confirmGetBytes
    }
    private var events: Task<Void, Never>?
    /// Get more models' Adds still running, by model id: where a failure of each is also reported.
    private var catalogFailures: [String: (String) -> Void] = [:]
    /// False once the app is quitting: rows start no new Get, Load or Delete.
    var admitting = true

    init(service: any ModelsService, chip: String? = HostChip.current) {
        self.service = service; self.chip = chip
        settingsCard.table = self
    }

    /// Reads rows, kinds and the catalog state from the service.
    func refresh() async {
        let sequence = nextRefreshSequence()
        async let r = service.rows()
        async let k = service.kinds()
        async let o = service.persistedKindOrder()
        async let c = service.catalogStatus()
        async let refusal = service.refreshRefusal()
        async let disk = Self.onDiskByID(service)
        async let orphans = (try? await service.orphanBytes()) ?? 0
        let (allRows, kinds, catalog, savedOrder, serviceRefusal, onDisk, orphanBytes) = await (r, k, c, o, refusal, disk, orphans)
        // Runner definitions can change without a catalog-status event (local-folder edits/overrides).
        // Refresh only on the existing request/event path, and drop removed rows' cached details.
        let entries = await service.catalog(query: CatalogQuery())
        // The table holds the models the user chose, and every model with weights on disk (its weights are deleted here).
        let rows = Unlisted.rows(allRows, disk: onDisk, catalog: entries.map(\.row))
        let ids = Set(rows.map(\.id))
        var runners = Dictionary(uniqueKeysWithValues: entries.filter { ids.contains($0.id) }.map { ($0.id, $0.runners) })
        for id in ids where runners[id] == nil { runners[id] = [] }
        let sizes = Dictionary(entries.filter { ids.contains($0.id) }.map { ($0.id, $0.sizesByDtype) }, uniquingKeysWith: { first, _ in first })
        var shown: [String: String] = [:]
        for e in entries where ids.contains(e.id) {
            let id = KindPresentation.id(kind: e.kind, tasks: e.tasks, kinds: kinds)
            if id != e.kind { shown[e.id] = id }
        }
        var settings: [String: [SettingState]] = [:]
        var panels: [String: SettingsPanel] = [:]
        var settingsError: (name: String, raw: String)?
        for row in rows where row.kind != Unlisted.kind {
            panels[row.id] = await service.settingsPanel(id: row.id)
            do { settings[row.id] = try await service.settings(id: row.id) } catch {
                settings[row.id] = []
                settingsError = (row.name, Self.message(error))
            }
            if let runner = Self.runnerSetting(row, runners[row.id] ?? []) { settings[row.id]?.insert(runner, at: 0) }
            // A model whose generic tool its operations place in capability windows names those windows.
            if let tools = Self.toolsSetting(row, declared: declaredOperations(row.id)) { settings[row.id]?.append(tools) }
            settings[row.id] = Self.withKeep(settings[row.id] ?? [], row)
        }
        for (id, keys) in pendingSettings {
            for (key, pending) in keys { if let i = settings[id]?.firstIndex(where: { $0.id == key }) { settings[id]?[i].value = pending.value } }
        }
        guard acceptRefresh(sequence) else { return }
        everyRow = allRows; catalogRows = entries.map(\.row)
        if self.onDisk != onDisk { self.onDisk = onDisk }
        if catalogSizes != sizes { catalogSizes = sizes }
        if self.orphanBytes != orphanBytes { self.orphanBytes = orphanBytes }
        if let settingsError {
            let raw = settingsError.raw
            report(text: "Could not read settings for \(settingsError.name): \(ErrorWording.plain(raw) ?? raw)", raw: raw)
        }
        if self.serviceRefusal != serviceRefusal { self.serviceRefusal = serviceRefusal }
        if self.runners != runners { self.runners = runners }
        if shownKinds != shown { shownKinds = shown }
        if self.settings != settings { self.settings = settings }
        if self.panels != panels { self.panels = panels }
        settingsCard.prune(keeping: Set(rows.map(\.id)))
        apply(rows: rows, kinds: kinds, catalog: catalog, savedOrder: savedOrder)
    }

    /// The on-disk listing by model id (empty when the service cannot list it).
    static func onDiskByID(_ service: any ModelsService) async -> [String: OnDiskModel] {
        Dictionary(((try? await service.onDiskModels()) ?? []).map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
    }

    /// A download's progress changes only its row's states: the rows alone are re-read. A row set that changed (an
    /// Add or Remove raced the download) takes the full refresh instead.
    func refreshRows() async {
        let sequence = nextRefreshSequence()
        let all = await service.rows()
        let rows = Unlisted.rows(all, disk: onDisk, catalog: catalogRows)
        guard sequence > appliedRefreshSequence else { return }
        guard orderLoaded, rows.map(\.id) == self.rows.map(\.id) else { await refresh(); return }
        guard acceptRefresh(sequence) else { return }
        everyRow = all
        apply(rows: rows, kinds: kinds, catalog: catalog)
    }
    private func nextRefreshSequence() -> UInt64 { refreshSequence += 1; return refreshSequence }
    private func acceptRefresh(_ sequence: UInt64) -> Bool {
        guard sequence > appliedRefreshSequence else { return false }
        appliedRefreshSequence = sequence; return true
    }

    /// Tests and renders: runner lists without a catalog read.
    func setRunners(_ value: [String: [CatalogRunner]]) { runners = value }
    static func runnerSetting(_ row: ModelRow, _ runners: [CatalogRunner]) -> SettingState? {
        let eligible = runners.filter(\.eligible)
        guard !eligible.isEmpty else { return nil }
        let fallback = row.defaultRunner ?? runners.first(where: \.acceptedDefault)?.name ?? eligible[0].name
        let declaration: JSONObject = [
            "id": .string(runnerKey), "title": "Runner", "type": "choice", "applies": "load", "default": .string(fallback),
            "choices": .array(eligible.map { .string($0.name) }), "help": "Which runner loads this model; the default is the folder's accepted one.",
            // Every runner named the same way: its name, then what it runs on ("Native (Swift · MLX)").
            "labels": .object(Dictionary(eligible.map { r in (r.name, .string(Self.runnerTitle(r.name, language: r.language))) }, uniquingKeysWith: { a, _ in a }))
        ]
        return SettingState(declaration: declaration, value: .string(row.runner), pending: row.residentRunner.map { $0 != row.runner } ?? false)
    }

    nonisolated static func runnerTitle(_ name: String, language: String) -> String {
        // Arsenal's own runner reads as a word ("Native"); an engine's name keeps its own spelling ("mlx-lm").
        let shown = name == "native" ? "Native" : name
        let engine = ModelDetail.engine(language)
        return engine.isEmpty ? shown : "\(shown) (\(engine))"
    }

    /// "Show in menu": whether a loaded model's tool section (Chat…, Make an Image…, Voice) is in the main menu.
    /// A background-only model (decisions, embeddings, rankings, turn detection) never appears in the main menu, so it has
    /// no switch: the same test as `ToolCenter.backgroundOnly`, over the declared operations the card already has.
    static func toolsSetting(_ row: ModelRow, declared: Set<String>? = nil) -> SettingState? {
        if backgroundOnly(row, declared: declared) { return nil }
        let names = ToolCatalog.resolve(tools: row.tools.compactMap(ToolKind.init(rawValue:)), declared: declared)
        guard !names.isEmpty else { return nil }
        let what = names.map { $0 == .voice ? "its voice controls" : $0.title.replacingOccurrences(of: "\u{2026}", with: "") }.joined(separator: " and ")
        let title = "Show " + names.map(\.noun).joined(separator: " and ") + " in menu"
        let declaration: JSONObject = [
            "id": .string(toolsKey), "title": .string(title), "type": "toggle", "applies": "app", "default": true,
            "help": .string("The main menu shows \(what).")
        ]
        return SettingState(declaration: declaration, value: .bool(row.toolsShown))
    }

    /// Works only behind the scenes: no tool of its own (the generic operation tool aside) and only background operations
    /// declared, or, without a declaration, a background kind.
    static func backgroundOnly(_ row: ModelRow, declared: Set<String>?) -> Bool {
        let own = row.tools.compactMap(ToolKind.init(rawValue:)).filter { $0 != .operation }
        guard own.isEmpty else { return false }
        return ToolCatalog.backgroundOnly(kind: row.kind, declared: declared)
    }

    /// Keep loaded: first, on every row, from the row's own `keepHot`.
    static func keepSetting(_ row: ModelRow) -> SettingState {
        let declaration: JSONObject = [
            "id": .string(keepKey), "title": "Keep loaded", "type": "toggle", "applies": "app", "default": false,
            "help": "Stays in memory until you unload it."
        ]
        return SettingState(declaration: declaration, value: .bool(row.keepHot))
    }
    static func withKeep(_ states: [SettingState], _ row: ModelRow) -> [SettingState] {
        [keepSetting(row)] + states.filter { $0.id != keepKey }
    }

    /// Tests and renders set the state directly.
    func apply(rows: [ModelRow], kinds: [KindSpec], catalog: CatalogStatus? = nil, savedOrder: [String]? = nil) {
        // Only what changed is published: every publish re-evaluates the table, and most refreshes change nothing.
        let voiceInvalidations = rows.filter { row in
            row.residentDType == nil || self.rows.first(where: { $0.id == row.id })?.residentDType != row.residentDType
        }.map(\.id)
        var nextVoices = voices, nextVoiceErrors = voiceErrors
        for id in voiceInvalidations { nextVoices[id] = nil; nextVoiceErrors[id] = nil }
        if voices != nextVoices { voices = nextVoices }
        if voiceErrors != nextVoiceErrors { voiceErrors = nextVoiceErrors }
        // Keep loaded follows the rows (a rows-only refresh, a render without the service's settings), but only when the
        // row's own value changed: a click shown at once is not undone by a progress refresh before the service answers.
        let previous = Dictionary(self.rows.map { ($0.id, $0.keepHot) }, uniquingKeysWith: { a, _ in a })
        var withKeep = settings
        for row in rows where settings[row.id]?.first?.id != Self.keepKey || previous[row.id] != row.keepHot {
            withKeep[row.id] = Self.withKeep(settings[row.id] ?? [], row)
        }
        if withKeep != settings { settings = withKeep }
        if self.rows != rows { self.rows = rows }
        // Seed once; refreshes never prune or normalize the remembered user/confirmed lists.
        if !orderLoaded { userOrder = savedOrder ?? kinds.map(\.id); confirmedOrder = userOrder; orderLoaded = true }
        let orderedKinds = Self.order(kinds: kinds, saved: userOrder)
        if self.kinds != orderedKinds { self.kinds = orderedKinds }
        if self.catalog != catalog { self.catalog = catalog }
    }

    /// Follows the service's events until `stop()`. Events arrive in bursts (a download reports every chunk), and a
    /// full refresh reads the rows, the catalog and every row's settings, so they are coalesced: one refresh runs at a
    /// time and keeps one trailing refresh for state changes received during its reads. Read feedback cannot retry itself. Progress alone re-reads only the rows, at most
    /// `progressInterval` apart. `onChange` follows each full refresh (progress changes nothing else the menu shows).
    func start(onChange: @escaping @MainActor () async -> Void = {}) {
        stop()
        self.onChange = onChange
        events = Task { [weak self] in
            guard let self else { return }
            let stream = await self.service.events()
            self.schedule(.full)
            for await event in stream {
                if Task.isCancelled { break }
                self.receive(event)
            }
        }
    }
    private func receive(_ event: ServiceEvent, feedback: Bool = false) {
        switch event {
        case .readFeedback(let event): receive(event, feedback: true)
        case .progress: schedule(.rows, feedback: feedback)
        case .error(let id, let message):
            // Failures display their reason without requesting a retry, regardless of their origin.
            if reading != nil { errors.append((id, message)) } else if id.map({ id in rows.contains { $0.id == id } }) ?? true { report(message, model: id) }
        case .catalog(let status):
            if catalog != status { catalog = status }
            schedule(.full, feedback: feedback)
        case .rowsChanged, .memory: schedule(.full, feedback: feedback)
        }
    }
    func stop() { events?.cancel(); events = nil; refresher?.cancel(); refresher = nil; wanted = nil; reading = nil; errors = []; generation += 1 }

    /// What the pending refresh must re-read; `.full` covers `.rows`.
    enum Need: Comparable { case rows, full }
    private var wanted: Need?
    private var refresher: Task<Void, Never>?
    private var onChange: @MainActor () async -> Void = {}
    /// Includes awaited menu reads. Only explicitly marked read feedback is covered by this refresh.
    private var reading: Need?
    /// Subscription work includes coalescing buffered read events, even after its views were published.
    var isRefreshing: Bool { refresher != nil }
    /// Setting writes include their follow-up refresh, not just the service's saved value.
    private var settingWrites: [UUID: Task<Void, Never>] = [:]
    /// Await setting writes and scheduled subscription work, including trailing reads and error application.
    /// Call after the action/event being observed has begun; an idle controller returns immediately.
    func waitUntilSettled() async {
        while !settingWrites.isEmpty || refresher != nil {
            for task in settingWrites.values { await task.value }
            if let refresher { await refresher.value }
        }
    }
    private var errors: [(id: String?, message: String)] = []
    /// Bumped by `stop()`: a cancelled refresher ending late must not clear its successor.
    private var generation = 0
    /// A progress bar redrawn ten times a second reads as continuous; faster only costs the service's time.
    private static let eventDeliveryInterval: Duration = .milliseconds(10)
    static let progressInterval: Duration = .milliseconds(100)

    private func schedule(_ need: Need, feedback: Bool = false) {
        guard !feedback || reading.map({ $0 < need }) == true else { return }
        wanted = max(wanted ?? need, need)
        guard refresher == nil else { return }
        let generation = generation
        refresher = Task { [weak self] in
            while let self, generation == self.generation, let need = self.wanted, !Task.isCancelled {
                self.wanted = nil
                self.reading = need
                if need == .full {
                    await self.refresh()
                    await self.onChange()
                    // AsyncStream delivers buffered notifications over several actor turns. Coalesce that delivery
                    // burst too; the refreshed views are already published, so this adds no presentation delay.
                    try? await Task.sleep(for: Self.eventDeliveryInterval)
                    guard generation == self.generation else { return }
                } else {
                    await self.refreshRows()
                    try? await Task.sleep(for: Self.eventDeliveryInterval)
                    guard generation == self.generation else { return }
                }
                let errors = self.errors; self.errors = []
                // Filter after either read kind: newly added rows count, removed rows do not.
                for (id, message) in errors where id.map({ id in self.rows.contains { $0.id == id } }) ?? true { self.report(message, model: id) }
                self.reading = nil
                if need == .rows && self.wanted != .full { try? await Task.sleep(for: Self.progressInterval) }
            }
            if let self, generation == self.generation { self.refresher = nil }
        }
    }

    // MARK: Groups

    /// The groups in display order with their rows (catalog order inside a group, never re-sorted: rows do not move
    /// under the pointer). Kinds without a row are left out; a kind the order does not know yet goes last, by its
    /// stable kind order.
    var groups: [(kind: KindSpec, rows: [ModelRow])] {
        // Where a row shows: its kind, or the kind whose group names what it does (KindPresentation.placed).
        let shownKinds = shownKinds
        func placed(_ r: ModelRow) -> String { KindPresentation.placed(kind: r.kind, shown: shownKinds[r.id]) }
        let listed = rows.filter { $0.kind != Unlisted.kind }
        var visibleKinds = kinds.filter { k in listed.contains { placed($0) == k.id } }
        // An invalid folder on startup has no readable kind; never hide its table row for that reason.
        for id in Set(listed.map(placed)).subtracting(visibleKinds.map(\.id)).sorted() {
            visibleKinds.append(KindSpec(id: id, title: "Local models", order: Int.max, columns: []))
        }
        let ordered = Self.order(kinds: visibleKinds, saved: userOrder)
        // A kind's rows show in its own group and in the groups of what they do (KindPresentation), side by side.
        var result = ordered.flatMap { k in KindPresentation.groups(k, rows: listed.filter { placed($0) == k.id }, shown: shownKinds) }
        // Weights on disk that the catalog does not list: their own group, last, not reorderable (not a kind).
        let unlisted = rows.filter { $0.kind == Unlisted.kind }
        if !unlisted.isEmpty { result.append((KindSpec(id: Unlisted.kind, title: Unlisted.title, order: Int.max, columns: []), unlisted)) }
        return result
    }
    /// What a row's weights show and allow now (its copies on disk, its Delete).
    func weights(_ row: ModelRow) -> WeightsPresentation {
        WeightsPresentation.make(row, disk: onDisk[row.id], sizes: catalogSizes[row.id] ?? [:], working: working, removing: removing)
    }
    /// The row's Precision column: a line per precision on disk.
    func precisionLines(_ row: ModelRow) -> [PrecisionLine] {
        PrecisionLine.make(row, presentation: presentation(row), weights: weights(row))
    }
    /// Tests and renders: the shown groups without a catalog read.
    func setShownKinds(_ value: [String: String]) { shownKinds = value }
    static func order(kinds: [KindSpec], saved: [String]) -> [KindSpec] {
        return kinds.sorted { a, b in
            let ia = saved.firstIndex(of: a.id) ?? Int.max, ib = saved.firstIndex(of: b.id) ?? Int.max
            return ia != ib ? ia < ib : (a.order, a.id) < (b.order, b.id)
        }
    }

    /// Moves a group to a slot (0 = before the first group, count = after the last) and saves the order through the
    /// service (config.json `kindOrder`, which the API and MCP list by too).
    func moveGroup(_ id: String, toSlot slot: Int) {
        let known = Set(kinds.map(\.id))
        guard known.contains(KindPresentation.parent(id)) else { return }
        var ids = groups.map(\.kind.id)
        guard let from = ids.firstIndex(of: id) else { return }
        let target = max(0, min(slot, ids.count))
        ids.remove(at: from)
        ids.insert(id, at: target > from ? target - 1 : target)
        // A shown split saves as its kind, which takes the place of its first group: the kind's groups move together.
        var parents: [String] = []
        for id in ids.map(KindPresentation.parent) where !parents.contains(id) { parents.append(id) }
        // Real kinds without rows follow the visible ones; synthetic groups never enter persistence.
        let hidden = Self.order(kinds: kinds, saved: groupOrder).map(\.id).filter { !parents.contains($0) }
        let order = parents.filter { known.contains($0) } + hidden
        guard order != groupOrder else { return }
        setLocalOrder(order)
        orderWrites.append(order)
        flushOrderWrites()
    }
    private func setLocalOrder(_ order: [String]) {
        let ordered = Self.order(kinds: kinds, saved: order)
        userOrder = order
        if kinds != ordered { kinds = ordered }
    }
    private func flushOrderWrites() {
        guard orderWriter == nil, !orderWrites.isEmpty else { return }
        orderWriter = Task { [weak self] in
            guard let self else { return }
            while !orderWrites.isEmpty {
                let next = orderWrites.removeFirst()
                do {
                    try await service.setKindOrder(next)
                    let stored = await service.persistedKindOrder()
                    guard stored == next else {
                        throw WorkerFailure(code: "kind_order", message: "The stored group order differs from the requested order.")
                    }
                    confirmedOrder = stored
                } catch {
                    let raw = Self.message(error)
                    report(text: "Could not save the group order: \(ErrorWording.plain(raw) ?? raw)", raw: raw)
                    // Keep a newer queued move optimistic; if it also fails, revert to the last
                    // successful write rather than reading an arbitrarily old refresh snapshot.
                    if orderWrites.isEmpty { setLocalOrder(confirmedOrder) }
                }
            }
            orderWriter = nil
            if !orderWrites.isEmpty { flushOrderWrites() }
        }
    }
    func moveGroup(_ id: String, by offset: Int) {
        guard let i = groups.firstIndex(where: { $0.kind.id == id }) else { return }
        let j = i + offset
        guard j >= 0, j < groups.count else { return }
        moveGroup(id, toSlot: offset > 0 ? j + 1 : j)
    }

    // MARK: Rows

    func presentation(_ row: ModelRow) -> RowPresentation {
        // The runner advertises its accepted fast path; there is no execution-path choice in this table.
        RowPresentation(row, previewPrecision: previewPrecision[row.id], previewFast: row.fastAvailable, chip: chip)
    }
    func select(_ row: ModelRow, column: String) {
        guard admitting, let dtype = PrecisionColumn.dtype(row, column), ModelHelp.unavailable(row)[column] == nil, !presentation(row).locked else { return }
        previewPrecision[row.id] = dtype
        precisionWrites.append((row.id, row.name, dtype))
        guard precisionWriter == nil else { return }
        precisionWriter = Task {
            while !precisionWrites.isEmpty {
                let next = precisionWrites.removeFirst()
                do { try await service.setPrecision(id: next.id, precision: next.dtype) } catch { report(Self.message(error), name: next.name) }
                await refresh()
                if !precisionWrites.contains(where: { $0.id == next.id }) { previewPrecision[next.id] = nil }
            }
            precisionWriter = nil
        }
    }
    /// Closing the menu leaves in-flight preference writes intact.
    func discardPreviews() { if drag != nil { drag = nil } }

    /// The row's action at its shown precision: Get, Load, Unload or Reload.
    func perform(_ row: ModelRow) {
        let p = presentation(row)
        // A second click while a Get, Load or delete of this model is in flight does nothing.
        guard p.busyText == nil, admitting, !working.contains(row.id), !removing.contains(row.id) else { return }
        rowErrors[row.id] = nil
        if p.action == .remove { remove(row); return }
        if p.action != .unload, PrecisionColumn.isEightBit(p.selected, formats: row.formats) {
            fail(row, "\(p.selected) is not offered; only 16-bit and int4 are."); return
        }
        // A format no runner here can run (the service's reason): Get and Load refuse with it.
        if p.action != .unload, let column = PrecisionColumn.of(p.selected, formats: row.formats), let reason = ModelHelp.unavailable(row)[column] {
            fail(row, "\(p.selected) \u{2014} \(reason)"); return
        }
        working.insert(row.id)
        guard p.action == .get else { start(row, p); return }
        // Get asks the service what it will download first (weights, assets and included engine). Without
        // that answer nothing starts: a guessed weights-only plan could skip the question for an asset download.
        Task {
            let plan: GetPlan
            do { plan = try await service.getPlan(id: row.id, dtype: p.selected) } catch {
                working.remove(row.id)
                fail(row, "could not check what Get needs, so nothing was downloaded. \(Self.message(error))"); return
            }
            guard approve(row, plan, revised: false) else { working.remove(row.id); return }
            start(row, p, plan: plan)
        }
    }
    /// Get more models' Add: this table's Get at `precision`, with its download plan, consent and code approval. The
    /// model joins the table as its download starts. Returns why nothing started, or nil (started, or the user declined).
    /// `failed`: a failure after the Get started (code consent, a refused format, the download itself), for Get more models,
    /// which shows it beside the model; the table shows it too once the row is there.
    func getFromCatalog(_ row: ModelRow, precision: String, failed: @escaping (String) -> Void = { _ in }) async -> String? {
        guard admitting else { return "Arsenal is quitting, so nothing was downloaded." }
        guard claimWork(row.id) else { return "a Get, Load or delete of it is already under way." }
        catalogFailures[row.id] = failed
        var p = presentation(row); p.selected = precision; p.action = .get
        let plan: GetPlan
        do { plan = try await service.getPlan(id: row.id, dtype: precision) } catch {
            working.remove(row.id)
            return "could not check what Get needs, so nothing was downloaded. \(Self.message(error))"
        }
        guard approve(row, plan, revised: false) else { working.remove(row.id); return nil }
        start(row, p, plan: plan)
        return nil
    }
    /// `plan`: for a Get, the plan the user saw (or that needed no question); the service downloads nothing beyond it.
    /// The caller has marked the row `working`; this clears the mark when the action settles.
    private func start(_ row: ModelRow, _ p: RowPresentation, plan: GetPlan? = nil) {
        lastError = nil; memoryNotice = nil
        // Get, Load and Reload start the folder's code, so each asks; Unload never approves anything. The approval
        // names only the contents this prompt was shown for: if the folder changed since, the service refuses it.
        var confirmed: String?
        if p.action != .unload, row.needsCodeConsent {
            guard let digest = row.contentDigest else { working.remove(row.id); fail(row, "its folder could not be read, so nothing was run."); return }
            guard confirmCode(row) else { working.remove(row.id); return }
            confirmed = digest
        }
        let consent = confirmed, id = row.id, precision = p.selected, fast = p.fast, loaded = p.loadedPrecision
        Task {
            do {
                if p.action == .load || p.action == .reload {
                    memoryNotice = await service.loadMemoryNotice(id: id, precision: precision).map { "\(row.name): \($0)" }
                }
                if let consent { try await service.approveLocalCode(id: id, contentDigest: consent, approvedBy: .app) }
                switch p.action {
                case .get:
                    guard let plan else { throw WorkerFailure(code: "get_plan", message: "Get needs its download plan first.") }
                    try await get(row, precision: precision, plan: plan)
                // Only load-time settings changed: the service reloads with them (same dtype and mode).
                case .reload where precision == loaded && fast == (row.fastAvailable && row.fast):
                    try await MemoryLoadPresenter.perform { force in
                        try await LoadMemoryContext.$force.withValue(force) { try await self.service.applySettings(id: id) }
                    }
                case .load, .reload:
                    try await MemoryLoadPresenter.perform { force in try await self.service.load(id: id, precision: precision, fast: fast, force: force) }
                case .unload: try await service.unload(id: id)
                case .remove: break // Routed through the existing Remove confirmation before starting work.
                }
            } catch is CancellationError {
                // Cancel in the memory warning leaves the row and resident untouched.
            } catch is HFGateError {
                // A gated model: the row says what to do and counts down to the retry; no footer line.
            } catch {
                fail(row, Self.message(error))
            }
            await refresh()
            working.remove(row.id)
        }
    }
    /// A Get bounded by the approved plan. When the service finds it needs more (a local copy failed its hash, an
    /// asset inventory changed) it stops before any network access and returns a revised plan; whenever that plan
    /// downloads anything it is asked about again (the button's size no longer holds), then the Get runs under it.
    private func get(_ row: ModelRow, precision: String, plan: GetPlan) async throws {
        var approved = plan
        for _ in 0..<3 {
            do {
                try await service.get(id: row.id, precision: precision, approvedPlan: approved); return
            } catch let changed as GetPlanChanged {
                approved = changed.revisedPlan
                guard approve(row, approved, revised: true) else { return }
            }
        }
        throw WorkerFailure(code: "get_plan_changed", message: "What Get needs kept changing; nothing more was downloaded. Try again.")
    }
    /// Whether a Get may run under `plan`: an unknown size is an error, never a question; a revised plan is always
    /// asked about afresh; a first plan only when `asks` says so.
    private func approve(_ row: ModelRow, _ plan: GetPlan, revised: Bool) -> Bool {
        guard plan.totalDownloadBytes != nil else { unknownGetSize(row, plan); return false }
        return revised || Self.asks(plan) ? confirmGet(row, plan) : true
    }
    func cancelGet(_ row: ModelRow) {
        Task {
            await service.cancelGet(id: row.id); await refresh()
        }
    }
    /// The row's trash and Remove: with weights on this Mac, its Delete (every weight, after one confirmation); without
    /// them, Add's inverse at once.
    func remove(_ row: ModelRow) {
        guard admitting else { return }
        if hasWeights(row) { deleteWeights(row); return }
        settingsCard.close(row.id)
        Task {
            do { try await service.remove(id: row.id, deleteWeights: false) } catch { report(Self.message(error), name: row.name) }
            await refresh()
        }
    }
    /// Deletes every weight of a model (each precision, revision and cache), then takes it off the table. Asks once,
    /// with the size that will be freed. A loaded model or one busy with a download, conversion or load is refused with
    /// its reason; a second call while this one runs does nothing.
    func deleteWeights(_ row: ModelRow) {
        guard admitting, !removing.contains(row.id), !working.contains(row.id) else { return }
        let current = rows.first { $0.id == row.id } ?? row
        let presentation = weights(current)
        if let reason = presentation.deleteBlocked { fail(current, reason); return }
        guard confirmDeleteWeights(current, presentation.bytes) else { return }
        // Confirmed: the sheet is gone, so the work is marked before anything else can click.
        removing.insert(row.id)
        rowErrors[row.id] = nil
        settingsCard.close(row.id)
        Task {
            do {
                _ = try await service.deleteAllWeights(id: row.id)
                // Off the table too, when the service knows the model (an unlisted one leaves with its weights).
                if everyRow.contains(where: { $0.id == row.id }) { try await service.remove(id: row.id, deleteWeights: false) }
            } catch { fail(current, Self.message(error)) }
            await refresh()
            removing.remove(row.id)
        }
    }
    /// Whether a Delete question still has its subject: the row is on the table with weights (`copy`: that copy is still on
    /// disk). Removed over MCP, or its last weights gone meanwhile, the question closes itself (`StaleQuestion`).
    func deleteQuestionStands(_ row: ModelRow, copy: DeleteConfirm.Copy? = nil) -> Bool {
        guard let current = rows.first(where: { $0.id == row.id }) else { return false }
        guard let copy else { return hasWeights(current) }
        return weights(current).copies.contains { $0.dtype == copy.dtype }
    }
    /// Deletes one precision's copy on this Mac, across every revision, after one question with the size it frees (the
    /// model's Delete question, worded for the copy: `DeleteConfirm.copy`). Refused with its reason while the copy is
    /// loaded or the model is busy; a second click while it runs does nothing.
    func deleteCopy(_ row: ModelRow, _ dtype: String) {
        let key = WeightsPresentation.copyKey(row.id, dtype)
        guard admitting, !removing.contains(key), !removing.contains(row.id) else { return }
        let current = rows.first { $0.id == row.id } ?? row
        guard let copy = weights(current).copies.first(where: { $0.dtype == dtype }) else { return }
        if let reason = copy.blocked { fail(current, reason); return }
        let question = DeleteConfirm.Copy(label: copy.label, downloadable: copyDownloadable(current, dtype), dtype: dtype)
        guard DeleteConfirm.$copy.withValue(question, operation: { confirmDeleteWeights(current, copyBytes(current, dtype)) }) else { return }
        // Confirmed: the alert is gone, so a second click while the delete starts does nothing.
        guard !removing.contains(key), !removing.contains(row.id) else { return }
        rowErrors[row.id] = nil
        removing.insert(key)
        Task {
            do { _ = try await service.deletePrecision(id: row.id, precision: dtype) } catch { fail(current, Self.message(error)) }
            await refresh()
            removing.remove(key)
        }
    }
    /// What deleting one copy frees, as its line shows it (the catalog's size for that precision, else the store's count).
    func copyBytes(_ row: ModelRow, _ dtype: String) -> Int64 {
        guard let copy = onDisk[row.id]?.precisions.first(where: { $0.dtype == dtype }) else { return 0 }
        return WeightsPresentation.size(copy, catalogSizes[row.id] ?? [:])
    }
    /// Get can bring this copy back: the model is still offered and this precision is one of its choices.
    func copyDownloadable(_ row: ModelRow, _ dtype: String) -> Bool {
        guard !row.catalogRemoved, row.kind != Unlisted.kind, let column = PrecisionColumn.of(dtype, formats: row.formats) else { return false }
        return PrecisionColumn.dtype(row, column) == dtype
    }
    /// Deletes the weights no model refers to. One click: they belong to no model, and the row shows their size.
    func deleteOrphans() {
        let key = WeightsPresentation.orphanKey
        guard admitting, orphanBytes > 0, !removing.contains(key) else { return }
        removing.insert(key)
        Task {
            do { _ = try await service.deleteOrphans() } catch { report(Self.message(error), name: "Unlisted weights") }
            await refresh()
            removing.remove(key)
        }
    }
    /// Forgets the owner's approval of a local model's code (the settings card's Revoke); the next Get or Load asks again.
    func revokeCode(_ row: ModelRow) {
        guard admitting else { return }
        rowErrors[row.id] = nil
        Task {
            do { try await service.revokeLocalCode(id: row.id) } catch { fail(row, Self.message(error)) }
            await refresh()
        }
    }
    /// Removes the old precisions on this Mac that the catalog no longer offers (the settings card's Remove). The offered
    /// precisions stay.
    func removeWithdrawnTiers(_ row: ModelRow) {
        guard admitting else { return }
        rowErrors[row.id] = nil
        Task {
            do { try await service.deleteWithdrawnTiers(id: row.id) } catch { fail(row, Self.message(error)) }
            await refresh()
        }
    }
    /// Claims a model for a Get or Load started outside the table (the menu's repair). False while a Get, Load or delete of
    /// it runs, or Arsenal is quitting; the claim is released by `releaseWork` when the action settles.
    func claimWork(_ id: String) -> Bool {
        guard admitting, !working.contains(id), !removing.contains(id) else { return false }
        working.insert(id)
        return true
    }
    func releaseWork(_ id: String) { working.remove(id) }
    /// Weights of some dtype are on this Mac (from the store's listing, or the row's own states before it is read).
    func hasWeights(_ row: ModelRow) -> Bool {
        weights(rows.first { $0.id == row.id } ?? row).hasWeights || row.precisions.contains { [.ready, .loaded].contains(row.states[$0]?.state) }
    }

    // MARK: Settings

    /// The gear: the row's settings card beside it, or closed when it is open. The table does not change and its menu
    /// stays open.
    func openSettings(_ row: ModelRow, from anchor: NSView? = nil) { settingsCard.toggle(row, from: anchor) }
    /// The card is about to show this row's settings.
    func settingsOpened(_ row: ModelRow) {
        // They start without an old failure (one from the table's own buttons stays in the table's footer).
        rowErrors[row.id] = nil
        if settingsRow != row.id { settingsRow = row.id }
    }
    /// A model's settings closed: a voice they were previewing stops (only that model's), and their error line goes.
    func settingsClosed(_ id: String) {
        rowErrors[id] = nil
        if settingsRow == id { settingsRow = nil }
        guard previewing?.model == id, let row = rows.first(where: { $0.id == id }) else { return }
        _ = previewVoice(row, nil); previewing = nil
    }
    /// A model's action failed: the table's footer line names the model; its settings say the rest.
    func fail(_ row: ModelRow, _ message: String) {
        catalogFailures.removeValue(forKey: row.id)?(message)
        report(message, name: row.name)
        rowErrors[row.id] = ErrorWording.short(message)
    }
    /// A failure the service reported for a model (by id) or for Arsenal as a whole (nil): attached to the model's name
    /// and worded plainly, its raw text kept for the footer's hover.
    func report(_ message: String, model id: String?) {
        report(message, name: id.flatMap { id in rows.first { $0.id == id }?.name })
    }
    func report(_ message: String, name: String?) {
        let line = ErrorWording.line(message, model: name)
        report(text: line.text, raw: line.help)
    }
    /// Shows `text` in the footer, `raw` on its hover.
    func report(text: String, raw: String) {
        lastErrorRaw = raw == text ? nil : (text, raw)
        if lastError != text { lastError = text }
    }
    /// Whether a setting can change now: while Arsenal admits work, except that a running Get or Load locks the
    /// load-time settings (the runner included), which the load in flight would not take. Request-time settings and the
    /// app's own rows (Keep loaded, Show in menu) stay live. The settings' controls follow it.
    func settingEnabled(_ row: ModelRow, _ key: String) -> Bool {
        guard admitting else { return false }
        guard let s = settings[row.id]?.first(where: { $0.id == key }), PaneLayout.locks(s) else { return true }
        return !presentation(rows.first { $0.id == row.id } ?? row).locked
    }
    func set(_ row: ModelRow, _ key: String, _ value: JSONValue) {
        // The settings' controls lock the same way; this also holds for a click that lands as a Get or Load starts.
        guard settingEnabled(row, key), settings[row.id]?.first(where: { $0.id == key })?.value != value else { return }
        // Shown at once; the service's answer (and its pending flag) follows.
        rowErrors[row.id] = nil
        if let i = settings[row.id]?.firstIndex(where: { $0.id == key }) { settings[row.id]?[i].value = value }
        let fallback = settings[row.id]?.first { $0.id == key }?.defaultValue
        let write = UUID()
        pendingSettings[row.id, default: [:]][key] = (write, value)
        settingWrites[write] = Task {
            defer { settingWrites[write] = nil }
            do {
                if key == Self.runnerKey {
                    try await service.setRunner(id: row.id, runner: value == fallback ? nil : value.string)
                } else if key == Self.toolsKey {
                    try await service.setToolsShown(id: row.id, on: value.bool ?? true)
                } else if key == Self.keepKey {
                    try await service.setKeepHot(id: row.id, on: value.bool ?? false)
                } else {
                    try await service.setSetting(id: row.id, key: key, value: value)
                }
            } catch {
                // Refused: the refresh shows the service's value again.
                settled(write, row.id, key); fail(row, Self.message(error))
            }
            await refresh()
            settled(write, row.id, key)
        }
    }
    /// The write has landed and its own refresh read it back (or it was refused): refreshes show the service's value.
    private func settled(_ write: UUID, _ id: String, _ key: String) {
        guard pendingSettings[id]?[key]?.write == write else { return }
        pendingSettings[id]?[key] = nil
        if pendingSettings[id]?.isEmpty == true { pendingSettings[id] = nil }
    }
    /// What Restore defaults would change: the card's load settings (the runner included) away from their default that
    /// can change now (`settingEnabled`, so nothing locked by a running Get or Load). Keep loaded and Show in menu are
    /// placement choices, not model defaults, and request settings belong to the apps: it leaves them alone.
    func restorable(_ row: ModelRow) -> [SettingState] {
        (settings[row.id] ?? []).filter { Self.restores($0) && settingEnabled(row, $0.id) }
    }
    /// A setting Restore defaults puts back: a load setting away from its default.
    nonisolated static func restores(_ s: SettingState) -> Bool { s.applies == "load" && s.value != s.defaultValue }
    func restoreDefaults(_ row: ModelRow) {
        let changed = restorable(row)
        guard !changed.isEmpty else { return }
        let values = Dictionary(uniqueKeysWithValues: changed.filter { ![Self.runnerKey, Self.toolsKey].contains($0.id) }.map { ($0.id, $0.defaultValue) })
        let runner = changed.contains { $0.id == Self.runnerKey }, tools = changed.contains { $0.id == Self.toolsKey }
        rowErrors[row.id] = nil
        Task {
            // All or none: a refused default leaves every setting as it was.
            do {
                if !values.isEmpty { try await service.setSettings(id: row.id, values: values) }
                if runner { try await service.setRunner(id: row.id, runner: nil) }
                if tools { try await service.setToolsShown(id: row.id, on: true) }
            } catch { fail(row, Self.message(error)) }
            await refresh()
        }
    }
    func toggleVoicePreview(_ row: ModelRow, _ voice: String) {
        let preview = VoicePreview(model: row.id, voice: voice)
        if previewing == preview { _ = previewVoice(row, nil); previewing = nil } else if previewVoice(row, voice) { previewing = preview }
    }
    /// Whether this model's settings show this voice as playing (Stop).
    func isPreviewing(_ row: ModelRow, _ voice: String) -> Bool { previewing == VoicePreview(model: row.id, voice: voice) }
    func previewEnded() { previewing = nil }
    func voiceHelp(_ id: String) -> String? {
        if loadingVoices.contains(id) { return "Listing this model’s voices…" }
        if let error = voiceErrors[id] { return error }
        if voices[id]?.isEmpty != false { return "This model returned no voices." }
        return nil
    }
    /// A `voices` op result for the row's voice setting (the QA renders pass the worker's reply shape without a load).
    func showVoices(_ id: String, _ result: JSONObject) { voices[id] = VoiceChoice.list(result) }

    /// What Reload would apply on a red row (title + new value; dtype and mode by name), for its tooltip.
    func pendingChanges(_ row: ModelRow) -> [String] {
        let p = presentation(row)
        guard p.pending else { return [] }
        var changes = (settings[row.id] ?? []).filter(\.pending).map { $0.id == Self.runnerKey ? "runner \($0.display($0.value))" : "\($0.title) \($0.display($0.value))" }
        if let resident = row.residentRunner, resident != row.runner, !changes.contains("runner \(row.runner)") { changes.insert("runner \(row.runner)", at: 0) }
        if p.selected != p.loadedPrecision { changes.append(p.selected) }
        if p.fast != (row.fastAvailable && row.fast) { changes.append("automatic execution path") }
        return changes
    }
    /// Load-time settings set away from their defaults: what the next Load uses besides dtype and mode.
    func loadChoices(_ row: ModelRow) -> [String] {
        (settings[row.id] ?? []).filter { $0.applies == "load" && $0.value != $0.defaultValue }.map { "\($0.title) \($0.display($0.value))" }
    }

    nonisolated static func message(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}

/// This Mac's chip as the table names it ("M5 Max"), from the CPU brand string ("Apple M5 Max").
enum HostChip {
    static let current: String? = {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return nil }
        return display(String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self))
    }()
    static func display(_ brand: String?) -> String? {
        guard let brand = brand?.trimmingCharacters(in: .whitespaces), !brand.isEmpty else { return nil }
        return brand.hasPrefix("Apple ") ? String(brand.dropFirst(6)) : brand
    }
}
