// SPDX-License-Identifier: MIT
import AppKit
import Combine
import ArsenalCore
import ServiceManagement
import SwiftUI

/// The menu-bar app: model access and Activity, loaded-model tools, memory controls, files, and app actions (About, Launch at Login,
/// Quit). The service does the work; this file only presents it.
@MainActor final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let service: any ModelsService
    let paths: ArsenalPaths
    let table: TableController
    let modelsMenu: ModelsMenu
    /// The Activity item below Models: what is running, in its own submenu.
    let activityMenu: ActivityMenu
    let runner: ActionRunner
    var status: NSStatusItem?
    let menu: NSMenu
    let menuDiagnostics: MenuDiagnostics
    private(set) var actions: [LoadedAction] = []
    /// Loaded models' menu sections (a folder's `[[menu.section]]`, present only while it is loaded).
    private(set) var sections: [LoadedMenuSection] = []
    private(set) var memory = MemorySnapshot()
    /// Every row the service holds, on the table or not (the table fetched it): a model an agent runs off the table is listed and unloaded.
    var serviceRows: [ModelRow] { table.everyRow }
    private(set) var residency = ResidencySettings()
    private(set) var memorySettings = MemorySettings()
    /// Model tools (chat, image, voice) and their windows.
    private(set) var tools: ToolCenter!
    /// The updater, when this build has one configured (unset: no item, no network).
    var updater: AppUpdating? { didSet { updater?.onChange = { [weak self] in self?.updateMenu() }; updateMenu() } }
    private var menuGeneration = 0
    private var rowsObserver: AnyCancellable?
    let api: APIHost
    /// Launch at Login (SMAppService): on from a fresh install's first launch, then the user's choice (applyLaunchAtLoginDefault);
    /// tests replace these.
    /// Launch at Login as the system reports it. `SMAppService.status` is a synchronous round trip to the system's login-item
    /// service, 60–150 ms while models run (measured, 1.0.1), so the menu never asks it while it is built: `loginState` is read
    /// once, after each change, and again off the main thread whenever the status menu opens. Tests replace this reader.
    var loginStatus: () -> SMAppService.Status = AppDelegate.systemLoginStatus { didSet { loginReaderReplaced = true; forgetLoginState() } }
    nonisolated static let systemLoginStatus: @Sendable () -> SMAppService.Status = {
        Bundle.main.bundleIdentifier == nil ? .notRegistered : SMAppService.mainApp.status
    }
    private var loginReaderReplaced = false
    private var loginStateCache: SMAppService.Status?
    /// Bumped by every change of the state (a toggle, the first-run default, a new reader): a background reading begun
    /// before it is stale and is dropped.
    private var loginGeneration = 0
    private func forgetLoginState() { loginStateCache = nil; loginGeneration += 1 }
    /// The status the menu shows: the last reading, read now only when there is none.
    var loginState: SMAppService.Status {
        if let loginStateCache { return loginStateCache }
        let status = loginStatus(); loginStateCache = status; return status
    }
    /// Reads the status again away from the main thread and updates the menu in place if it changed (Login Items can be
    /// changed in System Settings while Arsenal runs). A replaced reader (tests) is read on the main thread at the next use.
    func refreshLoginStateInBackground(reader: (@Sendable () -> SMAppService.Status)? = nil) {
        guard reader != nil || !loginReaderReplaced else { forgetLoginState(); return }
        let read = reader ?? AppDelegate.systemLoginStatus, generation = loginGeneration
        loginRead = Task.detached(priority: .utility) { [weak self] in
            let status = read()
            await MainActor.run {
                guard let self, generation == self.loginGeneration, self.loginStateCache != status else { return }
                self.loginStateCache = status
                self.rebuildMenu()
            }
        }
    }
    /// The last background reading (tests wait for it).
    private(set) var loginRead: Task<Void, Never>?
    var registerLogin: () throws -> Void = { try SMAppService.mainApp.register() }
    var unregisterLogin: () throws -> Void = { try SMAppService.mainApp.unregister() }
    var openLoginSettings: () -> Void = { SMAppService.openSystemSettingsLoginItems() }
    var openExternalURL: (URL) -> Bool = { NSWorkspace.shared.open($0) }

    init(service: any ModelsService, paths: ArsenalPaths, geometryWriter: (any MenuGeometryWriting)? = nil, menu: NSMenu = NSMenu()) {
        self.service = service; self.paths = paths; self.menu = menu
        menuDiagnostics = MenuDiagnostics(support: paths.support, writer: geometryWriter)
        table = TableController(service: service)
        modelsMenu = ModelsMenu(controller: table)
        activityMenu = ActivityMenu(controller: table)
        runner = ActionRunner(service: service, jobs: paths.directory("jobs"))
        api = APIHost(service: service, support: paths.support)
        super.init()
        MemoryLoadPresenter.dismissMenus = { [weak self] in
            self?.modelsMenu.dismiss(.memoryWarning)
            self?.menu.cancelTrackingFrom(.memoryWarning)
        }
        MemoryLoadPresenter.unloadModels = { [service] ids in await service.unloadIdle(ids: ids) }
        modelsMenu.diagnostics = menuDiagnostics
        activityMenu.diagnostics = menuDiagnostics
        let client: @MainActor @Sendable () -> ArsenalClient? = { [weak api] in api?.port.map { ArsenalClient(port: $0) } }
        tools = ToolCenter(client: client, playback: QueuePlayback())
        table.declaredOperations = { [weak self] id in self?.tools?.declared[id]?.names }
        // Settings refresh rows directly as well as through service events. Observe the actual rows, so a
        // Show in menu change reaches the menu even when no service event is emitted by a fixture/client.
        rowsObserver = table.$rows
            .map { rows in Dictionary(uniqueKeysWithValues: rows.map { ($0.id, $0.toolsShown) }) }
            .removeDuplicates().dropFirst()
            .sink { [weak self] _ in
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.updateMenu()
                }
            }
        tools.onChange = { [weak self] in self?.updateMenu() }
        tools.playback.onChange = { [weak self] in self?.updateMenu() }
        tools.kinds = { [weak self] in self?.table.kinds ?? [] }
        tools.inbox = paths.support.appendingPathComponent("inbox", isDirectory: true)
        tools.openModels = { [weak self] in self?.status?.button?.performClick(nil) }
        // A window's Download, Try Again and Approve… run the Models table's own Get and approval (its size
        // confirmation and code approval); an update still goes through Models.
        tools.prepareModel = { [weak self] row in
            guard let self else { return }
            // The same reading as the window's (ToolCenter.readiness): a resident model needs no download.
            switch row.residentDType == nil ? row.state : .loaded {
            case .notInstalled, .failed: self.modelsMenu.prepareTool(row, action: "Get")
            case .updateAvailable: self.tools.openModels()
            default: if row.needsCodeConsent { self.modelsMenu.prepareTool(row, action: "Load") }
            }
        }
        activityMenu.activity.source = ServiceActivitySource(service: service)
        runner.onChange = { [weak self] in
            if self?.runner.running == nil { self?.table.previewEnded() }
            self?.updateMenu()
        }
        modelsMenu.showBrowser = { [weak self] in self?.showBrowser() }
        modelsMenu.refreshModels = { [weak self] in self?.refreshFiles() }
        table.previewVoice = { [weak self] row, voice in
            guard let self else { return false }
            guard let voice else { return self.runner.cancel(actionID: "\(row.id)-preview") }
            return self.runner.run(
                LoadedAction(
                    id: "\(row.id)-preview", modelID: row.id, title: "Preview", op: "speak", input: "none",
                    params: ["text": .string(Self.previewSentence), "voice": .string(voice)]))
        }
    }

    /// The voice preview's fixed sample (four words).
    static let previewSentence = "This is my voice."
    /// The model browser: made on first open, kept for the session (its filters survive closing it). Releasing it on
    /// close was measured: no footprint returned, and each rebuild grew it.
    private(set) var browser: BrowserWindowController?
    private var browserState: BrowserController? { browser?.controller }
    func showBrowser() {
        if browser == nil {
            let controller = BrowserController(service: service)
            controller.openTool = { [weak self] tool, row in self?.tools.open(tool, row) }
            controller.confirmRemove = { [weak self] row in
                guard let self else { return .cancel }
                switch self.modelsMenu.presentAlert(ModelsMenu.removeAlert(row)) {
                case .alertSecondButtonReturn: return .deleteWeights
                default: return .cancel
                }
            }
            controller.getModel = { [weak self, weak controller] row, precision in
                await self?.modelsMenu.controller.getFromCatalog(row, precision: precision) { message in
                    controller?.lastError = "\(row.name): \(message)"
                }
            }
            controller.removeModel = { [weak self] row in self?.modelsMenu.controller.remove(row) }
            browser = BrowserWindowController(controller: controller)
        }
        browser?.show()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.appearance = nil
        Appearance.shared.followSystem()
        if let screen = NSScreen.main { ModelTable.maxHeight = max(400, screen.visibleFrame.height - 60) }
        let item = status ?? Self.makeStatusItem()
        menu.delegate = self; menu.autoenablesItems = false
        item.menu = menu
        status = item
        applyLaunchAtLoginDefault()
        rebuildMenu()
        table.start { [weak self] in
            await self?.refreshMenuState()
        }
        if let real = service as? RealModelsService {
            Task {
                do { try await real.watchLocalFolders() } catch { table.lastError = "Local models folder: \(error.localizedDescription)" }
            }
        }
        Task {
            await api.start()
            if let error = api.error { table.lastError = error }
        }
        // The updater (C8): only when this build names a release repository. After an update's relaunch it reloads
        // what was loaded, once the service (and so the saved settings) is open, which it is here.
        if updater == nil, let core = CoreUpdater.make(service: service, support: paths.support) {
            updater = core
            Task { await core.restoreAfterRelaunch() }
        }
    }

    static func makeStatusItem(
        visible: Bool = ProcessInfo.processInfo.environment["ARSENAL_HEADLESS"] != "1" && !CommandLine.arguments.contains("--smoke")
    ) -> NSStatusItem {
        // Start hidden: hiding a normal-width item after creation can flash on screen during a smoke.
        let item = NSStatusBar.system.statusItem(withLength: visible ? NSStatusItem.variableLength : 0)
        item.isVisible = visible
        item.button?.image = statusImage()
        return item
    }

    /// The cannon glyph (Resources/Brand/glyph.svg, rendered into the bundle as MenuBarTemplate.png and @2x by
    /// scripts/assemble.sh), tinted by the menu bar. A bare dev binary has no bundle resources and shows a stand-in.
    static func statusImage(bundle: Bundle = .main) -> NSImage? {
        let image = bundle.image(forResource: "MenuBarTemplate") ?? NSImage(systemSymbolName: "scope", accessibilityDescription: nil)
        image?.isTemplate = true
        image?.accessibilityDescription = "Arsenal"
        return image
    }

    /// Quit (menu, ⌘Q, logout, SIGTERM): the app stays until its supervised work has stopped, then exits.
    private(set) var quitting = false
    /// Tests replace this; the app tells AppKit the deferred termination may proceed.
    var replyToTerminate: @MainActor (Bool) -> Void = { NSApp.reply(toApplicationShouldTerminate: $0) }
    /// Generous for a worker that ignores EOF and SIGTERM (core escalates to SIGKILL within seconds), short enough that
    /// a stuck shutdown never keeps the app from quitting.
    static let quitLimit: Duration = .seconds(10)

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if quitting { return .terminateCancel } // the first Quit is still shutting down; it replies
        quitting = true; table.admitting = false
        Task {
            _ = await shutDown(); replyToTerminate(true)
        }
        return .terminateLater
    }

    /// Stops admitting work (menu actions, table clicks, the API), stops audio, then waits up to `limit` for the service
    /// to cancel its Gets and setup and stop every runner. True when the service finished in time.
    func shutDown(limit: Duration = quitLimit) async -> Bool {
        quitting = true
        table.stop(); table.admitting = false
        browser?.close()
        tools.closeAll()
        (updater as? CoreUpdater)?.stop()
        api.stop()
        runner.stop()
        updateMenu()
        // Process exit alone does not reach a worker blocked in native code or one that ignores stdin EOF.
        let service = self.service
        return await Bounded.wait(limit) { await service.shutdown() }
    }

    func applicationWillTerminate(_ notification: Notification) {
        api.stop()
        menuDiagnostics.drainForTermination()
    }

    /// Re-reads what the menu shows besides the table (actions of loaded models, memory).
    func refreshMenuState() async {
        // Explicit actions can overlap a subscription refresh: only the newest answers are applied.
        menuGeneration += 1
        let generation = menuGeneration
        async let a = service.menuActions()
        async let s = service.menuSections()
        async let m = service.memory()
        async let r = service.residencySettings()
        async let full = service.memorySettings()
        async let admission = tools.requestAdmissions(rows: table.menuRows, service: service)
        let (actions, sections, memory, residency, memorySettings, admissions) = await (a, s, m, r, full, admission)
        guard generation == menuGeneration else { return }
        self.actions = actions; self.sections = sections; self.memory = memory; self.residency = residency; self.memorySettings = memorySettings
        tools.applyAdmissions(admissions)
        tools.update(rows: table.menuRows)
        activityMenu.activity.memory = memory.models.mapValues(\.footprintMB)
        modelsMenu.resize()
        updateMenu()
        await tools.refreshVoices(table.menuRows, service: service)
    }

    /// Render harness and tests: what the menu shows besides the table, without a running service.
    func previewState(actions: [LoadedAction], memory: MemorySnapshot, sections: [LoadedMenuSection]? = nil) {
        self.actions = actions; self.memory = memory
        self.sections =
            sections ?? Dictionary(grouping: actions, by: \.modelID).keys.sorted().map { id in .init(id: id, modelID: id, title: id, items: actions.filter { $0.modelID == id }) }
    }

    func menuWillOpen(_ menu: NSMenu) {
        if menu === self.menu {
            menuDiagnostics.observeMainMenu(menu)
            menuDiagnostics.menuWillOpen(menu)
            rebuildMenu()
            menuOpen = true
            refreshLoginStateInBackground()
            menuDiagnostics.recordAfterOpen(menu: menu, name: .main)
        }
    }
    func menuDidClose(_ menu: NSMenu) {
        guard menu === self.menu else { return }
        menuDiagnostics.menuDidClose(menu)
        menuOpen = false
        // Changes have already been reconciled in place; what waited for the menus to close (an item added or removed)
        // follows one turn later: never restructure a native viewer while AppKit is removing its item views from the
        // closing menu window.
        guard structurePending else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self, !self.menuOpen, self.structurePending else { return }
            self.rebuildMenu()
        }
    }
    /// The status menu is open (and so any of its submenus can be): its structure waits until it closes.
    private(set) var menuOpen = false
    /// An item to add or remove is waiting for every Arsenal menu to close.
    private(set) var structurePending = false
    private func updateMenu() {
        status?.button?.toolTip = MenuContent.summary(table.menuRows)
        rebuildMenu()
    }

    func rebuildMenu() {
        // Build a desired snapshot off-menu. Reconciliation retains native item/submenu identity,
        // including during tracking; the live menu never passes through an empty state.
        let menu = NSMenu()
        if let updater, let line = MenuContent.update(updater.updateState) {
            let item = NSMenuItem(title: line.title, action: #selector(updateClicked), keyEquivalent: "")
            item.target = self; item.isEnabled = line.enabled && !quitting; item.toolTip = line.help
            item.image = Symbol.menu("arrow.down.circle.fill")
            item.showMenuImage()
            if line.orange {
                item.attributedTitle = NSAttributedString(string: line.title, attributes: [.foregroundColor: NSColor.systemOrange, .font: NSFont.menuFont(ofSize: 0)])
                item.image = item.image?.withSymbolConfiguration(.init(paletteColors: [.systemOrange]))
                item.showMenuImage()
            }
            menu.addItem(item)
            menu.addItem(.separator())
        }
        // Tasks first, by what a person does; the model is a choice inside each window.
        addTaskEntries(to: menu)
        // The newest result one click away (Vella's Copy Last Transcript); Recent holds the last
        // five with Open, Save As… and Show in Finder, even when the newest one's file has gone.
        if let lastOutput, lastOutput.title != nil || !lastOutput.recent.isEmpty { menu.addItem(.separator()) }
        if let lastOutput, let title = lastOutput.title {
            let copy = item(title, "doc.on.doc", #selector(copyLastOutput), in: menu)
            copy.toolTip = lastOutput.help
            _ = ProminentMenuItemView(item: copy)
        }
        if let recent = recentItem() { menu.addItem(recent) }
        menu.addItem(.separator())
        let models = NSMenuItem(title: "Models", action: nil, keyEquivalent: "")
        let activity = NSMenuItem(title: "Activity", action: nil, keyEquivalent: "")
        menu.addItem(models)
        menu.addItem(activity)
        menu.addItem(settingsItem())
        modelsMenu.updateRefresh(enabled: !quitting && !refreshing, notice: refreshNotice)
        menu.addItem(.separator())
        item("About Arsenal", "info.circle", #selector(about), in: menu)
        // Above Quit, to the author's support page.
        let support = item(Self.supportTitle, "heart.fill", #selector(supportDeveloper), in: menu)
        support.toolTip = Self.supportHelp
        ProminentMenuItemView(item: support).accent = Self.supportAccent
        let quit = item("Quit Arsenal", "power", #selector(QuitTarget.quit(_:)), key: "q", in: menu)
        quit.target = QuitTarget.shared; quit.toolTip = Self.quitHelp
        Self.tidySeparators(menu)
        let desired = menu.items.map { item in
            item === models ? modelsMenu.modelItem() : item === activity ? activityMenu.item() : item
        }
        // Release only the temporary snapshot's items, never the live menu's attached roots.
        menu.removeAllItems()
        // While a menu is open nothing is added or removed (AppKit closes an open submenu when its menu changes):
        // survivors update in place, and the rest follows once every Arsenal menu has closed.
        structurePending = !MenuReconciliation.apply(desired, to: self.menu, structure: !menuOpen)
        menuDiagnostics.observeMainMenu(self.menu)
        menuDiagnostics.observeSubmenus(in: self.menu)
    }

    private func addTaskEntries(to menu: NSMenu) {
        // Every task any table model serves is built; Show in menu only hides or shows it, so the switch changes no
        // structure and the menu updates while it is open.
        let entries = tools.taskEntries(table.menuRows, includeHidden: true)
        for entry in entries {
            let item = NSMenuItem(title: entry.title, action: #selector(openTask(_:)), keyEquivalent: "")
            item.target = self; item.isEnabled = !quitting; item.toolTip = entry.help; item.isHidden = !entry.shown
            item.representedObject = Self.taskKey(entry.task)
            item.image = Symbol.menu(entry.symbol)
            item.showMenuImage()
            // A model that cannot run yet still opens its window; the menu only says where it stands.
            if let note = entry.note { item.badge = NSMenuItemBadge(string: note) }
            if Self.perModelWindow(entry.task), entry.all.count > 1 {
                // A model's own operations open one window per model: the entry chooses the model.
                item.action = nil
                let sub = NSMenu(); sub.autoenablesItems = false
                for row in entry.all {
                    let choice = NSMenuItem(title: row.name, action: #selector(openTask(_:)), keyEquivalent: "")
                    choice.target = self; choice.isEnabled = !quitting; choice.isHidden = entry.shown && !row.toolsShown
                    choice.representedObject = [Self.taskKey(entry.task), row.id]
                    if let note = tools.readiness(row, task: entry.task)?.note { choice.badge = NSMenuItemBadge(string: note) }
                    sub.addItem(choice)
                }
                item.submenu = sub
            }
            menu.addItem(item)
        }
        // A folder's own actions (Speak Clipboard) follow the tasks: a declared section under its title, plain actions
        // with the model's name only where two models offer the same one. Background-only models offer none here.
        let shown = table.menuRows.filter { !tools.backgroundOnly($0) }
        let everyone = shown.map { row -> ModelRow in
            var r = row
            r.toolsShown = true
            return r
        }
        let groups = MenuContent.sections(sections, rows: everyone)
        let visibleIDs = Set(MenuContent.sections(sections, rows: shown).flatMap(\.entries).map(\.action.id))
        let plainTitles = Dictionary(grouping: groups.filter { $0.header == nil }.flatMap(\.entries).filter { visibleIDs.contains($0.action.id) }, by: \.title)
        var anyShown = entries.contains(where: \.shown)
        for group in groups {
            let groupShown = group.entries.contains { visibleIDs.contains($0.action.id) }
            // Always built, hidden where it would lead the menu or close an empty group: whether a task shows must not
            // add or remove it, or Show in menu would change structure while the menu is open.
            let separator = NSMenuItem.separator()
            separator.isHidden = !groupShown || !anyShown && group.header == nil
            menu.addItem(separator)
            if let header = group.header {
                let item = NSMenuItem.sectionHeader(title: header); item.isHidden = !groupShown
                menu.addItem(item)
            }
            for entry in group.entries {
                let named = group.header == nil && (plainTitles[entry.title]?.count ?? 0) > 1
                addAction(entry.action, title: entry.title, subtitle: named ? entry.subtitle : nil, in: menu)
                menu.items.last?.isHidden = !visibleIDs.contains(entry.action.id)
            }
            anyShown = anyShown || groupShown
        }
        // Stop Speaking only while something speaks.
        let playback = tools.playback
        let stop = NSMenuItem(title: "Stop Speaking", action: #selector(stopSpeaking), keyEquivalent: "")
        stop.target = self; stop.isEnabled = !quitting
        stop.image = Symbol.menu("stop.circle")
        stop.showMenuImage()
        if let now = playback.current, playback.playing {
            let model = table.rows.first { $0.id == now.model }?.name ?? now.model
            let voiceName = now.voice.map { v in tools.voices[now.model]?.first { $0.id == v }?.name ?? Self.voiceDisplayName(v) }
            stop.subtitle = ([now.requester] + [model, voiceName].compactMap { $0 }).joined(separator: " \u{00b7} ")
        }
        stop.isHidden = !playback.playing
        // Its own group: never glued to the last task (tidySeparators hides the line when nothing is above it).
        menu.addItem(.separator())
        menu.addItem(stop)
        // A menu action's failure, in red where the actions are; a success needs no line.
        // Built for every model in the menu and hidden with its Show in menu, so the switch changes no structure.
        for (id, error) in tools.errors.sorted(by: { $0.key < $1.key }) where shown.contains(where: { $0.id == id }) || id == "playback" {
            let line = Self.errorLine(error)
            line.isHidden = id != "playback" && !shown.contains { $0.id == id && $0.toolsShown }
            menu.addItem(line)
        }
        if let failure = runner.failure { menu.addItem(Self.errorLine(failure)) }
    }

    /// A separator shows only between two visible groups: none leads the menu (a fresh menu whose tasks are all hidden
    /// starts at Models), none follows another. Hiding, never removing, keeps the structure while the menu is open.
    static func tidySeparators(_ menu: NSMenu) {
        var seenItem = false, pending: NSMenuItem?
        for item in menu.items {
            if item.isSeparatorItem {
                item.isHidden = true
                if seenItem, pending == nil { pending = item }
            } else if !item.isHidden {
                pending?.isHidden = false
                pending = nil; seenItem = true
            }
        }
    }

    /// The generic window belongs to one model; the rest (Chat included) pick their model inside.
    static func perModelWindow(_ task: MenuTask) -> Bool {
        switch task {
        case .run: return true
        default: return false
        }
    }
    /// A voice id read as a name when its model has not listed it: Kokoro's "af_heart" is Heart.
    static func voiceDisplayName(_ id: String) -> String {
        let parts = id.split(separator: "_", maxSplits: 1)
        let name = parts.count == 2 && parts[0].count == 2 ? String(parts[1]) : id
        return name.split(separator: "_").map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined(separator: " ")
    }
    static func taskKey(_ task: MenuTask) -> String {
        switch task {
        case .sound(let s): return "sound:" + s.rawValue
        case .tool(let t): return "tool:" + t.rawValue
        case .run(let title): return "run:" + title
        }
    }
    @objc private func openTask(_ sender: NSMenuItem) {
        let key: String, model: String?
        if let pair = sender.representedObject as? [String], pair.count == 2 {
            key = pair[0]; model = pair[1]
        } else if let k = sender.representedObject as? String {
            key = k; model = nil
        } else {
            return
        }
        let entries = tools.taskEntries(table.menuRows, includeHidden: true)
        // The entry may have changed while the menu was open (a model's declared operations arrived and moved it from
        // Make a Sound to Make Music, or renamed its own Run entry): open the same kind of task as it is now.
        let kind = key.split(separator: ":").first.map(String.init) ?? key
        guard
            let entry = entries.first(where: { Self.taskKey($0.task) == key })
                ?? entries.first(where: { $0.shown && Self.taskKey($0.task).hasPrefix(kind + ":") && kind != "tool" })
        else { return }
        let row = model.flatMap { id in entry.all.first { $0.id == id } } ?? entry.lead
        tools.openTask(entry.task, row)
    }

    /// Settings: how models stay loaded and share memory, the voice agents speak with, and the app's own switches.
    private func settingsItem() -> NSMenuItem {
        let root = NSMenuItem(title: "Settings", action: nil, keyEquivalent: "")
        root.identifier = MenuDiagnosticName.settings.itemIdentifier
        root.image = Symbol.menu("gearshape")
        root.showMenuImage()
        let sub = NSMenu(); sub.autoenablesItems = false
        sub.addItem(keepHotItem())
        sub.addItem(memoryItem())
        // Every loaded voice model's root is built and hidden by its Show in menu, so the switch changes no structure.
        let voiceRows = table.menuRows.filter { $0.tools.contains(ToolKind.voice.rawValue) && $0.residentDType != nil && !tools.backgroundOnly($0) }
        let shownVoices = voiceRows.filter(\.toolsShown).count
        sub.addItem(.separator())
        for row in voiceRows {
            addVoice(row, named: shownVoices > 1, in: sub)
            sub.items.last?.isHidden = !row.toolsShown
        }
        let playback = tools.playback
        let mute = NSMenuItem(title: "Mute Agent Speech", action: #selector(toggleMute), keyEquivalent: "")
        mute.target = self; mute.isEnabled = !quitting; mute.state = playback.muted ? .on : .off
        mute.image = Symbol.menu("speaker.slash")
        mute.showMenuImage()
        mute.toolTip = "Agents and Speak Clipboard stay silent; their audio files are still made."
        sub.addItem(mute)
        sub.addItem(.separator())
        // The token for gated Hugging Face models; Vella's two items stay last.
        item(HuggingFaceToken.title, "key", #selector(editHuggingFaceToken), in: sub).toolTip = HuggingFaceToken.help
        let loginState = self.loginState
        let login = item(
            loginState == .requiresApproval ? Self.loginApprovalTitle : Self.loginTitle, "power.circle", #selector(toggleLogin), in: sub)
        login.state = loginState == .enabled ? .on : .off
        login.toolTip = loginState == .requiresApproval ? Self.loginApprovalHelp : Self.loginHelp
        item(MenuContent.openFiles, "folder", #selector(openFiles), in: sub).toolTip = MenuContent.openFilesHelp
        root.submenu = sub
        return root
    }

    /// The last results of every tool window, newest first: each opens, saves elsewhere, copies or shows in Finder.
    private func recentItem() -> NSMenuItem? {
        guard let lastOutput, !lastOutput.recent.isEmpty else { return nil }
        let root = NSMenuItem(title: "Recent", action: nil, keyEquivalent: "")
        root.image = Symbol.menu("clock.arrow.circlepath")
        root.showMenuImage()
        let sub = NSMenu(); sub.autoenablesItems = false
        for (index, entry) in lastOutput.recent.enumerated() {
            let item = NSMenuItem(title: LastOutput.title(entry), action: nil, keyEquivalent: "")
            item.subtitle = LastOutput.subtitle(entry)
            item.image = Symbol.menu(LastOutput.symbol(entry))
            item.showMenuImage()
            let actions = NSMenu(); actions.autoenablesItems = false
            func add(_ title: String, _ selector: Selector) {
                // The result itself, not its row number: a file deleted while the menu is open shifts the list.
                let a = NSMenuItem(title: title, action: selector, keyEquivalent: ""); a.target = self; a.tag = index; a.isEnabled = !quitting
                a.representedObject = entry.made
                actions.addItem(a)
            }
            if case .file = entry.content { add("Open", #selector(openRecent(_:))) }
            add("Copy", #selector(copyRecent(_:)))
            add("Save As\u{2026}", #selector(saveRecent(_:)))
            if case .file = entry.content { add("Show in Finder", #selector(revealRecent(_:))) }
            item.submenu = actions
            sub.addItem(item)
        }
        root.submenu = sub
        return root
    }
    @objc private func copyLastOutput() { lastOutput?.copy() }
    /// The Recent row's result where it is now; nil when it has gone.
    private func recentIndex(_ sender: NSMenuItem) -> Int? {
        guard let made = sender.representedObject as? Date else { return sender.tag }
        return lastOutput?.recent.firstIndex { $0.made == made }
    }
    @objc private func openRecent(_ sender: NSMenuItem) { if let i = recentIndex(sender) { lastOutput?.open(i) } }
    @objc private func copyRecent(_ sender: NSMenuItem) { if let i = recentIndex(sender) { lastOutput?.copy(i) } }
    @objc private func revealRecent(_ sender: NSMenuItem) { if let i = recentIndex(sender) { lastOutput?.reveal(i) } }
    @objc private func saveRecent(_ sender: NSMenuItem) {
        guard let index = recentIndex(sender) else { return }
        DispatchQueue.main.async { self.lastOutput?.saveAs(index) }
    }

    private func addAction(_ action: LoadedAction, title: String, subtitle: String?, in menu: NSMenu) {
        let item = NSMenuItem(title: title, action: #selector(runAction(_:)), keyEquivalent: "")
        item.target = self; item.subtitle = subtitle; item.representedObject = action.id
        item.image = Symbol.menu(Self.symbol(action.op))
        item.showMenuImage()
        item.isEnabled = runner.running == nil && !quitting
        menu.addItem(item)
    }

    static func unloadedToolHelp(_ row: ModelRow) -> String { "Load \(row.name) in Models to use this tool." }

    /// A tool's failure, in its place: a red line under its section until the next try.
    static func errorLine(_ text: String) -> NSMenuItem {
        let item = NSMenuItem(title: text, action: nil, keyEquivalent: "")
        item.attributedTitle = NSAttributedString(
            string: text, attributes: [.foregroundColor: NSColor.systemRed, .font: NSFont.menuFont(ofSize: NSFont.smallSystemFontSize)])
        item.isEnabled = false; item.toolTip = text
        return item
    }

    /// Each model owns its Voice root. Shared playback controls follow all model sections.
    private func addVoice(_ row: ModelRow, named: Bool, in menu: NSMenu) {
        let presets = tools.voices[row.id] ?? []
        let current = tools.currentVoice[row.id]
        let selection = presets.first { $0.id == current }?.name ?? current ?? (presets.isEmpty ? "No voices yet" : "Choose a voice")
        // The voice agents and Speak Clipboard speak with; the Speak window picks its own per take.
        let voice = NSMenuItem(title: (named ? row.name + " Voice: " : "Agent Voice: ") + selection, action: nil, keyEquivalent: "")
        voice.identifier = MenuDiagnosticName.voice.itemIdentifier
        voice.representedObject = [ToolKind.voice.rawValue, row.id]
        voice.image = Symbol.menu("person.wave.2")
        voice.showMenuImage()
        voice.isEnabled = !quitting && row.residentDType != nil
        voice.toolTip = "The voice agents and Speak Clipboard use with \(row.name). The Speak window chooses its own."
        let sub = NSMenu(); sub.autoenablesItems = false
        // Grouped by language under its name (American English, British English, …), each voice by its name.
        var language: String?, voiceGroup: Character?
        func heading(_ choice: VoiceChoice) -> String {
            let group = choice.group == "f" ? "Female" : choice.group == "m" ? "Male" : nil
            return [choice.languageName.isEmpty ? nil : choice.languageName, group].compactMap { $0 }.joined(separator: " \u{00b7} ")
        }
        for choice in presets {
            if choice.languageName != language || choice.group != voiceGroup {
                // Each language's female and male voices under their own heading (Kokoro's).
                if language != nil, heading(choice).isEmpty { sub.addItem(.separator()) }
                language = choice.languageName; voiceGroup = choice.group
                if !heading(choice).isEmpty { sub.addItem(.sectionHeader(title: heading(choice))) }
            }
            let item = NSMenuItem(title: choice.name, action: #selector(chooseVoice(_:)), keyEquivalent: "")
            item.target = self; item.representedObject = [row.id, choice.id]; item.state = choice.id == current ? .on : .off; item.isEnabled = !quitting
            sub.addItem(item)
        }
        if presets.isEmpty {
            let none = NSMenuItem(title: "No voices yet", action: nil, keyEquivalent: ""); none.isEnabled = false
            sub.addItem(none)
        }
        voice.submenu = sub
        menu.addItem(voice)
    }

    @objc private func openTool(_ sender: NSMenuItem) {
        guard let pair = sender.representedObject as? [String], pair.count == 2, let tool = ToolKind(rawValue: pair[0]),
            let row = table.rows.first(where: { $0.id == pair[1] })
        else { return }
        tools.open(tool, row)
    }
    @objc private func chooseVoice(_ sender: NSMenuItem) {
        guard let pair = sender.representedObject as? [String], pair.count == 2 else { return }
        tools.chooseVoice(pair[0], pair[1])
    }
    @objc private func toggleMute() { tools.toggleMute() }
    @objc private func stopSpeaking() { tools.stopSpeaking() }
    @objc private func updateClicked() {
        guard let updater else { return }
        if case .failed = updater.updateState { updater.checkAgain() } else { updater.installUpdate() }
    }

    static func symbol(_ op: String) -> String {
        switch op {
        case "speak": return "waveform"
        case "generate": return "photo"
        case "chat": return "text.bubble"
        case "decide": return "checkmark.circle"
        default: return "play.circle"
        }
    }

    @discardableResult private func item(_ title: String, _ icon: String, _ action: Selector, key: String = "", in menu: NSMenu) -> NSMenuItem {
        let entry = NSMenuItem(title: title, action: action, keyEquivalent: key)
        entry.target = self
        entry.image = Symbol.menu(icon)
        entry.showMenuImage()
        menu.addItem(entry)
        return entry
    }

    private func keepHotItem() -> NSMenuItem {
        let root = NSMenuItem(title: MenuContent.keepLoadedTitle, action: nil, keyEquivalent: "")
        root.identifier = MenuDiagnosticName.keepHot.itemIdentifier
        root.image = Symbol.menu("flame")
        root.showMenuImage()
        let sub = NSMenu(); sub.autoenablesItems = false
        for (index, group) in [
            (MenuContent.manualHeader, MenuContent.manualHelp, residency.manualIdleMinutes, true),
            (MenuContent.onDemandHeader, MenuContent.onDemandHelp, residency.onDemandIdleMinutes, false)
        ].enumerated() {
            let (title, help, current, manual) = group
            if index > 0 { sub.addItem(.separator()) }
            let header = NSMenuItem.sectionHeader(title: title); header.toolTip = help
            sub.addItem(header)
            for choice in MenuContent.idleChoices(current) {
                let item = NSMenuItem(title: choice.title, action: #selector(chooseIdle(_:)), keyEquivalent: "")
                item.target = self; item.state = choice.checked ? .on : .off; item.tag = choice.minutes
                item.representedObject = manual
                if choice.minutes == 0 { item.toolTip = MenuContent.alwaysHelp }
                sub.addItem(item)
            }
        }
        root.submenu = sub
        return root
    }

    private func memoryItem() -> NSMenuItem {
        let root = NSMenuItem(title: "Memory", action: nil, keyEquivalent: "")
        root.identifier = MenuDiagnosticName.memory.itemIdentifier
        root.image = Symbol.menu("memorychip")
        root.showMenuImage()
        let sub = NSMenu(); sub.autoenablesItems = false
        for (title, help, swap) in [(MenuContent.fitTitle, MenuContent.fitHelp, false), (MenuContent.swapTitle, MenuContent.swapHelp, true)] {
            let item = NSMenuItem(title: title, action: #selector(chooseSwap(_:)), keyEquivalent: "")
            item.target = self; item.state = residency.allowSwap == swap ? .on : .off; item.toolTip = help; item.representedObject = swap
            sub.addItem(item)
        }
        sub.addItem(.separator())
        sub.addItem(.sectionHeader(title: MenuContent.whenFullHeader))
        for (mode, title, help) in MenuContent.whenFull {
            let item = NSMenuItem(title: title, action: #selector(chooseWhenFull(_:)), keyEquivalent: "")
            item.target = self; item.state = memorySettings.whenFull == mode ? .on : .off; item.toolTip = help; item.representedObject = mode.rawValue
            sub.addItem(item)
        }
        sub.addItem(.separator())
        // Free memory and each loaded model with what it uses: information, readable and not clickable.
        for line in MenuContent.memoryLines(memory, rows: serviceRows) { sub.addItem(InfoMenuItem.make(line)) }
        let loaded = !MenuContent.loadedIDs(serviceRows).isEmpty
        sub.addItem(.separator())
        let unload = NSMenuItem(title: "Unload All", action: #selector(unloadAll), keyEquivalent: "")
        unload.target = self; unload.isEnabled = loaded
        unload.toolTip = "Stops every model's runner and frees its memory; the models stay on this Mac."
        sub.addItem(unload)
        root.submenu = sub
        return root
    }

    @objc private func runAction(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String, let action = (actions + sections.flatMap(\.items)).first(where: { $0.id == id }) else { return }
        runner.run(action)
    }
    @objc private func chooseIdle(_ sender: NSMenuItem) {
        var settings = residency
        if sender.representedObject as? Bool == true { settings.manualIdleMinutes = sender.tag } else { settings.onDemandIdleMinutes = sender.tag }
        apply(settings)
    }
    @objc private func chooseSwap(_ sender: NSMenuItem) {
        var settings = residency
        settings.allowSwap = sender.representedObject as? Bool == true
        apply(settings)
    }
    @objc private func chooseWhenFull(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let mode = MemoryWhenFull(rawValue: raw) else { return }
        let previous = memorySettings
        memorySettings.whenFull = mode
        Task {
            do { try await service.setMemorySettings(MemorySettings(whenFull: mode)) } catch {
                memorySettings = previous; table.lastError = "Could not save the memory setting: \(TableController.message(error))"
            }
            await refreshMenuState()
        }
    }
    private func apply(_ settings: ResidencySettings) {
        Task {
            do { try await service.setResidencySettings(settings) } catch { table.lastError = TableController.message(error) }
            await refreshMenuState()
        }
    }
    @objc private func unloadAll() {
        let ids = MenuContent.loadedIDs(serviceRows)
        Task {
            for id in ids { do { try await service.unload(id: id) } catch { table.lastError = TableController.message(error) } }
            await table.refresh(); await refreshMenuState()
        }
    }
    /// A refused Refresh's reason (a Get or Load is running), under the item until the next Refresh.
    private(set) var refreshNotice: String?
    private var refreshing = false
    /// Re-reads model folders, local models and config without restarting (the service refuses during a Get or Load).
    @objc func refreshFiles() {
        refreshNotice = nil; refreshing = true
        modelsMenu.updateRefresh(enabled: false, notice: nil)
        Task {
            do { try await service.refresh() } catch { refreshNotice = TableController.message(error) }
            refreshing = false
            modelsMenu.updateRefresh(enabled: !quitting, notice: refreshNotice)
            await table.refresh()
            if browser?.window?.isVisible == true { await browserState?.refresh() }
            await refreshMenuState()
        }
    }

    /// Finder at the support folder, `local-models` selected: adding a model is dropping a folder there.
    @objc func openFiles() {
        let local = paths.directory("local-models")
        try? FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([local])
    }
    /// The token for gated Hugging Face models: an alert after the menu closes (it opens a window, so the menu may close).
    @objc func editHuggingFaceToken() {
        let service = self.service
        DispatchQueue.main.async { _ = HuggingFaceToken.edit(service: service) }
    }
    @objc private func openBrowser() { showBrowser() }
    @objc private func about() {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.orderFrontStandardAboutPanel(options: [.applicationName: "Arsenal"])
    }
    static let loginTitle = "Launch at Login"
    static let loginApprovalTitle = "Launch at Login — Approval Required\u{2026}"
    static let loginHelp = "Opens Arsenal when you log in, so agents and the menu find it running."
    static let loginApprovalHelp = "Allow Arsenal in System Settings → General → Login Items & Extensions."
    /// Debug builds (developer runs and test hosts), `--smoke`, and any run with its own support folder (a lab or smoke copy)
    /// never register a login item: a registration from those runs points at a build or copy that does not stay.
    static func launchDefaultApplies(
        debugBuild: Bool = isDebugBuild, arguments: [String] = CommandLine.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        !debugBuild && !arguments.contains("--smoke") && environment["ARSENAL_SUPPORT_DIR"] == nil
    }
    static var isDebugBuild: Bool {
        #if DEBUG
            true
        #else
            false
        #endif
    }
    /// Launch at Login is on from a fresh install's first launch.
    /// A fresh install is marked as owed the default before its service writes anything (LaunchDelegate), so a launch cut
    /// short still owes it. An install that already had data keeps its login state exactly as it is: its first launch only
    /// records that. A registration that cannot be made stays owed and is tried again next launch. Once recorded, Arsenal
    /// never changes it on its own, so a user who turns it off stays off.
    func applyLaunchAtLoginDefault(applies: Bool = launchDefaultApplies()) {
        guard applies, !loginChoiceRecorded else { return }
        defer { forgetLoginState() }
        guard loginDefaultOwed else {
            recordLoginChoice()
            return
        }
        switch loginStatus() {
        // A never-registered app reports .notFound on macOS 27 until its first registration, which still succeeds.
        case .notRegistered, .notFound:
            do { try registerLogin() } catch { Self.markLoginDefaultOwed(support: paths.support); return }
        case .enabled, .requiresApproval:
            break
        default:
            Self.markLoginDefaultOwed(support: paths.support)
            return
        }
        recordLoginChoice()
    }
    /// Owes a fresh install the Launch at Login default until a launch records it. LaunchDelegate calls this before the service
    /// writes anything into the support folder, and only when the default applies (a release app with its own folder).
    static func markLoginDefaultOwed(support: URL) {
        try? FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: support.appendingPathComponent(loginOwedFileName).path, contents: nil)
    }
    /// A fresh install has nothing in its support folder but the instance lock and Finder's `.DS_Store`; a missing folder is
    /// fresh too. Checked before anything writes there (the launch delegate asks first).
    static func isFreshInstall(support: URL) -> Bool {
        let ignored: Set<String> = [".instance.lock", ".DS_Store"]
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: support.path)) ?? []
        return entries.allSatisfy(ignored.contains)
    }
    private static let loginOwedFileName = "launch-at-login.pending"
    /// The record of the login choice: a file in this support folder (tests give each run its own folder).
    var loginChoiceRecorded: Bool { FileManager.default.fileExists(atPath: loginChoiceFile.path) }
    private var loginChoiceFile: URL { paths.support.appendingPathComponent("launch-at-login.recorded") }
    /// The default is still owed: a fresh install whose launch has not yet recorded it.
    private var loginDefaultOwed: Bool { FileManager.default.fileExists(atPath: paths.support.appendingPathComponent(Self.loginOwedFileName).path) }
    private func recordLoginChoice() {
        try? FileManager.default.createDirectory(at: paths.support, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: paths.support.appendingPathComponent(Self.loginOwedFileName))
        FileManager.default.createFile(atPath: loginChoiceFile.path, contents: nil)
    }
    /// Off turns on, on turns off; waiting for approval opens Login Items, where the user allows it. Any change by the user
    /// is a choice the first-run default never overrides.
    func changeLogin() throws {
        recordLoginChoice()
        defer { forgetLoginState() }
        switch loginStatus() {
        case .requiresApproval: openLoginSettings()
        case .enabled: try unregisterLogin()
        default: try registerLogin()
        }
    }
    static func loginFailureAlert(_ error: Error) -> NSAlert {
        let alert = NSAlert(); alert.messageText = "Launch at Login not changed"
        let reason = error.localizedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        alert.informativeText = (reason.hasSuffix(".") ? reason : reason + ".") + " See Login Items & Extensions in System Settings."
        alert.addButton(withTitle: "OK"); return alert
    }
    @objc private func toggleLogin() {
        do { try changeLogin() } catch {
            NSApp.activate(ignoringOtherApps: true)
            Self.loginFailureAlert(error).runModal()
        }
        rebuildMenu()
    }
    /// The newest tool-window result for Copy Last (the app passes the shared one; tests none unless they test it).
    var lastOutput: LastOutput? {
        didSet {
            tools.lastOutput = lastOutput
            lastOutput?.changed = { [weak self] in self?.updateMenu() }
            updateMenu()
        }
    }
    static let supportTitle = "Support the Developer\u{2026}"
    static let supportURL = URL(string: "https://tobynoskillson.com/#support")!
    /// Support the Developer's warm accent: the system's pink, so it follows the system's light and dark variants.
    static let supportAccent = NSColor.systemPink
    static let supportHelp = "Opens tobynoskillson.com: GitHub Sponsors and Buy Me a Coffee."
    static let quitHelp = "Stops running work and every model, and frees their memory."
    @objc private func supportDeveloper() {
        DispatchQueue.main.async {
            guard !self.openExternalURL(Self.supportURL) else { return }
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.messageText = "Could not open the support page"
            alert.informativeText = Self.supportURL.absoluteString
            alert.runModal()
        }
    }
}

/// A line of information in a menu (free memory, a model's memory): drawn as text, never highlighted or clicked.
@MainActor enum InfoMenuItem {
    static func make(_ text: String) -> NSMenuItem {
        let item = NSMenuItem(title: text, action: nil, keyEquivalent: "")
        let label = NSTextField(labelWithString: text)
        label.font = .menuFont(ofSize: 0); label.textColor = .secondaryLabelColor
        label.translatesAutoresizingMaskIntoConstraints = false
        let view = NSView(frame: NSRect(x: 0, y: 0, width: max(200, label.intrinsicContentSize.width + 34), height: 22))
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 22), label.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
        item.view = view
        item.isEnabled = false
        return item
    }
}
