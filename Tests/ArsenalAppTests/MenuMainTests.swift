// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

/// The main status menu, checked in each of its states (MenuMainState): narrow enough to read, every row whole, Support
/// directly above Quit on the lighter row, and no Homebrew.
@MainActor @Suite(.menuState, .serialized) final class MenuMainTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-main-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// The main menu's width ceiling (pt), set just above the measured 317 pt: another menu-bar app's is 312 pt, Vella's 374 pt,
    /// and Arsenal's before this change 433 pt.
    static let maxWidth: CGFloat = 320

    private func app(_ state: MenuMainState) async throws -> AppDelegate {
        try await MenuMainState.app(state, support: support.appendingPathComponent(state.rawValue))
    }
    private func visible(_ app: AppDelegate) -> [NSMenuItem] { app.menu.items.filter { !$0.isHidden } }

    @Test(arguments: MenuMainState.allCases)
    func theMenuIsNarrowAndEveryRowFitsItsWidthInFull(state: MenuMainState) async throws {
        let app = try await app(state)
        _ = app.menu.size
        let width = app.menu.size.width
        #expect(width <= Self.maxWidth, "\(state.rawValue): the menu is \(width) pt wide")
        let font = NSFont.menuFont(ofSize: 0)
        for item in visible(app) where !item.isSeparatorItem {
            let needed = ProminentMenuItemView.titleX + (item.title as NSString).size(withAttributes: [.font: font]).width + ProminentMenuItemView.trailing
            #expect(needed <= width, "\(item.title) needs \(needed) pt; the menu is \(width) pt")
            if let view = item.view as? ProminentMenuItemView {
                #expect(view.frame.width >= needed, "\(item.title): its row is \(view.frame.width) pt, its title needs \(needed)")
            }
        }
    }

    /// The menu is by task, not by model: no model's name heads a section or names a top-level row, and the task entries
    /// (Speak…, Make an Image…, Chat…) lead the menu above Models, each opening its window. Fresh has nothing on this Mac,
    /// so no model is on the table and no task has an entry.
    @Test(arguments: MenuMainState.allCases)
    func noModelNameHeadsASectionAndTasksLeadTheMenu(state: MenuMainState) async throws {
        let app = try await app(state)
        let names = Set(app.table.rows.map(\.name))
        for item in visible(app) where !item.isSeparatorItem {
            #expect(!names.contains(item.title), "\(state.rawValue): \(item.title) is a model's name in the main menu")
            if item.isSectionHeader { #expect(!names.contains { item.title.contains($0) }, "\(item.title) heads a section by model") }
        }
        let rows = visible(app)
        let models = try #require(rows.firstIndex { $0.title == "Models" })
        let tasks = rows.prefix(models).filter { ["Speak\u{2026}", "Make an Image\u{2026}", "Chat\u{2026}"].contains($0.title) }
        let expected = state == .fresh ? [] : ["Speak\u{2026}", "Make an Image\u{2026}", "Chat\u{2026}"]
        #expect(tasks.map(\.title) == expected, "\(state.rawValue): \(rows.map(\.title))")
        #expect(tasks.allSatisfy { $0.isEnabled && $0.action != nil })
    }

    /// Core review 9: Copy Last and Support the Developer draw no grey pill at rest. Only the icon and the title are
    /// painted; the row's ground beside them stays the menu's own, and the highlight still fills the row.
    @Test func ownDrawnRowsAreUnlitAtRest() throws {
        let item = NSMenuItem(title: "Copy Last Image", action: nil, keyEquivalent: "")
        let view = ProminentMenuItemView(item: item)
        view.frame = NSRect(x: 0, y: 0, width: 300, height: ProminentMenuItemView.height)
        func alpha(highlighted: Bool) throws -> CGFloat {
            view.forcesHighlight = highlighted; view.rendersSelectionFill = highlighted
            let rep = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
            view.cacheDisplay(in: view.bounds, to: rep)
            // Past the title, inside the row: the ground a resting pill would fill.
            let scale = CGFloat(rep.pixelsWide) / view.bounds.width
            return try #require(rep.colorAt(x: Int((view.bounds.width - 30) * scale), y: rep.pixelsHigh / 2)).alphaComponent
        }
        #expect(try alpha(highlighted: false) == 0)
        #expect(try alpha(highlighted: true) > 0.9)
    }

    /// Support the Developer sits directly above Quit, on its own drawn row, in every state.
    @Test(arguments: MenuMainState.allCases)
    func supportIsDirectlyAboveQuitOnTheLighterRow(state: MenuMainState) async throws {
        let app = try await app(state)
        let rows = visible(app)
        let support = try #require(rows.dropLast().last)
        #expect(rows.last?.title == "Quit Arsenal" && rows.last?.keyEquivalent == "q")
        // During a question (app-modal) AppKit sends a menu action only to a target that works when modal: Quit's does,
        // so the menu and ⌘Q still quit with a "Delete …?" question up (VM run, 2 Nov).
        #expect(rows.last?.target === QuitTarget.shared && rows.last?.action == #selector(QuitTarget.quit(_:)))
        #expect(QuitTarget.shared.worksWhenModal)
        #expect(support.title == AppDelegate.supportTitle && support.action != nil)
        #expect(!support.isSeparatorItem && support.view is ProminentMenuItemView)
        #expect(support.toolTip == AppDelegate.supportHelp)
    }

    /// Homebrew is not a menu item, label or hint in any state, submenus included.
    @Test(arguments: MenuMainState.allCases)
    func nothingInTheMenuMentionsHomebrew(state: MenuMainState) async throws {
        let app = try await app(state)
        func texts(_ menu: NSMenu) -> [String] {
            menu.items.flatMap { item in
                [item.title, item.toolTip ?? "", item.subtitle ?? ""] + (item.submenu.map(texts) ?? [])
            }
        }
        #expect(!texts(app.menu).contains { $0.localizedCaseInsensitiveContains("homebrew") || $0.localizedCaseInsensitiveContains("brew ") })
    }

    /// The image and chat models on the table (each has weights on this Mac) without their selected precision downloaded:
    /// FLUX keeps an old 8-bit copy the catalog no longer offers, Qwen has its int4 while bf16 is chosen.
    static var onTableNotDownloaded: [ModelRow] {
        var rows = Fixtures.rows(.firstLaunch)
        rows[rows.firstIndex { $0.id == "flux2-klein-4b" }!].withdrawnTiers = ["int8": 8_500_000_000]
        rows[rows.firstIndex { $0.id == "qwen3.8-27b" }!].states["int4"] = PrecisionState(state: .ready)
        return rows
    }

    /// A model fixture that offers one tool and is on this Mac (ready), named by the test: never a shipped model's name.
    private func toolModel(_ id: String, _ name: String, kind: String, tool: String) -> ModelRow {
        var row = ModelRow(id: id, name: name, kind: kind, states: ["bf16": .init(state: .ready)])
        row.tools = [tool]
        return row
    }
    private func app(rows: [ModelRow], metadata: [String: JSONObject] = [:], folder: String) async -> AppDelegate {
        let directory = support.appendingPathComponent(folder)
        let app = AppDelegate(service: FixtureService(rows: rows, metadata: metadata, supportDirectory: directory), paths: ArsenalPaths(support: directory))
        app.tools.playback = FixturePlayback()
        app.loginStatus = { .notRegistered }
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        return app
    }

    /// A task no table model serves has no row at all: no "Get a model" placeholder ("I don't want that at all"). Show
    /// Models Folder stays, under Settings.
    @Test func aToolWithNoModelHasNoRow() async throws {
        let app = await app(rows: Self.onTableNotDownloaded, folder: "no-sound")
        let titles = visible(app).map(\.title)
        #expect(!titles.contains { $0.contains("Get a model") })
        #expect(!titles.contains { $0.hasPrefix("Make a Sound") || $0.hasPrefix("Make Music") })
        #expect(app.menu.settings?.items.contains { $0.title == MenuContent.openFiles && !$0.isHidden } == true)
        // The tasks whose models are on the table keep their entries, not downloaded yet: no "not installed — Get" row,
        // the entry opens its window (which offers Download) and carries no note.
        for title in ["Make an Image\u{2026}", "Chat\u{2026}"] {
            let item = try #require(visible(app).first { $0.title == title })
            #expect(item.isEnabled && item.action != nil && item.badge == nil)
        }
        #expect(!titles.contains { $0.contains("not installed") || $0.contains("\u{2014} Get") })
        // No two separators in a row where a placeholder used to sit.
        let rows = visible(app)
        #expect(!zip(rows, rows.dropFirst()).contains { $0.0.isSeparatorItem && $0.1.isSeparatorItem })
    }

    /// A kind that has a model gains its task entry; the model's name heads nothing.
    @Test func aToolWithAModelKeepsItsRow() async throws {
        let sound = toolModel("sound-fixture", "Sound Fixture", kind: "sound", tool: "sound")
        let app = await app(rows: Fixtures.rows(.firstLaunch) + [sound], metadata: [sound.id: ["tools": [.string("sound")]]], folder: "with-sound")
        let rows = visible(app)
        #expect(!rows.contains { $0.title.hasSuffix("Get a model") })
        #expect(!rows.contains { $0.title == "Sound Fixture" })
        let entry = try #require(rows.first { $0.title == "Make a Sound\u{2026}" })
        #expect(entry.isEnabled && entry.action != nil && entry.badge == nil && entry.submenu == nil)
        let models = try #require(rows.firstIndex { $0.title == "Models" })
        #expect(try #require(rows.firstIndex { $0 === entry }) < models)
    }

    /// A model hidden from the menu (Show in menu off) is neither shown nor replaced by a placeholder row.
    @Test func aHiddenModelToolIsNotReplacedByAPlaceholder() async throws {
        let sound = toolModel("sound-fixture", "Sound Fixture", kind: "sound", tool: "sound")
        let app = await app(rows: Fixtures.rows(.firstLaunch) + [sound], metadata: [sound.id: ["tools": [.string("sound")]]], folder: "hidden-sound")
        app.table.set(sound, TableController.toolsKey, .bool(false))
        try await waitUntil { app.table.rows.first { $0.id == sound.id }?.toolsShown == false }
        app.rebuildMenu()
        #expect(!app.menu.items.contains { $0.title.hasSuffix("Get a model") && !$0.isHidden })
        #expect(app.menu.items.first { $0.title == "Make a Sound\u{2026}" }?.isHidden == true)
    }

    /// Models that only serve agents and other models (decisions, embeddings, rankings, turn detection) reach no part of
    /// the main menu, Show in menu on or not: no task entry, no folder action, no agent voice, not even their name.
    @Test func backgroundOnlyModelsNeverReachTheMainMenu() async throws {
        var decide = toolModel("decide-fixture", "Decide Fixture", kind: "decisions", tool: "operation")
        var embed = toolModel("embed-fixture", "Embed Fixture", kind: "embeddings", tool: "operation")
        decide.states = ["bf16": .init(state: .loaded)]; embed.states = ["bf16": .init(state: .loaded)]
        let sound = toolModel("sound-fixture", "Sound Fixture", kind: "sound", tool: "sound")
        let rows = [decide, embed, sound]
        func op(_ name: String) -> ModelOperation { ModelOperation(name: name, title: name.capitalized, inputSchema: ["type": "object"], output: "json", stream: false) }
        let directory = support.appendingPathComponent("background")
        let service = FixtureService(
            rows: rows, metadata: Dictionary(uniqueKeysWithValues: rows.map { ($0.id, ["tools": .array($0.tools.map { .string($0) })]) }),
            supportDirectory: directory, operations: [decide.id: [op("decide")], embed.id: [op("embed"), op("count_tokens")]])
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: directory))
        app.tools.playback = FixturePlayback()
        app.loginStatus = { .notRegistered }
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        defer { app.table.stop(); app.tools.closeAll() }
        // Everything but Memory, which lists every loaded model's memory use (resources, not something to use).
        let memory = try #require(app.menu.settings?.items.first { $0.title == "Memory" }?.submenu)
        let reachable = app.menu.everyItem.filter { item in !memory.items.contains { $0 === item } }
        for id in [decide.id, embed.id] {
            let row = try #require(app.table.rows.first { $0.id == id })
            #expect(row.toolsShown && app.tools.backgroundOnly(row) && app.tools.tasks(row).isEmpty, "\(id)")
            #expect(!app.tools.taskEntries(app.table.menuRows, includeHidden: true).contains { $0.all.contains { $0.id == id } })
            #expect(!reachable.contains { $0.title.contains(row.name) || ($0.representedObject as? [String])?.contains(id) == true }, "\(id)")
            #expect(!reachable.contains { $0.title.hasPrefix("Run") })
        }
        // The model a person uses is there.
        #expect(app.menu.items.contains { $0.title == "Make a Sound\u{2026}" && !$0.isHidden })
    }

    /// A model not on this Mac: Make an Image… opens its window like any other entry, and the window's model choice carries
    /// the Download problem, titled with the download's size.
    @Test func aNotInstalledModelsEntryOpensItsWindowWithDownload() async throws {
        let app = await app(rows: Self.onTableNotDownloaded, folder: "not-downloaded")
        defer { app.table.stop(); app.tools.closeAll() }
        app.tools.present = { _ in }
        let flux = try #require(app.table.rows.first { $0.id == "flux2-klein-4b" })
        #expect(flux.state == .notInstalled)
        let entry = try #require(visible(app).first { $0.title == "Make an Image\u{2026}" })
        #expect(entry.isEnabled && entry.badge == nil)
        app.menu.performActionForItem(at: app.menu.index(of: entry))
        let open = try #require(app.tools.image)
        #expect(open.session.selected == flux.id)
        let choice = try #require(open.session.models.first { $0.id == flux.id })
        let bytes = try #require(flux.downloadBytes[flux.selectedPrecision] ?? flux.sizeBytes)
        #expect(choice.unavailable?.action == .download && choice.unavailable?.actionTitle == "Download " + ToolCenter.shortSize(bytes))
        #expect(choice.unavailable?.message == "\(flux.name) isn\u{2019}t on this Mac yet.")
    }

    /// A downloading model's entry shows its percent as a trailing note, which moves in place while the menu is open
    /// (the same item, no structure change) and goes once the model is on this Mac.
    @Test func aDownloadingModelsEntryShowsItsPercentAndUpdatesInPlace() async throws {
        var image = toolModel("image-fixture", "Image Fixture", kind: "images", tool: "image")
        image.states = ["bf16": .init(state: .downloading, progress: 0.42)]
        let directory = support.appendingPathComponent("downloading")
        let service = FixtureService(rows: [image], metadata: [image.id: ["tools": [.string("image")]]], supportDirectory: directory)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: directory))
        app.tools.playback = FixturePlayback()
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        defer { app.table.stop(); app.tools.closeAll() }
        let key = AppDelegate.taskKey(.tool(.image))
        let entry = try #require(app.menu.items.first { ($0.representedObject as? String) == key })
        #expect(entry.title == "Make an Image\u{2026}" && entry.isEnabled && entry.badge?.stringValue == "42 %")
        app.menuWillOpen(app.menu)
        for (progress, note) in [(0.43, "43 %"), (0.9, "90 %")] {
            image.states = ["bf16": .init(state: .downloading, progress: progress)]
            app.table.apply(rows: [image], kinds: app.table.kinds)
            app.rebuildMenu()
            #expect(app.menu.items.first { ($0.representedObject as? String) == key } === entry)
            #expect(entry.badge?.stringValue == note && !app.structurePending)
        }
        // The download finishes: the service has the model, and the menu's next refresh drops the note.
        try await service.get(id: image.id, precision: "bf16")
        await app.table.refresh(); await app.refreshMenuState()
        #expect(app.table.rows.first?.state == .ready)
        #expect(entry.badge == nil && entry.isEnabled && !app.structurePending, "\(entry.badge?.stringValue ?? "none")")
        app.menuDidClose(app.menu)
    }

    /// The smoke's menu check is strict: the fresh menu passes, and a missing row or Support anywhere but directly above
    /// Quit fails it; in Settings, Launch at Login may show its approval title, and a wrong login row or a missing one fails.
    @Test func theSmokeMenuCheckPassesTheFreshMenuAndRefusesAWrongLoginOrSupport() async throws {
        let app = await app(rows: [], folder: "smoke-check")
        // As the smoke runs: nothing speaks (Stop Speaking stays hidden), and the status item exists.
        (app.tools.playback as? FixturePlayback)?.playing = false
        let status = AppDelegate.makeStatusItem(visible: false)
        app.status = status
        defer { NSStatusBar.system.removeStatusItem(status) }
        app.rebuildMenu()
        let titles = app.menu.items.filter { !$0.isSeparatorItem && !$0.isHidden }.map(\.title)
        #expect(SmokeDelegate.menuMatches(titles), "\(titles)")
        let settings = try #require(app.menu.settings).items.filter { !$0.isSeparatorItem && !$0.isHidden }.map(\.title)
        #expect(SmokeDelegate.settingsMatch(settings), "\(settings)")
        let approval = settings.map { $0 == AppDelegate.loginTitle ? AppDelegate.loginApprovalTitle : $0 }
        #expect(SmokeDelegate.settingsMatch(approval), "the approval title is accepted")
        let login = try #require(settings.firstIndex(of: AppDelegate.loginTitle))
        var renamed = settings; renamed[login] = "Start at Login"
        #expect(!SmokeDelegate.settingsMatch(renamed), "a renamed login row fails")
        var missing = settings; missing.remove(at: login)
        #expect(!SmokeDelegate.settingsMatch(missing), "a missing login row fails")
        #expect(!SmokeDelegate.menuMatches(titles + [AppDelegate.loginTitle]), "a login row back in the main menu fails")
        let support = try #require(titles.firstIndex(of: AppDelegate.supportTitle))
        let quit = try #require(titles.firstIndex(of: "Quit Arsenal"))
        var swapped = titles; swapped.swapAt(support, quit)
        #expect(!SmokeDelegate.menuMatches(swapped), "Quit above Support fails")
        var aboveAbout = titles; aboveAbout.swapAt(support, support - 1)
        #expect(!SmokeDelegate.menuMatches(aboveAbout), "Support above About fails")
        var removed = titles; removed.remove(at: support)
        #expect(!SmokeDelegate.menuMatches(removed), "a missing Support row fails")
        // The whole check, as `Arsenal --smoke` runs it, passes the fresh menu.
        #expect(SmokeDelegate.check(app).ok, "\(SmokeDelegate.check(app).text)")
    }
}
