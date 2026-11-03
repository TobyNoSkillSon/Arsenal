// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import ObjectiveC
import Testing
@testable import Arsenal

/// A model's settings as a card beside its gear inside the Models table. Each failure seen
/// in the 13 Oct menu log is an invariant here, run on every build: a gear click acts in place; Show in menu changes no
/// open menu's structure; the card never moves or grows the table; no request setting reaches the card; nothing in the
/// Models menu pops up a menu. The real Models menu, its hosted table and gear, in AppKit's own item viewer in unordered
/// windows far off every screen. Nothing is ordered on screen, activated, focused or typed outside this process.
@MainActor @Suite(.menuState, .serialized) struct SettingsCardTests {
    private func support() -> URL { URL(fileURLWithPath: "/private/tmp/arsenal-r13-gear-card/tests/\(UUID())") }

    /// Every model folder in the catalog.
    static var catalogModels: [String] {
        let root = SettingsRenderDelegate.repo.appendingPathComponent("models")
        return ((try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []).filter {
            FileManager.default.fileExists(atPath: root.appendingPathComponent("\($0)/model.toml").path)
        }.sorted()
    }

    /// A model from its own folder (settings.toml included), as the user's table has it.
    /// Every catalog model, in tables of at most six: a user's table holds the models they added (A user has 5–6), and at
    /// Vella's row height the whole catalog in one table scrolls, which would hide the lower gears from a click.
    static var catalogTables: [[String]] {
        let ids = catalogModels
        return stride(from: 0, to: ids.count, by: 6).map { Array(ids[$0..<min($0 + 6, ids.count)]) }
    }

    static func folderRow(_ id: String, state: ModelState = .ready) throws -> (ModelRow, JSONObject) {
        let folder = SettingsRenderDelegate.repo.appendingPathComponent("models/\(id)")
        let source = try SettingsRenderDelegate.folder(folder)
        // The stub resolves tools from metadata, just as the real service does from the model manifest.
        // Supplying only row.tools loses sound and multi-tool declarations to its legacy kind defaults.
        var metadata: JSONObject = [
            "tools": .array(source.tools.map { .string($0) }),
            "setting": .array(source.settings.map { .object(SettingsRenderDelegate.declarationJSON($0)) })
        ]
        if let toml = try? String(contentsOf: folder.appendingPathComponent(SettingsPanel.fileName), encoding: .utf8) { metadata["settings_toml"] = .string(toml) }
        var row = ModelRow(id: id, name: source.name, kind: source.kind, states: ["bf16": .init(state: state)])
        row.tools = source.tools
        if state == .loaded { row.residentRunner = row.runner }
        return (row, metadata)
    }

    /// An app on the fixture service with these models, its menus opened as AppKit's tracking opens them (the Arsenal
    /// menu, then Models), its table in AppKit's item viewer.
    @MainActor struct Open {
        let app: AppDelegate
        let host: SettingsCardHost
        var tableMenu: NSMenu { host.menu }
        func close() {
            host.close()
            app.table.stop(); app.tools.closeAll()
        }
        /// Models, then the Arsenal menu, close as AppKit closes them.
        func closeMenus() {
            tableMenu.delegate?.menuDidClose?(tableMenu)
            app.menuDidClose(app.menu)
        }
    }
    static func open(_ ids: [String], support: URL, state: ModelState = .ready) async throws -> Open {
        var rows: [ModelRow] = [], metadata: [String: JSONObject] = [:]
        for id in ids {
            let (row, meta) = try folderRow(id, state: state)
            rows.append(row); metadata[id] = meta
        }
        let service = FixtureService(rows: rows, metadata: metadata, supportDirectory: support)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        #expect(app.table.rows.map(\.tools) == rows.map(\.tools), "the service keeps every folder's declared tools")
        app.menuDiagnostics.trackingRecorder.currentEvent = { nil }
        let host = SettingsCardHost(app.table, models: app.modelsMenu)
        app.menuWillOpen(app.menu); host.menu.delegate?.menuWillOpen?(host.menu)
        await host.settle()
        return Open(app: app, host: host)
    }

    /// Counts `cancelTracking` on any menu while `body` runs.
    static func countingCancellations(_ body: () async throws -> Void) async throws -> Int {
        let method = try #require(class_getInstanceMethod(NSMenu.self, #selector(NSMenu.cancelTracking)))
        var count = 0
        let replacement = imp_implementationWithBlock({ (_: NSMenu) in count += 1 } as @convention(block) (NSMenu) -> Void)
        let original = method_setImplementation(method, replacement)
        defer { method_setImplementation(method, original); imp_removeBlock(replacement) }
        try await body()
        return count
    }
    /// Counts the notifications AppKit posts for an item's action on any menu, and for items added to or removed from
    /// `root`'s menus (a snapshot menu the app builds off-menu and empties does not count: it is never shown).
    @MainActor final class MenuCalls {
        private(set) var actions = 0, structural = 0
        private var tokens: [NSObjectProtocol] = []
        init(root: NSMenu? = nil) {
            let center = NotificationCenter.default
            let rootID = root.map(ObjectIdentifier.init)
            tokens.append(center.addObserver(forName: NSMenu.willSendActionNotification, object: nil, queue: nil) { _ in MainActor.assumeIsolated { self.actions += 1 } })
            for name in [NSMenu.didAddItemNotification, NSMenu.didRemoveItemNotification] {
                tokens.append(
                    center.addObserver(forName: name, object: nil, queue: nil) { note in
                        nonisolated(unsafe) let menu = note.object as? NSMenu
                        MainActor.assumeIsolated {
                            var top = menu
                            while let parent = top?.supermenu { top = parent }
                            if rootID == nil || top.map(ObjectIdentifier.init) == rootID { self.structural += 1 }
                        }
                    })
            }
        }
        func stop() { tokens.forEach(NotificationCenter.default.removeObserver); tokens = [] }
    }
    private func closeRecords(_ log: MenuDiagnostics) async throws -> [[String: Any]] {
        await log.flush()
        return try String(contentsOf: log.url, encoding: .utf8).split(separator: "\n").map {
            try #require(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
        }.filter { ($0["record"] as? String) == "close" }
    }

    // MARK: The gear acts in place, as Load does

    /// A gear's mouse-down and mouse-up, through AppKit's hit test as the menu's tracking delivers them, open the card and
    /// then close it: no action goes through the enclosing menu item (no willSendAction on any menu), nothing ends
    /// tracking (no cancelTracking), exactly as the row's Load button in the same table. The close log shows no
    /// cancelTracking and no item action for Models or the Arsenal menu.
    @Test func aGearClickSendsNoMenuActionAndEndsNoTrackingAsLoadDoes() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let open = try await Self.open(["qwen3.8-27b", "flux2-klein-4b"], support: support)
        defer { open.close() }
        let host = open.host
        let calls = MenuCalls()
        defer { calls.stop() }
        let qwen = try #require(open.app.table.rows.first { $0.id == "qwen3.8-27b" })
        let cancellations = try await Self.countingCancellations {
            // Load, the mechanism that already acts in place.
            let load = try #require(SettingsCardHost.views(RowActionView.self, host.view).first)
            let target = try host.press(load)
            #expect(target === load)
            await host.settle()
            // The gear, twice: open, then closed.
            let gear = try #require(host.gear(qwen.id))
            try host.press(gear)
            await host.settle()
            #expect(open.app.table.settingsCard.openID == qwen.id && gear.active)
            try host.press(gear)
            await host.settle()
            #expect(open.app.table.settingsCard.openID == nil && !gear.active)
        }
        #expect(cancellations == 0, "nothing ended tracking")
        #expect(calls.actions == 0, "no action went through a menu item")
        #expect(open.tableMenu.items.first?.action == nil && open.tableMenu.items.first?.target == nil, "the table's item has no action to send")
        open.closeMenus()
        let closes = try await closeRecords(open.app.menuDiagnostics)
        #expect(closes.allSatisfy { ($0["cancelTrackingSites"] as? [String]) == [] && ($0["itemAction"] == nil) }, "\(closes)")
    }

    // MARK: Every control acts in place; the ways the card closes

    /// Every control in Qwen's, FLUX's and Kokoro's cards through its own event path (a switch's action, segments' hit-
    /// tested mouse-down, a field's commit, Restore defaults' press) while the Arsenal menu and Models are open: nothing
    /// ends tracking, no alert, panel or window comes, no write fails. Then the ways the card closes: Esc (the menu stays),
    /// a click elsewhere in the table (which does its own job: Refresh refreshes), another row's gear (its card takes the
    /// place), and Models closing.
    @Test(arguments: ["qwen3.8-27b", "flux2-klein-4b", "kokoro-82m"])
    func everyCardControlActsInPlaceAndTheCardClosesTheWaysItShould(id: String) async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let other = id == "qwen3.8-27b" ? "kokoro-82m" : "qwen3.8-27b"
        let open = try await Self.open([id, other], support: support)
        defer { open.close() }
        let app = open.app, host = open.host
        var alerts = 0, panels = 0, refreshes = 0
        app.modelsMenu.presentAlert = { _ in
            alerts += 1; return .alertFirstButtonReturn
        }
        app.modelsMenu.presentOpenPanel = { _ in
            panels += 1; return nil
        }
        app.modelsMenu.refreshModels = { refreshes += 1 }
        let windowsBefore = Set(NSApp.windows.map(ObjectIdentifier.init))
        let row = try #require(app.table.rows.first { $0.id == id })
        let calls = MenuCalls()
        defer { calls.stop() }
        let cancellations = try await Self.countingCancellations {
            try host.press(try #require(host.gear(id)))
            await host.settle()
            #expect(app.table.settingsCard.openID == id)
            var exercised: Set<String> = []
            for toggle in host.views(MenuSwitch.self) where toggle.isEnabled {
                toggle.state = toggle.state == .on ? .off : .on
                #expect(toggle.sendAction(toggle.action, to: toggle.target)); exercised.insert("switch")
            }
            await host.settle()
            for segments in host.views(MenuSegments.self) where segments.isEnabled && segments.segmentCount > 1 {
                try host.press(segments); exercised.insert("segments")
            }
            for field in host.views(MenuField.self) where field.isEnabled && field.alignment == .right {
                field.stringValue = field.stringValue.isEmpty ? "7" : field.stringValue
                #expect(field.sendAction(field.action, to: field.target)); exercised.insert("field")
            }
            for _ in 0..<20 { try await Task.sleep(for: .milliseconds(10)); SettingsCardHost.pump() }
            await app.table.waitUntilSettled()
            await host.settle()
            #expect(exercised.contains("switch"), "\(exercised)")
            #expect(app.table.settingsCard.openID == id, "the card stays open through its own controls")

            // Esc closes the card only.
            let esc = try #require(
                NSEvent.keyEvent(
                    with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: host.window?.windowNumber ?? 0, context: nil,
                    characters: "\u{1b}", charactersIgnoringModifiers: "\u{1b}", isARepeat: false, keyCode: 53))
            #expect(host.view.performKeyEquivalent(with: esc))
            #expect(app.table.settingsCard.openID == nil)

            // A click elsewhere in the table closes the card, and does its own job.
            try host.press(try #require(host.gear(id)))
            await host.settle()
            #expect(app.table.settingsCard.openID == id)
            try host.press(try #require(SettingsCardHost.views(RefreshControlView.self, host.view).first))
            await host.settle()
            #expect(app.table.settingsCard.openID == nil && refreshes == 1)

            // Another row's gear: its card takes the place.
            try host.press(try #require(host.gear(id)))
            await host.settle()
            try host.press(try #require(host.gear(other)))
            await host.settle()
            #expect(app.table.settingsCard.openID == other && host.gear(other)?.active == true && host.gear(id)?.active == false)
        }
        #expect(cancellations == 0 && calls.actions == 0)
        #expect(alerts == 0 && panels == 0 && app.table.lastError == nil)
        let made = NSApp.windows.filter { !windowsBefore.contains(ObjectIdentifier($0)) && $0 !== host.window }
        #expect(made.allSatisfy { !$0.isVisible }, "No window was shown: \(made)")
        // Models closing takes the card along.
        open.closeMenus()
        #expect(app.table.settingsCard.openID == nil)
        _ = row
    }

    // MARK: No structural change in an open menu

    /// Show in menu, switched off and on in the card for every catalog model with tools while the Arsenal menu and Models
    /// are open: the switch shows the new state at once and the setting is saved, and the main menu hides or shows the
    /// model's tool items at once. No menu gains or loses an item (the close log records no structural call), so
    /// nothing waits for the menus to close.
    @Test func showInMenuChangesNoOpenMenuAndAppliesAfterClose() async throws {
        let ids = try Self.catalogModels.filter { !(try Self.folderRow($0).0.tools.isEmpty) }
        #expect(ids.count >= 3, "\(ids)")
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let open = try await Self.open(ids, support: support)
        defer { open.close() }
        let app = open.app, host = open.host
        // A model's place in the main menu: the entries of the tasks it serves (and its choice inside one, for Chat).
        func entries(_ id: String) -> [NSMenuItem] {
            guard let row = app.table.rows.first(where: { $0.id == id }) else { return [] }
            let keys = Set(app.tools.tasks(row).map(AppDelegate.taskKey))
            return app.menu.everyItem.filter { item in
                if let key = item.representedObject as? String { return keys.contains(key) }
                if let pair = item.representedObject as? [String], pair.count == 2 { return keys.contains(pair[0]) && pair[1] == id }
                return false
            }
        }
        func toolItems(_ id: String) -> Int { entries(id).count }
        func visibleToolItems(_ id: String) -> Int { entries(id).filter { $0.menu === app.menu && !$0.isHidden }.count }
        let all = Dictionary(uniqueKeysWithValues: ids.map { ($0, toolItems($0)) })
        // Models whose tools have a main-menu item now (a tool with no repair to offer shows none).
        let before = all.filter { $0.value > 0 }
        #expect(before.count >= 2, "\(all)")
        let calls = MenuCalls(root: app.menu)
        defer { calls.stop() }
        for (pass, on) in [false, true, false].enumerated() {
            for id in ids {
                // Every catalog model with tools is more than the table shows at once (it scrolls past
                // ModelTable.maxHeight): scroll to the row first, as a user does.
                let gear = try #require(host.gear(id))
                await host.reveal(gear)
                try host.press(gear)
                await host.settle()
                let toggle = try #require(host.views(MenuSwitch.self).first { $0.exposed?.accessibilityLabel()?.hasSuffix("in menu") == true }, "\(id)")
                toggle.state = on ? .on : .off
                #expect(toggle.sendAction(toggle.action, to: toggle.target))
                await host.settle()
                // Shown at once in the card.
                #expect(app.table.settings[id]?.first { $0.id == TableController.toolsKey }?.value == .bool(on), "\(id) pass \(pass)")
            }
            await app.table.waitUntilSettled()
            for _ in 0..<10 { try await Task.sleep(for: .milliseconds(10)); SettingsCardHost.pump() }
            await host.settle()
            #expect(ids.allSatisfy { id in app.table.rows.first { $0.id == id }?.toolsShown == on }, "saved at once")
            // The main menu follows at once, with every menu open: the items are hidden or shown, none added or removed.
            #expect(ids.allSatisfy { (visibleToolItems($0) > 0) == (on && all[$0]! > 0) }, "pass \(pass): \(ids.map { ($0, visibleToolItems($0)) })")
            #expect(!app.structurePending, "pass \(pass): Show in menu waits for nothing")
        }
        #expect(calls.structural == 0, "no item was added to or removed from an open menu")
        #expect(ids.allSatisfy { toolItems($0) == all[$0] }, "the open menu keeps its items")
        open.closeMenus()
        // The close log keeps items' hidden flags in its structure, so it counts the visibility changes; `calls` above is
        // the check that nothing was added or removed.
        let closes = try await closeRecords(app.menuDiagnostics)
        #expect(!closes.isEmpty)
        // After every menu closed: nothing was pending, the items stay in place and are hidden (Show in menu is off).
        for _ in 0..<4 { await Task.yield(); SettingsCardHost.pump() }
        #expect(!app.structurePending && calls.structural == 0)
        #expect(ids.allSatisfy { toolItems($0) == all[$0] && visibleToolItems($0) == 0 }, "after close: \(ids.map { ($0, toolItems($0), visibleToolItems($0)) })")
    }

    // MARK: The card never changes the table

    /// For every catalog model in one table: its card opens beside its gear (trailing edge `gap` left of the gear, top
    /// level with the row, unless the table's bounds clamp it) and inside the table; the table's host keeps its frame and
    /// the item its natural size, before, while and after; the gear's place from the metrics is the gear's own frame.
    @Test func theCardNeverChangesTheTableAndSitsBesideItsGear() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        for ids in Self.catalogTables {
            let open = try await Self.open(ids, support: support)
            defer { open.close() }
            let app = open.app, host = open.host
            #expect(!ModelTable.scrolls(app.table))
            let natural = host.view.naturalSize, frame = host.view.frame, height = ModelTable.height(app.table)
            let table = SettingsCard.tableSize(app.table)
            for row in app.table.rows {
                let gear = try #require(host.gear(row.id))
                let metric = try #require(ModelTable.gearRect(app.table, id: row.id))
                let own = SettingsCard.gearRect(app.table, row: row, anchor: gear)
                #expect(abs(metric.minX - own.minX) < 0.5 && abs(metric.minY - own.minY) < 0.5, "\(row.id): metrics \(metric) vs view \(own)")
                try host.press(gear)
                await host.settle()
                let card = host.frame
                #expect(!card.isEmpty, "\(row.id)")
                #expect(CGRect(origin: .zero, size: table).insetBy(dx: SettingsCard.inset - 0.5, dy: SettingsCard.inset - 0.5).contains(card), "\(row.id): \(card) in \(table)")
                if own.minX - SettingsCard.gap - card.width >= SettingsCard.inset { #expect(abs(card.maxX - (own.minX - SettingsCard.gap)) <= 1, "\(row.id)") }
                if own.minY + card.height <= table.height - SettingsCard.inset { #expect(abs(card.minY - own.minY) <= 1, "\(row.id): top level with the row") }
                #expect(host.view.naturalSize == natural && host.view.frame == frame && ModelTable.height(app.table) == height, "\(row.id)")
                try host.press(gear)
                await host.settle()
            }
            #expect(host.view.naturalSize == natural && host.view.frame == frame)
        }
    }

    // MARK: Hosting settings only

    /// For every catalog model, loaded or not: the card lists only the app's rows and the model's `applies = "load"`
    /// settings; every request setting it declares is absent, as a row and as a control's name in the open card.
    @Test func noRequestSettingAppearsInAnyCard() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        var summary: [String] = []
        for ids in Self.catalogTables {
            let open = try await Self.open(ids, support: support)
            defer { open.close() }
            let app = open.app, host = open.host
            for id in ids {
                let states = app.table.settings[id] ?? []
                let request = states.filter { !PaneLayout.isAppRow($0) && $0.applies != "load" }
                let items = PaneLayout.items(states, panel: app.table.panels[id])
                for item in items {
                    if case .setting(let s, _) = item.kind { #expect(PaneLayout.isAppRow(s) || s.applies == "load", "\(id): \(s.id)") }
                    if case .presets(_, let presets, _, _) = item.kind {
                        let keys = Set(presets.flatMap(\.values.keys))
                        #expect(keys.allSatisfy { k in states.first { $0.id == k }?.applies == "load" }, "\(id): presets \(keys)")
                    }
                }
                let shown = items.compactMap { item -> String? in
                    if case .setting(let s, let style) = item.kind { return s.title(style) }; return nil
                }
                summary.append("\(id): " + shown.joined(separator: ", "))
                try host.press(try #require(host.gear(id)))
                await host.settle()
                let names = Set(
                    host.views(MenuSwitch.self).compactMap { $0.exposed?.accessibilityLabel() } + host.views(MenuSegments.self).compactMap { $0.accessibilityLabel() }
                        + host.views(MenuField.self).compactMap { $0.accessibilityLabel() })
                for s in request { #expect(!names.contains(s.title), "\(id): \(s.title) is a request setting") }
                app.table.settingsCard.close()
            }
        }
        try? summary.joined(separator: "\n").write(to: URL(fileURLWithPath: "/private/tmp/arsenal-r13-gear-card/card-settings.txt"), atomically: true, encoding: .utf8)
    }

    // MARK: Nothing pops up

    /// No `NSMenu.popUp` and no NSPopUpButton is reachable from the Models menu: none in the code of its table, card and
    /// controls, and none in the view tree of an open card for any catalog model.
    @Test func nothingInTheModelsMenuPopsUpAMenu() async throws {
        let sources = SettingsRenderDelegate.repo.appendingPathComponent("Sources/Arsenal")
        let files =
            ["Table", "Settings"].flatMap { dir -> [URL] in
                (FileManager.default.enumerator(at: sources.appendingPathComponent(dir), includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? [])
                    .filter { $0.pathExtension == "swift" }
            } + [sources.appendingPathComponent("Menu/ModelsMenu.swift"), sources.appendingPathComponent("Menu/MenuContentView.swift")]
        #expect(files.count > 10)
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for banned in [".popUp(", "popUpContextMenu", "NSPopUpButton", "NSComboBox"] {
                #expect(!text.contains(banned), "\(file.lastPathComponent) uses \(banned)")
            }
        }
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        for ids in Self.catalogTables {
            let open = try await Self.open(ids, support: support)
            defer { open.close() }
            for id in ids {
                try open.host.press(try #require(open.host.gear(id)))
                await open.host.settle()
                #expect(SettingsCardHost.views(NSPopUpButton.self, open.host.view).isEmpty && SettingsCardHost.views(NSComboBox.self, open.host.view).isEmpty, "\(id)")
                open.app.table.settingsCard.close()
            }
            #expect(open.host.menu.items.count == 1 && open.host.menu.items.allSatisfy { $0.submenu == nil }, "Models holds the table alone, with no submenu")
        }
    }

    // MARK: Renders

    /// Review renders (run with ARSENAL_CARD_RENDERS=<dir>): Qwen's, FLUX's and Kokoro's cards beside their gears in the owner's
    /// table, dark and light, at 2× on a stand-in of the menu material.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_CARD_RENDERS"] != nil))
    func cardRenders() async throws {
        let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ARSENAL_CARD_RENDERS"]))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            for (id, state) in [("qwen3.8-27b", ModelState.ready), ("flux2-klein-4b", .ready), ("kokoro-82m", .loaded)] {
                let support = support()
                defer { try? FileManager.default.removeItem(at: support) }
                var rows: [ModelRow] = [], metadata: [String: JSONObject] = [:]
                for model in SettingsRenderDelegate.tableModels {
                    let (row, meta) = try Self.folderRow(model, state: model == id ? state : .ready)
                    rows.append(row); metadata[model] = meta
                }
                let c = TableController(service: FixtureService(rows: rows, metadata: metadata, supportDirectory: support), chip: "M5 Max")
                await c.refresh()
                let row = try #require(c.rows.first { $0.id == id })
                c.openSettings(row)
                let size = try await SettingsCardRender.render(c, row: row, to: directory.appendingPathComponent("card-\(id)-\(mode.rawValue).png"))
                #expect(size.width > 0)
                c.stop()
            }
        }
        Appearance.shared.set(.dark)
    }
}

/// The Models table with a model's card open beside its gear, on a stand-in of the menu material (the real material is
/// the window server's), at 2× pixels, from the table host's layer tree (the card's shadow included).
@MainActor enum SettingsCardRender {
    static func render(_ c: TableController, row: ModelRow, to url: URL, focus: Bool = false) async throws -> NSSize {
        let host = SettingsCardHost(c)
        defer { host.close() }
        if c.settingsCard.openID != row.id { await host.open(row) }
        // The gear's own place, now that the table is laid out.
        c.settingsCard.state.gear = SettingsCard.gearRect(c, row: row, anchor: host.gear(row.id))
        await host.settle()
        if focus, let field = SettingsRenderDelegate.fields(in: host.view).first { host.window?.makeFirstResponder(field) }
        let dark = Appearance.shared.mode != .light
        let inset: CGFloat = 5
        let size = NSSize(width: host.view.naturalSize.width + 2 * inset, height: host.view.naturalSize.height + 2 * inset)
        let scale: CGFloat = 2
        let out = try #require(
            NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: Int(size.width * scale), pixelsHigh: Int(size.height * scale), bitsPerSample: 8,
                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        out.size = size
        // The rep's point size makes the context draw in points at 2× pixels.
        let context = try #require(NSGraphicsContext(bitmapImageRep: out)?.cgContext)
        // The menu: its material, corners and inset.
        let menu = CGPath(roundedRect: CGRect(origin: .zero, size: size), cornerWidth: 10, cornerHeight: 10, transform: nil)
        context.addPath(menu); context.clip()
        context.setFillColor(MenuMock.material(dark: dark).cgColor); context.fill(CGRect(origin: .zero, size: size))
        MenuMock.standInMaterials(in: host.view, dark: dark, picture: nil)
        host.view.layoutSubtreeIfNeeded(); host.view.displayIfNeeded(); CATransaction.flush()
        let layer = try #require(host.view.layer)
        context.saveGState()
        context.translateBy(x: inset, y: size.height - inset)
        context.scaleBy(x: 1, y: -1)
        layer.render(in: context)
        context.restoreGState()
        try out.representation(using: .png, properties: [:])?.write(to: url)
        return host.view.naturalSize
    }
}
