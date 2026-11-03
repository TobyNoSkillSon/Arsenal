// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Synchronization
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MenuRound3Tests {
    @Test(arguments: [false, true])
    func hidingOrRemovingFirstVoiceNeverDetachesSecond(remove: Bool) async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/two-voices-\(UUID())")
        defer { try? FileManager.default.removeItem(at: support) }
        let rows = [row("first", .voice), row("second", .voice)]
        let service = FixtureService(rows: rows, supportDirectory: support)
        let menu = StructuralMenuProbe()
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support), menu: menu)
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        // Each loaded voice model's agent voice is its own root in Settings ("second Voice: …" beside "first Voice: …").
        let settingsRoot = try #require(menu.items.first { $0.title == "Settings" })
        let settings = try #require(settingsRoot.submenu)
        let second = try #require(settings.items.first { ($0.representedObject as? [String]) == ["voice", "second"] })
        #expect(second.title.hasPrefix("second Voice: "))
        let voiceMenu = try #require(second.submenu)
        let voiceSize = voiceMenu.size
        let activity = try #require(menu.items.first { $0.title == "Activity" }?.submenu?.items.first?.view as? ActivityHostingView)
        let frames = MotionFrames(host: activity)
        defer { frames.close() }
        await frames.settle()
        let before = try frames.capture()
        app.menuWillOpen(menu); settings.delegate?.menuWillOpen?(settings); voiceMenu.delegate?.menuWillOpen?(voiceMenu)
        let survivors = menu.items.filter { ($0.representedObject as? [String])?.last != "first" && $0.title != "first" }
        let settingsItems = settings.items.map(ObjectIdentifier.init)
        menu.reset()
        // Remove unloads and deletes the weights (as the real service): the table holds only models with weights on this Mac.
        if remove {
            try await service.unload(id: "first"); try await service.remove(id: "first", deleteWeights: true)
        } else {
            try await service.setToolsShown(id: "first", on: false)
        }
        await app.table.refresh(); await app.refreshMenuState()
        for _ in 0..<12 {
            await frames.tick()
            let frame = try frames.capture()
            #expect(frame.bytes == before.bytes && frame.size == before.size && frame.controls == before.controls)
            #expect(second.menu === settings && second.submenu === voiceMenu && settingsRoot.submenu === settings)
            #expect(voiceMenu.size == voiceSize)
            // The open Settings submenu changes no structure either: the first voice's root waits for close.
            #expect(settings.items.map(ObjectIdentifier.init) == settingsItems)
        }
        #expect(menu.calls(on: [settingsRoot]).isEmpty, "The Settings root holding the surviving Voice root must stay attached")
        #expect(menu.calls(on: survivors.filter { $0.menu === menu }).isEmpty)
        voiceMenu.delegate?.menuDidClose?(voiceMenu); settings.delegate?.menuDidClose?(settings); app.menuDidClose(menu)
    }

    @Test func toolMembershipMatrixNeverRestructuresSurvivors() async throws {
        for tool in ToolKind.allCases {
            let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/matrix-\(UUID())")
            defer { try? FileManager.default.removeItem(at: support) }
            let changed = row("changed", tool)
            let service = FixtureService(
                rows: [changed, row("neighbour", .voice), row("last", .chat)], metadata: [changed.id: ["tools": [.string(tool.rawValue)]]],
                kinds: Fixtures.kinds + [LayersFixture.kind], supportDirectory: support,
                operations: tool == .operation
                    ? [changed.id: [LayersFixture.operation]] : tool == .background ? [changed.id: [CutoutFixture.operation]] : [:])
            let menu = StructuralMenuProbe()
            let app = AppDelegate(service: service, paths: ArsenalPaths(support: support), menu: menu)
            defer { app.table.stop(); app.tools.closeAll() }
            await app.table.refresh(); await app.refreshMenuState()
            // The model's place in the menu is its task's entry; a task it shares with another model shows its membership
            // as a model choice inside the entry (Chat), or not at all (Speak, which picks the model inside its window).
            let rows = app.table.rows
            let task = app.tools.tasks(try #require(rows.first { $0.id == changed.id })).first
            let shared = task.map { t in rows.contains { $0.id != changed.id && app.tools.tasks($0).contains(t) } } ?? false
            let tracks = task.map { !(shared && !AppDelegate.perModelWindow($0)) } ?? false
            func item() -> NSMenuItem? {
                guard let task else { return nil }
                let key = AppDelegate.taskKey(task)
                if shared && AppDelegate.perModelWindow(task) { return menu.everyItem.first { ($0.representedObject as? [String]) == [key, changed.id] } }
                return menu.items.first { ($0.representedObject as? String) == key }
            }
            #expect(task != nil || tool == .search, "\(tool.rawValue): every tool but Search has a task")
            for operation in ["hide", "show", "remove", "add", "refresh"] {
                app.menuWillOpen(menu)
                let before = menu.items
                func has() -> Bool { item() != nil }
                func isShown() -> Bool { item().map { !$0.isHidden } ?? false }
                let present = has(), shown = isShown()
                menu.reset()
                switch operation {
                case "hide": try await service.setToolsShown(id: changed.id, on: false)
                case "show": try await service.setToolsShown(id: changed.id, on: true)
                // Remove deletes the model's weights, which takes it off the table; Add downloads it again (a Get joins the
                // table with Show in menu off).
                case "remove":
                    try await service.unload(id: changed.id); try await service.remove(id: changed.id, deleteWeights: true)
                case "add": try await service.get(id: changed.id, precision: "bf16")
                default: try await service.refresh()
                }
                await app.table.refresh(); await app.refreshMenuState()
                let survivors = before.filter { old in menu.items.contains { $0 === old } }
                guard tracks else {
                    // Search (no entry) or a shared Speak…: whatever happens to this model, survivors stay attached and the
                    // shared entry stays shown for its other model.
                    #expect(menu.calls(on: survivors).isEmpty && menu.removeAllCalls == 0, "\(tool.rawValue)/\(operation)")
                    #expect(has() == (task != nil) && isShown() == (task != nil), "\(tool.rawValue)/\(operation)")
                    #expect(!app.structurePending || !["hide", "show"].contains(operation), "\(tool.rawValue)/\(operation): no structure waits for close")
                    app.menuDidClose(menu)
                    for _ in 0..<4 { await Task.yield(); SettingsCardHost.pump() }
                    #expect(has() == (task != nil) && isShown() == (task != nil), "\(tool.rawValue)/\(operation) after close")
                    continue
                }
                #expect(menu.calls(on: survivors).isEmpty, "\(tool.rawValue)/\(operation): surviving native items stay attached")
                #expect(menu.removeAllCalls == 0)
                // An open menu never changes its structure: a model's removal or return waits for the menu to close.
                #expect(has() == present, "\(tool.rawValue)/\(operation): nothing added or removed while open")
                if operation == "hide" || operation == "show" {
                    // Show in menu is a visibility change only: it applies at once, with no structural work pending.
                    #expect(isShown() == (operation == "show"), "\(tool.rawValue)/\(operation): visibility follows at once")
                    #expect(!app.structurePending, "\(tool.rawValue)/\(operation): no structure waits for close")
                } else {
                    #expect(isShown() == shown, "\(tool.rawValue)/\(operation): visibility unchanged while open")
                }
                app.menuDidClose(menu)
                for _ in 0..<4 { await Task.yield(); SettingsCardHost.pump() }
                #expect(has() == (operation != "remove"), "\(tool.rawValue)/\(operation): tool present after close unless its model was removed")
                // A model added back starts with Show in menu off, so it stays hidden through the refresh after it.
                #expect(isShown() == !["hide", "remove", "add", "refresh"].contains(operation), "\(tool.rawValue)/\(operation): tool visible after close")
            }
        }
    }

    @Test func aSoundToolOnAChatModelWithoutAudioOperationsIsRefused() async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/unsupported-sound-\(UUID())")
        defer { try? FileManager.default.removeItem(at: support) }
        var chat = row("chat-with-sound", .chat)
        chat.tools = [ToolKind.sound.rawValue]
        let service = FixtureService(rows: [chat], metadata: [chat.id: ["tools": ["sound"]]], supportDirectory: support)
        let refusal = await service.runRefusal(id: chat.id, op: "generate_sound")
        #expect(refusal?.code == "unsupported_op")
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        #expect(app.tools.admissions[chat.id]?[.sound]?.refusal?.code == "unsupported_op")
        #expect(!app.menu.items.contains { ($0.representedObject as? [String]) == ["sound", chat.id] })
    }

    /// Two models with one name, then a catalog rename and one hidden, while the menu is open: no model's name heads a
    /// section, and the voice roots in Settings follow in place, attached, with the task entry untouched.
    @Test func duplicateNamesAndCatalogRenameKeepModelHeadersAttached() async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/headers-\(UUID())")
        defer { try? FileManager.default.removeItem(at: support) }
        var first = row("first", .voice), second = row("second", .voice)
        first.name = "Same"; second.name = "Same"
        let app = AppDelegate(service: FixtureService(rows: [first, second], supportDirectory: support), paths: ArsenalPaths(support: support), menu: StructuralMenuProbe())
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        let menu = try #require(app.menu as? StructuralMenuProbe)
        #expect(!menu.items.contains { $0.isSectionHeader && $0.title.contains("Same") })
        let speak = try #require(menu.items.first { ($0.representedObject as? String) == AppDelegate.taskKey(.sound(.speak)) })
        let settings = try #require(menu.settings)
        let voice = try #require(settings.items.first { ($0.representedObject as? [String]) == ["voice", "second"] })
        #expect(voice.title.hasPrefix("Same Voice: "))
        first.toolsShown = false; second.name = "Renamed"
        menu.reset()
        app.menuWillOpen(menu)
        app.table.apply(rows: [first, second], kinds: app.table.kinds)
        await app.refreshMenuState()
        // The first voice hides with its Show in menu; the renamed one, now the only one shown, is the agents' voice and
        // names its model in its hint, in place.
        #expect(voice.menu === settings && menu.settings === settings && voice.title.hasPrefix("Agent Voice: ") && !voice.isHidden)
        #expect(voice.toolTip?.contains("Renamed") == true)
        #expect(settings.items.first { ($0.representedObject as? [String]) == ["voice", "first"] }?.isHidden == true)
        #expect(speak.menu === menu && !speak.isHidden && menu.calls(on: [speak]).isEmpty)
        #expect(!menu.items.contains { $0.title.contains("Renamed") || $0.title.contains("Same") })
        app.menuDidClose(menu)
    }

    @Test func catalogReorderingUpdatesWithoutDetachingSurvivingRoots() {
        let menu = StructuralMenuProbe()
        func item(_ id: String) -> NSMenuItem {
            let item = NSMenuItem(title: id, action: nil, keyEquivalent: "")
            item.representedObject = ["voice", id]; item.submenu = NSMenu()
            return item
        }
        let first = item("first"), second = item("second")
        MenuReconciliation.apply([first, second], to: menu)
        menu.reset()
        MenuReconciliation.apply([item("second"), item("added"), item("first")], to: menu)
        #expect(menu.calls(on: [first, second]).isEmpty)
        #expect(menu.items.count == 3 && first.menu === menu && second.menu === menu)
        menu.reset()
        let childMenu = StructuralMenuProbe()
        let child = NSMenuItem(title: "Choice", action: nil, keyEquivalent: "")
        childMenu.addItem(child); second.submenu = childMenu; childMenu.reset()
        let before = menu.items
        MenuReconciliation.apply(before, to: menu)
        #expect(menu.calls(on: before).isEmpty, "An already attached snapshot is idempotent")
        #expect(childMenu.calls(on: [child]).isEmpty, "The same snapshot must not empty/reinsert its live submenu")
    }

    private func row(_ id: String, _ tool: ToolKind) -> ModelRow {
        var row = ModelRow(
            id: id, name: id,
            kind: tool == .voice
                ? "voices" : tool == .image || tool == .background ? "images" : tool == .sound ? "sound" : tool == .operation ? LayersFixture.kind.id : "chat",
            states: ["bf16": .init(state: .loaded)])
        row.tools = [tool.rawValue]
        return row
    }
}

@MainActor final class StructuralMenuProbe: NSMenu {
    private struct Calls { var removals: [ObjectIdentifier] = []; var insertions: [ObjectIdentifier] = []; var removeAll = 0 }
    nonisolated private let state = Mutex(Calls())
    var removeAllCalls: Int { state.withLock { $0.removeAll } }
    override nonisolated func removeItem(_ item: NSMenuItem) { state.withLock { $0.removals.append(ObjectIdentifier(item)) }; super.removeItem(item) }
    override nonisolated func insertItem(_ item: NSMenuItem, at index: Int) { state.withLock { $0.insertions.append(ObjectIdentifier(item)) }; super.insertItem(item, at: index) }
    override nonisolated func removeAllItems() { state.withLock { $0.removeAll += 1 }; super.removeAllItems() }
    func reset() { state.withLock { $0 = Calls() } }
    func calls(on items: [NSMenuItem]) -> [ObjectIdentifier] {
        let ids = Set(items.map(ObjectIdentifier.init))
        return state.withLock { ($0.removals + $0.insertions).filter { ids.contains($0) } }
    }
}
