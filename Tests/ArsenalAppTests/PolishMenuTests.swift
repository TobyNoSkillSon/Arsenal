// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct PolishMenuTests {
    /// All tool kinds use the same setting. Click the real switch inside the Models menu view, while the
    /// parent menu is tracking, then close/reopen it as AppKit does. No manual refresh or restart rescues it.
    @Test(arguments: ["chat", "image", "voice", "operation"], [false, true])
    func showInMenuUpdatesThroughTheActualSwitch(tool: String, loaded: Bool) async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-polish-menu-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let modelKind = tool == "image" ? "images" : tool == "voice" ? "voices" : tool
        var row = ModelRow(id: "fixture", name: "Fixture", kind: modelKind, states: ["bf16": .init(state: loaded ? .loaded : .ready)])
        row.tools = [tool]
        let service = FixtureService(rows: [row], metadata: [row.id: ["tools": [.string(tool)]]], supportDirectory: support)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        let kind = try #require(ToolKind(rawValue: tool))
        // The model's task entry: Speak… for a voice (loaded or not), the tool's title, or the generic entry's own title.
        let entry = try #require(app.tools.taskEntries(app.table.menuRows, includeHidden: true).first)
        let title = entry.title
        switch kind {
        case .voice: #expect(title == "Speak\u{2026}")
        case .operation: #expect(title.hasSuffix("\u{2026}") && title != row.name)
        default: #expect(title == kind.title)
        }
        // Every task a table model serves is built; Show in menu only hides or shows its entry.
        func visible(_ title: String) -> Bool { app.menu.items.contains { $0.title == title && !$0.isHidden } }
        #expect(visible(title))
        for shown in [false, true] {
            app.menuWillOpen(app.menu)
            let modelItem = try #require(app.menu.items.first { $0.title == "Models" })
            let entryItem = try #require(app.menu.items.first { $0.title == title })
            let current = try #require(app.table.rows.first)
            // The switch lives in the model's settings card beside its gear (never shown here).
            let settings = SettingsCardHost(app.table, models: app.modelsMenu)
            await settings.open(current)
            let view = settings.view
            let window = try #require(settings.window)
            defer { settings.close() }
            func switches(_ view: NSView) -> [MenuSwitch] {
                (view as? MenuSwitch).map { [$0] } ?? view.subviews.flatMap(switches)
            }
            let toggle = try #require(switches(view).first { $0.toolTip?.contains("main menu shows") == true })
            let point = toggle.convert(NSPoint(x: toggle.bounds.midX, y: toggle.bounds.midY), to: nil)
            let frame = try #require(window.contentView?.superview)
            let hit = frame.hitTest(frame.convert(point, from: nil))
            #expect(hit === toggle || hit?.isDescendant(of: toggle) == true)
            toggle.performClick(nil)
            try await waitUntil { app.table.rows.first?.toolsShown == shown }
            // Visibility follows the switch at once, with the menu open: no item is added or removed, so nothing waits.
            try await waitUntil { visible(title) == shown }
            // Nothing waits for close: Mute lives in Settings, Stop Speaking is always built and only hides.
            #expect(!app.structurePending)
            #expect(app.menu.items.contains { $0 === entryItem })
            // The active Models item survives the setting change, so the menu's view is not detached.
            #expect(app.menu.items.contains { $0 === modelItem })
            app.menuDidClose(app.menu)
            #expect(visible(title) == shown)
            app.menuWillOpen(app.menu)
            #expect(visible(title) == shown)
            if shown {
                let item = try #require(app.menu.items.first { $0.title == title && !$0.isHidden })
                // Loaded or not, the entry opens its window: no "not loaded — Load" step and no note.
                #expect(item.isEnabled && item.action != nil && item.badge == nil, "\(item.toolTip ?? "No tooltip")")
            }
            // No model's name heads a section or names a row of the main menu, shown or hidden.
            #expect(!app.menu.items.contains { $0.title == row.name })
            #expect(await service.calls.contains("tools fixture \(shown)"))
            app.menuDidClose(app.menu)
        }
        app.table.stop(); app.tools.closeAll()
    }

    @Test func residencyAndMemoryChoicesReachTheActualMenuInBothDirections() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-choices-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let service = FixtureService(rows: [], supportDirectory: support)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        defer { app.table.stop(); app.tools.closeAll() }
        for manual in [true, false] {
            for choice in MenuContent.idleChoices(0) {
                app.menuWillOpen(app.menu)
                let submenu = try #require(app.menu.settings?.items.first { $0.title == MenuContent.keepLoadedTitle }?.submenu)
                let index = try #require(submenu.items.firstIndex { ($0.representedObject as? Bool) == manual && $0.tag == choice.minutes })
                submenu.performActionForItem(at: index)
                try await waitUntil { (manual ? app.residency.manualIdleMinutes : app.residency.onDemandIdleMinutes) == choice.minutes }
                #expect((manual ? await service.residencySettings().manualIdleMinutes : await service.residencySettings().onDemandIdleMinutes) == choice.minutes)
                app.menuDidClose(app.menu); app.menuWillOpen(app.menu)
                let updated = try #require(app.menu.settings?.items.first { $0.title == MenuContent.keepLoadedTitle }?.submenu)
                #expect(updated.items.filter { ($0.representedObject as? Bool) == manual && $0.state == .on }.map(\.tag) == [choice.minutes])
                app.menuDidClose(app.menu)
            }
        }
        for swap in [true, false] {
            app.menuWillOpen(app.menu)
            let sub = try #require(app.menu.settings?.items.first { $0.title == "Memory" }?.submenu)
            let index = try #require(sub.items.firstIndex { ($0.representedObject as? Bool) == swap })
            sub.performActionForItem(at: index)
            try await waitUntil { app.residency.allowSwap == swap }
            #expect(await service.residencySettings().allowSwap == swap)
            app.menuDidClose(app.menu); app.menuWillOpen(app.menu)
            let updated = try #require(app.menu.settings?.items.first { $0.title == "Memory" }?.submenu)
            #expect(updated.items.filter { $0.representedObject is Bool && $0.state == .on }.map { $0.representedObject as? Bool } == [swap])
            app.menuDidClose(app.menu)
        }
        for (mode, title, _) in MenuContent.whenFull.reversed() {
            app.menuWillOpen(app.menu)
            let sub = try #require(app.menu.settings?.items.first { $0.title == "Memory" }?.submenu)
            let index = try #require(sub.items.firstIndex { ($0.representedObject as? String) == mode.rawValue })
            sub.performActionForItem(at: index)
            try await waitUntil { await service.memorySettings().whenFull == mode && app.memorySettings.whenFull == mode }
            app.menuDidClose(app.menu); app.menuWillOpen(app.menu)
            let updated = try #require(app.menu.settings?.items.first { $0.title == "Memory" }?.submenu)
            #expect(updated.items.first { $0.title == title }?.state == .on)
            app.menuDidClose(app.menu)
        }
    }

    @Test func hiddenToolsAlsoHideTheModelsContributedActions() {
        var row = ModelRow(id: "voice", name: "Voice", kind: "voices")
        row.tools = ["voice"]; row.toolsShown = false
        let action = LoadedAction(id: "speak", modelID: row.id, title: "Speak Clipboard", op: "speak")
        let sections = [LoadedMenuSection(id: row.id, modelID: row.id, title: row.name, items: [action])]
        #expect(MenuContent.toolSections([row], sections: sections).isEmpty)
        #expect(MenuContent.sections(sections, rows: [row]).isEmpty)
    }

    @Test func systemAppearanceChangesThePaletteAndOpenNativeHostsWithoutAnAppSetting() async throws {
        let application = NSApplication.shared
        let previous = application.appearance
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-appearance-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: [], supportDirectory: support), paths: ArsenalPaths(support: support))
        app.rebuildMenu()
        let models = try #require(app.modelsMenu.modelItem().submenu)
        let activity = try #require(app.activityMenu.item().submenu)
        let menus = [models, activity]
        let views = try menus.map { try #require($0.items.first?.view) }
        // Supply the native material/frame hierarchy offscreen: menuWillOpen must leave the menu's material alone.
        let materials = views.map { NSVisualEffectView(frame: $0.frame) }
        let windows = zip(views, materials).map { view, material in
            let window = NSWindow(contentRect: view.frame, styleMask: .borderless, backing: .buffered, defer: false)
            window.contentView = material; material.addSubview(view)
            return window
        }
        for menu in menus { menu.delegate?.menuWillOpen?(menu) }
        defer {
            for menu in menus { menu.delegate?.menuDidClose?(menu) }
            for window in windows { window.contentView = nil }
            app.table.stop(); app.tools.closeAll()
            application.appearance = previous; Appearance.shared.followSystem()
        }
        for (name, light) in [(NSAppearance.Name.aqua, true), (.darkAqua, false), (.aqua, true)] {
            application.appearance = NSAppearance(named: name)
            try await waitUntil {
                Appearance.shared.palette.dark == !light
                    && menus.allSatisfy { $0.appearance?.name == name }
                    && views.allSatisfy { $0.appearance?.name == name }
            }
            #expect(Appearance.shared.colorScheme == (light ? .light : .dark))
            for (menu, view, material) in zip(menus, zip(views, materials)).map({ ($0.0, $0.1.0, $0.1.1) }) {
                #expect(menu.appearance?.name == name)
                #expect(view.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == name)
                // No fill of Arsenal's own: the material holds the hosted view only.
                #expect(material.subviews == [view])
            }
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MENU_RENDER"] == "1"))
    func renderFirstFrames() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = root.appendingPathComponent("docs/screens")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let application = NSApplication.shared
        let previous = application.appearance
        defer { application.appearance = previous; Appearance.shared.followSystem() }
        for (name, mode) in [(NSAppearance.Name.darkAqua, "dark"), (.aqua, "light")] {
            application.appearance = NSAppearance(named: name); Appearance.shared.followSystem()
            for loaded in [false, true] {
                var rows = Fixtures.rows(loaded ? .loaded : .firstLaunch)
                if loaded, let index = rows.firstIndex(where: { $0.id == "flux2-klein-4b" }) { rows[index].states["bf16"] = .init(state: .loaded) }
                let support = root.appendingPathComponent(".menu-render-" + UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: support) }
                let service = FixtureService(rows: rows, supportDirectory: support)
                let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
                await app.table.refresh(); await app.refreshMenuState()
                app.previewState(actions: await service.menuActions(), memory: .init(totalMB: 128_000, availableMB: 86_900), sections: await service.menuSections())
                app.tools.preview(voices: ["kokoro-82m": [.init(id: "heart", name: "Heart", language: "American English")]], current: ["kokoro-82m": "heart"])
                app.rebuildMenu()
                let suffix = loaded ? "loaded" : "empty"
                await capture(app.menu.items, to: directory.appendingPathComponent("polish-menu-\(suffix)-\(mode).png"))
                if !loaded {
                    for title in [MenuContent.keepLoadedTitle, "Memory"] {
                        let submenu = try #require(app.menu.settings?.items.first { $0.title == title }?.submenu)
                        await capture(submenu.items, to: directory.appendingPathComponent("polish-menu-\(title == "Memory" ? "memory" : "keep-hot")-\(mode).png"))
                    }
                }
                app.table.stop(); app.tools.closeAll()
            }
        }
    }

    private func capture(_ items: [NSMenuItem], to url: URL) async {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(items, width: 260, to: url) { done.resume() }
        }
    }
    private func waitUntil(_ condition: @MainActor () async -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while !(await condition()) {
            try #require(ContinuousClock.now < deadline, "Menu state did not update")
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
