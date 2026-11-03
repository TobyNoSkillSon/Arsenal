// SPDX-License-Identifier: MIT
import Foundation
import Testing
import AppKit
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) final class MenuTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-app-menu-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    private func titles(_ menu: NSMenu) -> [String] { menu.items.filter { !$0.isHidden }.map { $0.isSeparatorItem ? "—" : $0.title } }

    /// The table holds the models whose weights are on this Mac: here a voice, an image and a chat model, none loaded.
    static var downloaded: [ModelRow] {
        var rows = Fixtures.rows(.firstLaunch)
        for (id, dtype) in [("kokoro-82m", "fp16"), ("flux2-klein-4b", "bf16"), ("qwen3.8-27b", "bf16")] {
            rows[rows.firstIndex { $0.id == id }!].states[dtype] = PrecisionState(state: .ready)
        }
        return rows
    }

    @Test func headlessStatusRetainsTheButtonWithoutAppearing() {
        let item = AppDelegate.makeStatusItem(visible: false)
        defer { NSStatusBar.system.removeStatusItem(item) }
        #expect(!item.isVisible && item.length == 0)
        #expect(item.button?.image != nil)
    }

    @Test func theMenuOrderWithoutLoadedModels() async throws {
        let app = AppDelegate(service: FixtureService(rows: Self.downloaded), paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        // Tasks first, each opening its window though no model is loaded yet; then Models, Activity and Settings.
        #expect(
            titles(app.menu) == [
                "Speak\u{2026}", "Make an Image\u{2026}", "Chat\u{2026}", "—", "Models", "Activity", "Settings", "—",
                "About Arsenal", "Support the Developer\u{2026}", "Quit Arsenal"
            ])
        #expect(app.menu.items.filter { !$0.isHidden && !$0.isSeparatorItem }.prefix(3).allSatisfy { $0.isEnabled && $0.action != nil && $0.badge == nil })
        #expect(app.menu.items.last?.keyEquivalent == "q")
        #expect(app.menu.items.filter { !$0.isSeparatorItem && !$0.isSectionHeader }.allSatisfy { $0.image != nil && $0.menuImageIsVisible })
        let modelsRoot = try #require(app.menu.items.first { $0.title == "Models" }?.submenu)
        #expect(modelsRoot.items.count == 1 && modelsRoot.items.first?.view != nil)
        #expect(!app.menu.items.contains { $0.title == MenuContent.refresh })
        let settings = try #require(app.menu.items.first { $0.title == "Settings" }?.submenu)
        #expect(
            settings.items.map { $0.isSeparatorItem ? "—" : $0.title } == [
                MenuContent.keepLoadedTitle, "Memory", "—", "Mute Agent Speech", "—", HuggingFaceToken.title, AppDelegate.loginTitle, MenuContent.openFiles
            ])
        let keepHot = try #require(settings.items.first { $0.title == MenuContent.keepLoadedTitle }?.submenu)
        // Vella's idle windows: Manually loaded defaults to Always, on demand to 15 min.
        #expect(keepHot.items.first { $0.state == .on }?.title == "Always")
        #expect(keepHot.items.filter { $0.state == .on }.map(\.title) == ["Always", "15 min idle"])
        let memory = try #require(settings.items.first { $0.title == "Memory" }?.submenu)
        #expect(memory.items.prefix(2).map(\.title) == [MenuContent.fitTitle, MenuContent.swapTitle])
        #expect(memory.items[0].state == .on)
    }

    /// Voice choices follow speech; shared playback controls follow model sections. Unloading removes executable actions.
    @Test func loadedModelsContributeActionsThatComeAndGo() async throws {
        let service = FixtureService(rows: Self.downloaded)
        // Menu content depends on playback state, not the process-wide saved Mute preference.
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        app.tools.playback = FixturePlayback()
        try await service.get(id: "kokoro-82m", precision: "fp16")
        try await service.load(id: "kokoro-82m", precision: "fp16", fast: true)
        await app.table.refresh(); await app.refreshMenuState()
        // The folder's action follows the tasks; Stop Speaking shows while something speaks.
        #expect(
            titles(app.menu) == [
                "Speak\u{2026}", "Make an Image\u{2026}", "Chat\u{2026}", "—", "Speak Clipboard", "—", "Stop Speaking", "—",
                "Models", "Activity", "Settings", "—", "About Arsenal", "Support the Developer\u{2026}", "Quit Arsenal"
            ])
        let speak = try #require(app.menu.items.first { $0.title == "Speak Clipboard" })
        #expect(speak.subtitle == nil)
        #expect(speak.isEnabled)
        let settings = try #require(app.menu.items.first { $0.title == "Settings" }?.submenu)
        #expect(settings.items.contains { $0.title == "Agent Voice: Choose a voice" })
        #expect(try #require(settings.items.first { $0.title == "Memory" }?.submenu).items.contains { $0.title == "Unload All" && $0.isEnabled })
        try await service.unload(id: "kokoro-82m")
        await app.table.refresh(); await app.refreshMenuState()
        #expect(!(titles(app.menu).contains("Speak Clipboard")))
        // Unloaded, Speak still opens its window: a request loads the model.
        #expect(app.menu.items.contains { $0.title == "Speak\u{2026}" && $0.isEnabled && $0.badge == nil })
    }

    /// A model an agent runs off the table is listed in the memory lines and unloaded by Unload All.
    @Test func offTableResidentIsListedAndUnloadedByUnloadAll() {
        var offTable = Fixtures.rows(.loaded).first { $0.id == "kokoro-82m" }!
        offTable.onTable = false
        let memory = MemorySnapshot(totalMB: 137_439, availableMB: 86_900, models: ["kokoro-82m": WorkerMemory(footprintMB: 410)])
        #expect(MenuContent.memoryLines(memory, rows: [offTable]).contains("Kokoro 82M \u{00b7} 0.4 GB"))
        #expect(MenuContent.loadedIDs([offTable]) == ["kokoro-82m"])
        var unloaded = offTable
        unloaded.states = unloaded.states.mapValues { state in
            var state = state
            state.state = .ready
            return state
        }
        #expect(MenuContent.loadedIDs([unloaded]).isEmpty)
    }

    @Test func menuContentLines() {
        let rows = Fixtures.rows(.loaded)
        let memory = MemorySnapshot(totalMB: 137_439, availableMB: 86_900, models: ["kokoro-82m": WorkerMemory(footprintMB: 410)])
        #expect(MenuContent.memoryLines(memory, rows: rows) == ["Room for models: 52.9 GB", "Kokoro 82M \u{00b7} 0.4 GB", "Qwen3.8 27B"])
        #expect(MenuContent.summary(rows) == "Arsenal: 2 models loaded")
        // A runner still up for a model no longer in the table contributes no action.
        let actions = [
            LoadedAction(id: "k-speak", modelID: "kokoro-82m", title: "Speak Clipboard", op: "speak"),
            LoadedAction(id: "gone-speak", modelID: "removed-voice", title: "Speak Clipboard", op: "speak")
        ]
        let plain = actions.map { LoadedMenuSection(id: $0.modelID, modelID: $0.modelID, title: $0.modelID, items: [$0]) }
        let groups = MenuContent.sections(plain, rows: rows)
        #expect(groups.flatMap(\.entries).map(\.action.id) == ["k-speak"])
        #expect(groups.first?.header == nil && groups.first?.entries.first?.subtitle == "Kokoro 82M")
    }

    /// v2.2 item 7: Refresh (in the Models table's header) re-reads through the service; a refusal says why in the
    /// table's footer line until the next Refresh.
    @Test func refreshRereadsAndARefusalSaysWhy() async throws {
        let service = FixtureService(rows: Fixtures.rows(.firstLaunch))
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        await service.setRefreshError(WorkerFailure(code: "busy", message: "Finish Get or Load before refreshing models."))
        app.rebuildMenu()
        let models = try #require(app.menu.items.first { $0.title == "Models" }?.submenu)
        #expect(models.items.count == 1)
        let host = try #require(models.items.first?.view)
        let window = NSWindow(contentRect: host.frame.offsetBy(dx: -9000, dy: -9000), styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.contentView = host
        defer { window.contentView = nil; window.close() }
        host.layoutSubtreeIfNeeded()
        let refresh = try #require(Self.find(RefreshControlView.self, in: host))
        #expect(refresh.accessibilityPerformPress())
        #expect(!app.table.refreshEnabled)
        try await waitUntil { app.refreshNotice != nil }
        app.rebuildMenu()
        #expect(app.table.refreshEnabled && app.table.refreshNotice == "Finish Get or Load before refreshing models.")
        // Nothing is busy any more, so the busy refusal is stale: Refresh is allowed again and the footer drops it.
        #expect(app.table.refreshAllowed && ModelTable.footer(app.table) == nil)
        #expect(models.items.count == 1)
        await service.setRefreshError(nil)
        app.refreshFiles()
        try await waitUntil { await service.refreshes > 0 }
        try await waitUntil { app.table.refreshEnabled }
        app.rebuildMenu()
        #expect(app.refreshNotice == nil && app.table.refreshNotice == nil && ModelTable.footer(app.table) == nil)
    }

    static func find<V: NSView>(_ type: V.Type, in view: NSView) -> V? {
        if let match = view as? V { return match }
        for sub in view.subviews { if let match = find(type, in: sub) { return match } }
        return nil
    }

    /// v2.2 item 8: a loaded model's declared section shows under its own title and goes when it unloads.
    @Test func aDeclaredSectionShowsWhileItsModelIsLoaded() async throws {
        let app = AppDelegate(service: FixtureService(rows: Self.downloaded), paths: ArsenalPaths(support: support))
        await app.table.refresh()
        var rows = app.table.rows
        rows[0].states[rows[0].precisions[0]] = PrecisionState(state: .loaded)
        app.table.apply(rows: rows, kinds: app.table.kinds)
        let transcribe = LoadedAction(id: "\(rows[0].id)-transcribe", modelID: rows[0].id, title: "Transcribe Clipboard Audio", op: "transcribe")
        app.previewState(
            actions: [transcribe], memory: MemorySnapshot(),
            sections: [LoadedMenuSection(id: "\(rows[0].id)-section-0", modelID: rows[0].id, title: "Transcription", items: [transcribe])])
        app.rebuildMenu()
        let shown = titles(app.menu)
        let visible = app.menu.items.filter { !$0.isHidden }
        let header = try #require(shown.firstIndex(of: "Transcription"))
        #expect(visible[header].isSectionHeader)
        #expect(shown[header + 1] == "Transcribe Clipboard Audio" && visible[header + 1].subtitle == nil)
        app.previewState(actions: [], memory: MemorySnapshot(), sections: [])
        app.rebuildMenu()
        #expect(!titles(app.menu).contains("Transcription"))
    }
}

@MainActor @Suite(.menuState, .serialized) final class ActionRunnerTests {
    @Test func clipboardGoesUnderTheOpsInputName() throws {
        let out = URL(fileURLWithPath: "/tmp/job")
        func p(_ op: String) throws -> JSONObject {
            try ActionRunner.params(
                LoadedAction(id: "a", modelID: "m", title: "T", op: op, input: "clipboard", params: ["voice": "af_heart"]),
                clipboard: " hello ", outDir: out)
        }
        #expect(try p("speak")["text"] == "hello")
        #expect(try p("speak")["voice"] == "af_heart")
        // Speech is a file for the shared queue, not a stream.
        #expect(try p("speak")["stream"] == false && p("speak")["out_dir"] == "/tmp/job")
        #expect(try p("generate")["prompt"] == "hello")
        #expect(try p("generate")["out_dir"] == "/tmp/job")
        #expect(try p("chat")["messages"] == [["role": "user", "content": "hello"]])
        #expect(throws: (any Error).self) {
            try ActionRunner.params(
                LoadedAction(id: "a", modelID: "m", title: "T", op: "speak", input: "clipboard"),
                clipboard: "  ", outDir: out)
        }
    }

    @Test func resultLines() {
        let speak = LoadedAction(id: "a", modelID: "m", title: "Speak Clipboard", op: "speak")
        #expect(ActionRunner.summary(speak, ["ok": true, "duration_s": 4.21]) == "Spoke 4.2 s of audio")
        let image = LoadedAction(id: "b", modelID: "m", title: "Draw", op: "generate")
        #expect(ActionRunner.summary(image, ["images": [["path": "/x.png"]], "seconds": 4.0]) == "Made 1 image in 4.0 s; opened it")
    }

    @Test func runningAnActionThroughTheServiceCopiesAChatAnswer() async throws {
        let rows = [ModelRow(id: "c", name: "C", kind: "chat", states: ["bf16": .init(state: .loaded)])]
        let runner = ActionRunner(service: FixtureService(rows: rows), jobs: FileManager.default.temporaryDirectory)
        var copied: String?
        runner.readClipboard = { "Hi" }
        runner.writeClipboard = { copied = $0 }
        runner.run(LoadedAction(id: "c-chat", modelID: "c", title: "Answer Clipboard", op: "chat", input: "clipboard"))
        try await waitUntil { runner.running == nil && runner.lastResult != "Answer Clipboard\u{2026}" }
        #expect(copied == "Hello from Arsenal.")
        #expect(runner.lastResult == "Answer copied to the clipboard")
    }

    @Test func anEmptyClipboardSaysWhatToDo() {
        let runner = ActionRunner(service: FixtureService(rows: []), jobs: FileManager.default.temporaryDirectory)
        runner.readClipboard = { nil }
        runner.run(LoadedAction(id: "s", modelID: "m", title: "Speak Clipboard", op: "speak", input: "clipboard"))
        #expect(runner.lastResult == "The clipboard has no text; copy some text first.")
    }
}
