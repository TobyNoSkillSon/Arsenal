// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal
import ArsenalCore

/// The v3 menu: tool sections of loaded models (C1), the voice section (C7), Memory's When memory is full (C6), the
/// update item (C8), "Show in menu", and the activity panel beside the table (C3). Menu items
/// are clicked the way NSMenu does it (`performActionForItem`); nothing is shown, no window opens, the clipboard is
/// not touched.
@MainActor @Suite(.menuState, .serialized) final class V3MenuTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-menu-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// The chat model under a fixture's name: its rows are checked by the names this test gives them, never a shipped model's.
    static func chatFixture(_ rows: [ModelRow]) -> [ModelRow] {
        rows.map { row in
            guard row.id == "qwen3.8-27b" else { return row }
            var renamed = row
            renamed.name = "Chat Fixture"
            return renamed
        }
    }

    private func app(_ rows: [ModelRow] = V3MenuTests.chatFixture(Fixtures.rows(.loaded))) async -> (AppDelegate, FixtureService) {
        let service = FixtureService(rows: rows)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        // Each menu fixture owns its playback state; other suites and environment overrides cannot mute it.
        app.tools.playback = FixturePlayback()
        app.tools.present = { _ in }
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        return (app, service)
    }
    private func settings(_ app: AppDelegate) throws -> NSMenu { try #require(app.menu.items.first { $0.title == "Settings" }?.submenu) }
    private func index(_ app: AppDelegate, _ title: String, after header: String? = nil) throws -> Int {
        let items = app.menu.items
        let start = header.flatMap { h in items.firstIndex { $0.title == h } } ?? 0
        return try #require(items[start...].firstIndex { $0.title == title }, "\(title) missing")
    }
    private func click(_ menu: NSMenu, _ i: Int) {
        #expect(menu.items[i].isEnabled, "\(menu.items[i].title) is disabled")
        menu.performActionForItem(at: i)
    }

    @Test func loadedModelsWithToolsGetTheirSectionsAndChatOpensItsWindow() async throws {
        let (app, _) = await app()
        let titles = app.menu.items.map { $0.isSeparatorItem ? "—" : $0.title }
        let qwen = try #require(app.table.rows.first { $0.id == "qwen3.8-27b" })
        // Tasks, not models: no model name heads a section; Speak and Chat lead the menu, above Models.
        #expect(!titles.contains(qwen.name) && !titles.contains("Kokoro 82M"))
        #expect(titles.contains("Speak\u{2026}") && titles.contains("Chat\u{2026}") && titles.contains("Stop Speaking"))
        #expect(try index(app, "Chat\u{2026}") < (try index(app, "Models")))
        let settings = try settings(app)
        #expect(settings.items.contains { $0.title == "Agent Voice: Choose a voice" } && settings.items.contains { $0.title == "Mute Agent Speech" })
        click(app.menu, try index(app, "Chat\u{2026}"))
        #expect(app.tools.chats[qwen.id] != nil)
        // Opening it again brings the same window back.
        click(app.menu, try index(app, "Chat\u{2026}"))
        #expect(app.tools.chats.count == 1)
        app.tools.closeAll()
    }

    @Test func showInMenuHidesAModelsToolSection() async throws {
        let (app, service) = await app()
        let qwen = try #require(app.table.rows.first { $0.id == "qwen3.8-27b" })
        let toggle = try #require(app.table.settings[qwen.id]?.first { $0.id == TableController.toolsKey })
        #expect(toggle.title == "Show Chat in menu" && toggle.value == .bool(true) && toggle.control == .toggle && toggle.tooltip.contains("Chat"))
        app.table.set(qwen, TableController.toolsKey, .bool(false))
        try await waitUntil { app.table.rows.first { $0.id == qwen.id }?.toolsShown == false }
        #expect(await service.calls.contains("tools qwen3.8-27b false"))
        // The items stay in the menu, hidden: the toggle changes no structure.
        try await waitUntil { !app.menu.items.contains { $0.title == "Chat\u{2026}" && !$0.isHidden } }
        #expect(app.menu.items.contains { $0.title == "Chat\u{2026}" && $0.isHidden })
        // Unloaded: Speak stays and opens its window; there is no Load step.
        try await service.unload(id: "kokoro-82m")
        await app.table.refresh(); await app.refreshMenuState(); app.rebuildMenu()
        #expect(app.menu.items.contains { $0.title == "Speak\u{2026}" && $0.isEnabled && !$0.isHidden })
        #expect(!app.menu.items.contains { $0.title.contains("Load") })
    }

    /// The "Show in menu" switch in the model's settings menu: AppKit's hit-test at its centre reaches the switch
    /// (nothing covers it) and its click turns the tool section off.
    @Test func theShowInMenuSwitchTakesAPlainClick() async throws {
        let (app, service) = await app()
        let c = app.table
        app.menuWillOpen(app.menu)
        let qwen = try #require(c.rows.first { $0.id == "qwen3.8-27b" })
        // The switch lives in the model's settings card beside its gear (never shown here).
        let settings = SettingsCardHost(c, models: app.modelsMenu)
        await settings.open(qwen)
        let view = settings.view
        let window = try #require(settings.window)
        defer { settings.close() }
        func switches(_ v: NSView) -> [MenuSwitch] { (v as? MenuSwitch).map { [$0] } ?? v.subviews.flatMap(switches) }
        let toggle = try #require(switches(view).first { $0.toolTip?.contains("main menu shows") == true })
        let centre = toggle.convert(NSPoint(x: toggle.bounds.midX, y: toggle.bounds.midY), to: nil)
        let frame = try #require(window.contentView?.superview)
        let hit = frame.hitTest(frame.convert(centre, from: nil))
        #expect(hit === toggle || hit?.isDescendant(of: toggle) == true, "the click reaches \(hit.map { String(describing: type(of: $0)) } ?? "nothing")")
        #expect(toggle.state == .on)
        toggle.performClick(nil)
        try await waitUntil { await service.calls.contains("tools qwen3.8-27b false") }
        try await waitUntil { c.rows.first { $0.id == qwen.id }?.toolsShown == false }
        // The open menu follows at once (the items are hidden, none removed); no test-driven refresh/rebuild or app
        // restart may rescue the result.
        try await waitUntil { !app.menu.items.contains { $0.title == "Chat\u{2026}" && !$0.isHidden } }
        #expect(!app.structurePending)
        app.menuDidClose(app.menu)
        #expect(!app.menu.items.contains { $0.title == "Chat\u{2026}" && !$0.isHidden })
        app.table.set(try #require(c.rows.first { $0.id == qwen.id }), TableController.toolsKey, .bool(true))
        try await waitUntil { app.menu.items.contains { $0.title == "Chat\u{2026}" && !$0.isHidden } }
    }

    @Test func voiceSectionMutesStopsAndChoosesAVoiceWithFailuresInPlace() async throws {
        let (app, _) = await app()
        let playback = FixturePlayback()
        app.tools.playback = playback
        app.tools.preview(
            voices: ["kokoro-82m": [.init(id: "af_heart", name: "Heart", language: "American English"), .init(id: "bm_lewis", name: "Lewis", language: "")]],
            current: ["kokoro-82m": "af_heart"])
        app.rebuildMenu()
        let stop = app.menu.items[try index(app, "Stop Speaking")]
        #expect(stop.subtitle == "Claude Code \u{00b7} Kokoro 82M \u{00b7} Heart" && !stop.isHidden)
        var settings = try settings(app)
        let mute = try #require(settings.items.firstIndex { $0.title == "Mute Agent Speech" })
        #expect(settings.items[mute].state == .off)
        settings.performActionForItem(at: mute)
        #expect(playback.muted)
        app.rebuildMenu()
        settings = try self.settings(app)
        #expect(settings.items.first { $0.title == "Mute Agent Speech" }?.state == .on)
        click(app.menu, try index(app, "Stop Speaking"))
        #expect(!playback.playing)
        app.rebuildMenu()
        // Nothing speaks: Stop Speaking hides.
        #expect(app.menu.items[try index(app, "Stop Speaking")].isHidden)
        // A voice chosen without the API: the old voice stays and the reason is a red line in the menu.
        let voice = try #require(settings.items.first { $0.title == "Agent Voice: Heart" }?.submenu)
        #expect(voice.items.map { $0.isSeparatorItem ? "—" : $0.title } == ["American English \u{00b7} Female", "Heart", "Male", "Lewis"] && voice.items[1].state == .on)
        #expect(VoiceChoice(id: "af_heart", name: "Heart", language: "").group == "f" && VoiceChoice(id: "am_adam", name: "Adam", language: "").group == "m")
        voice.performActionForItem(at: 3)
        try await waitUntil { app.tools.errors["kokoro-82m"] != nil }
        #expect(app.tools.currentVoice["kokoro-82m"] == "af_heart")
        app.rebuildMenu()
        let line = try #require(app.menu.items.first { $0.title == app.tools.errors["kokoro-82m"] })
        #expect(!line.isEnabled && (line.attributedTitle?.attribute(.foregroundColor, at: 0, effectiveRange: nil) as? NSColor) == .systemRed)
    }

    @Test func memoryWhenFullChoosesRefuseOrWaitInQueue() async throws {
        let (app, service) = await app()
        let memory = try #require(try settings(app).items.first { $0.title == "Memory" }?.submenu)
        let header = try #require(memory.items.firstIndex { $0.title == MenuContent.whenFullHeader })
        #expect(memory.items[header + 1].title == "Refuse right away" && memory.items[header + 1].state == .on)
        #expect(memory.items[header + 2].title == "Wait in queue" && memory.items[header + 2].state == .off)
        memory.performActionForItem(at: header + 2)
        try await waitUntil { await service.memorySettings().whenFull == .queue }
        #expect(await service.calls.contains("memory queue"))
        app.rebuildMenu()
        let again = try #require(try settings(app).items.first { $0.title == "Memory" }?.submenu)
        #expect(again.items[header + 2].state == .on && again.items[header + 1].state == .off)
    }

    @Test func theOrangeUpdateItemInstallsAndAFailureRetries() async throws {
        let (app, _) = await app()
        #expect(!app.menu.items.contains { $0.title.hasPrefix("Update") })
        let updater = FixtureUpdater()
        app.updater = updater
        app.rebuildMenu()
        #expect(app.menu.items[0].title == "Update to 3.1" && app.menu.items[1].isSeparatorItem)
        #expect((app.menu.items[0].attributedTitle?.attribute(.foregroundColor, at: 0, effectiveRange: nil) as? NSColor) == .systemOrange)
        click(app.menu, 0)
        #expect(updater.installs == 1)
        updater.updateState = .downloading(progress: 0.42)
        app.rebuildMenu()
        #expect(app.menu.items[0].title == "Downloading update \u{00b7} 42 %" && !app.menu.items[0].isEnabled)
        updater.updateState = .failed(reason: "The download's checksum did not match.")
        app.rebuildMenu()
        #expect(app.menu.items[0].toolTip?.hasPrefix("The download's checksum") == true)
        click(app.menu, 0)
        #expect(updater.checks == 1)
    }

    /// An updater relaunch is acknowledged before the service (and its catalog verification) is created: the receipt
    /// is on disk when the factory starts. A launch without the update argument writes nothing.
    @Test func theUpdateRelaunchIsAcknowledgedBeforeTheServiceOpens() async throws {
        let paths = ArsenalPaths(support: support)
        let token = UUID().uuidString
        var seenAtCreation: Bool?
        let (service, error) = await LaunchDelegate.open(
            paths: paths, arguments: ["Arsenal", "--arsenal-update", token],
            make: { p in
                seenAtCreation = FileManager.default.fileExists(atPath: UpdateAcknowledgement.url(support: p.support, token: token).path)
                return (FixtureService(rows: []), nil)
            })
        #expect(seenAtCreation == true && error == nil && service is FixtureService)
        let plain = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-noack-\(UUID().uuidString)")
        _ = await LaunchDelegate.open(paths: ArsenalPaths(support: plain), arguments: ["Arsenal"], make: { _ in (FixtureService(rows: []), nil) })
        #expect(!FileManager.default.fileExists(atPath: plain.path))
    }

    // MARK: Activity panel

    /// The panel is its own item, below Models and above Keep Hot, opening as a submenu; the Models
    /// submenu holds the table alone. The panel subscribes only while its submenu is open.
    @Test func activityIsItsOwnItemBelowModelsAndSubscribesOnlyWhileOpen() async throws {
        let (app, _) = await app()
        let titles = app.menu.items.map(\.title)
        let models = try index(app, "Models"), activityIndex = try index(app, "Activity")
        #expect(activityIndex == models + 1 && titles[activityIndex + 1] == "Settings", "\(titles)")
        let item = app.menu.items[activityIndex]
        #expect(item.image != nil && item.isEnabled)
        let activity = app.activityMenu.activity
        activity.source = FixedActivitySource(feeds: [V3RenderDelegate.busyFeed()])
        #expect(activity.subscriptions == 0)
        // Opening the Models submenu no longer subscribes anything.
        let table = try #require(app.menu.items[models].submenu)
        table.delegate?.menuWillOpen?(table)
        #expect(activity.subscriptions == 0 && !activity.visible)
        table.delegate?.menuDidClose?(table)
        #expect(try #require(table.items.first?.view).frame.width == ModelTable.width(app.table))
        // The Activity submenu, opened as AppKit opens it (its delegate's menuWillOpen).
        let submenu = try #require(item.submenu)
        #expect(submenu.items.count == 1 && (submenu.delegate as? MenuTrackingDelegate)?.original === app.activityMenu)
        submenu.delegate?.menuWillOpen?(submenu)
        try await waitUntil { activity.feed.models.first?.requests.count == 2 }
        #expect(activity.subscriptions == 1 && activity.visible)
        try await Task.sleep(for: .milliseconds(100))
        let view = try #require(submenu.items.first?.view)
        #expect(view.frame.width == ActivityPanel.width && submenu.minimumWidth == ActivityPanel.width)
        // As tall as its three cards, never cut: the content's own height fits the item view.
        #expect(view.frame.height >= view.fittingSize.height.rounded(.down) && view.frame.height > ActivityPanel.minHeight, "\(view.frame.size) vs \(view.fittingSize)")
        submenu.delegate?.menuDidClose?(submenu)
        #expect(!activity.visible)
    }

    @Test func activityReadsC3AndNamesWhatEachModelIsDoing() {
        let json: [String: Any] = [
            "models": [
                [
                    "id": "qwen3.8-27b", "kind": "chat", "state": "decode", "queued": 1, "batch": 2.0, "memory_mb": 15_420,
                    "requests": [
                        ["requester": "Claude Code", "op": "chat", "phase": "decode", "prompt_tokens": 8214, "generated_tokens": 412, "tps": 30.8, "timing": ["decode_tps": 30.8]],
                        ["requester": "Codex", "op": "chat", "phase": "prefill", "prompt_tokens": 48_120, "cached_tokens": 2048]
                    ]
                ],
                ["id": "flux2-klein-4b", "kind": "images", "state": "step", "requests": [["requester": "Arsenal", "op": "generate", "phase": "step", "step": 3, "total": 4]]]
            ],
            "machine": [
                "gpu_busy_pct": 94, "cpu_busy_pct": 38, "cpu_temp_c": 52.4, "gpu_temp_c": 61.0, "memory_used_mb": 61_200, "memory_total_mb": 137_439, "arsenal_mb": 310,
                "thermal": "fair"
            ]
        ]
        let feed = ActivityFeed(json: json)
        #expect(feed.models.map(\.id) == ["qwen3.8-27b", "flux2-klein-4b"])
        let m = feed.machine
        #expect(m.gpuBusyPct == 94 && m.cpuBusyPct == 38 && m.cpuTempC == 52.4 && m.gpuTempC == 61 && m.memoryUsedMB == 61_200 && m.arsenalMB == 310)
        #expect(ActivityText.celsius(m.cpuTempC) == "52 \u{00b0}C" && ActivityText.celsius(nil) == "\u{2014}" && ActivityText.percent(m.gpuBusyPct) == "94")
        let qwen = feed.models[0], flux = feed.models[1]
        #expect(qwen.queued == 1 && qwen.batch == 2 && qwen.memoryMB == 15_420)
        // Chat: each stream's own figure; the header adds the total only when several streams write.
        #expect(ActivityText.summary(qwen, .chat, resident: true) == nil)
        var two = qwen; two.requests[1].phase = "decode"; two.requests[1].timing = ["decode_tps": 20]
        #expect(ActivityText.summary(two, .chat, resident: true)?.text == "50.8 tok/s")
        #expect(ActivityText.detail(qwen.requests[0], .chat) == "30.8 tok/s")
        #expect(ActivityText.detail(qwen.requests[1], .chat) == "Reading 46K tokens")
        var reading = qwen.requests[1]; reading.step = 19; reading.total = 50
        #expect(ActivityText.detail(reading, .chat) == "Reading" && ActivityText.showsBar(reading, .chat))
        // Image: step x of y with its bar.
        #expect(
            ActivityText.detail(flux.requests[0], .image) == "3 / 4 steps" && ActivityText.fraction(flux.requests[0]) == 0.75 && ActivityText.showsBar(flux.requests[0], .image))
        #expect(ActivityText.summary(flux, .image, resident: true) == nil)
        // Voice: "Speaking" once. A card listing its request ("Claude Code · Speaking") leaves the header to its memory
        // figure (round 1 said it twice); with no line to say it, the header does.
        let speaking = ActivityFeed.Model(
            id: "kokoro", kind: "voice", state: "resident", requests: [.init(id: "v", requester: "Claude Code", op: "speak", phase: "running")], queued: 0, batch: nil,
            running: 1)
        #expect(ActivityText.lines(speaking).count == 1 && ActivityText.summary(speaking, .voice, resident: true) == nil)
        var unlisted = speaking; unlisted.requests[0].phase = "loading"
        #expect(ActivityText.lines(unlisted).isEmpty && ActivityText.summary(unlisted, .voice, resident: true)?.text == "Speaking")
        // Memory: models, Arsenal, other, free.
        let parts = ActivityText.memoryParts(machine: m, models: [MemorySegment(id: "qwen3.8-27b", name: "Qwen", mb: 15_420, colour: .blue)])
        #expect(parts.map(\.name) == ["Qwen", "App", "Other", "Free"])
        let expected: [Double] = [15_420, 310, 45_470, 76_239]
        #expect(parts.map(\.mb) == expected)
        #expect(abs(parts.reduce(0) { $0 + $1.fraction } - 61_200.0 / 137_439) < 1e-9)
        #expect(ActivityText.memoryParts(machine: .init(), models: []).isEmpty)
        // The legend adds up to the header: the models together, the app, everything else make "used"; free the rest.
        let legend = ActivityText.legend(parts, models: [MemorySegment(id: "qwen3.8-27b", name: "Qwen", mb: 15_420, colour: .blue)])
        #expect(legend.map(\.name) == ["Models", "App", "Other", "Free"])
        #expect(legend.map { ActivityText.gb($0.mb) ?? "" } == ["14.4", "0.3", "42.3", "71.0"] && ActivityText.gb(61_200) == "57.0")
        // Rounded one by one, models, app and other would print 48.9 under a 48.8 header: other takes up the rounding.
        let idle = ActivityFeed.Machine(memoryUsedMB: 52_400, memoryTotalMB: 137_439, arsenalMB: 310)
        let segments = [MemorySegment(id: "a", name: "A", mb: 1_180, colour: .green), MemorySegment(id: "b", name: "B", mb: 15_420, colour: .orange)]
        let printed = ActivityText.legend(ActivityText.memoryParts(machine: idle, models: segments), models: segments).dropLast().map { ActivityText.gb($0.mb) ?? "" }
        #expect(printed == ["15.5", "0.3", "33.0"] && ActivityText.gb(52_400) == "48.8")
        // Sections: the folder's own, else its tool's, else its kind's, else the default.
        var row = ModelRow(id: "x", name: "X", kind: "sound")
        #expect(ActivitySection(row: row, kind: "sound") == .default)
        row.tools = ["voice"]
        #expect(ActivitySection(row: row, kind: "sound") == .voice)
        row.activity = "image"
        #expect(ActivitySection(row: row, kind: "sound") == .image)
        #expect(ActivitySection(row: nil, kind: "chat") == .chat)
        // Unknown values stay unknown.
        #expect(ActivityFeed(json: [:]).machine == .init())
    }

    /// Numbers changing never resizes the panel; a request line arriving or leaving does.
    @Test func onlyArrivalsAndDeparturesResize() {
        let model = ActivityModel()
        var feed = V3RenderDelegate.busyFeed()
        let before = model.layout(feed)
        feed.models[0].requests[0].generatedTokens = 999; feed.machine.gpuBusyPct = 3; feed.models[0].memoryMB = 1
        #expect(model.layout(feed) == before)
        feed.models[0].requests.removeFirst()
        #expect(model.layout(feed) != before)
        let loading = ActivityFeed.Request(id: "r", requester: "Codex", op: "chat", phase: "loading")
        #expect(ActivityText.detail(loading, .chat, resident: true) == "Starting" && ActivityText.detail(loading, .chat) == "Loading")
    }

    /// A model's colour is its own (hashed from its id): the same whatever else is loaded, never shared, never blue.
    @Test func modelColoursAreStableDistinctAndNotTheAccent() throws {
        let one = ActivityColours.assign(["qwen3.8-27b", "flux2-klein-4b", "kokoro-82m", "layers-fixture"])
        let reordered = ActivityColours.assign(["layers-fixture", "kokoro-82m", "qwen3.8-27b", "flux2-klein-4b"])
        #expect(one == reordered)
        #expect(Set(one.values).count == 4)
        #expect(!one.values.contains(ArsenalPalette.dark.accent) && !ActivityColours.palette.contains(ArsenalPalette.rgb(0x4C8DFF)))
        #expect(ActivityColours.assign(["qwen3.8-27b"])["qwen3.8-27b"] == one["qwen3.8-27b"] || ActivityColours.hash("qwen3.8-27b") >= 0)
        #expect(ActivityColours.other(.light) != ActivityColours.other(.dark))
        // Design review #8: settled over every installed model, a model's colour does not change when another loads.
        let ids = (0..<40).map { "model-\($0)" }
        let pair = ids.indices.lazy.flatMap { i in ids.indices.filter { $0 > i }.map { (ids[i], ids[$0]) } }
            .first { ActivityColours.hash($0.0) % ActivityColours.palette.count == ActivityColours.hash($0.1) % ActivityColours.palette.count }
        let (a, b) = try #require(pair)
        let installed = [a, b, "kokoro-82m"]
        #expect(ActivityColours.assign([b], over: installed)[b] == ActivityColours.assign([a, b], over: installed)[b])
        #expect(ActivityColours.assign([a, b], over: installed)[a] != ActivityColours.assign([a, b], over: installed)[b])
    }

    /// Design review r2 #2, #3, #5, #6, #7: the wait counts what is left of the running request; the legend's Models is
    /// its cards' printed figures added up; light mode darkens a pale meter; a busy bar sits on its printed percent; the
    /// minute's average wait reads as a past average.
    @Test func waitLegendMeterAndBarsAgreeWithWhatIsPrinted() {
        let now = Date()
        // A decision runner 80 ms into a 192 ms request with 40 behind it: 112 ms + 40 × 192 ms = 7.79 s.
        let running = ActivityFeed.Request(
            id: "d", requester: "Codex", op: "judge", phase: "prefill", startedAt: now.addingTimeInterval(-7.5), runningSince: now.addingTimeInterval(-0.08))
        let decisions = ActivityFeed.Model(
            id: "laya", kind: "decisions", state: "prefill", requests: [running], queued: 40, batch: nil, memoryMB: 1_180, running: 1,
            recent: .init(finished: 250, meanRunS: 0.192, meanWaitS: 7.4, busyS: 48, busy: []))
        #expect(abs((ActivityText.waitS(decisions, now: now) ?? 0) - 7.792) < 1e-6)
        #expect(ActivityText.countsHelp(decisions, now: now) == "Wait: 112 ms left on the running one, then 40 waiting \u{00d7} 192 ms each.")
        // An image at step 3 of 4 with two waiting: a quarter of 6.1 s left, then 2 × 6.1 s = 13.7 s ("14 s", not "12 s").
        let step = ActivityFeed.Request(id: "i", requester: "Arsenal", op: "generate", phase: "step", step: 3, total: 4, startedAt: now, runningSince: now)
        let image = ActivityFeed.Model(
            id: "flux", kind: "images", state: "step", requests: [step], queued: 2, batch: nil, memoryMB: nil, running: 1, recent: .init(finished: 2, meanRunS: 6.1))
        #expect(abs((ActivityText.waitS(image, now: now) ?? 0) - 13.725) < 1e-6)
        #expect(ActivityText.counts(image)?.last.map { "\($0.0) \($0.1)" } == "14 s wait")
        // A request past its average has nothing left to add.
        var late = decisions
        late.requests[0].runningSince = now.addingTimeInterval(-1)
        #expect(abs((ActivityText.waitS(late, now: now) ?? 0) - 7.68) < 1e-6)
        #expect(ActivityText.countsHelp(late, now: now) == "Wait: 40 waiting \u{00d7} 192 ms each.")
        // Without a reported start it counts from acceptance (7.5 s ago: nothing left); with neither, nothing is added.
        late.requests[0].runningSince = nil
        #expect(abs((ActivityText.waitS(late, now: now) ?? 0) - 7.68) < 1e-6)
        late.requests[0].startedAt = nil
        #expect(abs((ActivityText.waitS(late, now: now) ?? 0) - 7.68) < 1e-6)
        #expect(ActivityText.minuteHelp(decisions.recent!, .decision).contains("Each worked 192 ms and waited 7.4 s on average."))
        // Cards print 1.1, 14.4 and 0.4 GB: Models reads 15.9 (their sum), not 15.8 (the sum's rounding); Other absorbs it.
        let segments = [("laya", 1_180.0), ("qwen", 15_420.0), ("kokoro", 410.0)].map { MemorySegment(id: $0.0, name: $0.0, mb: $0.1, colour: .green) }
        #expect(segments.map { ActivityText.gb($0.mb) ?? "" } == ["1.1", "14.4", "0.4"])
        let machine = ActivityFeed.Machine(memoryUsedMB: 52_400, memoryTotalMB: 137_439, arsenalMB: 310)
        let legend = ActivityText.legend(ActivityText.memoryParts(machine: machine, models: segments), models: segments)
        #expect(legend.map { ActivityText.gb($0.mb) ?? "" } == ["15.9", "0.3", "32.6", "79.2"] && ActivityText.gb(52_400) == "48.8")
        // The progress meter's colour over light material is darker than the model's own; in dark it is the colour itself.
        let yellow = ActivityColours.palette[4]
        #expect(ActivityColours.ink(yellow, .dark) == yellow && ActivityColours.ink(yellow, .light) != yellow)
        // CPU and GPU bars fill to the whole percent printed beside them.
        #expect(ActivityText.percent(23.6) == "24" && ActivityText.busyFraction(23.6) == 0.24 && ActivityText.busyFraction(23.4) == 0.23)
        #expect(ActivityText.busyFraction(nil) == nil && ActivityText.busyFraction(140) == 1)
    }

    /// Every card leads with the same counts line: running, waiting and the wait, numbers strong; between one-at-a-time
    /// requests it reads Idle and keeps its place until the minute empties (design review #3, #6, #7).
    @Test func everyCardLeadsWithItsCountsAndKeepsItsHeight() {
        let model = ActivityModel()
        var decisions = ActivityFeed.Model(
            id: "laya", kind: "decisions", state: "prefill", requests: [], queued: 40, batch: nil, memoryMB: 1_180, running: 1,
            recent: .init(finished: 250, meanRunS: 0.192, busyS: 48.4, busy: Array(repeating: 1, count: 60)))
        #expect(ActivityText.counts(decisions)?.map { "\($0.0) \($0.1)" } == ["1 running", "40 waiting", "8 s wait"])
        #expect(ActivityText.countsHelp(decisions) == "Wait: 40 waiting \u{00d7} 192 ms each.")
        #expect(ActivityText.minute(decisions.recent!, .decision).text == "250 decisions/min \u{00b7} 192 ms each")
        let busy = model.layout(.init(models: [decisions]))
        decisions.running = 0; decisions.queued = 0
        #expect(ActivityText.counts(decisions)?.isEmpty == true && ActivityText.summary(decisions, .decision, resident: true) == nil)
        #expect(model.layout(.init(models: [decisions])) == busy)
        decisions.recent = nil
        #expect(ActivityText.counts(decisions) == nil && ActivityText.summary(decisions, .decision, resident: true)?.text == "Idle")
        #expect(model.layout(.init(models: [decisions])) != busy)
        // Chat and images: the same line, then their running requests; waiting ones are counted, not listed.
        let stream = ActivityFeed.Request(id: "s", requester: "Codex", op: "chat", phase: "decode", timing: ["decode_tps": 30.8])
        let waiting = ActivityFeed.Request(id: "w", requester: "Codex", op: "chat", phase: "queued")
        let chat = ActivityFeed.Model(
            id: "qwen", kind: "chat", state: "decode", requests: [stream, waiting], queued: 1, batch: 1, memoryMB: nil, running: 1,
            recent: .init(finished: 1, meanRunS: 15.8, tokens: 1_246, busyS: 41.7))
        #expect(ActivityText.counts(chat)?.map { "\($0.0) \($0.1)" } == ["1 running", "1 waiting", "16 s wait"])
        #expect(ActivityText.lines(chat).map(\.request.id) == ["s"])
        #expect(ActivityText.minute(chat.recent!, .chat).text == "1 reply/min \u{00b7} avg 29.9 tok/s")
        #expect(ActivityText.minute(.init(finished: 2, meanRunS: 6.1, steps: 11, images: 2), .image).text == "2 images/min \u{00b7} 6.1 s each")
        #expect(ActivityText.estimate(0.4) == "<1 s" && ActivityText.estimate(7.68) == "8 s" && ActivityText.estimate(150) == "3 min")
        // A request waiting for its model to load is waiting, and the header says the model is loading.
        let loading = ActivityFeed.Model(
            id: "flux", kind: "images", state: "loading", requests: [.init(id: "l", requester: "Arsenal", op: "generate", phase: "loading")], queued: 1, batch: nil,
            memoryMB: nil)
        #expect(ActivityText.counts(loading)?.map { "\($0.0) \($0.1)" } == ["1 waiting"] && ActivityText.lines(loading).isEmpty)
        #expect(ActivityText.summary(loading, .image, resident: false)?.text == "Loading")
    }

    /// Design review #10: a card over the menu's material stays a faint lift in light as in dark, never near-opaque.
    @Test func cardsStayTranslucent() {
        #expect(ModelActivityCard.fill(.light).alphaComponent <= 0.2 && ModelActivityCard.fill(.dark).alphaComponent <= 0.05)
    }

    /// Requesters: the client's own name (MCP client or X-Arsenal-Client), else Request 1, 2, … in arrival order (never an internal label).
    @Test func unnamedStreamsAreLetteredNeverGuessed() {
        let t = Date()
        let m = ActivityFeed.Model(
            id: "q", kind: "chat", state: "decode",
            requests: [
                .init(id: "1", requester: "API", op: "chat", phase: "decode", startedAt: t),
                .init(id: "2", requester: "Claude Code", op: "chat", phase: "decode", startedAt: t.addingTimeInterval(1)),
                .init(id: "3", requester: "API", op: "chat", phase: "prefill", startedAt: t.addingTimeInterval(2)),
                .init(id: "4", requester: "API", op: "chat", phase: "queued", startedAt: t.addingTimeInterval(3))
            ], queued: 1, batch: nil, memoryMB: nil)
        #expect(ActivityText.lines(m).map(\.name) == ["Request 1", "Claude Code", "Request 2"])
        #expect(ActivityText.lines(m).map(\.named) == [false, true, false])
    }

    @Test func bitWidthsComeFromDeclaredFormatsNeverFromDigitsInAName() {
        #expect(PrecisionColumn.of("exl3-5bpw") == nil)
        let exl3 = ModelFormat(id: "exl3-5bpw", bitsPerWeight: 5, source: .vendor, runners: ["native"])
        #expect(PrecisionColumn.of("exl3-5bpw", formats: [exl3]) == "4")
        #expect(PrecisionColumn.of("int4") == "4" && PrecisionColumn.of("q3_k") == "4" && PrecisionColumn.of("mxfp4") == "4")
        // 8-bit has no column (only 16-bit and int4): fp8 is recognised as 8-bit and not placed.
        #expect(PrecisionColumn.of("fp8") == nil && PrecisionColumn.isEightBit("fp8"))
        #expect(PrecisionColumn.of("bf16") == "16" && PrecisionColumn.of("fp32") == nil)
        let declared8 = ModelFormat(id: "oq8x", bitsPerWeight: 8.5, source: .vendor, runners: ["native"])
        #expect(PrecisionColumn.of("oq8x") == nil && PrecisionColumn.of("oq8x", formats: [declared8]) == nil)
        #expect(PrecisionColumn.isEightBit("oq8x", formats: [declared8]))
    }

    private func waitUntil(seconds: Double = 5, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            if Date() > deadline { Issue.record("timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
