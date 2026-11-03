// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore
import SwiftUI

/// `ARSENAL_QA="render-v3 <dir>"`: the v3 surfaces at real size (2× pixels) on the stub service, in the three
/// appearances: the chat window (finished reply, live reply, with and without a reference), the image window (result,
/// generating, reference images), the Models submenu with the activity panel (busy and idle), the main menu with tool
/// sections, the voice and Memory submenus, Qwen's settings window with "Show in menu". Fixture figures, no worker, no GPU.
/// Writes v3-*.png and v3-renders.txt.
@MainActor final class V3RenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private let log = QALog()
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        ToolWindow.keepOffscreenFrames = true
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            do { try await run(); exitCode = 0 } catch { log.say("render-v3 FAILED: \(TableController.message(error))") }
            log.write(to: directory.appendingPathComponent("v3-renders.txt"))
            await set(.dark)
            QA.stopApp()
        }
    }

    private func set(_ mode: RenderAppearance) async {
        Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
        try? await Task.sleep(for: .milliseconds(100))
    }
    private static let modes: [RenderAppearance] = [.dark, .light, .translucent]

    private func run() async throws {
        try await chat()
        try await image()
        let app = try await menuApp()
        try await models(app)
        try await menu(app)
        try await settings(app)
    }

    // MARK: Chat

    static let qwen = ToolModel(id: "qwen3.8-27b", name: "Qwen3.8 27B", kind: "chat", dtype: "int4", mode: "exact", contextLimit: 32_768)

    static func chatMessages() -> [ChatMessage] {
        [
            ChatMessage(role: .user, text: "Give me three tips for keeping a field notebook."),
            ChatMessage(
                role: .assistant,
                text:
                    "1. Date and place every page before you write anything else, so each note can be found and trusted later.\n2. Write what you saw before what you think it means; keep interpretation in the margin.\n3. Sketch, even badly: a quick drawing records proportions and positions that words skip."
            ),
            ChatMessage(role: .user, text: "", preset: PresetPrompt(tokens: 8192)),
            ChatMessage(
                role: .assistant,
                text:
                    "The notes follow daily life in a small harbour village through the seasons: the ferry, the market, the mill and the people who keep them running. Most entries record repairs, arrivals and the weather, and how the village prepares for winter."
            )
        ]
    }
    static func finishedRun(preset: PresetPrompt? = PresetPrompt(tokens: 8192)) -> ChatRun {
        var run = ChatRun(sent: Date(timeIntervalSinceNow: -6))
        run.preset = preset
        run.finish(
            usage: ["prompt_tokens": 8214, "completion_tokens": 96],
            timing: ["ttft_s": 1.84, "prefill_tps": 4610.0, "decode_tps": 31.2, "peak_mb": 15_420.0, "max_itl_ms": 41.0])
        return run
    }
    static let chatReference = ToolReference(
        match: .sameChip, chip: "M5 Max", date: "2026-10-02", metrics: ["ttft_s": 0.31, "prefill_tps": 410, "decode_tps": 31.5, "peak_mb": 14_750, "max_itl_ms": 38],
        suite: "arsenal-chat-suite-v1")
    /// A short typed question, comparable with the chat suite's reference.
    static func shortRun() -> ChatRun {
        var run = ChatRun(sent: Date(timeIntervalSinceNow: -4))
        run.finish(
            usage: ["prompt_tokens": 23, "completion_tokens": 61],
            timing: ["ttft_s": 0.36, "prefill_tps": 395.0, "decode_tps": 30.9, "peak_mb": 15_120.0, "max_itl_ms": 44.0])
        return run
    }

    private func chat() async throws {
        for mode in Self.modes {
            await set(mode)
            let session = ChatSession(model: Self.qwen, client: { nil })
            session.preview(messages: Self.chatMessages(), run: Self.finishedRun(), reference: .none)
            try await capture(window(session), "chat-\(mode.rawValue)", size: ChatView.size, translucent: mode == .translucent)
            log.say("chat \(mode.rawValue): presets " + PresetPrompt.all.map { "\($0.title)\(session.unavailable($0) == nil ? "" : " (off)")" }.joined(separator: " "))
        }
        await set(.dark)
        // With a measured reference on this chip.
        let referenced = ChatSession(model: Self.qwen, client: { nil })
        referenced.preview(messages: Array(Self.chatMessages().prefix(2)), run: Self.shortRun(), reference: .found(Self.chatReference))
        try await capture(window(referenced), "chat-reference-dark", size: ChatView.size)
        // Live: the reply is being written; the composer shows a 16K preset waiting.
        var live = ChatRun(sent: Date(timeIntervalSinceNow: -3.2))
        live.livePromptTokens = 41; live.liveGeneratedTokens = 87
        var messages = Array(Self.chatMessages().prefix(2))
        messages.append(ChatMessage(role: .user, text: "And how should I index it?"))
        messages.append(
            ChatMessage(
                role: .assistant, text: "Keep the first two pages free for an index. Each time a page holds something you will want again, add a line: the",
                streaming: true))
        let streaming = ChatSession(model: Self.qwen, client: { nil })
        streaming.preview(messages: messages, run: live, reference: .none, generating: true)
        try await capture(window(streaming), "chat-live-dark", size: ChatView.size)
        // Empty, as it opens: nothing attached, no benchmark prompts (they are Details).
        let empty = ChatSession(model: Self.qwen, client: { nil })
        empty.preview(messages: [], run: nil, reference: .none)
        try await capture(window(empty), "chat-empty-dark", size: ChatView.size)
        // Details on: the figures and the benchmark prompts, one waiting in the composer.
        let details = ToolDetails.shown
        ToolDetails.shown = true
        let bench = ChatSession(model: Self.qwen, client: { nil })
        bench.preview(messages: [], run: nil, reference: .none, preset: PresetPrompt(tokens: 4096))
        try await capture(window(bench), "chat-details-on-dark", size: ChatView.size)
        ToolDetails.shown = details
        // Hit-test: every preset, the send button and the new-chat button receive a plain click at real size.
    }

    private func window<V: View>(_ content: V, size: NSSize) -> ToolWindowController {
        ToolWindowController(title: "render", autosave: "", size: size, minSize: size, content: content)
    }
    private func window(_ session: ChatSession) -> ToolWindowController { window(ChatView(session: session), size: ChatView.size) }

    // MARK: Image

    static let flux = ToolModel(id: "flux2-klein-4b", name: "FLUX.2 Klein 4B", kind: "images", dtype: "bf16", mode: "exact")

    private func image() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-render-images", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let pictures = try (0..<3).map { try Self.picture(seed: $0, in: folder) }
        let results = pictures.enumerated().map { i, url in
            ImageResult(
                url: url, prompt: "A lighthouse on a rocky island at dusk, warm windows, long exposure sea", width: 1024, height: 1024, steps: 4,
                seed: 2_918_402 + i, seconds: 4.3)
        }
        let asked: JSONObject = ["width": 1024, "height": 1024, "steps": 4]
        var done = ToolRun(model: Self.flux.id, operation: "generate", started: Date(timeIntervalSinceNow: -4.3), input: asked)
        done.finished = Date(); done.step = 4; done.total = 4; done.memoryMB = 13_240
        let values: [String: FieldValue] = ["prompt": .text(results[2].prompt), "steps": .text("4")]
        for mode in Self.modes {
            await set(mode)
            let session = ImageSession(model: Self.flux, client: { nil }, outputDirectory: folder)
            session.preview(operations: [ImageFixture.generate], values: values, results: results, run: done)
            try await capture(window(ImageView(session: session), size: ImageView.size), "image-\(mode.rawValue)", size: ImageView.size, translucent: mode == .translucent)
        }
        await set(.dark)
        var running = ToolRun(model: Self.flux.id, operation: "generate", started: Date(timeIntervalSinceNow: -2.4), input: asked)
        running.step = 3; running.total = 4; running.memoryMB = 13_100; running.phase = "step"
        let generating = ImageSession(model: Self.flux, client: { nil }, outputDirectory: folder)
        generating.preview(operations: [ImageFixture.generate], values: values, results: Array(results.prefix(2)), run: running)
        try await capture(window(ImageView(session: generating), size: ImageView.size), "image-generating-dark", size: ImageView.size)
        // A model whose operation takes reference images (no folder declares one yet: fixture schema).
        let referencing = ImageSession(model: Self.flux, client: { nil }, outputDirectory: folder)
        referencing.preview(
            operations: [ImageFixture.withReferences()],
            values: ["prompt": .text("The same lighthouse in winter, snow on the rocks"), "steps": .text("4"), "reference_images": .files(Array(pictures.prefix(2)))],
            results: [results[0]], run: done)
        try await capture(window(ImageView(session: referencing), size: ImageView.size), "image-references-dark", size: ImageView.size)
        // Three images from one request (Images 3): the first shown, the batch in the strip; Copy acts on the shown one.
        let batch = ImageSession(model: Self.flux, client: { nil }, outputDirectory: folder)
        var three = done; three.input["n"] = 3
        batch.preview(operations: [ImageFixture.generate], values: values.merging(["n": .text("3")]) { _, b in b }, results: results, run: three)
        batch.shown = results[0].id
        try await capture(window(ImageView(session: batch), size: ImageView.size), "image-batch-dark", size: ImageView.size)
        // The smallest window, after a run that reported no memory (the figure says so).
        let small = ImageSession(model: Self.flux, client: { nil }, outputDirectory: folder)
        var unreported = done; unreported.memoryMB = nil
        small.preview(operations: [ImageFixture.generate], values: values, results: results, run: unreported)
        try await capture(window(ImageView(session: small), size: ImageView.minSize), "image-min-dark", size: ImageView.minSize)
        try await picker(folder: folder)
    }

    /// Make an Image's model pop-up open: the menu SwiftUI built for the picker, read from the window's own views.
    private func picker(folder: URL) async throws {
        let models = [
            ModelChoice(model: Self.flux),
            ModelChoice(
                model: ToolModel(id: "qwen-image", name: "Qwen Image", kind: "images", dtype: "bf16", mode: "exact"),
                unavailable: .init(message: "Qwen Image isn\u{2019}t on this Mac yet.", action: .download, actionTitle: "Download 41 GB"))
        ]
        let session = ImageSession(models: models, selected: Self.flux.id, client: { nil }, inbox: folder, outputDirectory: folder)
        session.preview(operations: [ImageFixture.generate], values: ["prompt": .text("A lighthouse")])
        let controller = window(ImageView(session: session), size: ImageView.size)
        guard let w = controller.window, let root = w.contentView else { return }
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: ImageView.size), display: false)
        w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await Task.sleep(for: .milliseconds(500))
        root.layoutSubtreeIfNeeded()
        func menus(_ v: NSView) -> [NSMenu] { (((v as? NSPopUpButton)?.menu ?? v.menu).map { [$0] } ?? []) + v.subviews.flatMap(menus) }
        let found = menus(root).first { $0.items.contains { $0.title.contains("Qwen Image") } }
        let items: [NSMenuItem]
        if let found {
            items = found.items.compactMap { $0.copy() as? NSMenuItem }
            log.say(
                "picker: the window's own menu, \(items.count) items: "
                    + items.map { "\($0.title)\($0.isEnabled ? "" : " (off)")\($0.state == .on ? " ✓" : "")" }.joined(separator: " | "))
        } else {
            // SwiftUI keeps the menu to itself until it opens: the same lines from the picker's own titles.
            items = models.map { m in
                let item = NSMenuItem(title: ModelPicker.title(m), action: nil, keyEquivalent: "")
                item.isEnabled = m.unavailable == nil; item.state = m.id == session.selected ? .on : .off
                return item
            }
            log.say("picker: SwiftUI's menu is not reachable offscreen; rendered the picker's own lines: " + items.map(\.title).joined(separator: " | "))
        }
        w.orderOut(nil); controller.close()
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(items, width: 340, to: directory.appendingPathComponent("v3-image-picker-dark.png")) { done.resume() }
        }
    }

    /// A stand-in result image (sky, sea, a lighthouse), drawn here so the render needs no model.
    static func picture(seed: Int, in folder: URL) throws -> URL {
        let size = NSSize(width: 1024, height: 1024)
        let image = NSImage(size: size, flipped: false) { r in
            let skies: [(NSColor, NSColor)] = [
                (NSColor(srgbRed: 0.98, green: 0.62, blue: 0.38, alpha: 1), NSColor(srgbRed: 0.28, green: 0.22, blue: 0.48, alpha: 1)),
                (NSColor(srgbRed: 0.95, green: 0.78, blue: 0.52, alpha: 1), NSColor(srgbRed: 0.36, green: 0.42, blue: 0.62, alpha: 1)),
                (NSColor(srgbRed: 0.99, green: 0.55, blue: 0.45, alpha: 1), NSColor(srgbRed: 0.18, green: 0.16, blue: 0.34, alpha: 1))
            ]
            let (low, high) = skies[seed % skies.count]
            NSGradient(starting: low, ending: high)?.draw(in: NSRect(x: 0, y: r.height * 0.38, width: r.width, height: r.height * 0.62), angle: 90)
            NSGradient(starting: NSColor(srgbRed: 0.08, green: 0.12, blue: 0.22, alpha: 1), ending: NSColor(srgbRed: 0.2, green: 0.26, blue: 0.4, alpha: 1))?
                .draw(in: NSRect(x: 0, y: 0, width: r.width, height: r.height * 0.4), angle: 90)
            NSColor(srgbRed: 0.1, green: 0.1, blue: 0.12, alpha: 1).setFill()
            NSBezierPath(ovalIn: NSRect(x: 380 + CGFloat(seed * 30), y: 300, width: 420, height: 160)).fill()
            NSColor(srgbRed: 0.92, green: 0.9, blue: 0.86, alpha: 1).setFill()
            NSBezierPath(rect: NSRect(x: 560 + CGFloat(seed * 30), y: 400, width: 56, height: 260)).fill()
            NSColor(srgbRed: 1, green: 0.86, blue: 0.5, alpha: 1).setFill()
            NSBezierPath(rect: NSRect(x: 552 + CGFloat(seed * 30), y: 660, width: 72, height: 40)).fill()
            return true
        }
        let url = folder.appendingPathComponent("render-\(seed).png")
        guard let tiff = image.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else {
            throw QAFailure("could not draw a picture")
        }
        try png.write(to: url)
        return url
    }

    // MARK: Menu and Models submenu

    static func busyFeed() -> ActivityFeed {
        let now = Date()
        return ActivityFeed(
            models: [
                .init(
                    id: "qwen3.8-27b", kind: "chat", state: "decode",
                    requests: [
                        .init(
                            id: "a", requester: "Claude Code", op: "chat", phase: "decode", promptTokens: 8214, cachedTokens: nil, generatedTokens: 412, step: nil, total: nil,
                            startedAt: now.addingTimeInterval(-15), tps: 30.8, timing: ["decode_tps": 30.8, "ttft_s": 1.2]),
                        .init(
                            id: "b", requester: "API", op: "chat", phase: "prefill", promptTokens: 48_120, cachedTokens: 2_048, generatedTokens: nil, step: nil,
                            total: nil, startedAt: now.addingTimeInterval(-4), tps: nil)
                    ], queued: 1, batch: 2.0, memoryMB: 15_420),
                .init(
                    id: "flux2-klein-4b", kind: "images", state: "step",
                    requests: [
                        .init(
                            id: "c", requester: "Arsenal", op: "generate", phase: "step", promptTokens: nil, cachedTokens: nil, generatedTokens: nil, step: 3, total: 4,
                            startedAt: now.addingTimeInterval(-3), tps: nil)
                    ], queued: 0, batch: nil, memoryMB: 13_240),
                .init(id: "kokoro-82m", kind: "voices", state: "idle", requests: [], queued: 0, batch: nil, memoryMB: 410)
            ],
            machine: .init(
                cpuBusyPct: 38, gpuBusyPct: 94, cpuTempC: 52.4, gpuTempC: 71.2, memoryUsedMB: 61_200, memoryTotalMB: 137_439, arsenalMB: 310))
    }

    /// Claude Code's request on Qwen: waiting behind a load, reading a 12K prompt, writing, done; then idle.
    static func sequence() -> [ActivityFeed] {
        let start = Date()
        func qwen(_ phase: String, prompt: Int? = nil, step: Int? = nil, total: Int? = nil, generated: Int? = nil, tps: Double? = nil, gpu: Double) -> ActivityFeed {
            var feed = busyFeed()
            feed.models[0].requests =
                phase == "idle"
                ? []
                : [
                    .init(
                        id: "seq", requester: "Claude Code", op: "chat", phase: phase, promptTokens: prompt, cachedTokens: nil, generatedTokens: generated, step: step,
                        total: total, startedAt: start, tps: tps, timing: tps.map { ["decode_tps": $0] } ?? [:])
                ]
            feed.models[0].state = phase; feed.models[0].queued = 0; feed.models[0].batch = nil
            feed.models[1].requests = []; feed.models[1].state = "idle"
            feed.machine = .init(cpuBusyPct: 9, gpuBusyPct: gpu, cpuTempC: 44, gpuTempC: 40 + gpu / 4, memoryUsedMB: 48_000, memoryTotalMB: 137_439, arsenalMB: 300)
            return feed
        }
        return [
            qwen("idle", gpu: 2), qwen("loading", gpu: 12), qwen("prefill", prompt: 12_288, step: 4, total: 12, gpu: 97),
            qwen("prefill", prompt: 12_288, step: 10, total: 12, gpu: 99), qwen("decode", prompt: 12_288, generated: 48, tps: 29.4, gpu: 100),
            qwen("decode", prompt: 12_288, generated: 212, tps: 29.1, gpu: 100), qwen("idle", gpu: 4)
        ]
    }

    private func menuApp() async throws -> AppDelegate {
        var rows = Fixtures.rows(.loaded)
        if let i = rows.firstIndex(where: { $0.id == "flux2-klein-4b" }) { rows[i].states["bf16"] = .init(state: .loaded) }
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-render-\(UUID().uuidString)")
        let app = AppDelegate(service: FixtureService(rows: rows), paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        let speak = LoadedAction(id: "kokoro-82m-speak", modelID: "kokoro-82m", title: "Speak Clipboard", op: "speak")
        app.previewState(
            actions: [speak],
            memory: MemorySnapshot(
                totalMB: 128_000, availableMB: 86_900,
                models: ["kokoro-82m": WorkerMemory(footprintMB: 410), "flux2-klein-4b": WorkerMemory(footprintMB: 13_240), "qwen3.8-27b": WorkerMemory(footprintMB: 15_420)]),
            sections: [LoadedMenuSection(id: "kokoro-82m", modelID: "kokoro-82m", title: "Kokoro 82M", items: [speak])])
        app.tools.preview(
            voices: ["kokoro-82m": [.init(id: "af_heart", name: "Heart", language: "American English"), .init(id: "bm_lewis", name: "Lewis", language: "British English")]],
            current: ["kokoro-82m": "af_heart"])
        app.tools.playback = FixturePlayback()
        app.updater = FixtureUpdater()
        log.say("menu rows: " + app.table.rows.map { "\($0.name) tools \($0.tools) shown \($0.toolsShown) loaded \($0.residentDType ?? "-")" }.joined(separator: "; "))
        return app
    }

    private func models(_ app: AppDelegate) async throws {
        let activity = app.activityMenu.activity
        for mode in Self.modes {
            await set(mode)
            activity.preview(Self.busyFeed())
            try await renderModels(app, "models-\(mode.rawValue)")
        }
        await set(.dark)
        var idle = Self.busyFeed()
        for i in idle.models.indices { idle.models[i].requests = []; idle.models[i].queued = 0; idle.models[i].batch = nil; idle.models[i].state = "idle" }
        idle.machine = .init(cpuBusyPct: 4, gpuBusyPct: 3, cpuTempC: 41, gpuTempC: 38, memoryUsedMB: 45_600, memoryTotalMB: 137_439, arsenalMB: 300)
        activity.preview(idle)
        try await renderModels(app, "models-idle-dark")
        // macOS reports no GPU statistics or thermal state: the strip says so instead of drawing empty values.
        activity.preview(ActivityFeed(models: idle.models, machine: .init()), sampled: true)
        try await renderModels(app, "models-unreported-dark")
        // A request's life as the panel shows it, one frame per snapshot (offscreen frame sequence).
        for (index, feed) in Self.sequence().enumerated() {
            activity.preview(feed)
            try await renderModels(app, String(format: "activity-sequence-%02d", index + 1))
        }
    }

    /// The Activity submenu's panel (it used to sit beside the table in the Models submenu).
    private func renderModels(_ app: AppDelegate, _ name: String) async throws {
        let size = await QA.renderActivity(app.activityMenu.activity, to: directory.appendingPathComponent("v3-\(name).png"))
        log.say("wrote v3-\(name).png (\(Int(size.width)) × \(Int(size.height)))")
    }

    private func menu(_ app: AppDelegate) async throws {
        await set(.dark)
        app.rebuildMenu()
        log.say("menu: " + app.menu.items.map { $0.isSeparatorItem ? "—" : $0.title + ($0.subtitle.map { " [\($0)]" } ?? "") }.joined(separator: " | "))
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("v3-menu-dark.png")) { done.resume() }
        }
        for (title, name) in [("Memory", "memory"), ("Voice", "voice")] {
            guard let sub = app.menu.items.first(where: { $0.title == title || (title == "Voice" && $0.title.hasPrefix("Voice:")) })?.submenu else {
                log.say("no submenu \(title)"); continue
            }
            log.say("\(title): " + sub.items.map { $0.isSeparatorItem ? "—" : ($0.state == .on ? "✓ " : "") + $0.title }.joined(separator: " | "))
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                MenuMock.render(sub.items, width: 260, to: directory.appendingPathComponent("v3-menu-\(name).png")) { done.resume() }
            }
        }
    }

    private func settings(_ app: AppDelegate) async throws {
        let c = app.table
        guard let qwen = c.rows.first(where: { $0.id == "qwen3.8-27b" }) else { return }
        for mode in [RenderAppearance.dark, .light] {
            await set(mode)
            c.openSettings(qwen)
            log.say("settings \(qwen.name): " + (c.settings[qwen.id] ?? []).map { "\($0.title)=\($0.display($0.value))" }.joined(separator: ", "))
            _ = try await SettingsRenderDelegate.render(c, row: qwen, to: directory.appendingPathComponent("v3-settings-\(mode.rawValue).png"))
            c.settingsCard.close()
        }
        await set(.dark)
    }

    // MARK: Capture

    /// The window's frame view (traffic lights included, so the title's indent can be checked) at 2× pixels.
    private func capture(_ controller: ToolWindowController, _ name: String, size: NSSize, translucent: Bool = false) async throws {
        guard let window = controller.window, let root = window.contentView?.superview else { return }
        controller.applyAppearance()
        window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
        window.orderFrontRegardless()
        window.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await Task.sleep(for: .milliseconds(700))
        root.layoutSubtreeIfNeeded(); window.displayIfNeeded()
        let scale: CGFloat = 2
        guard
            let out = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: Int(root.bounds.width * scale), pixelsHigh: Int(root.bounds.height * scale), bitsPerSample: 8,
                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return }
        out.size = root.bounds.size
        root.cacheDisplay(in: root.bounds, to: out)
        var rep = out
        if translucent, let composed = V21RenderDelegate.compose(rep, over: V21RenderDelegate.blurred(V21RenderDelegate.darkPicture(), size: size)) { rep = composed }
        try rep.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("v3-\(name).png"))
        window.orderOut(nil); controller.close()
        log.say("wrote v3-\(name).png")
    }
}

/// Speech playing for an agent (renders).
@MainActor final class FixturePlayback: PlaybackControlling {
    var muted = false
    var playing = true
    var current: (requester: String, model: String, voice: String?)? = ("Claude Code", "kokoro-82m", "af_heart")
    var onChange: () -> Void = {}
    func setMuted(_ muted: Bool) throws { self.muted = muted }
    func stop() { playing = false }
}

/// An offered release (renders).
@MainActor final class FixtureUpdater: AppUpdating {
    var updateState: AppUpdateState = .available(version: "3.1", notes: "", size: 38_000_000)
    var onChange: () -> Void = {}
    private(set) var installs = 0, checks = 0
    func installUpdate() { installs += 1 }
    func checkAgain() { checks += 1 }
}
