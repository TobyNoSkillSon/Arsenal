// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore
import SceneKit
import SwiftUI

/// `ARSENAL_QA="render-capabilities <dir>"`: every capability window at its real default size (and its minimum), 2×
/// pixels, through the app's own API over the stub service with the capability fixtures (CapabilityFixture): the first
/// frame (nothing to work on yet), a run in progress, the result of a real run, a failure, a model not on this Mac, then
/// the changed windows (Make a Sound for a song model, Chat with attachments and a JSON answer, Remove Background with
/// a video) and the menu with the new entries. Writes cap-*.png and capability-renders.txt. No worker, no GPU.
@MainActor final class CapabilityRenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private let log = QALog()
    /// Only these windows (`ARSENAL_QA_ONLY="transcribe,select"`), for a quick look at one change.
    private let only = ProcessInfo.processInfo.environment["ARSENAL_QA_ONLY"].map { Set($0.split(separator: ",").map(String.init)) }
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        ToolWindow.keepOffscreenFrames = true
        SoundPlayer.silent = true
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            do { try await run(); exitCode = 0 } catch { log.say("render-capabilities FAILED: \(TableController.message(error))") }
            log.write(to: directory.appendingPathComponent("capability-renders.txt"))
            await set(.dark)
            QA.stopApp()
        }
    }

    private func set(_ mode: RenderAppearance) async {
        Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
        try? await Task.sleep(for: .milliseconds(100))
    }

    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-capability-render-\(UUID().uuidString)", isDirectory: true)
    private var port = 0
    private var service: FixtureService!

    private func run() async throws {
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        await set(.dark)
        let support = scratch.appendingPathComponent("support")
        service = await CapabilityFixture.service(support: support)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        guard let port = api.port else { throw QAFailure("the API did not start") }
        self.port = port
        let media = scratch.appendingPathComponent("media")
        let photo = try CapabilityFixture.photo(media.appendingPathComponent("Portrait on the beach.jpg"), width: 1200, height: 800)
        let screen = try CapabilityFixture.photo(media.appendingPathComponent("Screenshot.png"), width: 1440, height: 900)
        let clip = try CapabilityFixture.clip(media.appendingPathComponent("Team meeting.wav"))
        let song = try CapabilityFixture.clip(media.appendingPathComponent("Song demo.wav"), seconds: 3)
        let pdf = try CapabilityFixture.pdf(media.appendingPathComponent("Quarterly report.pdf"))
        let video = try CapabilityFixture.video(media.appendingPathComponent("Beach walk.mp4"))

        try await window(.transcribe, "whisper-fixture", TranscribeSession.self) { s in
            s.setSubject([clip]); s.set("diarize", .toggle(true)); s.set("language", .text("en"))
        }
        try await window(.separate, "demucs-fixture", SeparateSession.self) { s in
            s.setSubject([song]); s.set("stems", .text("vocals, drums, bass, other"))
        }
        try await window(.translate, "translate-fixture", TranslateSession.self) { s in
            s.set("text", .text("The meeting moves to Thursday at ten. Bring the quarterly numbers.")); s.set("target_lang", .text("de"))
        }
        try await window(.document, "ocr-fixture", DocumentSession.self) { s in s.setSubject([pdf]) }
        try await window(.extract, "extract-fixture", ExtractSession.self) { s in
            s.set("text", .text("Invoice 2041 from Northwind Traders, due 14 November, total 1,250.00 EUR."))
            s.set("json_schema", .text(SchemaField.schema([.init(name: "invoice", kind: .text), .init(name: "vendor", kind: .text), .init(name: "total", kind: .number)])))
        }
        try await window(.search, "rerank-fixture", SearchSession.self) { s in
            s.set("query", .text("how do I reset my password"))
            s.set(
                "documents",
                .text(
                    "Reset your password from Settings › Account.\nOur offices are closed on public holidays.\nPasswords must be 12 characters long.\nContact support to reset two-factor sign-in."
                ))
        }
        try await window(
            .select, "sam-fixture", SelectSession.self,
            fill: { s in
                s.setSubject([photo]); s.pointsBinding.wrappedValue = [SpatialPoint(x: 0.5, y: 0.55, label: "foreground"), SpatialPoint(x: 0.15, y: 0.2, label: "background")]
            },
            after: { s, controller in
                // A click on the shown selection: it stays on screen with no progress card while the model answers, then
                // the new mask replaces it in place; nothing joins the history until it is kept.
                let kept = s.outcomes.count
                await self.service.setDelay("segment", .milliseconds(1200))
                s.pointsBinding.wrappedValue += [SpatialPoint(x: 0.62, y: 0.4, label: "foreground")]
                try await self.until(seconds: 5) { s.running }
                try await self.capture(controller, "select-click-busy", keep: true, settle: 300)
                try await self.until(seconds: 10) { !s.running }
                await self.service.setDelay("segment", nil)
                try await self.capture(controller, "select-click-draft", keep: true, settle: 600)
                self.log.say("select click: quiet \(s.draft != nil), history \(kept) → \(s.outcomes.count)")
            })
        try await window(.find, "dino-fixture", FindSession.self) { s in
            s.setSubject([photo]); s.set("labels", .text("person, umbrella, towel"))
        }
        try await window(.depth, "depth-fixture", DepthSession.self) { s in
            s.setSubject([photo]); s.set("intrinsics", .toggle(true))
        }
        try await window(.upscale, "upscale-fixture", UpscaleSession.self) { s in
            s.setSubject([photo]); s.set("scale", .choice(0))
        }
        try await window(.smooth, "rife-fixture", SmoothSession.self) { s in
            s.setSubject([video]); s.set("factor", .text("4"))
        }
        try await window(.make3d, "trellis-fixture", Make3DSession.self) { s in s.setSubject([photo]) }
        // A splat model (IDOL): the window shows the model's render of the splat and its Gaussian count.
        if want("make3d") {
            CapabilityFixture.splatResult = true
            defer { CapabilityFixture.splatResult = false }
            try await window(.make3d, "trellis-fixture", Make3DSession.self, name: "make3d-splat", states: false) { s in s.setSubject([photo]) }
        }
        // Screen grounding: pixels converted to the picture.
        if want("find") {
            try await window(.find, "dino-fixture", FindSession.self, name: "find-ground", states: false) { s in
                s.select("ground_screen"); s.setSubject([screen]); s.set("instruction", .text("the Save button"))
            }
        }
        // A video selection on a later frame.
        if want("select") {
            try await window(.select, "sam-fixture", SelectSession.self, name: "select-video", states: false) { s in
                s.select("track_video"); s.setSubject([video]); _ = await VideoInfo.load(video)
                s.boxesBinding.wrappedValue = [SpatialBox(xmin: 0.2, ymin: 0.25, xmax: 0.55, ymax: 0.8)]
            }
        }
        // Live transcription, listening.
        if want("transcribe") { try await live() }
        if want("changes") { try await changes(photo: photo, video: video) }
        if want("menu") { try await menu() }
    }

    private func want(_ name: String) -> Bool { only.map { $0.contains(name) } ?? true }

    private func make<S: CapabilitySession>(_ tool: ToolKind, _ model: String, _: S.Type, choices: [ModelChoice]? = nil) -> (ToolWindowController, S) {
        let models =
            choices
            ?? CapabilityFixture.models.filter { ToolCatalog.windows(declared: Set($0.ops)).contains(tool) }.map {
                CapabilityFixture.choice($0.id, resident: true)
            }
        let port = self.port
        let (session, view) = CapabilityWindows.make(
            tool, models: models, selected: model, client: { ArsenalClient(port: port) }, inbox: scratch.appendingPathComponent("support/inbox"),
            output: scratch.appendingPathComponent("Results"), scratch: scratch.appendingPathComponent("window-\(tool.rawValue)"))
        let controller = ToolWindowController(title: tool.windowTitle, autosave: "", size: ToolLayout.size, minSize: ToolLayout.minSize, content: view)
        // swiftlint:disable:next force_cast
        return (controller, session as! S)
    }

    /// One window's states: the first frame, the subject given, a run in progress, the result (dark and light), the
    /// minimum size, a failure and a model that cannot run here.
    private func window<S: CapabilitySession>(
        _ tool: ToolKind, _ model: String, _ type: S.Type, name: String? = nil, states: Bool = true, fill: (S) async throws -> Void,
        after: (@MainActor (S, ToolWindowController) async throws -> Void)? = nil
    ) async throws {
        let name = name ?? tool.rawValue
        guard want(tool.rawValue) else { return }
        let (controller, session) = make(tool, model, type)
        defer { session.discard(); controller.close() }
        session.prepare()
        if states { try await capture(controller, "\(name)-opening", keep: true, settle: 40) }
        try await until { !session.loading }
        if states { try await capture(controller, "\(name)-empty", keep: true) }
        try await fill(session)
        if states { try await capture(controller, "\(name)-ready", keep: true) }
        await service.setDelay(session.operation ?? "", .milliseconds(1500))
        guard session.canRun else { throw QAFailure("\(name): cannot run (\(session.problem?.message ?? "not ready"))") }
        session.start()
        try await until(seconds: 8) { session.run?.step != nil || session.run?.phase != nil }
        if states { try await capture(controller, "\(name)-busy", keep: true) }
        try await until(seconds: 20) { !session.running }
        await service.setDelay(session.operation ?? "", nil)
        if let problem = session.problem { throw QAFailure("\(name): the run failed: \(problem.message)") }
        try await capture(controller, "\(name)-result-dark", keep: true, settle: 900)
        if let after { try await after(session, controller) }
        // SceneKit draws through Metal, which a view capture cannot read: the model's own view is snapshotted too.
        if let scene = Self.find(SCNView.self, in: controller.window?.contentView) {
            let image = scene.snapshot()
            if let tiff = image.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) {
                try png.write(to: directory.appendingPathComponent("cap-\(name)-scene.png"))
                log.say("wrote cap-\(name)-scene.png (SceneKit snapshot \(Int(image.size.width)) × \(Int(image.size.height)))")
            }
        }
        // The switches as the session holds them (an offscreen NSSwitch can draw its old state).
        log.say("\(name) switches: " + (session.form?.fields.filter { $0.control == .generic(.toggle) }.map { "\($0.key)=\(session.value($0.key))" } ?? []).joined(separator: ", "))
        log.say("\(name): " + session.figures().map { "\($0.title)=\($0.value ?? "-")" }.joined(separator: ", ") + "; options \(session.optionFields.map(\.key))")
        guard states else { return }
        await set(.light)
        try await capture(controller, "\(name)-result-light", keep: true, settle: 600)
        await set(.dark)
        try await capture(controller, "\(name)-min", keep: true, size: ToolLayout.minSize, settle: 600)
        // A failure, in place; then a model not on this Mac.
        let (failed, f) = make(tool, model, type)
        f.preview(
            operations: session.operations, selected: session.operation, values: session.values,
            problem: OperationProblem.from(ToolError("Worker connection closed.", status: 502, code: "worker_exit"), model: f.model?.name ?? "", task: f.stoppedNoun))
        try await capture(failed, "\(name)-error", settle: 400)
        f.discard()
        // The model not on this Mac, as ToolCenter words it for the window (its display name, its one repair).
        let center = ToolCenter(client: { nil }, playback: FixturePlayback())
        var missing = CapabilityFixture.row(model, state: .notInstalled)
        missing.sizeBytes = 1_340_000_000
        _ = await center.requestAdmissions(rows: [missing], service: service)
        let (blocked, b) = make(tool, model, type, choices: center.models([missing], for: tool))
        b.preview(operations: session.operations, selected: session.operation)
        try await capture(blocked, "\(name)-blocked", settle: 300)
        b.discard()
    }

    private func live() async throws {
        let (controller, session) = make(.transcribe, "whisper-fixture", TranscribeSession.self)
        defer { session.discard(); controller.close() }
        session.liveAvailable = true
        session.prepare()
        try await until { !session.loading }
        session.liveMode = true
        session.live.preview(
            lines: ["Thanks for coming in.", "Let's look at the numbers for the third quarter."], partial: "Revenue is up eight", listening: true, level: 0.45)
        try await capture(controller, "transcribe-live", settle: 500)
    }

    // MARK: Changed windows

    private func changes(photo: URL, video: URL) async throws {
        // Make a Sound with a song model: lyrics, language and instrumental from the music schema.
        let music = SoundSession(
            models: [ModelChoice(model: ToolModel(id: "song-fixture", name: "Song Fixture", kind: "sound", dtype: "fp16", mode: "exact"))], selected: "song-fixture",
            client: { nil }, inbox: scratch, outputDirectory: scratch)
        let schema = CapabilityFixture.schemaFromKind("sound", "generate_music")
        music.preview(
            operations: [
                OperationDescriptor(
                    name: "generate_sound", title: "Generate Sound", inputSchema: CapabilityFixture.schemaFromKind("sound", "generate_sound"), output: "audio", stream: true),
                OperationDescriptor(name: "generate_music", title: "Generate Music", inputSchema: schema, output: "audio", stream: true)
            ], selected: "generate_music",
            values: [
                "prompt": .text("Warm acoustic folk, finger-picked guitar"), "lyrics": .text("[verse]\nMorning light on the harbour wall\nGulls above the market stall"),
                "language": .text("en")
            ])
        let sound = ToolWindowController(title: "Make a Sound", autosave: "", size: SoundView.size, minSize: SoundView.minSize, content: SoundView(session: music))
        try await capture(sound, "sound-song", keep: true)
        music.set("instrumental", .toggle(true))
        try await capture(sound, "sound-song-instrumental")
        log.say("sound-song: fields \(music.optionFields.map { "\($0.key)" })")

        // Chat with a model whose schema takes pictures and a JSON answer.
        let chatModel = ToolModel(id: "vision-chat", name: "Vision Chat Fixture", kind: "chat", dtype: "int4", mode: "exact")
        let chat = ChatSession(model: chatModel, client: { nil })
        chat.preview(
            capabilities: ChatCapabilities(schema: CapabilityFixture.schemaFromKind("chat", "chat")), attachments: [photo], structured: true,
            schema: SchemaField.schema([.init(name: "people", kind: .integer), .init(name: "setting", kind: .text)]))
        chat.preview(
            messages: [
                ChatMessage(role: .user, text: "How many people are in this photo, and where is it?", attachments: [photo]),
                ChatMessage(role: .assistant, text: "{\"people\": 1, \"setting\": \"a beach at midday\"}", structured: true)
            ], run: nil, reference: .none, draft: "And the weather?")
        let chatWindow = ToolWindowController(title: "Chat", autosave: "", size: ChatView.size, minSize: ChatView.minSize, content: ChatView(session: chat))
        try await capture(chatWindow, "chat-attachments", size: ChatView.size)
        log.say("chat: images \(chat.capabilities.images), audio \(chat.capabilities.audioFormats), json \(chat.capabilities.jsonSchema)")
        // The same chat with a text-only model (qwen's narrowed schema): no paperclip, no braces.
        let plain = ChatSession(model: chatModel, client: { nil })
        plain.preview(capabilities: ChatCapabilities(schema: SoundFixture.schema("qwen3.8-27b", "chat")))
        plain.preview(messages: [], run: nil, reference: .none, draft: "")
        try await capture(
            ToolWindowController(title: "Chat", autosave: "", size: ChatView.size, minSize: ChatView.minSize, content: ChatView(session: plain)), "chat-text-only",
            size: ChatView.size)
        log.say("chat text-only: images \(plain.capabilities.images), json \(plain.capabilities.jsonSchema)")

        // Remove Background with a model that also tracks a selection through a video: a real job through the API, so
        // its figures are the run's.
        let row = CutoutFixture.row
        let bgSupport = scratch.appendingPathComponent("background-support")
        let bgService = FixtureService(
            rows: [row], metadata: CutoutFixture.metadata, supportDirectory: bgSupport,
            operations: [row.id: [CutoutFixture.operation, CapabilityFixture.operation("track_video")]])
        await CapabilityFixture.install(bgService)
        let bgAPI = APIHost(service: bgService, support: bgSupport, port: 0)
        await bgAPI.start()
        defer { bgAPI.stop() }
        guard let bgPort = bgAPI.port else { throw QAFailure("the background API did not start") }
        let background = BackgroundSession(
            models: [ModelChoice(model: ToolModel(row), resident: true)], selected: row.id, client: { ArsenalClient(port: bgPort) },
            inbox: bgSupport.appendingPathComponent("inbox"), outputDirectory: scratch.appendingPathComponent("Results"), scratch: scratch.appendingPathComponent("bg-window"))
        defer { background.discard() }
        background.prepare()
        try await until { !background.loading }
        background.setInput([video])
        _ = await VideoInfo.load(video)
        background.set("points", .text(SpatialValue.text([SpatialPoint(x: 0.42, y: 0.5, label: "foreground")])))
        let bg = ToolWindowController(
            title: "Remove Background", autosave: "", size: BackgroundView.size, minSize: BackgroundView.minSize, content: BackgroundView(session: background))
        try await capture(bg, "background-video", keep: true)
        background.start()
        try await until(seconds: 20) { !background.running }
        if let problem = background.problem { throw QAFailure("background video: \(problem.message)") }
        try await capture(bg, "background-video-result", settle: 900)
        log.say(
            "background video: mode \(background.videoMode), options \(background.optionFields.map(\.key)), figures "
                + background.figures().map { "\($0.title)=\($0.value ?? "-")" }.joined(separator: ", "))
    }

    // MARK: Menu

    private func menu() async throws {
        let support = scratch.appendingPathComponent("menu-support")
        let service = await CapabilityFixture.service(support: support)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        app.tools.playback = FixturePlayback()
        app.rebuildMenu()
        log.say("menu: " + app.menu.items.filter { !$0.isHidden }.map { $0.isSeparatorItem ? "\u{2014}" : $0.title }.joined(separator: " | "))
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(app.menu.items.filter { !$0.isHidden }, width: 340, to: directory.appendingPathComponent("cap-menu-dark.png")) { done.resume() }
        }
        app.table.stop()
    }

    // MARK: Capture

    static func find<T: NSView>(_ type: T.Type, in view: NSView?) -> T? {
        guard let view else { return nil }
        if let hit = view as? T { return hit }
        for sub in view.subviews { if let hit = find(type, in: sub) { return hit } }
        return nil
    }

    private func until(seconds: Double = 5, _ condition: @MainActor () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() > deadline { throw QAFailure("timed out") }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    /// The window's frame view (traffic lights included) at 2× pixels; `keep` leaves the window open for the next state.
    private func capture(_ controller: ToolWindowController, _ name: String, keep: Bool = false, size: NSSize = ToolLayout.size, settle: Int = 500) async throws {
        guard let window = controller.window, let root = window.contentView?.superview else { return }
        controller.applyAppearance()
        if window.frame.size != size { window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false) }
        if !window.isVisible {
            window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
            window.orderFrontRegardless()
            window.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        }
        try await Task.sleep(for: .milliseconds(settle))
        root.layoutSubtreeIfNeeded(); window.displayIfNeeded()
        let scale: CGFloat = 2
        guard
            let out = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: Int(root.bounds.width * scale), pixelsHigh: Int(root.bounds.height * scale), bitsPerSample: 8,
                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return }
        out.size = root.bounds.size
        root.cacheDisplay(in: root.bounds, to: out)
        try out.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("cap-\(name).png"))
        if !keep { window.orderOut(nil); controller.close() }
        log.say("wrote cap-\(name).png (\(Int(root.bounds.width)) × \(Int(root.bounds.height)))")
    }
}
