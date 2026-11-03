// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore
import SwiftUI

/// `ARSENAL_QA="render-sound <dir>"`: Make a Sound at its real size, 2× pixels. The opening frame, the first frame, a
/// typed prompt, a take being made and the take come from a real run through the app's own API over the stub service
/// (Stable Audio's and Kokoro's schemas read from their folders); takes with sound in them, voices, a clip to clone,
/// playing, failure and a model that cannot run are previews. Then the menu with the sound entries. Writes sound-*.png
/// and sound-renders.txt. No worker, no GPU; playback is silent.
@MainActor final class SoundRenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private let log = QALog()
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        ToolWindow.keepOffscreenFrames = true
        SoundPlayer.silent = true
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            do { try await run(); exitCode = 0 } catch { log.say("render-sound FAILED: \(TableController.message(error))") }
            log.write(to: directory.appendingPathComponent("sound-renders.txt"))
            await set(.dark)
            QA.stopApp()
        }
    }

    private func set(_ mode: RenderAppearance) async {
        Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
        try? await Task.sleep(for: .milliseconds(100))
    }

    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-sound-render-\(UUID().uuidString)", isDirectory: true)

    private func run() async throws {
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        await set(.dark)
        try await realRun()
        try await previews()
        try await menu()
    }

    private func window(_ session: SoundSession) -> ToolWindowController {
        ToolWindowController(title: "Make a Sound", autosave: "", size: SoundView.size, minSize: SoundView.minSize, content: SoundView(session: session))
    }

    // MARK: A real run through the API

    private func realRun() async throws {
        let service = SoundFixture.service()
        await service.setDelay("generate_sound", .milliseconds(1200))
        let support = scratch.appendingPathComponent("support")
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        guard let port = api.port else { throw QAFailure("the API did not start") }
        let session = SoundSession(
            models: SoundFixture.models, selected: "stable-audio-3-medium", client: { ArsenalClient(port: port) }, inbox: support.appendingPathComponent("inbox"),
            outputDirectory: scratch.appendingPathComponent("Music"))
        defer { session.close() }
        let controller = window(session)
        session.prepare()
        // The opening frame: the operations are still being read.
        try await capture(controller, "opening-dark", keep: true, settle: 30)
        log.say("opening: loading \(session.loading), operations \(session.operations.count), generate \(session.canRun)")
        try await until { !session.loading }
        try await capture(controller, "first-dark", keep: true)
        log.say("first frame: " + (session.form?.fields.map { "\($0.key) \($0.control)" }.joined(separator: "; ") ?? "none") + "; generate \(session.canRun)")
        session.set("prompt", .text("Heavy rain on a tin roof, distant thunder rolling in"))
        session.set("duration", .text("10"))
        try await capture(controller, "idle-dark", keep: true)
        session.start()
        try await until { session.run?.phase != nil }
        try await capture(controller, "generating-dark", keep: true)
        log.say("generating: \(session.run?.line(model: "Stable Audio 3 Medium") ?? "none")")
        try await until(seconds: 10) { !session.running }
        if let problem = session.problem { throw QAFailure("the take failed: \(problem.message)") }
        try await capture(controller, "real-result-dark", keep: true)
        log.say("take: " + (session.takes.last.map { "\($0.url.lastPathComponent) \(AudioShape.clock($0.shape.seconds)) seed \($0.seed ?? -1)" } ?? "none"))
        log.say("figures: " + session.figures().map { "\($0.title)=\($0.value ?? "-")" }.joined(separator: ", "))
        controller.close()
    }

    // MARK: Previews

    private func previews() async throws {
        let takes = scratch.appendingPathComponent("takes")
        let stable = SoundFixture.descriptors("stable-audio-3-medium")
        let series = [
            try SoundFixture.take(in: takes, seconds: 10, text: "Heavy rain on a tin roof, distant thunder rolling in", seed: 3_811_204_417, made: 0.45, wave: 1),
            try SoundFixture.take(in: takes, seconds: 6, text: "Old wooden door creaking open slowly", seed: 90_412, made: 0.31, wave: 2),
            try SoundFixture.take(in: takes, seconds: 4, text: "Footsteps on gravel, then a car door", seed: 7_741_009, made: 0.22, wave: 3),
            try SoundFixture.take(in: takes, seconds: 12, text: "Heavy rain on a tin roof, distant thunder rolling in", seed: 1_234_567, made: 0.51, wave: 4)
        ]
        func done(_ take: SoundTake) -> ToolRun {
            var r = ToolRun(model: take.model, operation: "generate_sound", started: Date(timeIntervalSinceNow: -take.seconds))
            r.finished = Date(); r.peakMB = 6_420
            return r
        }
        let values: [String: FieldValue] = ["prompt": .text(series[3].text), "duration": .text("12")]

        // The result, with earlier takes listed under it.
        let result = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: scratch, outputDirectory: scratch)
        result.preview(operations: stable, values: values, run: done(series[3]), takes: series)
        let resultWindow = window(result)
        try await capture(resultWindow, "result-dark", keep: true)
        for mode in [RenderAppearance.light, .translucent] {
            await set(mode)
            try await capture(resultWindow, "result-\(mode.rawValue)", keep: true, translucent: mode == .translucent)
        }
        await set(.dark)
        try await capture(resultWindow, "result-min-dark", keep: true, size: SoundView.minSize)
        // Playing: the playhead part-way (silent).
        try await capture(resultWindow, "result-dark-again", keep: true)
        result.player.play(series[3].url, from: 0.38)
        try await capture(resultWindow, "playing-dark", keep: true, settle: 300)
        log.say("playing: \(result.player.playing?.lastPathComponent ?? "none") at \(String(format: "%.2f", result.player.fraction))")
        result.player.stop()
        try? FileManager.default.removeItem(at: directory.appendingPathComponent("sound-result-dark-again.png"))
        resultWindow.close()

        // Making another take over the last one: the step bar over a scrim.
        let again = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: scratch, outputDirectory: scratch)
        var running = ToolRun(model: "stable-audio-3-medium", operation: "generate_sound", started: Date(timeIntervalSinceNow: -0.2))
        running.phase = "step"; running.step = 3; running.total = 8; running.memoryMB = 6_100
        again.preview(operations: stable, values: values, run: running, takes: Array(series.prefix(2)))
        try await both(window(again), "regenerating")
        // Loading the model first.
        let loading = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: scratch, outputDirectory: scratch)
        var load = ToolRun(model: "stable-audio-3-medium", operation: "generate_sound")
        load.phase = "loading"
        loading.preview(operations: stable, values: values, run: load)
        try await both(window(loading), "loading")

        // A voice: Kokoro's speech with its voices listed.
        let speech = [
            try SoundFixture.take(
                in: takes, seconds: 3.4, text: "The quick brown fox jumps over the lazy dog.", model: SoundFixture.kokoro, operation: "Speak", seed: nil, voice: "af_heart",
                rate: 24_000, channels: 1, made: 0.08, wave: 5)
        ]
        let voice = SoundSession(models: SoundFixture.models, selected: "kokoro-82m", client: { nil }, inbox: scratch, outputDirectory: scratch)
        var spoke = ToolRun(model: "kokoro-82m", operation: "speak", started: Date(timeIntervalSinceNow: -0.08))
        spoke.finished = Date(); spoke.memoryMB = 410
        voice.preview(
            operations: SoundFixture.descriptors("kokoro-82m"), values: ["text": .text(speech[0].text), "voice": .text("af_heart")], voices: SoundFixture.kokoroVoices,
            run: spoke, takes: speech)
        let voiceWindow = window(voice)
        try await capture(voiceWindow, "voice-dark", keep: true)
        await set(.light)
        try await capture(voiceWindow, "voice-light", keep: true)
        await set(.dark)
        voiceWindow.close()
        log.say("voice form: " + (voice.form?.fields.map { "\($0.key) \($0.control)" }.joined(separator: "; ") ?? "none"))

        // Not loaded yet: no voices to list, Generate on (it loads the model).
        let cold = SoundSession(
            models: [SoundFixture.model(SoundFixture.stableAudio), SoundFixture.model(SoundFixture.row("kokoro-82m", "Kokoro 82M", kind: "voices", tools: ["voice", "sound"]))],
            selected: "kokoro-82m", client: { nil }, inbox: scratch, outputDirectory: scratch)
        cold.preview(operations: SoundFixture.descriptors("kokoro-82m"), values: ["text": .text("Testing one two three.")])
        try await both(window(cold), "voice-not-loaded")

        // A clip to clone, with what it says (Qwen Base needs it).
        let qwen = SoundFixture.row("qwen3-tts-base", "Qwen3 Base 1.7B", kind: "voices", tools: ["voice", "sound"], loaded: true)
        let clip = try SoundFixture.take(in: takes, seconds: 5, text: "", model: qwen, seed: nil, rate: 24_000, channels: 1, wave: 6)
        let named = clip.url.deletingLastPathComponent().appendingPathComponent("my voice sample.wav")
        try FileManager.default.moveItem(at: clip.url, to: named)
        let clone = SoundSession(models: [SoundFixture.model(qwen)], selected: qwen.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        clone.newVoice = true // New Voice… chosen: Speak shows its clip to clone
        clone.preview(
            operations: SoundFixture.descriptors("qwen3-tts-base"),
            values: ["text": .text("Hello, this should sound like me."), "reference": .files([named]), "reference.text": .text("This is a short sample of my voice.")],
            voices: [])
        try await both(window(clone), "clone")
        log.say("clone form: " + (clone.form?.fields.map { "\($0.key) \($0.control)" }.joined(separator: "; ") ?? "none"))

        // A failure, in place.
        let failed = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: scratch, outputDirectory: scratch)
        failed.preview(
            operations: stable, values: values, takes: [series[1]],
            problem: OperationProblem.from(
                ToolError.from(
                    status: 409,
                    body: try WireJSON.encode([
                        "code": "insufficient_memory",
                        "details": .object(LoadMemoryRefusal(model: "stable-audio-3-medium", name: "Stable Audio 3 Medium", precision: "bf16", neededGB: 6.4, freeGB: 3.1).details)
                    ])),
                model: "Stable Audio 3 Medium"))
        try await both(window(failed), "error")
        // A field's own problem, under it.
        let field = SoundSession(models: [SoundFixture.model(qwen)], selected: qwen.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        field.newVoice = true
        field.preview(
            operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello, this should sound like me."), "reference": .files([named])], voices: [],
            problem: .init(message: "Type what the clip says: this model needs its transcript.", action: nil, field: "reference.text"))
        try await both(window(field), "field-error")
        // The longest form at the smallest window: the transcript's problem scrolled into view, the run button below.
        let small = SoundSession(models: [SoundFixture.model(qwen)], selected: qwen.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        small.newVoice = true
        small.advancedOpen = true
        small.preview(
            operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello, this should sound like me."), "reference": .files([named])], voices: [])
        // Generate pressed with the clip and no transcript: the problem appears while the window is open.
        let smallWindow = window(small)
        try await capture(smallWindow, "field-error-min-before", keep: true, size: SoundView.minSize)
        small.start()
        try await capture(smallWindow, "field-error-min-dark", keep: true, size: SoundView.minSize)
        await set(.light)
        try await capture(smallWindow, "field-error-min-light", size: SoundView.minSize)
        await set(.dark)
        try? FileManager.default.removeItem(at: directory.appendingPathComponent("sound-field-error-min-before.png"))
        log.say("min field error: \(small.problem?.message ?? "none") (field \(small.problem?.field ?? "none"))")
        // A model that cannot run here.
        let blocked = SoundSession(models: SoundFixture.models, selected: "qwen3-tts-base", client: { nil }, inbox: scratch, outputDirectory: scratch)
        blocked.preview(operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello")])
        try await both(window(blocked), "blocked")
        try await uxPass(series: series, speech: speech, values: values, qwen: qwen, clip: named)
    }

    /// The UX pass (30 Oct): the three task windows, a model not downloaded with its Download as the run button, the
    /// Advanced fold closed and open, Details off and on, a voice being made, and Cancel in the memory alert.
    private func uxPass(series: [SoundTake], speech: [SoundTake], values: [String: FieldValue], qwen: ModelRow, clip: URL) async throws {
        let stable = SoundFixture.descriptors("stable-audio-3-medium")
        func task(_ t: SoundTask, _ models: [ModelChoice], _ selected: String) -> SoundSession {
            SoundSession(models: models, selected: selected, client: { nil }, inbox: scratch, outputDirectory: scratch, task: t)
        }
        // Speak: Kokoro with its voices.
        let speak = task(.speak, SoundFixture.models, "kokoro-82m")
        speak.preview(
            operations: SoundFixture.descriptors("kokoro-82m"), values: ["text": .text(speech[0].text), "voice": .text("af_heart")], voices: SoundFixture.kokoroVoices,
            takes: speech)
        try await both(windowTitled(speak, "Speak"), "task-speak")
        // Make Music and Make a Sound from the same model.
        let music = task(.music, [SoundFixture.model(SoundFixture.stableAudio)], "stable-audio-3-medium")
        music.preview(operations: stable, values: ["prompt": .text("Slow piano over soft rain, warm tape hiss")])
        try await both(windowTitled(music, "Make Music"), "task-music")
        let effects = task(.sound, [SoundFixture.model(SoundFixture.stableAudio)], "stable-audio-3-medium")
        effects.preview(operations: stable, values: values, takes: [series[3]])
        try await both(windowTitled(effects, "Make a Sound"), "task-sound")
        log.say(
            "tasks: speak \(speak.operations.map(\.name)) \(speak.runVerb); music \(music.operations.map(\.name)) \(music.runVerb); sound \(effects.operations.map(\.name)) \(effects.runVerb)"
        )
        // Details on: the figures above the form (off is every other render).
        let details = ToolDetails.shown
        ToolDetails.shown = true
        let figures = task(.sound, [SoundFixture.model(SoundFixture.stableAudio)], "stable-audio-3-medium")
        var done = ToolRun(model: "stable-audio-3-medium", operation: "generate_sound", started: Date(timeIntervalSinceNow: -series[3].seconds))
        done.finished = Date(); done.peakMB = 6_420
        figures.preview(operations: stable, values: values, run: done, takes: series)
        try await both(windowTitled(figures, "Make a Sound"), "details-on")
        ToolDetails.shown = details
        // Not downloaded: chosen, its reason, and the run button is its Download.
        var models = SoundFixture.models
        models[2].unavailable = .init(message: "Qwen3 Base 1.7B isn\u{2019}t on this Mac yet.", action: .download, actionTitle: "Download 3.4 GB")
        let missing = task(.speak, models, "qwen3-tts-base")
        missing.preview(operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello")], voices: [])
        try await both(windowTitled(missing, "Speak"), "not-downloaded")
        // Advanced closed (the default) and open: Qwen's samplers and seed.
        var ready = SoundFixture.model(qwen); ready.unavailable = nil
        let advanced = task(.speak, [ready], qwen.id)
        advanced.preview(operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello, this should sound like me.")], voices: [])
        let advancedWindow = windowTitled(advanced, "Speak")
        try await capture(advancedWindow, "advanced-closed-dark", keep: true)
        advanced.advancedOpen = true
        try await capture(advancedWindow, "advanced-open-dark", keep: true)
        await set(.light)
        try await capture(advancedWindow, "advanced-open-light")
        await set(.dark)
        // A voice being made: the clip and what it says, the Voice picker on New Voice….
        let making = task(.speak, [ready], qwen.id)
        making.newVoice = true
        making.preview(
            operations: SoundFixture.descriptors("qwen3-tts-base"),
            values: ["text": .text("Hello, this should sound like me."), "reference": .files([clip]), "reference.text": .text("This is a short sample of my voice.")],
            voices: [])
        try await both(windowTitled(making, "Speak"), "new-voice")
        // Cancel in the memory alert: the run did not happen; the earlier take stays at full strength and the run button
        // is Make Sound Anyway (slower).
        let declined = task(.sound, [SoundFixture.model(SoundFixture.stableAudio)], "stable-audio-3-medium")
        let short = LoadMemoryRefusal(model: "stable-audio-3-medium", name: "Stable Audio 3 Medium", precision: "bf16", neededGB: 6.4, freeGB: 3.1)
        declined.preview(operations: stable, values: values, takes: [series[1]], problem: OperationProblem.declinedRun(short))
        try await both(windowTitled(declined, "Make a Sound"), "declined")
    }

    private func windowTitled(_ session: SoundSession, _ title: String) -> ToolWindowController {
        ToolWindowController(title: title, autosave: "", size: SoundView.size, minSize: SoundView.minSize, content: SoundView(session: session))
    }

    // MARK: Menu

    private func menu() async throws {
        let support = scratch.appendingPathComponent("menu-support")
        let service = FixtureService(rows: SoundFixture.rows, metadata: SoundFixture.metadata, supportDirectory: support, operations: SoundFixture.allOperations)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        app.tools.playback = FixturePlayback()
        app.rebuildMenu()
        log.say("menu: " + app.menu.items.filter { !$0.isHidden }.map { $0.isSeparatorItem ? "—" : $0.title }.joined(separator: " | "))
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(app.menu.items.filter { !$0.isHidden }, width: 320, to: directory.appendingPathComponent("sound-menu-dark.png")) { done.resume() }
        }
        app.table.stop()
    }

    // MARK: Capture

    /// The state in dark, then light; the window closes after.
    private func both(_ controller: ToolWindowController, _ name: String) async throws {
        try await capture(controller, name + "-dark", keep: true)
        await set(.light)
        try await capture(controller, name + "-light")
        await set(.dark)
    }

    private func until(seconds: Double = 5, _ condition: @MainActor () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() > deadline { throw QAFailure("timed out") }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    /// The window's frame view (traffic lights included) at 2× pixels; `keep` leaves the window open for the next state.
    private func capture(
        _ controller: ToolWindowController, _ name: String, keep: Bool = false, translucent: Bool = false, size: NSSize = SoundView.size, settle: Int = 500
    ) async throws {
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
        var rep = out
        if translucent, let composed = V21RenderDelegate.compose(rep, over: V21RenderDelegate.blurred(V21RenderDelegate.darkPicture(), size: size)) { rep = composed }
        try rep.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("sound-\(name).png"))
        if !keep { window.orderOut(nil); controller.close() }
        log.say("wrote sound-\(name).png (\(Int(root.bounds.width)) × \(Int(root.bounds.height)))")
    }
}
