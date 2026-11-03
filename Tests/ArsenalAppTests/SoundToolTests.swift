// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

/// Make a Sound: its inputs from each audio operation's schema (the repository's model folders as they are today), the
/// request each makes, a take end to end through the app's own API over the stub service, the window's controls
/// clicked through the real event path on a never-shown window (as V3ToolTests does), and the menu entry opening the
/// one window with that model chosen. Playback runs at zero volume; nothing is shown or heard.
@MainActor @Suite(.menuState, .serialized) final class SoundToolTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        ControlFrames.recording = true
        SoundPlayer.silent = true
    }
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-sound-tests-\(UUID().uuidString)", isDirectory: true)
    deinit { try? FileManager.default.removeItem(at: scratch) }

    // MARK: Inputs from schemas

    @Test func stableAudioSchemaBecomesPromptLengthStepsGuidanceSeedAndAvoid() throws {
        let schema: JSONObject = [
            "type": "object", "required": ["prompt"], "additionalProperties": false,
            "properties": [
                "prompt": ["type": "string", "minLength": 1], "duration": ["type": "number", "exclusiveMinimum": 0, "maximum": 120, "default": 5],
                "steps": ["type": "integer", "minimum": 1, "maximum": 100, "default": 8], "seed": ["type": "integer", "minimum": 0],
                "guidance": ["type": "number", "minimum": 0, "maximum": 20, "default": 1], "negative_prompt": ["type": "string"],
                "out_dir": ["type": "string"], "stream": ["type": "boolean"]
            ]
        ]
        let form = ToolForm(schema: schema, operation: "generate_sound")
        #expect(form.fields.count == 6)
        #expect(form.fields.map(\.key) == ["prompt", "duration", "steps", "guidance", "seed", "negative_prompt"])
        #expect(form.fields[0].control == .main(placeholder: "Describe the sound"))
        #expect(form.fields[1].control == .seconds(min: 0, max: 120))
        #expect(form.fields[2].control == .generic(.integer(min: 1, max: 100, stepper: true)))
        #expect(form.fields[3].control == .generic(.number(min: 0, max: 20)))
        #expect(form.fields[4].control == .seed)
        #expect(form.fields[5].control == .note && form.fields[5].title == "Avoid")
        #expect(ToolForm(schema: schema, operation: "generate_music").main?.control == .main(placeholder: "Describe the music"))
        // Defaults from the schema, a model setting of the same name over them.
        #expect(form.initialValue(form.fields[1], setting: nil) == .text("5"))
        #expect(form.initialValue(form.fields[2], setting: .number(12)) == .text("12"))
        // Empty prompt: refused in the window, nothing sent.
        #expect(throws: OperationForm.Problem(key: "prompt", message: "Type something first.")) {
            try form.input([:], seed: 7, stage: { $0.path })
        }
        let values: [String: FieldValue] = ["prompt": .text("  rain on a tin roof "), "duration": .text("10"), "steps": .text("8"), "guidance": .text("1")]
        // An empty seed field sends the window's random seed, kept with the take.
        let input = try form.input(values, seed: 42, stage: { $0.path })
        #expect(input == ["prompt": "rain on a tin roof", "duration": 10, "steps": 8, "guidance": 1, "seed": 42])
        try OperationSchemaValidator.validate(.object(input), schema: .object(schema))
        #expect(throws: OperationForm.Problem(key: "duration", message: "Length must be more than 0 and at most 120 s.")) {
            try form.input(values.merging(["duration": .text("400")]) { _, b in b }, seed: 1, stage: { $0.path })
        }
        #expect(throws: OperationForm.Problem(key: "seed", message: "The seed must be a whole number from 0.")) {
            try form.input(values.merging(["seed": .text("-3")]) { _, b in b }, seed: nil, stage: { $0.path })
        }
    }

    @Test func speechSchemaGetsTextVoiceAClipToCloneAndHidesFixedValues() throws {
        let schema: JSONObject = [
            "type": "object", "required": ["text"],
            "properties": [
                "text": ["type": "string", "minLength": 1], "voice": ["type": "string"], "speed": ["type": "number", "const": 1],
                "language": ["type": "string", "enum": ["en", "zh"]], "instructions": ["type": "string"], "seed": ["type": "integer", "minimum": 0],
                "reference": [
                    "type": "object", "required": ["path", "text"], "properties": ["path": ["type": "string"], "text": ["type": "string", "minLength": 1]],
                    "additionalProperties": false
                ]
            ]
        ]
        let form = ToolForm(schema: schema, operation: "speak")
        #expect(form.fields.map(\.key) == ["text", "voice", "language", "seed", "reference", "instructions"])
        #expect(form.fixed == ["speed": 1])
        #expect(form.main?.control == .main(placeholder: "What should it say"))
        #expect(form.fields[1].control == .voice)
        #expect(form.fields[2].control == .generic(.choice(["en", "zh"])))
        #expect(form.fields[4].control == .reference(transcript: true) && form.fields[4].title == "Voice to clone")
        #expect(FieldControl.choiceTitle("zh") == "Chinese" && FieldControl.choiceTitle("en-GB") == "British English")
        // Plain lowercase words read as menu lines; anything else as it is.
        #expect(FieldControl.choiceTitle("english") == "English" && FieldControl.choiceTitle("birefnet-general") == "birefnet-general")
        let clip = URL(fileURLWithPath: "/tmp/me.wav")
        var values: [String: FieldValue] = ["text": .text("Hello there"), "voice": .text("af_heart"), "reference": .files([clip])]
        // The model needs the clip's transcript: the window says so under the field, before staging anything.
        #expect(throws: OperationForm.Problem(key: "reference.text", message: "Type what the clip says: this model needs its transcript.")) {
            try form.input(values, seed: nil, stage: { "/inbox/1/" + $0.lastPathComponent })
        }
        values["reference.text"] = .text("This is my voice.")
        let input = try form.input(values, seed: 9, stage: { "/inbox/1/" + $0.lastPathComponent })
        #expect(input == ["text": "Hello there", "voice": "af_heart", "speed": 1, "seed": 9, "reference": ["path": "/inbox/1/me.wav", "text": "This is my voice."]])
        try OperationSchemaValidator.validate(.object(input), schema: .object(schema))
    }

    /// Every audio operation in the repository's model folders: a main text, nothing the service manages, and the
    /// window's request from defaults plus a line of text is one the service's validator accepts.
    @Test func everyAudioOperationInTheModelFoldersMakesAValidRequest() throws {
        var checked: [String] = []
        for (model, ops) in SoundFixture.declared.sorted(by: { $0.key < $1.key }) {
            for op in ops where op.output == "audio" {
                let schema = SoundFixture.schema(model, op.name)
                let form = ToolForm(schema: schema, operation: op.name)
                let main = try #require(form.main, "\(model) \(op.name) has no main text")
                #expect(!form.fields.contains { ToolForm.hidden.contains($0.key) })
                var values: [String: FieldValue] = [:]
                for f in form.fields { values[f.key] = form.initialValue(f, setting: nil) }
                values[main.key] = .text("A short test line.")
                let input = try form.input(values, seed: 3, stage: { $0.path })
                // Conditional requirements (a voice or a description, say) are the validator's to explain; anything
                // else must pass as the window sends it.
                let conditional = ["anyOf", "oneOf", "allOf", "if", "dependentRequired"].contains { schema[$0] != nil }
                do { try OperationSchemaValidator.validate(.object(input), schema: .object(schema)) } catch {
                    #expect(conditional, "\(model) \(op.name): \(error)")
                }
                checked.append("\(model)/\(op.name)")
            }
        }
        // Counts first: three generate operations and five speak operations.
        #expect(checked.count == 8, "\(checked)")
    }

    // MARK: A take through the app's own API

    @Test func aTakeIsMadeSavedAndReadThroughTheAppsOwnAPI() async throws {
        let service = SoundFixture.service()
        await service.setDelay("generate_sound", .milliseconds(300))
        let support = scratch.appendingPathComponent("support")
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        let port = try #require(api.port)
        let music = scratch.appendingPathComponent("Music")
        let session = SoundSession(
            models: SoundFixture.models, selected: "stable-audio-3-medium", client: { ArsenalClient(port: port) }, inbox: support.appendingPathComponent("inbox"),
            outputDirectory: music)
        defer { session.close() }
        session.prepare()
        try await waitUntil { !session.loading }
        #expect(session.operations.map(\.name) == ["generate_sound", "generate_music"])
        #expect(session.operation == "generate_sound")
        // The inputs are the folder's: whatever it declares today, prompt first.
        let declared = ToolForm(schema: SoundFixture.schema("stable-audio-3-medium", "generate_sound"), operation: "generate_sound")
        #expect(session.form == declared && session.form?.fields.first?.key == "prompt")
        #expect(Set(["duration", "seed"]).isSubset(of: Set(declared.fields.map(\.key))))
        #expect(!session.canRun)
        session.set("prompt", .text("Thunder rolling over hills"))
        session.set("duration", .text("3"))
        #expect(session.canRun)
        session.start()
        #expect(session.running)
        try await waitUntil { !session.running }
        #expect(session.problem == nil)
        #expect(session.takes.count == 1)
        let take = try #require(session.takes.first)
        #expect(take.url.deletingLastPathComponent().resolvingSymlinksInPath() == music.resolvingSymlinksInPath() && take.url.pathExtension == "wav")
        #expect(abs(take.shape.seconds - 0.2) < 0.01 && take.shape.sampleRate == 24_000)
        #expect(take.text == "Thunder rolling over hills" && take.seed != nil)
        // The request carried what the window showed, with the seed it keeps.
        let sent = try #require(await service.runInputs.last)
        #expect(sent["prompt"] == "Thunder rolling over hills" && sent["duration"] == 3 && sent["seed"]?.number.flatMap(\.roundedInt) == take.seed)
        let figures = session.figures()
        #expect(figures.map(\.title).prefix(3) == ["Time", "Length", "vs real time"])
        #expect(figures[1].value == "0.20 s" && figures[0].value != nil && figures[2].value != nil)
        // Music keeps its own values; the text carries over.
        session.select("generate_music")
        #expect(session.value("prompt").text == "Thunder rolling over hills")
        let musicDefault = SoundFixture.schema("stable-audio-3-medium", "generate_music")["properties"]?["duration"]?["default"]?.number
        #expect(session.value("duration").text == musicDefault.map(OperationForm.number))
        // Another model: its own operation and inputs; the main text carries over to its text.
        session.choose("kokoro-82m")
        try await waitUntil { !session.loading && session.voices != nil }
        #expect(session.operations.map(\.name) == ["speak"])
        #expect(session.value("text").text == "Thunder rolling over hills")
        #expect(session.voices?.map(\.id) == ["af_heart"])
        // The menu's voice is the stub's only voice; an unknown one is the service's refusal, in its own words.
        session.set("voice", .text("zz_none"))
        session.start()
        try await waitUntil { !session.running }
        #expect(session.problem?.message.contains("Unknown stub voice") == true)
        #expect(session.takes.count == 1)
        session.set("voice", .text("af_heart"))
        session.start()
        try await waitUntil { !session.running }
        #expect(session.problem == nil && session.takes.count == 2 && session.takes.last?.voice == "af_heart")
        // A model that is not on this Mac can be chosen: the window then offers its Download.
        session.choose("qwen3-tts-base")
        #expect(session.selected == "qwen3-tts-base" && session.blocked != nil && !session.canRun)
        #expect(session.notRunnable == session.blocked?.message)
    }

    @Test func withoutTheAPITheFailureIsALineAndRetryRunsAgain() async throws {
        let session = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: scratch, outputDirectory: scratch)
        session.prepare()
        try await waitUntil { !session.loading }
        #expect(session.problem == .init(message: ChatSession.noAPI, action: .retry))
        #expect(!session.canRun)
    }

    // MARK: The window's controls

    private func open(_ session: SoundSession) async throws -> (ToolWindowController, NSWindow) {
        let wc = ToolWindowController(title: "test", autosave: "", size: SoundView.size, minSize: SoundView.size, content: SoundView(session: session))
        let w = try #require(wc.window)
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: SoundView.size), display: false)
        w.orderFrontRegardless()
        w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await settle(w)
        scope = wc.scope
        return (wc, w)
    }
    private func settle(_ w: NSWindow) async throws {
        w.contentView?.layoutSubtreeIfNeeded(); w.displayIfNeeded()
        try await Task.sleep(for: .milliseconds(150))
    }
    private var scope: UUID?
    private func centre(_ id: String, _ w: NSWindow) throws -> NSPoint {
        let f = try #require(scope.flatMap { ControlFrames.frames[$0]?[id] }, "\(id) was not laid out")
        #expect(f.width >= 14 && f.height >= 14, "\(id) is too small to tap: \(f.size)")
        let height = try #require(w.contentView).bounds.height
        return NSPoint(x: f.midX, y: height - f.midY)
    }
    /// `text`: a text field or editor, whose own AppKit text view rightly takes the click.
    private func expectReachable(_ id: String, _ w: NSWindow, scrolled: Bool = false, text: Bool = false) throws {
        let p = try centre(id, w)
        let frame = try #require(w.contentView?.superview)
        let view = frame.hitTest(frame.convert(p, from: nil))
        let host = try #require(w.contentView)
        let reached = view === host || (scrolled && view?.isDescendant(of: host) == true) || (text && (view is NSTextView || view is NSTextField))
        #expect(reached, "\(id): the click reaches \(view.map { String(describing: type(of: $0)) } ?? "nothing")")
    }
    private func tap(_ id: String, _ w: NSWindow, scrolled: Bool = false) async throws {
        try expectReachable(id, w, scrolled: scrolled)
        let p = try centre(id, w)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            let e = try #require(
                NSEvent.mouseEvent(
                    with: type, location: p, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: w.windowNumber, context: nil,
                    eventNumber: 0, clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0))
            w.sendEvent(e)
        }
        try await settle(w)
    }

    @Test func pillsPlayTakesSeedAndGenerateTakeAPlainTap() async throws {
        let session = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: scratch, outputDirectory: scratch)
        let takes = [
            try SoundFixture.take(in: scratch, seconds: 2, text: "Rain on a tin roof", seed: 11),
            try SoundFixture.take(in: scratch, seconds: 3, text: "A door creaks open", seed: 22, wave: 2)
        ]
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), values: ["prompt": .text("Rain")], takes: takes)
        session.advancedOpen = true // the seed and steps are under Advanced
        let (wc, w) = try await open(session)
        defer { wc.close(); session.close() }
        // First frame: the model's name, the last take shown; no Sound | Music pills (Make Music is its own menu entry, so
        // the window looks the same whichever entry opened it).
        #expect(session.operation == "generate_sound" && session.current?.id == takes[1].id)
        #expect(scope.flatMap { ControlFrames.frames[$0]?["op-generate_music"] } == nil)
        #expect(session.value("prompt").text == "Rain")
        // Play and stop the shown take; an earlier take from the list.
        try await tap("play", w)
        #expect(session.player.playing == takes[1].url)
        try await tap("play", w)
        #expect(session.player.playing == nil)
        try await tap("take-\(takes[0].id)", w, scrolled: true)
        #expect(session.current?.id == takes[0].id)
        try await tap("play-\(takes[0].id)", w, scrolled: true)
        #expect(session.player.playing == takes[0].url)
        // Clicking the waveform plays from there.
        try await tap("waveform", w)
        #expect(session.player.playing == takes[0].url)
        session.player.stop()
        // Reuse last: the shown take's seed.
        try await tap("seed-back", w, scrolled: true)
        #expect(session.value("seed").text == "11")
        // The model pop-up, the length field and Show in Finder are reachable (a menu or Finder would open: not clicked).
        for id in ["model", "reveal"] { try expectReachable(id, w) }
        #expect(ControlFrames.frames[wc.scope]?["reveal-result"] == nil, "one Show in Finder per window")
        try expectReachable("field-prompt", w, text: true)
        try expectReachable("field-duration", w, scrolled: true, text: true)
        // Generate without the API: the failure is a line in place, with Try Again.
        try await tap("generate", w)
        try await waitUntil { !session.running }
        #expect(session.problem == .init(message: ChatSession.noAPI, action: .retry))
        #expect(session.takes.count == 2)
    }

    /// Review round 1: the slider moves where it is clicked; the takes list shows from the first take; not enough memory
    /// offers Load Anyway, which runs again.
    @Test func theSliderTakesAClickOneTakeHasAListAndMemoryOffersItsAction() async throws {
        let session = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: scratch, outputDirectory: scratch)
        let take = try SoundFixture.take(in: scratch, seconds: 2, text: "Rain on a tin roof", seed: 11)
        let memory = OperationProblem.from(
            ToolError("Not enough memory for Stable Audio 3 Medium (needs 6.4 GB, 3.1 GB free).", status: 507, code: "insufficient_memory"),
            model: "Stable Audio 3 Medium")
        #expect(memory.action == .loadAnyway)
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), values: ["prompt": .text("Rain")], takes: [take], problem: memory)
        session.advancedOpen = true // guidance is under Advanced
        let (wc, w) = try await open(session)
        defer { wc.close(); session.close() }
        // The run button is Make Sound Anyway (slower) and runs again here (without the API, that run fails as its own
        // line); no menu opens and no second link offers it.
        var opened = false
        session.openModels = { opened = true }
        #expect(scope.flatMap { ControlFrames.frames[$0]?["action"] } == nil)
        try await tap("generate", w)
        try await waitUntil { !session.running }
        #expect(!opened && session.problem == .init(message: ChatSession.noAPI, action: .retry))
        // The guidance slider's middle is the middle of its range.
        let range = try #require(session.form?.fields.first { $0.key == "guidance" }).control
        guard case .generic(.number(let lo?, let hi?)) = range else { Issue.record("guidance is \(range)"); return }
        try await tap("field-guidance", w, scrolled: true)
        let set = try #require(Double(session.value("guidance").text))
        #expect(abs(set - (lo + hi) / 2) <= (hi - lo) / 50, "guidance \(set) of \(lo)...\(hi)")
        // While a take is made the slider is off, as the rest of the form is.
        var running = ToolRun(model: "stable-audio-3-medium", operation: "generate_sound")
        running.phase = "step"; running.step = 1; running.total = 8
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), values: ["prompt": .text("Rain"), "guidance": .text("4")], run: running, takes: [take])
        try await settle(w)
        let p = try centre("field-guidance", w)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            w.sendEvent(
                try #require(
                    NSEvent.mouseEvent(
                        with: type, location: p, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: w.windowNumber, context: nil,
                        eventNumber: 0, clickCount: 1, pressure: 1)))
        }
        try await settle(w)
        #expect(session.running && session.value("guidance").text == "4")
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), values: ["prompt": .text("Rain")], takes: [take])
        try await settle(w)
        // One take: the list shows it already, so it is plain that takes are kept; its Play plays.
        try await tap("play-\(take.id)", w, scrolled: true)
        #expect(session.player.playing == take.url)
        session.player.stop()
    }

    @Test func whatTheClipSaysAppearsOnceThereIsAClip() async throws {
        let qwen = SoundFixture.models[2]
        let session = SoundSession(models: [qwen], selected: qwen.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        session.preview(operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello")], voices: [])
        // The clip to clone in Speak itself shows after New Voice….
        session.newVoice = true
        let (wc, w) = try await open(session)
        defer { wc.close(); session.close() }
        #expect(ControlFrames.frames[try #require(scope)]?["field-reference.text"] == nil)
        let clip = try SoundFixture.take(in: scratch, seconds: 1, text: "", seed: nil).url
        session.addFiles([clip], to: "reference")
        try await settle(w)
        try expectReachable("field-reference.text", w, scrolled: true, text: true)
    }

    /// Review r1 (A1): the longest form at the smallest window keeps a field's problem in view above Generate, and the
    /// prompt gives up height before the options do.
    @Test func atTheSmallestSizeAFieldsProblemIsScrolledIntoView() async throws {
        let qwen = SoundFixture.models[2]
        var ready = qwen; ready.unavailable = nil
        let session = SoundSession(models: [ready], selected: qwen.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        session.preview(operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello")], voices: [])
        session.newVoice = true
        // The longest form: Advanced open and the figures shown.
        session.advancedOpen = true
        let details = ToolDetails.shown
        ToolDetails.shown = true
        defer { ToolDetails.shown = details }
        let clip = try SoundFixture.take(in: scratch, seconds: 1, text: "", seed: nil).url
        session.addFiles([clip], to: "reference")
        let wc = ToolWindowController(title: "test", autosave: "", size: SoundView.minSize, minSize: SoundView.minSize, content: SoundView(session: session))
        let w = try #require(wc.window)
        defer { wc.close(); session.close() }
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: SoundView.minSize), display: false)
        w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await settle(w)
        scope = wc.scope
        let frames = { ControlFrames.frames[wc.scope] ?? [:] }
        let prompt = try #require(frames()["field-text"])
        #expect(prompt.height >= MainEditor.minHeight - 1 && prompt.height < MainEditor.height, "the prompt gave up height: \(prompt.height)")
        session.start()
        #expect(session.problem?.field == "reference.text")
        try await settle(w)
        try await Task.sleep(for: .milliseconds(300))
        let transcript = try #require(frames()["field-reference.text"])
        let generate = try #require(frames()["generate"])
        let prompted = try #require(frames()["field-text"])
        #expect(transcript.maxY <= generate.minY && transcript.minY >= prompted.maxY, "the transcript \(transcript) is out of view (prompt \(prompted), Generate \(generate))")
    }

    @Test func aModelThatCannotRunHereSaysWhyAndGenerateIsOff() async throws {
        let blocked = SoundFixture.models[2]
        let session = SoundSession(models: SoundFixture.models, selected: blocked.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        session.preview(operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello")])
        let (wc, w) = try await open(session)
        defer { wc.close(); session.close() }
        #expect(session.blocked?.message == "Qwen3 Base 1.7B isn\u{2019}t on this Mac yet.")
        #expect(!session.canRun)
        // The run button is the Download; no second link offers it.
        var prepared: [String] = []
        session.prepareModel = { prepared.append($0) }
        #expect(scope.flatMap { ControlFrames.frames[$0]?["action"] } == nil)
        try await tap("generate", w)
        #expect(prepared == ["qwen3-tts-base"] && !session.running && session.takes.isEmpty)
    }

    // MARK: Menu

    @Test func theMenuEntryOpensTheOneWindowWithThatModelChosen() async throws {
        let root = scratch.appendingPathComponent("menu")
        // The menu names fixtures, never shipped models: the two rows that carry a sound model's or a voice model's name are
        // renamed here. Their ids stay, so the operations and the window's selection still match.
        var sound = SoundFixture.stableAudio; sound.name = "Sound Fixture"
        var voice = SoundFixture.qwenBase; voice.name = "Voice Fixture"
        // On the table through an old copy the catalog no longer offers; its offered precision is not on this Mac.
        voice.withdrawnTiers = ["int8": 1_700_000_000]
        let service = FixtureService(rows: [sound, SoundFixture.kokoro, voice], metadata: SoundFixture.metadata, supportDirectory: root, operations: SoundFixture.allOperations)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root))
        await app.api.start(); defer { app.api.stop(); app.table.stop(); app.tools.closeAll() }
        app.tools.present = { _ in }
        app.tools.soundDirectory = scratch.appendingPathComponent("Music")
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        // Placement by declared operations (ToolCenter.tasks): Stable Audio declares generate_sound and generate_music, so
        // it serves Make Music and Make a Sound; Kokoro and the Qwen voice declare speak, so they serve Speak.
        let rows = app.table.rows
        func tasks(_ id: String) throws -> [MenuTask] { app.tools.tasks(try #require(rows.first { $0.id == id })) }
        #expect(try tasks("stable-audio-3-medium") == [.sound(.music), .sound(.sound)])
        #expect(try tasks("kokoro-82m") == [.sound(.speak)] && (try tasks("qwen3-tts-base")) == [.sound(.speak)])
        let keys = SoundTask.allCases.map { AppDelegate.taskKey(.sound($0)) }
        let items = app.menu.items.filter { keys.contains(($0.representedObject as? String) ?? "") && !$0.isHidden }
        #expect(items.map(\.title) == ["Speak\u{2026}", "Make Music\u{2026}", "Make a Sound\u{2026}"], "\(items.map(\.title))")
        #expect(items.allSatisfy { $0.isEnabled && $0.action != nil && $0.badge == nil && $0.submenu == nil })
        // No model's name heads a section, and the voice not on this Mac has no "not installed" row.
        #expect(!app.menu.items.contains { $0.isSectionHeader || $0.title == "Sound Fixture" || $0.title == "Voice Fixture" })
        #expect(!app.menu.items.contains { $0.title.contains("not installed") })
        // Make a Sound opens its window with the one model that makes sounds.
        app.menu.performActionForItem(at: app.menu.index(of: items[2]))
        let effects = try #require(app.tools.soundTasks[.sound])
        #expect(effects.session.selected == "stable-audio-3-medium" && effects.session.models.map(\.id) == ["stable-audio-3-medium"])
        #expect(effects.window.window?.title == SoundTask.sound.windowTitle)
        // Speak opens its own window with the model that runs now (Kokoro, loaded); the voice not on this Mac is a
        // choice whose problem offers its download.
        app.menu.performActionForItem(at: app.menu.index(of: items[0]))
        let speak = try #require(app.tools.soundTasks[.speak])
        #expect(speak.window !== effects.window && speak.window.window?.title == SoundTask.speak.windowTitle)
        #expect(speak.session.selected == "kokoro-82m" && speak.session.models.map(\.id) == ["kokoro-82m", "qwen3-tts-base"])
        #expect(speak.session.models[1].unavailable?.action == .download)
        try await waitUntil { !speak.session.loading }
        #expect(speak.session.operations.map(\.name) == ["speak"])
        // Make Music: Stable Audio's window narrowed to music.
        app.menu.performActionForItem(at: app.menu.index(of: items[1]))
        let music = try #require(app.tools.soundTasks[.music])
        try await waitUntil { music.session.selected == "stable-audio-3-medium" && !music.session.loading }
        #expect(music.session.operations.map(\.name) == ["generate_music"])
        // Again: the same Speak window.
        app.menu.performActionForItem(at: app.menu.index(of: items[0]))
        #expect(app.tools.soundTasks[.speak]?.window === speak.window)
        // The settings' switch names the tool.
        #expect(TableController.toolsSetting(SoundFixture.kokoro)?.declaration["title"]?.string == "Show Voice and Sound in menu")
    }

    private func waitUntil(seconds: Double = 5, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            if Date() > deadline { Issue.record("timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
