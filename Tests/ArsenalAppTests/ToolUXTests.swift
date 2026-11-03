// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

/// The tool windows' UX pass (minimal visual change): the band titled by
/// task, lab figures behind Details, task verbs on the run button, Save As… on every result, history from the first
/// result, problems worded and dimming the old result, no plumbing or destructive pills, sampler knobs under Advanced,
/// New Voice… in the Voice picker, and the sound tasks. Windows are never shown; no model runs.
@MainActor @Suite(.menuState, .serialized) final class ToolUXTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        ControlFrames.recording = true
        SoundPlayer.silent = true
        details = ToolDetails.shown
        UserDefaults.standard.removeObject(forKey: ToolDetails.key)
    }
    private let details: Bool
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-ux-tests-\(UUID().uuidString)", isDirectory: true)
    deinit {
        try? FileManager.default.removeItem(at: scratch)
        ToolDetails.shown = details
    }

    // MARK: Window plumbing (as SoundToolTests drives a window)

    private var scope: UUID?
    private func open<V: View>(_ view: V, size: NSSize = ToolLayout.size) async throws -> (ToolWindowController, NSWindow) {
        let wc = ToolWindowController(title: "test", autosave: "", size: size, minSize: size, content: view)
        let w = try #require(wc.window)
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
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
    private func frame(_ id: String) -> CGRect? { scope.flatMap { ControlFrames.frames[$0]?[id] } }
    private func tap(_ id: String, _ w: NSWindow) async throws {
        let f = try #require(frame(id), "\(id) was not laid out")
        let height = try #require(w.contentView).bounds.height
        let p = NSPoint(x: f.midX, y: height - f.midY)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            w.sendEvent(
                try #require(
                    NSEvent.mouseEvent(
                        with: type, location: p, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: w.windowNumber, context: nil,
                        eventNumber: 0, clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0)))
        }
        try await settle(w)
    }
    private func sound(_ models: [ModelChoice] = SoundFixture.models, selected: String = "stable-audio-3-medium", task: SoundTask? = nil) -> SoundSession {
        SoundSession(models: models, selected: selected, client: { nil }, inbox: scratch, outputDirectory: scratch, scratch: scratch, task: task)
    }
    private func qwen() -> SoundSession {
        var ready = SoundFixture.models[2]; ready.unavailable = nil
        let session = sound([ready], selected: ready.id)
        session.preview(operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hello")], voices: [])
        return session
    }

    // MARK: Band and Details

    @Test func theBandNamesTheTaskAndTheModelPopUpKeepsThePrecisionInItsHover() async throws {
        let session = sound(task: .speak)
        session.preview(operations: SoundFixture.descriptors("kokoro-82m"))
        #expect(session.taskTitle == "Speak" && session.runVerb == "Speak")
        let model = try #require(session.model?.model)
        #expect(ModelPicker.help(model).hasPrefix(model.name + " \u{00b7} " + model.detail))
        let (wc, _) = try await open(SoundView(session: session))
        defer { wc.close(); session.close() }
        let title = try #require(frame("task-title")), picker = try #require(frame("model"))
        #expect(title.maxX <= picker.minX, "the task comes first, the model after it")
    }

    @Test func detailsAreHiddenByDefaultAndTheBandButtonShowsThemForEveryWindow() async throws {
        #expect(!ToolDetails.shown)
        let session = sound()
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), values: ["prompt": .text("Rain")])
        let (wc, w) = try await open(SoundView(session: session))
        defer { wc.close(); session.close() }
        let before = try #require(frame("field-prompt")).minY
        try await tap("details", w)
        #expect(ToolDetails.shown)
        let after = try #require(frame("field-prompt")).minY
        #expect(after > before + 30, "the figures take their place above the form: \(before) → \(after)")
        try await tap("details", w)
        let back = try #require(frame("field-prompt")).minY
        #expect(!ToolDetails.shown && abs(back - before) < 1)
    }

    @Test func chatOpensWithoutBenchmarkPromptsAndOnePeakMemoryName() async throws {
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { nil })
        let (wc, w) = try await open(ChatView(session: session), size: ChatView.size)
        defer { wc.close() }
        #expect(frame("preset-1K") == nil && session.preset == nil && frame("details") != nil)
        try await tap("details", w)
        #expect(frame("preset-1K") != nil)
        #expect(session.figures().contains { $0.title == "Peak memory" } && !session.figures().contains { $0.title == "Peak RAM" })
        if case .unavailable(let why)? = session.referenceLine { #expect(why != "No reference yet") }
    }

    // MARK: Run verbs

    @Test func theRunButtonSaysTheTasksVerb() {
        let music = sound(task: .music)
        music.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"))
        #expect(music.operations.map(\.name) == ["generate_music"] && music.runVerb == "Make Music" && music.taskTitle == "Make Music")
        let effects = sound(task: .sound)
        effects.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"))
        #expect(effects.operations.map(\.name) == ["generate_sound"] && effects.runVerb == "Make Sound")
        // Without a task: the chosen operation's task.
        let any = sound()
        any.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), selected: "generate_music")
        #expect(any.runVerb == "Make Music" && any.taskTitle == "Make Music")
        let image = ImageSession(model: V3RenderDelegate.flux, client: { nil }, outputDirectory: scratch)
        image.preview(operations: [ImageFixture.generate])
        #expect(image.runVerb == "Make Image" && image.taskTitle == "Make an Image")
    }

    @Test func aDisabledRunButtonSaysWhy() {
        let session = sound(task: .speak)
        var kokoro = SoundFixture.models[1]; kokoro.unavailable = nil
        session.update(models: [kokoro])
        session.preview(operations: SoundFixture.descriptors("kokoro-82m"))
        #expect(!session.canRun && session.notRunnable == "Type something to speak")
        session.set("text", .text("Hello"))
        #expect(session.canRun && session.notRunnable == nil)
    }

    // MARK: Results

    @Test func everyResultKindOffersSaveAsAndItCopiesTheFile() async throws {
        // A take.
        let take = try SoundFixture.take(in: scratch, seconds: 1, text: "Rain", seed: 3)
        let session = sound()
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), takes: [take])
        let target = scratch.appendingPathComponent("saved/copy.wav")
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        var suggested: URL?
        session.chooseSaveURL = {
            suggested = $0; return target
        }
        var (wc, w) = try await open(SoundView(session: session))
        try await tap("save-as", w)
        #expect(suggested == take.url && FileManager.default.contentsEqual(atPath: target.path, andPath: take.url.path))
        wc.close(); session.close()
        // An image.
        let png = try CapabilityFixture.photo(scratch.appendingPathComponent("p/one.png"), width: 64, height: 48)
        let image = ImageSession(model: V3RenderDelegate.flux, client: { nil }, outputDirectory: scratch)
        image.preview(operations: [ImageFixture.generate], results: [ImageResult(url: png, prompt: "a", width: 64, height: 48, seconds: 1)])
        (wc, w) = try await open(ImageView(session: image))
        #expect(frame("save-as") != nil)
        wc.close(); image.close()
        // A cut-out.
        let background = BackgroundSession(
            models: [ModelChoice(model: ToolModel(CutoutFixture.row))], selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch,
            scratch: scratch)
        let cutout = Cutout(original: png, url: png, mask: nil, model: CutoutFixture.id, modelName: "BiRefNet", width: 64, height: 48, seconds: 1)
        background.preview(operations: [CutoutFixture.descriptor], input: png, cutouts: [cutout], shown: cutout.id)
        (wc, w) = try await open(BackgroundView(session: background))
        #expect(frame("save-as") != nil)
        wc.close(); background.close()
        // A capability result with a file, and stems with one Save each and Save All.
        for (tool, model, op) in [(ToolKind.upscale, "upscale-fixture", "upscale_image"), (.separate, "demucs-fixture", "separate_audio")] {
            let (cap, view) = CapabilityWindows.make(
                tool, models: [CapabilityFixture.choice(model)], selected: model, client: { nil }, inbox: scratch, output: scratch, scratch: scratch)
            cap.preview(operations: [CapabilityFixture.descriptor(op)])
            let file =
                op == "separate_audio"
                ? ResultFile(id: "vocals", label: "Vocals", order: 0, mime: "audio/wav", url: take.url, bytes: 1, extras: [])
                : ResultFile(id: "image", label: "Upscaled", order: 0, mime: "image/png", url: png, bytes: 1, width: 64, height: 48, extras: [])
            cap.append(
                Outcome(
                    model: model, modelName: model, operation: op, operationTitle: op, inputs: [], text: "", values: [:], input: [:], reply: [:], files: [file],
                    seconds: 1))
            (wc, w) = try await open(view)
            if tool == .separate {
                #expect(frame("save-vocals") != nil && frame("save-all") != nil)
            } else {
                #expect(frame("save-as") != nil)
            }
            wc.close(); cap.close()
        }
        // The generic Run window's files.
        let run = OperationSession(model: ToolModel(LayersFixture.row), client: { nil }, inbox: scratch, results: scratch)
        run.preview(
            operations: [LayersFixture.descriptor],
            result: OperationResult(operation: "s", files: [ResultFile(id: "layer-0", label: "Near", order: 0, mime: "image/png", url: png, bytes: 1, extras: [])], seconds: 1))
        (wc, w) = try await open(OperationView(session: run))
        #expect(frame("save-layer-0") != nil)
        wc.close(); run.close()
    }

    @Test func theTakesListShowsFromTheFirstTakeAndChoosingOneBringsItsOptionsBack() async throws {
        var take = try SoundFixture.take(in: scratch, seconds: 1, text: "Rain on a tin roof", seed: 3)
        take.operationName = "generate_sound"; take.values = ["prompt": .text("Rain on a tin roof"), "steps": .text("12")]
        let session = sound()
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), values: ["prompt": .text("Thunder")], takes: [take])
        let (wc, w) = try await open(SoundView(session: session))
        defer { wc.close(); session.close() }
        #expect(frame("take-\(take.id)") != nil && frame("take-title") != nil)
        try await tap("take-\(take.id)", w)
        #expect(session.value("prompt").text == "Rain on a tin roof" && session.value("steps").text == "12")
    }

    @Test func fileErrorsAreWorded() {
        let raw = NSError(domain: NSCocoaErrorDomain, code: NSFileReadNoSuchFileError, userInfo: [NSFilePathErrorKey: "/tmp/x/diarize_session.input.json"])
        let worded = OperationProblem.from(raw, model: "M")
        #expect(worded.message == "Couldn\u{2019}t read diarize_session.input.json." && worded.detail == raw.localizedDescription)
        #expect(!OperationProblem.wording(raw, model: "M").contains("couldn\u{2019}t be opened"))
    }

    @Test func theMemoryLineIsShortAndLoadAnywaySaysItIsSlower() {
        let refusal = LoadMemoryRefusal(model: "m", name: "Stable Audio 3 Medium", precision: "bf16", neededGB: 6.4, freeGB: 3.1)
        let line = OperationProblem.memoryLine(refusal)
        #expect(line == "Needs \(OperationProblem.gigabytes(6.4)) of memory; \(OperationProblem.gigabytes(3.1)) free.")
        #expect(OperationProblem.gigabytes(1.0737418240 * 6) == "6 GB" && OperationProblem.gigabytes(1.0737418240 * 2.9) == "2.9 GB")
        #expect(OperationProblem.declined.action == .loadAnyway && OperationProblem.declined.message == "Not run: not enough free memory.")
        let short = LoadMemoryRefusal(model: "m", name: "M", precision: "fp16", neededGB: 6.44, freeGB: 3.11)
        let words = OperationProblem.memoryLine(short)
        #expect(words.hasPrefix("Needs ") && words.hasSuffix(" of memory; 2.9 GB free."), "the error names the resource: \(words)")
        let declined = OperationProblem.declinedRun(short)
        #expect(declined.message == "Not run: n" + words.dropFirst(), "Cancel keeps the alert's figures")
        #expect(declined.action == OperationProblem.Action.loadAnyway)
        #expect(OperationProblem.declinedRun(nil) == OperationProblem.declined)
        #expect(OperationProblem.anywayHelp == "Uses disk as memory; the Mac may slow down.")
        #expect(OperationProblem.anywayTitle("Make Sound") == "Make Sound Anyway (slower)")
    }

    @Test func aModelThatIsNotDownloadedIsChoosableAndTheRunButtonDownloadsIt() async throws {
        var models = SoundFixture.models
        models[2].unavailable = .init(message: "Qwen3 Base 1.7B isn\u{2019}t on this Mac yet.", action: .download, actionTitle: "Download 3.4 GB")
        let session = sound(models, selected: "kokoro-82m")
        session.preview(operations: SoundFixture.descriptors("kokoro-82m"), values: ["text": .text("Hi")])
        session.choose("qwen3-tts-base")
        #expect(session.selected == "qwen3-tts-base" && session.preparesInstead?.actionTitle == "Download 3.4 GB")
        session.preview(operations: SoundFixture.descriptors("qwen3-tts-base"), values: ["text": .text("Hi")])
        var prepared: String?
        session.prepareModel = { prepared = $0 }
        let (wc, w) = try await open(SoundView(session: session))
        defer { wc.close(); session.close() }
        try await tap("generate", w)
        #expect(prepared == "qwen3-tts-base")
    }

    // MARK: Pills, Advanced, voices

    @Test func plumbingAndDestructiveOperationsAreNeverPills() {
        let ops = ["speak", "clone_voice", "delete_saved_voice", "delete_voice", "voices"].map {
            OperationDescriptor(name: $0, title: $0, inputSchema: ["type": "object", "properties": ["text": ["type": "string"]]], output: "audio", stream: false)
        }
        let run = OperationSession(model: ToolModel(id: "ming", name: "Ming", kind: "voices", dtype: "bf16", mode: "exact"), client: { nil }, inbox: scratch)
        run.preview(operations: ops)
        #expect(run.operations.map(\.name) == ["speak", "clone_voice"])
        #expect(run.pillOperations.map(\.name) == ["speak"], "cloning is a Voice picker item in a window that speaks")
        let s = sound()
        s.preview(operations: ops)
        #expect(!s.operations.contains { ToolSession.neverPills.contains($0.name) })
    }

    @Test func samplerKnobsAndTheSeedFoldUnderAdvanced() async throws {
        let session = qwen()
        let keys = Set(session.optionFields.filter(session.isAdvanced).map(\.key))
        #expect(keys.isSuperset(of: ["temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "seed"]))
        #expect(!session.isAdvanced(try #require(session.optionFields.first { $0.key == "voice" })))
        let (wc, w) = try await open(SoundView(session: session))
        defer { wc.close(); session.close() }
        #expect(frame("advanced") != nil && frame("field-temperature") == nil && frame("field-voice") != nil)
        try await tap("advanced", w)
        #expect(session.advancedOpen && frame("field-temperature") != nil)
    }

    @Test func newVoiceShowsTheClipInSpeakAndAVoiceLeadsBack() {
        // Qwen3 TTS Base in Speak: its clone_voice saves a voice (JSON), so New Voice… shows Speak's own clip to clone.
        let session = qwen()
        #expect(session.newVoiceChoices.map(\.title) == ["New Voice\u{2026}"] && session.newVoiceChoices.first?.operation == nil)
        let reference = try? #require(session.form?.fields.first { $0.key == "reference" })
        #expect(reference.map(session.showsField) == false, "no clip to clone in Speak until New Voice…")
        session.chooseNewVoice(nil)
        #expect(session.newVoice && session.makingVoice && reference.map(session.showsField) == true)
        session.chooseVoice("zz_saved")
        #expect(!session.newVoice && session.value("voice").text == "zz_saved" && reference.map(session.showsField) == false)
    }

    @Test func newVoiceSwitchesToACloneOperationWhereTheWindowOffersOne() {
        let schema: JSONObject = ["type": "object", "required": ["text"], "properties": ["text": ["type": "string"], "voice": ["type": "string"]]]
        let ops = ["speak", "clone_voice"].map { OperationDescriptor(name: $0, title: $0 == "speak" ? "Speak" : "Clone Voice", inputSchema: schema, output: "json", stream: false) }
        let run = OperationSession(model: ToolModel(id: "ming", name: "Ming", kind: "voices", dtype: "bf16", mode: "exact"), client: { nil }, inbox: scratch)
        run.preview(operations: ops, selected: "speak")
        #expect(run.newVoiceChoices.map(\.title) == ["New Voice\u{2026}"] && run.newVoiceChoices.first?.operation == "clone_voice")
        run.chooseNewVoice("clone_voice")
        #expect(run.operation == "clone_voice" && run.makingVoice)
        run.chooseVoice("zz_saved")
        #expect(run.operation == "speak" && run.value("voice").text == "zz_saved" && !run.makingVoice)
        #expect(run.taskTitle == "Speak")
    }

    @Test func kokorosVoicesAreGroupedUnderNamedHeadings() {
        let groups = VoicePicker.groups(SoundFixture.kokoroVoices)
        #expect(groups.allSatisfy { $0.heading != nil })
        #expect(groups.map(\.heading) == ["American English \u{00b7} Female", "American English \u{00b7} Male", "British English"])
        let one = VoicePicker.groups(Array(SoundFixture.kokoroVoices.prefix(3)))
        #expect(one.map(\.heading) == ["Female", "Male"])
    }

    @Test func showInFinderNamesItsFolder() {
        let session = sound()
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"))
        session.outputDirectory = URL(fileURLWithPath: "/Users/x/Music/Arsenal", isDirectory: true)
        #expect(session.revealFolderName == "Music \u{203a} Arsenal")
    }

    @Test func plainWordsAcrossTheCapabilityWindows() {
        // Transcribe's speaker pill, Remove Background's video run, Depth's verb, raw JSON and a cut's tuning folded.
        let (transcribe, _) = CapabilityWindows.make(
            .transcribe, models: [CapabilityFixture.choice("whisper-fixture")], selected: "whisper-fixture", client: { nil }, inbox: scratch, output: scratch)
        let diarize = CapabilityFixture.descriptor("diarize")
        #expect(transcribe.pillTitle(diarize) == "Speaker timing")
        #expect(BackgroundView.runTitle(OperationDescriptor(name: BackgroundSession.video, title: "Track", inputSchema: [:], output: "file", stream: false)) == "Track Mask")
        let (depth, _) = CapabilityWindows.make(
            .depth, models: [CapabilityFixture.choice("depth-fixture")], selected: "depth-fixture", client: { nil }, inbox: scratch, output: scratch)
        depth.preview(operations: [CapabilityFixture.descriptor("estimate_depth")])
        #expect(depth.runVerb == "Make Depth Map" && depth.taskTitle == ToolKind.depth.windowTitle)
        let json = ToolField(key: "exemplars", title: "Visual exemplars", help: nil, control: .generic(.json), required: false, defaultValue: nil)
        #expect(depth.isAdvanced(json))
        let background = BackgroundSession(
            models: [ModelChoice(model: ToolModel(CutoutFixture.row))], selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch, scratch: scratch)
        #expect(background.advancedKeys(BackgroundSession.operation).isSuperset(of: ["threshold", "resolution"]))
        // The generic Run window's files keep readable columns: two at its smallest width.
        #expect(Int((ToolLayout.minSize.width - ToolLayout.controlsWidth - 4 * ToolLayout.inset + 12) / FileGrid.minColumn) >= 2)
    }

    @Test func aSoundWindowStillReadingItsOptionsSaysItsVerb() {
        let s = sound()
        #expect(s.descriptor == nil && s.runVerb == "Make Sound")
    }

    /// Review of the pass: a hidden clip is never sent, New Voice… does not follow another operation, a decoding error is
    /// not a file error, an unwired Download still does something, and Save As never loses the file it replaces.
    @Test func reviewFixesHold() throws {
        let session = qwen()
        // Without a voice or a clip on show, the schema's alternatives are said in plain words.
        session.start()
        #expect(session.run == nil && session.problem?.message == "Choose a voice, or make one with New Voice\u{2026}")
        session.set("voice", .text("voice-0a1b"))
        session.set("reference", .files([try SoundFixture.take(in: scratch, seconds: 1, text: "", seed: nil).url]))
        #expect(session.canRun, "\(session.notRunnable ?? "") \(String(describing: session.problem))")
        session.start()
        let sent = session.run?.input
        #expect(session.run != nil && sent?["reference"] == nil, "a clip not on show is not sent: \(String(describing: session.problem)) \(String(describing: sent))")
        session.stop()
        let run = OperationSession(model: ToolModel(id: "ming", name: "Ming", kind: "voices", dtype: "bf16", mode: "exact"), client: { nil }, inbox: scratch)
        let schema: JSONObject = ["type": "object", "properties": ["text": ["type": "string"], "reference": ["type": "string", "x-arsenal-file": ["audio/*"]]]]
        run.preview(operations: ["speak", "describe"].map { OperationDescriptor(name: $0, title: $0, inputSchema: schema, output: "json", stream: false) }, selected: "speak")
        run.newVoice = true
        run.select("describe")
        #expect(!run.newVoice && !run.makingVoice)
        let decoding = NSError(domain: NSCocoaErrorDomain, code: 4864)
        #expect(FileProblem(decoding) == nil)
        var opened = false
        run.openModels = { opened = true }
        run.performAction(.download)
        #expect(opened)
        let folder = scratch.appendingPathComponent("place")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let a = folder.appendingPathComponent("a.txt"), b = folder.appendingPathComponent("b.txt")
        try Data("new".utf8).write(to: a); try Data("old".utf8).write(to: b)
        try ToolSession.place(a, at: b)
        #expect(try String(contentsOf: b, encoding: .utf8) == "new")
        #expect(throws: (any Error).self) { try ToolSession.place(folder.appendingPathComponent("missing"), at: b) }
        #expect(try String(contentsOf: b, encoding: .utf8) == "new", "a failed copy leaves the file in place")
    }

    // MARK: Review round 1 (30 Oct)

    /// A memory refusal: the run button becomes the rerun named by the window's verb, with no word Load and no second
    /// bright button; its click runs past the refusal.
    @Test func aMemoryRefusalMakesTheRunButtonTheAnywayRun() async throws {
        let session = sound(task: .sound)
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), values: ["prompt": .text("Rain")], problem: OperationProblem.declined)
        #expect(session.rerunInstead == .loadAnyway && session.runButtonTitle(session.runVerb) == "Make Sound Anyway (slower)")
        #expect(session.duplicatesRun(OperationProblem.declined), "the line by the button offers no second way")
        #expect(!session.worded(OperationProblem.declined).actionTitle!.contains("Load"))
        let (wc, _) = try await open(SoundView(session: session))
        defer { wc.close(); session.close() }
        #expect(frame("action") == nil && frame("generate") != nil)
        // Speak's own verb.
        let speak = sound(selected: "kokoro-82m", task: .speak)
        speak.preview(operations: SoundFixture.descriptors("kokoro-82m"), values: ["text": .text("Hi")], problem: OperationProblem.declined)
        #expect(speak.runButtonTitle(speak.runVerb) == "Speak Anyway (slower)")
        // Nothing to run (the text cleared): the button is off and the line keeps its own action, still not "Load".
        speak.set("text", .text(""))
        #expect(speak.rerunInstead == nil && speak.worded(OperationProblem.declined).actionTitle == "Speak Anyway (slower)")
    }

    /// Core review 12: an isolated profile (ARSENAL_SUPPORT_DIR set) keeps window results inside itself; the user's own
    /// profile keeps them in ~/Pictures/Arsenal, ~/Music/Arsenal and their siblings.
    @Test func anIsolatedProfileKeepsResultsInsideItself() {
        let isolated = ["ARSENAL_SUPPORT_DIR": "/tmp/arsenal-review/support"]
        #expect(ToolFiles.kept(.picturesDirectory, environment: isolated).path == "/tmp/arsenal-review/support/Outputs/Pictures")
        #expect(ToolFiles.kept(.musicDirectory, environment: isolated).path == "/tmp/arsenal-review/support/Outputs/Music")
        #expect(ToolFiles.kept(.moviesDirectory, environment: isolated).path == "/tmp/arsenal-review/support/Outputs/Movies")
        #expect(ToolFiles.kept(.documentDirectory, environment: isolated).path == "/tmp/arsenal-review/support/Outputs/Documents")
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        #expect(ToolFiles.kept(.picturesDirectory, environment: [:]).path == home + "/Pictures/Arsenal")
        #expect(ToolFiles.kept(.musicDirectory, environment: ["ARSENAL_SUPPORT_DIR": ""]).path == home + "/Music/Arsenal")
    }

    /// Core review 5: the Anyway button re-checks memory. Once enough is freed (by the figures in use when the refusal
    /// was shown), the refusal goes and the button is the window's own verb again; too little freed keeps it.
    @Test func freedMemoryTurnsTheAnywayButtonBackIntoTheRun() throws {
        let session = sound(task: .sound)
        let refusal = LoadMemoryRefusal(model: "stable-audio-3-medium", name: "Stable Audio 3 Medium", precision: "fp16", neededGB: 12, freeGB: 8)
        session.preview(operations: SoundFixture.descriptors("stable-audio-3-medium"), values: ["prompt": .text("Rain")], problem: .declinedRun(refusal))
        defer { session.close() }
        #expect(session.runButtonTitle(session.runVerb) == "Make Sound Anyway (slower)")
        #expect(!session.memoryFreed(refusal, baselineMB: 90_000, usedMB: 87_000), "3 GB freed of the 4 GB missing")
        #expect(session.runButtonTitle(session.runVerb) == "Make Sound Anyway (slower)")
        #expect(session.memoryFreed(refusal, baselineMB: 90_000, usedMB: 85_500))
        #expect(session.problem == nil && session.rerunInstead == nil && session.runButtonTitle(session.runVerb) == "Make Sound")
        let error = ToolError.from(status: 409, body: try WireJSON.encode(["code": "insufficient_memory", "error": "x", "details": .object(refusal.details)]))
        #expect(OperationProblem.from(error, model: refusal.name).refusal == refusal, "the inline refusal carries its figures too")
    }

    /// A model process that ended mid-run reads by the task, never "worker", and the run button is Try Again with the
    /// inputs kept.
    @Test func aCrashIsWordedByTaskAndTheRunButtonIsTryAgain() throws {
        let crash = OperationProblem.from(ToolError("Worker connection closed.", status: 502, code: "worker_exit"), model: "Whisper", task: "Transcription")
        #expect(crash.message == "Transcription stopped unexpectedly." && crash.action == .retry && crash.detail == "Worker connection closed.")
        #expect(ToolSession.stoppedNoun("Transcribe") == "Transcription" && ToolSession.stoppedNoun("Speak") == "Speech")
        let (base, _) = CapabilityWindows.make(
            .transcribe, models: [CapabilityFixture.choice("whisper-fixture")], selected: "whisper-fixture", client: { nil }, inbox: scratch, output: scratch)
        let session = try #require(base as? TranscribeSession)
        #expect(session.stoppedNoun == "Transcription")
        session.preview(operations: [CapabilityFixture.descriptor("transcribe")])
        let clip = try CapabilityFixture.clip(scratch.appendingPathComponent("c/Team meeting.wav"))
        session.setSubject([clip])
        session.problem = crash
        #expect(session.runButtonTitle(session.runVerb) == "Try Again" && session.subject == [clip])
        session.discard()
    }

    /// The generic Run window waits for its required file, says why on the button's hover, and has the folder button.
    @Test func theRunWindowWaitsForItsFileAndShowsTheFolder() async throws {
        let session = OperationSession(model: ToolModel(LayersFixture.row), client: { nil }, inbox: scratch)
        session.preview(operations: [LayersFixture.descriptor])
        #expect(!session.canRun && session.notRunnable == "Add an image first")
        let (wc, _) = try await open(OperationView(session: session), size: OperationView.size)
        defer { wc.close() }
        #expect(frame("reveal") != nil)
        let image = try CapabilityFixture.photo(scratch.appendingPathComponent("r/a.png"))
        session.set("image", .files([image]))
        #expect(session.canRun && session.notRunnable == nil)
    }

    /// Pills only for the window's own task, opened from any entry: never Sound beside Music.
    @Test func soundPillsBelongToTheTaskWhateverOpenedTheWindow() {
        let ops = SoundFixture.descriptors("stable-audio-3-medium")
        for task in [SoundTask.sound, .music, nil] {
            let session = sound(task: task)
            session.previewOptions(operations: ops, selected: task == .music ? "generate_music" : "generate_sound", values: [:])
            #expect(session.pillOperations.count <= 1, "\(String(describing: task)): \(session.pillOperations.map(\.name))")
        }
    }

    /// Remove Background: the title says the task, the operation pills Cut-out and Alpha Matte (never the result pane's
    /// Cut-out | Mask pair again).
    @Test func removeBackgroundPillsAreCutoutAndMatte() {
        let matte = OperationDescriptor(
            name: "matte", title: "Alpha Matte", inputSchema: CutoutFixture.descriptor.inputSchema, outputSchema: nil, output: "file", stream: false, timeout: 60)
        #expect(BackgroundSession.pillName(CutoutFixture.descriptor) == "Cut-out" && BackgroundSession.pillName(matte) == "Alpha Matte")
        #expect(BackgroundView.wellAspect(PixelSize(width: 1200, height: 900)) == CGFloat(1200.0 / 900.0) && BackgroundView.wellAspect(nil) == nil)
    }

    /// Small fixes: capability history from two results; Auto sliders rest at the default; plain sampler hovers; one
    /// transcript placeholder; the Details strip names the precision.
    @Test func historyAutoSlidersHoversTranscriptAndPrecision() {
        #expect(!OutcomeHistory.shows(1) && OutcomeHistory.shows(2))
        #expect(FieldControl.restingValue(key: "temperature", declared: nil, lo: 0, hi: 2) == 1)
        #expect(FieldControl.restingValue(key: "top_k", declared: 50, lo: 1, hi: 200) == 50)
        #expect(FieldControl.restingValue(key: "guidance", declared: nil, lo: 1, hi: 10) == 1)
        #expect(FieldControl.rangeHelp(unset: true, declared: true, "0\u{2013}2", defaultValue: "0.9") == "Auto: the model\u{2019}s default, 0.9. Takes 0\u{2013}2.")
        for key in ["top_k", "top_p", "temperature", "repetition_penalty"] { #expect(FieldControl.plainHelp[key] != nil, "\(key)") }
        #expect(ToolSlider.restingOpacity > 0 && ToolSlider.restingOpacity < 1)
        #expect(FieldControl.transcriptPlaceholder == "Type what the clip says")
        let model = ToolModel(id: "a", name: "A", kind: "images", dtype: "bf16", mode: "fast")
        #expect(model.precisionFigure.value == "16-bit \u{00b7} Fast" && model.precisionFigure.title == "Precision")
    }

    /// Transcribe's Label speakers switch follows the value the run used, set in code (a result chosen, a render).
    @Test func aSwitchSetInCodeIsDrawnOn() async throws {
        let (base, view) = CapabilityWindows.make(
            .transcribe, models: [CapabilityFixture.choice("whisper-fixture")], selected: "whisper-fixture", client: { nil }, inbox: scratch, output: scratch)
        let session = try #require(base as? TranscribeSession)
        defer { session.discard() }
        session.preview(operations: [CapabilityFixture.descriptor("transcribe")])
        session.setSubject([try CapabilityFixture.clip(scratch.appendingPathComponent("s/Team meeting.wav"))])
        let (wc, w) = try await open(view)
        defer { wc.close() }
        session.set("diarize", .toggle(true))
        try await settle(w)
        func switches(_ v: NSView) -> [NSSwitch] { (v as? NSSwitch).map { [$0] } ?? v.subviews.flatMap(switches) }
        let all = switches(try #require(w.contentView))
        #expect(!all.isEmpty)
        #expect(all.contains { $0.state == .on }, "states: \(all.map(\.state.rawValue))")
        // The result follows the switch: off, no line is labelled with a speaker.
        let lines = [TranscriptSegment(id: 0, start: 0, end: 1, speaker: "SPEAKER_00", text: "Hi")]
        #expect(session.labelsSpeakers && session.labelled(lines)[0].speaker == "SPEAKER_00")
        session.set("diarize", .toggle(false))
        #expect(!session.labelsSpeakers && session.labelled(lines)[0].speaker == nil)
    }

    /// The take header says voice or seed and sample rate; the words are the list row.
    @Test func theTakeHeaderShowsItsSeedAndRate() async throws {
        let take = try SoundFixture.take(in: scratch.appendingPathComponent("t"), seconds: 2, text: "Thunder", seed: 42, wave: 1)
        #expect(SoundResult.meta(take, voices: nil).hasPrefix("seed 42 \u{00b7} "))
    }

    // MARK: Review 2

    /// An unset slider's figure matches its knob: "Auto (0.9)" at the declared default; no default, "Auto" (no knob).
    @Test func autoSlidersSayWhereTheKnobRests() {
        #expect(FieldControl.autoFigure(declared: 0.9, lo: 0, hi: 2) == "Auto (0.9)")
        #expect(FieldControl.autoFigure(declared: 1, lo: 0.5, hi: 2) == "Auto (1)")
        #expect(FieldControl.autoFigure(declared: nil, lo: 0, hi: 2) == "Auto")
    }

    /// A reply's keys read as words, numbers grouped; the raw key and value stay for copy and export.
    @Test func resultKeysReadAsWords() {
        #expect(ResultWords.label("depth_mean") == "Mean depth" && ResultWords.label("depth_units") == "Depth units")
        #expect(ResultWords.label("nearest") == "Nearest" && ResultWords.label("depth_max") == "Max depth" && ResultWords.label("RMS") == "RMS")
        #expect(ResultWords.value("1048576") == "1,048,576" && ResultWords.value("0.4182") == "0.4182" && ResultWords.value("512") == "512")
        let pair = Pair(key: "pixels", value: "1048576")
        #expect(pair.line == "Pixels 1,048,576" && pair.key == "pixels" && pair.value == "1048576")
    }

    /// Layers are named nearest first with Background last, by order past the named slots.
    @Test func layerNamesKeepBackgroundLast() {
        #expect(LayersFixture.names(4) == ["Near", "Middle", "Far", "Background"])
        #expect(LayersFixture.names(2) == ["Near", "Background"])
        #expect(LayersFixture.names(6) == ["Near", "2", "3", "4", "5", "Background"])
        for n in 2...8 { #expect(LayersFixture.names(n).count == n && LayersFixture.names(n).last == "Background") }
    }

    /// Steps and guidance fold under Advanced in every window, images too; Rank Passages folds its tuning.
    @Test func diffusionKnobsFoldEverywhere() {
        let image = ImageSession(models: [], selected: "", client: { nil }, inbox: scratch, outputDirectory: scratch, scratch: scratch)
        for op in ["generate", "separate", "generate_sound", nil] {
            #expect(image.advancedKeys(op).isSuperset(of: ["steps", "guidance", "num_inference_steps", "seed"]), "\(op ?? "nil")")
        }
        let search = SearchSession(models: [], selected: "", client: { nil }, inbox: scratch, outputDirectory: scratch, scratch: scratch)
        #expect(search.advancedKeys("rerank").isSuperset(of: ["max_tokens_per_doc", "top_n"]))
    }

    /// Stem chips read as their result rows do; other chips keep what was typed.
    @Test func stemChipsAreCapitalised() {
        #expect(TagField.shown("vocals", key: "stems") == "Vocals" && TagField.shown("dog", key: "labels") == "dog")
    }

    /// Activity: a queue with nothing counted as running follows the first line; one unnamed request has no "Request"
    /// label; Models has no figure beside an empty bar.
    @Test func activityQueueFollowsTheClient() {
        let now = Date()
        let model = ActivityFeed.Model(
            id: "q", kind: "chat", state: "decode",
            requests: [
                .init(
                    id: "a", requester: "Claude Code", op: "chat", phase: "decode", promptTokens: 10, cachedTokens: nil, generatedTokens: 5, step: nil, total: nil,
                    startedAt: now.addingTimeInterval(-2), tps: 30.8),
                .init(
                    id: "b", requester: "API", op: "chat", phase: "prefill", promptTokens: 48_120, cachedTokens: nil, generatedTokens: nil, step: nil, total: nil,
                    startedAt: now, tps: nil)
            ], queued: 1, batch: nil, memoryMB: 1)
        let lines = ActivityText.lines(model)
        #expect(lines.map(\.name) == ["Claude Code", ""], "a lone unnamed request has no generic label")
        #expect(ActivityText.trailingCounts(model, lines: lines)?.map { "\($0.0) \($0.1)" } == ["1 waiting"])
        var counted = model; counted.running = 2
        #expect(ActivityText.trailingCounts(counted, lines: lines) == nil, "with a running count the counts line stays")
        let segment = MemorySegment(id: "q", name: "Q", mb: 27_000, colour: .systemTeal)
        let legend = ActivityText.shownLegend(parts: [], models: [segment], totalMB: nil, p: .dark)
        #expect(legend.first?.name == "Models" && ActivityText.gb(legend.first?.mb) == nil, "a dash beside an empty bar")
    }
}
