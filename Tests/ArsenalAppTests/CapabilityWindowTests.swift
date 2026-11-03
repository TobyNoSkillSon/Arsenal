// SPDX-License-Identifier: MIT
import AppKit
import SceneKit
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import Arsenal
import ArsenalAPI
import ArsenalCore

/// The capability windows (ToolCatalog) without models: fixture models declaring the kinds' operations with their
/// real schemas, the app's own API over the stub service with stand-in runners whose replies follow the kinds' output
/// schemas, each window run end to end (files staged, jobs polled and cancelled, files fetched by id and saved), what a
/// model does not declare never shown, the menu placing each model in its windows, and the memory warning before a
/// load. No panel, Finder window, menu or window is shown on screen.
@MainActor @Suite(.menuState, .serialized) final class CapabilityWindowTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        ControlFrames.recording = true
        SoundPlayer.silent = true
    }
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-capability-tests-\(UUID().uuidString)", isDirectory: true)
    deinit { try? FileManager.default.removeItem(at: scratch) }

    private struct Lab<S: CapabilitySession> {
        let service: FixtureService
        let api: APIHost
        let session: S
        let support: URL
        @MainActor func stop() { session.discard(); api.stop() }
    }

    /// The fixture models and runners behind the app's API, and a window of `tool` with `model` chosen (every fixture
    /// model that declares the window's operations offered).
    private func lab<S: CapabilitySession>(_ tool: ToolKind, _ model: String, as: S.Type, resident: Bool = true, prepare: Bool = true) async throws -> Lab<S> {
        let support = scratch.appendingPathComponent("\(tool.rawValue)-\(UUID().uuidString.prefix(6))")
        let service = await CapabilityFixture.service(support: support)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        let port = try #require(api.port)
        let models = CapabilityFixture.models.filter { m in ToolCatalog.windows(declared: Set(m.ops)).contains(tool) }.map { CapabilityFixture.choice($0.id, resident: resident) }
        let (session, _) = CapabilityWindows.make(
            tool, models: models, selected: model, client: { ArsenalClient(port: port) }, inbox: support.appendingPathComponent("inbox"),
            output: support.appendingPathComponent("Results"), scratch: support.appendingPathComponent("window"))
        let typed = try #require(session as? S)
        if prepare {
            typed.prepare()
            try await waitUntil(timeout: 10) { !typed.loading }
        }
        return Lab(service: service, api: api, session: typed, support: support)
    }

    // MARK: Catalog

    @Test func operationsPlaceEachModelInItsWindowsAndHelpersInNone() {
        let windows = Dictionary(uniqueKeysWithValues: CapabilityFixture.models.map { ($0.id, ToolCatalog.resolve(tools: [.operation], declared: Set($0.ops))) })
        #expect(windows["whisper-fixture"] == [.transcribe])
        #expect(windows["demucs-fixture"] == [.separate])
        #expect(windows["translate-fixture"] == [.translate] && windows["ocr-fixture"] == [.document] && windows["extract-fixture"] == [.extract])
        #expect(windows["rerank-fixture"] == [.search] && windows["embed-fixture"] == [.search])
        #expect(windows["sam-fixture"] == [.select], "sessions are the window's own, never a menu entry")
        #expect(windows["dino-fixture"] == [.find] && windows["depth-fixture"] == [.depth])
        #expect(windows["upscale-fixture"] == [.upscale] && windows["rife-fixture"] == [.smooth] && windows["trellis-fixture"] == [.make3d])
        #expect(windows["smart-turn-fixture"] == [], "an end-of-turn model serves live transcription and agents, not a window")
        // An operation no window offers keeps the generic tool beside the windows; explicit tools stay as they are.
        #expect(ToolCatalog.resolve(tools: [.operation], declared: ["detect", "segment", "vectorize"]) == [.select, .find, .operation])
        #expect(ToolCatalog.resolve(tools: [.background], declared: ["remove_background", "segment"]) == [.background])
        #expect(ToolCatalog.resolve(tools: [.operation], declared: nil) == [.operation], "before the operations are read the generic tool stands in")
        // Every window's operations exist in the kinds the integrators declare against.
        for tool in ToolCatalog.windows {
            for op in ToolCatalog.operations(tool) { #expect(CapabilityFixture.declared[op] != nil, "\(op) has no kind schema") }
            #expect(!tool.title.isEmpty && tool.title.hasSuffix("\u{2026}") && NSImage(systemSymbolName: tool.symbol, accessibilityDescription: nil) != nil, "\(tool)")
        }
    }

    @Test func theMenuOffersEachModelsWindowsAndOpensThem() async throws {
        let support = scratch.appendingPathComponent("menu")
        let service = await CapabilityFixture.service(support: support)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        defer { app.tools.closeAll(); app.table.stop() }
        app.tools.present = { _ in }
        app.tools.capabilityDirectory = support.appendingPathComponent("Results")
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        let titles = app.menu.items.filter { !$0.isHidden }.map(\.title)
        // By task: one entry per window, serving the model (no model's name heads a section: see the rows below).
        for (title, model) in capabilityMenuEntries {
            #expect(capabilityMenuRow(app, title, serving: model) != nil, "\(title) for \(model): \(titles)")
        }
        #expect(!titles.contains { $0.contains("Smart Turn") }, "no entry for a model whose operations are all behind the scenes")
        try expectBackgroundModelsAbsent(app)
        #expect(!titles.contains { $0.hasPrefix("Run") }, "no generic entry once every operation has a window")
        try openCapabilityMenuRows(app)
        // The entry opens the one window of that tool with the model chosen; a second model's entry reuses it.
        let rerank = try #require(app.table.rows.first { $0.id == "rerank-fixture" })
        app.tools.open(.search, rerank)
        let search = try #require(app.tools.capability[.search])
        #expect(search.session.selected == "rerank-fixture")
        #expect(Set(search.session.models.map(\.id)) == ["rerank-fixture", "embed-fixture"], "a window lists only models that declare its operations")
        let embed = try #require(app.table.rows.first { $0.id == "embed-fixture" })
        app.tools.open(.operation, embed)
        #expect(app.tools.capability[.search]?.window === search.window && search.session.selected == "embed-fixture", "the generic entry of a covered model opens its window")
        #expect(search.window.window?.title == "Rank Passages")
        // Transcribe knows the installed end-of-turn model for live mode.
        let whisper = try #require(app.table.rows.first { $0.id == "whisper-fixture" })
        app.tools.open(.transcribe, whisper)
        #expect((app.tools.capability[.transcribe]?.session as? TranscribeSession)?.turnModels == ["smart-turn-fixture"])
        // Show in menu names the windows, not the generic tool.
        let setting = try #require(TableController.toolsSetting(whisper, declared: ["transcribe", "diarize"]))
        #expect(setting.declaration["title"]?.string == "Show Transcribe in menu")
    }

    // MARK: Visibility

    /// Each window's options are the declared schema's, and an option a model's narrowed schema leaves out is absent.
    @Test func optionsFollowTheDeclaredSchemaAndNothingElse() {
        func keys(_ tool: ToolKind, _ op: String, drop: [String] = []) -> (options: [String], subject: [String], spatial: [String], main: String?) {
            var schema = CapabilityFixture.schema(op, "input")
            var properties = schema["properties"]?.object ?? [:]
            for key in drop { properties[key] = nil }
            schema["properties"] = .object(properties)
            schema["required"] = .array((schema["required"]?.array ?? []).filter { !drop.contains($0.string ?? "") })
            let (session, _) = CapabilityWindows.make(
                tool, models: [CapabilityFixture.choice("whisper-fixture")], selected: "whisper-fixture", client: { nil }, inbox: scratch, output: scratch)
            session.preview(operations: [CapabilityFixture.descriptor(op, input: schema)])
            // Every option, chooser or not (a chooser shows one option's controls at a time; the rest stay declared).
            let options = (session.optionFields.map(\.key) + session.groupedKeys.filter { k in !session.fileFields.contains { $0.key == k } && k != session.form?.main?.key })
            return (options.sorted(), session.fileFields.map(\.key), session.spatialFields.map(\.key).sorted(), session.form?.main?.key)
        }
        let full = keys(.transcribe, "transcribe")
        #expect(full.subject == ["audio"] && full.main == nil)
        #expect(full.options == ["diarize", "language", "prompt", "task", "word_timestamps"])
        let narrow = keys(.transcribe, "transcribe", drop: ["diarize", "word_timestamps", "task"])
        #expect(narrow.options == ["language", "prompt"], "a model without speakers shows no Speakers switch")
        #expect(keys(.separate, "separate_audio").main == "prompt")
        #expect(keys(.separate, "separate_audio", drop: ["prompt"]).main == nil)
        let segment = keys(.select, "segment")
        #expect(segment.subject == ["image"] && segment.spatial == ["boxes", "points"] && !segment.options.contains("session_id"))
        #expect(keys(.select, "segment", drop: ["boxes"]).spatial == ["points"], "no box tool for a model without boxes")
        #expect(keys(.select, "track_video").subject == ["video"] && !keys(.select, "track_video").options.contains("prompt_updates"))
        #expect(keys(.find, "ground_screen").main == "instruction" && keys(.find, "detect").main == nil)
        #expect(keys(.depth, "estimate_depth").subject.sorted() == ["image", "images"])
        #expect(keys(.smooth, "interpolate_video").subject.sorted() == ["frames", "video"])
        #expect(keys(.translate, "translate").main == "text" && keys(.translate, "translate").options.sorted() == ["context", "source_lang", "target_lang"])
        #expect(keys(.extract, "extract").subject.sorted() == ["document", "image"])
        #expect(keys(.document, "read_document").options.sorted() == ["document_annotation_format", "include_image_base64", "pages", "table_format"])
        #expect(keys(.make3d, "image_to_3d").options.sorted() == ["resolution", "seed", "simplify", "texture_size"])
    }

    @Test func semanticControlsComeFromTheFoldersHints() throws {
        let music = ToolForm(schema: CapabilityFixture.schemaFromKind("sound", "generate_music"), operation: "generate_music", output: "audio")
        let lyrics = try #require(music.fields.first { $0.key == "lyrics" })
        #expect(lyrics.control == .lyrics)
        #expect(music.fields.first { $0.key == "language" }?.control == .language(choices: nil))
        #expect(music.fields.first { $0.key == "instrumental" }?.control == .generic(.toggle))
        // Lyrics sit right under the prompt; an instrumental sends none.
        #expect(music.fields.map(\.key).prefix(2) == ["prompt", "lyrics"])
        let sung = try music.input(["prompt": .text("folk"), "lyrics": .text("la la"), "instrumental": .toggle(false)], seed: 1, stage: { $0.path })
        #expect(sung["lyrics"] == "la la")
        let instrumental = try music.input(["prompt": .text("folk"), "lyrics": .text("la la"), "instrumental": .toggle(true)], seed: 1, stage: { $0.path })
        #expect(instrumental["lyrics"] == nil && instrumental["instrumental"] == true)
        // Generate Sound (no song options) shows none of them.
        let sound = ToolForm(schema: CapabilityFixture.schemaFromKind("sound", "generate_sound"), operation: "generate_sound", output: "audio")
        #expect(!sound.fields.contains { ["lyrics", "language", "instrumental"].contains($0.key) })
        // Labels are chips, pages are typed as people count them, a schema is fields.
        let detect = ToolForm(schema: CapabilityFixture.schema("detect", "input"))
        #expect(detect.fields.first { $0.key == "labels" }?.control == .tags)
        let labels = try detect.input(["labels": .text("cat, dog\nbird"), "image": .files([URL(fileURLWithPath: "/tmp/a.png")])], seed: nil, stage: { $0.path })
        #expect(labels["labels"] == ["cat", "dog", "bird"])
        let read = ToolForm(schema: CapabilityFixture.schema("read_document", "input"))
        #expect(read.fields.first { $0.key == "pages" }?.control == .pages)
        let document: FieldValue = .files([URL(fileURLWithPath: "/tmp/a.pdf")])
        #expect(try read.input(["pages": .text("1-3, 5"), "document": document], seed: nil, stage: { $0.path })["pages"] == [0, 1, 2, 4])
        #expect(throws: OperationForm.Problem(key: "pages", message: "Type pages as 1-3, 5.")) {
            try read.input(["pages": .text("three"), "document": document], seed: nil, stage: { $0.path })
        }
        let extract = ToolForm(schema: CapabilityFixture.schema("extract", "input"), main: ["text": "Text to read"])
        #expect(extract.fields.first { $0.key == "json_schema" }?.control == .schema)
        #expect(
            SchemaField.fields(SchemaField.schema([.init(name: "name", kind: .text), .init(name: "age", kind: .integer)])) == [
                .init(name: "name", kind: .text), .init(name: "age", kind: .integer)
            ])
        #expect(SchemaField.fields("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"object\"}}}") == nil, "a richer schema is edited as JSON")
        let translate = ToolForm(schema: CapabilityFixture.schema("translate", "input"), main: ["text": "Text to translate"])
        #expect(try translate.input(["text": .text(" Hello "), "target_lang": .text("de")], seed: nil, stage: { $0.path })["text"] == ["Hello"])
        let interpolate = ToolForm(schema: CapabilityFixture.schema("interpolate_video", "input"))
        #expect(interpolate.fields.first { $0.key == "t" }?.control == .numbers(min: 0, max: 1))
        #expect(throws: OperationForm.Problem.self) { try interpolate.input(["t": .text("0.5, 2")], seed: nil, stage: { $0.path }) }
    }

    // MARK: Subject and spatial prompts

    @Test func theWellRoutesEachFileToTheFieldThatTakesIt() throws {
        let photo = try CapabilityFixture.photo(scratch.appendingPathComponent("in/a.jpg"))
        let photo2 = try CapabilityFixture.photo(scratch.appendingPathComponent("in/b.png"))
        let pdf = try CapabilityFixture.pdf(scratch.appendingPathComponent("in/report.pdf"))
        let video = try CapabilityFixture.video(scratch.appendingPathComponent("in/clip.mp4"))
        // Depth: one picture to `image`, several to `images`; the other is cleared.
        let (depth, _) = CapabilityWindows.make(
            .depth, models: [CapabilityFixture.choice("depth-fixture")], selected: "depth-fixture", client: { nil }, inbox: scratch, output: scratch)
        depth.preview(operations: [CapabilityFixture.descriptor("estimate_depth")])
        depth.setSubject([photo])
        #expect(depth.value("image").files == [photo] && depth.value("images").files.isEmpty && depth.readyToRun)
        depth.setSubject([photo, photo2])
        #expect(depth.value("images").files == [photo, photo2] && depth.value("image").files.isEmpty && depth.readyToRun)
        // Extract: a PNG to the image field (not the document field that also takes PNG), a PDF to the document.
        let (extract, _) = CapabilityWindows.make(
            .extract, models: [CapabilityFixture.choice("extract-fixture")], selected: "extract-fixture", client: { nil }, inbox: scratch, output: scratch)
        extract.preview(operations: [CapabilityFixture.descriptor("extract")])
        extract.setSubject([photo2])
        #expect(extract.value("image").files == [photo2] && extract.value("document").files.isEmpty)
        extract.setSubject([pdf])
        #expect(extract.value("document").files == [pdf] && extract.value("image").files.isEmpty)
        #expect(!extract.readyToRun, "fields or labels are still needed")
        extract.set("labels", .text("person"))
        #expect(extract.readyToRun && extract.canRun)
        // Smooth: a video, or two frames.
        let (smooth, _) = CapabilityWindows.make(
            .smooth, models: [CapabilityFixture.choice("rife-fixture")], selected: "rife-fixture", client: { nil }, inbox: scratch, output: scratch)
        smooth.preview(operations: [CapabilityFixture.descriptor("interpolate_video")])
        smooth.setSubject([video])
        #expect(smooth.value("video").files == [video] && smooth.canRun)
        smooth.setSubject([photo, photo2])
        #expect(smooth.value("frames").files == [photo, photo2] && smooth.value("video").files.isEmpty)
        // Something the window does not take: a line, nothing changes.
        let text = scratch.appendingPathComponent("in/notes.txt")
        try Data("x".utf8).write(to: text)
        smooth.setSubject([text])
        #expect(smooth.problem?.message.hasPrefix("notes.txt is not something") == true && smooth.value("frames").files.count == 2)
        // Upscale switches to its video operation for a video.
        let (upscale, _) = CapabilityWindows.make(
            .upscale, models: [CapabilityFixture.choice("upscale-fixture")], selected: "upscale-fixture", client: { nil }, inbox: scratch, output: scratch)
        upscale.preview(operations: [CapabilityFixture.descriptor("upscale_image"), CapabilityFixture.descriptor("upscale_video")])
        #expect(upscale.operation == "upscale_image")
        upscale.setSubject([video])
        #expect(upscale.operation == "upscale_video" && upscale.value("video").files == [video])
        upscale.setSubject([photo])
        #expect(upscale.operation == "upscale_image" && upscale.value("image").files == [photo])
    }

    @Test func pointsAndBoxesAreNormalizedToThePictureNotItsLetterbox() throws {
        // A 1600 × 900 picture in a 400 × 400 view: 400 × 225, centred with 87.5 pt above and below.
        let box = Letterbox(container: CGSize(width: 400, height: 400), pixels: CGSize(width: 1600, height: 900))
        #expect(box.rect == CGRect(x: 0, y: 87.5, width: 400, height: 225))
        #expect(box.normalized(CGPoint(x: 100, y: 87.5 + 112.5)) == CGPoint(x: 0.25, y: 0.5))
        #expect(box.normalized(CGPoint(x: 200, y: 20)) == nil, "a click in the letterbox is not on the picture")
        #expect(box.view(CGPoint(x: 1, y: 1)) == CGPoint(x: 400, y: 312.5))
        // The session keeps them as the schema's JSON; a new picture clears them.
        let (session, _) = CapabilityWindows.make(
            .select, models: [CapabilityFixture.choice("sam-fixture")], selected: "sam-fixture", client: { nil }, inbox: scratch, output: scratch)
        let select = try #require(session as? SelectSession)
        select.preview(operations: [CapabilityFixture.descriptor("segment")])
        select.pointsBinding.wrappedValue = [SpatialPoint(x: 0.25, y: 0.5, label: "foreground"), SpatialPoint(x: 0.7, y: 0.4, label: "background")]
        select.boxesBinding.wrappedValue = [SpatialBox(xmin: 0.1, ymin: 0.2, xmax: 0.5, ymax: 0.9)]
        let points = SpatialValue.points(select.value("points").text)
        #expect(points.count == 2 && points[1].excludes && select.hasPrompts)
        #expect(PromptTools.modes(select) == [.include, .exclude, .box])
        select.setSubject([try CapabilityFixture.photo(scratch.appendingPathComponent("p/one.jpg"))])
        #expect(!select.hasPrompts)
    }

    // MARK: End to end

    @Test func transcribeShowsSpeakersAndTimesAndSavesSubtitles() async throws {
        let lab = try await lab(.transcribe, "whisper-fixture", as: TranscribeSession.self)
        defer { lab.stop() }
        #expect(lab.session.operations.map(\.name) == ["transcribe", "diarize"])
        #expect(!lab.session.canRun, "nothing to transcribe yet")
        lab.session.setSubject([try CapabilityFixture.clip(scratch.appendingPathComponent("clips/meeting.wav"))])
        lab.session.set("diarize", .toggle(true))
        lab.session.set("language", .text("en"))
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running }
        #expect(lab.session.problem == nil)
        let outcome = try #require(lab.session.current)
        let segments = TranscriptSegment.list(outcome.reply)
        #expect(segments.count == 3 && segments.allSatisfy(\.timed))
        #expect(TranscriptSegment.speakers(segments, reply: outcome.reply) == ["SPEAKER_00", "SPEAKER_01"])
        #expect(TranscriptSegment.speakerName("SPEAKER_01", in: ["SPEAKER_00", "SPEAKER_01"]) == "Speaker 2")
        #expect(TranscriptSegment.srt(segments, speakers: ["SPEAKER_00", "SPEAKER_01"]).hasPrefix("1\n00:00:00,000 --> 00:00:02,400\nSpeaker 1: Thanks for coming in.\n"))
        let sent = try #require(await lab.service.runInputs.last)
        #expect(sent["diarize"] == true && sent["language"] == "en" && sent["audio"]?["path"]?.string?.hasSuffix("meeting.wav") == true)
        #expect(lab.session.figures().map(\.title) == ["Time", "Peak memory", "Length", "vs real time"])
        #expect(lab.session.figures()[2].value != nil)
        // Who spoke when: segments without words.
        lab.session.select("diarize")
        #expect(lab.session.runTitle(lab.session.descriptor) == "Find Speakers")
        lab.session.setSubject([try CapabilityFixture.clip(scratch.appendingPathComponent("clips/call.wav"))])
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running }
        #expect(lab.session.outcomes.count == 2 && TranscriptSegment.list(lab.session.current!.reply).allSatisfy { $0.text.isEmpty && $0.speaker != nil })
    }

    @Test func liveModeStreamsTurnsAndKeepsTheTranscript() async throws {
        let lab = try await lab(.transcribe, "whisper-fixture", as: TranscribeSession.self)
        defer { lab.stop() }
        let socket = FakeLiveSocket()
        let source = FakeLiveSource()
        lab.session.live.makeSocket = { _, _ in socket }
        lab.session.live.makeSource = { source }
        lab.session.turnModels = ["smart-turn-fixture"]
        lab.session.liveAvailable = true
        lab.session.liveMode = true
        lab.session.startLive()
        try await waitUntil(timeout: 5) { source.started && socket.sent.count >= 1 }
        let update = try #require(socket.sent.first)
        #expect(update["type"] as? String == "session.update")
        let input = (((update["session"] as? [String: Any])?["audio"] as? [String: Any])?["input"] as? [String: Any])
        #expect((input?["turn_detection"] as? [String: Any])?["model"] as? String == "smart-turn-fixture")
        #expect((input?["transcription"] as? [String: Any])?["model"] as? String == "whisper-fixture")
        socket.push(["type": "session.updated", "session": [:]])
        source.emit(Data(repeating: 0x40, count: 4800))
        socket.push(["type": "input_audio_buffer.speech_started", "item_id": "item_1"])
        try await waitUntil(timeout: 5) { socket.sent.contains { $0["type"] as? String == "input_audio_buffer.append" } }
        socket.push(["type": "conversation.item.input_audio_transcription.delta", "delta": "Hello"])
        socket.push(["type": "conversation.item.input_audio_transcription.completed", "transcript": "Hello there."])
        try await waitUntil(timeout: 5) { lab.session.live.lines == ["Hello there."] && lab.session.live.partial.isEmpty }
        // A refused chunk after setup shows its words and listening goes on.
        socket.push(["type": "error", "error": ["code": "invalid_request", "message": "One chunk was refused."]])
        try await waitUntil(timeout: 5) { lab.session.live.problem == "One chunk was refused." }
        #expect(lab.session.live.listening)
        // More speech, then Stop: the unfinished turn is committed after its audio and kept once its text arrives.
        source.emit(Data(repeating: 0x40, count: 4800))
        socket.push(["type": "input_audio_buffer.speech_started", "item_id": "item_2"])
        lab.session.stopLive()
        #expect(source.stopped && lab.session.live.stopping)
        try await waitUntil(timeout: 5) { socket.sent.last?["type"] as? String == "input_audio_buffer.commit" }
        let types = socket.sent.compactMap { $0["type"] as? String }
        #expect(types == ["session.update", "input_audio_buffer.append", "input_audio_buffer.append", "input_audio_buffer.commit"], "chunks and the commit go in order")
        #expect(lab.session.current?.operation != "live", "nothing is kept before the last turn's text")
        socket.push(["type": "input_audio_buffer.committed", "item_id": "item_2"])
        socket.push(["type": "conversation.item.input_audio_transcription.completed", "item_id": "item_2", "transcript": "Goodbye."])
        try await waitUntil(timeout: 5) { lab.session.current?.operation == "live" }
        #expect(lab.session.current?.text == "Hello there.\nGoodbye." && !lab.session.live.stopping)
        // A refusal from the service ends listening with its words.
        lab.session.startLive()
        try await waitUntil(timeout: 5) { lab.session.live.listening }
        socket.push(["type": "error", "error": ["message": "No transcription model is loaded."]])
        try await waitUntil(timeout: 5) { !lab.session.live.listening }
        #expect(lab.session.live.problem == "No transcription model is loaded.")
    }

    @Test func liveDiarizationShowsWhoSpokeWhenAndKeepsTheFinalTimeline() async throws {
        let lab = try await lab(.transcribe, "whisper-fixture", as: TranscribeSession.self)
        defer { lab.stop() }
        let socket = FakeLiveSocket()
        let source = FakeLiveSource()
        lab.session.live.makeSocket = { _, _ in socket }
        lab.session.live.makeSource = { source }
        lab.session.turnModels = ["smart-turn-fixture"]
        lab.session.liveAvailable = true
        // A streaming diarizer (as Nemotron declares it): live mode streams its speaker timeline.
        lab.session.operations = ["diarize", "diarize_session"].map {
            OperationDescriptor(name: $0, title: $0, inputSchema: [:], outputSchema: nil, output: "json", stream: false, timeout: 120)
        }
        #expect(lab.session.offersLive && lab.session.liveDiarizes)
        lab.session.liveMode = true
        lab.session.startLive()
        try await waitUntil(timeout: 5) { source.started && socket.sent.count >= 1 }
        let input = (((socket.sent.first?["session"] as? [String: Any])?["audio"] as? [String: Any])?["input"] as? [String: Any])
        #expect(input?["turn_detection"] == nil, "one recording keeps its speaker numbers: no turns")
        socket.push(["type": "session.updated", "session": [:]])
        source.emit(Data(repeating: 0x40, count: 4800))
        socket.push(["type": "input_audio_buffer.speech_started", "item_id": "item_1"])
        socket.push(["type": "arsenal.diarization.snapshot", "final": false, "segments": [["start": 0, "end": 1.2, "speaker": "speaker-0"]], "speakers": ["speaker-0"]])
        try await waitUntil(timeout: 5) { (lab.session.live.timeline?["segments"]?.array?.count ?? 0) == 1 }
        lab.session.stopLive()
        try await waitUntil(timeout: 5) { socket.sent.last?["type"] as? String == "input_audio_buffer.commit" }
        socket.push(["type": "input_audio_buffer.committed", "item_id": "item_1"])
        socket.push([
            "type": "arsenal.diarization.snapshot", "final": true,
            "segments": [["start": 0, "end": 1.0, "speaker": "speaker-0"], ["start": 1.1, "end": 2.0, "speaker": "speaker-1"]], "speakers": ["speaker-0", "speaker-1"]
        ])
        socket.push(["type": "conversation.item.input_audio_transcription.completed", "item_id": "item_1", "transcript": ""])
        try await waitUntil(timeout: 5) { lab.session.current?.operation == "live" }
        let outcome = try #require(lab.session.current)
        let segments = TranscriptSegment.list(outcome.reply)
        #expect(segments.map(\.end) == [1.0, 2.0], "the finished timeline replaces the live one")
        #expect(TranscriptSegment.speakers(segments, reply: outcome.reply) == ["speaker-0", "speaker-1"])
        #expect(outcome.text.contains("Speaker 2"))
    }

    @Test func separateSavesEachStemAndListsThemToPlay() async throws {
        let lab = try await lab(.separate, "demucs-fixture", as: SeparateSession.self, resident: false)
        defer { lab.stop() }
        lab.session.setSubject([try CapabilityFixture.clip(scratch.appendingPathComponent("clips/song.wav"), seconds: 1.5)])
        lab.session.set("stems", .text("vocals, drums"))
        #expect(lab.session.runsAsJob(try #require(lab.session.descriptor)), "separation runs as a job")
        lab.session.start()
        try await waitUntil(timeout: 15) { !lab.session.running }
        #expect(lab.session.problem == nil)
        let outcome = try #require(lab.session.current)
        #expect(outcome.files.map(\.id) == ["vocals", "drums"] && outcome.files.map(\.label) == ["Vocals", "Drums"])
        #expect(outcome.files.allSatisfy { $0.url.deletingLastPathComponent() == lab.support.appendingPathComponent("Results") && AudioShape.read($0.url) != nil })
        #expect(outcome.files[0].url.lastPathComponent.hasSuffix("demucs-fixture song Vocals.wav"))
        #expect(await lab.service.calls.contains("load demucs-fixture -"), "a job loads the model first, through the memory check")
    }

    @Test func translateReadAndExtractAnswerAsTheirOutputsDeclare() async throws {
        let translate = try await lab(.translate, "translate-fixture", as: TranslateSession.self)
        defer { translate.stop() }
        translate.session.set("text", .text("Good morning"))
        #expect(!translate.session.canRun, "the language to translate into comes first")
        translate.session.set("target_lang", .text("de"))
        translate.session.start()
        try await waitUntil(timeout: 10) { !translate.session.running }
        let t = try #require(translate.session.current)
        #expect(TranslateSession.translations(t).map(\.text) == ["[de] Good morning"] && TranslateSession.translations(t).first?.detected == "en")

        let read = try await lab(.document, "ocr-fixture", as: DocumentSession.self)
        defer { read.stop() }
        read.session.setSubject([try CapabilityFixture.pdf(scratch.appendingPathComponent("docs/report.pdf"))])
        read.session.set("pages", .text("1-2"))
        read.session.start()
        try await waitUntil(timeout: 10) { !read.session.running }
        let d = try #require(read.session.current)
        #expect(DocumentSession.pages(d).count == 2 && d.input["pages"] == [0, 1])
        let blocks = MarkdownBlocks.blocks(DocumentSession.pages(d)[0].markdown)
        #expect(blocks.first == .heading(1, "Quarterly report"))
        #expect(blocks.contains(.table([["Region", "Q1", "Q2"], ["North", "12", "14"], ["South", "9", "11"]])))
        #expect(
            blocks.filter {
                if case .item = $0 { return true }; return false
            }.count == 2)

        let extract = try await lab(.extract, "extract-fixture", as: ExtractSession.self)
        defer { extract.stop() }
        extract.session.set("text", .text("Ada Lovelace, 36, mathematician"))
        extract.session.set("json_schema", .text(SchemaField.schema([.init(name: "name", kind: .text), .init(name: "age", kind: .integer)])))
        extract.session.start()
        try await waitUntil(timeout: 10) { !extract.session.running }
        let e = try #require(extract.session.current)
        #expect(ExtractSession.data(e)?["name"] == "name value" && ExtractSession.violation(e) == nil)
        // An answer that breaks the fields asked for says so.
        var broken = e
        broken.reply = ["data": ["name": 3]]
        #expect(ExtractSession.violation(broken)?.hasPrefix("Does not match the fields") == true)
        // Labels instead of fields: entities.
        extract.session.set("json_schema", .text(""))
        extract.session.set("labels", .text("person"))
        extract.session.start()
        try await waitUntil(timeout: 10) { !extract.session.running }
        #expect(ExtractSession.entities(try #require(extract.session.current)) == [.init(label: "person", text: "Ada Lovelace", score: 0.9)])
    }

    @Test func searchRanksWithARerankerAndWithEmbeddings() async throws {
        let lab = try await lab(.search, "rerank-fixture", as: SearchSession.self)
        defer { lab.stop() }
        lab.session.set("query", .text("red apples"))
        lab.session.set("documents", .text("bananas are yellow\nred apples are sweet\napples grow on trees"))
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running }
        let reranked = SearchSession.ranked(try #require(lab.session.current))
        #expect(reranked.map(\.text) == ["red apples are sweet", "apples grow on trees", "bananas are yellow"])
        // The embedding model in the same window: query and passages embedded, ranked by cosine similarity, its own
        // options only (no raw input field).
        lab.session.choose("embed-fixture")
        try await waitUntil(timeout: 10) { !lab.session.loading }
        #expect(lab.session.form?.main?.key == "query" && !lab.session.optionFields.contains { $0.key == "input" })
        #expect(lab.session.optionFields.map(\.key).contains("documents"))
        lab.session.set("documents", .text("bananas are yellow\nred apples are sweet"))
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running }
        #expect(lab.session.problem == nil)
        let embedded = SearchSession.ranked(try #require(lab.session.current))
        #expect(embedded.first?.text == "red apples are sweet" && embedded.count == 2)
        // Query and documents went as their own tasks.
        let tasks = await lab.service.runInputs.suffix(2).compactMap { $0["task"]?.string }
        #expect(tasks == ["query", "document"])
        #expect(abs(SearchSession.cosine([1, 0], [1, 0]) - 1) < 1e-9 && SearchSession.cosine([1, 0], [0, 1]) == 0)
    }

    @Test func selectKeepsTheModelsSessionBetweenClicksAndReleasesIt() async throws {
        let lab = try await lab(.select, "sam-fixture", as: SelectSession.self)
        defer { lab.stop() }
        let photo = try CapabilityFixture.photo(scratch.appendingPathComponent("photos/dog.jpg"))
        lab.session.setSubject([photo])
        #expect(!lab.session.canRun, "a point, a box or a description first")
        lab.session.pointsBinding.wrappedValue = [SpatialPoint(x: 0.4, y: 0.5, label: "foreground")]
        #expect(lab.session.canRun)
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running && lab.session.current != nil }
        #expect(lab.session.problem == nil)
        #expect(await lab.service.runLog() == ["sam-fixture segment_session", "sam-fixture segment"])
        let session = try #require(lab.session.sessionID)
        let first = try #require(await lab.service.runInputs.last)
        #expect(first["session_id"]?.string == session && first["image"] == nil, "the picture is sent once, to the session")
        #expect(first["points"]?.array?.first?["x"] == 0.4)
        #expect(lab.session.canvasMasks.count == 1 && lab.session.masks.count == 1)
        // A second click selects again by itself, through the same session.
        lab.session.pointsBinding.wrappedValue += [SpatialPoint(x: 0.6, y: 0.5, label: "background")]
        try await waitUntil(timeout: 10) { await lab.service.runLog().count == 3 && !lab.session.running }
        #expect(await lab.service.runInputs.last?["session_id"]?.string == session)
        // Several masks: the best one shows first, the cut-out keeps only the selection.
        lab.session.set("multimask", .toggle(true))
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running && lab.session.masks.count == 3 }
        #expect(lab.session.maskIndex == 0 && lab.session.score(2) == 0.75)
        lab.session.view = .cutout
        let cut = try #require(lab.session.shownResultFile())
        #expect(try BackgroundToolTests.alpha(cut, x: 2, y: 2) == 0)
        // A new picture releases the session.
        lab.session.setSubject([try CapabilityFixture.photo(scratch.appendingPathComponent("photos/cat.jpg"))])
        #expect(lab.session.sessionID == nil)
        try await waitUntil(timeout: 10) { await lab.service.runInputs.last?["action"]?.string == "close" }
        #expect(await lab.service.runInputs.last?["session_id"]?.string == session)
    }

    @Test func aStaleSessionIsReopenedOnce() async throws {
        let lab = try await lab(.select, "sam-fixture", as: SelectSession.self)
        defer { lab.stop() }
        let photo = try CapabilityFixture.photo(scratch.appendingPathComponent("photos/dog.jpg"))
        lab.session.setSubject([photo])
        lab.session.pointsBinding.wrappedValue = [SpatialPoint(x: 0.4, y: 0.5, label: "foreground")]
        // The model was reloaded: the window's session is gone there.
        lab.session.preview(session: "gone", for: photo)
        await lab.service.setRunner("segment") { params in
            if params["session_id"]?.string == "gone" { throw WorkerFailure(code: "invalid_request", message: "Unknown session gone.") }
            return try CapabilityFixture.run("segment", params: params)
        }
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running && lab.session.current != nil }
        #expect(lab.session.problem == nil && lab.session.sessionID != "gone" && lab.session.sessionID != nil)
        #expect(await lab.service.runLog() == ["sam-fixture segment", "sam-fixture segment_session", "sam-fixture segment"])
    }

    @Test func trackingAVideoIsAJobWithProgressThatStopCancels() async throws {
        let lab = try await lab(.select, "sam-fixture", as: SelectSession.self)
        defer { lab.stop() }
        lab.session.select("track_video")
        #expect(lab.session.operation == "track_video")
        let video = try CapabilityFixture.video(scratch.appendingPathComponent("videos/walk.mp4"))
        lab.session.setSubject([video])
        _ = await VideoInfo.load(video)
        #expect(VideoInfo.cachedSize(video) == PixelSize(width: 480, height: 270) && VideoInfo.cachedRate(video) == 12)
        lab.session.boxesBinding.wrappedValue = [SpatialBox(xmin: 0.1, ymin: 0.2, xmax: 0.5, ymax: 0.9)]
        // Prompts on a later frame go as that frame's prompt update.
        lab.session.frame = 6
        await lab.service.setDelay("track_video", .seconds(5))
        lab.session.start()
        try await waitUntil(timeout: 10) { lab.session.run?.step == 1 }
        #expect(lab.session.run?.total == 2, "the job's progress reaches the window")
        lab.session.stop()
        #expect(!lab.session.running && lab.session.run?.cancelled == true)
        try await waitUntil(timeout: 10) {
            let records = (try? FileManager.default.contentsOfDirectory(at: lab.support.appendingPathComponent("jobs/operations"), includingPropertiesForKeys: nil)) ?? []
            return records.contains { (try? String(contentsOf: $0.appendingPathComponent("record.json"), encoding: .utf8))?.contains("\"canceled\"") == true }
        }
        let sent = try #require(await lab.service.runInputs.last { $0["prompt_updates"] != nil || $0["video"] != nil || $0["session_id"] != nil && $0["action"] == nil })
        let update = try #require(sent["prompt_updates"]?.array?.first)
        #expect(update["frame_index"] == 6 && update["boxes"] != nil && sent["boxes"] == nil)
        // Without the delay it finishes: the mask video plays.
        await lab.service.setDelay("track_video", nil)
        lab.session.start()
        try await waitUntil(timeout: 20) { !lab.session.running && lab.session.current != nil }
        #expect(lab.session.current?.file("video/")?.url.pathExtension == "mp4")
    }

    /// The model reads a session's video from its staged path on every track_video, so the staged copy stays
    /// until the session is released (create → track → track → release).
    @Test func aVideoSessionKeepsItsStagedVideoUntilReleased() async throws {
        let lab = try await lab(.select, "sam-fixture", as: SelectSession.self)
        defer { lab.stop() }
        lab.session.select("track_video")
        let video = try CapabilityFixture.video(scratch.appendingPathComponent("videos/walk.mp4"))
        lab.session.setSubject([video])
        _ = await VideoInfo.load(video)
        lab.session.boxesBinding.wrappedValue = [SpatialBox(xmin: 0.1, ymin: 0.2, xmax: 0.5, ymax: 0.9)]
        lab.session.start()
        try await waitUntil(timeout: 20) { !lab.session.running && lab.session.current != nil }
        #expect(lab.session.problem == nil)
        #expect(await lab.service.runLog() == ["sam-fixture video_session", "sam-fixture track_video"])
        let create = try #require(await lab.service.runInputs.first { $0["action"]?.string == "create" })
        let staged = URL(fileURLWithPath: try #require(create["video"]?["path"]?.string))
        let session = try #require(lab.session.sessionID)
        #expect(FileManager.default.fileExists(atPath: staged.path), "the session's video outlives the create request")
        // A second track reuses the session and the same staged file.
        lab.session.boxesBinding.wrappedValue = [SpatialBox(xmin: 0.2, ymin: 0.2, xmax: 0.6, ymax: 0.9)]
        lab.session.start()
        try await waitUntil(timeout: 20) { await lab.service.runLog().count == 3 && !lab.session.running }
        #expect(await lab.service.runInputs.last?["session_id"]?.string == session)
        #expect(FileManager.default.fileExists(atPath: staged.path))
        // Release closes the session, then removes the staged copy.
        lab.session.release()
        #expect(lab.session.sessionID == nil && lab.session.sessionFolder == nil)
        try await waitUntil(timeout: 10) { await lab.service.runInputs.last?["action"]?.string == "close" }
        try await waitUntil(timeout: 10) { !FileManager.default.fileExists(atPath: staged.path) }
    }

    /// Hour-long file operations (TRELLIS, SeedVR2, RIFE declare 3600 s) run as durable jobs with polled progress even
    /// when they do not stream; short ones keep the direct request.
    @Test func longFileOperationsRunAsJobs() {
        let session = CapabilitySession(
            tool: .upscale, models: [], selected: "", client: { nil }, inbox: scratch.appendingPathComponent("inbox"),
            outputDirectory: scratch.appendingPathComponent("Results"), scratch: scratch.appendingPathComponent("window"))
        func op(stream: Bool, timeout: Double?, output: String = "file") -> OperationDescriptor {
            OperationDescriptor(name: "upscale_video", title: "Upscale video", inputSchema: [:], output: output, stream: stream, timeout: timeout)
        }
        #expect(session.runsAsJob(op(stream: false, timeout: 3600)))
        #expect(session.runsAsJob(op(stream: true, timeout: 120)))
        #expect(!session.runsAsJob(op(stream: false, timeout: 590)))
        #expect(!session.runsAsJob(op(stream: false, timeout: nil)))
        #expect(!session.runsAsJob(op(stream: false, timeout: 3600, output: "json")))
    }

    @Test func findConvertsScreenPixelsAndDrawsEachDetection() async throws {
        let lab = try await lab(.find, "dino-fixture", as: FindSession.self)
        defer { lab.stop() }
        let shot = try CapabilityFixture.photo(scratch.appendingPathComponent("shots/screen.png"), width: 1440, height: 900)
        lab.session.setSubject([shot])
        lab.session.set("labels", .text("button, field"))
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running }
        let found = lab.session.detections(try #require(lab.session.current))
        #expect(found.map(\.label) == ["button", "field"] && found[0].box?.minX == 0.12)
        #expect(lab.session.figures().last?.value == "2")
        // Screen grounding answers in pixels: normalized with the reply's own image size.
        lab.session.select("ground_screen")
        lab.session.setSubject([shot])
        lab.session.set("instruction", .text("the Save button"))
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running && lab.session.outcomes.count == 2 }
        let grounded = try #require(lab.session.current)
        let point = try #require(lab.session.detections(grounded).first?.point)
        #expect(abs(point.x - 0.7) < 1e-9 && abs(point.y - 0.2) < 1e-9)
        #expect(FindView.place(Detection(id: 0, label: "x", score: 1, box: nil, point: point), pixels: PixelSize(width: 1440, height: 900)) == "x 1008, y 180")
    }

    @Test func depthUpscaleSmoothAndMake3DShowWhatTheyReturn() async throws {
        let photo = try CapabilityFixture.photo(scratch.appendingPathComponent("photos/room.jpg"), width: 320, height: 240)

        let depth = try await lab(.depth, "depth-fixture", as: DepthSession.self)
        defer { depth.stop() }
        depth.session.setSubject([photo])
        depth.session.set("intrinsics", .toggle(true))
        depth.session.start()
        try await waitUntil(timeout: 10) { !depth.session.running }
        let d = try #require(depth.session.current)
        #expect(!DepthSession.metric(d) && DepthSession.legend(d, map: d.files[0]) == ("Near", "Far") && DepthSession.focal(d) == 812)
        let rendered = try #require(DepthImage.render(d.files[0].url, colour: true, nearIsHigh: true))
        #expect(rendered.high > rendered.low && rendered.image.size == NSSize(width: 320, height: 240))

        let upscale = try await lab(.upscale, "upscale-fixture", as: UpscaleSession.self)
        defer { upscale.stop() }
        upscale.session.setSubject([photo])
        upscale.session.set("scale", .choice(1))
        upscale.session.start()
        try await waitUntil(timeout: 15) { !upscale.session.running }
        #expect(upscale.session.problem == nil)
        #expect(PixelSize.read(try #require(upscale.session.current?.file("image/")).url) == PixelSize(width: 1280, height: 960))
        #expect(upscale.session.figures().last?.value == "1280 \u{00d7} 960")

        let smooth = try await lab(.smooth, "rife-fixture", as: SmoothSession.self)
        defer { smooth.stop() }
        smooth.session.setSubject([photo, try CapabilityFixture.photo(scratch.appendingPathComponent("photos/room2.jpg"), width: 320, height: 240)])
        smooth.session.start()
        try await waitUntil(timeout: 15) { !smooth.session.running }
        #expect(smooth.session.current?.files.count == 3 && smooth.session.current?.files.allSatisfy(\.isImage) == true)

        let make = try await lab(.make3d, "trellis-fixture", as: Make3DSession.self)
        defer { make.stop() }
        make.session.setSubject([photo])
        make.session.start()
        try await waitUntil(timeout: 15) { !make.session.running }
        #expect(make.session.problem == nil)
        let model = try #require(make.session.current.flatMap(Make3DSession.model))
        #expect(model.url.pathExtension == "glb" && Make3DSession.triangles(model.url) == 12)
        #expect(make.session.shownFile == model.url)
    }

    @Test func aGLBIsReadIntoSceneKitAndABrokenOneIsAStatedFailure() throws {
        let scene = try GLB.scene(GLBMaker.data())
        var triangles = 0, materials: [SCNMaterial] = []
        scene.rootNode.enumerateHierarchy { node, _ in
            triangles += node.geometry?.elements.reduce(0) { $0 + $1.primitiveCount } ?? 0
            materials += node.geometry?.materials ?? []
        }
        #expect(triangles == 12 && (materials.first?.diffuse.contents as? NSColor)?.redComponent ?? 0 > 0.8)
        #expect(throws: GLB.Failure.self) { try GLB.scene(Data("not a model".utf8)) }
        var truncated = GLBMaker.data()
        truncated.removeLast(40)
        #expect(throws: GLB.Failure.self) { try GLB.scene(truncated) }
        // It renders: SceneKit's own renderer draws pixels of the cube.
        let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
        renderer.scene = scene; renderer.autoenablesDefaultLighting = true
        let image = renderer.snapshot(atTime: 0, with: CGSize(width: 64, height: 64), antialiasingMode: .none)
        let rep = try #require(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
        #expect((0..<64).contains { x in (rep.colorAt(x: x, y: 32)?.alphaComponent ?? 0) > 0.5 })
    }

    // MARK: Review fixes

    @Test func aClickOnTheShownSelectionRefinesItInPlaceAndKeepsNothingUntilAsked() async throws {
        let lab = try await lab(.select, "sam-fixture", as: SelectSession.self)
        defer { lab.stop() }
        lab.session.setSubject([try CapabilityFixture.photo(scratch.appendingPathComponent("photos/dog.jpg"))])
        lab.session.pointsBinding.wrappedValue = [SpatialPoint(x: 0.4, y: 0.5, label: "foreground")]
        lab.session.start()
        try await waitUntil(timeout: 10) { !lab.session.running && lab.session.current != nil }
        let first = try #require(lab.session.current)
        #expect(lab.session.outcomes.count == 1 && lab.session.draft == nil)
        // A click: the shown selection stays (no progress card), the new one replaces it, nothing is saved or added.
        await lab.service.setDelay("segment", .milliseconds(600))
        lab.session.pointsBinding.wrappedValue += [SpatialPoint(x: 0.6, y: 0.5, label: "foreground")]
        try await waitUntil(timeout: 5) { lab.session.running }
        #expect(lab.session.quietRun && lab.session.current?.id == first.id, "the mask stays on screen while the click runs")
        try await waitUntil(timeout: 10) { !lab.session.running }
        await lab.service.setDelay("segment", nil)
        let draft = try #require(lab.session.draft)
        #expect(lab.session.current?.id == draft.id && lab.session.outcomes.count == 1)
        let results = lab.support.appendingPathComponent("Results")
        #expect(!draft.files.contains { $0.url.deletingLastPathComponent() == results }, "a click's mask is not saved")
        // Select with nothing changed keeps it: saved and in the history, no new run.
        let runs = await lab.service.runLog().count
        lab.session.start()
        #expect(lab.session.draft == nil && lab.session.outcomes.count == 2 && lab.session.outcomes[1].files.allSatisfy { $0.url.deletingLastPathComponent() == results })
        #expect(await lab.service.runLog().count == runs)
        // Copy keeps a draft too.
        lab.session.pointsBinding.wrappedValue += [SpatialPoint(x: 0.3, y: 0.3, label: "background")]
        try await waitUntil(timeout: 10) { lab.session.draft != nil && !lab.session.running }
        let board = NSPasteboard(name: .init("arsenal-select-test-\(UUID().uuidString)"))
        defer { board.releaseGlobally() }
        lab.session.copyPasteboard = board
        _ = lab.session.copyFile(try #require(lab.session.current))
        #expect(lab.session.draft == nil && lab.session.outcomes.count == 3)
        // The history shows each selection as its cut-out, not the same picture three times.
        let tiles = lab.session.outcomes.compactMap(lab.session.historyPicture)
        #expect(Set(tiles).count == 3 && tiles.allSatisfy { $0.lastPathComponent.hasSuffix("cut-out.png") })
    }

    @Test func alternativesShowAsOneChooserAndOnlyTheChosenIsSent() throws {
        let (session, _) = CapabilityWindows.make(
            .upscale, models: [CapabilityFixture.choice("upscale-fixture")], selected: "upscale-fixture", client: { nil }, inbox: scratch, output: scratch)
        session.preview(operations: [CapabilityFixture.descriptor("upscale_image")])
        let group = try #require(session.optionGroups.first)
        #expect(group.options.map(\.title) == ["Scale", "Height"] && !session.optionFields.contains { ["scale", "target_height"].contains($0.key) })
        session.set("scale", .choice(1)); session.set("target_height", .text("2000"))
        #expect(session.chosenIndex(group) == 0 && session.hidden("target_height") && session.requestValues()["target_height"] == .text(""))
        session.choose(group, 1)
        #expect(session.hidden("scale") && session.requestValues()["scale"] == .choice(nil) && session.requestValues()["target_height"] == .text("2000"))
        // Extract: a file given chooses File; a text main editor shows only for Text.
        let (extract, _) = CapabilityWindows.make(
            .extract, models: [CapabilityFixture.choice("extract-fixture")], selected: "extract-fixture", client: { nil }, inbox: scratch, output: scratch)
        extract.preview(operations: [CapabilityFixture.descriptor("extract")])
        #expect(extract.optionGroups.map(\.id) == ["source", "want"] && extract.subjectFields.isEmpty && !extract.showsMain)
        extract.setSubject([try CapabilityFixture.pdf(scratch.appendingPathComponent("groups/a.pdf"))])
        #expect(extract.subjectFields.map(\.key).contains("document") && extract.hidden("text"))
        // Smooth Video: positions only for two frames.
        let (smooth, _) = CapabilityWindows.make(
            .smooth, models: [CapabilityFixture.choice("rife-fixture")], selected: "rife-fixture", client: { nil }, inbox: scratch, output: scratch)
        smooth.preview(operations: [CapabilityFixture.descriptor("interpolate_video")])
        #expect(smooth.hidden("t"))
        smooth.setSubject([
            try CapabilityFixture.photo(scratch.appendingPathComponent("groups/f1.png")), try CapabilityFixture.photo(scratch.appendingPathComponent("groups/f2.png"))
        ])
        #expect(!smooth.hidden("t"))
        // Words for unset fields: what Auto means and what the field takes, never a range typed into the field.
        #expect(FieldControl.rangeHelp(unset: true, declared: false, "1\u{2013}2048") == "Auto: the model decides. Takes 1\u{2013}2048.")
        #expect(FieldControl.placeholders["focal_length"] == "Estimated" && FieldControl.placeholders["start_frame"] == "First")
    }

    @Test func aModelNotOnThisMacIsInItsWindowAndItsMenuEntryKeepsItsName() async throws {
        let support = scratch.appendingPathComponent("not-installed")
        let missing = CapabilityFixture.row("depth-fixture", state: .notInstalled)
        let service = await CapabilityFixture.service(support: support)
        let center = ToolCenter(client: { nil }, playback: FixturePlayback())
        _ = await center.requestAdmissions(rows: [missing], service: service)
        #expect(center.declared["depth-fixture"]?.names == ["estimate_depth"] && center.windowTools(missing) == [.depth])
        let choice = try #require(center.models([missing], for: .depth).first)
        // The window's model choice offers the download itself (the table's Get), not a trip to Models.
        #expect(choice.unavailable?.message == "Depth Fixture isn\u{2019}t on this Mac yet." && choice.unavailable?.actionTitle?.hasPrefix("Download") == true)
        #expect(choice.unavailable?.action == .download)
        // The window opens complete on its first frame from the operations already read.
        let (session, _) = CapabilityWindows.make(.depth, models: [choice], selected: "depth-fixture", client: { nil }, inbox: scratch, output: scratch)
        session.cachedOperations = { center.descriptors[$0] }
        session.prepare()
        #expect(!session.loading && session.operation == "estimate_depth" && session.subjectFields.map(\.key) == ["image", "images"])
        session.discard()
    }

    // MARK: Changed windows

    @Test func chatOffersAttachmentsAndJSONAnswersOnlyWhereTheSchemaDeclaresThem() throws {
        let full = ChatCapabilities(schema: CapabilityFixture.schemaFromKind("chat", "chat"))
        #expect(full.images && full.audioFormats == ["wav", "mp3"] && full.jsonSchema)
        let qwen = ChatCapabilities(schema: SoundFixture.schema("qwen3.8-27b", "chat"))
        #expect(!qwen.attaches && !qwen.jsonSchema, "a text-only model shows no paperclip and no braces")
        let session = ChatSession(model: ToolModel(id: "m", name: "Vision", kind: "chat", dtype: "int4", mode: "exact"), client: { nil })
        let photo = try CapabilityFixture.photo(scratch.appendingPathComponent("chat/a.png"))
        let clip = try CapabilityFixture.clip(scratch.appendingPathComponent("chat/b.wav"), seconds: 0.2)
        let flac = scratch.appendingPathComponent("chat/c.flac")
        try Data("x".utf8).write(to: flac)
        session.preview(capabilities: qwen)
        session.attach([photo])
        #expect(session.attachments.isEmpty)
        session.preview(capabilities: full)
        session.attach([photo, clip, flac])
        #expect(session.attachments == [photo, clip], "a FLAC clip is not a format the schema takes")
        #expect(session.canSend, "attachments alone make a message")
        let parts = try #require(try ChatSession.content("What is this?", attachments: [photo, clip]) as? [[String: Any]])
        #expect(parts.map { $0["type"] as? String } == ["text", "image_url", "input_audio"])
        #expect(((parts[1]["image_url"] as? [String: Any])?["url"] as? String)?.hasPrefix("data:image/png;base64,") == true)
        #expect((parts[2]["input_audio"] as? [String: Any])?["format"] as? String == "wav")
        #expect(try ChatSession.content("plain", attachments: []) as? String == "plain")
        // A JSON answer needs a schema that is an object.
        session.preview(capabilities: full, structured: true, schema: "not json")
        #expect(!session.canSend)
        session.preview(capabilities: full, attachments: [photo], structured: true, schema: SchemaField.schema([.init(name: "count", kind: .integer)]))
        #expect(session.canSend && ChatSession.schemaObject(session.responseSchema)?["type"] as? String == "object")
    }

    @Test func removeBackgroundTakesAVideoOnlyFromAModelThatReturnsMasks() async throws {
        let support = scratch.appendingPathComponent("background-video")
        let service = FixtureService(
            rows: [CutoutFixture.row], metadata: CutoutFixture.metadata, supportDirectory: support,
            operations: [CutoutFixture.id: [CutoutFixture.operation, CapabilityFixture.operation("track_video")]])
        await CapabilityFixture.install(service)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        let port = try #require(api.port)
        let session = BackgroundSession(
            models: [ModelChoice(model: ToolModel(CutoutFixture.row), resident: true)], selected: CutoutFixture.id, client: { ArsenalClient(port: port) },
            inbox: support.appendingPathComponent("inbox"), outputDirectory: support.appendingPathComponent("Pictures"), scratch: support.appendingPathComponent("window"))
        defer { session.discard() }
        session.prepare()
        try await waitUntil(timeout: 10) { !session.loading }
        #expect(session.operations.map(\.name) == ["remove_background", "track_video"] && session.operation == "remove_background")
        let video = try CapabilityFixture.video(scratch.appendingPathComponent("bgv/walk.mp4"))
        session.setInput([video])
        #expect(session.videoMode && session.input == video && !session.canRun, "a video needs a point, a box or words")
        #expect(!session.optionFields.contains { ["video", "points", "boxes", "session_id", "prompt_updates"].contains($0.key) })
        session.set("points", .text(SpatialValue.text([SpatialPoint(x: 0.4, y: 0.5, label: "foreground")])))
        #expect(session.canRun)
        session.start()
        try await waitUntil(timeout: 20) { !session.running }
        #expect(session.problem == nil)
        let cut = try #require(session.current)
        #expect(cut.isVideo && cut.mask == nil && cut.url.lastPathComponent.hasSuffix(" walk mask.mp4"))
        // A photo goes back to the image operation.
        session.setInput([try CapabilityFixture.photo(scratch.appendingPathComponent("bgv/p.jpg"))])
        #expect(!session.videoMode && session.operation == "remove_background")
        // BiRefNet alone: a video is refused in words, nothing changes.
        let plain = BackgroundSession(
            models: [ModelChoice(model: ToolModel(CutoutFixture.row))], selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        plain.preview(operations: [CutoutFixture.descriptor])
        plain.setInput([video])
        #expect(plain.input == nil && plain.problem?.message == "BiRefNet removes backgrounds from images only.")
    }

    // MARK: Failures, cancel, memory

    @Test func aFailureIsALineAndARunThatMadeNothingSaysSo() async throws {
        let find = try await lab(.find, "dino-fixture", as: FindSession.self)
        defer { find.stop() }
        find.session.setSubject([try CapabilityFixture.photo(scratch.appendingPathComponent("photos/x.jpg"))])
        find.session.set("labels", .text("cat"))
        await find.service.setRunner("detect") { _ in throw WorkerFailure(code: "execution", message: "The detector ran out of anchors.") }
        find.session.start()
        try await waitUntil(timeout: 10) { !find.session.running }
        #expect(find.session.problem?.message == "The detector ran out of anchors." && find.session.current == nil && find.session.run?.cancelled == true)
        // A transcription that heard nothing is reported, not shown blank.
        let transcribe = try await lab(.transcribe, "whisper-fixture", as: TranscribeSession.self)
        defer { transcribe.stop() }
        await transcribe.service.setRunner("transcribe") { _ in ["text": ""] }
        transcribe.session.setSubject([try CapabilityFixture.clip(scratch.appendingPathComponent("clips/silence.wav"), seconds: 1)])
        transcribe.session.start()
        try await waitUntil(timeout: 10) { !transcribe.session.running }
        #expect(transcribe.session.problem?.message == "Whisper Fixture heard no speech in silence.wav." && transcribe.session.outcomes.isEmpty)
    }

    @Test func aModelThatCannotRunHereIsListedWithItsReasonAndNothingRuns() async throws {
        let (session, _) = CapabilityWindows.make(
            .depth,
            models: [
                CapabilityFixture.choice(
                    "depth-fixture", unavailable: .init(message: "Depth Fixture isn\u{2019}t on this Mac yet.", action: .download, actionTitle: "Download 1.3 GB"))
            ],
            selected: "depth-fixture", client: { nil }, inbox: scratch, output: scratch)
        session.preview(operations: [CapabilityFixture.descriptor("estimate_depth")])
        session.setSubject([try CapabilityFixture.photo(scratch.appendingPathComponent("p/a.jpg"))])
        // Not downloaded: the run button is its Download (no disabled button beside an Open Models link).
        #expect(session.blocked?.action == .download && !session.canRun && session.preparesInstead?.actionTitle == "Download 1.3 GB")
        var prepared: [String] = []
        session.prepareModel = { prepared.append($0) }
        session.performAction(.download)
        #expect(prepared == ["depth-fixture"])
        // No model at all (its row was removed while the window was open): nothing chosen, nothing to run.
        session.update(models: [])
        #expect(session.models.isEmpty && !session.canRun)
    }

    @Test(arguments: [false, true]) func aJobAsksBeforeLoadingAModelThatDoesNotFit(_ anyway: Bool) async throws {
        let respond = MemoryLoadPresenter.respond, activate = MemoryLoadPresenter.activate
        defer { MemoryLoadPresenter.respond = respond; MemoryLoadPresenter.activate = activate }
        MemoryLoadPresenter.activate = {}
        var shown: [String] = []
        MemoryLoadPresenter.respond = { alert in
            shown.append(alert.messageText); return anyway ? .alertSecondButtonReturn : .alertFirstButtonReturn
        }
        let root = scratch.appendingPathComponent("memory-\(anyway)")
        let service = TightMemoryService()
        let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
        let port = try await server.start()
        defer { server.stop() }
        for (tool, model, op) in [(ToolKind.upscale, "upscale-fixture", "upscale_image"), (.find, "dino-fixture", "detect")] {
            await service.reset()
            let (session, _) = CapabilityWindows.make(
                tool, models: [CapabilityFixture.choice(model)], selected: model, client: { ArsenalClient(port: port) }, inbox: root.appendingPathComponent("inbox"),
                output: root.appendingPathComponent("Results"), scratch: root.appendingPathComponent("window"))
            defer { session.discard() }
            session.preview(operations: [CapabilityFixture.descriptor(op)])
            session.setSubject([try CapabilityFixture.photo(scratch.appendingPathComponent("p/\(op).jpg"), width: 64, height: 48)])
            if op == "detect" { session.set("labels", .text("cat")) }
            session.start()
            try await waitUntil(timeout: 15) { !session.running }
            #expect(shown.last == "Not enough free memory to load Fixture", "\(op)")
            #expect((session.current != nil) == anyway, "\(op)")
            // Cancel says it did not run, with the alert's figures; Load Anyway runs it.
            #expect(anyway ? session.problem == nil : session.problem?.message.hasPrefix("Not run: needs ") == true && session.problem?.action == .loadAnyway, "\(op)")
            #expect(await service.forced == anyway, "\(op)")
        }
        #expect(shown.count == 2 && !LoadMemoryContext.force)
    }
}

// MARK: Test doubles

/// The capability windows the menu offers, each with the fixture model whose section header its row sits under.
private let capabilityMenuEntries: [(title: String, model: String)] = [
    ("Transcribe\u{2026}", "Whisper Fixture"), ("Separate Audio\u{2026}", "Demucs Fixture"), ("Translate\u{2026}", "Translate Fixture"),
    ("Read Document\u{2026}", "OCR Fixture"), ("Extract\u{2026}", "Extract Fixture"), ("Select Objects\u{2026}", "SAM Fixture"),
    ("Find Objects\u{2026}", "Grounding Fixture"), ("Depth\u{2026}", "Depth Fixture"), ("Upscale\u{2026}", "Upscale Fixture"), ("Smooth Video\u{2026}", "RIFE Fixture"),
    ("Make 3D\u{2026}", "TRELLIS Fixture")
]

/// The visible task entry titled `title`, when the model (by name) is one of the models it serves.
@MainActor private func capabilityMenuRow(_ app: AppDelegate, _ title: String, serving model: String) -> NSMenuItem? {
    guard let entry = app.tools.taskEntries(app.table.menuRows).first(where: { $0.title == title }), entry.rows.contains(where: { $0.name == model })
    else { return nil }
    return app.menu.items.first { !$0.isHidden && ($0.representedObject as? String) == AppDelegate.taskKey(entry.task) }
}

/// Rankings, embeddings and turn detection work in the background: no Search entry, and none of those models appears
/// anywhere in the main menu (titles, submenus or the ids items carry).
@MainActor private func expectBackgroundModelsAbsent(_ app: AppDelegate) throws {
    #expect(!app.menu.everyItem.contains { $0.title == "Search\u{2026}" })
    for id in ["rerank-fixture", "embed-fixture", "smart-turn-fixture"] {
        let row = try #require(app.table.rows.first { $0.id == id })
        #expect(app.tools.tasks(row).isEmpty, "\(id)")
        #expect(!app.menu.everyItem.contains { $0.title.contains(row.name) || ($0.representedObject as? [String])?.contains(id) == true }, "\(id)")
    }
}

/// Each window opens from its own entry, with a model that serves it chosen and that model among its choices (the
/// windows are stubbed: nothing is shown).
@MainActor private func openCapabilityMenuRows(_ app: AppDelegate) throws {
    for (title, model) in capabilityMenuEntries {
        let tool = try #require(ToolCatalog.windows.first { $0.title == title })
        let item = try #require(capabilityMenuRow(app, title, serving: model))
        #expect(item.isEnabled && item.action != nil, "\(title) for \(model) is disabled")
        app.menu.performActionForItem(at: app.menu.index(of: item))
        let open = try #require(app.tools.capability[tool], "\(title) for \(model) opened no window")
        #expect(open.session.models.contains { $0.name == model }, "\(title) offers \(model)")
    }
}

extension CapabilityFixture {
    /// A schema from Resources/kinds/<kind>/ops (for kinds outside the capability table, such as sound).
    static func schemaFromKind(_ kind: String, _ op: String) -> JSONObject {
        let url = repo.appendingPathComponent("Resources/kinds/\(kind)/ops/\(op).input.json")
        guard let data = try? Data(contentsOf: url), case .object(let object)? = try? JSONDecoder().decode(JSONValue.self, from: data) else { return [:] }
        return object
    }
}

@MainActor final class FakeLiveSource: LiveAudioSource {
    var started = false, stopped = false
    private var chunk: (@Sendable (Data) -> Void)?
    func start(_ chunk: @escaping @Sendable (Data) -> Void) async throws { started = true; self.chunk = chunk }
    func stop() { stopped = true }
    func emit(_ data: Data) { chunk?(data) }
}

final class FakeLiveSocket: LiveSocket, @unchecked Sendable {
    private let lock = NSLock()
    private var _sent: [[String: Any]] = []
    private var queue: [[String: Any]] = []
    private var waiting: CheckedContinuation<[String: Any], Error>?
    var sent: [[String: Any]] {
        lock.lock(); defer { lock.unlock() }
        return _sent
    }
    func send(_ object: [String: Any]) async throws { record(object) }
    private func record(_ object: [String: Any]) {
        lock.lock(); _sent.append(object); lock.unlock()
    }
    func receive() async throws -> [String: Any] {
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<[String: Any], Error>) in
            lock.lock()
            if queue.isEmpty {
                waiting = c; lock.unlock()
            } else {
                nonisolated(unsafe) let next = queue.removeFirst()
                lock.unlock()
                c.resume(returning: next)
            }
        }
    }
    func push(_ event: [String: Any]) {
        nonisolated(unsafe) let value = event
        lock.lock()
        if let w = waiting {
            waiting = nil; lock.unlock()
            w.resume(returning: value)
        } else {
            queue.append(value); lock.unlock()
        }
    }
    func close() {}
}

/// A model that does not fit until the load is forced; its runs answer as the fixture does once it is loaded.
private actor TightMemoryService: ModelsService {
    let refusal = LoadMemoryRefusal(model: "fixture", name: "Fixture", precision: "fp16", neededGB: 40, freeGB: 8)
    var loaded = false
    var forced = false
    func reset() { loaded = false; forced = false }
    func rows() async -> [ModelRow] { CapabilityFixture.rows.filter { ["upscale-fixture", "dino-fixture"].contains($0.id) } }
    func kinds() async -> [KindSpec] { KindSpec.defaults }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws {
        guard LoadMemoryContext.force else { throw refusal.failure }
        loaded = true; forced = true
    }
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        if !loaded {
            guard LoadMemoryContext.force else { throw refusal.failure }
            loaded = true; forced = true
        }
        return try CapabilityFixture.run(op, params: params)
    }
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
    func operations(id: String) async -> [ModelOperation] {
        (CapabilityFixture.models.first { $0.id == id }?.ops ?? []).map { CapabilityFixture.operation($0) }
    }
}
