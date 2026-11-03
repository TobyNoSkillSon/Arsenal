// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

/// The v3 tool windows' controls on never-shown windows at their real size, far off every screen (nothing is shown,
/// no cursor, focus or keyboard is touched). Each control is clicked through the real event path: AppKit's hit-test at
/// the control's laid-out centre must reach the window's SwiftUI host (no other view covers it), then the window gets
/// the mouse-down and mouse-up as it would from the trackpad (a plain tap, no hard click). Controls that open a menu,
/// a panel or Finder are hit-tested only.
@MainActor @Suite(.menuState, .serialized) final class V3ToolTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        ControlFrames.recording = true
    }

    private func open<V: View>(_ content: V, size: NSSize) async throws -> (ToolWindowController, NSWindow) {
        let wc = ToolWindowController(title: "test", autosave: "", size: size, minSize: size, content: content)
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
    /// The control's centre in window coordinates (AppKit, bottom-left origin).
    private var scope: UUID?
    private func centre(_ id: String, _ w: NSWindow) throws -> NSPoint {
        let f = try #require(scope.flatMap { ControlFrames.frames[$0]?[id] }, "\(id) was not laid out")
        #expect(f.width >= 14 && f.height >= 14, "\(id) is too small to tap: \(f.size)")
        let height = try #require(w.contentView).bounds.height
        return NSPoint(x: f.midX, y: height - f.midY)
    }
    /// What AppKit's hit-test picks at the control's centre.
    private func hit(_ id: String, _ w: NSWindow) throws -> NSView? {
        let p = try centre(id, w)
        let frame = try #require(w.contentView?.superview)
        return frame.hitTest(frame.convert(p, from: nil))
    }
    /// The hit must reach the SwiftUI host itself, or an AppKit control the host made for this control, never a
    /// scroll view or another surface lying over it.
    /// `scrolled`: the control lives inside a SwiftUI scroll view, whose own platform views then rightly take the click.
    private func expectReachable(_ id: String, _ w: NSWindow, control: AnyClass? = nil, scrolled: Bool = false) throws {
        let view = try hit(id, w)
        let host = try #require(w.contentView)
        let reached =
            view === host || control.map { c in view.map { $0.isKind(of: c) || $0.superview?.isKind(of: c) == true } ?? false } ?? false
            || (scrolled && view?.isDescendant(of: host) == true)
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

    // MARK: Chat

    /// The model's name starts right after the real traffic lights, as the browser's collapsed toolbar does.
    @Test func theTitleStartsAfterTheTrafficLights() async throws {
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { nil })
        let (wc, w) = try await open(ChatView(session: session), size: ChatView.size)
        defer { wc.close() }
        let zoom = try #require(w.standardWindowButton(.zoomButton))
        let end = try #require(zoom.superview).convert(zoom.frame, to: nil).maxX
        print("traffic lights end at \(end); band leading \(ToolLayout.bandLeading)")
        #expect(ToolLayout.bandLeading >= end + 10 && ToolLayout.bandLeading <= end + 18)
    }

    @Test func chatPresetsSendStopAndNewChatTakeAPlainTap() async throws {
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { nil })
        // The benchmark prompts sit behind Details.
        let details = ToolDetails.shown
        ToolDetails.shown = true
        defer { ToolDetails.shown = details }
        let (wc, w) = try await open(ChatView(session: session), size: ChatView.size)
        defer { wc.close() }
        try await tap("preset-1K", w)
        #expect(session.preset == PresetPrompt(tokens: 1024))
        try await tap("preset-8K", w)
        #expect(session.preset == PresetPrompt(tokens: 8192))
        try await tap("remove-preset", w)
        #expect(session.preset == nil)
        // Above the loaded 32K context (with room for a reply): drawn, reachable, does nothing.
        try await tap("preset-32K", w)
        #expect(session.preset == nil)
        #expect(session.unavailable(PresetPrompt(tokens: 32_768)) == "Longer than Qwen3.8 27B\u{2019}s 32K context")
        try expectReachable("message", w, control: NSTextView.self)
        // Send without the API: the failure is a line in the conversation, the window keeps working.
        session.draft = "Hello"
        try await settle(w)
        try await tap("send", w)
        try await waitUntil { !session.generating }
        #expect(session.messages.map(\.role) == [.user, .assistant])
        #expect(session.messages.last?.error == ChatSession.noAPI)
        try await tap("new-chat", w)
        #expect(session.messages.isEmpty)
        // Stop while a reply streams.
        session.preview(messages: V3RenderDelegate.chatMessages(), run: V3RenderDelegate.finishedRun(), reference: .none, generating: true)
        try await settle(w)
        try await tap("stop", w)
        #expect(!session.generating)
    }

    /// The whole path a reply takes: the app's own API (hosted as in the app, over the stub service), streamed deltas,
    /// usage, the requester header; the figures fill from what the stream returned.
    @Test func aChatReplyStreamsThroughTheAppsOwnAPI() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-chat-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        let api = APIHost(service: FixtureService(rows: Fixtures.rows(.loaded)), support: support, port: 0)
        await api.start()
        defer { api.stop() }
        let port = try #require(api.port)
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { ArsenalClient(port: port) })
        session.draft = "Say hello"
        session.send()
        try await waitUntil(seconds: 10) { !session.generating }
        #expect(session.messages.last?.text == "Hello from Arsenal.")
        #expect(session.messages.last?.error == nil)
        let run = try #require(session.run)
        #expect(run.done)
        #expect(run.promptTokens == 1 && run.completionTokens == 4)
        let tokens = session.figures().first { $0.title == "Tokens" }
        #expect(tokens?.value == "1 in \u{2192} 4 out")
    }

    /// The app's own API over the stub service with C3, a qwen chat window (never shown) and its session.
    private func liveChat(_ configure: (FixtureService) async -> Void = { _ in }) async throws -> (FixtureService, APIHost, ToolCenter, ChatSession, ToolWindowController, URL) {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-live-\(UUID().uuidString)")
        let service = FixtureService(rows: Fixtures.rows(.loaded))
        await configure(service)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        let port = try #require(api.port)
        let center = ToolCenter(client: { ArsenalClient(port: port) }, playback: FixturePlayback())
        center.present = { _ in }
        let row = try #require(await service.rows().first { $0.id == "qwen3.8-27b" })
        center.open(.chat, row)
        let open = try #require(center.chats[row.id])
        return (service, api, center, open.session, open.window, support)
    }

    /// A tool window's activity stream never reads the machine; closing the window stops its reply, that stream and its
    /// settings read;
    /// the conversation stays for the next opening.
    @Test func closingAToolWindowStopsItsWork() async throws {
        let (service, api, _, session, window, support) = try await liveChat { await $0.setDelay("chat", .seconds(30)) }
        defer { api.stop(); try? FileManager.default.removeItem(at: support) }
        session.draft = "Hello"
        session.send()
        try await waitUntil(seconds: 10) {
            let subscribers = await service.activitySubscribers(), runs = await service.runLog()
            return subscribers >= 1 && runs.contains("qwen3.8-27b chat")
        }
        // Code review #5: the window follows its own request; it never starts the machine readings Activity uses.
        try await Task.sleep(for: .milliseconds(400))
        #expect(await service.activitySamples() == 0, "a tool window read the machine")
        window.close()
        try await waitUntil(seconds: 10) {
            let subscribers = await service.activitySubscribers()
            return subscribers == 0 && !session.hasRunningWork
        }
        let samples = await service.activitySamples()
        try await Task.sleep(for: .milliseconds(800))
        #expect(await service.activitySamples() == samples, "the sampler kept running")
        #expect(session.messages.map(\.role) == [.user, .assistant] && !session.generating)
    }

    /// A failure inside the stream (after tokens arrived) ends as the reply's error, not as a finished reply.
    @Test func aFailureMidStreamIsAnError() async throws {
        let (service, api, _, session, _, support) = try await liveChat {
            await $0.setChatFailure(WorkerFailure(code: "worker_exited", message: "The chat worker stopped."))
        }
        defer { api.stop(); try? FileManager.default.removeItem(at: support) }
        session.draft = "Hello"
        session.send()
        try await waitUntil(seconds: 10) { !session.generating }
        #expect(session.messages.last?.error?.contains("stopped") == true)
        #expect(session.run?.done != true || session.messages.last?.error != nil)
        _ = service
    }

    /// Stop or New Chat while the model counts a preset's tokens: nothing more happens for that request (no chat run,
    /// no activity stream), and the next preset still counts first.
    @Test func stopDuringPresetCalibrationEndsTheRequest() async throws {
        let (service, api, _, session, _, support) = try await liveChat { await $0.setDelay("count_tokens", .seconds(30)) }
        defer { api.stop(); try? FileManager.default.removeItem(at: support) }
        session.insert(PresetPrompt(tokens: 1024))
        session.send()
        try await waitUntil(seconds: 10) { await service.runLog().contains("qwen3.8-27b count_tokens") }
        session.stop()
        try await Task.sleep(for: .milliseconds(600))
        let afterStop = await service.runLog(), subscribers = await service.activitySubscribers()
        #expect(!afterStop.contains("qwen3.8-27b chat") && !session.hasRunningWork && subscribers == 0)
        session.insert(PresetPrompt(tokens: 1024))
        session.send()
        try await waitUntil(seconds: 10) { await service.runLog().filter { $0 == "qwen3.8-27b count_tokens" }.count == 2 }
        session.newChat()
        try await Task.sleep(for: .milliseconds(600))
        let afterNew = await service.runLog()
        #expect(session.messages.isEmpty && !afterNew.contains("qwen3.8-27b chat") && !session.hasRunningWork)
    }

    @Test func chatFiguresComeOnlyFromTheServiceAndCompareLikeWithLike() throws {
        // A short typed question against the chat suite's reference (one-line questions on this chip).
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { nil })
        session.preview(messages: [], run: V3RenderDelegate.shortRun(), reference: .found(V3RenderDelegate.chatReference))
        let figures = Dictionary(uniqueKeysWithValues: session.figures().map { ($0.title, $0) })
        #expect(session.figures().map(\.title) == ["First token", "Prompt", "Decode", "Peak memory", "Tokens", "Longest pause"])
        #expect(figures["First token"]?.value == "0.36 s" && figures["First token"]?.reference == "0.31 s")
        #expect(figures["Prompt"]?.value == "395 tok/s" && figures["Decode"]?.reference == "31.5 tok/s")
        #expect(figures["Peak memory"]?.value == "14.1 GB" && figures["Tokens"]?.value == "23 in \u{2192} 61 out")
        #expect(figures["Longest pause"]?.value == "44 ms")
        #expect(session.referenceLine == .compared(V3RenderDelegate.chatReference))
        #expect(V3RenderDelegate.chatReference.baseline == "M5 Max \u{00b7} short prompt \u{00b7} 2 Oct")
        // An 8K preset: nothing measured that size, so no comparison.
        session.preview(messages: [], run: V3RenderDelegate.finishedRun(), reference: .found(V3RenderDelegate.chatReference))
        #expect(session.referenceLine == .unavailable("No reference for this prompt size"))
        #expect(session.figures().allSatisfy { $0.reference == nil })
        // Before any run every value is empty; nothing is ever "?".
        let fresh = ChatSession(model: V3RenderDelegate.qwen, client: { nil })
        #expect(fresh.figures().allSatisfy { $0.value == nil })
        #expect(FigureText.tokens(in: nil, out: 37) == "\u{2014} in \u{2192} 37 out" && FigureText.tokens(in: nil, out: nil) == nil)
    }

    /// C3's per-request timing while a reply runs (shapes as the aggregator sends them): First token and Prompt as soon
    /// as the first token is out, Peak RAM and Longest pause live; a prefill rate is never shown as Decode.
    @Test func liveFiguresComeFromTheRequestsTimingByPhase() throws {
        let prefill = ActivityFeed(json: [
            "models": [
                [
                    "id": "q", "kind": "chat", "state": "prefill",
                    "requests": [["requester": "Arsenal", "op": "chat", "phase": "prefill", "prompt_tokens": 8214, "tps": 840.0, "timing": ["queue_s": 0.001]]]
                ]
            ]
        ])
        #expect(prefill.models[0].requests[0].phaseRate == nil && ActivityText.detail(prefill.models[0].requests[0], .chat) == "Reading 8,214 tokens")
        let decode = ActivityFeed(json: [
            "models": [
                [
                    "id": "q", "kind": "chat", "state": "decode",
                    "requests": [
                        [
                            "requester": "Arsenal", "op": "chat", "phase": "decode", "prompt_tokens": 8214, "generated_tokens": 40, "tps": 840.0,
                            "timing": ["ttft_s": 1.41, "prefill_tps": 857.0, "peak_mb": 16_498.0, "max_itl_ms": 36.5]
                        ]
                    ]
                ]
            ]
        ])
        let r = decode.models[0].requests[0]
        // Decode has no rate of its own yet: the line shows tokens, never the prompt rate the request's `tps` carries.
        #expect(r.phaseRate == nil && ActivityText.detail(r, .chat) == "40 tok")
        #expect(ActivityText.chatSpeed(decode.models[0]) == nil)
        var run = ChatRun()
        run.livePromptTokens = r.promptTokens; run.liveGeneratedTokens = r.generatedTokens; run.liveTiming = r.timing
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { nil })
        session.preview(messages: [], run: run, reference: .none, generating: true)
        let f = Dictionary(uniqueKeysWithValues: session.figures().map { ($0.title, $0.value) })
        #expect(f["First token"] == "1.41 s" && f["Prompt"] == "857 tok/s" && f["Peak memory"] == "15.4 GB" && f["Longest pause"] == "37 ms")
        #expect(f["Decode"] == .some(nil) && f["Tokens"] == "8,214 in \u{2192} 40 out")
        run.liveTiming["decode_tps"] = 29.3
        run.finish(usage: ["prompt_tokens": 8214, "completion_tokens": 56], timing: ["decode_tps": 29.6, "ttft_s": 1.41])
        #expect(run.figure("decode_tps") == 29.6 && run.figure("peak_mb") == 16_498)
    }

    /// Figures arrive from workers and the API: a finite but absurd value reads as unknown and never traps.
    @Test func extremeValuesNeverTrap() {
        #expect(Double(1e20).roundedInt == nil && Double.nan.roundedInt == nil && Double(41.6).roundedInt == 42)
        #expect(FigureText.rate(1e20) == nil && FigureText.milliseconds(1e20) == nil && FigureText.seconds(1e20) == nil && FigureText.gigabytes(mb: 1e20) == nil)
        #expect(FigureText.rate(1_234.4) == "1,234 tok/s" && FigureText.milliseconds(-1) == nil)
        var m = ActivityFeed.Model(id: "q", kind: "chat", state: "decode", requests: [], queued: 0, batch: 1e20, memoryMB: nil)
        m.requests = [.init(id: "r", requester: "A", op: "chat", phase: "decode", step: nil, total: nil, startedAt: nil, tps: 1e20, timing: ["decode_tps": 1e20])]
        #expect(ActivityText.detail(m.requests[0], .chat) == "Writing")
        #expect(PresetPrompt(tokens: 8192).words(ratio: 1e-300) == PresetPrompt(tokens: 8192).words())
    }

    @Test func presetPromptsAreFreshAtTheStartAndSizedFromTheMeasuredRatio() {
        let a = PresetPrompt.text(words: 400, run: "aaaa"), b = PresetPrompt.text(words: 400, run: "bbbb")
        #expect(a != b && a.dropFirst(12) == b.dropFirst(12))
        let words = a.split { $0 == " " || $0 == "\n" }.count
        #expect(words > 400 && words < 440)
        #expect(a.hasSuffix(PresetPrompt.question))
        #expect(PresetPrompt.all.map(\.title) == ["1K", "2K", "4K", "8K", "16K", "32K", "64K"])
        #expect(PresetPrompt(tokens: 8192).words(ratio: 1.6) == 5120)
        #expect(PresetPrompt(tokens: 8192).unavailable(contextLimit: 8192, model: "Q") != nil)
        #expect(PresetPrompt(tokens: 4096).unavailable(contextLimit: 8192, model: "Q") == nil)
        #expect(PresetPrompt(tokens: 65_536).unavailable(contextLimit: nil, model: "Q") == nil)
    }

    @Test func presetsCountEveryCompletePromptWithinTwoPercent() async throws {
        for preset in PresetPrompt.all {
            var calls = 0
            let sized = try await preset.sized(run: "fixture", typed: "Include this final question.", contextLimit: 100_000) { text in
                calls += 1
                // Template overhead and a tokenizer whose ratio differs from the seed estimate.
                return 49 + text.split(whereSeparator: { $0.isWhitespace }).count * 2
            }
            #expect(calls > 1)
            #expect(abs(sized.tokens - preset.tokens) <= preset.tokens / 50)
            #expect(sized.text.hasSuffix("Include this final question."))
            #expect(sized.tokens == 49 + sized.text.split(whereSeparator: { $0.isWhitespace }).count * 2)
        }
        #expect(PresetPrompt.replyTokens == ChatSession.maxReplyTokens)
        let preset = PresetPrompt(tokens: 1024)
        #expect(preset.unavailable(contextLimit: 1280, model: "Q") != nil)
        #expect(preset.unavailable(contextLimit: 2048, model: "Q") == nil)
        await #expect(throws: ToolError.self) {
            try await preset.sized(run: "fixture", typed: "", contextLimit: 2048) { _ in throw ToolError("Count failed") }
        }
        await #expect(throws: ToolError.self) {
            try await preset.sized(run: "fixture", typed: "too much", contextLimit: 2048) { _ in 3000 }
        }
    }

    @Test func consecutivePresetsSendOnlyTheirSizedPromptAndShowItsCount() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-preset-http-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fixture = Process()
        fixture.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/python3")
        fixture.arguments = [root.appendingPathComponent("Tests/Fixtures/Workers/chat_preset_http.py").path, support.path]
        fixture.standardOutput = FileHandle.nullDevice; fixture.standardError = FileHandle.nullDevice
        try fixture.run()
        defer { fixture.terminate(); fixture.waitUntilExit(); try? FileManager.default.removeItem(at: support) }
        let portFile = support.appendingPathComponent("port")
        try await waitUntil { FileManager.default.fileExists(atPath: portFile.path) }
        let port = try #require(Int(String(contentsOf: portFile, encoding: .utf8)))
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { ArsenalClient(port: port) })
        session.draft = "First conversation"; session.send()
        try await waitUntil { !session.generating }
        for tokens in [1024, 2048, 1024] {
            session.draft = "Include my question"; session.insert(PresetPrompt(tokens: tokens)); session.send()
            try await waitUntil { !session.generating }
            #expect(session.messages.count == 2 && session.messages.last?.error == nil)
            let actual = try #require(session.run?.promptTokens)
            #expect(abs(actual - tokens) <= tokens / 50)
            #expect(session.figures().first { $0.title == "Tokens" }?.value == "\(actual.grouped) in \u{2192} 2 out")
        }
        let requests = try String(contentsOf: support.appendingPathComponent("requests.jsonl"), encoding: .utf8).split(separator: "\n").map {
            try #require(eventObject(Data($0.utf8)))
        }
        let chats = requests.filter { $0["path"] as? String == "/v1/chat/completions" }
        #expect(chats.count == 4)
        #expect(chats.allSatisfy { ($0["body"] as? [String: Any])?["max_tokens"] == nil }, "Typed and preset chats both leave the limit to model settings")
        for request in chats.dropFirst() {
            let body = try #require(request["body"] as? [String: Any]), messages = try #require(body["messages"] as? [[String: Any]])
            #expect(messages.count == 1 && body["max_tokens"] == nil)
            let content = try #require(messages.first?["content"] as? String)
            #expect(content.hasSuffix("Include my question") && !content.contains("First conversation") && !content.contains("Fixture reply"))
        }
    }

    @Test func selectingAPresetStartsAFreshConversation() async throws {
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { nil })
        session.preview(messages: V3RenderDelegate.chatMessages(), run: V3RenderDelegate.finishedRun(), reference: .none)
        session.draft = "My question"; session.insert(PresetPrompt(tokens: 1024)); session.send()
        #expect(session.messages.count == 2 && session.messages[0].text == "My question")
        #expect(session.messages[0].preset?.tokens == 1024 && session.run == nil)
        try await waitUntil { !session.generating }
    }

    @Test func imageWorkerFootprintIsNeverComparedWithMLXPeak() {
        let session = ImageSession(model: V3RenderDelegate.flux, client: { nil })
        let reference = ToolReference(match: .sameChip, chip: "M5 Max", date: "2026-10-06", metrics: ["peak_mb": 12_500], suite: "arsenal-images-suite-v1")
        var run = ToolRun(model: V3RenderDelegate.flux.id, operation: "generate", input: ["width": 512, "height": 512, "steps": 4]); run.memoryMB = 13_240
        session.preview(operations: [ImageFixture.generate], run: run, reference: .found(reference))
        let footprint = session.figures().last
        #expect(footprint?.title == "Peak memory" && footprint?.value == "12.3 GB" && footprint?.reference == nil)
        run.peakMB = 12_000
        session.preview(operations: [ImageFixture.generate], run: run, reference: .found(reference))
        #expect(session.figures().last?.title == "Peak memory" && session.figures().last?.value == "11.2 GB")
        #expect(session.figures().last?.reference != nil)
    }

    // MARK: Image

    private static func pictures() throws -> [URL] {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-test-images", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return try (0..<2).map { try V3RenderDelegate.picture(seed: $0, in: folder) }
    }

    @Test func imageControlsTakeAPlainTapAndFailuresStayInPlace() async throws {
        let pictures = try Self.pictures()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-test-out-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let session = ImageSession(model: V3RenderDelegate.flux, client: { nil }, outputDirectory: folder)
        let results = pictures.map { ImageResult(url: $0, prompt: "p", width: 1024, height: 1024, steps: 4, seed: 7, seconds: 1) }
        session.preview(
            operations: [ImageFixture.withReferences()], values: ["prompt": .text("A lighthouse"), "reference_images": .files(pictures)], results: results)
        // The seed is folded under Advanced.
        session.advancedOpen = true
        let (wc, w) = try await open(ImageView(session: session), size: ImageView.size)
        defer { wc.close() }
        // The options are the folder's: prompt, size, steps, how many, seed, and the fixture's reference images.
        #expect(session.form?.fields.map(\.key) == ["prompt", "size", "steps", "n", "seed", "reference_images"])
        try await tap("seed-back", w, scrolled: true)
        #expect(session.value("seed").text == "7")
        try await tap("result-\(pictures[0].lastPathComponent)", w, scrolled: true)
        #expect(session.current?.url == pictures[0])
        try await tap("remove-reference_images-\(pictures[1].lastPathComponent)", w, scrolled: true)
        #expect(session.value("reference_images").files == [pictures[0]])
        // A menu, a stepper, a field, an open panel, Copy and Finder: reachable, not opened here.
        try expectReachable("field-size", w, control: NSPopUpButton.self, scrolled: true)
        try expectReachable("field-steps", w, control: NSStepper.self, scrolled: true)
        try expectReachable("field-seed", w, control: NSTextField.self, scrolled: true)
        try expectReachable("well-reference_images", w, scrolled: true)
        for id in ["model", "reveal", "copy"] { try expectReachable(id, w) }
        #expect(scope.flatMap { ControlFrames.frames[$0]?["reveal-result"] } == nil, "one Show in Finder per window")
        try await tap("generate", w)
        try await waitUntil { !session.running }
        #expect(session.problem == .init(message: ChatSession.noAPI, action: .retry))
        #expect(
            OperationProblem.wording(ToolError("Arsenal's API did not answer: Could not connect to the server."), model: "FLUX.2 Klein 4B")
                == "Couldn\u{2019}t reach FLUX.2 Klein 4B \u{2014} try again.")
    }

    @Test func anImageIsMadeThroughTheAPIAndSavedToItsFolder() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-image-\(UUID().uuidString)")
        let folder = support.appendingPathComponent("Pictures")
        defer { try? FileManager.default.removeItem(at: support) }
        let rows = Fixtures.rows(.loaded)
        let flux = try #require(rows.first { $0.kind == "images" })
        let service = FixtureService(rows: rows, operations: [flux.id: [ImageFixture.operation]])
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        let port = try #require(api.port)
        let session = ImageSession(model: ToolModel(flux), client: { ArsenalClient(port: port) }, outputDirectory: folder)
        session.prepare()
        try await waitUntil { !session.loading }
        #expect(session.operations.map(\.name) == ["generate"] && session.problem == nil)
        session.set("prompt", .text("A lighthouse"))
        session.set("seed", .text("42"))
        session.set("n", .text("2"))
        session.start()
        try await waitUntil(seconds: 10) { !session.running }
        #expect(session.problem == nil)
        // Counts first: two images from one request, the first shown.
        #expect(session.results.count == 2 && session.current?.id == session.results.first?.id)
        let saved = try #require(session.results.last)
        #expect(saved.seed == 42)
        #expect(saved.url.deletingLastPathComponent().resolvingSymlinksInPath() == folder.resolvingSymlinksInPath())
        #expect(NSImage(data: try Data(contentsOf: saved.url)) != nil)
        // A pinned seed makes the same images again: Generate steps back; a random seed makes new ones.
        #expect(session.isFresh)
        session.set("seed", .text(""))
        #expect(!session.isFresh)
        let sent = try #require(await service.runInputs.last)
        #expect(sent["prompt"] == "A lighthouse" && sent["seed"] == 42 && sent["n"] == 2 && sent["width"] == 1024 && sent["height"] == 1024 && sent["resolution"] == nil)
        // A model the service does not have: the service's own reason, in place; nothing saved.
        let missing = ImageSession(
            model: ToolModel(id: "no-such-model", name: "Nothing", kind: "images", dtype: "bf16", mode: "exact"), client: { ArsenalClient(port: port) },
            outputDirectory: folder)
        missing.preview(operations: [ImageFixture.generate], values: ["prompt": .text("A lighthouse")])
        missing.start()
        try await waitUntil(seconds: 10) { !missing.running }
        #expect(missing.problem != nil && missing.results.isEmpty)
    }

    /// Review r1 (A7): earlier images stay browsable while a new one is made.
    @Test func theStripIsBrowsableDuringARun() async throws {
        let pictures = try Self.pictures()
        let session = ImageSession(model: V3RenderDelegate.flux, client: { nil })
        let results = pictures.map { ImageResult(url: $0, prompt: "p", width: 1024, height: 1024, steps: 4, seed: 7, seconds: 1) }
        var run = ToolRun(model: V3RenderDelegate.flux.id, operation: "generate", input: ["steps": 4]); run.step = 1; run.total = 4
        session.preview(operations: [ImageFixture.generate], values: ["prompt": .text("p")], results: results, run: run)
        let (wc, w) = try await open(ImageView(session: session), size: ImageView.size)
        defer { wc.close() }
        #expect(session.running && session.current?.url == pictures[1])
        try await tap("result-\(pictures[0].lastPathComponent)", w, scrolled: true)
        #expect(session.current?.url == pictures[0] && session.running)
    }

    /// Stop while the image is made: the last step seen stays, no time is reported, nothing claims all steps.
    @Test func stoppingAnImageKeepsTheLastStep() {
        let session = ImageSession(model: V3RenderDelegate.flux, client: { nil })
        var run = ToolRun(model: V3RenderDelegate.flux.id, operation: "generate", input: ["width": 1024, "height": 1024, "steps": 4]); run.step = 2; run.total = 4
        session.preview(operations: [ImageFixture.generate], run: run)
        #expect(session.running)
        session.stop()
        let f = Dictionary(uniqueKeysWithValues: session.figures().map { ($0.title, $0.value) })
        #expect(f["Steps"] == "2 of 4" && f["Time"] == .some(nil) && !session.running)
    }

    @Test func imageOptionsComeFromTheOperationSchema() {
        let flux: JSONObject = [
            "properties": [
                "prompt": ["type": "string"], "width": ["type": "integer", "minimum": 64, "maximum": 2048], "height": ["type": "integer", "minimum": 64, "maximum": 2048],
                "resolution": ["enum": ["512x512", "1024x1024"]], "steps": ["type": "integer", "minimum": 1, "maximum": 100], "seed": ["type": "integer"]
            ]
        ]
        let form = ToolForm(schema: flux, operation: "generate", output: "image")
        // Width and height make one Size (the resolution list is not shown, never sent).
        #expect(form.fields.map(\.key) == ["prompt", "size", "steps", "seed"])
        #expect(form.fields[0].control == .main(placeholder: "Describe the image"))
        guard case .size(let sizes) = form.fields[1].control else { Issue.record("no size field"); return }
        #expect(sizes.free && sizes.sizes == ImageSize.free && form.unsupported == nil)
        #expect(form.fields[2].control == .generic(.integer(min: 1, max: 100, stepper: true)))
        // Reversed bounds (a local folder can declare anything): no size, a reason, Generate off; no trap.
        let reversedSchema: JSONObject = ["properties": ["prompt": ["type": "string"], "width": ["minimum": 2048, "maximum": 512], "height": ["minimum": 64, "maximum": 2048]]]
        #expect(ToolForm(schema: reversedSchema).unsupported != nil)
        let session = ImageSession(model: V3RenderDelegate.flux, client: { nil })
        session.preview(
            operations: [OperationDescriptor(name: "generate", title: "Generate", inputSchema: reversedSchema, output: "image", stream: false)],
            values: ["prompt": .text("A lighthouse")])
        #expect(!session.canRun)
        // A resolution list alone is the Size; reference images take their limit.
        let refs = ToolForm(schema: ["properties": ["resolution": ["enum": ["512x512", "1024x1024"]], "reference_images": ["type": "array", "maxItems": 2]]])
        guard case .size(let listed) = refs.fields.first(where: { $0.key == "resolution" })?.control else { Issue.record("no resolution size"); return }
        #expect(!listed.free && listed.sizes.map(\.title) == ["512 \u{00d7} 512", "1024 \u{00d7} 1024"])
        #expect(refs.fields.first { $0.key == "reference_images" }?.control == .images(max: 2))
        let input = try? refs.input(["resolution": .text("1024x1024")], seed: nil, stage: { $0.path })
        #expect(input == ["resolution": "1024x1024"])
    }

    /// C5 as the API encodes core's ModelReference (suite object, null metrics): a short typed question is compared,
    /// an 8K preset and a 300-token prompt are not.
    @Test func referencesFromARealResultsShapeCompareOnlyTheSameWork() throws {
        let json = #"""
            {"match":"same_chip","chip":"m5-max","date":"2026-10-02","dtype":"int4","mode":"exact","runner":"native",
             "suite":{"id":"arsenal-chat-suite-v1","version":1,"sha256":"5f0c"},
             "metrics":{"ttft_s":0.31,"prefill_tps":410.2,"decode_tps":31.5,"peak_mb":14750,"accuracy":null}}
            """#
        let object = try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let reference = try #require(ToolReference(object))
        #expect(reference.workload == .shortPrompt && reference.baseline == "M5 Max \u{00b7} short prompt \u{00b7} 2 Oct")
        let session = ChatSession(model: V3RenderDelegate.qwen, client: { nil })
        var short = ChatRun(); short.finish(usage: ["prompt_tokens": 23, "completion_tokens": 40], timing: [:])
        session.preview(messages: [], run: short, reference: .found(reference))
        #expect(session.referenceLine == .compared(reference))
        var long = ChatRun(); long.finish(usage: ["prompt_tokens": 300, "completion_tokens": 40], timing: [:])
        session.preview(messages: [], run: long, reference: .found(reference))
        #expect(session.referenceLine == .unavailable("No reference for this prompt size"))
        session.preview(messages: [], run: V3RenderDelegate.finishedRun(), reference: .found(reference))
        #expect(session.referenceLine == .unavailable("No reference for this prompt size"))
    }

    @Test func apiErrorsAndReferencesReadPlainly() {
        #expect(ToolError.from(status: 507, body: Data(#"{"error":{"message":"Not enough memory"}}"#.utf8)).message == "Not enough memory")
        #expect(ToolError.from(status: 404, body: Data(#"{"error":"Unknown model x."}"#.utf8)).message == "Unknown model x.")
        let r = ToolReference(["match": "same_family", "chip": "m4-max", "date": "2026-10-01", "metrics": ["decode_tps": 28.0, "ttft_s": NSNull()]])
        #expect(r?.chip == "M4 Max" && r?.match == .sameFamily && r?.metrics == ["decode_tps": 28])
        #expect(r?.help.contains("closest chip in this Mac's family") == true)
        #expect(ToolReference(["chip": NSNull()]) == nil)
    }

    // MARK: Helpers

    private func waitUntil(seconds: Double = 5, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            if Date() > deadline { Issue.record("timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
