// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore
import SwiftUI

/// `ARSENAL_QA="real-v3 <dir>"` (GPU: only under `scripts/mlock gpulock`, alarm ≤ 180 s): the real service on an
/// isolated, cloned support dir with Qwen3.8 int4 and Kokoro fp16 installed. Loads Qwen, opens the chat tool (its window
/// never shown), sends a short prompt through the app's own API, renders the chat window and the Models submenu with
/// the activity panel while the reply streams and when it is done; then loads Kokoro and speaks one sentence with
/// playback muted, renders the voice section; unloads everything and checks no worker is left.
/// Env: ARSENAL_SUPPORT_DIR, ARSENAL_CACHE_DIR, ARSENAL_CATALOG (repo), ARSENAL_HELPERS, ARSENAL_API_PORT=0, ARSENAL_PLAYBACK_MUTED=1.
@MainActor final class RealV3Delegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    /// Only the idle baseline: the same process and service, no tool, panel or model ever used.
    var baselineOnly = false
    var exitCode: Int32 = 1
    private let log = QALog()
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        ToolWindow.keepOffscreenFrames = true
        Task {
            do { try await run(); exitCode = 0 } catch { log.say("real-v3 FAILED: \(TableController.message(error))") }
            log.write(to: directory.appendingPathComponent(baselineOnly ? "v3-real-idle-baseline.txt" : "v3-real.txt"))
            QA.stopApp()
        }
    }

    private var app: AppDelegate!

    private func run() async throws {
        let paths = ArsenalPaths()
        log.say("support \(paths.support.path)")
        let (service, error) = await ServiceFactory.make(paths: paths)
        if let error { throw QAFailure(error) }
        let app = AppDelegate(service: service, paths: paths)
        self.app = app
        app.tools.present = { _ in }
        await app.api.start()
        guard let port = app.api.port else { throw QAFailure(app.api.error ?? "no API port") }
        log.say("API on 127.0.0.1:\(port)")
        defer { app.api.stop() }
        await app.table.refresh()
        if baselineOnly {
            await app.refreshMenuState(); app.rebuildMenu()
            try await renderMenuItems("real-menu-idle")
            app.activityMenu.activity.show()
            try await Task.sleep(for: .seconds(1))
            try await renderModels("real-models-nothing-loaded")
            app.activityMenu.activity.hide()
            try await Task.sleep(for: .seconds(10))
            log.say("idle baseline (no tool used): " + (try idle(seconds: 60)))
            await service.shutdown()
            return
        }

        // Qwen3.8 int4.
        let qwenID = "qwen3.8-27b"
        let started = Date()
        try await service.load(id: qwenID, precision: "int4", fast: nil)
        log.say(String(format: "loaded %@ int4 in %.2f s", qwenID, Date().timeIntervalSince(started)))
        await app.table.refresh(); await app.refreshMenuState(); app.rebuildMenu()
        let qwen = try require(app.table.rows.first { $0.id == qwenID }, "no Qwen row")
        log.say("Qwen row: tools \(qwen.tools) shown \(qwen.toolsShown) resident \(qwen.residentDType ?? "-"); menu: " + titles(app.menu))
        if let chatItem = app.menu.items.firstIndex(where: { $0.title == "Chat — Qwen3.8 27B\u{2026}" }) {
            app.menu.performActionForItem(at: chatItem)
        } else {
            // The signed catalog in use predates C1's `tools` (re-signed by core before the release build): open the
            // tool the way its menu item does.
            log.say("no Chat… item (row declares no tools in this catalog); opening the chat tool directly")
            app.tools.open(.chat, qwen)
        }
        let (window, session) = try require(app.tools.chats[qwenID], "chat tool did not open")
        try await QA.wait(timeout: 20) { session.reference != .loading }
        log.say("context limit \(session.model.contextLimit.map(String.init) ?? "?"); reference \(session.reference)")

        // The activity panel, subscribed as when the Models submenu opens.
        let activity = app.activityMenu.activity
        activity.show()
        defer { activity.hide() }

        // 1. The first request after load (cold), 2. the same kind again (warm, as the reference suite measures after
        // its warm-up), 3. a 1K preset. Panel frames are captured while the first reply streams (twice a second).
        for (index, prompt) in ["In six sentences, explain why the sea looks blue.", "In two sentences, why is the sky blue at noon?", nil].enumerated() {
            if let prompt { session.draft = prompt } else { session.insert(PresetPrompt(tokens: 1024)) }
            session.send()
            var frame = 0, captured = false
            while session.generating, index == 0, frame < 12 {
                try await Task.sleep(for: .milliseconds(500))
                frame += 1
                try await renderModels(String(format: "real-activity-%02d", frame))
                log.say("frame \(frame): " + describe(activity.feed) + " · grid: " + figures(session))
                // The chat window mid-reply: once the first tokens are out (first token and prompt rate are known).
                if !captured, (session.run?.liveGeneratedTokens ?? 0) >= 10 { try await capture(window, "real-chat-live"); captured = true }
            }
            try await QA.wait(timeout: 120) { !session.generating }
            if let error = session.messages.last?.error { throw QAFailure("reply \(index + 1) failed: \(error)") }
            log.say("reply \(index + 1) (\(prompt == nil ? "1K preset" : index == 0 ? "first after load" : "warm")): \(session.messages.last?.text.prefix(160) ?? "")")
            log.say("figures \(index + 1): " + figures(session) + " · timing \(session.run?.timing ?? [:]) · reference line \(String(describing: session.referenceLine))")
            try await capture(window, "real-chat-done-\(index + 1)")
        }
        try await renderModels("real-models-idle")
        app.tools.closeAll()

        try await voice(service)
        try await renderModels("real-models-voice")

        // Everything back: no runner outlives the run.
        for id in [qwenID, "kokoro-82m"] { try? await service.unload(id: id) }
        try QA.noWorkersLeft(after: "unload", log: log)
        // Idle after a tool window was used and closed and the panel hidden: nothing loaded, nothing subscribed. The
        // service's C3 sampler would wake this process four times a second if it still ran.
        activity.hide()
        await app.table.refresh(); await app.refreshMenuState(); app.rebuildMenu()
        try await renderMenuItems("real-menu-idle")
        try await Task.sleep(for: .seconds(10))
        log.say("idle after the chat window closed: " + (try idle(seconds: 60)))
        await service.shutdown()
    }

    /// Kokoro fp16 speaks one sentence through the menu's action path; playback is muted for the whole run.
    private func voice(_ service: any ModelsService) async throws {
        let id = "kokoro-82m"
        try await service.load(id: id, precision: "fp16", fast: nil)
        await app.table.refresh(); await app.refreshMenuState()
        try app.tools.playback.setMuted(true)
        app.rebuildMenu()
        log.say("voices: \(app.tools.voices[id]?.count ?? 0), current \(app.tools.currentVoice[id] ?? "-"); errors \(app.tools.errors)")
        app.runner.run(LoadedAction(id: "qa-speak", modelID: id, title: "Speak", op: "speak", params: ["text": "Arsenal is ready."]))
        try await QA.wait(timeout: 60) { self.app.runner.running == nil }
        log.say("speak: \(app.runner.lastResult ?? "-"); playback muted \(app.tools.playback.muted)")
        app.rebuildMenu()
        log.say("menu: " + app.menu.items.map { $0.isSeparatorItem ? "—" : $0.title + ($0.subtitle.map { " [\($0)]" } ?? "") }.joined(separator: " | "))
        for (title, name) in [("Memory", "memory"), ("Voice", "voice")] {
            guard let sub = app.menu.items.first(where: { $0.title == title || (title == "Voice" && $0.title.hasPrefix("Voice:")) })?.submenu else { continue }
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                MenuMock.render(sub.items, width: 260, to: directory.appendingPathComponent("v3-real-menu-\(name).png")) { done.resume() }
            }
        }
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("v3-real-menu.png")) { done.resume() }
        }
    }

    // MARK: Helpers

    private func require<T>(_ value: T?, _ message: String) throws -> T {
        guard let value else { throw QAFailure(message) }
        return value
    }
    private func titles(_ menu: NSMenu) -> String { menu.items.map { $0.isSeparatorItem ? "—" : $0.title }.joined(separator: " | ") }
    private func figures(_ s: ChatSession) -> String { s.figures().map { "\($0.title) \($0.value ?? "—")" }.joined(separator: ", ") }
    private func describe(_ feed: ActivityFeed) -> String {
        let models = feed.models.map { m in
            "\(m.id) \(m.state) mem \(m.memoryMB.map { String(format: "%.0f MB", $0) } ?? "?") ["
                + m.requests.map {
                    "\($0.requester) \($0.op) \($0.phase) p\($0.promptTokens ?? -1) g\($0.generatedTokens ?? -1) \($0.tps.map { String(format: "%.1f", $0) } ?? "-")"
                }
                .joined(separator: "; ") + "]"
        }
        let m = feed.machine
        func f(_ v: Double?, _ unit: String) -> String { v.map { String(format: "%.0f", $0) + unit } ?? "?" }
        let machine = "cpu \(f(m.cpuBusyPct, "%")) \(f(m.cpuTempC, "°C")) gpu \(f(m.gpuBusyPct, "%")) \(f(m.gpuTempC, "°C")) mem \(f(m.memoryUsedMB, " MB"))"
        return (models + [machine]).joined(separator: " · ")
    }

    private func renderMenuItems(_ name: String) async throws {
        log.say("menu \(name): " + app.menu.items.map { $0.isSeparatorItem ? "—" : $0.title }.joined(separator: " | "))
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("v3-\(name).png")) { done.resume() }
        }
    }

    /// scripts/idle.py on this process (kernel rusage: CPU time and wakeups over the window).
    private func idle(seconds: Int) throws -> String {
        let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("scripts/idle.py")
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [script.path, "--seconds", String(seconds), "--label", "v3-ui-after-tools", String(ProcessInfo.processInfo.processIdentifier)]
        let pipe = Pipe(); process.standardOutput = pipe
        try process.run(); process.waitUntilExit()
        return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The Activity submenu's panel.
    private func renderModels(_ name: String) async throws {
        await QA.renderActivity(app.activityMenu.activity, to: directory.appendingPathComponent("v3-\(name).png"))
        log.say("wrote v3-\(name).png")
    }

    private func capture(_ controller: ToolWindowController, _ name: String) async throws {
        guard let window = controller.window, let root = window.contentView?.superview else { return }
        window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: ChatView.size), display: false)
        window.orderFrontRegardless()
        try await Task.sleep(for: .milliseconds(400))
        root.layoutSubtreeIfNeeded(); window.displayIfNeeded()
        guard
            let out = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: Int(root.bounds.width * 2), pixelsHigh: Int(root.bounds.height * 2), bitsPerSample: 8, samplesPerPixel: 4,
                hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return }
        out.size = root.bounds.size
        root.cacheDisplay(in: root.bounds, to: out)
        try out.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("v3-\(name).png"))
        window.orderOut(nil)
        log.say("wrote v3-\(name).png")
    }
}
