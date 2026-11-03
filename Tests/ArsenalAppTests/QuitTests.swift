// SPDX-License-Identifier: MIT
import AppKit
import Darwin
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// A service with one real supervised stub worker: its shutdown is the supervisor's (core's `RealModelsService.shutdown`
/// ends in the same call), so the test proves the app's quit path reaches a real process group.
actor SupervisedService: ModelsService {
    let supervisor = Supervisor()
    private(set) var shutdowns = 0
    func rows() -> [ModelRow] { [] }
    func kinds() -> [KindSpec] { [] }
    func events() -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) {}
    func cancelGet(id: String) {}
    func load(id: String, precision: String?, fast: Bool?) {}
    func unload(id: String) async { await supervisor.unload(id: id) }
    func setKeepHot(id: String, on: Bool) {}
    func delete(id: String) {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        try await supervisor.request(id: id, op: op, params: params, timeout: 60, sink: sink)
    }
    func runRaw(id: String, op: String, params: JSONObject) async throws -> Data { Data() }
    func menuActions() -> [LoadedAction] { [] }
    func memory() -> MemorySnapshot { MemorySnapshot() }
    func catalogStatus() -> CatalogStatus { CatalogStatus() }
    func shutdown() async { shutdowns += 1; await supervisor.shutdown() }
}

/// A service whose shutdown never finishes.
actor StuckService: ModelsService {
    func rows() -> [ModelRow] { [] }
    func kinds() -> [KindSpec] { [] }
    func events() -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) {}
    func cancelGet(id: String) {}
    func load(id: String, precision: String?, fast: Bool?) {}
    func unload(id: String) {}
    func setKeepHot(id: String, on: Bool) {}
    func delete(id: String) {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject { [:] }
    func runRaw(id: String, op: String, params: JSONObject) async throws -> Data { Data() }
    func menuActions() -> [LoadedAction] { [] }
    func memory() -> MemorySnapshot { MemorySnapshot() }
    func catalogStatus() -> CatalogStatus { CatalogStatus() }
    func shutdown() async { try? await Task.sleep(for: .seconds(3600)) }
}

@MainActor @Suite(.menuState, .serialized) final class QuitTests {
    private let support = URL(fileURLWithPath: "/tmp/arsenal-appfix/quit-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    private func stubLaunch() -> WorkerLaunch {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return WorkerLaunch(
            // scripts/test.sh builds into its own scratch path and names the stub worker it built.
            executable: ProcessInfo.processInfo.environment["ARSENAL_TEST_STUB_WORKER"].map { URL(fileURLWithPath: $0) }
                ?? root.appendingPathComponent(".build/debug/arsenal-stub-worker"), arguments: ["--kind", "chat"], kind: "chat",
            runner: "native", environment: ["ARSENAL_SUPPORT_DIR": support.path, "ARSENAL_CACHE_DIR": support.path, "PATH": "/usr/bin:/bin"],
            log: support.appendingPathComponent("worker.log"))
    }

    /// Quit while a runner is blocked in work that ignores stdin EOF and SIGTERM: AppKit is told to wait
    /// (`terminateLater`), the app stops admitting work, the service's shutdown escalates to the process group, and only
    /// then does the app answer that termination may proceed. No process survives.
    @Test func quitWaitsForAWorkerThatIgnoresEOFToBeGone() async throws {
        let service = SupervisedService()
        let worker = try await service.supervisor.worker(id: "busy", launch: stubLaunch())
        let pid = await worker.pid
        let work = Task { try await service.run(id: "busy", op: "test_ignore_term", params: [:], sink: nil) }
        try await Task.sleep(for: .milliseconds(200)) // the worker is now in its uninterruptible sleep, SIGTERM ignored
        #expect(kill(pid, 0) == 0)

        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        var replies: [Bool] = []
        app.replyToTerminate = { replies.append($0) }
        let started = Date()
        #expect(app.applicationShouldTerminate(NSApplication.shared) == .terminateLater)
        #expect(app.applicationShouldTerminate(NSApplication.shared) == .terminateCancel) // a second Quit while waiting
        #expect(!app.table.admitting)
        while replies.isEmpty, Date().timeIntervalSince(started) < 15 { try await Task.sleep(for: .milliseconds(20)) }

        #expect(replies == [true])
        #expect(kill(pid, 0) == -1 && errno == ESRCH)
        #expect(await worker.exited)
        #expect(await service.shutdowns == 1)
        #expect(await service.supervisor.residentIDs().isEmpty)
        #expect(Date().timeIntervalSince(started) < 10)
        do { _ = try await work.value; Issue.record("The blocked request should fail when its worker is stopped") } catch {}
    }

    /// A shutdown that never finishes cannot keep the app from quitting: the bounded wait gives up.
    @Test func aStuckShutdownIsBounded() async {
        let app = AppDelegate(service: StuckService(), paths: ArsenalPaths(support: support))
        let started = Date()
        #expect(await app.shutDown(limit: .milliseconds(200)) == false)
        #expect(Date().timeIntervalSince(started) < 2)
        #expect(app.quitting)
    }

    /// A service without supervised work (the stub catalog) quits at once; menu actions start nothing afterwards.
    @Test func quittingStopsMenuActions() async throws {
        let service = FixtureService(rows: Fixtures.rows(.firstLaunch))
        try await service.get(id: "kokoro-82m", precision: "fp16")
        try await service.load(id: "kokoro-82m", precision: "fp16", fast: true)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        #expect(await app.shutDown(limit: .seconds(1)))
        app.runner.readClipboard = { "Hello" }
        app.runner.run(LoadedAction(id: "speak", modelID: "kokoro-82m", title: "Speak Clipboard", op: "speak", input: "clipboard"))
        #expect(app.runner.running == nil)
        #expect(app.menu.items.first { $0.title == "Speak Clipboard" }?.isEnabled != true)
    }
}

/// Speak Clipboard through the shared playback queue (C7): the action asks for a file, not a stream, and hands it to
/// the queue as Arsenal's speech with its model and voice; muted, the file is kept and the line says so. Nothing sounds
/// here: the queue is replaced by a recorder.
@MainActor @Suite(.menuState, .serialized) final class SpeechPlaybackTests {
    /// Starts an action and returns when it has finished (the runner's own change callback, no polling).
    private func finished(_ runner: ActionRunner, _ start: () -> Void) async {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            var resumed = false
            runner.onChange = { if runner.running == nil, !resumed { resumed = true; done.resume() } }
            start()
        }
        runner.onChange = {}
    }

    @Test func speakClipboardGoesToTheSharedQueue() async throws {
        let service = FixtureService(rows: Fixtures.rows(.firstLaunch))
        try await service.get(id: "kokoro-82m", precision: "fp16")
        try await service.load(id: "kokoro-82m", precision: "fp16", fast: true)
        let jobs = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-v3-speech-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: jobs) }
        let runner = ActionRunner(service: service, jobs: jobs)
        var queued: [(URL, String, String?)] = []
        var muted = false
        runner.play = { file, model, voice in
            queued.append((file, model, voice)); return !muted
        }
        runner.stopOwnSpeech = {}
        runner.readClipboard = { "Hello" }
        let speak = LoadedAction(id: "speak", modelID: "kokoro-82m", title: "Speak Clipboard", op: "speak", input: "clipboard", params: ["voice": "af_heart"])
        #expect(try ActionRunner.params(speak, clipboard: "Hello", outDir: jobs)["stream"] == false)
        await finished(runner) { runner.run(speak) }
        #expect(runner.lastResult == "Spoke 0.2 s of audio")
        let first = try #require(queued.first)
        #expect(first.1 == "kokoro-82m" && first.2 == "af_heart" && FileManager.default.fileExists(atPath: first.0.path))
        muted = true
        await finished(runner) { runner.run(speak) }
        #expect(runner.lastResult == "Muted: 0.2 s of audio not played")
        #expect(queued.count == 2)
    }
}
