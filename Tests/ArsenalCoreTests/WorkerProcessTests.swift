// SPDX-License-Identifier: MIT
import Foundation
import Testing
import Darwin
@testable import ArsenalCore

func stubLaunch(_ kind: String = "chat") -> WorkerLaunch {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    return WorkerLaunch(
        executable: ProcessInfo.processInfo.environment["ARSENAL_TEST_STUB_WORKER"].map { URL(fileURLWithPath: $0) }
            ?? root.appendingPathComponent(".build/debug/arsenal-stub-worker"), arguments: ["--kind", kind], kind: kind, runner: "native",
        environment: [
            "ARSENAL_SUPPORT_DIR": "/tmp/arsenal-core/support", "ARSENAL_CACHE_DIR": "/tmp/arsenal-core/cache", "PATH": "/usr/bin:/bin", "HF_TOKEN": "must-not-leak",
            "DYLD_TEST": "must-not-leak"
        ], log: URL(fileURLWithPath: "/tmp/arsenal-core/logs/\(UUID().uuidString).log"))
}
actor FrameCollector { var values: [WireFrame] = []; func add(_ frame: WireFrame) { values.append(frame) } }
@Test func workerLifecycleAndMultiplex() async throws {
    let w = WorkerProcess(); try await w.start(stubLaunch("voices")); let pid = await w.pid
    _ = try await w.request(op: "load", params: ["weights_dir": "/tmp/none"])
    #expect(await w.memory.footprintMB == 16)
    let frames = FrameCollector()
    _ = try await w.request(op: "speak", params: ["text": "hello", "stream": true], sink: { await frames.add($0) })
    #expect(await frames.values.count == 2)
    try await withThrowingTaskGroup(of: Void.self) { group in
        for _ in 0..<32 { group.addTask { _ = try await w.request(op: "ping") } }; try await group.waitForAll()
    }
    await w.unload(); #expect(await w.exited); #expect(kill(pid, 0) == -1); #expect(await w.memory.footprintMB == 0)
}
@Test func hostileWorkerIsolation() async throws {
    let good = WorkerProcess(); try await good.start(stubLaunch())
    for op in ["test_crash", "test_hang", "test_ignore_term", "test_garbage", "test_oversize"] {
        let w = WorkerProcess(); try await w.start(stubLaunch()); let pid = await w.pid
        do { _ = try await w.request(op: op, timeout: 0.2); Issue.record("Hostile worker should fail") } catch {}
        await w.unload(grace: 0.1); #expect(await w.exited); #expect(kill(pid, 0) == -1)
        #expect(try await good.request(op: "ping")["ok"] == true)
    }
    await good.unload()
}
@Test func cancellationStopsHang() async throws {
    let w = WorkerProcess(); try await w.start(stubLaunch())
    let task = Task { try await w.request(op: "test_hang", timeout: 5) }
    try await Task.sleep(for: .milliseconds(30)); task.cancel()
    do { _ = try await task.value; Issue.record("Cancelled request succeeded") } catch is CancellationError {} catch { Issue.record("Expected cancellation") }
    await w.unload(grace: 0.1); #expect(await w.exited)
}
extension RuntimeTimingTests {
    @Test static func runnerQuarantine() async throws {
        let s = Supervisor(crashLimit: 2)
        for _ in 0..<2 { let w = try await s.worker(id: "bad", launch: stubLaunch()); do { _ = try await w.request(op: "test_crash", timeout: 10) } catch {}; await w.unload() }
        do { _ = try await s.worker(id: "bad", launch: stubLaunch()); Issue.record("Runner should be quarantined") } catch let e as WorkerFailure {
            #expect(e.code == "quarantined")
        }
        await s.reset(id: "bad"); _ = try await s.worker(id: "bad", launch: stubLaunch()); await s.shutdown()
    }
}

@Test func rawDecisionReplyAndFailure() async throws {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let code =
        "import sys; sys.path.insert(0, \(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers").path))); exec(open(\(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers/raw_worker.py").path))).read())"
    let w = WorkerProcess(); var launch = stubLaunch("decisions"); launch.executable = URL(fileURLWithPath: "/opt/homebrew/bin/python3"); launch.arguments = ["-I", "-c", code]
    try await w.start(launch)
    let result = try await w.request(op: "decide", params: ["body": "original"])
    #expect(result["_raw_body"]?.string == "{\"answers\":{\"é\":1,\"e\\u0301\":2},\"usage\":{\"x\":0.1234567890123456789}}")
    do { _ = try await w.request(op: "decide", params: ["body": "error"]); Issue.record("Expected worker error") } catch let error as WorkerFailure {
        #expect(error.code == "wrong_model"); #expect(error.details["model"] == "laya-multilingual"); #expect(error.details["status"] == 409)
        #expect(error.rawBody == "{\"é\":1,\"e\\u0301\":2}"); #expect(error.rawFrame != nil)
    }
    await w.unload()
}

extension RuntimeTimingTests {
    @Test static func slowReaderHasBoundedDeadline() async throws {
        let w = WorkerProcess(); try await w.start(stubLaunch())
        let hang = Task { try await w.request(op: "test_hang", timeout: 5) }
        try await Task.sleep(for: .milliseconds(20))
        let started = Date()
        do {
            _ = try await w.request(op: "ping", params: ["payload": .string(String(repeating: "x", count: 2 * 1024 * 1024))], timeout: 0.1)
            Issue.record("Slow reader accepted request")
        } catch {}
        _ = try? await hang.value
        await w.unload(grace: 0.1)
        #expect(Date().timeIntervalSince(started) < 15)
        #expect(await w.exited)
    }
}

actor RawFixtureService: ModelsService {
    let worker: WorkerProcess
    init(worker: WorkerProcess) { self.worker = worker }
    func rows() -> [ModelRow] { [] }
    func kinds() -> [KindSpec] { [] }
    func events() -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) {}
    func cancelGet(id: String) {}
    func load(id: String, precision: String?, fast: Bool?) {}
    func unload(id: String) {}
    func setKeepHot(id: String, on: Bool) {}
    func delete(id: String) {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject { try await worker.request(op: op, params: params, sink: sink) }
    func menuActions() -> [LoadedAction] { [] }
    func memory() -> MemorySnapshot { MemorySnapshot() }
    func catalogStatus() -> CatalogStatus { CatalogStatus() }
}
@Test func rawFacadeReturnsFailureBytes() async throws {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let code =
        "import sys; sys.path.insert(0, \(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers").path))); exec(open(\(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers/raw_worker.py").path))).read())"
    let w = WorkerProcess(); var launch = stubLaunch("decisions"); launch.executable = URL(fileURLWithPath: "/opt/homebrew/bin/python3"); launch.arguments = ["-I", "-c", code]
    try await w.start(launch)
    let service: any ModelsService = RawFixtureService(worker: w)
    let bytes = try await service.runRaw(id: "fixture", op: "decide", params: ["body": "error"])
    #expect(String(decoding: bytes, as: UTF8.self).contains("\"é\":1,\"e\\u0301\":2"))
    #expect(try WireJSON.decode(bytes)["ok"] == false)
    let result = try await service.runRawResult(id: "fixture", op: "decide", params: ["body": "error"])
    #expect(result.status == 409); #expect(!result.ok)
    await w.unload()
}

func runtimeFixtureLaunch() -> WorkerLaunch {
    var launch = stubLaunch()
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    launch.executable = URL(fileURLWithPath: "/opt/homebrew/bin/python3")
    launch.arguments = ["-I", root.appendingPathComponent("Fixtures/runtime_worker.py").path]
    launch.log = URL(fileURLWithPath: "/tmp/arsenal-runtime/logs/\(UUID().uuidString).log")
    return launch
}
struct SinkGone: Error {}
extension RuntimeTimingTests {
    @Test static func runtimeSinkFailurePreservesWorker() async throws {
        for op in ["stream", "binary"] {
            let w = WorkerProcess(cancellationGrace: 10); try await w.start(runtimeFixtureLaunch())
            do { _ = try await w.request(op: op, sink: { _ in throw SinkGone() }); Issue.record("Expected sink failure") } catch is SinkGone {} catch {
                Issue.record("Sink error misreported: \(error)")
            }
            #expect(try await w.request(op: "ping", timeout: 15)["ok"] == true)
            #expect(!(await w.exited))
            await w.unload()
        }
    }
}
extension RuntimeTimingTests {
    @Test static func runtimeStreamingUsesStallNotTotalDeadline() async throws {
        let w = WorkerProcess(); try await w.start(runtimeFixtureLaunch())
        for op in ["stream", "binary"] { #expect(try await w.request(op: op, params: ["chunks": 120], timeout: 2, firstResponseTimeout: 15)["ok"] == true) }
        await w.unload()
    }
}
extension RuntimeTimingTests {
    @Test static func runtimeCancelAcknowledgementDoesNotEndDrain() async throws {
        let w = WorkerProcess(cancellationGrace: 0.15); try await w.start(runtimeFixtureLaunch())
        let task = Task { try await w.request(op: "ack_ignore", timeout: 10) }
        try await Task.sleep(for: .milliseconds(40)); task.cancel()
        do { _ = try await task.value; Issue.record("Expected cancellation") } catch is CancellationError {}
        try await Task.sleep(for: .milliseconds(650))
        #expect(await w.exited); #expect(!(await w.unexpectedExit))
        await w.unload()
    }
}

extension RuntimeTimingTests {
    @Test static func runtimeSlowSinkHasBoundedDrain() async throws {
        let w = WorkerProcess(cancellationGrace: 0.1); try await w.start(runtimeFixtureLaunch())
        let start = Date()
        do { _ = try await w.request(op: "stream", timeout: 0.1, sink: { _ in try await Task.sleep(for: .seconds(20)) }); Issue.record("Expected stall") } catch let error
            as WorkerFailure
        { #expect(error.code == "timeout") }
        await w.unload(grace: 0.1)
        #expect(Date().timeIntervalSince(start) < 15)
    }
}
@Test func runtimeLeaderExitCleansChildrenAndReaders() async throws {
    let w = WorkerProcess(); try await w.start(runtimeFixtureLaunch())
    let reply = try await w.request(op: "exit_child")
    let child = Int32(try #require(reply["child"]?.number))
    let start = Date(); await w.unload(grace: 0.1)
    #expect(Date().timeIntervalSince(start) < 15)
    #expect(kill(child, 0) == -1)
    let log = URL(fileURLWithPath: "/tmp/arsenal-runtime/logs/command-\(UUID().uuidString).log")
    // Time this command independently of the preceding worker's shutdown. Python startup and
    // the 200 ms reader grace need the same busy-machine allowance as the other lifecycle checks.
    let commandStart = ContinuousClock.now
    let result = try await runBoundedCommand(
        URL(fileURLWithPath: "/opt/homebrew/bin/python3"), ["-I", "-c", "import os,time; p=os.fork(); time.sleep(60) if p == 0 else None"], environment: [:], log: log, timeout: 15)
    #expect(result.isEmpty); #expect(ContinuousClock.now - commandStart < .seconds(15))
}
@Test func runtimeEnvironmentAllowlistAndLogCap() async throws {
    var launch = runtimeFixtureLaunch()
    launch.environment.merge([
        "OPENAI_API_KEY": "secret", "ANTHROPIC_API_KEY": "secret", "AWS_ACCESS_KEY_ID": "secret", "AZURE_TOKEN": "secret", "CUSTOM_SECRET": "secret", "UV_INDEX_URL": "secret",
        "UV_CACHE_DIR": "/tmp/no", "ARSENAL_SUPPORT_DIR": "/tmp/arsenal-runtime/support"
    ]) { _, new in new }
    let w = WorkerProcess(); try await w.start(launch)
    let env = try #require(try await w.request(op: "env")["env"]?.object)
    #expect(env["ARSENAL_SUPPORT_DIR"] == "/tmp/arsenal-runtime/support")
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID", "AZURE_TOKEN", "CUSTOM_SECRET", "UV_INDEX_URL", "UV_CACHE_DIR", "HF_TOKEN", "DYLD_TEST"] {
        #expect(env[key] == nil)
    }
    _ = try await w.request(op: "logs"); await w.unload()
    let size = try #require(try FileManager.default.attributesOfItem(atPath: launch.log.path)[.size] as? Int)
    #expect(size > 0); #expect(size <= ChildProcess.logLimit)

}
extension RuntimeTimingTests {
    @Test static func runtimeManualLoadResetsQuarantineAndSuccessBudget() async throws {
        let s = Supervisor(crashLimit: 1)
        let bad = try await s.worker(id: "bad", launch: stubLaunch())
        _ = try? await bad.request(op: "test_crash", timeout: 10); await bad.unload()
        do { _ = try await s.worker(id: "bad", launch: stubLaunch()); Issue.record("Expected quarantine") } catch let e as WorkerFailure { #expect(e.code == "quarantined") }
        let recovered = try await s.worker(id: "bad", launch: stubLaunch(), manual: true)
        _ = try await recovered.request(op: "ping")
        _ = try? await recovered.request(op: "test_crash", timeout: 10); await recovered.unload()
        // Success resets the old budget; this new crash consumes the one allowed failure.
        do { _ = try await s.worker(id: "bad", launch: stubLaunch()); Issue.record("Expected new crash quarantine") } catch let e as WorkerFailure {
            #expect(e.code == "quarantined")
        }
        await s.shutdown()
    }
}

@Test func runtimeManualLoadImmediatelyAfterCrash() async throws {
    let s = Supervisor(crashLimit: 1)
    let old = try await s.worker(id: "bad", launch: stubLaunch())
    _ = try? await old.request(op: "test_crash", timeout: 10); await old.unload()
    let recovered = try await s.worker(id: "bad", launch: stubLaunch(), manual: true)
    #expect(try await recovered.request(op: "ping")["ok"] == true)
    await s.shutdown()
}
@Test func runtimeHealthyReplyResetsOldCrashBudget() async throws {
    let s = Supervisor(crashLimit: 2)
    let first = try await s.worker(id: "bad", launch: stubLaunch())
    _ = try? await first.request(op: "test_crash", timeout: 10); await first.unload()
    let second = try await s.worker(id: "bad", launch: stubLaunch())
    _ = try await second.request(op: "ping")
    _ = try? await second.request(op: "test_crash", timeout: 10); await second.unload()
    let third = try await s.worker(id: "bad", launch: stubLaunch())
    #expect(try await third.request(op: "ping")["ok"] == true)
    await s.shutdown()
}
@Test func runtimeBoundedCommandEnvironmentAndLogs() async throws {
    let log = URL(fileURLWithPath: "/tmp/arsenal-runtime/logs/fixture-command-\(UUID().uuidString).log")
    let text = try await runBoundedCommand(
        URL(fileURLWithPath: "/opt/homebrew/bin/python3"), ["-I", "-c", "import os,json; os.write(2,b'x'*(6*1024*1024)); print(json.dumps(dict(os.environ)))"],
        environment: [
            "UV_CACHE_DIR": "/tmp/arsenal-runtime/uv", "UV_NO_CONFIG": "1", "OPENAI_API_KEY": "secret", "AWS_SECRET_ACCESS_KEY": "secret", "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": "secret", "UV_INDEX_URL": "secret"
        ], log: log, timeout: 15)
    let env = try WireJSON.decode(Data(text.utf8))
    #expect(env["HF_HUB_OFFLINE"] == "1")
    for key in ["OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "UV_CACHE_DIR", "UV_NO_CONFIG", "PYTHONPATH", "UV_INDEX_URL"] { #expect(env[key] == nil) }
    let size = try #require(try FileManager.default.attributesOfItem(atPath: log.path)[.size] as? Int)
    #expect(size > 0); #expect(size <= ChildProcess.logLimit)
}
@Test(.timeLimit(.minutes(1))) func runtimePipeReadersStopAfterEscapedDescendant() async throws {
    let log = URL(fileURLWithPath: "/tmp/arsenal-runtime/logs/escaped-\(UUID().uuidString).log")
    // Keep a separate PID receipt so even a timeout or a malformed stdout reply cleans up the
    // deliberately escaped process, which the command's process group cannot kill.
    let receipt = log.appendingPathExtension("pid")
    defer {
        if let text = try? String(contentsOf: receipt, encoding: .utf8), let child = Int32(text) { _ = kill(child, SIGKILL) }
        try? FileManager.default.removeItem(at: receipt)
    }
    let start = ContinuousClock.now
    let text = try await runBoundedCommand(
        URL(fileURLWithPath: "/opt/homebrew/bin/python3"),
        [
            "-I", "-c",
            "import os,time; r,w=os.pipe(); p=os.fork()\nif p==0:\n os.close(r); os.setsid(); open(\(String(reflecting: receipt.path)), 'w').write(str(os.getpid())); print(os.getpid(),flush=True); os.write(w,b'1'); time.sleep(60)\nelse:\n os.close(w); os.read(r,1)"
        ], environment: [:], log: log, timeout: 15)
    let child = try #require(Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)))
    #expect(try String(contentsOf: receipt, encoding: .utf8) == String(child))
    #expect(kill(child, 0) == 0)
    // Finishing well before the 60 s descendant exits proves readers do not wait for pipe EOF.
    #expect(ContinuousClock.now - start < .seconds(15))
}
