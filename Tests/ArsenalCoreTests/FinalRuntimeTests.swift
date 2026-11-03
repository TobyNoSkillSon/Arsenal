// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import Testing
@testable import ArsenalCore

@Test func finalSpawnResetsIgnoredSignalsAndBlockedMask() async throws {
    // Preserve parent state; only synchronous spawn observes these temporary dispositions.
    let previous = signal(SIGTERM, SIG_IGN)
    var blocked = sigset_t(), saved = sigset_t()
    sigemptyset(&blocked); sigaddset(&blocked, SIGTERM)
    pthread_sigmask(SIG_BLOCK, &blocked, &saved)
    let child: ChildProcess
    do {
        defer { signal(SIGTERM, previous); pthread_sigmask(SIG_SETMASK, &saved, nil) }
        let launch = runtimeFixtureLaunch()
        child = try ChildProcess(executable: launch.executable, arguments: launch.arguments, environment: RunnerEnvironment.scrub(launch.environment), log: launch.log)
    }
    _ = try await child.readPacket() // hello, process definitely executed
    child.signal(SIGTERM)
    try await Task.sleep(for: .milliseconds(120))
    var info = siginfo_t()
    let observed = waitid(P_PID, id_t(child.pid), &info, WEXITED | WNOHANG | WNOWAIT)
    #expect(observed == 0 && info.si_pid == child.pid) // Must exit at TERM, not require KILL.
    child.signal(SIGKILL) // bounded cleanup even when reproducing old broken behavior
    _ = await child.awaitExitAsync()
    child.finishOutput(); await child.finishReaders(); child.closeInput()
    #expect(kill(child.pid, 0) == -1)
}

@Test func finalResidentRequestsDoNotReverifyButExplicitLoadDoes() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: ProtocolFixtureArtifact(scenario: "normal")); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    try await f.service.load(id: "fixture-chat")
    let extra = f.paths.directory("models").appendingPathComponent("fixture-chat/unlisted.py")
    try Data("not executable resident code".utf8).write(to: extra)
    do { #expect(try await f.service.run(id: "fixture-chat", op: "inspect_load")["ok"] == true) } catch { Issue.record("Resident request unexpectedly reverified: \(error)") }
    await #expect(throws: (any Error).self) { try await f.service.load(id: "fixture-chat") }
    await f.service.shutdown()
}

@Test func finalFinderMetadataDoesNotBlockLoadButOtherFilesDo() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: ProtocolFixtureArtifact(scenario: "normal")); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    let folder = f.paths.directory("models").appendingPathComponent("fixture-chat")
    for path in [".DS_Store", "gate/.DS_Store"] { try Data("finder metadata".utf8).write(to: folder.appendingPathComponent(path)) }
    do { try await f.service.load(id: "fixture-chat") } catch { Issue.record("Finder metadata rejected: \(error)") }
    try Data("not Finder metadata".utf8).write(to: folder.appendingPathComponent(".DS_Store.py"))
    await #expect(throws: (any Error).self) { try await f.service.load(id: "fixture-chat") }
    await f.service.shutdown()
}

extension RuntimeTimingTests {
    @Test static func finalLongPrefillThenFramesUsesSeparateStallBudget() async throws {
        var launch = runtimeFixtureLaunch()
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let code =
            "import sys,time; sys.path.insert(0, \(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers").path))); from arsenal_worker import Worker; w=Worker('chat',runner='native'); w.op('chat')(lambda r: (time.sleep(3),w.emit(r,'token',text='first'),time.sleep(.04),{'text':'ok'})[-1]); w.run()"
        launch.arguments = ["-I", "-c", code]
        let worker = WorkerProcess(); try await worker.start(launch)
        do { #expect(try await worker.request(op: "chat", timeout: 2, firstResponseTimeout: 15)["ok"] == true) } catch {
            Issue.record("Prefill hit streaming stall deadline: \(error)")
        }
        await worker.unload(grace: 0.1)
    }
}

extension RuntimeTimingTests {
    @Test static func finalFirstFrameSwitchesToShortStallDeadline() async throws {
        var launch = runtimeFixtureLaunch()
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let code =
            "import sys,time; sys.path.insert(0, \(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers").path))); from arsenal_worker import Worker; w=Worker('chat',runner='native'); w.op('chat')(lambda r: (w.emit(r,'token',text='first'),time.sleep(6),{'text':'late'})[-1]); w.run()"
        launch.arguments = ["-I", "-c", code]
        let worker = WorkerProcess(); try await worker.start(launch)
        let start = Date()
        do { _ = try await worker.request(op: "chat", timeout: 2, firstResponseTimeout: 15); Issue.record("Post-token stall inherited prefill allowance") } catch let error
            as WorkerFailure
        { #expect(error.code == "timeout"); #expect(Date().timeIntervalSince(start) < 15) }
        await worker.unload(grace: 0.1)
    }
}
