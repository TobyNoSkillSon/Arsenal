// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func cancelledWorkerCleanupWaitsForProcessExitWithoutSpinning() async throws {
    let worker = WorkerProcess(); try await worker.start(stubLaunch())
    let pid = await worker.pid
    let cleanup = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        await worker.unload(grace: 0.02)
    }
    await cleanup.value
    #expect(await worker.exited)
    #expect(kill(pid, 0) == -1)
}

@Test func workerDeadlineCancellingItsOwnTimerStillTerminates() async throws {
    let worker = WorkerProcess(); try await worker.start(stubLaunch())
    let pid = await worker.pid
    await #expect(throws: WorkerFailure.self) { _ = try await worker.request(op: "test_hang", timeout: 0.03) }
    await worker.unload(grace: 0.02)
    #expect(await worker.exited)
    #expect(kill(pid, 0) == -1)
}

@Test func cancelledWorkerCleanupOfAnUnresponsiveChildYieldsDuringGrace() async throws {
    let worker = WorkerProcess(); try await worker.start(stubLaunch())
    let hang = Task { try? await worker.request(op: "test_ignore_term", timeout: 5) }
    try await Task.sleep(for: .milliseconds(50))
    var before = rusage(), after = rusage()
    getrusage(RUSAGE_SELF, &before)
    let started = Date()
    let cleanup = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        await worker.unload(grace: 0.1)
    }
    await cleanup.value
    getrusage(RUSAGE_SELF, &after)
    let cpu =
        Double(after.ru_utime.tv_sec - before.ru_utime.tv_sec) + Double(after.ru_utime.tv_usec - before.ru_utime.tv_usec) / 1e6
        + Double(after.ru_stime.tv_sec - before.ru_stime.tv_sec) + Double(after.ru_stime.tv_usec - before.ru_stime.tv_usec) / 1e6
    print("EXIT-CANCEL: wall=\(Date().timeIntervalSince(started)) cpu=\(cpu)")
    // Process-wide CPU is meaningful only in the explicitly isolated probe, not beside the parallel suite.
    if ProcessInfo.processInfo.environment["ARSENAL_EXIT_CPU_CHECK"] == "1" { #expect(cpu < 0.05) }
    #expect(await worker.exited)
    hang.cancel(); _ = await hang.value
}
