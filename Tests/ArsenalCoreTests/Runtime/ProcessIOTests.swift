// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func manyIdleWorkersDoNotExhaustTheExecutor() async throws {
    try await withThrowingTaskGroup(of: Void.self) { group in
        for _ in 0..<40 {
            group.addTask {
                let worker = WorkerProcess()
                do {
                    try await worker.start(stubLaunch(), handshakeTimeout: 15)
                    #expect(try await worker.request(op: "ping", timeout: 15)["ok"] == true)
                    await worker.unload()
                } catch { await worker.unload(); throw error }
            }
        }
        try await group.waitForAll()
    }
}

@Test(arguments: 0..<32) func processExitSourcesReleaseTheirChild(_ iteration: Int) async throws {
    let log = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-exit-\(iteration)-\(UUID().uuidString).log")
    defer { try? FileManager.default.removeItem(at: log) }
    var child: ChildProcess? = try ChildProcess(
        executable: URL(fileURLWithPath: "/usr/bin/true"), arguments: [], environment: [:], log: log)
    weak var observed = child
    _ = await child?.awaitExitAsync()
    await child?.finishReaders()
    child?.closeInput(); child?.finishOutput(); child = nil
    // Exit completion includes cancellation of the dispatch source; no release polling is needed.
    #expect(observed == nil)
}
