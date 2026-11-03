// SPDX-License-Identifier: MIT
import Darwin
import Foundation
import Testing
@testable import ArsenalCore

@Test func pipeReaderClosesOwnedDescriptorAfterEOF() async throws {
    var descriptors: [Int32] = [0, 0]
    try #require(pipe(&descriptors) == 0)
    // Keep the observed slot away from unrelated tests' low-FD allocations.
    let descriptor = fcntl(descriptors[0], F_DUPFD_CLOEXEC, 65_536)
    close(descriptors[0])
    try #require(descriptor >= 0)
    var reader: AsyncPipeReader? = AsyncPipeReader(descriptor: descriptor)
    close(descriptors[1])
    #expect(try await reader?.read(count: 1) == nil)
    let deadline = Date().addingTimeInterval(1)
    while fcntl(descriptor, F_GETFD) != -1 && Date() < deadline { try await Task.sleep(for: .milliseconds(1)) }
    let result = fcntl(descriptor, F_GETFD), error = errno
    try #require(result == -1 && error == EBADF)
    reader?.finish()
    #expect(try await reader?.read(count: 1) == nil)
    // Reuse the exact slot; repeated finish and deinit must not close the next owner's FD.
    try #require(pipe(&descriptors) == 0)
    try #require(dup2(descriptors[0], descriptor) == descriptor)
    close(descriptors[0])
    defer { close(descriptor); close(descriptors[1]) }
    reader?.finish(); reader = nil
    try await Task.sleep(for: .milliseconds(10))
    #expect(fcntl(descriptor, F_GETFD) >= 0)
}

@Test func badPacketFinishesOutputBeforeProcessExit() async throws {
    let child = try ChildProcess(
        executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", "printf '\\002\\000\\000\\000x'; sleep 60"], environment: ["PATH": "/usr/bin:/bin"],
        log: URL(fileURLWithPath: "/tmp/arsenal-mb-pol-fix-core/bad-frame-\(UUID().uuidString).log"))
    await #expect(throws: (any Error).self) { _ = try await child.readPacket() }
    // An oversized header must cancel the reader now, not when the still-running child exits.
    #expect(try await child.readOutputAsync(count: 1) == nil)
    child.signal(SIGKILL); _ = await child.awaitExitAsync()
    await child.finishReaders(); child.closeInput()
}

@Test func rapidWorkersKeepTheirOwnFramesAcrossDescriptorReuse() async throws {
    try await withThrowingTaskGroup(of: Void.self) { group in
        for lane in 0..<12 {
            group.addTask {
                for turn in 0..<8 {
                    let worker = WorkerProcess(), identity = "\(lane)-\(turn)"
                    do {
                        try await worker.start(stubLaunch(), handshakeTimeout: 15)
                        let result = try await worker.request(op: "test_echo", params: ["identity": .string(identity)], timeout: 15)
                        #expect(result["params"]?["identity"]?.string == identity)
                        if turn % 2 == 0 { _ = try? await worker.request(op: "test_oversize", timeout: 15) }
                        await worker.unload(grace: 0.1)
                        #expect(await worker.exited)
                    } catch { await worker.unload(); throw error }
                }
            }
        }
        try await group.waitForAll()
    }
}
