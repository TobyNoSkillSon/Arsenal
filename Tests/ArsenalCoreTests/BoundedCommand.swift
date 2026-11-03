// SPDX-License-Identifier: MIT
import Foundation
@testable import ArsenalCore

/// Drains output continuously, keeps only a bounded tail, and kills the whole group on timeout.
func runBoundedCommand(_ executable: URL, _ arguments: [String], environment: [String: String], log: URL, timeout: Double) async throws -> String {
    let child = try ChildProcess(executable: executable, arguments: arguments, environment: RunnerEnvironment.scrub(environment), log: log)
    child.closeInput()
    let timer = Task.detached { do { try await Task.sleep(for: .seconds(timeout)); child.signal(SIGTERM); try await Task.sleep(for: .seconds(1)); child.signal(SIGKILL) } catch {} }
    let reader = Task.detached { () throws -> String in
        defer { child.finishOutput() }
        var tail = Data()
        while let data = try await child.readOutputAsync(count: 16_384), !data.isEmpty { tail.append(data); if tail.count > 65_536 { tail = Data(tail.suffix(65_536)) } }
        return String(decoding: tail, as: UTF8.self)
    }
    return try await withTaskCancellationHandler {
        let status = await child.awaitExitAsync(); timer.cancel()
        let output = try await reader.value
        await child.finishReaders()
        try Task.checkCancellation()
        guard status == 0 else { throw WorkerFailure(code: "environment", message: "Fixture command failed; see \(log.lastPathComponent).") }
        return output
    } onCancel: {
        child.signal(SIGKILL)
    }
}
