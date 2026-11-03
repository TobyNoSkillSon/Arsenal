// SPDX-License-Identifier: MIT
import Foundation
import ArsenalAPI
import ArsenalCore

@main struct TestServer {
    static func main() async {
        do { try await execute() } catch {
            let line = error.localizedDescription.components(separatedBy: .newlines).filter { !$0.isEmpty }.joined(separator: " ")
            try? FileHandle.standardError.write(contentsOf: Data("Arsenal QA failed: \(line)\n".utf8))
            Foundation.exit(1)
        }
    }
    private static func execute() async throws {
        guard ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] != nil, ProcessInfo.processInfo.environment["ARSENAL_CACHE_DIR"] != nil else {
            print("Set ARSENAL_SUPPORT_DIR and ARSENAL_CACHE_DIR to private test directories."); Foundation.exit(2)
        }
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.first == "--prepare-real", arguments.count == 2 {
            try await RealTestSetup.prepare(repository: URL(fileURLWithPath: arguments[1]), paths: ArsenalPaths()); return
        }
        let service: any ModelsService
        if arguments.first == "--real", arguments.count == 3 {
            let paths = ArsenalPaths(), repository = URL(fileURLWithPath: arguments[1]), helpers = URL(fileURLWithPath: arguments[2])
            service = try await RealTestSetup.open(repository: repository, paths: paths, helpers: helpers)
        } else if ProcessInfo.processInfo.environment["ARSENAL_TESTSERVER_CATALOG"] == "stub" {
            service = StubModelsService()
        } else {
            service = FixtureModelsService()
        }
        let server = ArsenalAPIServer(service: service)
        let port = try await server.start()
        try FileHandle.standardOutput.write(contentsOf: Data("Arsenal test API http://127.0.0.1:\(port)\n".utf8))
        await TestServerSignals().wait()
        server.stop()
        await service.shutdown()
    }
}

/// Fixture and Layer 3 hosts shut down their own workers on termination, rather
/// than leave process groups behind when a test fails.
private final class TestServerSignals: @unchecked Sendable {
    private let lock = NSLock()
    private var finished = false
    private var sources: [DispatchSourceSignal] = []
    func wait() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            for value in [SIGTERM, SIGINT] {
                signal(value, SIG_IGN)
                let source = DispatchSource.makeSignalSource(signal: value, queue: .global())
                source.setEventHandler { [self] in
                    let first = lock.withLock {
                        if finished { return false }; finished = true; return true
                    }
                    if first { continuation.resume() }
                }
                sources.append(source); source.resume()
            }
        }
        for source in sources { source.cancel() }
        sources.removeAll()
    }
}
