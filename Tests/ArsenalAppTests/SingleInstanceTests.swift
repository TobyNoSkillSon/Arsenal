// SPDX-License-Identifier: MIT
import Darwin
import Foundation
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct SingleInstanceTests {
    private var executable: URL {
        if let server = ProcessInfo.processInfo.environment["ARSENAL_TEST_SERVER"] {
            return URL(fileURLWithPath: server).deletingLastPathComponent().appendingPathComponent("Arsenal")
        }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent(".build/debug/Arsenal")
    }
    private func launch(_ support: URL) throws -> Process {
        let process = Process(); process.executableURL = executable
        process.environment = [
            "HOME": NSHomeDirectory(), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "TMPDIR": "/tmp", "ARSENAL_STUB": "1",
            "ARSENAL_SUPPORT_DIR": support.path, "ARSENAL_CACHE_DIR": support.appendingPathComponent("cache").path
        ]
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        try process.run(); return process
    }
    private func waitUntil(_ condition: () throws -> Bool) async throws {
        let deadline = Date().addingTimeInterval(10)
        while try !condition() {
            guard Date() < deadline else { throw TestLaunchError.deadline }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
    private func status(_ process: Process, _ support: URL) async throws -> [String: Int] {
        let file = support.appendingPathComponent("status.json")
        try await waitUntil {
            guard process.isRunning else { throw TestLaunchError.exited }
            guard let data = try? Data(contentsOf: file), let saved = try? JSONDecoder().decode([String: Int].self, from: data) else { return false }
            return saved["pid"] == Int(process.processIdentifier) && (saved["port"] ?? 0) > 0
        }
        return try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: file))
    }
    private func stop(_ process: Process) {
        if process.isRunning { kill(process.processIdentifier, SIGKILL); process.waitUntilExit() }
    }
    private enum TestLaunchError: Error { case deadline, exited }

    @Test func sameSupportSecondLaunchExitsWithoutChangingFirstInstance() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-single-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try launch(root); defer { stop(first) }
        let original = try await status(first, root), file = root.appendingPathComponent("status.json")
        let bytes = try Data(contentsOf: file), modified = try file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        let second = try launch(root); defer { stop(second) }
        try await waitUntil { !second.isRunning }
        #expect(second.terminationStatus == 0 && first.isRunning)
        #expect(try Data(contentsOf: file) == bytes)
        #expect(try file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate == modified)
        #expect(try await status(first, root) == original)
    }

    @Test func differentSupportLaunchRunsAndFallsBackFromOccupiedPreferredPort() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-separate-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a"), b = root.appendingPathComponent("b")
        let first = try launch(a); defer { stop(first) }
        let firstStatus = try await status(first, a), firstBytes = try Data(contentsOf: a.appendingPathComponent("status.json"))
        let second = try launch(b); defer { stop(second) }
        let secondStatus = try await status(second, b)
        #expect(first.isRunning && second.isRunning)
        #expect(firstStatus["pid"] != secondStatus["pid"] && firstStatus["port"] != secondStatus["port"])
        // Either the installed app or the first test instance owns 8798; the second must fall back.
        #expect(secondStatus["port"] != 8798)
        #expect(try Data(contentsOf: a.appendingPathComponent("status.json")) == firstBytes)
    }

    @Test func crashedOwnerLeavesLockFileThatTheNextLaunchReclaims() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-stale-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try launch(root); defer { stop(first) }
        _ = try await status(first, root)
        let lock = root.appendingPathComponent(".instance.lock")
        let inode = try FileManager.default.attributesOfItem(atPath: lock.path)[.systemFileNumber] as? NSNumber
        kill(first.processIdentifier, SIGKILL); first.waitUntilExit()
        #expect(FileManager.default.fileExists(atPath: lock.path))
        let recovered = try launch(root); defer { stop(recovered) }
        let recoveredStatus = try await status(recovered, root)
        #expect(recovered.isRunning && recoveredStatus["pid"] == Int(recovered.processIdentifier))
        #expect(try FileManager.default.attributesOfItem(atPath: lock.path)[.systemFileNumber] as? NSNumber == inode)
    }
}
