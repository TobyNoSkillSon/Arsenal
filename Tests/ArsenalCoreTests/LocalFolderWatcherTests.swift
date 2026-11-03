// SPDX-License-Identifier: MIT
import Foundation
import Synchronization
import Testing
@testable import ArsenalCore

/// A model folder dropped into local-models reaches the service without a click: one callback, after the copy ends.
@Suite(.serialized) struct LocalFolderWatcherTests {
    static func wait(_ calls: borrowing Atomic<Int>, above: Int, seconds: Double) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while calls.load(ordering: .relaxed) <= above, Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
    }

    @Test func aDroppedFolderIsReportedOnceAfterItSettlesAndAQuietFolderNever() async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-watch-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let root = support.appendingPathComponent("local-models", isDirectory: true)
        let calls = Atomic<Int>(0)
        let watcher = try LocalFolderWatcher(root: root, settle: 1) { calls.add(1, ordering: .relaxed) }
        // The folder Open Arsenal Files shows exists before anything is dropped.
        #expect(FileManager.default.fileExists(atPath: root.path))
        watcher.start()
        defer { watcher.stop() }
        #expect(watcher.running)
        try await Task.sleep(for: .seconds(1))
        #expect(calls.load(ordering: .relaxed) == 0)

        // A slow copy: a file every 0.4 s for 4 s. Nothing fires while it runs (a rescan would hash half-written files).
        let folder = root.appendingPathComponent("my-model", isDirectory: true)
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("weights"), withIntermediateDirectories: true)
        let handle = try FileHandle(
            forWritingTo: {
                let f = folder.appendingPathComponent("weights/model.safetensors"); FileManager.default.createFile(atPath: f.path, contents: nil); return f
            }())
        for _ in 0..<10 {
            try handle.write(contentsOf: Data(repeating: 1, count: 1 << 16))
            try handle.synchronize()
            try await Task.sleep(for: .milliseconds(400))
        }
        try handle.close()
        #expect(calls.load(ordering: .relaxed) == 0, "fired while the copy was still running")
        try Data("schema = 3\n".utf8).write(to: folder.appendingPathComponent("model.toml"))
        try await Self.wait(calls, above: 0, seconds: LocalFolderWatcher.latency + 3 * watcher.settle + 3)
        try await Task.sleep(for: .seconds(LocalFolderWatcher.latency + watcher.settle + 0.5))
        #expect(calls.load(ordering: .relaxed) == 1)

        // Removal is a change too.
        try FileManager.default.removeItem(at: folder)
        try await Self.wait(calls, above: 1, seconds: LocalFolderWatcher.latency + 3 * watcher.settle + 3)
        #expect(calls.load(ordering: .relaxed) == 2)
        watcher.stop()
        #expect(!watcher.running)
        // Nothing after stop.
        try FileManager.default.createDirectory(at: root.appendingPathComponent("late"), withIntermediateDirectories: true)
        try await Task.sleep(for: .seconds(LocalFolderWatcher.latency + watcher.settle + 1))
        #expect(calls.load(ordering: .relaxed) == 2)
    }
}
