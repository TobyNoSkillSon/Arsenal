// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal

/// AppKit menus and Appearance.shared are process state, even when their windows are offscreen.
/// MainActor alone does not isolate an async test across its suspension points.
struct MenuStateIsolation: TestTrait, SuiteTrait, TestScoping {
    var isRecursive: Bool { true }
    private static let gate = Gate()

    func provideScope(for test: Test, testCase: Test.Case?, performing function: @Sendable () async throws -> Void) async throws {
        await Self.gate.acquire()
        let held = ContinuousClock.now
        let previous = await MainActor.run { AppearanceSnapshot() }
        do {
            try await function()
        } catch {
            await previous.restore()
            Self.record(test, since: held)
            await Self.gate.release()
            throw error
        }
        await previous.restore()
        Self.record(test, since: held)
        await Self.gate.release()
    }

    /// scripts/test-targets.py splits the app suites across processes by how long each holds this gate, which is the
    /// work one process must do in series. It names the file; without it nothing is written.
    private static func record(_ test: Test, since held: ContinuousClock.Instant) {
        guard let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_GATE_TIMES"],
            let suite = test.id.nameComponents.first
        else { return }
        let elapsed = ContinuousClock.now - held
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let line = Data("\(suite)\t\(seconds)\n".utf8)
        // Only the gate holder writes, so appends never interleave within the process.
        if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
        guard let handle = FileHandle(forWritingAtPath: path) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: line)
    }

    @MainActor private final class AppearanceSnapshot {
        let appearance = NSApplication.shared.appearance
        let mode = Appearance.shared.mode
        func restore() {
            NSApplication.shared.appearance = appearance
            // Drain appearance notifications queued by the test before restoring the fixture mode.
            RunLoop.main.run(until: Date(timeIntervalSinceNow: 1.0 / 120))
            Appearance.shared.set(mode)
            RunLoop.main.run(until: Date(timeIntervalSinceNow: 1.0 / 120))
        }
    }

    private actor Gate {
        var held = false
        var waiters: [CheckedContinuation<Void, Never>] = []
        func acquire() async {
            if !held { held = true; return }
            await withCheckedContinuation { waiters.append($0) }
        }
        func release() {
            if waiters.isEmpty { held = false } else { waiters.removeFirst().resume() }
        }
    }
}

extension Trait where Self == MenuStateIsolation {
    static var menuState: Self { Self() }
}
