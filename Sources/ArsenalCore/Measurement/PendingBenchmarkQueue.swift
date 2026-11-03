// SPDX-License-Identifier: MIT
import Foundation
import Darwin

public struct PendingBenchmark: Codable, Sendable, Equatable, Identifiable {
    public var model: String
    public var kind: ModelStatKind
    public var dtype: String
    public var sourceDtype: String
    public var revision: String
    public var runner: String
    public var mode: String
    public var introducedAt: String
    public var state: String = "pending"
    public var error: String?
    public var result: String?
    public var id: String { [model, revision, runner, dtype, mode].joined(separator: ":") }
    public init(model: String, kind: ModelStatKind, dtype: String, sourceDtype: String, revision: String, runner: String, mode: String) {
        self.model = model; self.kind = kind; self.dtype = dtype; self.sourceDtype = sourceDtype
        self.revision = revision; self.runner = runner; self.mode = mode
        introducedAt = ISO8601DateFormatter().string(from: Date())
    }
}

/// Shared Swift/script JSON contract. A lock protects read-modify-write against another app/agent.
/// No queue method loads, downloads or measures a model. Only the explicit agent runner drains it.
public struct PendingBenchmarkQueue: Sendable {
    public let file: URL
    public init(support: URL) { file = support.appendingPathComponent("benchmarks/pending.json") }
    public func read() throws -> [PendingBenchmark] { try locked { try readUnlocked() } }
    private func readUnlocked() throws -> [PendingBenchmark] {
        guard FileManager.default.fileExists(atPath: file.path) else { return [] }
        return try JSONDecoder().decode([PendingBenchmark].self, from: Data(contentsOf: file))
    }
    @discardableResult public func enqueue(_ entries: [PendingBenchmark]) throws -> Int {
        try locked {
            var values = try readUnlocked(), ids = Set(values.map(\.id))
            var revived = 0
            for entry in entries {
                if let index = values.firstIndex(where: { $0.id == entry.id && $0.state == "superseded" }) { values[index] = entry; revived += 1 }
            }
            let new = entries.filter { ids.insert($0.id).inserted }
            guard !new.isEmpty || revived > 0 else { return 0 }
            values.append(contentsOf: new)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(values).write(to: file, options: .atomic)
            return new.count + revived
        }
    }
    @discardableResult public func introduce(_ model: ModelManifest, runner: String, mode: String) throws -> Int {
        guard let kind = ModelStatKind(rawValue: model.kind), !kind.isOperation else { return 0 }
        try locked {
            var entries = try readUnlocked()
            var changed = false
            for index in entries.indices where entries[index].model == model.id && entries[index].state == "pending" {
                if entries[index].revision != model.weights.revision || (entries[index].runner == runner && entries[index].mode != mode) {
                    entries[index].state = "superseded"
                    entries[index].error = "Revision or automatic serving mode changed; current entries replace this pending run."
                    changed = true
                }
            }
            if changed { try JSONEncoder().encode(entries).write(to: file, options: .atomic) }
        }
        return try enqueue(
            model.precisions.map { dtype in
                PendingBenchmark(model: model.id, kind: kind, dtype: dtype, sourceDtype: model.basePrecision, revision: model.weights.revision, runner: runner, mode: mode)
            })
    }
    public func complete(id: String, result: String) throws {
        try update(id: id) { entry in
            entry.state = "complete"; entry.result = result; entry.error = nil
        }
    }
    public func recordFailure(id: String, reason: String) throws {
        try update(id: id) { entry in
            entry.state = "pending"; entry.error = reason
        }
    }
    private func update(id: String, operation: (inout PendingBenchmark) -> Void) throws {
        try locked {
            var entries = try readUnlocked()
            guard let index = entries.firstIndex(where: { $0.id == id }) else { throw ManifestError.invalid("Benchmark queue entry missing") }
            operation(&entries[index])
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(entries).write(to: file, options: .atomic)
        }
    }
    private func locked<T>(_ operation: () throws -> T) throws -> T {
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let descriptor = Darwin.open(file.path + ".lock", O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw ManifestError.invalid("Cannot open benchmark queue lock") }
        defer { Darwin.close(descriptor) }
        // Bounded nonblocking acquisition avoids macOS blocking-flock missed wakeups.
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while flock(descriptor, LOCK_EX | LOCK_NB) != 0 {
            guard errno == EWOULDBLOCK, ProcessInfo.processInfo.systemUptime < deadline else { throw ManifestError.invalid("Benchmark queue is busy") }
            Thread.sleep(forTimeInterval: 0.05)
        }
        defer { flock(descriptor, LOCK_UN) }
        return try operation()
    }
}
