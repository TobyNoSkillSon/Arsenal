// SPDX-License-Identifier: MIT
import Foundation

/// Persisted kernel verdicts: compile results per instantiation and Exact/Fast gate results per model, keyed by
/// everything that changes the answer (kernel identity, OS build, GPU, the caller's own key: model, dtype, weights,
/// check revision…). File: `$ARSENAL_SUPPORT_DIR/state/kernels.json` (default `~/Library/Application Support/Arsenal/
/// state/`). A verdict is a cache: deleting the file only costs one recheck per kernel. Bounded: keys of an older
/// schema are dropped and at most `limit` entries are kept (oldest evicted first), so contributor iteration across
/// source revisions, builds and OS updates cannot grow it without end.
public final class KernelVerdicts: @unchecked Sendable {
    public struct Verdict: Codable, Sendable, Equatable {
        public var pass: Bool
        public var detail: String
        public var date: Double
        public init(pass: Bool, detail: String, date: Double = Date().timeIntervalSince1970) { self.pass = pass; self.detail = detail; self.date = date }
    }

    /// Prefix of every key. Bump when what a verdict means changes (v2: full kernel identity, engagement-checked
    /// gates, weights in model gates): verdicts saved under another schema are never read and are pruned on write.
    public static let schema = "v2"
    /// Entries kept: ~a few hundred bytes each; enough for every kernel × dtype × instantiation of the folders on
    /// one Mac with room for a contributor's iterations.
    public static let defaultLimit = 512

    public static let shared = KernelVerdicts(file: defaultFile)

    public static var defaultFile: URL {
        let env = ProcessInfo.processInfo.environment
        let support = env["ARSENAL_SUPPORT_DIR"] ?? ((env["HOME"] ?? NSHomeDirectory()) + "/Library/Application Support/Arsenal")
        return URL(fileURLWithPath: support, isDirectory: true).appendingPathComponent("state/kernels.json")
    }

    public let file: URL
    public let limit: Int
    private let lock = NSLock()
    public init(file: URL, limit: Int = defaultLimit) { self.file = file; self.limit = max(1, limit) }

    /// Schema + stage + parts joined; always includes the host key.
    public static func key(_ stage: String, _ parts: String...) -> String {
        ([schema, stage] + parts + [KernelHost.key]).joined(separator: "|")
    }

    public func read(_ key: String) -> Verdict? {
        lock.lock(); defer { lock.unlock() }
        return all()[key]
    }

    public func write(_ key: String, _ verdict: Verdict) {
        lock.lock(); defer { lock.unlock() }
        var entries = all().filter { $0.key.hasPrefix(Self.schema + "|") }
        entries[key] = verdict
        if entries.count > limit {
            // Oldest first; the entry being written is always kept.
            let evict = entries.filter { $0.key != key }.sorted { ($0.value.date, $0.key) < ($1.value.date, $1.key) }
            for (old, _) in evict.prefix(entries.count - limit) { entries[old] = nil }
        }
        let directory = file.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        guard let data = try? encoder.encode(entries) else { return }
        // Write-then-rename: concurrent runners never read a torn file.
        let tmp = directory.appendingPathComponent("kernels.\(getpid()).\(UInt32.random(in: 0 ... .max)).tmp")
        if (try? data.write(to: tmp)) != nil, rename(tmp.path, file.path) != 0 { try? FileManager.default.removeItem(at: tmp) }
    }

    /// The saved verdict for `key`, or `run()`'s result saved under it.
    public func verdict(_ key: String, run: () -> Verdict) -> (verdict: Verdict, persisted: Bool) {
        if let saved = read(key) { return (saved, true) }
        let fresh = run()
        write(key, fresh)
        return (fresh, false)
    }

    /// A gate verdict whose pass counts only when the check really ran the kernels: `check` runs (kernels on and off)
    /// and its pass is replaced by a failure when any kernel failed its preflight or was never dispatched during it
    /// (the caller's code fell back to its stock expression, so "on" compared stock with stock). Such a run can never
    /// save a pass.
    public func gate(_ key: String, kernels: [any GatedKernel], check: () -> Verdict) -> (verdict: Verdict, persisted: Bool) {
        verdict(key) {
            let before = kernels.map(\.calls)
            let result = check()
            if let failed = kernels.first(where: { $0.failure != nil }), let failure = failed.failure {
                return Verdict(pass: false, detail: "kernel \(failed.gateName) failed its preflight: \(failure)")
            }
            if let idle = zip(kernels, before).first(where: { $0.0.calls <= $0.1 })?.0 {
                return Verdict(pass: false, detail: "the check never dispatched kernel \(idle.gateName)")
            }
            return result
        }
    }

    private func all() -> [String: Verdict] {
        guard let data = try? Data(contentsOf: file) else { return [:] }
        return (try? JSONDecoder().decode([String: Verdict].self, from: data)) ?? [:]
    }
}

/// What a gate needs from a kernel to know the check exercised it.
public protocol GatedKernel: AnyObject {
    /// For the failure detail (the manifest id).
    var gateName: String { get }
    /// Dispatches so far.
    var calls: Int { get }
    /// The first preflight failure, nil while every instantiation compiled.
    var failure: String? { get }
}

/// What a runner reports per resolved kernel in its load reply and status (`kernels: [...]`).
public struct KernelState: Sendable, Equatable {
    public var id: String
    public var replaces: String
    public var mode: KernelSpec.Mode
    public var active: Bool
    /// One sentence; why the stock path serves this site when not active, or the passed check when active.
    public var reason: String
    public init(id: String, replaces: String, mode: KernelSpec.Mode, active: Bool, reason: String) {
        self.id = id; self.replaces = replaces; self.mode = mode; self.active = active; self.reason = reason
    }
    public var wire: [String: Any] {
        ["id": id, "replaces": replaces, "mode": mode.rawValue, "state": active ? "active" : "stock", "reason": reason]
    }
}
