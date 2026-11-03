// SPDX-License-Identifier: MIT
import Foundation
public struct ArsenalPaths: Sendable {
    public var support: URL; public var cache: URL
    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        support = environment["ARSENAL_SUPPORT_DIR"].map { URL(fileURLWithPath: $0) } ?? home.appendingPathComponent("Library/Application Support/Arsenal")
        cache = environment["ARSENAL_CACHE_DIR"].map { URL(fileURLWithPath: $0) } ?? home.appendingPathComponent("Library/Caches/Arsenal")
    }
    public init(support: URL, cache: URL? = nil) { self.support = support; self.cache = cache ?? support.appendingPathComponent("cache") }
    public func directory(_ name: String) -> URL { support.appendingPathComponent(name) }
    public func create() throws {
        for name in ["catalog", "models", "weights", "state", "jobs", "logs"] {
            try FileManager.default.createDirectory(at: directory(name), withIntermediateDirectories: true)
        }
    }
}
public enum RunnerEnvironment {
    private static let common: Set<String> = ["HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "ARSENAL_SUPPORT_DIR", "ARSENAL_CACHE_DIR"]
    private static let worker: Set<String> = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "ARSENAL_MLX_METALLIB"]
    /// Exact keys only: no credential-shaped prefixes or ambient Python/uv configuration.
    public static func scrub(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> [String: String] {
        filtered(environment, allowed: common.union(worker))
    }
    private static func filtered(_ environment: [String: String], allowed: Set<String>) -> [String: String] {
        var result = environment.filter { allowed.contains($0.key) }
        // Absolute worker/interpreter paths need no parent shell's executable search path.
        result["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        return result
    }
}
