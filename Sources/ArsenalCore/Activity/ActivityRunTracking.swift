// SPDX-License-Identifier: MIT
import Foundation

/// One clock across queue/load/runner admission. Worker TTFT begins at its own acceptance; add the service prefix once.
public final class ActivityRunTracking: @unchecked Sendable {
    private let lock = NSLock()
    private let accepted: Double
    private var loading: Double?
    private var loadSeconds: Double?
    private var scheduled: Double?
    public let id: UUID
    public init(id: UUID) { self.id = id; accepted = ProcessInfo.processInfo.systemUptime }
    init(id: UUID, accepted: Double) { self.id = id; self.accepted = accepted }
    public func startLoading(at now: Double = ProcessInfo.processInfo.systemUptime) { lock.withLock { loading = now } }
    public func finishLoading(at now: Double = ProcessInfo.processInfo.systemUptime) { lock.withLock { if let loading { loadSeconds = now - loading } } }
    public func schedule(at now: Double = ProcessInfo.processInfo.systemUptime) { lock.withLock { scheduled = now } }
    public func decorate(_ result: JSONObject) -> JSONObject {
        lock.withLock {
            var result = result
            let legacyFirstToken = result.removeValue(forKey: "_mobius_first_token_at")?.number // legacy-compat: local runners written before the rename
            let firstTokenAt = result.removeValue(forKey: "_arsenal_first_token_at")?.number ?? legacyFirstToken
            var timing = result["timing"]?.object ?? [:]
            if let scheduled {
                let prefix = max(0, scheduled - accepted)
                timing["queue_s"] = .number(max(0, prefix - (loadSeconds ?? 0)) + (timing["queue_s"]?.number ?? 0))
                if let loadSeconds { timing["load_s"] = .number(loadSeconds) }
                if let firstTokenAt {
                    timing["ttft_s"] = .number(max(0, firstTokenAt - accepted))
                } else if let ttft = timing["ttft_s"]?.number {
                    timing["ttft_s"] = .number(prefix + ttft)
                }
            }
            if !timing.isEmpty { result["timing"] = .object(timing) }
            return result
        }
    }
}
extension ActivityRequestContext {
    @TaskLocal static var tracking: ActivityRunTracking?
}
