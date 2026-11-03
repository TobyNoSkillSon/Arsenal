// SPDX-License-Identifier: MIT
import Foundation

/// What a row shows while a Get waits for a person to accept a gated model's terms on Hugging Face.
public struct GateWait: Codable, Sendable, Equatable {
    public var repo: String
    /// The model's page (where the terms are accepted).
    public var page: String
    public var hasToken: Bool
    /// Hugging Face refused the token Arsenal found (expired or revoked): a new one is needed.
    public var tokenRejected: Bool?
    /// "auto" (access on the click) or "manual" (the owner approves each request); nil unknown.
    public var mode: String?
    /// Retries made so far in this wait (automatic or a click on Get).
    public var tries: Int
    public var limit: Int
    /// When the next automatic retry runs; nil once Arsenal stopped retrying.
    public var retryAt: Date?
    public var gaveUp: Bool { retryAt == nil }
    public init(
        repo: String, page: String, hasToken: Bool, tokenRejected: Bool? = nil, mode: String? = nil, tries: Int = 0, limit: Int = 10, retryAt: Date? = nil
    ) {
        self.repo = repo; self.page = page; self.hasToken = hasToken; self.tokenRejected = tokenRejected; self.mode = mode; self.tries = tries
        self.limit = limit; self.retryAt = retryAt
    }
}

/// The waiting and retrying of a gated Get, per model: one episode from the first refusal until success, another failure,
/// a cancel or the limit. The pages open once per episode (`began`); each refusal schedules the next retry after
/// `interval`; a Get started meanwhile (a click) cancels the pending retry and the counter. Its bookkeeping is synchronous
/// (a lock, no actor hop), so the service records a refusal without a suspension point in which another Get could start.
/// The clock is injected, so tests never sleep.
public final class GateRetry: @unchecked Sendable {
    public typealias Sleep = @Sendable (Duration) async throws -> Void
    public let interval: Duration
    public let limit: Int
    private let sleep: Sleep
    private let now: @Sendable () -> Date
    /// The wait an automatic retry belongs to, set while it runs: a Get checks it is still current before it registers.
    @TaskLocal public static var episode: UUID?
    private struct Episode {
        let id = UUID()
        var tries = 0
        var timer: Task<Void, Never>?
        var timerID: UUID?
        var retry: (@Sendable () async -> Void)?
    }
    private let lock = NSLock()
    private var episodes: [String: Episode] = [:]

    public init(
        interval: Duration = .seconds(30), limit: Int = 10, sleep: @escaping Sleep = { try await Task.sleep(for: $0) },
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.interval = interval; self.limit = max(1, limit); self.sleep = sleep; self.now = now
    }
    deinit { for episode in episodes.values { episode.timer?.cancel() } }

    /// A Get of `key` met the gate. Returns what the row shows and whether this wait just began (open the pages then).
    /// Until the limit, `retry` runs after the interval; at the limit the episode ends and the wait says it gave up.
    public func refused(_ key: String, gate: HFGateError, retry: @escaping @Sendable () async -> Void) -> (wait: GateWait, began: Bool) {
        lock.withLock {
            let began = episodes[key] == nil
            var episode = episodes[key] ?? Episode()
            episode.timer?.cancel(); episode.timer = nil; episode.timerID = nil
            if !began { episode.tries += 1 }
            var wait = GateWait(
                repo: gate.repo, page: gate.page.absoluteString, hasToken: gate.hasToken, tokenRejected: gate.tokenRejected ? true : nil, mode: gate.mode,
                tries: episode.tries, limit: limit)
            guard episode.tries < limit else {
                episodes[key] = nil
                return (wait, began)
            }
            let id = UUID(), sleep = sleep, interval = interval, owner = episode.id
            wait.retryAt = now().addingTimeInterval(Self.seconds(interval))
            episode.retry = retry; episode.timerID = id
            episode.timer = Task { [weak self] in
                do { try await sleep(interval) } catch { return }
                guard let self, self.fire(key, id) else { return }
                await Self.$episode.withValue(owner) { await retry() }
            }
            episodes[key] = episode
            return (wait, began)
        }
    }
    /// A Get of `key` starts now (a click or an agent): the pending retry and its counter go; the wait goes on.
    public func retryingNow(_ key: String) {
        lock.withLock { cancelTimer(key) }
    }
    /// The Get succeeded, failed otherwise or was cancelled: the wait is over.
    public func end(_ key: String) {
        lock.withLock {
            episodes[key]?.timer?.cancel(); episodes[key] = nil
        }
    }
    /// The retries waiting now, their counters removed: the caller runs them (a token was just saved).
    public func takePendingRetries() -> [String: @Sendable () async -> Void] {
        lock.withLock {
            var due: [String: @Sendable () async -> Void] = [:]
            for (key, episode) in episodes where episode.timerID != nil {
                if let retry = episode.retry { due[key] = retry }
                cancelTimer(key)
            }
            return due
        }
    }
    public func isWaiting(_ key: String) -> Bool { lock.withLock { episodes[key] != nil } }
    /// `episode` is still `key`'s wait (not cancelled, removed or replaced).
    public func isCurrent(_ key: String, episode: UUID) -> Bool { lock.withLock { episodes[key]?.id == episode } }
    /// A counter is running for `key`.
    public func hasPendingRetry(_ key: String) -> Bool { lock.withLock { episodes[key]?.timerID != nil } }

    private func cancelTimer(_ key: String) {
        episodes[key]?.timer?.cancel(); episodes[key]?.timer = nil; episodes[key]?.timerID = nil
    }
    private func fire(_ key: String, _ id: UUID) -> Bool {
        lock.withLock {
            guard episodes[key]?.timerID == id else { return false }
            episodes[key]?.timer = nil; episodes[key]?.timerID = nil
            return true
        }
    }
    static func seconds(_ d: Duration) -> TimeInterval {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }
}
