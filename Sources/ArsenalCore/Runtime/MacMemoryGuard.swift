// SPDX-License-Identifier: MIT
import Foundation
import os
import ArsenalWire

/// The Mac-wide memory line: the whole Mac stays at or under 100/128 of its RAM in use. Every worker with a request running joins it; each worker's half-second
/// sampler reports its footprint and re-reads the Mac's memory in use (`MemoryProbe`'s VM counters: RAM that cannot
/// be reclaimed without swap). Past the line, the guard stops one worker: among those that grew by more than 512 MB
/// during their current requests and are not Load Anyway residents, the one furthest past its expected peak plus
/// 1 GB (without a known peak, past where its requests started plus 1 GB). One stop at a time: the next decision waits
/// until that worker has exited and its memory is back, then reads the Mac again. A worker that did not grow is never
/// blamed for memory other apps or earlier Loads took.
public final class MacMemoryGuard: Sendable {
    public static let shared = MacMemoryGuard()
    public static let safeInUseFraction = 100.0 / 128.0
    /// The Mac-wide line in MB for a Mac with `totalMB` of RAM. Load admission and the guard both use this one figure.
    public static func safeInUseMB(totalMB: Double) -> Double { totalMB * safeInUseFraction }
    /// Room under the Mac-wide line now, in MB: the RAM usable without swap, less the RAM already above the line. Load
    /// admission and the menu's memory line both show this figure; it may be negative when the Mac is already above it.
    public static func roomUnderLineMB(totalMB: Double, noSwapAvailableMB: Double) -> Double {
        noSwapAvailableMB - (totalMB - safeInUseMB(totalMB: totalMB))
    }
    /// This Mac's line and RAM in GB, for copy and wire details when the caller has no probe at hand.
    public static var macLimitGB: Double { safeInUseMB(totalMB: MemoryProbe(environment: [:]).totalMB) / 1000 }
    public static var macTotalGB: Double { MemoryProbe(environment: [:]).totalMB / 1000 }
    /// Growth during a request below this is noise, not a cause.
    public static let minimumGrowthBytes: UInt64 = 512_000_000
    /// Slack above the expected peak before a worker counts as past it.
    public static let expectedSlackBytes: UInt64 = 1_000_000_000
    /// A stop that has not reported its worker's exit by then no longer blocks the next decision.
    static let stopTimeout: TimeInterval = 15
    public let limitBytes: UInt64
    private let inUse: @Sendable () -> UInt64
    private let state = OSAllocatedUnfairLock(initialState: State())

    struct Member: Sendable {
        var ceiling: WorkerMemoryCeiling
        var startBytes: UInt64
        var footprintBytes: UInt64
        var stop: @Sendable (WorkerFailure) -> Void
    }
    struct State: Sendable {
        var members: [UUID: Member] = [:]
        var stopping: (token: UUID, until: Date)?
    }
    public enum Decision: Sendable, Equatable { case none, stop(UUID, WorkerFailure) }

    /// `limitBytes` and `inUseBytes` replace the Mac's figures in tests (a simulated Mac); the default reads this Mac.
    public init(limitBytes: UInt64? = nil, inUseBytes: (@Sendable () -> UInt64)? = nil) {
        let live = MemoryProbe(environment: [:])
        self.limitBytes = limitBytes ?? UInt64(Self.safeInUseMB(totalMB: live.totalMB) * 1e6)
        self.inUse = inUseBytes ?? { UInt64(max(0, live.totalMB - live.noSwapAvailableMB(loadedMB: 0)) * 1e6) }
    }
    public var inUseBytes: UInt64 { inUse() }

    /// A worker's sampler starts: `startBytes` is its footprint when its first current request began.
    func join(_ token: UUID, ceiling: WorkerMemoryCeiling, startBytes: UInt64, stop: @escaping @Sendable (WorkerFailure) -> Void) {
        state.withLock { $0.members[token] = Member(ceiling: ceiling, startBytes: startBytes, footprintBytes: startBytes, stop: stop) }
    }
    /// The worker has no request running any more, or it has exited after a stop.
    func leave(_ token: UUID) {
        state.withLock { state in
            state.members[token] = nil
            if state.stopping?.token == token { state.stopping = nil }
        }
    }
    /// One sampler tick. Returns the failure when this worker is the one to stop; another worker chosen as the victim
    /// is told through its `stop` callback.
    func check(_ token: UUID, ceiling: WorkerMemoryCeiling, footprintBytes: UInt64) -> WorkerFailure? {
        let inUse = inUse()
        let (decision, callback) = state.withLock { state -> (Decision, (@Sendable (WorkerFailure) -> Void)?) in
            state.members[token]?.ceiling = ceiling; state.members[token]?.footprintBytes = footprintBytes
            let decision = Self.decide(&state, inUseBytes: inUse, limitBytes: limitBytes, now: Date())
            guard case .stop(let victim, _) = decision, victim != token else { return (decision, nil) }
            return (decision, state.members[victim]?.stop)
        }
        guard case .stop(let victim, let failure) = decision else { return nil }
        if victim == token { return failure }
        callback?(failure)
        return nil
    }
    /// The decision, on recorded footprints. Marks the chosen worker as stopping.
    static func decide(_ state: inout State, inUseBytes: UInt64, limitBytes: UInt64, now: Date) -> Decision {
        if let stopping = state.stopping, stopping.until > now, state.members[stopping.token] != nil { return .none }
        state.stopping = nil
        guard inUseBytes > limitBytes else { return .none }
        func excess(_ member: Member) -> Double {
            let expected = member.ceiling.expectedMB.map { UInt64($0 * 1e6) } ?? member.startBytes
            return Double(member.footprintBytes) - Double(expected) - Double(expectedSlackBytes)
        }
        let candidates = state.members.filter { _, member in
            !member.ceiling.forced && member.footprintBytes >= member.startBytes + minimumGrowthBytes
        }
        guard let (token, member) = candidates.max(by: { excess($0.value) < excess($1.value) || excess($0.value) == excess($1.value) && $0.key.uuidString < $1.key.uuidString })
        else { return .none }
        state.stopping = (token, now.addingTimeInterval(stopTimeout))
        return .stop(
            token, member.ceiling.macFailure(footprintBytes: member.footprintBytes, startBytes: member.startBytes, inUseBytes: inUseBytes, limitBytes: limitBytes))
    }
}
