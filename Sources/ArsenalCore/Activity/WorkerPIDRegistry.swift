// SPDX-License-Identifier: MIT
import Foundation

/// Every WorkerProcess registers, including precision converters outside a resident Supervisor.
/// Tokens prevent a late exit callback from removing a subsequently reused process ID.
enum WorkerPIDRegistry {
    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var workers: [UUID: Int32] = [:]
    }
    private static let state = State()
    static func register(_ pid: Int32, token: UUID) { state.lock.withLock { state.workers[token] = pid } }
    static func remove(_ token: UUID) { state.lock.withLock { state.workers[token] = nil } }
    static func snapshot() -> Set<Int32> { state.lock.withLock { Set(state.workers.values) } }
}
