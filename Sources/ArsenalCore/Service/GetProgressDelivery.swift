// SPDX-License-Identifier: MIT
import Foundation

/// One demand-driven task per Get, not per downloaded chunk. Keeps the latest count for every content ID.
/// There is no timer at idle; the final pending batch is drained before Get publishes its final state.
/// Batches keep submission order (by each key's latest update), so a conversion marker is applied after the
/// download events that preceded it and before those of the next file.
final class GetProgressDelivery: @unchecked Sendable {
    /// Content ID of the marker `submitDeriving()` sends: a conversion step has started.
    static let derivingContentID = "\u{1}deriving"
    private let lock = NSLock()
    private var pending: [String: DownloadProgress] = [:]
    private var order: [String] = []
    private var task: Task<Void, Never>?
    private var accepting = true
    private var nextDelivery = ContinuousClock.now
    private let interval: Duration
    private let sink: @Sendable ([DownloadProgress], Bool) async -> Void
    init(interval: Duration = .milliseconds(100), sink: @escaping @Sendable ([DownloadProgress], Bool) async -> Void) {
        self.interval = interval; self.sink = sink
    }
    func submit(_ event: DownloadProgress) {
        lock.withLock {
            guard accepting else { return }
            let key = event.contentID ?? event.file
            var value = event
            if let previous = pending[key], value.resetsProgress != true {
                value.completedBytes = max(value.completedBytes, previous.completedBytes)
            }
            // A later count must not erase the reset of a rejected local donor in the same batch.
            if pending[key]?.resetsProgress == true { value.resetsProgress = true }
            pending[key] = value
            order.removeAll { $0 == key }; order.append(key)
            if task == nil { task = Task { await self.deliver() } }
        }
    }
    /// Converting has started; delivered in order with the download counts around it.
    func submitDeriving() {
        submit(DownloadProgress(file: "", completedBytes: 0, totalBytes: 0, bytesPerSecond: 0, contentID: Self.derivingContentID, phase: "deriving"))
    }
    func finish(discard: Bool = false) async {
        let (running, final) = lock.withLock {
            accepting = false
            let final = discard ? [] : order.compactMap { pending[$0] }
            pending.removeAll(); order.removeAll(); task?.cancel()
            return (task, final)
        }
        await running?.value
        // Final row publication is unconditional; don't delay completion to emit another progress tick.
        if !final.isEmpty { await sink(final, true) }
    }
    private func deliver() async {
        while true {
            let deadline = lock.withLock { nextDelivery }
            do { try await ContinuousClock().sleep(until: deadline) } catch {
                lock.withLock {
                    pending.removeAll(); order.removeAll(); task = nil
                }; return
            }
            let batch = lock.withLock { () -> [DownloadProgress]? in
                guard !pending.isEmpty else { task = nil; return nil }
                let values = order.compactMap { pending[$0] }
                pending.removeAll(); order.removeAll()
                return values
            }
            guard let batch else { return }
            await sink(batch, false)
            let done = lock.withLock {
                nextDelivery = ContinuousClock.now + interval
                if pending.isEmpty { task = nil; return true }
                return false
            }
            if done { return }
        }
    }
}
