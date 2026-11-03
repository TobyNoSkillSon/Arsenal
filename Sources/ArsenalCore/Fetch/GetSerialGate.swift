// SPDX-License-Identifier: MIT
import Foundation

/// Serializes Get transactions, not inference. This keeps shared .part/blob activation and delete accounting coherent.
actor GetSerialGate {
    private var held = false
    private var queue: [UUID] = []
    private var waiters: [UUID: CheckedContinuation<Void, Error>] = [:]
    private var cancelled: Set<UUID> = []
    func acquire(_ ticket: UUID) async throws {
        try Task.checkCancellation()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if cancelled.remove(ticket) != nil || Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                } else if !held {
                    held = true; continuation.resume()
                } else {
                    queue.append(ticket); waiters[ticket] = continuation
                }
            }
        } onCancel: {
            Task { await self.cancel(ticket) }
        }
    }
    private func cancel(_ ticket: UUID) {
        if let waiter = waiters.removeValue(forKey: ticket) { queue.removeAll { $0 == ticket }; waiter.resume(throwing: CancellationError()) } else { cancelled.insert(ticket) }
    }
    func release(_ ticket: UUID) {
        cancelled.remove(ticket)
        while !queue.isEmpty { let next = queue.removeFirst(); if let waiter = waiters.removeValue(forKey: next) { waiter.resume(); return } }
        held = false
    }
}
