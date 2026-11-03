// SPDX-License-Identifier: MIT
import Darwin
import Foundation

/// Owns its descriptor until Dispatch completes cancellation. Idle workers park continuations, not threads.
final class AsyncPipeReader: @unchecked Sendable {
    private static let queue = DispatchQueue(label: "dev.arsenal.pipe-read", qos: .utility)
    private let descriptor: Int32
    private let source: DispatchSourceRead
    // Accessed only on queue; one consumer per pipe preserves framing and bounded backpressure.
    private var pending: (count: Int, reply: CheckedContinuation<Data?, any Error>)?
    private var armed = false
    private var finished = false

    init(descriptor: Int32) {
        self.descriptor = descriptor
        _ = fcntl(descriptor, F_SETFL, fcntl(descriptor, F_GETFL) | O_NONBLOCK)
        source = DispatchSource.makeReadSource(fileDescriptor: descriptor, queue: Self.queue)
        source.setEventHandler { [weak self] in self?.consume() }
        // Cancellation is asynchronous: callers must never close/reuse the monitored FD themselves.
        source.setCancelHandler { close(descriptor) }
    }
    func read(count: Int) async throws -> Data? {
        try await withCheckedThrowingContinuation { reply in
            Self.queue.async {
                guard !self.finished else { reply.resume(returning: nil); return }
                guard self.pending == nil, count > 0 else { reply.resume(throwing: WorkerFailure(code: "read", message: "Worker pipe already has a reader.")); return }
                self.pending = (count, reply)
                self.consume()
            }
        }
    }
    private func consume() {
        guard !finished, let pending else { return }
        var data = Data(count: min(pending.count, 65_536))
        var count: Int
        repeat { count = data.withUnsafeMutableBytes { Darwin.read(descriptor, $0.baseAddress!, $0.count) } } while count < 0 && errno == EINTR
        if count < 0 && errno == EAGAIN {
            if !armed { armed = true; source.resume() }
            return
        }
        self.pending = nil
        if armed { armed = false; source.suspend() }
        if count < 0 {
            finishOnQueue()
            pending.reply.resume(throwing: WorkerFailure(code: "read", message: "Worker pipe failed."))
        } else if count == 0 {
            finishOnQueue()
            pending.reply.resume(returning: nil)
        } else {
            data.count = count; pending.reply.resume(returning: data)
        }
    }
    func finish() { Self.queue.async { self.finishOnQueue() } }
    private func finishOnQueue() {
        guard !finished else { return }
        finished = true
        source.cancel()
        // A suspended Dispatch source must be resumed before disposal.
        if !armed { armed = true; source.resume() }
        let reply = pending?.reply; pending = nil; reply?.resume(returning: nil)
    }
    deinit { source.cancel(); if !armed { source.resume() } }
}
