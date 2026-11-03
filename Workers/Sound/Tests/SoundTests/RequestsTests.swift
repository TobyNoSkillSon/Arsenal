// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalWire
import WorkerTransport
@testable import SoundWorker

@Test func requestAdmissionAndCancellationStayBounded() throws {
    let requests = Requests()
    let active = try #require(requests.admit(7, progress: { _ in }))
    #expect(requests.admit(8, progress: { _ in }) == nil)
    #expect(!requests.cancel(8))
    #expect(!active.isCancelled)
    #expect(requests.cancel(7))
    #expect(active.isCancelled)
    requests.release()
    let next = try #require(requests.admit(8, progress: { _ in }))
    #expect(!next.isCancelled)
    requests.stop()
    #expect(next.isCancelled)
}

private final class SoundStepProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [(Int, Int)] = []
    func add(_ step: Int, _ total: Int) { lock.withLock { values.append((step, total)) } }
    var last: (Int, Int)? { lock.withLock { values.last } }
}
@Test func stepCountsReachTheWorkerCallback() throws {
    let requests = Requests(), probe = SoundStepProbe()
    let active = try #require(requests.admit(7, progress: { _ in }, step: { _, step, total in probe.add(step, total) }))
    active.report(step: 3, total: 8)
    let value = try #require(probe.last)
    #expect(value.0 == 3 && value.1 == 8)
    active.report(step: 10, total: 8)
    #expect(probe.last?.0 == 3)
    requests.release()
}

@Test func samplerFractionAndCountsShareOneWireFrame() throws {
    let pipe = Pipe(), channel = WorkerChannel(output: pipe.fileHandleForWriting), requests = Requests()
    let active = try #require(
        requests.admit(
            7,
            progress: { try? channel.send(SoundWorker.progressFrame(7, fraction: $0)) },
            step: { try? channel.send(SoundWorker.progressFrame(7, fraction: $0, step: $1, total: $2)) }))
    active.report(progress: 0.375, step: 3, total: 8)
    try pipe.fileHandleForWriting.close()
    let received = try FrameIO.read(from: pipe.fileHandleForReading)
    let frame = try #require(received)
    guard case .json(let value) = frame else { Issue.record("Expected JSON progress"); return }
    #expect(value["fraction"] == 0.375 && value["step"] == 3 && value["total"] == 8)
    #expect(try FrameIO.read(from: pipe.fileHandleForReading) == nil)
}
