// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private actor ProgressRecorder {
    var batches: [[DownloadProgress]] = []
    var times: [ContinuousClock.Instant] = []
    func append(_ batch: [DownloadProgress], final: Bool) { batches.append(batch); if !final { times.append(.now) } }
}

@Test func progressDeliveryCoalescesBurstAndKeepsAllContentFinalCounts() async {
    let recorder = ProgressRecorder()
    let delivery = GetProgressDelivery { await recorder.append($0, final: $1) }
    for count in 1...10_000 {
        for id in ["a", "b"] { delivery.submit(.init(file: "same.bin", completedBytes: Int64(count), totalBytes: 10_000, bytesPerSecond: 1, contentID: id)) }
    }
    // A reordered callback cannot erase an unseen higher count in the pending batch.
    delivery.submit(.init(file: "same.bin", completedBytes: 1, totalBytes: 10_000, bytesPerSecond: 1, contentID: "a"))
    await delivery.finish()
    let batches = await recorder.batches
    #expect(batches.count < 20_000)
    var counts: [String: Int64] = [:]
    for batch in batches { for value in batch { counts[value.contentID!] = max(counts[value.contentID!] ?? 0, value.completedBytes) } }
    #expect(counts == ["a": 10_000, "b": 10_000])
    delivery.submit(.init(file: "late", completedBytes: 1, totalBytes: 1, bytesPerSecond: 1))
    #expect(await recorder.batches.count == batches.count)
}

@Test func progressDeliveryRateIsBoundedAndFinalPendingBatchIsDrained() async throws {
    let recorder = ProgressRecorder()
    let delivery = GetProgressDelivery { await recorder.append($0, final: $1) }
    for count in 1...12 {
        delivery.submit(.init(file: "a", completedBytes: Int64(count), totalBytes: 12, bytesPerSecond: 1))
        try await Task.sleep(for: .milliseconds(30))
    }
    await delivery.finish()
    let times = await recorder.times
    for pair in zip(times, times.dropFirst()) { #expect(pair.1 - pair.0 >= .milliseconds(95)) }
    #expect(await recorder.batches.last?.last?.completedBytes == 12)
}

@Test func progressDeliveryRetainsRejectedDonorResetInSameBatch() async throws {
    let recorder = ProgressRecorder()
    let (stream, continuation) = AsyncStream<Void>.makeStream()
    let delivery = GetProgressDelivery { batch, final in
        await recorder.append(batch, final: final); continuation.yield(())
    }
    delivery.submit(.init(file: "a", completedBytes: 100, totalBytes: 100, bytesPerSecond: 1, contentID: "a"))
    var iterator = stream.makeAsyncIterator(); _ = await iterator.next()
    delivery.submit(.init(file: "a", completedBytes: 0, totalBytes: 100, bytesPerSecond: 1, contentID: "a", resetsProgress: true))
    delivery.submit(.init(file: "a", completedBytes: 3, totalBytes: 100, bytesPerSecond: 1, contentID: "a"))
    await delivery.finish(); continuation.finish()
    let last = try #require(await recorder.batches.last?.last)
    #expect(last.resetsProgress == true && last.completedBytes == 3)
    var accumulator = try GetProgressAccumulator(files: [.init(contentID: "a", file: "a", bytes: 100)])
    _ = accumulator.record(contentID: "a", file: "a", completedBytes: 100)
    _ = accumulator.record(last)
    #expect(accumulator.doneBytes == 3)
}

@Test func progressDeliveryDiscardDrainsTaskAndRejectsLateCallbacks() async {
    let recorder = ProgressRecorder()
    let delivery = GetProgressDelivery { await recorder.append($0, final: $1) }
    delivery.submit(.init(file: "a", completedBytes: 1, totalBytes: 10, bytesPerSecond: 1))
    await delivery.finish(discard: true)
    let count = await recorder.batches.count
    delivery.submit(.init(file: "a", completedBytes: 10, totalBytes: 10, bytesPerSecond: 1))
    #expect(await recorder.batches.count == count)
}
