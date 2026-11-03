// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func aggregateGetProgressKeepsContentIdentityAndMonotonicCounts() throws {
    let start = Date(timeIntervalSince1970: 0)
    var progress = try GetProgressAccumulator(
        files: [.init(contentID: "a", file: "config.json", bytes: 3), .init(contentID: "b", file: "config.json", bytes: 5), .init(contentID: "a", file: "shared.json", bytes: 3)],
        started: start)
    #expect(progress.totalBytes == 8)
    progress.record(contentID: "a", file: "config.json", completedBytes: 3)
    progress.record(contentID: "b", file: "config.json", completedBytes: 2)
    #expect(progress.doneBytes == 5)
    progress.record(contentID: "a", file: "config.json", completedBytes: 1) // reordered callback cannot go backward
    progress.record(contentID: "already-installed", file: "config.json", completedBytes: 999)
    #expect(progress.doneBytes == 5)
    progress.record(contentID: "b", file: "config.json", completedBytes: 5)
    #expect(progress.doneBytes == 8 && progress.fraction == 1)
    #expect(progress.rate(now: start.addingTimeInterval(2)) == 4)
}

@Test func localVerificationProgressKeepsCancelStateAndPhase() throws {
    var progress = try GetProgressAccumulator(files: [.init(contentID: "fixture", file: "weights.bin", bytes: 3)])
    let recorded = progress.record(DownloadProgress(file: "weights.bin", completedBytes: 1, totalBytes: 3, bytesPerSecond: 1, contentID: "fixture", phase: "verifyingLocalCopy"))
    let state = try #require(recorded)
    #expect(state.state == .downloading && state.phase == "verifyingLocalCopy")
    #expect(state.bytes == 1 && state.totalBytes == 3 && state.bytesPerSecond != nil)
}
