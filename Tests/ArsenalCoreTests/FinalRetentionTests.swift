// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private func receiptFixture() -> SourceRetentionReceipt {
    let file = DerivedFile(path: "weights.bin", bytes: 55_563_199_860, sha256: String(repeating: "a", count: 64))
    let source = SnapshotManifest(model: "tiny-test", revision: String(repeating: "b", count: 40), precision: "bf16", files: [file], derived: nil)
    return SourceRetentionReceipt(
        id: "fixture", model: source.model, revision: source.revision, sourcePrecision: "bf16", derivedPrecision: "int4", path: "/fixture", bytes: file.bytes,
        reclaimedBytes: file.bytes, retainedSharedBytes: 0, sourceFiles: [file], sourceManifest: source, timestamp: "fixture")
}
@Test func retentionNotesDistinguishLegacyLocalAndNetworkRecovery() throws {
    let legacy = receiptFixture()
    #expect(legacy.note == "16-bit copy removed; Get restores it.")
    var downloaded = legacy; downloaded.importedFromLocalCopy = false
    #expect(downloaded.note == "16-bit copy removed; Get re-downloads it (55.6 GB)")
    var local = legacy; local.importedFromLocalCopy = true; local.localCopyBytes = legacy.bytes; local.localSources = ["/Users/test/.omlx/models/Qwen/Qwen3.8-27B/weights.bin"]
    #expect(local.note == "16-bit copy removed; Get re-imports it from oMLX")
    #expect(!local.note.contains("55.6") && !local.note.contains("/Users"))
    local.localSources = ["/Users/test/.cache/huggingface/hub/models--test--good/snapshots/revision/weights.bin"]
    #expect(local.note == "16-bit copy removed; Get re-imports it from Hugging Face cache")
    local.localCopyBytes = 1 // mixed origins cannot promise a pure re-import
    #expect(local.note == "16-bit copy removed; Get restores it.")
    // Optional provenance permits historical receipts to remain unchanged on disk.
    let restored = try JSONDecoder().decode(SourceRetentionReceipt.self, from: JSONEncoder().encode(legacy))
    #expect(restored.note == legacy.note)
}
@Test func realLegacyRetentionReceiptIsReadOnlyAndNeverClaimsSavings() throws {
    let path = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
        "Library/Application Support/Arsenal-dev/final-support/weights/retention/qwen3.8-27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/bf16.json")
    guard FileManager.default.fileExists(atPath: path.path) else { return } // portable fixture above covers absent evidence
    let original = try Data(contentsOf: path)
    let receipt = try JSONDecoder().decode(SourceRetentionReceipt.self, from: original)
    #expect(receipt.reclaimedBytes == 55_563_199_860)
    #expect(receipt.importedFromLocalCopy == nil)
    #expect(receipt.note == "16-bit copy removed; Get restores it.")
    #expect(try Data(contentsOf: path) == original)
}
@Test func progressPublishesPhaseWithoutIncreasingBytesAndOnlyExplicitResetRegresses() throws {
    var progress = try GetProgressAccumulator(files: [.init(contentID: "a", file: "weights.bin", bytes: 3)])
    _ = progress.record(DownloadProgress(file: "weights.bin", completedBytes: 3, totalBytes: 3, bytesPerSecond: 0, contentID: "a", phase: "verifyingLocalCopy"))
    let phase = progress.record(DownloadProgress(file: "weights.bin", completedBytes: 3, totalBytes: 3, bytesPerSecond: 0, contentID: "a", phase: "downloading"))
    #expect(phase?.phase == "downloading" && phase?.bytes == 3)
    #expect(progress.record(DownloadProgress(file: "weights.bin", completedBytes: 1, totalBytes: 3, bytesPerSecond: 0, contentID: "a", phase: "downloading")) == nil)
    let reset = progress.record(
        DownloadProgress(file: "weights.bin", completedBytes: 0, totalBytes: 3, bytesPerSecond: 0, contentID: "a", phase: "downloading", resetsProgress: true))
    #expect(reset?.bytes == 0 && reset?.progress == 0)
}

@Test func localImportRetentionRowSaysReimportNotSavings() async throws {
    let donor = URL(fileURLWithPath: "/tmp/arsenal-final-b/donor-\(UUID().uuidString)/.omlx/models")
    let file = donor.appendingPathComponent("test/fixture/base.bin")
    try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data([12, 34, 56]).write(to: file)
    defer { try? FileManager.default.removeItem(at: donor.deletingLastPathComponent().deletingLastPathComponent()) }
    var configuration = ArsenalConfiguration(); configuration.get = ["localSources": [.string(donor.path)]]
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], threshold: 1, configuration: configuration); defer { f.stop() }
    f.stop() // a local copy must not silently fall back to HTTP
    do {
        try await f.service.get(id: "fixture-chat", precision: "int4")
        try await f.service.load(id: "fixture-chat", precision: "int4", fast: false)
        let loaded = try #require(await f.service.rows().first)
        #expect(loaded.states["bf16"]?.state == .ready && loaded.retentionNote == nil)
        // An older build's receipt for this local import: the row must say re-import, never savings.
        try await simulateOlderSourceRetention(f, derivedPrecision: "int4")
        let row = try #require(await f.service.rows().first)
        #expect(row.retentionNote == "16-bit copy removed; Get re-imports it from oMLX")
        #expect(row.retentionNote?.contains("save") == false)
        #expect(try Data(contentsOf: file) == Data([12, 34, 56]))
        await f.service.shutdown()
    } catch { await f.service.shutdown(); throw error }
}
