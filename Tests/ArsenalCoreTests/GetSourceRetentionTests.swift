// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

// Live fault, 20 Oct (Arsenal 3.5.3): after qwen3-reranker-8b's int8 was loaded, source retention removed its 16.4 GB
// fp16 copy and the row only said fp16=notInstalled. The following int4 Get re-downloaded the BF16 source shard by
// shard, converting each, but the row sat in "deriving" for 16+ minutes because every download count that arrived
// after the first conversion step was dropped.

private func marker() -> DownloadProgress {
    DownloadProgress(file: "", completedBytes: 0, totalBytes: 0, bytesPerSecond: 0, contentID: GetProgressDelivery.derivingContentID, phase: "deriving")
}
private func bytes(_ id: String, _ count: Int64, of total: Int64 = 100) -> DownloadProgress {
    DownloadProgress(file: id + ".safetensors", completedBytes: count, totalBytes: total, bytesPerSecond: 1, contentID: id)
}

@Test func nextShardDownloadAfterAConversionStepShowsDownloadingAgain() throws {
    var progress = try GetProgressAccumulator(files: [
        .init(contentID: "shard1", file: "shard1.safetensors", bytes: 100), .init(contentID: "shard2", file: "shard2.safetensors", bytes: 100)
    ])
    var states: [PrecisionState?] = []
    for event in [bytes("shard1", 50), bytes("shard1", 90), marker(), bytes("shard1", 100), bytes("shard2", 30), bytes("shard2", 60), bytes("shard2", 100), marker()] {
        states.append(progress.record(event))
    }
    #expect(states[0]?.state == .downloading && states[1]?.bytes == 90)
    #expect(states[2]?.state == .deriving)
    // A late callback for the shard being converted counts its bytes, but does not claim the row is downloading.
    #expect(states[3] == nil && progress.doneBytes >= 100)
    // The next shard's download is shown, with its bytes, even though a conversion step came before it.
    #expect(states[4]?.state == .downloading && states[4]?.bytes == 130 && states[4]?.totalBytes == 200)
    #expect(states[5]?.bytes == 160 && states[6]?.bytes == 200)
    #expect(states[7]?.state == .deriving)
}

@Test func progressDeliveryKeepsConversionMarkerBetweenTheShardsAroundIt() async throws {
    let batches = BatchRecorder()
    let (gate, open) = AsyncStream<Void>.makeStream()
    let delivery = GetProgressDelivery { batch, _ in
        await batches.append(batch); open.yield(())
    }
    delivery.submit(bytes("zz-first", 10))
    var iterator = gate.makeAsyncIterator(); _ = await iterator.next()
    // Everything below lands in one pending batch: order must follow submission, not key order.
    delivery.submit(bytes("zz-first", 100))
    delivery.submitDeriving()
    delivery.submit(bytes("aa-second", 5))
    await delivery.finish(); open.finish()
    let flattened = await batches.values.flatMap { $0 }.map(\.contentID)
    #expect(flattened == ["zz-first", "zz-first", GetProgressDelivery.derivingContentID, "aa-second"])
}

private actor BatchRecorder {
    var values: [[DownloadProgress]] = []
    func append(_ batch: [DownloadProgress]) { values.append(batch) }
}

@Test func retainedSourceIsVisibleAndAStalePlanFailsFastBeforeRestoring() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], threshold: 1); defer { f.stop() }
    let s = f.service, id = "fixture-chat"
    try await s.get(id: id, precision: "int4")
    // int8 is no longer offered, so this scenario's derived tier is int4. Reviewed while the 16-bit copy was installed and int4
    // was not: deriving int4 then needed no download.
    try await s.deletePrecision(id: id, precision: "int4")
    let stale = try await s.getPlan(id: id, dtype: "int4")
    #expect(stale.totalDownloadBytes == 0)
    try await s.get(id: id, precision: "int4")
    try await s.load(id: id, precision: "int4", fast: false)
    // Loading the derived tier keeps the 16-bit copy installed and ready.
    #expect(await s.rows().first?.states["bf16"]?.state == .ready)
    #expect(await s.rows().first?.retentionNote == nil)
    // Now the 16-bit copy goes missing for another reason (an older build's receipt): the visible state and the stale plan are checked from here.
    try await simulateOlderSourceRetention(f, derivedPrecision: "int4")

    let row = try #require(await s.rows().first)
    #expect(row.states["bf16"]?.state == .notInstalled)
    #expect(row.states["bf16"]?.phase == PrecisionState.sourceRemovedPhase)
    #expect(row.retentionNote == "16-bit copy removed while int4 was in use; Get of bf16 or another tier downloads it again (3 bytes)")

    // int4 leaves too, so the stale plan would derive it from a source that must now be restored.
    try await s.unload(id: id)
    try await s.deletePrecision(id: id, precision: "int4")
    // The source is gone: the old plan no longer covers the Get, so it is refused before any state change.
    do {
        try await s.get(id: id, precision: "int4", approvedPlan: stale)
        Issue.record("A Get whose source was removed ran under a plan that did not include restoring it")
    } catch let changed as GetPlanChanged {
        let weights = try #require(changed.revisedPlan.items.first { $0.id == "weights" })
        #expect(weights.downloadBytes == 3)
        #expect(weights.name.hasSuffix("restoring the 16-bit copy removed while int4 was in use"))
    }
    // Nothing of it is on this Mac now, so it is read from Get more models.
    #expect(await catalogRow(s, id)?.states["int4"]?.state == .notInstalled)
    #expect(await s.rows().isEmpty)

    let revised = try await s.getPlan(id: id, dtype: "int4")
    try await s.get(id: id, precision: "int4", approvedPlan: revised)
    let restored = try #require(await s.rows().first)
    #expect(restored.states["int4"]?.state == .ready)
    #expect(restored.states["bf16"]?.state == .ready && restored.states["bf16"]?.phase == nil)
    #expect(restored.retentionNote == nil)
    #expect(try await s.getPlan(id: id, dtype: "bf16").items.allSatisfy { !$0.name.contains("restoring") })
    await s.shutdown()
}

@Test func retentionNoteNamesTheRestoreDownloadSize() throws {
    let file = DerivedFile(path: "w.bin", bytes: 16_388_574_793, sha256: String(repeating: "a", count: 64))
    let receipt = SourceRetentionReceipt(
        id: "r", model: "m", revision: String(repeating: "b", count: 40), sourcePrecision: "fp16", derivedPrecision: "int8", path: "/tmp/x", bytes: file.bytes,
        reclaimedBytes: file.bytes, retainedSharedBytes: 0, sourceFiles: [file],
        sourceManifest: SnapshotManifest(model: "m", revision: String(repeating: "b", count: 40), precision: "fp16", files: [file], derived: nil), timestamp: "t")
    #expect(
        SourceRetentionPolicy.rowNote(receipt, downloadBytes: 16_388_574_793)
            == "16-bit copy removed while int8 was in use; Get of fp16 or another tier downloads it again (16.4 GB)")
    // Nothing to download (a local copy covers it): the receipt's own wording stands.
    #expect(SourceRetentionPolicy.rowNote(receipt, downloadBytes: 0) == receipt.note)
    #expect(SourceRetentionPolicy.size(42_000_000) == "42 MB")
}
