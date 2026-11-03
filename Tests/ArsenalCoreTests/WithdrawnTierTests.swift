// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Snapshots of a precision the model's entry no longer offers (the int8/int4 tiers withdrawn from TIPSv2, Harrier,
/// Jina, RIFE and SAM 3.1, or Smart Turn's int4 on the development Mac) are reported on the row with their size and removed only
/// on request, through the store's logged deletion; the offered tiers stay.
@Test func withdrawnTiersAreReportedAndRemovedOnlyOnRequest() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let service = f.service
    let row = try #require(await catalogRow(service, "fixture-chat"))
    let withdrawn = try #require(["fp16", "int2", "q3"].first { !row.precisions.contains($0) })
    let offered = try #require(row.precisions.first)
    let snapshots = f.paths.directory("weights").appendingPathComponent("snapshots/\(row.id)/\(String(repeating: "a", count: 40))")
    for (precision, bytes) in [(withdrawn, 4096), (offered, 1024)] {
        let directory = snapshots.appendingPathComponent(precision)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(count: bytes).write(to: directory.appendingPathComponent("model.safetensors"))
        let manifest = SnapshotManifest(model: row.id, revision: snapshots.lastPathComponent, precision: precision, files: [], derived: nil)
        try JSONEncoder().encode(manifest).write(to: directory.appendingPathComponent("arsenal-snapshot.json"))
    }
    await service.refreshInstalled(forceDownloadSizes: true)
    // Weights of a withdrawn tier on this Mac put the model on the table, where the user can delete them.
    let reported = try #require(await service.rows().first { $0.id == row.id })
    #expect(Array(reported.withdrawnTiers.keys) == [withdrawn] && (reported.withdrawnTiers[withdrawn] ?? 0) > 4096, "\(reported.withdrawnTiers)")
    // Nothing removes them on its own: another refresh keeps them.
    await service.refreshInstalled(forceDownloadSizes: true)
    #expect(FileManager.default.fileExists(atPath: snapshots.appendingPathComponent(withdrawn).path))
    // The row survives encoding for the app and API.
    var plain = ModelRow(id: row.id, name: row.name, kind: row.kind)
    plain.withdrawnTiers = reported.withdrawnTiers
    #expect(try JSONDecoder().decode(ModelRow.self, from: JSONEncoder().encode(plain)).withdrawnTiers == reported.withdrawnTiers)

    try await service.deleteWithdrawnTiers(id: row.id)
    #expect(!FileManager.default.fileExists(atPath: snapshots.appendingPathComponent(withdrawn).path))
    #expect(FileManager.default.fileExists(atPath: snapshots.appendingPathComponent(offered).appendingPathComponent("model.safetensors").path))
    #expect(try #require(await catalogRow(service, row.id)).withdrawnTiers.isEmpty)
    let log = try String(contentsOf: f.paths.directory("logs").appendingPathComponent("store-deletions.jsonl"), encoding: .utf8)
    #expect(log.contains("delete_withdrawn_precision") && log.contains("/\(withdrawn)"), "\(log)")
    await service.shutdown()
}
