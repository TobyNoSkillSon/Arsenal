// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func installedRowCacheTracksExternalSnapshotRemovalAndNativeConfigEdits() async throws {
    let config = Data(#"{"max_position_embeddings":1024}"#.utf8)
    let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], filePayloads: ["base.bin": Data([1, 2, 3]), "config.json": config])
    defer { fixture.stop() }
    let service = fixture.service
    try await service.get(id: "fixture-chat", precision: "bf16")
    #expect(await service.catalog().first?.row.metrics["native_context"] == 1024)
    let snapshot = fixture.paths.directory("weights").appendingPathComponent("snapshots/fixture-chat/\(fixture.revision)/bf16")
    let native = snapshot.appendingPathComponent("config.json")
    try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: native.path)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: snapshot.path)
    let modified = try FileManager.default.attributesOfItem(atPath: native.path)[.modificationDate]
    try Data(#"{"max_position_embeddings":2048}"#.utf8).write(to: native)
    if let modified { try FileManager.default.setAttributes([.modificationDate: modified], ofItemAtPath: native.path) }
    // The edited file no longer verifies, so the snapshot is not ready weights: the row is read from Get more models.
    #expect(await catalogRow(service, "fixture-chat")?.metrics["native_context"] == 2048)
    await #expect(throws: (any Error).self) { try await service.load(id: "fixture-chat") }
    try config.write(to: native)
    let metadata = snapshot.appendingPathComponent("arsenal-snapshot.json")
    let held = fixture.paths.support.appendingPathComponent("held-snapshot.json")
    try FileManager.default.moveItem(at: metadata, to: held)
    #expect(await catalogRow(service, "fixture-chat")?.states["bf16"]?.state == .notInstalled)
    #expect(await service.rows().isEmpty)
    #expect(await service.catalog().first?.installed == false)
    try FileManager.default.moveItem(at: held, to: metadata)
    #expect(await service.catalog().first?.installed == true)
    try await service.delete(id: "fixture-chat")
    #expect(await service.catalog().first?.installed == false)
    await service.shutdown()
}

@Test func localDiscoveryCacheNoticesEmptySubtreeAdditionAndRestoredMtimeEdit() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-pol-core2/local-" + UUID().uuidString)
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let folder = root.appendingPathComponent("local-models/qwen3.8-27b")
    try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: repo.appendingPathComponent("models/qwen3.8-27b"), to: folder)
    let empty = folder.appendingPathComponent("empty")
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    let scanner = LocalFolderScanner()
    let first = try await scanner.scan(support: root, signedIDs: [], kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    let file = empty.appendingPathComponent("new.txt")
    try Data("aaaa".utf8).write(to: file)
    let added = try await scanner.scan(support: root, signedIDs: [], kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    #expect(first.folders.first?.contentDigest != added.folders.first?.contentDigest)
    let modified = try FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate]
    try Data("bbbb".utf8).write(to: file)
    if let modified { try FileManager.default.setAttributes([.modificationDate: modified], ofItemAtPath: file.path) }
    let edited = try await scanner.scan(support: root, signedIDs: [], kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    #expect(added.folders.first?.contentDigest != edited.folders.first?.contentDigest)
}

@Test func bundleBatchEvidenceKeepsInventoryAndIndividualHashChecks() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["voices"])
    defer { fixture.stop() }
    let resources = fixture.paths.support.appendingPathComponent("bundle-fixture")
    for (path, data) in fixture.source.files {
        let target = resources.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: target)
    }
    let catalog = try JSONDecoder().decode(CatalogDocument.self, from: fixture.catalogData)
    let source = try BundleCatalogSource(resources: resources, catalog: catalog)
    let paths = fixture.source.files.keys.sorted()
    #expect(await source.readAvailable(paths: paths, commit: catalog.sourceCommit) == fixture.source.files)
    let path = try #require(paths.first { $0.hasSuffix("README.md") })
    try Data("changed".utf8).write(to: resources.appendingPathComponent(path))
    let partial = await source.readAvailable(paths: paths, commit: catalog.sourceCommit)
    #expect(partial[path] == nil && partial.count == paths.count - 1)
    let extra = resources.appendingPathComponent("models/fixture-voices/unlisted.txt")
    try Data([1]).write(to: extra)
    #expect(await source.readAvailable(paths: paths, commit: catalog.sourceCommit).isEmpty)
    await fixture.service.shutdown()
}
