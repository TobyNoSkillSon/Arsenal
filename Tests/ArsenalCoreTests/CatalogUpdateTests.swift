// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import Testing
@testable import ArsenalCore

private func bumpedCatalog(_ f: LocalCatalogFixture, weightsChange: Bool = false, recipeChange: Bool = false) throws -> (Data, Data, Data, FixtureCatalogSource) {
    var catalog = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData)
    catalog.sequence += 1
    var files = f.source.files
    let path = "models/fixture-chat/model.toml"
    var text = String(decoding: files[path]!, as: UTF8.self)
    // Semantically identical, but exact signed bytes change: the candidate's real failure.
    let lines = text.components(separatedBy: "\n")
    text = ([lines[1], lines[0]] + lines.dropFirst(2)).joined(separator: "\n")
    if weightsChange {
        let revision = String(repeating: "b", count: 40)
        text = text.replacingOccurrences(of: f.revision, with: revision)
        catalog.models[0].manifest.weights.revision = revision
    }
    if recipeChange {
        text = text.replacingOccurrences(of: "opaque-fixture-recipe", with: "opaque-updated-recipe")
        catalog.models[0].manifest.derive?["int4"] = "opaque-updated-recipe"
        catalog.models[0].manifest.derive?["int8"] = "opaque-updated-recipe"
    }
    files[path] = Data(text.utf8)
    let index = catalog.models[0].files.firstIndex { $0.path == "model.toml" }!
    catalog.models[0].files[index] = CatalogFile(path: "model.toml", sha256: FileDigest.sha256(files[path]!), bytes: Int64(files[path]!.count))
    let data = try catalog.encoded(), key = Curve25519.Signing.PrivateKey()
    return (data, try key.signature(for: data), key.publicKey.rawRepresentation, FixtureCatalogSource(files: files))
}

@Test(arguments: [false, true]) func folderOnlyCatalogAdvanceLoadsWithoutTouchingWeights(legacy: Bool) async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "int4")
    await f.service.shutdown()
    if legacy {
        let record = f.paths.directory("state").appendingPathComponent("installed-catalog/fixture-chat.json")
        try FileManager.default.moveItem(at: record, to: record.appendingPathExtension("legacy"))
    }
    let store = BlobStore(support: f.paths.support)
    let weights = try await store.snapshotURL(model: "fixture-chat", revision: f.revision, precision: "int4")
    let before = try Data(contentsOf: weights.appendingPathComponent("arsenal-snapshot.json"))
    let weightFiles = try SafePath.files(in: weights)
    let identities = try weightFiles.map { try FileIdentity.read(weights.appendingPathComponent($0)) }
    let (data, signature, key, source) = try bumpedCatalog(f)
    let s = try await RealModelsService.open(
        catalogData: data, signature: signature, publicKey: key, source: source, kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    do { try await s.load(id: "fixture-chat", precision: "int4", fast: false) } catch { await s.shutdown(); throw error }
    #expect(await s.rows().first?.state == .loaded)
    #expect(try Data(contentsOf: weights.appendingPathComponent("arsenal-snapshot.json")) == before)
    #expect(try weightFiles.map { try FileIdentity.read(weights.appendingPathComponent($0)) } == identities)
    #expect(try Data(contentsOf: f.paths.directory("models").appendingPathComponent("fixture-chat/model.toml")) == source.files["models/fixture-chat/model.toml"])
    await s.shutdown()
}

@Test func failedFolderRefreshPreservesOldFolderAndWeights() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "int4"); await f.service.shutdown()
    let folder = f.paths.directory("models").appendingPathComponent("fixture-chat/model.toml")
    let before = try Data(contentsOf: folder)
    let (data, signature, key, source) = try bumpedCatalog(f)
    var badSource = source; badSource.files["models/fixture-chat/model.toml"] = Data("corrupt source".utf8)
    let s = try await RealModelsService.open(
        catalogData: data, signature: signature, publicKey: key, source: badSource, kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    do { try await s.load(id: "fixture-chat", precision: "int4", fast: false); Issue.record("Accepted corrupt source") } catch let e as WorkerFailure {
        #expect(e.code == "integrity"); #expect(e.message.contains("\(MCPNames.prefix)download(model: fixture-chat) to repair"))
    }
    #expect(try Data(contentsOf: folder) == before)
    #expect(await s.supervisor.residentIDs().isEmpty)
    await s.shutdown()
}

@Test func sameRevisionPinCorrectionGetsOnlyChangedBlobAndRebuildsDerived() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], filePayloads: ["base.bin": Data([1, 2, 3]), "shared.bin": Data([7, 8, 9])]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "int4"); await f.service.shutdown()
    var catalog = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData); catalog.sequence += 1
    var files = f.source.files
    let payload = Data([4, 5, 6]), hash = FileDigest.sha256(payload)
    catalog.models[0].manifest.weights.sha256?["base.bin"] = hash
    let path = "models/fixture-chat/model.toml"
    files[path] = Data(String(decoding: files[path]!, as: UTF8.self).replacingOccurrences(of: FileDigest.sha256(Data([1, 2, 3])), with: hash).utf8)
    let index = catalog.models[0].files.firstIndex { $0.path == "model.toml" }!
    catalog.models[0].files[index] = CatalogFile(path: "model.toml", sha256: FileDigest.sha256(files[path]!), bytes: Int64(files[path]!.count))
    let hf = f.paths.support.deletingLastPathComponent().appendingPathComponent("hf/test/fixture/resolve/\(f.revision)/base.bin")
    try payload.write(to: hf)
    let data = try catalog.encoded(), key = Curve25519.Signing.PrivateKey()
    let s = try await RealModelsService.open(
        catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: FixtureCatalogSource(files: files), kinds: KindSpec.defaults,
        paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    #expect(await s.rows().first?.downloadBytes["int4"] == 3)
    try await s.get(id: "fixture-chat", precision: "int4")
    try await s.load(id: "fixture-chat", precision: "int4", fast: false)
    #expect(await s.rows().first?.states["int4"]?.state == .loaded)
    await s.shutdown()
    let reopened = try await RealModelsService.open(
        catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: FixtureCatalogSource(files: files), kinds: KindSpec.defaults,
        paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    #expect(await reopened.rows().first?.states["int4"]?.state == .ready)
    #expect(await reopened.rows().first?.downloadBytes["int4"] == 0)
    await reopened.shutdown()
}

@Test func realFinalSupportFolderRefreshReadOnly() async throws {
    guard let path = ProcessInfo.processInfo.environment["ARSENAL_FINAL_SUPPORT_READONLY"] else { return }
    let original = URL(fileURLWithPath: path)
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let data = try Data(contentsOf: repository.appendingPathComponent("Resources/catalog/catalog.json"))
    let signature = try Data(contentsOf: repository.appendingPathComponent("Resources/catalog/catalog.json.sig"))
    let key = try Data(contentsOf: repository.appendingPathComponent("Resources/catalog-public-key"))
    let accepted = try CatalogTrust.verify(data: data, signature: signature, publicKey: key)
    let current = try #require(accepted.models.first { $0.id == "qwen3.8-27b" })
    let oldFolder = original.appendingPathComponent("models/qwen3.8-27b")
    let old = try ModelManifest.load(oldFolder)
    #expect(current.manifest.hasSameInstallationPins(as: old, precision: "int4"))
    let support = URL(fileURLWithPath: "/tmp/arsenal-final-c/real-folder-\(UUID().uuidString)")
    let destination = support.appendingPathComponent("models/qwen3.8-27b")
    for file in try SafePath.files(in: oldFolder) {
        let source = try SafePath.resolve(file, inside: oldFolder)
        #expect(try FileDigest.bytes(source) < 1_048_576)
        let target = destination.appendingPathComponent(file)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: source, to: target)
    }
    let before = try Data(contentsOf: oldFolder.appendingPathComponent("model.toml"))
    let installer = ModelFolderInstaller(support: support)
    try await installer.accept(data: data, signature: signature, publicKey: key)
    await #expect(throws: (any Error).self) { try await installer.verifyInstalled(id: old.id) }
    let folder = try await installer.verifiedOrRefresh(id: old.id, source: LocalGitCatalogSource(repo: repository))
    _ = try await installer.verifyInstalled(id: old.id)
    #expect(try Data(contentsOf: oldFolder.appendingPathComponent("model.toml")) == before)
    print(
        "Read-only real confirmation: qwen int4 pins unchanged; signed seq \(accepted.sequence) folder refreshed and verified in \(folder.path); original files and all weights untouched."
    )
}

@Test(arguments: [false, true]) func weightsOrRecipeAdvanceOffersUpdateInsteadOfLoad(recipeOnly: Bool) async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "int4"); await f.service.shutdown()
    let (data, signature, key, source) = try bumpedCatalog(f, weightsChange: !recipeOnly, recipeChange: recipeOnly)
    let s = try await RealModelsService.open(
        catalogData: data, signature: signature, publicKey: key, source: source, kinds: KindSpec.defaults, paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    let row = try #require(await s.rows().first)
    #expect(row.states["int4"]?.state == .updateAvailable)
    #expect(row.states["int4"]?.phase == "updateAvailable")
    #expect(row.states["int4"]?.error == nil)
    #expect(row.downloadBytes["int4"] == 0) // same verified blob, new immutable revision
    do { try await s.load(id: "fixture-chat", precision: "int4", fast: false); Issue.record("Loaded obsolete pins") } catch let e as WorkerFailure {
        #expect(e.code == "update_available"); #expect(e.message.contains("\(MCPNames.prefix)download(model: fixture-chat"))
        #expect(!e.message.contains("0.0 MB"))
    }
    #expect(await s.rows().first?.states["int4"]?.state == .updateAvailable)
    #expect(await s.supervisor.residentIDs().isEmpty)
    await s.resetInstalledState("fixture-chat", precision: "int4")
    #expect(await s.rows().first?.states["int4"]?.state == .updateAvailable)
    try await s.get(id: "fixture-chat", precision: "int4")
    #expect(await s.rows().first?.states["int4"]?.state == .ready)
    #expect(await s.rows().first?.states["int4"]?.error == nil)
    try await s.load(id: "fixture-chat", precision: "int4", fast: false)
    await s.shutdown()
}

@Test func snapshotReplacementKeepsSharedBlobAndRefusesResidentLease() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-final-c/replace-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let store = BlobStore(support: root)
    func blob(_ payload: Data, _ name: String) async throws -> DerivedFile {
        let file = root.appendingPathComponent(name); try payload.write(to: file)
        let pin = DerivedFile(path: name, bytes: Int64(payload.count), sha256: FileDigest.sha256(payload))
        try await store.promote(file, sha256: pin.sha256, bytes: pin.bytes)
        return pin
    }
    let shared = try await blob(Data([1, 2, 3]), "shared.bin"), old = try await blob(Data([4, 5, 6]), "old.bin"), new = try await blob(Data([7, 8, 9]), "new.bin")
    let revision = String(repeating: "a", count: 40)
    let first = SnapshotManifest(model: "first", revision: revision, precision: "bf16", files: [shared, old], derived: nil)
    let other = SnapshotManifest(model: "other", revision: revision, precision: "bf16", files: [shared], derived: nil)
    try await store.commitSnapshot(first); try await store.commitSnapshot(other)
    let replacement = SnapshotManifest(model: "first", revision: revision, precision: "bf16", files: [shared, new], derived: nil)
    await store.retain(model: "first")
    await #expect(throws: (any Error).self) { try await store.commitSnapshot(replacement, replacingExisting: true) }
    await store.release(model: "first")
    try await store.commitSnapshot(replacement, replacingExisting: true)
    #expect(try await store.verifySnapshot(model: "first", revision: revision, precision: "bf16") == replacement)
    #expect(try await store.verifySnapshot(model: "other", revision: revision, precision: "bf16") == other)
    #expect(try await store.referenceCounts()[shared.sha256] == 2)
    #expect(try await store.hasBlob(sha256: shared.sha256, bytes: shared.bytes))
    #expect(try await !store.hasBlob(sha256: old.sha256, bytes: old.bytes))
}
