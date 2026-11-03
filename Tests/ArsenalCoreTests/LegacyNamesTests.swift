// SPDX-License-Identifier: MIT
import ArsenalWire
import Foundation
import Testing
@testable import ArsenalCore

/// Stores and model folders written before the rename keep their legacy names on disk; Arsenal reads them as they are
/// and writes only the current names.
struct LegacyNamesTests {
    @Test func aSnapshotUnderLegacyNamesStillVerifiesAndCounts() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-legacy-names-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let store = BlobStore(support: root)
        let payload = Data([1, 2, 3]), file = root.appendingPathComponent("weights.bin"); try payload.write(to: file)
        let pin = DerivedFile(path: "weights.bin", bytes: Int64(payload.count), sha256: FileDigest.sha256(payload))
        try await store.promote(file, sha256: pin.sha256, bytes: pin.bytes)
        let revision = String(repeating: "a", count: 40)
        let manifest = SnapshotManifest(model: "legacy", revision: revision, precision: "fp16", files: [pin], derived: nil)
        let directory = try await store.commitSnapshot(manifest)
        #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent(StoredNames.snapshot).path))
        // As an older app wrote it: the same manifest under the legacy name.
        try FileManager.default.moveItem(
            at: directory.appendingPathComponent(StoredNames.snapshot), to: directory.appendingPathComponent(StoredNames.legacySnapshot))
        #expect(StoredNames.hasSnapshot(directory) && StoredNames.snapshotURL(in: directory).lastPathComponent == StoredNames.legacySnapshot)
        #expect(try await store.verifySnapshot(model: "legacy", revision: revision, precision: "fp16") == manifest)
        #expect(try RealModelsService.installedDtypes(id: "legacy", support: root) == ["fp16"])
        // Nothing renames the legacy file.
        #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent(StoredNames.snapshot).path))
    }

    @Test func legacySchemaKeywordsReadAsCurrentOnes() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-legacy-schema-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let prefix = "x-" + "mob" + "ius-"
        let schema = """
            {"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false,"required":["image"],
             "properties":{"image":{"type":"object","\(prefix)file":{"accept":["image/png"]},"required":["path"],"additionalProperties":false,
             "properties":{"path":{"type":"string"}}},"language":{"type":"string","\(prefix)control":"language"}}}
            """
        try Data(schema.utf8).write(to: folder.appendingPathComponent("input.json"))
        let object = try WireJSON.decode(OperationSchema.read(path: "input.json", folder: folder))
        #expect(object["properties"]?["image"]?["x-arsenal-file"]?["accept"] == ["image/png"])
        #expect(object["properties"]?["language"]?["x-arsenal-control"] == "language")
        #expect(object["properties"]?["image"]?[prefix + "file"] == nil)
    }

    @Test func legacyLayoutAndKeysFallBackOnlyWhenTheCurrentOneIsAbsent() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-legacy-marker-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        #expect(StoredNames.current("new.json", legacy: "old.json", in: folder).lastPathComponent == "new.json")
        try Data("{}".utf8).write(to: folder.appendingPathComponent("old.json"))
        #expect(StoredNames.current("new.json", legacy: "old.json", in: folder).lastPathComponent == "old.json")
        try Data("{}".utf8).write(to: folder.appendingPathComponent("new.json"))
        #expect(StoredNames.current("new.json", legacy: "old.json", in: folder).lastPathComponent == "new.json")
        #expect(StoredNames.reserved.count == 4 && StoredNames.reserved.contains(StoredNames.legacyDerived))
    }
}
