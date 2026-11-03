// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import Testing
@testable import ArsenalCore

/// The store half of the quiet weights collection: copies relinked to their blob, interrupted downloads removed at startup.
/// Every store lives in a private temp directory; nothing here touches an installed app or a real model.
@Suite struct WeightsCollectionTests {
    static let revision = String(repeating: "a", count: 40)
    func temp() throws -> URL {
        let root = URL(fileURLWithPath: "/tmp/arsenal-weights-collection/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    func blob(_ store: BlobStore, _ root: URL, _ text: String) async throws -> DerivedFile {
        let data = Data(text.utf8), file = DerivedFile(path: "model.safetensors", bytes: Int64(data.count), sha256: FileDigest.sha256(data))
        let part = root.appendingPathComponent("part-" + UUID().uuidString); try data.write(to: part)
        try await store.promote(part, sha256: file.sha256, bytes: file.bytes)
        return file
    }
    func inode(_ url: URL) -> UInt64 { var s = stat(); return lstat(url.path, &s) == 0 ? UInt64(s.st_ino) : 0 }

    @Test func committedSnapshotsLinkTheirBlobInsteadOfCopyingIt() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root), file = try await blob(store, root, "fp16-weights")
        let snapshot = try await store.commitSnapshot(SnapshotManifest(model: "m", revision: Self.revision, precision: "fp16", files: [file], derived: nil))
        #expect(inode(snapshot.appendingPathComponent(file.path)) == inode(try await store.blobURL(sha256: file.sha256)))
    }

    @Test func aSnapshotFileThatIsACopyIsRelinkedToItsBlobAndItsBytesFreed() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root), file = try await blob(store, root, "fp16-weights")
        let snapshot = try await store.commitSnapshot(SnapshotManifest(model: "m", revision: Self.revision, precision: "fp16", files: [file], derived: nil))
        let target = snapshot.appendingPathComponent(file.path), blobURL = try await store.blobURL(sha256: file.sha256)
        try FileManager.default.removeItem(at: target); try Data("fp16-weights".utf8).write(to: target) // a separate copy
        #expect(inode(target) != inode(blobURL))
        #expect(await store.relinkSnapshotCopies() == (file.bytes, 0))
        #expect(inode(target) == inode(blobURL))
        #expect(try await store.verifySnapshot(model: "m", revision: Self.revision, precision: "fp16") != nil)
        #expect(await store.relinkSnapshotCopies() == (0, 0)) // a repeat finds nothing
    }

    @Test func aSnapshotFileWhoseBlobIsMissingBecomesTheBlob() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root), file = try await blob(store, root, "fp16-weights")
        let snapshot = try await store.commitSnapshot(SnapshotManifest(model: "m", revision: Self.revision, precision: "fp16", files: [file], derived: nil))
        let blobURL = try await store.blobURL(sha256: file.sha256)
        try FileManager.default.removeItem(at: blobURL)
        #expect(await store.relinkSnapshotCopies() == (0, 0))
        #expect(inode(blobURL) == inode(snapshot.appendingPathComponent(file.path)))
        #expect(try await store.verifySnapshot(model: "m", revision: Self.revision, precision: "fp16", forceHash: true) != nil)
    }

    @Test func aLeasedModelIsNotRelinked() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root), file = try await blob(store, root, "fp16-weights")
        let snapshot = try await store.commitSnapshot(SnapshotManifest(model: "m", revision: Self.revision, precision: "fp16", files: [file], derived: nil))
        let target = snapshot.appendingPathComponent(file.path)
        try FileManager.default.removeItem(at: target); try Data("fp16-weights".utf8).write(to: target)
        await store.retain(model: "m")
        #expect(await store.relinkSnapshotCopies() == (0, 0))
        #expect(inode(target) != inode(try await store.blobURL(sha256: file.sha256)))
    }

    @Test func interruptedDownloadsAreRemovedExceptRecentPinnedOnesAndOnesAGetReserves() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root), downloads = root.appendingPathComponent("weights/downloads")
        try FileManager.default.createDirectory(at: downloads, withIntermediateDirectories: true)
        let dropped = String(repeating: "1", count: 64), active = String(repeating: "2", count: 64)
        let recent = String(repeating: "3", count: 64), old = String(repeating: "4", count: 64)
        let names = [
            dropped + ".part", dropped + ".json", recent + ".part", recent + ".json", recent + ".part.old-" + UUID().uuidString,
            old + ".part", active + ".part", "notes.txt"
        ]
        for name in names { try Data(repeating: 7, count: 10).write(to: downloads.appendingPathComponent(name)) }
        let dayOld = Date().addingTimeInterval(-2 * 86_400)
        try FileManager.default.setAttributes([.modificationDate: dayOld], ofItemAtPath: downloads.appendingPathComponent(old + ".part").path)
        await store.reserve(files: [DerivedFile(path: "x", bytes: 10, sha256: active)])
        let freed = try await store.removeInterruptedDownloads(pinned: [recent, old, active], resumableSince: Date().addingTimeInterval(-86_400))
        #expect(freed == 40) // the dropped pair, the quarantined copy, the stale pinned partial
        #expect(Set(try FileManager.default.contentsOfDirectory(atPath: downloads.path)) == [recent + ".part", recent + ".json", active + ".part", "notes.txt"])
        let log = try String(contentsOf: root.appendingPathComponent("logs/store-deletions.jsonl"), encoding: .utf8)
        #expect(log.components(separatedBy: "delete_interrupted_download").count - 1 == 4)
    }
    @Test func aStaleRelinkLeftoverIsSweptAndNeverLandsInASnapshot() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root), file = try await blob(store, root, "fp16-weights")
        let snapshot = try await store.commitSnapshot(SnapshotManifest(model: "m", revision: Self.revision, precision: "fp16", files: [file], derived: nil))
        let leftover = root.appendingPathComponent("weights/.relink-\(UUID().uuidString)")
        try FileManager.default.linkItem(at: try await store.blobURL(sha256: file.sha256), to: leftover)
        _ = await store.relinkSnapshotCopies()
        #expect(!FileManager.default.fileExists(atPath: leftover.path))
        #expect(Set(try FileManager.default.contentsOfDirectory(atPath: snapshot.path)) == [file.path, "arsenal-snapshot.json"])
    }

    @Test func aPinnedPartialAndItsJournalStayOrGoTogether() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root), downloads = root.appendingPathComponent("weights/downloads")
        try FileManager.default.createDirectory(at: downloads, withIntermediateDirectories: true)
        let hash = String(repeating: "5", count: 64)
        for name in [hash + ".part", hash + ".json"] { try Data(repeating: 1, count: 4).write(to: downloads.appendingPathComponent(name)) }
        // The journal is old, the partial was written a minute ago: the pair is recent and both stay.
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-3 * 86_400)], ofItemAtPath: downloads.appendingPathComponent(hash + ".json").path)
        #expect(try await store.removeInterruptedDownloads(pinned: [hash], resumableSince: Date().addingTimeInterval(-86_400)) == 0)
        #expect(Set(try FileManager.default.contentsOfDirectory(atPath: downloads.path)) == [hash + ".part", hash + ".json"])
    }

    @Test func anAssetACurrentModelDeclaresSurvivesOrphanCleanupWithoutAReferenceFile() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root), file = try await blob(store, root, "g2p-lexicon")
        let kept = String(repeating: "c", count: 40), stale = String(repeating: "d", count: 40)
        for revision in [kept, stale] {
            try await store.commitSnapshot(SnapshotManifest(model: "asset-g2p", revision: revision, precision: "fp16", files: [file], derived: nil))
        }
        _ = try await store.deleteOrphans(keepAssets: ["g2p/" + kept])
        let assets = root.appendingPathComponent("weights/snapshots/asset-g2p")
        #expect(Set(try FileManager.default.contentsOfDirectory(atPath: assets.path)) == [kept])
        #expect(FileManager.default.fileExists(atPath: try await store.blobURL(sha256: file.sha256).path))
    }
}
