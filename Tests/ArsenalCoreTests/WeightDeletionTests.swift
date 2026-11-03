// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Per-precision deletion, inventory and orphan cleanup in the weights store (edge cases 1 to 8 of the brief). Every store lives
/// in a private temp directory; nothing here touches an installed app, the user's store or a real model.
@Suite struct WeightDeletionTests {
    static let revision = String(repeating: "a", count: 40)
    static let otherRevision = String(repeating: "b", count: 40)
    /// The store lives in `support/weights`; tests read the directory from the temp support root (the store is an actor).
    func weightsDir(_ support: URL) -> URL { support.appendingPathComponent("weights") }

    func temp() throws -> URL {
        let root = URL(fileURLWithPath: "/tmp/arsenal-weight-deletion/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    /// A verified blob holding `text`, promoted from a scratch file the way a finished download is.
    func blob(_ store: BlobStore, _ root: URL, _ text: String, name: String = "weights.safetensors") async throws -> DerivedFile {
        let data = Data(text.utf8)
        let file = DerivedFile(path: name, bytes: Int64(data.count), sha256: FileDigest.sha256(data))
        let part = root.appendingPathComponent("part-" + UUID().uuidString)
        try data.write(to: part)
        try await store.promote(part, sha256: file.sha256, bytes: file.bytes)
        return file
    }
    func commit(_ store: BlobStore, model: String, precision: String, files: [DerivedFile], revision: String = Self.revision) async throws {
        try await store.commitSnapshot(SnapshotManifest(model: model, revision: revision, precision: precision, files: files, derived: nil))
    }
    /// Asserts that `action` throws and that the message names the reason.
    func expectRefused(_ fragment: String, _ action: () async throws -> Void) async {
        do {
            try await action()
            Issue.record("the deletion was allowed; expected a refusal mentioning \"\(fragment)\"")
        } catch {
            #expect(error.localizedDescription.contains(fragment), "message was: \(error.localizedDescription)")
        }
    }
    func manifestBytes(_ store: BlobStore, model: String, precision: String, revision: String = Self.revision) async throws -> Int64 {
        let directory = try await store.snapshotURL(model: model, revision: revision, precision: precision)
        return try FileDigest.bytes(directory.appendingPathComponent("arsenal-snapshot.json"))
    }

    // Core review, 2 Nov: a delete whose blob could not be unlinked (here an unreadable snapshot elsewhere blocks the
    // collection) must not leave it verified, or the next run restores the deleted weights without asking.
    @Test func deletedBlobThatCannotBeUnlinkedIsNoLongerVerifiedPresent() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let file = try await blob(store, root, "ddcolor-weights")
        try await commit(store, model: "ddcolor", precision: "fp16", files: [file])
        #expect(try await store.isVerifiedPresent(sha256: file.sha256, bytes: file.bytes))
        // Another model's snapshot manifest is unreadable, so collection keeps every released blob.
        let other = weightsDir(root).appendingPathComponent("snapshots/other/\(Self.revision)/fp16")
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        try Data("not json".utf8).write(to: other.appendingPathComponent(StoredNames.snapshotNames.sorted().first!))
        _ = try await store.delete(model: "ddcolor")
        #expect(FileManager.default.fileExists(atPath: try await store.blobURL(sha256: file.sha256).path))
        #expect(try await store.isVerifiedPresent(sha256: file.sha256, bytes: file.bytes) == false)
    }

    // Edge 1: a precision that is loaded refuses, with a message that says what to do.
    @Test func loadedPrecisionRefusesAndOtherPrecisionsOfALoadedModelStayDeletable() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let fp16 = try await blob(store, root, "fp16-weights"), int8 = try await blob(store, root, "int8-weights")
        try await commit(store, model: "m", precision: "fp16", files: [fp16])
        try await commit(store, model: "m", precision: "int8", files: [int8])
        await store.retain(model: "m") // a worker is loaded; its precision is fp16
        await expectRefused("unload m") { _ = try await store.delete(model: "m", precision: "fp16", residentPrecision: "fp16") }
        await expectRefused("unload m") { _ = try await store.delete(model: "m") }
        // The loaded worker holds fp16, so int8 can go while fp16 stays.
        let int8Manifest = try await manifestBytes(store, model: "m", precision: "int8")
        let freed = try await store.delete(model: "m", precision: "int8", residentPrecision: "fp16")
        #expect(freed == int8.bytes + int8Manifest)
        #expect(try await store.verifySnapshot(model: "m", revision: Self.revision, precision: "fp16") != nil)
        await store.release(model: "m")
        #expect(try await store.delete(model: "m", precision: "fp16") > 0)
    }

    // Edge 2: deleting while the same model is downloading or converting refuses.
    @Test func modelWithAGetOrConversionInFlightRefusesDeletion() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let fp16 = try await blob(store, root, "fp16-weights"), int8 = try await blob(store, root, "int8-weights")
        try await commit(store, model: "m", precision: "fp16", files: [fp16])
        try await commit(store, model: "m", precision: "int8", files: [int8])
        await store.retain(model: "m") // a Get of this model is running
        await expectRefused("finish the Get") { _ = try await store.delete(model: "m", precision: "int8") }
        await expectRefused("unload m") { _ = try await store.delete(model: "m") }
        await store.release(model: "m")
        await store.reserve(files: [fp16]) // a conversion is reading fp16's blob
        await expectRefused("in use by a download or conversion") { _ = try await store.delete(model: "m", precision: "fp16") }
        await store.unreserve(files: [fp16])
        #expect(try await store.delete(model: "m", precision: "fp16") > 0)
    }

    // Edge 3: two precisions on disk; deleting one leaves the other intact, and a blob both share stays.
    @Test func deletingOnePrecisionLeavesTheOtherIntact() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let fp16 = try await blob(store, root, "fp16-only-weights", name: "fp16.safetensors")
        let int8 = try await blob(store, root, "int8-only-weights", name: "int8.safetensors")
        let shared = try await blob(store, root, "shared-tensor-bytes", name: "shared.safetensors")
        try await commit(store, model: "m", precision: "fp16", files: [fp16, shared])
        try await commit(store, model: "m", precision: "int8", files: [int8, shared])
        let int8Manifest = try await manifestBytes(store, model: "m", precision: "int8")
        let freed = try await store.delete(model: "m", precision: "int8")
        // The int8 blob and its manifest come back; the shared blob stays with fp16 and is not counted.
        #expect(freed == int8.bytes + int8Manifest)
        #expect(try await store.verifySnapshot(model: "m", revision: Self.revision, precision: "fp16") != nil)
        #expect(try await store.hasBlob(sha256: fp16.sha256, bytes: fp16.bytes))
        #expect(try await store.hasBlob(sha256: shared.sha256, bytes: shared.bytes))
        #expect(try await store.hasBlob(sha256: int8.sha256, bytes: int8.bytes) == false)
        #expect(try await store.precisionBytes(model: "m").keys.sorted() == ["fp16"])
    }

    // Deletes collect only what they released: an orphan (a blob nothing refers to) stays listed through a precision delete and
    // a whole-model delete, and only deleteOrphans removes it.
    @Test func deletingAPrecisionLeavesAnUnrelatedOrphanIntact() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let fp16 = try await blob(store, root, "fp16-weights", name: "fp16.safetensors")
        let int8 = try await blob(store, root, "int8-weights", name: "int8.safetensors")
        let orphan = try await blob(store, root, "orphan-unlisted-weights")
        try await commit(store, model: "m", precision: "fp16", files: [fp16])
        try await commit(store, model: "m", precision: "int8", files: [int8])
        let int8Manifest = try await manifestBytes(store, model: "m", precision: "int8")
        #expect(try await store.delete(model: "m", precision: "int8") == int8.bytes + int8Manifest)
        #expect(try await store.hasBlob(sha256: orphan.sha256, bytes: orphan.bytes))
        #expect(try await store.orphanBytes() == orphan.bytes)
        let fp16Manifest = try await manifestBytes(store, model: "m", precision: "fp16")
        #expect(try await store.delete(model: "m") == fp16.bytes + fp16Manifest)
        #expect(try await store.hasBlob(sha256: orphan.sha256, bytes: orphan.bytes))
        #expect(try await store.deleteOrphans() == orphan.bytes)
        #expect(try await store.hasBlob(sha256: orphan.sha256, bytes: orphan.bytes) == false)
    }

    // A local donor copied across volumes (EXDEV) gets the Get's disk-full message, like a clone or a promote.
    @Test func localCopyDiskFullUsesTheGetMessage() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let donor = root.appendingPathComponent("donor.safetensors"); try Data("donor-weights".utf8).write(to: donor)
        let stage = root.appendingPathComponent("stage.part")
        await store.injectWriteFault { name in name == "import.copy" ? POSIXError(.ENOSPC) : nil }
        do {
            try await store.copyDonor(donor, to: stage)
            Issue.record("a full disk copied the donor")
        } catch {
            #expect(error.localizedDescription == "disk full; free space and retry Get", "message was: \(error.localizedDescription)")
        }
        await store.injectWriteFault(nil)
        try await store.copyDonor(donor, to: stage)
        #expect(FileManager.default.fileExists(atPath: stage.path))
    }

    // Edge 4: a full disk during a promote or a commit removes the partial file and any blob it created, names the cause, and
    // leaves no snapshot, so nothing reads ready. A retry after the disk frees up publishes the snapshot.
    @Test func diskFullDuringPromoteLeavesNoPartialOrBlob() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        for point in ["promote.move", "promote.record"] {
            let data = Data("disk-full-\(point)".utf8)
            let file = DerivedFile(path: "weights.safetensors", bytes: Int64(data.count), sha256: FileDigest.sha256(data))
            let part = root.appendingPathComponent("part-\(point)")
            try data.write(to: part)
            await store.injectWriteFault { name in name == point ? POSIXError(.ENOSPC) : nil }
            do {
                try await store.promote(part, sha256: file.sha256, bytes: file.bytes)
                Issue.record("\(point) succeeded on a full disk")
            } catch {
                #expect(error.localizedDescription.contains("disk full"), "message was: \(error.localizedDescription)")
            }
            #expect(FileManager.default.fileExists(atPath: part.path) == false)
            #expect(try await store.hasBlob(sha256: file.sha256, bytes: file.bytes) == false)
            #expect(try await store.isVerifiedPresent(sha256: file.sha256, bytes: file.bytes) == false)
        }
        await store.injectWriteFault(nil)
    }
    @Test func diskFullDuringCommitIsNeverReadyAndRetryPublishes() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let file = try await blob(store, root, "commit-weights")
        await store.injectWriteFault { name in name == "commit.link" ? POSIXError(.ENOSPC) : nil }
        do {
            try await commit(store, model: "m", precision: "fp16", files: [file])
            Issue.record("commit succeeded on a full disk")
        } catch {
            #expect(error.localizedDescription.contains("disk full"), "message was: \(error.localizedDescription)")
        }
        #expect(try await store.verifySnapshot(model: "m", revision: Self.revision, precision: "fp16") == nil)
        #expect(try await store.precisionBytes(model: "m").isEmpty)
        let staging = try FileManager.default.contentsOfDirectory(atPath: weightsDir(root).path).filter { $0.hasPrefix(".snapshot-") }
        #expect(staging.isEmpty)
        await store.injectWriteFault(nil)
        try await commit(store, model: "m", precision: "fp16", files: [file])
        #expect(try await store.verifySnapshot(model: "m", revision: Self.revision, precision: "fp16") != nil)
    }

    // Edge 5: a corrupt weight file (hash mismatch) is never verified, and its precision can still be deleted.
    @Test func corruptWeightIsNotVerifiedAndDeleteStillWorks() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let file = try await blob(store, root, "good-weights-bytes")
        try await commit(store, model: "m", precision: "fp16", files: [file])
        let path = try await store.blobURL(sha256: file.sha256)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: path.path)
        try Data("bad-weights-bytes!".utf8).write(to: path) // same length, different content
        do {
            _ = try await store.verifySnapshot(model: "m", revision: Self.revision, precision: "fp16")
            Issue.record("a corrupt weight file verified")
        } catch {}
        let freed = try await store.delete(model: "m", precision: "fp16")
        #expect(freed >= file.bytes)
        #expect(try await store.precisionBytes(model: "m").isEmpty)
    }

    // Edge 6: an orphan blob, an unowned asset snapshot and stale snapshot staging are reported as orphan bytes, and deletable.
    @Test func orphansAreReportedAndDeletable() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let orphan = try await blob(store, root, "orphan-blob-bytes")
        let asset = try await blob(store, root, "unowned-asset-bytes")
        try await commit(store, model: "asset-old", precision: "fp16", files: [asset], revision: Self.otherRevision)
        let assetManifest = try await manifestBytes(store, model: "asset-old", precision: "fp16", revision: Self.otherRevision)
        let staging = weightsDir(root).appendingPathComponent(".snapshot-stale")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 40).write(to: staging.appendingPathComponent("arsenal-snapshot.json"))
        let expected = orphan.bytes + asset.bytes + assetManifest + 40
        #expect(try await store.orphanBytes() == expected)
        #expect(try await store.deleteOrphans() == expected)
        #expect(try await store.orphanBytes() == 0)
        #expect(try await store.hasBlob(sha256: orphan.sha256, bytes: orphan.bytes) == false)
        #expect(FileManager.default.fileExists(atPath: staging.path) == false)
    }

    // Edge 7: a repeated delete (rapid clicks) finds nothing and frees nothing; it never crashes or throws.
    @Test func repeatedDeleteIsANoOp() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let file = try await blob(store, root, "repeat-weights")
        try await commit(store, model: "m", precision: "fp16", files: [file])
        let manifest = try await manifestBytes(store, model: "m", precision: "fp16")
        #expect(try await store.delete(model: "m", precision: "fp16") == file.bytes + manifest)
        #expect(try await store.delete(model: "m", precision: "fp16") == 0)
        #expect(try await store.delete(model: "m", precision: "fp16") == 0)
        // The first delete collected the blob, so the recommit promotes it again from a fresh download.
        let again = try await blob(store, root, "repeat-weights")
        try await commit(store, model: "m", precision: "fp16", files: [again])
        #expect(try await store.delete(model: "m") > 0)
        #expect(try await store.delete(model: "m") == 0)
    }

    // Edge 8: a relaunch mid-download finds only the partial file. The partial is resumable, never a snapshot, never ready,
    // and the model's deletion clears it.
    @Test func relaunchMidDownloadKeepsOnlyAResumablePartial() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let payload = Data(repeating: 9, count: 4096)
        let hash = FileDigest.sha256(payload)
        let downloads = weightsDir(root).appendingPathComponent("downloads")
        try FileManager.default.createDirectory(at: downloads, withIntermediateDirectories: true)
        try payload.prefix(1000).write(to: downloads.appendingPathComponent(hash + ".part"))
        try Data("{}".utf8).write(to: downloads.appendingPathComponent(hash + ".json"))
        let relaunched = BlobStore(support: root) // the app starts again on the same disk
        #expect(try await relaunched.precisionBytes(model: "m").isEmpty)
        #expect(try await relaunched.verifySnapshot(model: "m", revision: Self.revision, precision: "fp16") == nil)
        #expect(try await relaunched.onDiskModelIDs().contains("m") == false)
        #expect(await relaunched.partialBytes(sha256s: [hash]) == 1002)
        #expect(try await relaunched.delete(model: "m", partialSHA256s: [hash]) == 1002)
        #expect(await relaunched.partialBytes(sha256s: [hash]) == 0)
    }

    // A Get refused at a gated repository leaves only the download's journal (VM run, 2 Nov): no weights, so the model
    // is not listed with "Frees up to 0 KB" to delete. A started download still counts its journal with its bytes.
    @Test func aJournalWithoutDownloadedBytesIsNoPartial() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let hash = FileDigest.sha256(Data(repeating: 7, count: 4096))
        let downloads = weightsDir(root).appendingPathComponent("downloads")
        try FileManager.default.createDirectory(at: downloads, withIntermediateDirectories: true)
        try Data(String(repeating: "j", count: 250).utf8).write(to: downloads.appendingPathComponent(hash + ".json"))
        let store = BlobStore(support: root)
        #expect(await store.partialBytes(sha256s: [hash]) == 0)
        try Data().write(to: downloads.appendingPathComponent(hash + ".part"))
        #expect(await store.partialBytes(sha256s: [hash]) == 0)
        try Data(repeating: 7, count: 100).write(to: downloads.appendingPathComponent(hash + ".part"))
        #expect(await store.partialBytes(sha256s: [hash]) == 350)
        // The model's Delete still clears the journal.
        #expect(try await store.delete(model: "m", partialSHA256s: [hash]) == 350)
    }

    // Inventory: every model with weights is listed, catalog or not, with its precisions, asset references and cache bytes.
    @Test func listsEveryModelWithWeightsAndItsCaches() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let fp16 = try await blob(store, root, "listed-fp16"), int8 = try await blob(store, root, "listed-int8")
        let speech = try await blob(store, root, "speech-encoder-bytes")
        let gone = try await blob(store, root, "gone-model-bytes")
        try await commit(store, model: "catalog-model", precision: "fp16", files: [fp16])
        try await commit(store, model: "catalog-model", precision: "int8", files: [int8])
        try await commit(store, model: "gone-model", precision: "bf16", files: [gone])
        try await commit(store, model: "asset-speech", precision: "fp16", files: [speech], revision: Self.otherRevision)
        try await store.recordAssets(model: "catalog-model", revisions: ["speech": Self.otherRevision])
        #expect(try await store.onDiskModelIDs() == ["catalog-model", "gone-model"])
        #expect(try await store.precisionBytes(model: "catalog-model").keys.sorted() == ["fp16", "int8"])
        #expect(try await store.precisionBytes(model: "gone-model").keys.sorted() == ["bf16"])
        let assetManifest = try await manifestBytes(store, model: "asset-speech", precision: "fp16", revision: Self.otherRevision)
        #expect(try await store.assetCacheBytes(model: "catalog-model") == speech.bytes + assetManifest)
        #expect(try await store.assetCacheBytes(model: "gone-model") == 0)
    }

    // Model deletion takes every precision, revision, separate-asset snapshot, retention receipt and partial download.
    @Test func deleteAllRemovesEveryPrecisionRevisionAssetAndCache() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let store = BlobStore(support: root)
        let fp16A = try await blob(store, root, "rev-a-fp16"), int8A = try await blob(store, root, "rev-a-int8")
        let fp16B = try await blob(store, root, "rev-b-fp16"), speech = try await blob(store, root, "speech-bytes")
        try await commit(store, model: "m", precision: "fp16", files: [fp16A])
        try await commit(store, model: "m", precision: "int8", files: [int8A])
        try await commit(store, model: "m", precision: "fp16", files: [fp16B], revision: Self.otherRevision)
        try await commit(store, model: "asset-speech", precision: "fp16", files: [speech], revision: Self.otherRevision)
        try await store.recordAssets(model: "m", revisions: ["speech": Self.otherRevision])
        let retention = weightsDir(root).appendingPathComponent("retention/m/\(Self.revision)")
        try FileManager.default.createDirectory(at: retention, withIntermediateDirectories: true)
        try Data(repeating: 2, count: 30).write(to: retention.appendingPathComponent("bf16.json"))
        let payload = Data(repeating: 7, count: 500), hash = FileDigest.sha256(payload)
        let downloads = weightsDir(root).appendingPathComponent("downloads")
        try FileManager.default.createDirectory(at: downloads, withIntermediateDirectories: true)
        try payload.write(to: downloads.appendingPathComponent(hash + ".part"))
        let precisions = try await store.precisionBytes(model: "m").values.reduce(0, +)
        let assets = try await store.assetCacheBytes(model: "m")
        let freed = try await store.delete(model: "m", partialSHA256s: [hash])
        // Nothing is shared here, so every logical byte comes back, plus the 30-byte receipt.
        #expect(freed == precisions + assets + 500 + 30)
        #expect(try await store.onDiskModelIDs().contains("m") == false)
        #expect(try await store.assetReferences(model: "m").isEmpty)
        #expect(FileManager.default.fileExists(atPath: weightsDir(root).appendingPathComponent("snapshots/asset-speech").path) == false)
        #expect(FileManager.default.fileExists(atPath: weightsDir(root).appendingPathComponent("retention/m").path) == false)
        #expect(try await store.orphanBytes() == 0)
    }
}
