// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import Darwin

public struct SnapshotManifest: Codable, Sendable, Equatable {
    public var model: String; public var revision: String; public var precision: String
    public var files: [DerivedFile]; public var derived: DerivedManifest?
}
/// All references live in committed snapshot manifests; no separate ref-count database can drift after a crash.
public actor BlobStore {
    public let root: URL
    var reserved: [String: Int] = [:]
    var leases: [String: Int] = [:]
    /// Tests: a named write point (`promote.move`, `promote.record`, `commit.link`, `commit.manifest`) fails with the returned
    /// error, which simulates a full disk without filling one. Nil in the app.
    private var writeFault: (@Sendable (String) -> Error?)?
    public init(support: URL) { root = support.appendingPathComponent("weights") }
    public func injectWriteFault(_ fault: (@Sendable (String) -> Error?)?) { writeFault = fault }
    func faultCheck(_ point: String) throws { if let error = writeFault?(point) { throw error } }
    static let diskFullMessage = "disk full; free space and retry Get"
    /// A full disk is reported in the Get's words, whichever write hit it (a POSIX ENOSPC or Cocoa's out-of-space error).
    static func diskFull(_ error: Error) -> Error { isDiskFull(error) ? ManifestError.invalid(diskFullMessage) : error }
    static func isDiskFull(_ error: Error) -> Bool {
        if let manifest = error as? ManifestError, manifest.errorDescription == diskFullMessage { return true }
        let nsError = error as NSError
        return (nsError.domain == NSPOSIXErrorDomain && nsError.code == Int(ENOSPC))
            || (nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileWriteOutOfSpaceError)
    }
    public func blobURL(sha256: String) throws -> URL {
        guard ModelManifest.isHex(sha256, count: 64) else { throw ManifestError.invalid("invalid blob hash") }
        return try SafePath.resolve("blobs/\(sha256.lowercased())", inside: root)
    }
    public func hasBlob(sha256: String, bytes: Int64) throws -> Bool {
        let path = try blobURL(sha256: sha256)
        guard FileManager.default.fileExists(atPath: path.path) else { return false }
        try verifyRecorded(path, sha256: sha256, bytes: bytes, forceHash: bytes < 16 * 1024 * 1024); return true
    }
    /// The downloader's .part is promoted only after a complete hash/size check, on the same support filesystem.
    public func promote(_ file: URL, sha256: String, bytes: Int64) throws {
        try FileDigest.verify(file, sha256: sha256, bytes: bytes)
        try promoteVerified(file, sha256: sha256, bytes: bytes)
    }
    private func promoteVerified(_ file: URL, sha256: String, bytes: Int64, localSource: String? = nil, sourceKind: String? = nil) throws {
        let dest = try blobURL(sha256: sha256)
        var created = false
        do {
            try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: dest.path) {
                try FileDigest.verify(dest, sha256: sha256, bytes: bytes); try FileManager.default.removeItem(at: file)
            } else {
                try faultCheck("promote.move")
                try FileManager.default.moveItem(at: file, to: dest); created = true
                try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: dest.path)
            }
            try faultCheck("promote.record")
            try recordVerified(dest, sha256: sha256, localSource: localSource, sourceKind: sourceKind)
        } catch {
            // A failed promotion leaves neither the partial file nor a blob it created, so no snapshot can ever read ready from it.
            if created { try? FileManager.default.removeItem(at: dest); try? FileManager.default.removeItem(at: verificationURL(sha256)) }
            try? FileManager.default.removeItem(at: file)
            throw Self.diskFull(error)
        }
    }
    func promote(_ file: VerifiedDownload) throws {
        guard try FileIdentity.read(file.url) == file.identity, file.identity.bytes == file.bytes else { throw ManifestError.invalid("verified download changed before promotion") }
        try promoteVerified(file.url, sha256: file.sha256, bytes: file.bytes, sourceKind: "download")
    }
    public func isVerifiedPresent(sha256: String, bytes: Int64) throws -> Bool {
        let blob = try blobURL(sha256: sha256)
        guard let identity = try? FileIdentity.read(blob), identity.bytes == bytes,
            let record = try? JSONDecoder().decode(BlobVerificationRecord.self, from: Data(contentsOf: verificationURL(sha256))),
            record.sha256 == sha256.lowercased(), record.identity == identity
        else { return false }
        return true
    }
    func verificationURL(_ hash: String) throws -> URL { try SafePath.resolve("verified/\(hash.lowercased()).json", inside: root) }
    func recordVerified(_ file: URL, sha256: String, localSource: String? = nil, sourceKind: String? = nil) throws {
        let previous = try? JSONDecoder().decode(BlobVerificationRecord.self, from: Data(contentsOf: verificationURL(sha256)))
        // A forced hash must preserve the verified blob's import history, not turn a clone into a download.
        let record = BlobVerificationRecord(
            sha256: sha256.lowercased(), identity: try FileIdentity.read(file), localSource: sourceKind == nil ? previous?.localSource : localSource,
            sourceKind: sourceKind ?? previous?.sourceKind)
        let target = try verificationURL(sha256)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(record).write(to: target, options: .atomic)
    }
    private func verifyRecorded(_ file: URL, sha256: String, bytes: Int64, forceHash: Bool) throws {
        let identity = try FileIdentity.read(file)
        guard identity.bytes == bytes else { throw ManifestError.invalid("verified file size changed") }
        let record = try? JSONDecoder().decode(BlobVerificationRecord.self, from: Data(contentsOf: verificationURL(sha256)))
        if !forceHash, record?.sha256 == sha256.lowercased(), record?.identity == identity { return }
        try FileDigest.verify(file, sha256: sha256, bytes: bytes)
        try recordVerified(file, sha256: sha256)
    }
    /// Copies a donor that cannot be cloned (another volume). A full disk gets the Get's words, like every other write here.
    func copyDonor(_ source: URL, to stage: URL) throws {
        do {
            try faultCheck("import.copy")
            try FileManager.default.copyItem(at: source, to: stage)
        } catch {
            throw Self.diskFull(error)
        }
    }
    /// Clone first, then hash the owned clone: a cache changing during Get cannot invalidate a prior source hash.
    public func importLocal(file: URL, sha256: String, bytes: Int64, progress: @escaping @Sendable (Int64) -> Void = { _ in }) throws -> Bool {
        guard ModelManifest.isHex(sha256, count: 64), bytes >= 0 else { throw ManifestError.invalid("invalid local import pin") }
        let source = file.resolvingSymlinksInPath()
        guard (try? source.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true, (try? FileDigest.bytes(source)) == bytes else { return false }
        let imports = root.appendingPathComponent("imports"); let stage = try SafePath.resolve(UUID().uuidString + ".part", inside: imports)
        try FileManager.default.createDirectory(at: imports, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: stage) }
        progress(0)
        if clonefile(source.path, stage.path, 0) != 0 {
            let code = errno
            if code == EXDEV {
                try copyDonor(source, to: stage)
            } else if code == ENOENT || code == EACCES {
                return false
            } else if code == ENOSPC {
                throw ManifestError.invalid(Self.diskFullMessage)
            } else {
                throw ManifestError.invalid("local clone failed: \(String(cString: strerror(code))); disable get.localSources to use a download")
            }
        }
        guard try FileDigest.bytes(stage) == bytes, try FileDigest.sha256(stage, progress: progress) == sha256.lowercased() else { return false }
        try promoteVerified(stage, sha256: sha256, bytes: bytes, localSource: source.path, sourceKind: "localCopy")
        return true
    }
    /// Only Get calls this, immediately after verifying/admitting every blob in the transaction; no shared-store writer runs concurrently.
    func commitAlreadyVerifiedSnapshot(_ source: SnapshotManifest) throws -> URL { try commitSnapshot(source, verifyFiles: false, replacingExisting: true, getLease: true) }
    func linkVerifiedBlob(_ file: DerivedFile, to target: URL) throws {
        guard target.path.hasPrefix(root.path + "/") else { throw ManifestError.invalid("transient snapshot must stay inside weights") }
        _ = try SafePath.resolve(String(target.path.dropFirst(root.path.count + 1)), inside: root)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.linkItem(at: blobURL(sha256: file.sha256), to: target)
    }
    /// FP32 cast sources have no committed snapshot. Collect only their unreferenced content after Get releases its reservations.
    func collectUnreferenced(files: [DerivedFile]) throws {
        let refs = try referenceCounts()
        for hash in Set(files.map { $0.sha256.lowercased() }) where refs[hash] == nil && reserved[hash] == nil {
            let blob = try blobURL(sha256: hash)
            if FileManager.default.fileExists(atPath: blob.path) {
                try logDeletion(path: blob, bytes: FileDigest.bytes(blob), reason: "collect_fp32_cast_source"); try FileManager.default.removeItem(at: blob)
            }
        }
    }
    public func snapshotURL(model: String, revision: String, precision: String) throws -> URL {
        guard ModelManifest.validID(model), ModelManifest.isHex(revision, count: 40), ModelManifest.validDtype(precision) else {
            throw ManifestError.invalid("invalid snapshot identity")
        }
        return try SafePath.resolve("snapshots/\(model)/\(revision)/\(precision)", inside: root)
    }
    @discardableResult public func commitSnapshot(_ source: SnapshotManifest, replacingExisting: Bool = false) throws -> URL {
        try commitSnapshot(source, verifyFiles: true, replacingExisting: replacingExisting)
    }
    private func commitSnapshot(_ source: SnapshotManifest, verifyFiles: Bool, replacingExisting: Bool = false, getLease: Bool = false) throws -> URL {
        var manifest = source
        manifest.files = source.files.map { DerivedFile(path: $0.path, bytes: $0.bytes, sha256: $0.sha256.lowercased()) }
        if manifest.derived != nil { manifest.derived?.files = manifest.files }
        let dest = try snapshotURL(model: manifest.model, revision: manifest.revision, precision: manifest.precision)
        guard !manifest.files.isEmpty, Set(manifest.files.map(\.path)).count == manifest.files.count else { throw ManifestError.invalid("empty or duplicate snapshot files") }
        let exists = FileManager.default.fileExists(atPath: dest.path)
        if exists {
            if !replacingExisting { _ = try verifySnapshot(model: manifest.model, revision: manifest.revision, precision: manifest.precision); return dest }
            let old = try? verifySnapshot(model: manifest.model, revision: manifest.revision, precision: manifest.precision)
            if old == manifest { return dest }
            guard leases[manifest.model, default: 0] <= (getLease ? 1 : 0) else { throw ManifestError.invalid("unload \(manifest.model) before updating its weights") }
        }
        let staging = root.appendingPathComponent(".snapshot-\(UUID().uuidString)")
        do {
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
            for file in manifest.files {
                guard !StoredNames.reserved.contains(file.path) else { throw ManifestError.invalid("reserved snapshot filename") }
                let blob = try blobURL(sha256: file.sha256)
                if verifyFiles {
                    try verifyRecorded(blob, sha256: file.sha256, bytes: file.bytes, forceHash: file.bytes < 16 * 1024 * 1024)
                } else {
                    guard try FileDigest.bytes(blob) == file.bytes else { throw ManifestError.invalid("admitted blob size changed") }
                }
                let target = try SafePath.resolve(file.path, inside: staging)
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                // Hardlinks share bytes without exposing escaping symlinks in a runnable snapshot.
                try faultCheck("commit.link")
                try FileManager.default.linkItem(at: blob, to: target)
            }
            try faultCheck("commit.manifest")
            try JSONEncoder().encode(manifest).write(to: StoredNames.snapshotURL(in: staging), options: .atomic)
            if let derived = manifest.derived { try JSONEncoder().encode(derived).write(to: staging.appendingPathComponent("arsenal-derived.json"), options: .atomic) }
        } catch {
            // Nothing is published until the rename below: a failed build leaves no snapshot and no staging directory.
            try? FileManager.default.removeItem(at: staging)
            throw Self.diskFull(error)
        }
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        if exists {
            // Only the blobs the superseded snapshot held are candidates for release once the swap is done.
            let superseded = manifestHashes(under: dest)
            guard renamex_np(staging.path, dest.path, UInt32(RENAME_SWAP)) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            // Receipt precedes unlinking the superseded snapshot; shared blobs remain referenced.
            let files = try SafePath.files(in: staging).map { path in
                ["path": staging.appendingPathComponent(path).path, "bytes": try FileDigest.bytes(staging.appendingPathComponent(path))] as [String: Any]
            }
            let receipt = root.appendingPathComponent("deletions/update-\(UUID().uuidString).json")
            try FileManager.default.createDirectory(at: receipt.deletingLastPathComponent(), withIntermediateDirectories: true)
            var record: [String: Any] = ["reason": "replace_snapshot", "files": files, "verified": false]
            try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]).write(to: receipt, options: .atomic)
            try FileManager.default.removeItem(at: staging)
            record["verified"] = !FileManager.default.fileExists(atPath: staging.path)
            try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]).write(to: receipt, options: .atomic)
            try collectReleased(superseded)
        } else {
            try FileManager.default.moveItem(at: staging, to: dest)
        }
        return dest
    }
    public func verifySnapshot(model: String, revision: String, precision: String, forceHash: Bool = false) throws -> SnapshotManifest? {
        let directory = try snapshotURL(model: model, revision: revision, precision: precision)
        guard FileManager.default.fileExists(atPath: directory.path) else { return nil }
        // A snapshot written before the rename keeps its legacy manifest names (StoredNames); both are read as they are.
        let manifestName = StoredNames.snapshotURL(in: directory).lastPathComponent, derivedName = StoredNames.derivedURL(in: directory).lastPathComponent
        let manifestURL = try SafePath.resolve(manifestName, inside: directory)
        let manifest = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.model == model, manifest.revision == revision, manifest.precision == precision else { throw ManifestError.invalid("snapshot identity mismatch") }
        let listed = Set(manifest.files.map(\.path) + [manifestName] + (manifest.derived == nil ? [] : [derivedName]))
        guard !manifest.files.isEmpty, Set(manifest.files.map(\.path)).count == manifest.files.count,
            manifest.files.allSatisfy({ $0.bytes >= 0 && ModelManifest.isHex($0.sha256, count: 64) }), Set(try SafePath.files(in: directory)) == listed
        else { throw ManifestError.invalid("snapshot has missing or unlisted files") }
        for file in manifest.files {
            let largeWeight = file.bytes >= 16 * 1024 * 1024 && ["safetensors", "gguf", "bin"].contains(URL(fileURLWithPath: file.path).pathExtension)
            try verifyRecorded(SafePath.resolve(file.path, inside: directory), sha256: file.sha256, bytes: file.bytes, forceHash: forceHash || !largeWeight)
        }
        if let derived = manifest.derived {
            let persisted = try JSONDecoder().decode(DerivedManifest.self, from: Data(contentsOf: SafePath.resolve(derivedName, inside: directory)))
            guard persisted == derived else { throw ManifestError.invalid("derived provenance differs from snapshot manifest") }
        }
        return manifest
    }
    /// Core owns selection/residency policy: invoke only after a derived Load succeeds and source is not selected/in use.
    public func retainVerifiedDerived(model: String, revision: String, sourcePrecision: String, derivedPrecision: String, thresholdBytes: Int64 = 8_000_000_000) throws
        -> SourceRetentionReceipt?
    {
        guard ["fp16", "bf16"].contains(sourcePrecision), ["int8", "int4"].contains(derivedPrecision), thresholdBytes >= 0 else {
            throw ManifestError.invalid("source retention requires a 16-bit source and an int8/int4 derive")
        }
        guard let output = try verifySnapshot(model: model, revision: revision, precision: derivedPrecision), let derived = output.derived else {
            throw ManifestError.invalid("source retention requires verified derived output")
        }
        guard let source = try verifySnapshot(model: model, revision: revision, precision: sourcePrecision) else {
            return try sourceRetentionReceipt(model: model, revision: revision, sourcePrecision: sourcePrecision)
        }
        // Legacy derives without complete recovery pins stay installed rather than lose a reproducible source.
        guard let pins = derived.sourceFiles, !pins.isEmpty else { return nil }
        guard derived.schema == 1, derived.model == model, derived.precision == derivedPrecision,
            derived.sourceRevision == revision, derived.sourceDtype == sourcePrecision,
            Set(pins) == Set(source.files), pins.count == source.files.count,
            derived.sourceHashes == Dictionary(uniqueKeysWithValues: pins.map { ($0.path, $0.sha256) }),
            derived.files == output.files
        else { throw ManifestError.invalid("source recovery provenance does not match the verified source") }
        let bytes = try summedBytes(source.files)
        guard bytes > thresholdBytes, !source.files.contains(where: { reserved[$0.sha256.lowercased(), default: 0] > 0 }) else { return nil }
        let references = try referenceCounts()
        let own = Dictionary(grouping: source.files, by: { $0.sha256.lowercased() })
        var localSources: Set<String> = [], localCopyBytes: Int64 = 0, allDownloaded = true
        for (hash, files) in own {
            let record = try? JSONDecoder().decode(BlobVerificationRecord.self, from: Data(contentsOf: verificationURL(hash)))
            allDownloaded = allDownloaded && record?.sourceKind == "download"
            if let donor = record?.localSource { localSources.insert(donor); localCopyBytes += files[0].bytes }
        }
        var reclaimed: Int64 = 0
        for (hash, files) in own where references[hash] == files.count && reserved[hash, default: 0] == 0 {
            let blob = try blobURL(sha256: hash)
            if FileManager.default.fileExists(atPath: blob.path) { reclaimed += try FileDigest.bytes(blob) }
        }
        let path = try snapshotURL(model: model, revision: revision, precision: sourcePrecision)
        var receipt = SourceRetentionReceipt(
            id: UUID().uuidString, model: model, revision: revision, sourcePrecision: sourcePrecision, derivedPrecision: derivedPrecision, path: path.path, bytes: bytes,
            reclaimedBytes: reclaimed, retainedSharedBytes: max(0, try summedBytes(own.values.compactMap(\.first)) - reclaimed), sourceFiles: source.files, sourceManifest: source,
            timestamp: ISO8601DateFormatter().string(from: Date()))
        receipt.importedFromLocalCopy = !localSources.isEmpty ? true : allDownloaded ? false : nil
        receipt.localCopyBytes = localCopyBytes
        receipt.localSources = localSources.sorted()
        receipt.reclaimedBytesMeaning = "logical bytes unlinked, not physical disk space saved"
        let receiptURL = try retentionReceiptURL(model: model, revision: revision, sourcePrecision: sourcePrecision)
        try FileManager.default.createDirectory(at: receiptURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(receipt).write(to: receiptURL, options: .atomic)
        let receiptHandle = try FileHandle(forWritingTo: receiptURL); defer { try? receiptHandle.close() }; try receiptHandle.synchronize()
        // A failed receipt/log write leaves both snapshots untouched. Intent is not a claim that removal succeeded.
        try logDeletion(path: path, bytes: bytes, reason: "verified_derive_source_retention", receiptID: receipt.id)
        try FileManager.default.removeItem(at: path)
        try collectReleased(Set(source.files.map { $0.sha256.lowercased() }))
        return receipt
    }
    public func sourceRetentionReceipt(model: String, revision: String, sourcePrecision: String) throws -> SourceRetentionReceipt? {
        let source = try snapshotURL(model: model, revision: revision, precision: sourcePrecision)
        guard !FileManager.default.fileExists(atPath: source.path) else { return nil }
        let url = try retentionReceiptURL(model: model, revision: revision, sourcePrecision: sourcePrecision)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let receipt = try JSONDecoder().decode(SourceRetentionReceipt.self, from: Data(contentsOf: url))
        guard receipt.schema == 1, receipt.model == model, receipt.revision == revision, receipt.sourcePrecision == sourcePrecision, receipt.path == source.path else {
            throw ManifestError.invalid("source retention receipt identity mismatch")
        }
        return receipt
    }
    private func retentionReceiptURL(model: String, revision: String, sourcePrecision: String) throws -> URL {
        _ = try snapshotURL(model: model, revision: revision, precision: sourcePrecision)
        return try SafePath.resolve("retention/\(model)/\(revision)/\(sourcePrecision).json", inside: root)
    }
    private func summedBytes(_ files: [DerivedFile]) throws -> Int64 {
        var result: Int64 = 0
        for file in files {
            let sum = result.addingReportingOverflow(file.bytes); guard file.bytes >= 0, !sum.overflow else { throw ManifestError.invalid("invalid source size") }
            result = sum.partialValue
        }
        return result
    }
    private func logicalBytes(_ directory: URL) throws -> Int64 {
        var result: Int64 = 0
        for path in try SafePath.files(in: directory) { result += try FileDigest.bytes(SafePath.resolve(path, inside: directory)) }
        return result
    }
    func logDeletion(path: URL, bytes: Int64, reason: String, receiptID: String? = nil) throws {
        let logs = root.deletingLastPathComponent().appendingPathComponent("logs")
        let file = try SafePath.resolve("store-deletions.jsonl", inside: logs)
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let record: [String: Any] = [
            "event": "deletion_intent", "path": path.path, "bytes": bytes, "reason": reason,
            "receipt_id": receiptID ?? "", "timestamp": ISO8601DateFormatter().string(from: Date())
        ]
        var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys, .withoutEscapingSlashes]); data.append(10)
        if !FileManager.default.fileExists(atPath: file.path) {
            guard FileManager.default.createFile(atPath: file.path, contents: nil) else { throw ManifestError.invalid("cannot create deletion log") }
        }
        let handle = try FileHandle(forWritingTo: file); defer { try? handle.close() }
        try handle.seekToEnd(); try handle.write(contentsOf: data); try handle.synchronize()
    }
    public func reserve(files: [DerivedFile]) { for f in files { reserved[f.sha256.lowercased(), default: 0] += 1 } }
    public func unreserve(files: [DerivedFile]) {
        for f in files {
            let count = (reserved[f.sha256.lowercased()] ?? 0) - 1
            if count <= 0 { reserved.removeValue(forKey: f.sha256.lowercased()) } else { reserved[f.sha256.lowercased()] = count }
        }
    }
    public func retain(model: String) { leases[model, default: 0] += 1 }
    public func release(model: String) { leases[model] = max(0, (leases[model] ?? 0) - 1) }
    public func recordAssets(model: String, revisions: [String: String]) throws {
        guard ModelManifest.validID(model), revisions.allSatisfy({ ModelManifest.validID($0.key) && ModelManifest.isHex($0.value, count: 40) }) else {
            throw ManifestError.invalid("invalid asset references")
        }
        let dir = root.appendingPathComponent("asset-references"); let target = try SafePath.resolve(model + ".json", inside: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(revisions).write(to: target, options: .atomic)
    }
    /// Removes every snapshot, separate-asset snapshot, source-retention receipt and partial download of a model, each logged.
    /// `partialSHA256s` are the model's pinned hashes; a partial that a running Get reserves is kept. Refused while the model is
    /// leased (a Get, or a loaded worker). Returns the bytes freed.
    @discardableResult
    public func delete(model: String, partialSHA256s: Set<String> = []) throws -> Int64 {
        guard ModelManifest.validID(model), leases[model, default: 0] == 0 else { throw ManifestError.invalid("unload \(model) before deleting its weights") }
        let path = try SafePath.resolve("snapshots/\(model)", inside: root)
        let refDir = root.appendingPathComponent("asset-references")
        let refFile = try SafePath.resolve(model + ".json", inside: refDir)
        // An unreadable reference file (this model's or another's) means its asset snapshots cannot be proven unowned: keep them.
        let own = FileManager.default.fileExists(atPath: refFile.path) ? (try? JSONDecoder().decode([String: String].self, from: Data(contentsOf: refFile))) : [:]
        var released = manifestHashes(under: path)
        var freed = try removeLogged(path, reason: "delete_model_snapshots")
        if FileManager.default.fileExists(atPath: refFile.path) { try FileManager.default.removeItem(at: refFile) }
        var remaining: Set<String> = []; var readable = own != nil
        for url in (try? FileManager.default.contentsOfDirectory(at: refDir, includingPropertiesForKeys: nil)) ?? [] {
            _ = try SafePath.resolve(url.lastPathComponent, inside: refDir)
            guard let pairs = try? JSONDecoder().decode([String: String].self, from: Data(contentsOf: url)) else { readable = false; continue }
            for (id, revision) in pairs { remaining.insert(id + "/" + revision) }
        }
        for (id, revision) in own ?? [:] where readable && !remaining.contains(id + "/" + revision) && leases["asset-" + id, default: 0] == 0 {
            let assetDir = try SafePath.resolve("snapshots/asset-\(id)/\(revision)", inside: root)
            released.formUnion(manifestHashes(under: assetDir))
            freed += try removeLogged(assetDir, reason: "delete_last_asset_reference")
        }
        freed += try removeLogged(try SafePath.resolve("retention/\(model)", inside: root), reason: "delete_model_retention")
        freed += try removePartials(partialSHA256s)
        freed += try collectReleased(released)
        forgetWithheld(released)
        pruneEmpty(root.appendingPathComponent("snapshots"))
        return freed
    }
    /// Removes one precision's snapshots of a model (every revision), logged like any store deletion. `residentPrecision` is the
    /// precision a loaded worker holds (nil when none). Its own precision is refused; other precisions of a loaded model stay
    /// deletable, but a Get or load of the model (a lease with no resident precision named) refuses everything. A repeat finds
    /// nothing and returns 0. Returns the bytes freed.
    @discardableResult
    public func delete(model: String, precision: String, residentPrecision: String? = nil, reason: String = "delete_precision") throws -> Int64 {
        guard ModelManifest.validID(model), ModelManifest.validDtype(precision) else { throw ManifestError.invalid("invalid snapshot precision") }
        if residentPrecision == precision { throw ManifestError.invalid("unload \(model) before removing its \(precision) files") }
        if leases[model, default: 0] > 0, residentPrecision == nil {
            throw ManifestError.invalid("finish the Get or unload \(model) before removing its \(precision) files")
        }
        let revisions = try SafePath.resolve("snapshots/\(model)", inside: root)
        let directories = dirs(in: revisions)
        for revision in directories {
            let manifest = StoredNames.snapshotURL(in: revision.appendingPathComponent(precision))
            if let data = try? Data(contentsOf: manifest), let snapshot = try? JSONDecoder().decode(SnapshotManifest.self, from: data),
                snapshot.files.contains(where: { reserved[$0.sha256.lowercased(), default: 0] > 0 })
            {
                throw ManifestError.invalid("\(precision) files are in use by a download or conversion")
            }
        }
        var released = Set<String>()
        for revision in directories { released.formUnion(manifestHashes(under: try SafePath.resolve(precision, inside: revision))) }
        var freed: Int64 = 0
        for revision in directories { freed += try removeLogged(try SafePath.resolve(precision, inside: revision), reason: reason) }
        freed += try collectReleased(released)
        forgetWithheld(released)
        pruneEmpty(root.appendingPathComponent("snapshots"))
        return freed
    }
    /// A released blob that could not be unlinked now (a snapshot was unreadable, or a Get reserved it) stays for the
    /// orphan sweep, but loses its verification record unless a readable snapshot still uses it: a later Get or run then
    /// treats it as missing and asks for the download again, instead of restoring the deleted weights without the user's
    /// yes and calling it a download (core review, 2 Nov: DDColor came back after an HTTP delete).
    private func forgetWithheld(_ released: Set<String>) {
        let counts = referenceScan().counts
        for hash in released.map({ $0.lowercased() }) where ModelManifest.isHex(hash, count: 64) && counts[hash] == nil {
            guard let blob = try? blobURL(sha256: hash), FileManager.default.fileExists(atPath: blob.path), let record = try? verificationURL(hash) else { continue }
            try? FileManager.default.removeItem(at: record)
        }
    }
    /// Logical bytes of each precision on disk for a model that `offered` does not list (every revision summed).
    public func withdrawnPrecisions(model: String, offered: Set<String>) -> [String: Int64] {
        guard ModelManifest.validID(model), let revisions = try? SafePath.resolve("snapshots/\(model)", inside: root) else { return [:] }
        var result: [String: Int64] = [:]
        for revision in (try? FileManager.default.contentsOfDirectory(at: revisions, includingPropertiesForKeys: nil)) ?? [] {
            for tier in (try? FileManager.default.contentsOfDirectory(at: revision, includingPropertiesForKeys: [.isDirectoryKey])) ?? [] {
                let name = tier.lastPathComponent
                guard !offered.contains(name), (try? tier.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true, ModelManifest.validID(name) else { continue }
                result[name, default: 0] += (try? logicalBytes(tier)) ?? 0
            }
        }
        return result
    }
    public func referenceCounts() throws -> [String: Int] {
        var counts: [String: Int] = [:]
        let dir = root.appendingPathComponent("snapshots")
        if let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.isSymbolicLinkKey]) {
            for case let url as URL in e {
                if try url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true { throw ManifestError.invalid("linked snapshot rejected") }
                if StoredNames.snapshotNames.contains(url.lastPathComponent) {
                    let manifest = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: url))
                    for file in manifest.files { counts[file.sha256.lowercased(), default: 0] += 1 }
                }
            }
        }; return counts
    }
}

extension BlobStore {
    /// Explicit Get retires a cold, formerly published half snapshot after its normalized replacement is verified.
    /// A resident lease or another model's blob reference remains authoritative.
    func retireUnnormalizedSource(_ model: ModelManifest) throws {
        guard model.normalizesNativeHalf, model.weights.dtype != model.basePrecision,
            ["fp16", "bf16"].contains(model.weights.dtype), leases[model.id, default: 0] == 0,
            try verifyInstalledSnapshot(model: model, precision: model.basePrecision) != nil,
            let source = try? verifyPinnedSnapshot(model: model.id, source: model.weights, precision: model.weights.dtype)
        else { return }
        let path = try snapshotURL(model: model.id, revision: model.weights.revision, precision: model.weights.dtype)
        try logDeletion(path: path, bytes: summedBytes(source.files), reason: "native_half_normalization")
        try FileManager.default.removeItem(at: path)
        try collectReleased(Set(source.files.map { $0.sha256.lowercased() }))
    }
}
