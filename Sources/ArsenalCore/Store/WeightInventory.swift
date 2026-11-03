// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// Inventory and orphan cleanup for the weights store. Under the store root (`support/weights`) the store keeps, for a model:
/// blobs (`blobs/<sha256>`, shared by content), blob verification records (`verified/<sha256>.json`), precision snapshots
/// (`snapshots/<model>/<revision>/<precision>/`), separate-asset snapshots (`snapshots/asset-<id>/<revision>/`) with their
/// references (`asset-references/<model>.json`), source-retention receipts (`retention/<model>/<revision>/<precision>.json`),
/// partial downloads (`downloads/<sha256>.part` and `.json`), and two transient kinds that no model owns: local-clone staging
/// (`imports/*.part`) and snapshot staging (`.snapshot-*`). The deletion log is `support/logs/store-deletions.jsonl`.
extension BlobStore {
    /// Model IDs with a precision snapshot or an asset reference on disk. Separate-asset snapshots are not models.
    public func onDiskModelIDs() throws -> [String] {
        var ids = Set<String>()
        for name in names(in: root.appendingPathComponent("snapshots")) where !name.hasPrefix("asset-") && ModelManifest.validID(name) { ids.insert(name) }
        for name in names(in: root.appendingPathComponent("asset-references")) where name.hasSuffix(".json") {
            let id = String(name.dropLast(".json".count))
            if ModelManifest.validID(id) { ids.insert(id) }
        }
        return ids.sorted()
    }
    /// Logical bytes of each precision a model has on disk, summed across its revisions.
    public func precisionBytes(model: String) throws -> [String: Int64] {
        guard ModelManifest.validID(model) else { return [:] }
        var result: [String: Int64] = [:]
        for revision in dirs(in: try SafePath.resolve("snapshots/\(model)", inside: root)) {
            for tier in dirs(in: revision) where ModelManifest.validDtype(tier.lastPathComponent) {
                result[tier.lastPathComponent, default: 0] += measure(tier).logical
            }
        }
        return result
    }
    /// The separate assets a model references: asset ID to the pinned revision.
    public func assetReferences(model: String) throws -> [String: String] {
        guard ModelManifest.validID(model) else { return [:] }
        let file = try SafePath.resolve(model + ".json", inside: root.appendingPathComponent("asset-references"))
        guard FileManager.default.fileExists(atPath: file.path) else { return [:] }
        return try JSONDecoder().decode([String: String].self, from: Data(contentsOf: file))
    }
    /// Logical bytes of the separate-asset snapshots a model references. An asset two models share counts for both.
    public func assetCacheBytes(model: String) throws -> Int64 {
        var total: Int64 = 0
        for (id, revision) in try assetReferences(model: model) where ModelManifest.validID(id) && ModelManifest.isHex(revision, count: 40) {
            total += measure(try SafePath.resolve("snapshots/asset-\(id)/\(revision)", inside: root)).logical
        }
        return total
    }
    /// Bytes of the partial downloads of these pinned hashes: the `.part` file, its journal and any quarantined copy. A
    /// download that received nothing (its journal alone, as a Get refused at a gated repository leaves it) holds no
    /// weights and counts 0, so its model is not listed with "0 KB" to delete.
    public func partialBytes(sha256s: Set<String>) -> Int64 {
        let downloads = root.appendingPathComponent("downloads"); let entries = names(in: downloads); var total: Int64 = 0
        for hash in sha256s.map({ $0.lowercased() }) where ModelManifest.isHex(hash, count: 64) {
            let files = entries.filter { $0.hasPrefix(hash + ".") }.map { (name: $0, bytes: fileBytes(downloads.appendingPathComponent($0))) }
            guard files.contains(where: { $0.name != hash + ".json" && $0.bytes > 0 }) else { continue }
            total += files.reduce(0) { $0 + $1.bytes }
        }
        return total
    }
    /// Removes the partial downloads of these hashes, except any a running Get reserves. Logged.
    func removePartials(_ hashes: Set<String>) throws -> Int64 {
        let downloads = root.appendingPathComponent("downloads"); let entries = names(in: downloads); var freed: Int64 = 0
        for hash in hashes.map({ $0.lowercased() }).sorted() where ModelManifest.isHex(hash, count: 64) && reserved[hash, default: 0] == 0 {
            for entry in entries where entry.hasPrefix(hash + ".") {
                freed += try removeLogged(try SafePath.resolve(entry, inside: downloads), reason: "delete_partial_download")
            }
        }
        return freed
    }

    struct OrphanPlan {
        var assetDirectories: [URL] = []
        var staging: [URL] = []
        var imports: [URL] = []
        var blobs: [(name: String, url: URL, bytes: Int64)] = []
        var records: [URL] = []
    }
    /// What `deleteOrphans` would remove, decided without touching the disk. An unowned asset snapshot is one no model
    /// references, unless a Get holds its lease (it commits the reference last); a blob is an orphan when no snapshot that
    /// survives the cleanup references it and no Get reserves it. `keepAssets` ("id/revision") are assets a current model
    /// declares; they stay even without a reference file (installs older than asset references have none).
    func orphanPlan(keepAssets: Set<String> = []) -> OrphanPlan {
        var plan = OrphanPlan()
        let (referenced, unreadableReferences) = assetReferenceScan()
        let scan = referenceScan()
        // A record that cannot be read might be what refers to a blob, so while any exists nothing is reported as an orphan.
        guard unreadableReferences + scan.unreadable == 0 else { return plan }
        for asset in dirs(in: root.appendingPathComponent("snapshots")) where asset.lastPathComponent.hasPrefix("asset-") {
            let id = String(asset.lastPathComponent.dropFirst("asset-".count))
            for revision in dirs(in: asset) where leases["asset-" + id, default: 0] == 0 {
                let key = id + "/" + revision.lastPathComponent
                guard !referenced.contains(key), !keepAssets.contains(key) else { continue }
                plan.assetDirectories.append(revision)
            }
        }
        var refs = scan.counts
        for directory in plan.assetDirectories {
            for file in treeFiles(directory) where StoredNames.snapshotNames.contains(file.lastPathComponent) {
                for entry in (try? JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: file)))?.files ?? [] {
                    refs[entry.sha256.lowercased(), default: 0] -= 1
                }
            }
        }
        let blobDirectory = root.appendingPathComponent("blobs")
        for name in names(in: blobDirectory) where ModelManifest.isHex(name, count: 64) && (refs[name] ?? 0) <= 0 && reserved[name, default: 0] == 0 {
            let url = blobDirectory.appendingPathComponent(name)
            plan.blobs.append((name, url, fileBytes(url)))
        }
        let orphaned = Set(plan.blobs.map(\.name))
        plan.staging = names(in: root).filter { $0.hasPrefix(".snapshot-") }.map { root.appendingPathComponent($0) }
        plan.imports = names(in: root.appendingPathComponent("imports")).map { root.appendingPathComponent("imports/" + $0) }
        let verified = root.appendingPathComponent("verified")
        plan.records = names(in: verified).filter { $0.hasSuffix(".json") }.compactMap { name in
            let hash = String(name.dropLast(".json".count))
            guard orphaned.contains(hash) || !FileManager.default.fileExists(atPath: blobDirectory.appendingPathComponent(hash).path) else { return nil }
            return verified.appendingPathComponent(name)
        }
        return plan
    }
    /// Bytes of weights nothing refers to: orphan blobs, unowned asset snapshots, stale snapshot staging and stale imports.
    func orphanBytes() -> Int64 {
        let plan = orphanPlan()
        var total = plan.blobs.reduce(0) { $0 + $1.bytes }
        for directory in plan.assetDirectories + plan.staging + plan.imports { total += measure(directory).freed }
        return total
    }
    /// Removes what `orphanPlan` found, each removal logged. Returns the bytes freed (the figure `orphanBytes` reported).
    func deleteOrphans(keepAssets: Set<String> = []) throws -> Int64 {
        let plan = orphanPlan(keepAssets: keepAssets)
        var freed: Int64 = 0
        for directory in plan.assetDirectories { freed += try removeLogged(directory, reason: "delete_orphan_asset") }
        for directory in plan.staging { freed += try removeLogged(directory, reason: "delete_stale_snapshot_staging") }
        for file in plan.imports { freed += try removeLogged(file, reason: "delete_stale_import") }
        for blob in plan.blobs {
            try logDeletion(path: blob.url, bytes: blob.bytes, reason: "delete_orphan_blob")
            try FileManager.default.removeItem(at: blob.url)
            freed += blob.bytes
        }
        for record in plan.records { try? FileManager.default.removeItem(at: record) }
        pruneEmpty(root.appendingPathComponent("snapshots"))
        return freed
    }

    /// Logical bytes of the regular files under `url`, and the bytes removing them frees now: a file with no other hard link.
    /// A snapshot file shares its blob's inode, so its bytes come back only when the blob goes.
    func measure(_ url: URL) -> (logical: Int64, freed: Int64) {
        var logical: Int64 = 0, freed: Int64 = 0
        for file in treeFiles(url) {
            var status = stat()
            guard lstat(file.path, &status) == 0, status.st_mode & S_IFMT == S_IFREG else { continue }
            logical += Int64(status.st_size)
            if status.st_nlink == 1 { freed += Int64(status.st_size) }
        }
        return (logical, freed)
    }
    /// Logs, then removes, a store path. Absent paths cost nothing. Returns the bytes the removal frees.
    @discardableResult
    func removeLogged(_ url: URL, reason: String) throws -> Int64 {
        guard FileManager.default.fileExists(atPath: url.path) else { return 0 }
        let measured = measure(url)
        try logDeletion(path: url, bytes: measured.logical, reason: reason)
        try FileManager.default.removeItem(at: url)
        return measured.freed
    }
    /// Removes directories left empty under `directory`, deepest first; `directory` itself goes when it empties too.
    func pruneEmpty(_ directory: URL) {
        for child in dirs(in: directory) { pruneEmpty(child) }
        if directory != root, names(in: directory).isEmpty { try? FileManager.default.removeItem(at: directory) }
    }

    /// The asset references every readable reference file names ("asset/revision"), and how many files could not be read.
    func assetReferenceScan() -> (referenced: Set<String>, unreadable: Int) {
        let references = root.appendingPathComponent("asset-references"); var result = Set<String>(); var unreadable = 0
        for name in names(in: references) where name.hasSuffix(".json") {
            guard let data = try? Data(contentsOf: references.appendingPathComponent(name)),
                let pairs = try? JSONDecoder().decode([String: String].self, from: data)
            else { unreadable += 1; continue }
            for (id, revision) in pairs { result.insert(id + "/" + revision) }
        }
        return (result, unreadable)
    }
    /// Snapshot manifests and asset reference files that cannot be read. While any exist, orphan cleanup reports and deletes nothing.
    func unreadableRecords() -> Int { assetReferenceScan().unreadable + referenceScan().unreadable }
    private func treeFiles(_ url: URL) -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return [] }
        guard isDirectory.boolValue else { return [url] }
        return FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL } ?? []
    }
    private func names(in directory: URL) -> [String] { (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? [] }
    func dirs(in directory: URL) -> [URL] {
        let children = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])) ?? []
        return children.filter { child in
            let values = try? child.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            return values?.isDirectory == true && values?.isSymbolicLink != true
        }
    }
    private func fileBytes(_ url: URL) -> Int64 { (try? FileDigest.bytes(url)) ?? 0 }

    /// The blob hashes the readable snapshot manifests under `url` reference. An unreadable manifest contributes nothing here.
    func manifestHashes(under url: URL) -> Set<String> {
        var result = Set<String>()
        for file in treeFiles(url) where StoredNames.snapshotNames.contains(file.lastPathComponent) {
            guard let data = try? Data(contentsOf: file), let manifest = try? JSONDecoder().decode(SnapshotManifest.self, from: data) else { continue }
            for entry in manifest.files { result.insert(entry.sha256.lowercased()) }
        }
        return result
    }
    /// Reference counts from every readable snapshot manifest, and how many entries could not be read (undecodable or linked).
    func referenceScan() -> (counts: [String: Int], unreadable: Int) {
        var counts: [String: Int] = [:]; var unreadable = 0
        guard let walk = FileManager.default.enumerator(at: root.appendingPathComponent("snapshots"), includingPropertiesForKeys: [.isSymbolicLinkKey]) else {
            return (counts, 0)
        }
        for case let url as URL in walk {
            if (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true { unreadable += 1; continue }
            guard StoredNames.snapshotNames.contains(url.lastPathComponent) else { continue }
            guard let data = try? Data(contentsOf: url), let manifest = try? JSONDecoder().decode(SnapshotManifest.self, from: data) else { unreadable += 1; continue }
            for file in manifest.files { counts[file.sha256.lowercased(), default: 0] += 1 }
        }
        return (counts, unreadable)
    }
    /// Unlinks the blobs a deletion released, but only those no readable snapshot still references and no Get reserves, with
    /// their verification records. Blobs nothing released are never touched: a sweep of unreferenced blobs is `deleteOrphans`.
    /// If any snapshot cannot be read, a released blob might be its, so nothing is unlinked and the blob remains as an orphan.
    @discardableResult
    func collectReleased(_ hashes: Set<String>) throws -> Int64 {
        let scan = referenceScan()
        guard scan.unreadable == 0 else { return 0 }
        var freed: Int64 = 0
        for hash in hashes.map({ $0.lowercased() }).sorted() where ModelManifest.isHex(hash, count: 64) && scan.counts[hash] == nil && reserved[hash, default: 0] == 0 {
            let blob = try blobURL(sha256: hash)
            guard FileManager.default.fileExists(atPath: blob.path) else { continue }
            let bytes = fileBytes(blob)
            try logDeletion(path: blob, bytes: bytes, reason: "collect_released_blob")
            try FileManager.default.removeItem(at: blob)
            try? FileManager.default.removeItem(at: verificationURL(hash))
            freed += bytes
        }
        return freed
    }
}
