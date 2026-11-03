// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import Darwin

/// The store half of the weights collection (`RealModelsService.collectWeights`): interrupted downloads and snapshot files
/// that are copies of their blob instead of hard links to it.
extension BlobStore {
    /// Removes interrupted downloads (`downloads/*`), each logged: every quarantined copy, every partial of a hash no current
    /// model pins, and pinned partials untouched since `resumableSince`. A recent `<sha>.part` and its journal of a pinned hash stay
    /// together so the next Get resumes them; one a running Get reserves is never touched. Returns the bytes freed.
    func removeInterruptedDownloads(pinned: Set<String>, resumableSince: Date) throws -> Int64 {
        let downloads = root.appendingPathComponent("downloads"); var freed: Int64 = 0
        let pins = Set(pinned.map { $0.lowercased() })
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: downloads.path)) ?? []).sorted()
        func modified(_ name: String) -> Date? {
            try? downloads.appendingPathComponent(name).resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        }
        for name in names {
            let hash = String(name.prefix(64)).lowercased()
            guard ModelManifest.isHex(hash, count: 64), reserved[hash, default: 0] == 0 else { continue }
            if name == hash + ".part" || name == hash + ".json", pins.contains(hash) {
                // The pair is judged by its newest member, so a partial and its journal go or stay together.
                let newest = [modified(hash + ".part"), modified(hash + ".json")].compactMap { $0 }.max() ?? .distantPast
                if newest >= resumableSince { continue }
            }
            freed += try removeLogged(try SafePath.resolve(name, inside: downloads), reason: "delete_interrupted_download")
        }
        return freed
    }
    /// Makes every snapshot file a hard link to its blob. A file that is a separate copy is replaced by a link to the blob when the
    /// blob exists, or is linked in as the blob when it does not (verified on its next use, so launch never hashes weights).
    /// Leased models are skipped; a file that cannot be relinked is counted and left. Returns the bytes freed and the failures.
    func relinkSnapshotCopies() -> (freed: Int64, failures: Int) {
        var freed: Int64 = 0, failures = 0
        // A link staged by an interrupted run is a second name for a blob; it never sits inside a snapshot.
        for name in ((try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []) where name.hasPrefix(".relink-") {
            try? FileManager.default.removeItem(at: root.appendingPathComponent(name))
        }
        let snapshots = root.appendingPathComponent("snapshots")
        for model in dirs(in: snapshots) where leases[model.lastPathComponent, default: 0] == 0 {
            for manifestURL in snapshotManifests(under: model) {
                guard let data = try? Data(contentsOf: manifestURL), let manifest = try? JSONDecoder().decode(SnapshotManifest.self, from: data) else { continue }
                let directory = manifestURL.deletingLastPathComponent()
                for file in manifest.files {
                    do { freed += try relink(file, in: directory) } catch { failures += 1 }
                }
            }
        }
        return (freed, failures)
    }
    private func relink(_ file: DerivedFile, in directory: URL) throws -> Int64 {
        let target = try SafePath.resolve(file.path, inside: directory), blob = try blobURL(sha256: file.sha256)
        var t = stat(), b = stat()
        guard lstat(target.path, &t) == 0, t.st_mode & S_IFMT == S_IFREG, t.st_size == file.bytes else { return 0 }
        guard lstat(blob.path, &b) == 0 else {
            try FileManager.default.createDirectory(at: blob.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.linkItem(at: target, to: blob)
            return 0
        }
        guard b.st_ino != t.st_ino, b.st_size == file.bytes else { return 0 }
        let staged = root.appendingPathComponent(".relink-\(UUID().uuidString)")
        try FileManager.default.linkItem(at: blob, to: staged)
        guard rename(staged.path, target.path) == 0 else {
            try? FileManager.default.removeItem(at: staged); throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        try logDeletion(path: target, bytes: Int64(t.st_size), reason: "relink_snapshot_copy")
        return t.st_nlink == 1 ? Int64(t.st_size) : 0
    }
    /// Appends one summary record of a collection to the deletion log.
    func logCollection(_ summary: [String: Any]) throws {
        let logs = root.deletingLastPathComponent().appendingPathComponent("logs")
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let file = try SafePath.resolve("store-deletions.jsonl", inside: logs)
        var record = summary; record["event"] = "weights_collection"; record["timestamp"] = ISO8601DateFormatter().string(from: Date())
        var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys, .withoutEscapingSlashes]); data.append(10)
        if !FileManager.default.fileExists(atPath: file.path) { FileManager.default.createFile(atPath: file.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: file); defer { try? handle.close() }
        try handle.seekToEnd(); try handle.write(contentsOf: data)
    }
    private func snapshotManifests(under url: URL) -> [URL] {
        FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL }
            .filter { StoredNames.snapshotNames.contains($0.lastPathComponent) } ?? []
    }
}
