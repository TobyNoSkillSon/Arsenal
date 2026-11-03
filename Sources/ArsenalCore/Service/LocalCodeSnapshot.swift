// SPDX-License-Identifier: MIT
import Foundation

/// Run the reviewed bytes, not the mutable folder the user dropped. Changes create a new consent/snapshot identity.
enum LocalCodeSnapshot {
    static func install(_ local: LocalModelFolder, support: URL) throws -> URL {
        let parent = support.appendingPathComponent("state/local-folders/\(local.contentDigest)")
        let destination = try SafePath.resolve(local.manifest.id, inside: parent)
        func digest(_ folder: URL) throws -> String {
            let files = try SafePath.files(in: folder).map { path in
                CatalogFile(path: path, sha256: try FileDigest.sha256(SafePath.resolve(path, inside: folder)), bytes: try FileDigest.bytes(SafePath.resolve(path, inside: folder)))
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            return FileDigest.sha256(try encoder.encode(files))
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            guard try digest(destination) == local.contentDigest else {
                throw WorkerFailure(code: "integrity", message: "Local model snapshot changed; restore the dropped folder and approve it again.")
            }
            return destination
        }
        let stageParent = support.appendingPathComponent("state/local-folders/.stage-\(UUID().uuidString)")
        let staging = stageParent.appendingPathComponent(local.manifest.id)
        defer {
            if FileManager.default.fileExists(atPath: stageParent.path) { try? OwnedCacheCleanup.remove([stageParent], support: support, operation: "local_snapshot_staging") }
        }
        for path in try SafePath.files(in: local.directory) {
            let source = try SafePath.resolve(path, inside: local.directory)
            let target = try SafePath.resolve(path, inside: staging)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: source, to: target)
            // Preserve an executable's mode, but never permit code writes from the runner.
            let mode = (try FileManager.default.attributesOfItem(atPath: source.path)[.posixPermissions] as? NSNumber)?.intValue ?? 0o644
            try FileManager.default.setAttributes([.posixPermissions: mode & 0o555], ofItemAtPath: target.path)
        }
        guard try digest(staging) == local.contentDigest else {
            throw WorkerFailure(code: "code_consent", message: "The local folder changed after approval; review and approve its new contents.")
        }
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: staging, to: destination)
        return destination
    }
    static func prune(support: URL, protecting digests: Set<String>, limit: Int = 8) throws {
        let root = support.appendingPathComponent("state/local-folders")
        let candidates = ((try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []).filter {
            ModelManifest.isHex($0.lastPathComponent, count: 64) && !digests.contains($0.lastPathComponent)
        }
        let sorted = candidates.sorted {
            ((try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
                > ((try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
        }
        for candidate in sorted.dropFirst(max(0, limit)) {
            _ = try SafePath.resolve(candidate.lastPathComponent, inside: root); try OwnedCacheCleanup.remove([candidate], support: support, operation: "local_snapshot_retention")
        }
    }
}
