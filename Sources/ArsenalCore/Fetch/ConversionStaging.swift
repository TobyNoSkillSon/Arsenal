// SPDX-License-Identifier: MIT
import Foundation
import Darwin

/// A kernel-held lease survives task cancellation and vanishes on process death. Never reclaim a live Get.
final class ConversionStaging {
    let root: URL
    private let fd: Int32
    init(_ root: URL) throws {
        self.root = root
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let handle = open(root.appendingPathComponent(".owner-lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
        guard handle >= 0 else { throw ManifestError.invalid("Cannot create conversion lease") }
        guard flock(handle, LOCK_EX | LOCK_NB) == 0 else { close(handle); throw ManifestError.invalid("Conversion staging is in use") }
        fd = handle
    }
    deinit { close(fd); try? FileManager.default.removeItem(at: root) }
    /// Staging directories this module owns: `.cast-` and `.derive-` conversions, and `.asset-` incorporated-asset inputs.
    static let prefixes = [".cast-", ".derive-", ".asset-"]
    static func recover(in weights: URL) throws {
        guard FileManager.default.fileExists(atPath: weights.path) else { return }
        for dir in try FileManager.default.contentsOfDirectory(at: weights, includingPropertiesForKeys: [.contentModificationDateKey, .isSymbolicLinkKey, .isDirectoryKey])
        where prefixes.contains(where: { dir.lastPathComponent.hasPrefix($0) }) {
            guard let prefix = prefixes.first(where: { dir.lastPathComponent.hasPrefix($0) }),
                UUID(uuidString: String(dir.lastPathComponent.dropFirst(prefix.count))) != nil
            else { continue }
            let attrs = try dir.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
            guard attrs.isDirectory == true, attrs.isSymbolicLink != true else { continue }
            let marker = dir.appendingPathComponent(".owner-lock"), lease = SiblingLease.leaseURL(for: dir)
            let owner: URL
            if FileManager.default.fileExists(atPath: marker.path) {
                owner = marker
            } else if FileManager.default.fileExists(atPath: lease.path) {
                // A converter-read directory keeps its lease beside it: the lock alone decides, no age needed.
                owner = lease
            } else {
                // Old versions wrote no lease. Reclaim only after their maximum request lifetime has passed.
                let date = try dir.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? Date()
                if Date().timeIntervalSince(date) < 86_400 { continue }
                owner = marker
            }
            let fd = open(owner.path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
            guard fd >= 0 else { continue }
            defer { close(fd) }
            if flock(fd, LOCK_EX | LOCK_NB) == 0 {
                try FileManager.default.removeItem(at: dir)
                try? FileManager.default.removeItem(at: lease)
            }
        }
    }
    /// Ownership of a staging directory whose own contents a converter reads (a normalized source or an asset input):
    /// the lease sits beside the directory, so no file is added to what the converter sees.
    final class SiblingLease {
        static func leaseURL(for dir: URL) -> URL { dir.deletingLastPathComponent().appendingPathComponent(dir.lastPathComponent + ".lease") }
        private let fd: Int32
        private let url: URL
        // Properties are assigned only after success: a failed init must not reach deinit, which would remove a live lease.
        init(_ dir: URL) throws {
            let url = SiblingLease.leaseURL(for: dir)
            let handle = open(url.path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
            guard handle >= 0 else { throw ManifestError.invalid("Cannot create staging lease") }
            guard flock(handle, LOCK_EX | LOCK_NB) == 0 else { close(handle); throw ManifestError.invalid("Staging is in use") }
            self.url = url; fd = handle
        }
        deinit { close(fd); try? FileManager.default.removeItem(at: url) }
    }
    static func preflight(in root: URL, sourceBytes: Int64) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let attrs = try FileManager.default.attributesOfFileSystem(forPath: root.path)
        // Conservative source-size preflight; the writer also checks actual tensor bytes before every append.
        guard let free = attrs[.systemFreeSize] as? NSNumber, sourceBytes <= (Int64.max - (64 << 20)) / 2,
            free.int64Value >= 2 * sourceBytes + (64 << 20)
        else { throw ManifestError.invalid("Insufficient disk capacity for source and converted file") }
    }
}
