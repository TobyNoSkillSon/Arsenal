// SPDX-License-Identifier: MIT
import Foundation

extension ModelManifest {
    /// Network bytes on a fresh install, including required shared assets. Derived precisions download only the source.
    /// Nil means a contributor has not pinned every file; never present unknown size as zero.
    public func downloadBytes(precision: String) -> Int64? {
        guard precisions.contains(precision) else { return nil }
        let primary = vendor?[precision] ?? weights
        let sources = [primary] + (asset ?? []).filter { assets == nil || assets!.contains($0.id) }.map(\.weights)
        var sizes: [String: Int64] = [:]
        for source in sources {
            for file in source.files {
                guard let hash = source.sha256?[file], ModelManifest.isHex(hash, count: 64), let bytes = source.bytes?[file], bytes >= 0 else { return nil }
                if let old = sizes[hash], old != bytes { return nil }; sizes[hash] = bytes
            }
        }
        var total: Int64 = 0
        for size in sizes.values { let sum = total.addingReportingOverflow(size); if sum.overflow { return nil }; total = sum.partialValue }
        return total
    }
    public var downloadBytesByPrecision: [String: Int64] {
        Dictionary(uniqueKeysWithValues: precisions.compactMap { p in downloadBytes(precision: p).map { (p, $0) } })
    }
}
extension BlobStore {
    /// Remaining bytes ignores already verified blobs. Call away from the UI thread; hashing belongs to Get/work jobs.
    public func remainingDownloadBytes(files: [DerivedFile]) throws -> Int64 {
        var hashes: Set<String> = []; var total: Int64 = 0
        for file in files where hashes.insert(file.sha256).inserted {
            if try !isVerifiedPresent(sha256: file.sha256, bytes: file.bytes) { total += file.bytes }
        }; return total
    }
}
