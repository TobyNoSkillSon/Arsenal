// SPDX-License-Identifier: MIT
import Foundation

/// Receipt is durable before any unlink; directory inventories refuse links and special files.
enum OwnedCacheCleanup {
    private static let lock = NSRecursiveLock()
    static func remove(_ roots: [URL], support: URL, operation: String) throws {
        try lock.withLock {
            try removeWithReceipt(roots, support: support, operation: operation)
            try? pruneReceipts(support: support)
        }
    }
    /// Keep bounded durable evidence. Rotation itself gets an inventory receipt before any unlink.
    static func pruneReceipts(support: URL, limit: Int = 256) throws {
        try lock.withLock {
            let directory = support.appendingPathComponent("logs/cleanup-receipts")
            let receipts = try ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []).filter {
                $0.pathExtension == "json" && UUID(uuidString: $0.deletingPathExtension().lastPathComponent) != nil
            }.map { file in
                (
                    try SafePath.resolve(file.lastPathComponent, inside: directory),
                    try file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? .distantPast
                )
            }
            .sorted { ($0.1, $0.0.path) > ($1.1, $1.0.path) }
            guard receipts.count > max(2, limit) else { return }
            let expired = receipts.dropFirst(max(2, limit) - 1).map(\.0)
            try removeWithReceipt(Array(expired), support: support, operation: "cleanup_receipt_retention")
        }
    }
    private static func removeWithReceipt(_ roots: [URL], support: URL, operation: String) throws {
        var files: [URL] = []
        for root in roots {
            let attributes = try FileManager.default.attributesOfItem(atPath: root.path)
            if attributes[.type] as? FileAttributeType == .typeDirectory {
                files += try SafePath.files(in: root).map { try SafePath.resolve($0, inside: root) }
            } else {
                guard attributes[.type] as? FileAttributeType == .typeRegular else { throw ManifestError.invalid("cleanup refused special file") }; files.append(root)
            }
        }
        let listing = try files.map { ["path": JSONValue.string($0.path), "bytes": .number(Double(try FileDigest.bytes($0)))] }
        let bytes = listing.reduce(0.0) { $0 + ($1["bytes"]?.number ?? 0) }
        let receipts = support.appendingPathComponent("logs/cleanup-receipts")
        try FileManager.default.createDirectory(at: receipts, withIntermediateDirectories: true)
        let receipt = receipts.appendingPathComponent(UUID().uuidString + ".json")
        var record: JSONObject = ["operation": .string(operation), "files": .array(listing.map(JSONValue.object)), "bytes": .number(bytes), "verified": false]
        try WireJSON.encode(record).write(to: receipt, options: .atomic)
        let handle = try FileHandle(forWritingTo: receipt); try handle.synchronize(); try handle.close()
        for root in roots { try FileManager.default.removeItem(at: root) }
        let gone = roots.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) }
        record["verified"] = .bool(gone); record["unlinkedBytes"] = .number(gone ? bytes : 0)
        try WireJSON.encode(record).write(to: receipt, options: .atomic)
    }
}
