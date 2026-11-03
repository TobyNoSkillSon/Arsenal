// SPDX-License-Identifier: MIT
import Foundation

/// Written durably before deleting a source. `bytes` is logical source size; reclaimedBytes is logical bytes unlinked, excludes shared/reserved blobs, and is NOT physical space saved.
public struct SourceRetentionReceipt: Codable, Sendable, Equatable {
    public var schema = 1
    public var id: String
    public var model: String
    public var revision: String
    public var sourcePrecision: String
    public var derivedPrecision: String
    public var path: String
    public var bytes: Int64
    public var reclaimedBytes: Int64
    public var retainedSharedBytes: Int64
    public var sourceFiles: [DerivedFile]
    /// Includes cast provenance when a 16-bit source itself came from FP32.
    public var sourceManifest: SnapshotManifest
    public var timestamp: String
    /// Optional for compatibility with existing receipts, whose physical reclaim was never measured.
    public var importedFromLocalCopy: Bool?
    public var localCopyBytes: Int64?
    public var localSources: [String]?
    public var reclaimedBytesMeaning: String?

    public var note: String {
        let uniqueBytes = Dictionary(grouping: sourceFiles, by: \.sha256).values.compactMap(\.first).reduce(Int64(0)) { $0 + $1.bytes }
        if importedFromLocalCopy == true, localCopyBytes == uniqueBytes, let sources = localSources, !sources.isEmpty {
            let names = Set(
                sources.map { path in
                    let parts = URL(fileURLWithPath: path).pathComponents
                    if parts.contains(".omlx") { return "oMLX" }
                    if parts.contains("huggingface") || parts.contains(where: { $0.hasPrefix("models--") || $0.hasPrefix("datasets--") }) { return "Hugging Face cache" }
                    return "local copy"
                }
            ).sorted().joined(separator: ", ")
            return "16-bit copy removed; Get re-imports it from \(names)"
        }
        if importedFromLocalCopy == false, reclaimedBytes > 0 {
            let size = String(format: "%.1f", Double(reclaimedBytes) / 1e9)
            return "16-bit copy removed; Get re-downloads it (\(size) GB)"
        }
        // Legacy/mixed receipts cannot promise network use or physical savings.
        return "16-bit copy removed; Get restores it."
    }
}
