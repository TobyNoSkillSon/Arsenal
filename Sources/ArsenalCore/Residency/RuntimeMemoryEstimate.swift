// SPDX-License-Identifier: MIT
import Foundation

public enum RuntimeMemoryEstimate {
    public static func megabytes(snapshot: SnapshotManifest, measuredMB: Double? = nil, overheadMB: Double = 768) throws -> Double {
        if let measuredMB, measuredMB.isFinite, measuredMB > 0 { return measuredMB }
        var bytes: Int64 = 0
        for file in snapshot.files {
            let next = bytes.addingReportingOverflow(file.bytes)
            guard file.bytes >= 0, !next.overflow else { throw WorkerFailure(code: "memory", message: "Installed snapshot size is invalid.") }; bytes = next.partialValue
        }
        return Double(bytes) / 1e6 + overheadMB
    }
}
