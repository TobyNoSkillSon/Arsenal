// SPDX-License-Identifier: MIT
import Foundation

/// A one-shot decision on a verified derived load, never an idle/background cleanup pass.
enum SourceRetentionPolicy {
    static let defaultThresholdBytes: Int64 = 8_000_000_000
    static func eligible(
        sourcePrecision: String, derivedPrecision: String, selectedPrecision: String, residentPrecision: String?, verifiedDerived: Bool, sourceBytes: Int64, thresholdBytes: Int64,
        getInFlight: Bool
    ) -> Bool {
        // "Don't delete any models or 16-bit copies; they may be useful later." A source is never retired.
        false
    }
}

extension SourceRetentionPolicy {
    /// The row note for a removed 16-bit copy. When the current Get plan needs a download, it says so and how much;
    /// every tier still to be made needs that copy too (a quantized tier is never re-quantized from another).
    static func rowNote(_ receipt: SourceRetentionReceipt, downloadBytes: Int64?) -> String {
        guard let bytes = downloadBytes, bytes > 0 else { return receipt.note }
        return "16-bit copy removed while \(receipt.derivedPrecision) was in use; Get of \(receipt.sourcePrecision) or another tier downloads it again (\(size(bytes)))"
    }
    static func size(_ bytes: Int64) -> String {
        if bytes >= 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
        if bytes >= 1_000_000 { return String(format: "%.0f MB", Double(bytes) / 1e6) }
        return "\(bytes) bytes"
    }
}

extension PrecisionState {
    /// Phase of a not-installed 16-bit tier that source retention removed; the row's retentionNote explains it.
    public static let sourceRemovedPhase = "sourceRemoved"
}
