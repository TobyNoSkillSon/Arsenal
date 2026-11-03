// SPDX-License-Identifier: MIT
import Foundation

/// Resident tensor payloads for the actual component coverage of each tier, never download/file sizes.
public struct PrecisionMemoryRecipe: Codable, Sendable, Equatable {
    public var residentBytes: [String: Int64]
    public var basis: String
    enum CodingKeys: String, CodingKey { case residentBytes = "resident_bytes", basis }
    public init(residentBytes: [String: Int64], basis: String) { self.residentBytes = residentBytes; self.basis = basis }
    var problems: [String] {
        guard !basis.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !residentBytes.isEmpty,
            residentBytes.allSatisfy({ ["fp16", "bf16", "int8", "int4"].contains($0.key) && $0.value > 0 })
        else { return ["memory needs a basis and positive resident bytes per precision"] }
        return []
    }
}
