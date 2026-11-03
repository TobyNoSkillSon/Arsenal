// SPDX-License-Identifier: MIT
import Foundation

/// The decision models this worker runs, with Verdict's precision rules (Projects/Verdict/Resources/models.json and
/// VerdictHelper/Catalog.swift ModelSpec @ f8177ed). Verdict's "helper bits" are kept internally: 0 = the
/// stored half precision, else 16 / 8 / 4.
public struct DecisionsSpec: Sendable {
    public let id: String
    public let runtime: String          // "laya" | "von"
    public let context: Int
    /// Stored executable precision, after source conversion.
    public var nativeBits: Int { 16 }
    /// Tiers offered to users, highest first (models.json tiers_offered).
    public let tiers: [Int]
    public func effectiveBits(_ bits: Int) -> Int { bits == 0 ? nativeBits : bits }
    /// Helper bits a request may name (System One `bits` extension): the offered tiers, plus 0 where 16 is native.
    public var acceptedBits: Set<Int> {
        var set = Set(tiers)
        if set.contains(nativeBits) { set.insert(0) }
        return set
    }
    public var bitsMessage: String {
        let t = tiers.map(String.init)
        return "\(id) runs at " + (t.count > 1 ? t.dropLast().joined(separator: ", ") + " or " + t.last! : t.joined()) + " bits"
    }
    public func helperBits(tier: Int) -> Int { tier == nativeBits ? 0 : tier }

    /// Arsenal dtype (wire `load.dtype`) -> helper bits. FP32 is never executable.
    public func helperBits(dtype: String) -> Int? {
        switch dtype {
        case "fp16", "bf16": return helperBits(tier: 16)
        case "int8": return 8
        case "int4": return 4
        case "fp32": return nil
        default: return nil
        }
    }
    /// Helper bits -> Arsenal dtype.
    public func dtype(bits: Int) -> String {
        switch effectiveBits(bits) { case 32: return "fp32"; case 8: return "int8"; case 4: return "int4"; default: return "fp16" }
    }
}

public struct DecisionsCatalog: Sendable {
    public let entries: [DecisionsSpec]
    /// TypeSafe's hosted model: named in errors, never served.
    public let hosted: [String]
    public var local: [String] { entries.map(\.id) }
    public func spec(_ id: String) -> DecisionsSpec? { entries.first { $0.id == id } }

    public init(entries: [DecisionsSpec], hosted: [String] = ["jev"]) { self.entries = entries; self.hosted = hosted }

    /// The five Verdict models (context from each checkpoint: Laya 512/1024, Von 2048/8192), all offering 16 / 8 / 4.
    public static let verdict = DecisionsCatalog(entries: [
        DecisionsSpec(id: "laya-english", runtime: "laya", context: 512, tiers: [16, 8, 4]),
        DecisionsSpec(id: "laya-multilingual", runtime: "laya", context: 1024, tiers: [16, 8, 4]),
        DecisionsSpec(id: "laya-typed-decisions", runtime: "laya", context: 1024, tiers: [16, 8, 4]),
        DecisionsSpec(id: "von-1.2", runtime: "von", context: 8192, tiers: [16, 8, 4]),
        DecisionsSpec(id: "von-1.1", runtime: "von", context: 2048, tiers: [16, 8, 4])
    ])
}

public struct DecisionsError: Error, LocalizedError, CustomStringConvertible {
    public let code: String
    public let message: String
    public init(_ code: String, _ message: String) { self.code = code; self.message = message }
    public var description: String { message }
    public var errorDescription: String? { message }
}

/// Verdict's helper error type (kept for the ported parser's messages).
public struct ServiceError: Error, CustomStringConvertible, LocalizedError {
    public let description: String
    public init(_ description: String) { self.description = description }
    public var errorDescription: String? { description }
}
