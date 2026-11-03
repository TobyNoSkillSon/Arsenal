// SPDX-License-Identifier: MIT
import Foundation

public enum FormatSource: String, Codable, Sendable {
    case vendor, derived
    case runnerNative = "runner-native"
}
/// A folder-defined dtype/quantization label. Pins and recipes stay in vendor/derive; this declares interpretation.
public struct ModelFormat: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var bitsPerWeight: Double
    public var storage: String
    public var method: String?
    public var profile: JSONObject?
    public var source: FormatSource
    public var runners: [String]
    public var sizeBytes: Int64?
    public var ramMB: Double?
    public init(
        id: String, bitsPerWeight: Double, source: FormatSource, runners: [String], sizeBytes: Int64? = nil, ramMB: Double? = nil, storage: String = "affine",
        method: String? = nil, profile: JSONObject? = nil
    ) {
        self.storage = storage; self.method = method; self.profile = profile
        self.id = id; self.bitsPerWeight = bitsPerWeight; self.source = source; self.runners = runners; self.sizeBytes = sizeBytes; self.ramMB = ramMB
    }
    enum CodingKeys: String, CodingKey {
        case id, storage, method, profile, source, runners
        case bitsPerWeight = "bits_per_weight", sizeBytes = "size_bytes", ramMB = "ram_mb"
    }
}
extension ModelManifest {
    public static func validDtype(_ text: String) -> Bool { text.range(of: "^[A-Za-z0-9][A-Za-z0-9_.-]*$", options: .regularExpression) != nil }
    public var formats: [ModelFormat] {
        precisions.map { dtype in
            if let declared = format?.first(where: { $0.id == dtype }) { return declared }
            // MLX affine g64 includes fp16 scale + bias: 32 bits per 64 weights.
            let overhead = derive?[dtype] != nil ? 0.5 : 0
            let bits: Double = dtype == "int4" ? 4 + overhead : dtype == "int8" ? 8 + overhead : 16
            return ModelFormat(
                id: dtype, bitsPerWeight: bits, source: conversionRecipe(for: dtype) != nil ? .derived : .vendor, runners: runner.map(\.name),
                storage: vendor?[dtype]?.format ?? (dtype == "int4" || dtype == "int8" ? "affine" : dtype))
        }
    }
    public func supports(dtype: String, runner: String) -> Bool { formats.first { $0.id == dtype }?.runners.contains(runner) == true }
    /// The precision a row starts on: the base format when some runner serves it, otherwise the first listed format that has a runner
    /// (fish-drama-3 lists its 16-bit release with no runner, so it starts on the vendor int4). The base stays listed, refused with its reason.
    public var defaultPrecision: String { formats.first { !$0.runners.isEmpty }?.id ?? basePrecision }
    /// A saved or requested precision when it is listed and has a runner; otherwise `defaultPrecision`.
    public func runnablePrecision(_ dtype: String) -> String {
        formats.first { $0.id == dtype }.map { $0.runners.isEmpty } == false ? dtype : defaultPrecision
    }
    /// Downloaded formats use their own pinned source; runner-native formats use the unchanged base weights.
    public func source(for dtype: String) -> WeightSpec? {
        if conversionRecipe(for: dtype) != nil { return nil }
        return vendor?[dtype] ?? (dtype == weights.dtype || format?.first(where: { $0.id == dtype })?.source == .runnerNative ? weights : nil)
    }
    public var sizesByDtype: [String: Int64] {
        let base = weights.bytes?.values.reduce(Int64(0), +) ?? 0
        return Dictionary(
            uniqueKeysWithValues: formats.map { f in
                let vendorSize = vendor?[f.id]?.bytes?.values.reduce(Int64(0), +)
                // Core is a forecast; a folder's exact size (including scales/metadata) wins over the bit-ratio estimate.
                let estimated = Double(base) * f.bitsPerWeight / (weights.dtype == "fp32" ? 32 : 16)
                return (f.id, f.sizeBytes ?? vendorSize ?? Int64(min(Double(Int64.max - 1024), estimated)))
            })
    }
    func formatProblems() -> [String] {
        var errors: [String] = []
        let declared = format ?? []
        if Set(declared.map(\.id)).count != declared.count { errors.append("duplicate format id") }
        for f in declared {
            if f.storage.isEmpty || !Self.validDtype(f.id) || !f.bitsPerWeight.isFinite || f.bitsPerWeight <= 0 || f.bitsPerWeight > 64 {
                errors.append("invalid format id/bits_per_weight: \(f.id)")
            }
            if Set(f.runners).count != f.runners.count || f.runners.contains(where: { name in !runner.contains { $0.name == name } }) {
                errors.append("format \(f.id) needs known, unique runners")
            }
            if (f.sizeBytes ?? 0) < 0 || !(f.ramMB ?? 0).isFinite || (f.ramMB ?? 0) < 0 { errors.append("invalid format size/RAM: \(f.id)") }
            switch f.source {
            case .vendor: if f.id != weights.dtype && vendor?[f.id] == nil { errors.append("format \(f.id) needs vendor pins") }
            case .derived: if derive?[f.id] == nil { errors.append("format \(f.id) needs a derive recipe") }
            case .runnerNative:
                if vendor?[f.id] != nil || derive?[f.id] != nil || f.id == basePrecision { errors.append("runner-native format \(f.id) must use only the base weights") }
            }
        }
        for key in Set((derive ?? [:]).keys).union((vendor ?? [:]).keys) where !["fp16", "bf16", "int8", "int4"].contains(key) && !declared.contains(where: { $0.id == key }) {
            errors.append("format declaration required for \(key)")
        }
        return errors.map { "\(id): " + $0 }
    }
}
