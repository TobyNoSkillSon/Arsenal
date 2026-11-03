// SPDX-License-Identifier: MIT
import Foundation

/// Storage and serving agree. The source's weights.dtype still describes its pinned download bytes.
public struct NativeHalfPolicy: Codable, Sendable, Equatable {
    public var dtype: String
    public var reason: String?
    public var evidence: String?
    /// `"raw"`: Get hands the model's own `cast:` recipe the verified pinned files unchanged instead of normalizing
    /// them to half first, because the recipe must transform full-range values before the half cast (LaMa folds
    /// BatchNorm running variances past FP16's range into its convolutions). The recipe still stores the selected half.
    public var source: String?
    public init(dtype: String, reason: String? = nil, evidence: String? = nil, source: String? = nil) {
        self.dtype = dtype; self.reason = reason; self.evidence = evidence; self.source = source
    }
    public static let recipe = "cast:native-half-v1"
    /// The native families whose weights are normalized at Get and may declare a [precision] block. Scripts repeat this
    /// list (scripts/model-check.py, scripts/pkg-lib.sh); NormalizedFamilyTests keeps the three equal.
    public static let normalizedWorkers = ["language", "embed", "audio", "music", "restoration", "perception", "geometry", "qwenimage"]
}

extension ModelManifest {
    public var normalizesNativeHalf: Bool {
        precision != nil || runners.contains { NativeHalfPolicy.normalizedWorkers.contains($0.worker ?? "") }
    }
    /// The model's own cast receives the raw pinned source (`[precision] source = "raw"`).
    public var castsRawSource: Bool { precision?.source == "raw" }
    public var nativeHalfNormalization: String? {
        normalizesNativeHalf ? (castsRawSource ? "raw-cast-v1:" : "native-half-v1:") + basePrecision : nil
    }
    /// Checks for `[precision] source = "raw"`: float32 source, the model's own cast recipe, a reason and evidence.
    func rawSourceProblems(_ policy: NativeHalfPolicy, folder: URL) -> [String] {
        guard let source = policy.source else { return [] }
        var problems: [String] = []
        if source != "raw" { problems.append("precision.source must be \"raw\"") }
        if weights.dtype != "fp32" { problems.append("precision.source = \"raw\" is for float32 sources") }
        if derive?[policy.dtype]?.hasPrefix("cast:") != true || derive?[policy.dtype] == NativeHalfPolicy.recipe {
            problems.append("a raw source needs the model's own cast: recipe")
        }
        if policy.reason?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false { problems.append("a raw source needs precision.reason") }
        let evidence = policy.evidence.flatMap { SafePath.isRelative($0) ? try? SafePath.resolve($0, inside: folder) : nil }
        if evidence.map({ FileManager.default.fileExists(atPath: $0.path) }) != true { problems.append("a raw source needs a relative precision.evidence file") }
        return problems
    }
    public func conversionRecipe(for dtype: String) -> String? {
        if let declared = derive?[dtype] { return declared }
        return normalizesNativeHalf && dtype == basePrecision ? NativeHalfPolicy.recipe : nil
    }
}
