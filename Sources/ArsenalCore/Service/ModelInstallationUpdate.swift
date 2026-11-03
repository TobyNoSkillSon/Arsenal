// SPDX-License-Identifier: MIT
import Foundation

extension ModelManifest {
    /// Identity of the content needed for one precision; presentation/key ordering is irrelevant.
    func hasSameInstallationPins(as old: ModelManifest, precision: String) -> Bool {
        func same(_ a: WeightSpec, _ b: WeightSpec) -> Bool {
            a.repo == b.repo && a.repoType == b.repoType && a.revision == b.revision && a.dtype == b.dtype
                && Set(a.files) == Set(b.files) && a.sha256 == b.sha256 && a.bytes == b.bytes
        }
        guard old.precisions.contains(precision) else { return false }
        if let a = vendor?[precision], let b = old.vendor?[precision] {
            guard same(a, b) else { return false }
        } else {
            guard vendor?[precision] == nil, old.vendor?[precision] == nil, same(weights, old.weights),
                conversionRecipe(for: precision) == old.conversionRecipe(for: precision), conversionRecipe(for: basePrecision) == old.conversionRecipe(for: old.basePrecision),
                nativeHalfNormalization == old.nativeHalfNormalization
            else { return false }
        }
        let a = (asset ?? []).filter { assets == nil || assets!.contains($0.id) }
        let b = (old.asset ?? []).filter { old.assets == nil || old.assets!.contains($0.id) }
        return a.count == b.count && a.allSatisfy { item in b.contains { $0.id == item.id && same(item.weights, $0.weights) } }
    }
}

extension RealModelsService {
    func updateMessage(_ id: String, _ precision: String) -> String {
        let bytes = modelRows[id]?.downloadBytes[precision] ?? 0
        let size = bytes >= 1_000_000_000 ? String(format: "%.1f GB", Double(bytes) / 1e9) : String(format: "%.1f MB", Double(bytes) / 1e6)
        let transfer = bytes > 0 ? " · \(size)" : ""
        return "Update available · Get\(transfer); use \(MCPNames.prefix)download(model: \(id), dtype: \(precision))."
    }
}
