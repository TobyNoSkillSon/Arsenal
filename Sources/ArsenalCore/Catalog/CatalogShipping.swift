// SPDX-License-Identifier: MIT
import Foundation

/// General text chat stays user-added; curated multimodal and task-specific language models can ship.
public enum CatalogShipping {
    public static func includes(_ model: ModelManifest) -> Bool {
        guard model.kind == "chat" else { return true }
        if model.facets?.modalitiesIn.contains(where: { ["image", "audio", "video"].contains($0) }) == true { return true }
        return model.operations.contains { ["translate", "read_document", "extract", "ground_screen"].contains($0.name) }
    }

    /// The signed Get inventory is also the bundle inventory. Repository qualification evidence is never
    /// an installation dependency: table results already contain the verdict and quality identity that
    /// ModelStats/PrecisionEvidence check against the current runner, recipe, kernels and load settings.
    public static func files(in folder: URL, model: ModelManifest) throws -> [String] {
        var required = Set(["model.toml", "README.md", SettingsPanel.fileName, model.gate.recipe, model.bench.recipe, "gate/runner-check.json"])
        required.formUnion(model.runners.flatMap { [$0.entry, $0.lock].compactMap { $0 } })
        required.formUnion(model.operations.flatMap { [$0.input, $0.outputSchema].compactMap { $0 } }.filter { !$0.hasPrefix("kinds/") })
        if let kindFile = model.kindFile { required.insert(kindFile) }
        let kernels = try KernelManifest.loadIfPresent(folder: folder)
        required.insert("kernels/manifest.toml")
        required.formUnion(kernels.kernels.flatMap { $0.files + ($0.header.map { [$0] } ?? []) }.map { "kernels/" + $0 })
        // A folder-local Swift entry needs its whole closure: LocalSwiftRunner's build receipt covers every file under runners/swift/.
        let localSwiftClosure = model.runners.contains { $0.entry != nil }
        return try SafePath.files(in: folder).filter { path in
            if path.hasPrefix("gate/evidence/") || path.hasPrefix("results/intake/") || path.hasPrefix(LocalPrecisionResults.prefix) { return false }
            if required.contains(path) { return true }
            // Keep complete runner closures (including source/build receipts for local Swift admission)
            // and schema resources. Test and fixture trees are development evidence, not runtime files.
            if path.hasPrefix("runners/") {
                return localSwiftClosure || !path.split(separator: "/").contains { ["Tests", "fixtures"].contains(String($0)) }
            }
            // Shared kinds have their own complete signed inventory.
            if path.hasPrefix("ops/") || (model.kindFile != nil && path.hasPrefix("kinds/")) { return true }
            if path.hasPrefix("log/") { return path.hasSuffix(".md") }
            return path.hasPrefix("results/") && path.split(separator: "/").count == 2 && (path.hasSuffix(".json") || path.hasSuffix(".md"))
        }
    }
}
