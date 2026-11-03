// SPDX-License-Identifier: MIT
import Foundation

/// The served Load recipe, also used by intake. Paths are resolved only here.
enum ModelLoadConfiguration {
    static func defaults(_ model: ModelManifest) throws -> ModelSettings {
        var settings = ModelSettings()
        settings.declare(model.id, metadata: try WireJSON.decode(JSONEncoder().encode(model)))
        return settings
    }

    static func defaultFast(model: ModelManifest, runner: RunnerSpec, folder: URL) throws -> Bool {
        if runner.fastExact { return true }
        return try KernelManifest.loadIfPresent(folder: folder).kernels.contains { $0.mode == "fast" }
    }

    static func config(
        model: ModelManifest, folder: URL, assets: [String: URL], chip: String, fast: Bool,
        settings: ModelSettings, maximumContext: Int? = nil
    ) throws -> JSONObject {
        var config: JSONObject = ["assets": .object(assets.mapValues { .string($0.path) })]
        if let context = model.context { config["context_limit"] = .number(Double(context)) }
        let manifest = try SafePath.resolve("kernels/manifest.toml", inside: folder)
        if FileManager.default.fileExists(atPath: manifest.path) {
            config["kernels"] = .array(
                try KernelManifest.loadIfPresent(folder: folder).resolved(chip: chip, fast: fast).map { k in
                    var object = try WireJSON.decode(JSONEncoder().encode(k))
                    object["files"] = .array(try k.files.map { .string(try SafePath.resolve($0, inside: folder.appendingPathComponent("kernels")).path) })
                    object["api"] = .string(k.resolvedAPI)
                    object["header"] = try k.header.map { .string(try SafePath.resolve($0, inside: folder.appendingPathComponent("kernels")).path) } ?? .null
                    return .object(object)
                })
        }
        config = settings.loadConfig(model.id, base: config)
        if let architecture = model.architecture { config["architecture"] = .string(architecture) }
        if model.normalizesNativeHalf { config["serving_dtype"] = .string(model.basePrecision) }
        if model.kind == "chat", let maximumContext { config = ModelContext.loadConfig(config, maximum: maximumContext) }
        return config
    }

    /// Location-independent identity of this model's folder implementation and the settings selecting it.
    /// Models with neither folder kernels nor load settings retain their source-only receipt.
    static func identity(
        model: ModelManifest, runner: RunnerSpec, folder: URL, chip: String = HostChip.current,
        fast: Bool? = nil, settings: ModelSettings? = nil
    ) throws -> String {
        let manifest = try KernelManifest.loadIfPresent(folder: folder)
        let manifestURL = folder.appendingPathComponent("kernels/manifest.toml")
        let hasManifest = FileManager.default.fileExists(atPath: manifestURL.path)
        var hashes: JSONObject = [:]
        for file in Set(manifest.kernels.flatMap { $0.files + ($0.header.map { [$0] } ?? []) }).sorted() {
            hashes[file] = .string(try FileDigest.sha256(SafePath.resolve(file, inside: folder.appendingPathComponent("kernels"))))
        }
        return try identity(
            model: model, runner: runner, manifest: manifest, manifestHash: hasManifest ? FileDigest.sha256(manifestURL) : nil,
            hashes: hashes, chip: chip, fast: fast, settings: settings)
    }

    /// A not-yet-Got row has no folder. Its signed file inventory supplies the very same content hashes.
    static func identity(
        catalog: CatalogModel, runner: RunnerSpec, kernelManifest: Data?, chip: String = HostChip.current,
        fast: Bool? = nil, settings: ModelSettings? = nil
    ) throws -> String {
        let files = Dictionary(uniqueKeysWithValues: catalog.files.map { ($0.path, $0) })
        var manifest = KernelManifest(), manifestHash: String?
        if let file = files["kernels/manifest.toml"] {
            guard let kernelManifest, kernelManifest.count == file.bytes, FileDigest.sha256(kernelManifest) == file.sha256.lowercased() else {
                throw ManifestError.invalid("catalog kernel manifest checksum rejected")
            }
            manifest = try KernelManifest.decode(kernelManifest) { files["kernels/" + $0] != nil }
            manifestHash = file.sha256.lowercased()
        }
        var hashes: JSONObject = [:]
        for path in Set(manifest.kernels.flatMap { $0.files + ($0.header.map { [$0] } ?? []) }) {
            guard let file = files["kernels/" + path] else { throw ManifestError.invalid("unlisted catalog kernel source \(path)") }
            hashes[path] = .string(file.sha256.lowercased())
        }
        return try identity(
            model: catalog.manifest, runner: runner, manifest: manifest, manifestHash: manifestHash,
            hashes: hashes, chip: chip, fast: fast, settings: settings)
    }

    private static func identity(
        model: ModelManifest, runner: RunnerSpec, manifest: KernelManifest, manifestHash: String?, hashes: JSONObject,
        chip: String, fast: Bool?, settings: ModelSettings?
    ) throws -> String {
        let settings = try settings ?? defaults(model)
        guard manifestHash != nil || model.normalizesNativeHalf || model.architecture != nil || !settings.loadValues(model.id).isEmpty else { return "" }
        let fast = fast ?? (runner.fastExact || manifest.kernels.contains { $0.mode == "fast" })
        var config: JSONObject = ["assets": .object([:])]
        if let context = model.context { config["context_limit"] = .number(Double(context)) }
        config = settings.loadConfig(model.id, base: config)
        if let architecture = model.architecture { config["architecture"] = .string(architecture) }
        if model.normalizesNativeHalf { config["serving_dtype"] = .string(model.basePrecision) }
        // Both sources use relative declarations, never checkout/install-specific absolute paths.
        config["kernels"] = .array(try manifest.resolved(chip: chip, fast: fast).map { .object(try WireJSON.decode(JSONEncoder().encode($0))) })
        return FileDigest.sha256(
            try WireJSON.encode([
                "config": .object(config), "fast": .bool(fast), "chip_family": .string(RunnerSelection.family(chip)),
                "manifest_sha256": manifestHash.map(JSONValue.string) ?? .null,
                "kernel_sha256": .object(hashes)
            ]))
    }
}
