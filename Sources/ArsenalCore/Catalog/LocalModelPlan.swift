// SPDX-License-Identifier: MIT
import Foundation

/// The rules behind `Arsenal catalog add-local`: what a folder built from a Hugging Face repository may claim.
/// Pure decisions only (no network, no writes), so every refusal can be tested without a download.
public enum LocalModelPlan {
    /// Licences Arsenal treats as open: commercial use is allowed with notice or attribution at most.
    /// Any other licence, including a missing one, needs the user's yes before a local folder is written.
    public static let openLicences: Set<String> = [
        "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "cc-by-4.0", "mpl-2.0", "unlicense", "zlib"
    ]

    public static func isOpen(licence: String?) -> Bool {
        guard let name = licence.map(normalised), !name.isEmpty else { return false }
        return openLicences.contains(name)
    }

    public static func normalised(_ licence: String) -> String {
        licence.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// The repository files a template's weights need: each file the template lists that the repository has, plus the
    /// base safetensors shards and their index. Adapters (LoRA, PEFT) are not base weights, and other formats (pickle,
    /// GGUF, ONNX) are never fetched. A sharded checkpoint has no single `model.safetensors`, so a template's safetensors
    /// file counts as missing only when the repository has no safetensors at all.
    public static func weightFiles(repository: [HFMetadataFile], template: [String]) -> (files: [String], missing: [String]) {
        let present = Set(repository.map(\.path))
        let hasSafetensors = repository.contains { $0.path.hasSuffix(".safetensors") }
        let listed = template.filter { present.contains($0) }
        let isAdapter: (String) -> Bool = { $0.contains("adapter") || $0.contains("lora") }
        let shards = repository.map(\.path).filter {
            ($0.hasSuffix(".safetensors") || $0.hasSuffix(".safetensors.index.json")) && !isAdapter($0)
        }
        let files = Array(Set(listed + shards)).sorted()
        let missing = template.filter { !present.contains($0) && !($0.hasSuffix(".safetensors") && hasSafetensors) }
        return (files, missing)
    }

    /// The weight dtype a checkpoint declares, or nil when config.json does not declare one. Half precisions are kept as
    /// shipped (bf16 is cast to fp16 at Get). Nothing is guessed: the caller refuses an undeclared dtype.
    public static func dtype(torchDType: String?) -> String? {
        switch torchDType?.lowercased() {
        case "bfloat16", "bf16": "bf16"
        case "float16", "fp16": "fp16"
        case "float32", "fp32": "fp32"
        default: nil
        }
    }

    /// The `model.toml` a local folder gets: the template's operations, runner, derivation and facets, with this repository's
    /// identity, licence, weights and pins. Template data that describes a different checkpoint (assets, quantized tiers,
    /// popularity, copy-specific prose and the template's context limit) is dropped.
    public static func manifest(
        template: [String: TOMLValue], id: String, metadata: HFModelMetadata, architecture: String?, dtype: String, files: [String],
        sha256: [String: String], bytes: [String: Int64], licenceAccepted: Bool, year: Int
    ) -> [String: TOMLValue] {
        var doc = template
        for key in ["scaffold", "asset", "vendor", "format", "precision", "quantization", "popularity", "good_for", "limits", "purpose", "context", "native_context"] {
            doc.removeValue(forKey: key)
        }
        let name = String(metadata.repo.split(separator: "/").last ?? Substring(id))
        let publisher = String(metadata.repo.split(separator: "/").first ?? "")
        let licence = metadata.licence.map(normalised).flatMap { $0.isEmpty ? nil : $0 } ?? "unknown"
        let open = isOpen(licence: metadata.licence)
        doc["id"] = .string(id)
        doc["name"] = .string(name)
        doc["publisher"] = .string(publisher)
        doc["released"] = .integer(Int64(year))
        doc["suggested"] = .bool(false)
        doc["summary"] = .string("\(name), added locally from \(metadata.repo)")
        doc["languages"] = .array((metadata.languages.isEmpty ? ["und"] : metadata.languages).map(TOMLValue.string))
        doc["params"] = .string(parameterLabel(metadata.parameterCount))
        if let architecture { doc["architecture"] = .string(architecture) }
        // The template's own selector stays unless the registry names another one for this checkpoint.
        doc["licence"] = .table([
            "spdx": .string(licence), "name": .string(licenceAccepted && !open ? "\(licence), accepted by the user" : licence),
            "commercial": .bool(open), "url": .string("https://huggingface.co/\(metadata.repo)/tree/\(metadata.revision)")
        ])
        doc["weights"] = .table([
            "repo": .string(metadata.repo), "revision": .string(metadata.revision), "dtype": .string(dtype),
            "files": .array(files.map(TOMLValue.string)),
            "sha256": .table(sha256.mapValues(TOMLValue.string)), "bytes": .table(bytes.mapValues(TOMLValue.integer))
        ])
        return doc
    }

    /// `596M`, `8B`: the parameter count in the label style the folders use.
    public static func parameterLabel(_ count: Int64?) -> String {
        guard let count, count > 0 else { return "unknown" }
        if count >= 1_000_000_000 { return String(format: "%.1fB", Double(count) / 1e9).replacingOccurrences(of: ".0B", with: "B") }
        return "\(max(1, count / 1_000_000))M"
    }
}

/// One accepted checkpoint family: a Hugging Face `model_type` that a bundled worker already runs for a kind.
/// `template` names a shipped folder with that runner; its operations and settings are copied into the new folder.
public struct ArchitectureEntry: Codable, Sendable, Equatable {
    public var modelType: String
    public var kind: String
    public var worker: String
    public var template: String
    /// The folder's `architecture` selector when the worker registers a name other than the checkpoint's model_type.
    public var architecture: String?
    /// Hugging Face pipeline tags this row serves. A repository tagged with another pipeline is refused, because one
    /// model_type can be a chat model and an embedding model alike (Qwen3). Nil means any tag is accepted.
    public var pipelineTags: [String]?

    enum CodingKeys: String, CodingKey { case modelType = "model_type", kind, worker, template, architecture, pipelineTags = "pipeline_tags" }
}

public struct ArchitectureRegistry: Codable, Sendable, Equatable {
    public var schema: Int
    public var architectures: [ArchitectureEntry]

    public static func load(from url: URL) throws -> Self {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }

    /// The entries that serve a model_type, optionally for one kind.
    public func entries(modelType: String, kind: String?) -> [ArchitectureEntry] {
        architectures.filter { $0.modelType == modelType && (kind == nil || $0.kind == kind) }
    }
}
