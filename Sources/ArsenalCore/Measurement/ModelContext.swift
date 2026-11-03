// SPDX-License-Identifier: MIT
import Foundation

public enum ModelContext {
    /// Inspect config metadata only. Never loads weights or allocates a KV cache.
    public static func maximum(config: Data?, fallback: Int?) -> Int? {
        if let config, config.count <= 1_048_576, let object = try? WireJSON.decode(config),
            let raw = (object["text_config"]?["max_position_embeddings"] ?? object["max_position_embeddings"])?.number,
            let count = Int(exactly: raw), count > 0
        {
            return count
        }
        return fallback.flatMap { $0 > 0 ? $0 : nil }
    }
    public static func maximum(snapshot: URL, model: ModelManifest) -> Int? {
        let file = try? SafePath.resolve("config.json", inside: snapshot)
        let data = file.flatMap { url -> Data? in
            guard let bytes = try? FileDigest.bytes(url), bytes <= 1_048_576 else { return nil }
            return try? Data(contentsOf: url)
        }
        return maximum(config: data, fallback: model.nativeContext ?? model.context)
    }
    /// Old saved caps are migrated to the current model maximum, including nested load settings.
    public static func loadConfig(_ config: JSONObject, maximum: Int) -> JSONObject {
        var config = config
        config["context_limit"] = .number(Double(maximum))
        var settings = config["settings"]?.object ?? [:]
        settings["context_limit"] = .number(Double(maximum)); config["settings"] = .object(settings)
        return config
    }
}

extension RealModelsService {
    func updateContextMetrics(_ model: ModelManifest, row: ModelRow, metrics: inout JSONObject) {
        if model.kind == "chat", let maximum = configureChatContext(model, snapshot: snapshot(model, row.selectedPrecision)) {
            metrics["context"] = .number(Double(maximum)); metrics["native_context"] = .number(Double(maximum))
            if let date = declaredProvenanceDate(model) {
                let declaration = StatValue(
                    state: .declared, value: Double(maximum),
                    provenance: StatProvenance(
                        chip: Self.chip,
                        date: date, method: "Model config max_position_embeddings; no KV preallocation", source: "config.json / model.toml"))
                for dtype in [model.basePrecision, "int8", "int4"] { modelRows[model.id]?.precisionStats[dtype]?.set(.maximumContextTokens, declaration) }
            }
        } else if let context = model.context {
            metrics["context"] = .number(Double(context))
        }
    }
    func effectiveMode(_ model: ModelManifest, runner name: String) -> String {
        let supportsFast = model.runners.first { $0.name == name }?.fastExact == true || modelRows[model.id]?.hasFastKernels == true
        return supportsFast ? "fast" : "exact"
    }
    func enqueueMeasurements(_ model: ModelManifest) throws {
        let runner = selectedRunner(model)
        try PendingBenchmarkQueue(support: paths.support).introduce(model, runner: runner?.name ?? "Unavailable", mode: effectiveMode(model, runner: runner?.name ?? "Unavailable"))
    }
    func maximumContext(_ model: ModelManifest, snapshot: URL) -> Int? { ModelContext.maximum(snapshot: snapshot, model: model) }
    func configureChatContext(_ model: ModelManifest, snapshot: URL) -> Int? {
        guard model.kind == "chat", let maximum = maximumContext(model, snapshot: snapshot) else { return nil }
        // One config-derived choice, so settings cannot impose a lower fixed cap at Load.
        for index in (modelSettings.declarations[model.id] ?? []).indices where modelSettings.declarations[model.id]?[index]["id"]?.string == "context_limit" {
            modelSettings.declarations[model.id]?[index]["default"] = .number(Double(maximum))
            modelSettings.declarations[model.id]?[index]["choices"] = .array([.number(Double(maximum))])
            modelSettings.declarations[model.id]?[index]["help"] = .string("Model maximum, read from config. Memory grows with the conversation.")
        }
        return maximum
    }
}
