// SPDX-License-Identifier: MIT
import Foundation

extension RealModelsService {
    func cachePrecisionEvidence(_ model: ModelManifest, files: [String: Data], local: LocalModelFolder? = nil) {
        precisionResultCache[model.id] = files.keys.sorted().filter { $0.hasPrefix("results/") }.compactMap { path in
            guard let data = files[path] else { return nil }
            if let result = try? PrecisionStatsResult.decode(data), result.model == model.id, result.stats.kind.rawValue == model.kind {
                return PrecisionEvidence.current(result.stats, revision: model.weights.revision) ? result : nil
            }
            return (try? JSONDecoder().decode(ResultsFile.self, from: data)).flatMap { result in
                guard result.model == model.id, result.kind == model.kind else { return nil }
                if model.schema == 3, let local = local ?? localModels[model.id] {
                    guard let checks = result.checks, !checks.stale, ImplementationEvidence.compatible(checks, local: local) else { return nil }
                }
                return PrecisionStatsReader.legacy(result, source: path)
            }
        }
    }
    func statsForRunner(_ model: ModelManifest, runner: String) -> [String: PrecisionStatSet] {
        let directory = localModels[model.id]?.directory ?? paths.directory("models").appendingPathComponent(model.id)
        // Existing folders remain authoritative: a missing/tampered installed kernel must never fall back to catalog evidence.
        let catalogEntry = localModels[model.id] == nil && !FileManager.default.fileExists(atPath: directory.path) ? catalog.models.first { $0.id == model.id } : nil
        let spec = model.runners.first { $0.name == runner }
        let implementation = spec.flatMap {
            if let catalogEntry { return artifacts.qualityImplementation(model: model, runner: $0, directory: directory, catalogFiles: catalogEntry.files) }
            return artifacts.qualityImplementation(model: model, runner: $0, directory: directory)
        }
        let loadIdentity = spec.flatMap { spec in
            if let catalogEntry {
                return try? ModelLoadConfiguration.identity(
                    catalog: catalogEntry, runner: spec, kernelManifest: catalogKernelManifests[model.id], chip: Self.chip,
                    fast: effectiveMode(model, runner: runner) == "fast", settings: modelSettings)
            }
            return try? ModelLoadConfiguration.identity(
                model: model, runner: spec, folder: directory, chip: Self.chip,
                fast: effectiveMode(model, runner: runner) == "fast", settings: modelSettings)
        }
        let supported = model.formats.filter { $0.runners.contains(runner) }.map(\.id)
        var sets = PrecisionStatsReader.sets(
            kind: model.kind, baseDtype: model.basePrecision, runner: runner,
            supported: model.formats.isEmpty ? model.precisions : supported,
            results: ((precisionResultCache[model.id] ?? []) + LocalPrecisionResults.read(folder: paths.directory("models").appendingPathComponent(model.id), model: model)).filter
            { $0.mode == effectiveMode(model, runner: runner) }.map { result in
                return PrecisionEvidence.scopeQuality(
                    result,
                    identity: PrecisionEvidence.qualityIdentity(
                        model: model, dtype: result.stats.dtype, runner: runner, implementation: implementation,
                        loadIdentity: loadIdentity))
            })
        if model.kind == "chat", let maximum = maximumContext(model, snapshot: snapshot(model, modelRows[model.id]?.selectedPrecision ?? model.defaultPrecision)),
            let date = declaredProvenanceDate(model)
        {
            let cell = StatValue(
                state: .declared, value: Double(maximum),
                provenance: StatProvenance(
                    chip: Self.chip,
                    date: date, method: "Model config maximum context; no KV preallocation", source: "config.json / model.toml"))
            for dtype in sets.keys { sets[dtype]?.set(.maximumContextTokens, cell) }
        }
        return sets
    }
    /// Declared figures are dated by their source: the local folder's commit, else the signed catalog's source commit.
    func declaredProvenanceDate(_ model: ModelManifest) -> String? {
        if let local = localModels[model.id] { return local.declaredDate }
        return catalog.models.contains { $0.id == model.id } ? catalog.sourceDate : nil
    }
    func runnerDetails(_ model: ModelManifest) -> [CatalogRunner] {
        let eligible = Set(eligibleRunners(model).map(\.name))
        return model.runners.map { runner in
            let results = runnerResultsCache[model.id]?[runner.name] ?? []
            var detail = CatalogRunner(
                name: runner.name, language: runner.language, engine: runner.worker ?? runner.entry ?? runner.language, acceptedDefault: runner.name == defaultRunner(model),
                eligible: eligible.contains(runner.name), selected: runner.name == selectedRunner(model)?.name, results: results, history: historyCache[model.id]?[runner.name],
                ineligibleReason: eligible.contains(runner.name) ? nil : ineligibleReason(runner))
            detail.stats = statsForRunner(model, runner: runner.name)
            detail.features = runner.features ?? (runner.name == "native" ? RunnerFeature.native(worker: runner.worker) : [])
            return detail
        }
    }
}
