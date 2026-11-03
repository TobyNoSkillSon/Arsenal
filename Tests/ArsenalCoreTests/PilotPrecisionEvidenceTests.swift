// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct PilotPrecisionEvidenceTests {
    private var repo: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private func table(_ model: ModelManifest) throws -> [String: Data] {
        let folder = intakeFixtureRepository.appendingPathComponent("models/" + model.id + "/results")
        return try Dictionary(
            uniqueKeysWithValues: FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
                .filter { $0.lastPathComponent.hasPrefix("intake-m5-max-") }
                .map { ("results/" + $0.lastPathComponent, try Data(contentsOf: $0)) })
    }
    @Test(arguments: ["m5-max", "m4-max"]) func fluxUsesDefaultWorkloadForEveryPrecision(chip: String) throws {
        let model = try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/flux2-klein-4b"))
        let current = try table(model).values.map(PrecisionStatsResult.decode)
        let legacyFile = try JSONDecoder().decode(
            ResultsFile.self, from: Data(contentsOf: intakeFixtureRepository.appendingPathComponent("models/flux2-klein-4b/results/m5-max.json")))
        let legacy = try #require(PrecisionStatsReader.legacy(legacyFile))
        #expect(legacy.stats.value(.secondsPerImage).value == 1.0684575414925348)
        let sets = PrecisionStatsReader.sets(
            kind: model.kind, baseDtype: model.basePrecision, runner: "native", supported: model.precisions, results: current + [legacy], chip: chip, gpuCores: 40)
        #expect(Set(sets.keys) == Set(["bf16", "int8", "int4"]))
        for result in current {
            let stat = try #require(sets[result.stats.dtype])
            #expect(stat.value(.secondsPerImage).value == result.stats.value(.secondsPerImage).value)
            #expect(stat.value(.peakMemoryGB).value == result.stats.value(.peakMemoryGB).value)
            for provenance in [stat.availability.provenance] + stat.metrics.values.compactMap(\.provenance).map(Optional.some) {
                #expect(provenance?.gpuCores == 40 && provenance?.defaultWorkload == true)
            }
        }
        let folder = intakeFixtureRepository.appendingPathComponent("models/flux2-klein-4b")
        let binding = try latestIntakeBinding(folder), run = try #require(binding["run"]?.string)
        let capture = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("results/intake/" + run + ".json")))
        let baseline = try #require(capture["rows"]?.array?.first { $0["precision"] == "bf16" })
        let seconds = try #require(baseline["warm"]?["median_s"]?.number), peak = try #require(baseline["peak_gb"]?.number)
        #expect(sets["bf16"]?.value(.secondsPerImage).value == seconds)
        #expect(sets["bf16"]?.value(.peakMemoryGB).value == peak)
        // A host-matching historical result still cannot displace a default-workload reference.
        var historicalHost = legacy
        historicalHost.stats.availability.provenance?.chip = chip
        var defaultReference = current.first { $0.stats.dtype == "bf16" }!
        defaultReference.stats.set(.peakMemoryGB, StatValue())
        let strict = PrecisionStatsReader.sets(
            kind: model.kind, baseDtype: model.basePrecision, runner: "native", supported: model.precisions,
            results: [historicalHost, defaultReference], chip: chip, gpuCores: 40)
        #expect(strict["bf16"]?.value(.secondsPerImage).value == seconds)
        #expect(strict["bf16"]?.value(.peakMemoryGB).state == .notMeasured)
    }
    @Test func pilotOutcomesFollowTheirSourceIdentityAndMismatchedInt8IsUncheckedAndOffered() async throws {
        let helpers = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: helpers, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: helpers) }
        let command = Process()
        command.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        command.arguments = [repo.appendingPathComponent("scripts/quality-identities.py").path, helpers.appendingPathComponent("quality-identities.json").path]
        try command.run(); command.waitUntilExit()
        #expect(command.terminationStatus == 0)
        let artifact = BundledWorkerArtifact(helpers: helpers, qualityIdentities: helpers.appendingPathComponent("quality-identities.json"))
        let fixture = try await makeLocalCatalogFixture(kinds: ["images"], artifacts: artifact)
        defer { fixture.stop() }
        for id in ["flux2-klein-4b", "stable-audio-3-medium", "d1-3b"] {
            var model = try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/" + id))
            let files = try table(model)
            let stats = await fixture.service.pilotStats(model, files: files)
            let folder = intakeFixtureRepository.appendingPathComponent("models/" + id), runner = model.runners[0]
            let implementation = artifact.qualityImplementation(model: model, runner: runner, directory: folder)
            let loadIdentity = try ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder)
            let saved = try files.values.map(PrecisionStatsResult.decode)
            #expect(Set(stats.keys) == Set(model.precisions))
            for dtype in model.precisions {
                let stat = try #require(stats[dtype])
                let prior = try #require(saved.first { $0.stats.dtype == dtype })
                let current = PrecisionEvidence.qualityIdentity(model: model, dtype: dtype, runner: runner.name, implementation: implementation, loadIdentity: loadIdentity)
                // Engine changes may stale a family receipt while leaving the retained measurements intact.
                let expected: Bool? = current == prior.stats.qualityIdentity ? prior.stats.qualityCheckPassed : nil
                #expect(stat.qualityCheckPassed == expected)
                #expect(stat.selectionUnavailableReason == nil)
                #expect(stat.value(.peakMemoryGB).value != nil)
            }
            var row = ModelRow(id: id, name: model.name, kind: model.kind, precisions: model.precisions)
            row.precisionStats = stats
            #expect(
                row.precisionQualityWarnings == ["int4": stats["int4"]?.qualityCheckPassed == nil ? PrecisionStatSet.int4UncheckedWarning : PrecisionStatSet.int4QualityWarning])
            model.derive?["int8"] = "changed-recipe:" + id
            model.derive?["int4"] = "changed-recipe:" + id
            let stale = await fixture.service.pilotStats(model, files: files)
            #expect(stale["int8"]?.qualityCheckPassed == nil)
            #expect(stale["int8"]?.selectionUnavailableReason == nil)
            #expect(stale["int4"]?.qualityCheckPassed == nil)
            #expect(stale["int4"]?.selectionUnavailableReason == nil)
            #expect(stale["int8"]?.value(.peakMemoryGB) == stats["int8"]?.value(.peakMemoryGB))
            row.precisionStats = stale
            #expect(row.precisionQualityWarnings == ["int4": PrecisionStatSet.int4UncheckedWarning])
        }
        await fixture.service.shutdown()
    }
    /// "why does the D1 3 billion does not allow int 8, but it allows int 4". Through the real service path,
    /// every catalog model that declares int8 offers it on each runner that supports it, unless a recorded int8 check
    /// failed. Any model may offer its 16-bit base alone.
    @Test func everyCatalogModelOffersInt8UnlessARecordedCheckFailed() async throws {
        let helpers = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: helpers, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: helpers) }
        let command = Process()
        command.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        command.arguments = [repo.appendingPathComponent("scripts/quality-identities.py").path, helpers.appendingPathComponent("quality-identities.json").path]
        try command.run(); command.waitUntilExit()
        #expect(command.terminationStatus == 0)
        let artifact = BundledWorkerArtifact(helpers: helpers, qualityIdentities: helpers.appendingPathComponent("quality-identities.json"))
        let fixture = try await makeLocalCatalogFixture(kinds: ["images"], artifacts: artifact)
        defer { fixture.stop() }
        let folders = try FileManager.default.contentsOfDirectory(at: intakeFixtureRepository.appendingPathComponent("models"), includingPropertiesForKeys: nil)
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path) }.sorted { $0.path < $1.path }
        let directories = try FileManager.default.contentsOfDirectory(at: intakeFixtureRepository.appendingPathComponent("models"), includingPropertiesForKeys: [.isDirectoryKey])
            .filter { !$0.lastPathComponent.hasPrefix(".") && (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
        #expect(folders.count == directories.count && folders.count >= 48)
        var offered: [String] = [], greyed: [String] = [], shipped: Set<String> = [], halfOnly: Set<String> = []
        for folder in folders {
            let model = try ModelManifest.load(folder)
            guard CatalogShipping.includes(model) else { continue }
            // Precision stats (and so the int8 offer) exist only for measured kinds; PrecisionStatsReader.sets returns none for the rest.
            guard ModelStatKind(rawValue: model.kind) != nil else { continue }
            shipped.insert(model.id)
            if !model.precisions.contains("int8") && !model.formats.contains(where: { $0.id == "int8" }) {
                // The int8-with-int4 rule is about tiers Arsenal derives. A vendor-published checkpoint pinned in [vendor.int4]
                // (Drama 3's mlx-community 4-bit) is the release as its makers ship it, not a derivation Arsenal chose to skip int8 for.
                // An int8 withdrawn on the record (removedQualityTiers: Kokoro's, no faster and no smaller than FP16) leaves its int4.
                let int8Withdrawn = removedQualityTiers.contains { $0.model == model.id && $0.tier == "int8" }
                #expect(!model.precisions.contains("int4") || model.vendor?["int4"] != nil || int8Withdrawn, "\(model.id) offers a derived int4 without int8")
                halfOnly.insert(model.id)
                continue
            }
            let resultsFolder = folder.appendingPathComponent("results")
            let names = (try? FileManager.default.contentsOfDirectory(atPath: resultsFolder.path)) ?? []
            let files = try Dictionary(
                uniqueKeysWithValues: names.filter { $0.hasSuffix(".json") }.map { ("results/" + $0, try Data(contentsOf: resultsFolder.appendingPathComponent($0))) })
            // Recorded int8 failures, read straight from the files.
            let recordedFailure = files.values.contains { data in
                if let current = try? PrecisionStatsResult.decode(data) {
                    let runner = model.runners.first { $0.name == current.stats.runner }
                    let identity = PrecisionEvidence.qualityIdentity(
                        model: model, dtype: current.stats.dtype, runner: current.stats.runner,
                        implementation: runner.flatMap { artifact.qualityImplementation(model: model, runner: $0, directory: folder) },
                        loadIdentity: runner.flatMap { try? ModelLoadConfiguration.identity(model: model, runner: $0, folder: folder) })
                    let scoped = PrecisionEvidence.scopeQuality(current, identity: identity)
                    return scoped.stats.dtype == "int8" && (scoped.stats.qualityCheckPassed == false || scoped.stats.hasLegacyQualityFailure)
                }
                guard let legacy = try? JSONDecoder().decode(ResultsFile.self, from: data) else { return false }
                return legacy.precision == "int8" && legacy.recordedQualityFailure != nil
            }
            try await fixture.service.cachePilotCatalogEvidence(model, files: files, source: folder)
            for runner in model.runners where model.formats.first(where: { $0.id == "int8" })?.runners.contains(runner.name) ?? model.precisions.contains("int8") {
                let stats = await fixture.service.statsForRunner(model, runner: runner.name)
                let reason = try #require(stats["int8"], "\(model.id)/\(runner.name) has no int8 stats").selectionUnavailableReason
                if recordedFailure {
                    #expect(reason == "Did not pass the quality check", "\(model.id)/\(runner.name)")
                    greyed.append(model.id + "/" + runner.name)
                } else {
                    #expect(reason == nil, "\(model.id)/\(runner.name): \(reason ?? "")")
                    offered.append(model.id + "/" + runner.name)
                }
                // int4 is offered wherever it is declared; a tier removed after a failed check (removedQualityTiers) is never offered.
                if declaredTiers(model).contains("int4") {
                    #expect(stats["int4"]?.selectionUnavailableReason == nil, "\(model.id)/\(runner.name)")
                } else {
                    #expect(stats["int4"] == nil || stats["int4"]?.selectionUnavailableReason != nil, "\(model.id)/\(runner.name) offers a removed int4")
                }
            }
        }
        #expect(offered.contains("d1-3b/native"))
        // The Laya models ship 16-bit only since 26 Oct (under 1 GB at 16-bit; removedQualityTiers).
        #expect(["laya-english", "laya-multilingual", "laya-typed-decisions"].allSatisfy(halfOnly.contains))
        // Every shipped model lands in exactly one list (the original 16 at least).
        let quantized = Set(offered.map { $0.split(separator: "/")[0] }).count + Set(greyed.map { $0.split(separator: "/")[0] }).count
        #expect(quantized + halfOnly.count == shipped.count && shipped.count >= 16)
        await fixture.service.shutdown()
    }
    /// The Laya intake captures exactly the components the core expects, read from the committed bounds of each run.
    @Test func layaIntakeKeysMatchCommittedCaptures() throws {
        for (id, layers) in [("laya-multilingual", 22), ("laya-english", 28), ("laya-typed-decisions", 28)] {
            // Round 2 (Linears only) and round 3 (packed embedding, language-graded probes); the table binds the latest.
            let runs = try FileManager.default.contentsOfDirectory(atPath: intakeFixtureRepository.appendingPathComponent("models/\(id)/gate/evidence/intake").path).sorted()
            #expect(runs.count >= 2)
            let expected = ModelIntakeCommand.layaComponentKeys(layers: layers, headLayers: 2)
            #expect(expected.count == 2 * (layers + 4))
            for run in runs {
                let evidence = intakeFixtureRepository.appendingPathComponent("models/\(id)/gate/evidence/intake/" + run)
                for file in ["bounds.json", "quantized-bounds.json"] {
                    let bounds = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent(file)))
                    #expect(Set(bounds["components"]?.object?.keys.map { $0 } ?? []) == expected)
                }
                let policy = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("policy.json")))
                #expect(Set(policy["component_keys"]?.array?.compactMap(\.string) ?? []) == expected)
            }
            // Graded only in the model's supported languages: German only for Multilingual.
            let latest = try WireJSON.decode(Data(contentsOf: intakeFixtureRepository.appendingPathComponent("models/\(id)/gate/evidence/intake/" + runs.last! + "/policy.json")))
            let states = latest["component_states"]?.array?.compactMap(\.string) ?? []
            #expect(states.count == 2 && states[1].hasPrefix(id == "laya-multilingual" ? "Das Paket" : "The package"))
            let binding = try latestIntakeBinding(intakeFixtureRepository.appendingPathComponent("models/" + id))
            #expect(binding["run"]?.string == runs.last!)
            // The recorded outcomes stay in the binding: int8 passed, int4 failed. The tiers themselves were withdrawn on 26 Oct
            // (under 1 GB at 16-bit), so their table results are gone and the folder declares 16-bit only.
            let outcomes = binding["outcomes"]?.array ?? []
            #expect(outcomes.first { $0["dtype"] == "int8" }?["passed"] == true && outcomes.first { $0["dtype"] == "int4" }?["passed"] == false)
            let model = try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/" + id))
            #expect(model.derive?["int8"] == nil && model.derive?["int4"] == nil && model.precisions == [model.basePrecision])
        }
        #expect(ModelIntakeCommand.offered(dtype: "int8", passed: nil) && ModelIntakeCommand.offered(dtype: "int8", passed: true))
        #expect(!ModelIntakeCommand.offered(dtype: "int8", passed: false) && ModelIntakeCommand.offered(dtype: "int4", passed: false))
        #expect(!ModelIntakeCommand.offered(dtype: "fp16", passed: nil))
    }
}

private extension RealModelsService {
    func cachePilotCatalogEvidence(_ model: ModelManifest, files: [String: Data], source: URL) throws {
        let directory = paths.directory("models").appendingPathComponent(model.id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let kernels = source.appendingPathComponent("kernels")
        if FileManager.default.fileExists(atPath: kernels.path) {
            try FileManager.default.copyItem(at: kernels, to: directory.appendingPathComponent("kernels"))
        }
        // The served identity includes folder kernels and declared default settings, not only the worker receipt.
        modelSettings.declare(model.id, metadata: try WireJSON.decode(JSONEncoder().encode(model)))
        cachePrecisionEvidence(model, files: files)
    }
    func pilotStats(_ model: ModelManifest, files: [String: Data]) -> [String: PrecisionStatSet] {
        cachePrecisionEvidence(model, files: files)
        return statsForRunner(model, runner: "native")
    }
}
