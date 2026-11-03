// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct DecisionIntakeRevisionTests {
    /// Rewrites every absolute path inside the repository to start at `models/`.
    /// Recorded evidence names paths relative to the checkout that ran it. Any absolute path compares from `models/` on,
    /// and any other path under `root` (this checkout) as relative to it.
    static func checkoutRelative(_ value: JSONValue, root: URL) -> JSONValue {
        switch value {
        case .string(let text):
            guard text.hasPrefix("/") else { return value }
            if let range = text.range(of: "/models/") { return .string(String(text[text.index(after: range.lowerBound)...])) }
            return text.hasPrefix(root.path + "/") ? .string(String(text.dropFirst(root.path.count + 1))) : value
        case .array(let items): return .array(items.map { checkoutRelative($0, root: root) })
        case .object(let fields): return .object(fields.mapValues { checkoutRelative($0, root: root) })
        default: return value
        }
    }
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @Test func intakeMatchesOrdinaryDefaultLoadIncludingM5FolderKernelsAndSettings() async throws {
        let source = intakeFixtureRepository.appendingPathComponent("models/laya-english")
        let declarations = try KernelManifest.loadIfPresent(folder: source)
        var files: [String: Data] = ["kernels/manifest.toml": try Data(contentsOf: source.appendingPathComponent("kernels/manifest.toml"))]
        for name in Set(declarations.kernels.flatMap(\.files)) { files["kernels/" + name] = try Data(contentsOf: source.appendingPathComponent("kernels/" + name)) }
        let extra: [String: TOMLValue] = [
            "schema": .integer(2), "selection": .table(["default": .string("native")]),
            "setting": .array([
                .table([
                    "id": .string("stock_mlx"), "title": .string("Stock MLX"), "type": .string("toggle"),
                    "default": .bool(false), "applies": .string("load")
                ])
            ])
        ]
        let f = try await makeLocalCatalogFixture(kinds: ["decisions"], extraTable: extra, extraFiles: files)
        defer { f.stop() }
        let id = "fixture-decisions", folder = f.paths.directory("models").appendingPathComponent(id)
        try await f.service.get(id: id, precision: "bf16")
        try await f.service.load(id: id)
        let served = try #require(await f.service.intakeTestLoad(id))
        let model = try ModelManifest.load(folder)
        let runner = try #require(model.runners.first)
        let intake = try ModelIntakeCommand.defaultLoadParameters(
            model: model, runner: runner, folder: folder,
            weights: URL(fileURLWithPath: try #require(served["weights_dir"]?.string)), assets: [:])
        #expect(served == intake)
        #expect(intake["config"]?["settings"]?["stock_mlx"] == false)
        let m5 = try ModelIntakeCommand.defaultLoadParameters(
            model: model, runner: runner, folder: folder,
            weights: folder, assets: [:], chip: "m5-max")
        let kernel = try #require(m5["config"]?["kernels"]?.array?.first)
        #expect(kernel["id"] == "smallm-tile" && kernel["entry"] == "smallm_tile")
        #expect(kernel["header"]?.string == folder.appendingPathComponent("kernels/m5/smallm_tile.h.metal").path)
        #expect(kernel["files"]?.array?.count == 2)
        await f.service.shutdown()
    }

    @Test func folderKernelManifestHeaderAndLoadSettingsInvalidatePassAndFailurePerModel() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/laya-english")), runner = model.runners[0]
        let folder = root.appendingPathComponent("laya-english"), other = root.appendingPathComponent("laya-multilingual")
        for target in [folder, other] {
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: intakeFixtureRepository.appendingPathComponent("models/laya-english/kernels"), to: target.appendingPathComponent("kernels"))
        }
        func identity(_ directory: URL, settings: ModelSettings? = nil) throws -> String {
            try #require(
                PrecisionEvidence.qualityIdentity(
                    model: model, dtype: "int8", runner: runner.name, implementation: "worker-source",
                    loadIdentity: ModelLoadConfiguration.identity(model: model, runner: runner, folder: directory, chip: "m5-max", settings: settings)))
        }
        let original = try identity(folder)
        #expect(try identity(other) == original) // Location-independent.
        for name in ["m5/smallm_tile.metal", "m5/smallm_tile.h.metal", "manifest.toml"] {
            let file = folder.appendingPathComponent("kernels/" + name), saved = try Data(contentsOf: file)
            var changed = saved; changed.append(Data("\n# revision\n".replacingOccurrences(of: "#", with: name == "manifest.toml" ? "#" : "//").utf8))
            try changed.write(to: file)
            let current = try identity(folder)
            #expect(current != original)
            for passed in [true, false] {
                var stats = PrecisionStatSet(kind: .decisions, dtype: "int8", runner: runner.name)
                stats.qualityIdentity = original; stats.qualityCheckPassed = passed
                let scoped = PrecisionEvidence.scopeQuality(.init(model: model.id, mode: "fast", stats: stats), identity: current)
                #expect(scoped.stats.qualityCheckPassed == nil && scoped.stats.selectionUnavailableReason == nil)
            }
            #expect(try identity(other) == original) // Another model's folder was not changed.
            try saved.write(to: file)
        }
        var settings = try ModelLoadConfiguration.defaults(model)
        try settings.set(model.id, key: "stock_mlx", value: true)
        #expect(try identity(folder, settings: settings) != original)
        var changedModel = model
        changedModel.setting?[0].defaultValue = .bool(true)
        #expect(
            try ModelLoadConfiguration.identity(model: changedModel, runner: runner, folder: folder, chip: "m5-max")
                != ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder, chip: "m5-max"))
    }

    @Test func completedDecisionRunsBindExactStatsBoundsAndDefaultLoadConfiguration() throws {
        for id in ["d1-3b", "laya-english", "laya-multilingual", "laya-typed-decisions"] {
            let folder = intakeFixtureRepository.appendingPathComponent("models/" + id), model = try ModelManifest.load(folder), runner = model.runners[0]
            let receipt = try latestIntakeBinding(folder)
            let run = try #require(receipt["run"]?.string)
            #expect(receipt["schema"] == 3)
            let resultPath = "results/intake/" + run + ".json", result = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(resultPath)))
            #expect(receipt["worker_sha256"] == result["worker_sha256"] && receipt["implementation_identity"] == result["implementation_identity"])
            #expect(receipt["load_identity"]?.string == (try ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder)))
            let rows = try #require(result["rows"]?.array), outcomes = try #require(receipt["outcomes"]?.array)
            #expect(rows.count == 3 && outcomes.count == 3)
            for row in rows {
                let dtype = try #require(row["precision"]?.string)
                let outcome = try #require(outcomes.first { $0["dtype"]?.string == dtype })
                // A tier withdrawn since (removedQualityTiers) keeps its recorded outcome; its table file is gone.
                if removedQualityTiers.contains(where: { $0.model == id && $0.tier == dtype }) {
                    #expect(!model.precisions.contains(dtype)); continue
                }
                let statsPath = try #require(outcome["result"]?.string), statsData = try Data(contentsOf: folder.appendingPathComponent(statsPath))
                let stats = try PrecisionStatsResult.decode(statsData).stats
                #expect(outcome["capture_result"]?.string == resultPath)
                #expect(outcome["capture_result_sha256"]?.string == (try FileDigest.sha256(folder.appendingPathComponent(resultPath))))
                #expect(outcome["result_sha256"]?.string == FileDigest.sha256(statsData))
                #expect(outcome["passed"]?.bool == stats.qualityCheckPassed && outcome["passed"] == row["gate"]?["component_gate_passed"])
                #expect(outcome["bounds_sha256"] == row["gate"]?["bounds_sha256"] && outcome["quality_identity"]?.string == stats.qualityIdentity)
                #expect(
                    stats.qualityIdentity
                        == PrecisionEvidence.qualityIdentity(
                            model: model, dtype: dtype, runner: runner.name,
                            implementation: receipt["implementation_identity"]?.string, loadIdentity: receipt["load_identity"]?.string))
                let params = try #require(row["load"]?["requested_load"]?.object)
                let recordedWeights = try #require(params["weights_dir"]?.string)
                var expected = try ModelIntakeCommand.defaultLoadParameters(
                    model: model, runner: runner, folder: folder,
                    weights: recordedWeights.hasPrefix("/") ? URL(fileURLWithPath: recordedWeights) : repo.appendingPathComponent(recordedWeights),
                    assets: [:])
                expected["dtype"] = .string(dtype)
                // The evidence records checkout-relative paths; compare so any checkout of the same repository verifies it.
                #expect(Self.checkoutRelative(.object(params), root: repo) == Self.checkoutRelative(.object(expected), root: repo))
                if id != "d1-3b" {
                    #expect(params["config"]?["kernels"]?.array?.first?["id"] == "smallm-tile")
                    #expect(row["load"]?["kernels"]?.array?.count == 1)
                    #expect(row["load"]?["optimizations"]?["smallm"]?.string?.contains("declares no smallm_tile") != true)
                }
            }
        }
    }

    @Test func completionWritesReceiptForExactRunAndEmittedStats() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/d1-3b")), run = "explicit-run"
        let evidence = root.appendingPathComponent("gate/evidence/intake/" + run), helpers = root.appendingPathComponent("Helpers")
        for directory in [evidence, helpers, root.appendingPathComponent("results/intake")] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        try Data("test-worker".utf8).write(to: helpers.appendingPathComponent("arsenal-worker-native"))
        let ids = root.appendingPathComponent("identities.json")
        try JSONEncoder().encode(["d1": "exact-source"]).write(to: ids)
        let artifact = BundledWorkerArtifact(helpers: helpers, qualityIdentities: ids)
        let fixture = evidence.appendingPathComponent("fixture")
        try FileManager.default.createDirectory(at: fixture, withIntermediateDirectories: true)
        try Data("retained input".utf8).write(to: fixture.appendingPathComponent("inputs.safetensors"))
        let bound: JSONValue = ["relative_l2_max": 0.05, "cosine_min": 0.999]
        let bounds = try WireJSON.encode(["components": ["a": bound]]), boundsHash = FileDigest.sha256(bounds)
        for name in ["bounds.json", "quantized-bounds.json"] { try bounds.write(to: evidence.appendingPathComponent(name)) }
        let plan = try IntakeQualification.freeze(
            selection: fixture, selectionBounds: evidence.appendingPathComponent("bounds.json"), to: evidence.appendingPathComponent("qualification-plan-bf16.json"))
        let quantPlan = try IntakeQualification.freeze(
            selection: fixture, selectionBounds: evidence.appendingPathComponent("quantized-bounds.json"), to: evidence.appendingPathComponent("qualification-plan-quantized.json"))
        let rows: [JSONValue] = try ["bf16", "int8", "int4"].map { dtype in
            let passed = dtype != "int4"
            let gate: JSONObject = [
                "bounds_sha256": .string(boundsHash), "component_gate_passed": .bool(passed),
                "chosen_answers_unchanged": true, "max_probability_delta": 0, "probability_delta_max": 0.017,
                "fixture_inputs_sha256": plan["selection"]!["inputs_sha256"]!, "reference_outputs_sha256": .string(String(repeating: "a", count: 64)),
                "candidate_outputs_sha256": .string(String(repeating: "b", count: 64)),
                "components": ["a": ["bounds": bound, "relative_l2": .number(passed ? 0 : 1.05), "cosine": 1, "finite": true, "activation_passed": true, "passed": .bool(passed)]]
            ]
            let qualification = try IntakeQualification.complete(
                plan: dtype == "bf16" ? plan : quantPlan, selection: gate, heldout: nil, searched: false, ordinaryLoad: ["ok": true, "requested_load": ["dtype": .string(dtype)]])
            return .object([
                "precision": .string(dtype), "warm": ["median_s": 0.01], "peak_gb": 2, "qualified": .bool(passed),
                "gate": .object(gate), "selection_gate": .object(gate), "qualification": .object(qualification)
            ])
        }
        let result = root.appendingPathComponent("results/intake/" + run + ".json")
        let workerHash = try FileDigest.sha256(helpers.appendingPathComponent("arsenal-worker-native"))
        try WireJSON.encode([
            "model": .string(model.id), "run": .string(run), "date": "2026-10-15T19:00:00Z", "rows": .array(rows),
            "worker_sha256": .string(workerHash), "implementation_identity": "exact-source", "load_identity": ""
        ]).write(to: result)
        // An unrelated, later file must never be selected by completion.
        try Data("{}".utf8).write(to: root.appendingPathComponent("results/intake/zzzz-unrelated.json"))
        var missing = rows; var invalid = missing[1].object!; invalid.removeValue(forKey: "qualification"); missing[1] = .object(invalid)
        #expect(throws: (any Error).self) { try ModelIntakeCommand.writeDecisionStats(rows: missing, model: model, folder: root, artifact: artifact, run: run, result: result) }
        #expect(!FileManager.default.fileExists(atPath: evidence.appendingPathComponent("quality-binding.json").path))
        try ModelIntakeCommand.writeDecisionStats(rows: rows, model: model, folder: root, artifact: artifact, run: run, result: result)
        let receipt = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("quality-binding.json")))
        #expect(receipt["run"]?.string == run && receipt["worker_sha256"]?.string == workerHash)
        #expect(receipt["implementation_identity"] == "exact-source")
        let outcomes = try #require(receipt["outcomes"]?.array)
        #expect(outcomes.count == 3)
        for outcome in outcomes {
            let path = try #require(outcome["result"]?.string), dtype = try #require(outcome["dtype"]?.string)
            let data = try Data(contentsOf: root.appendingPathComponent(path)), stat = try PrecisionStatsResult.decode(data).stats
            #expect(outcome["capture_result"]?.string == "results/intake/" + run + ".json")
            #expect(outcome["capture_result_sha256"]?.string == (try FileDigest.sha256(result)))
            #expect(outcome["result_sha256"]?.string == FileDigest.sha256(data))
            #expect(outcome["bounds_sha256"]?.string == boundsHash && outcome["passed"]?.bool == stat.qualityCheckPassed)
            #expect(outcome["quality_identity"]?.string == stat.qualityIdentity && stat.availability.provenance?.run == run + "-" + dtype)
        }
        #expect(
            try Data(contentsOf: evidence.appendingPathComponent("quality-binding.json"))
                == Data(contentsOf: root.appendingPathComponent("gate/evidence/quality-binding-2026-10-15.json")))
    }
}

private extension RealModelsService {
    func intakeTestLoad(_ id: String) -> JSONObject? { residentLoads[id] }
}
