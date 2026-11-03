// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalCommands

struct CatalogResultValidationTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var folder: URL { intakeFixtureRepository.appendingPathComponent("models/stable-audio-3-medium") }
    private let tablePath = "results/intake-m5-max-int8-native-fast.json"
    private let runPath = "results/intake/intake-2026-10-15T11-57-03Z-619ba9d4-cb51-46ca-a784-caed82dc2fd0.json"

    @Test func trackedFoldersValidateWithoutHistoricalEvidence() throws {
        let kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
        let folders = try FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("models"), includingPropertiesForKeys: nil)
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path) }
        let directories = try FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("models"), includingPropertiesForKeys: [.isDirectoryKey])
            .filter { !$0.lastPathComponent.hasPrefix(".") && (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
        #expect(folders.count == directories.count && folders.count >= 17)
        for folder in folders {
            let errors = CatalogCommand.validation(folder: folder, kinds: kinds)
            #expect(errors.isEmpty, "\(folder.lastPathComponent): \(errors.joined(separator: "; "))")
        }
    }

    @Test func allRepositoryModelFoldersValidate() throws {
        let kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
        let folders = try FileManager.default.contentsOfDirectory(at: intakeFixtureRepository.appendingPathComponent("models"), includingPropertiesForKeys: [.isDirectoryKey])
            .filter { !$0.lastPathComponent.hasPrefix(".") && (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
        #expect(!folders.isEmpty)
        var resultCount = 0
        for folder in folders {
            let errors = CatalogCommand.validation(folder: folder, kinds: kinds)
            #expect(errors.isEmpty, "\(folder.lastPathComponent): \(errors.joined(separator: "; "))")
            resultCount += try SafePath.files(in: folder).filter { $0.hasPrefix("results/") && $0.hasSuffix(".json") }.count
        }
        #expect(resultCount > 0)
    }

    @Test(arguments: ["missing-audio", "blank-path", "streaming"]) func malformedEnhancementWorkloadsFail(mutation: String) throws {
        let folder = intakeFixtureRepository.appendingPathComponent("models/re-use"), model = try ModelManifest.load(folder)
        let binding = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/evidence/quality-binding-2026-10-15.json")))
        let run = try #require(binding["run"]?.string), path = "results/intake/" + run + ".json"
        var root = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(path)))
        try CatalogResultValidation.validate(data: WireJSON.encode(root), path: path, folder: folder, manifest: model)
        var policy = try #require(root["policy"]?.object), fixture = try #require(policy["fixture"]?.object)
        switch mutation {
        case "missing-audio": fixture.removeValue(forKey: "audio")
        case "blank-path": fixture["audio"] = ["path": ""]
        default: fixture["stream"] = true
        }
        policy["fixture"] = .object(fixture); root["policy"] = .object(policy)
        #expect(throws: (any Error).self) { try CatalogResultValidation.validate(data: WireJSON.encode(root), path: path, folder: folder, manifest: model) }
    }

    @Test(arguments: ["laya-english", "stable-audio-3-medium", "stable-audio-3-small-sfx", "von-1.1", "re-use", "birefnet"])
    func currentFixedRecipesRejectTheReviewersDoctoredComponentAndMissingQualification(id: String) throws {
        let folder = intakeFixtureRepository.appendingPathComponent("models/" + id), model = try ModelManifest.load(folder)
        let bindings = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("gate/evidence"), includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("quality-binding-20") }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        let binding = try WireJSON.decode(Data(contentsOf: #require(bindings.last))), run = try #require(binding["run"]?.string)
        let path = "results/intake/" + run + ".json"
        let original = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(path)))
        try CatalogResultValidation.validate(data: WireJSON.encode(original), path: path, folder: folder, manifest: model)
        for mutation in [
            "failed-component", "forged-component-pass", "missing-qualification", "stripped-qualification", "searched", "input", "load", "count", "row-verdict", "missing-task",
            "missing-format"
        ] {
            var root = original, rows = try #require(root["rows"]?.array)
            if mutation == "missing-task" && model.kind != "decisions" { continue }
            if mutation == "missing-format" && model.weights.dtype != "fp32" { continue }
            let tier = mutation == "missing-format" ? model.basePrecision : "int8"
            let index = try #require(rows.firstIndex { $0["precision"]?.string == tier })
            var row = try #require(rows[index].object), q = try #require(row["qualification"]?.object)
            switch mutation {
            case "failed-component", "forged-component-pass":
                var gate = try #require(row["gate"]?.object), components = try #require(gate["components"]?.object)
                let key = try #require(components.keys.sorted().first), old = try #require(components[key]?.object)
                var entry = old
                entry["relative_l2"] = 1.05; entry["finite"] = false; entry["passed"] = .bool(mutation == "forged-component-pass")
                components[key] = .object(entry); gate["components"] = .object(components)
                // Preserve aggregate/row pass, and forge every redundant gate copy together.
                gate["component_gate_passed"] = true; row["qualified"] = true; q["passed"] = true
                row["gate"] = .object(gate); row["selection_gate"] = .object(gate); q["selection_gate"] = .object(gate)
            case "missing-qualification": row.removeValue(forKey: "qualification")
            case "stripped-qualification": row.removeValue(forKey: "qualification"); row.removeValue(forKey: "selection_gate")
            case "missing-task", "missing-format":
                var gate = try #require(row["gate"]?.object)
                for key in mutation == "missing-task"
                    ? ["chosen_answers_unchanged", "max_probability_delta", "probability_delta_max"] : ["format_qualification", "format_qualification_sha256", "selected_half"]
                { gate.removeValue(forKey: key) }
                row["gate"] = .object(gate); row["selection_gate"] = .object(gate); q["selection_gate"] = .object(gate)
            case "searched": q["searched"] = true
            case "input":
                var gate = try #require(row["gate"]?.object); gate["fixture_inputs_sha256"] = .string(String(repeating: "0", count: 64))
                row["gate"] = .object(gate); row["selection_gate"] = .object(gate); q["selection_gate"] = .object(gate)
            case "load": q.removeValue(forKey: "ordinary_load")
            case "row-verdict": row["qualified"] = false
            default:
                var plan = try #require(q["fixtures"]?.object), selection = try #require(plan["selection"]?.object)
                selection["component_count"] = 1; plan["selection"] = .object(selection); q["fixtures"] = .object(plan)
            }
            if !["missing-qualification", "stripped-qualification"].contains(mutation) { row["qualification"] = .object(q) }
            rows[index] = .object(row); root["rows"] = .array(rows)
            #expect(throws: (any Error).self, "\(id): \(mutation)") {
                try CatalogResultValidation.validate(data: WireJSON.encode(root), path: path, folder: folder, manifest: model)
            }
        }
    }

    private func document(_ path: String) throws -> JSONObject { try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(path))) }
    private func validate(_ root: JSONObject, path: String) throws {
        try CatalogResultValidation.validate(data: WireJSON.encode(root), path: path, folder: folder, manifest: ModelManifest.load(folder))
    }

    @Test(arguments: [
        "valid", "missing-qualification", "missing-heldout", "copied-heldout", "missing-load", "wrong-load-tier", "diagnostic", "count", "unknown", "forged-metric", "stripped-v3"
    ])
    func heldoutVoiceResultsCannotDowngradeOrForgeQualification(mutation: String) throws {
        let voice = intakeFixtureRepository.appendingPathComponent("models/breeze-tts-2")
        let binding = try WireJSON.decode(Data(contentsOf: voice.appendingPathComponent("gate/evidence/quality-binding-2026-10-16.json")))
        let path = "results/intake/" + (try #require(binding["run"]?.string)) + ".json"
        var root = try WireJSON.decode(Data(contentsOf: voice.appendingPathComponent(path)))
        var rows = try #require(root["rows"]?.array), row = try #require(rows[1].object), q = try #require(row["qualification"]?.object)
        switch mutation {
        case "missing-qualification": row.removeValue(forKey: "qualification")
        case "stripped-v3": row.removeValue(forKey: "qualification"); row.removeValue(forKey: "selection_gate")
        case "missing-heldout": q.removeValue(forKey: "heldout_gate")
        case "copied-heldout": q["heldout_gate"] = q["selection_gate"]; row["gate"] = q["heldout_gate"]
        case "missing-load": q.removeValue(forKey: "ordinary_load")
        case "wrong-load-tier":
            var load = try #require(q["ordinary_load"]?.object), request = try #require(load["requested_load"]?.object)
            request["dtype"] = "int4"; load["requested_load"] = .object(request); q["ordinary_load"] = .object(load)
        case "diagnostic":
            var gate = try #require(q["heldout_gate"]?.object); gate["diagnostic_only"] = true; q["heldout_gate"] = .object(gate); row["gate"] = .object(gate)
        case "count":
            var plan = try #require(q["fixtures"]?.object), heldout = try #require(plan["heldout"]?.object)
            heldout["component_count"] = 1; plan["heldout"] = .object(heldout); q["fixtures"] = .object(plan)
        case "unknown": q["unchecked"] = true
        case "forged-metric":
            var gate = try #require(q["heldout_gate"]?.object), components = try #require(gate["components"]?.object)
            let key = try #require(components.keys.sorted().first); var entry = try #require(components[key]?.object)
            entry["relative_l2"] = 1; entry["passed"] = true; components[key] = .object(entry); gate["components"] = .object(components)
            q["heldout_gate"] = .object(gate); row["gate"] = .object(gate)
        default: break
        }
        if !["missing-qualification", "stripped-v3"].contains(mutation) { row["qualification"] = .object(q) }
        rows[1] = .object(row); root["rows"] = .array(rows)
        let validate = { try CatalogResultValidation.validate(data: WireJSON.encode(root), path: path, folder: voice, manifest: ModelManifest.load(voice)) }
        if mutation == "valid" { try validate() } else { #expect(throws: (any Error).self) { try validate() } }
    }

    @Test(arguments: ["unknown-schema", "missing-schema", "missing-model", "wrong-model", "missing-stats", "wrong-kind", "unknown-field", "missing-metric", "negative-metric"])
    func malformedTableResultsFail(mutation: String) throws {
        var root = try document(tablePath), stats = try #require(root["stats"]?.object)
        switch mutation {
        case "unknown-schema": root["schema"] = 99
        case "missing-schema": root.removeValue(forKey: "schema")
        case "missing-model": root.removeValue(forKey: "model")
        case "wrong-model": root["model"] = "another-model"
        case "missing-stats": root.removeValue(forKey: "stats")
        case "wrong-kind": stats["kind"] = "images"; root["stats"] = .object(stats)
        case "unknown-field": root["new_format"] = true
        case "missing-metric": stats["metrics"] = [:]; root["stats"] = .object(stats)
        default:
            var metrics = try #require(stats["metrics"]?.object), peak = try #require(metrics["peakMemoryGB"]?.object)
            peak["value"] = -1; metrics["peakMemoryGB"] = .object(peak); stats["metrics"] = .object(metrics); root["stats"] = .object(stats)
        }
        #expect(throws: (any Error).self) { try validate(root, path: tablePath) }
    }

    @Test(arguments: ["missing-run", "wrong-run", "missing-source", "empty-source-path", "missing-file", "escape", "absolute", "directory", "other-model"])
    func tableEvidenceMustResolveInsideItsModel(mutation: String) throws {
        var root = try document(tablePath), stats = try #require(root["stats"]?.object)
        var availability = try #require(stats["availability"]?.object), provenance = try #require(availability["provenance"]?.object)
        switch mutation {
        case "missing-run": provenance.removeValue(forKey: "run")
        case "wrong-run": provenance["run"] = "intake-nonexistent-int8"
        case "missing-source": provenance.removeValue(forKey: "source")
        case "empty-source-path": provenance["source"] = "; no path"
        case "missing-file": provenance["source"] = "results/missing.json"
        case "escape": provenance["source"] = "../flux2-klein-4b/results/intake-m5-max-int8-native-exact.json"
        case "absolute": provenance["source"] = .string(folder.appendingPathComponent(runPath).path)
        case "directory": provenance["source"] = "results/intake"
        default: provenance["source"] = "results/m5-max.json"
        }
        availability["provenance"] = .object(provenance); stats["availability"] = .object(availability); root["stats"] = .object(stats)
        #expect(throws: (any Error).self) { try validate(root, path: tablePath) }
    }

    @Test(arguments: [
        "missing-policy", "missing-rows", "empty-rows", "wrong-kind", "wrong-run", "unsafe-run", "missing-run", "wrong-policy-run", "missing-precision", "missing-gate",
        "bad-bounds", "negative-peak", "empty-warm", "unknown-field"
    ])
    func malformedIntakeResultsFail(mutation: String) throws {
        var root = try document(runPath)
        switch mutation {
        case "missing-policy": root.removeValue(forKey: "policy")
        case "missing-rows": root.removeValue(forKey: "rows")
        case "empty-rows": root["rows"] = []
        case "wrong-kind": root["kind"] = "images"
        case "wrong-run": root["run"] = "intake-nonexistent"
        case "unsafe-run": root["run"] = "../escape"
        case "missing-run": root.removeValue(forKey: "run")
        case "wrong-policy-run":
            var policy = try #require(root["policy"]?.object); policy["run"] = "another-run"; root["policy"] = .object(policy)
        case "unknown-field": root["new_format"] = true
        default:
            var rows = try #require(root["rows"]?.array), row = try #require(rows[0].object)
            switch mutation {
            case "missing-precision": row.removeValue(forKey: "precision")
            case "missing-gate": row.removeValue(forKey: "gate")
            case "bad-bounds":
                var gate = try #require(row["gate"]?.object); gate["bounds_sha256"] = .string(String(repeating: "0", count: 64)); row["gate"] = .object(gate)
            case "negative-peak": row["peak_gb"] = -1
            default: row["warm"] = ["median_s": 0, "samples_s": []]
            }
            rows[0] = .object(row); root["rows"] = .array(rows)
        }
        #expect(throws: (any Error).self) { try validate(root, path: runPath) }
    }

    @Test(arguments: ["missing-tier", "wrong-recipe", "wrong-half", "not-object"])
    func mixedVoiceRecipesMustMatchTheirCapturedSnapshots(mutation: String) throws {
        let voice = intakeFixtureRepository.appendingPathComponent("models/pocket-tts")
        let binding = try WireJSON.decode(Data(contentsOf: voice.appendingPathComponent("gate/evidence/quality-binding-2026-10-15.json")))
        let run = try #require(binding["run"]?.string), path = "results/intake/" + run + ".json"
        var document = try WireJSON.decode(Data(contentsOf: voice.appendingPathComponent(path)))
        let manifest = try ModelManifest.load(voice)
        try CatalogResultValidation.validate(data: WireJSON.encode(document), path: path, folder: voice, manifest: manifest)
        var recipes = try #require(document["recipes"]?.object)
        switch mutation {
        case "missing-tier": recipes.removeValue(forKey: "int4")
        case "wrong-recipe": recipes["int8"] = "intake-v2:pocket-tts:all"
        case "wrong-half": document["selected_half"] = "fp32"
        default: document["recipes"] = []
        }
        if mutation != "not-object" { document["recipes"] = .object(recipes) }
        #expect(throws: (any Error).self) {
            try CatalogResultValidation.validate(data: WireJSON.encode(document), path: path, folder: voice, manifest: manifest)
        }
    }

    @Test(arguments: ["valid", "capture", "missing-count", "invalid-pid", "missing-load", "unknown-field"])
    func servingPeakReceiptsAreStrictAndCaptureFree(mutation: String) throws {
        var root = try document(runPath), rows = try #require(root["rows"]?.array), row = try #require(rows[0].object)
        var worker: JSONObject = ["pid": 123, "measurement_components_requests": 0]
        row["capture_load"] = row["load"]
        switch mutation {
        case "capture": worker["measurement_components_requests"] = 1
        case "missing-count": worker.removeValue(forKey: "measurement_components_requests")
        case "invalid-pid": worker["pid"] = 0
        case "missing-load": row.removeValue(forKey: "capture_load")
        case "unknown-field": worker["unknown"] = true
        default: break
        }
        row["serving_worker"] = .object(worker); rows[0] = .object(row); root["rows"] = .array(rows)
        if mutation == "valid" {
            try validate(root, path: runPath)
        } else {
            #expect(throws: (any Error).self) { try validate(root, path: runPath) }
        }
    }

    @Test func renamedFilesStillUseDeclaredSchemaAndBenchmarkValidationStaysStrict() throws {
        try validate(document(tablePath), path: "results/arbitrary.json")
        let benchmarkFolder = intakeFixtureRepository.appendingPathComponent("models/flux2-klein-4b")
        let manifest = try ModelManifest.load(benchmarkFolder)
        var benchmark = try WireJSON.decode(Data(contentsOf: benchmarkFolder.appendingPathComponent("results/m5-max.json")))
        try CatalogResultValidation.validate(data: WireJSON.encode(benchmark), path: "results/intake-renamed.json", folder: benchmarkFolder, manifest: manifest)
        benchmark.removeValue(forKey: "conditions")
        #expect(throws: (any Error).self) {
            try CatalogResultValidation.validate(data: WireJSON.encode(benchmark), path: "results/intake-renamed.json", folder: benchmarkFolder, manifest: manifest)
        }
    }

    @Test(arguments: ["implementation_identity", "load_identity"])
    func intakeBuildAndLoadIdentitiesAreExplicitAndStrict(field: String) throws {
        var root = try document(runPath)
        root["implementation_identity"] = .string(String(repeating: "a", count: 64))
        root["load_identity"] = "" // FLUX has no folder kernels or load settings.
        try validate(root, path: runPath)
        let valid = root
        root[field] = "bad-hash"
        #expect(throws: (any Error).self) { try validate(root, path: runPath) }
        root = valid; root[field] = true
        #expect(throws: (any Error).self) { try validate(root, path: runPath) }
        root = valid; root.removeValue(forKey: field)
        #expect(throws: (any Error).self) { try validate(root, path: runPath) }
    }

    @Test(arguments: ["qualityIdentity", "qualityOnlyUnavailable", "defaultWorkload"])
    func boundQualityAndWorkloadFieldsStayStrict(field: String) throws {
        var root = try document(tablePath)
        try validate(root, path: tablePath)
        var stats = try #require(root["stats"]?.object)
        if field == "defaultWorkload" {
            var availability = try #require(stats["availability"]?.object), provenance = try #require(availability["provenance"]?.object)
            provenance[field] = "true"
            availability["provenance"] = .object(provenance); stats["availability"] = .object(availability)
        } else {
            stats[field] = field == "qualityIdentity" ? "not-a-sha256" : "false"
        }
        root["stats"] = .object(stats)
        #expect(throws: (any Error).self) { try validate(root, path: tablePath) }
    }

    @Test(arguments: ["results/intake-2026-10-15.json", "results/intake-original-audio-2026-10-15.json"])
    func historicalScreeningsHaveRequiredRowsAndWorkloadKind(path: String) throws {
        let root = try document(path)
        try validate(root, path: path)
        var broken = root, policy = try #require(root["policy"]?.object)
        policy["fixture"] = ["width": 1024, "height": 1024, "prompt": "image", "steps": 4]; broken["policy"] = .object(policy)
        #expect(throws: (any Error).self) { try validate(broken, path: path) }
        broken = root; broken.removeValue(forKey: "worker_sha256")
        #expect(throws: (any Error).self) { try validate(broken, path: path) }
    }

    @Test(arguments: ["1e999", "NaN", "Infinity", "\"NaN\""])
    func nonfiniteResultNumbersFail(literal: String) throws {
        var root = try document(tablePath), stats = try #require(root["stats"]?.object)
        var metrics = try #require(stats["metrics"]?.object), peak = try #require(metrics["peakMemoryGB"]?.object)
        peak["value"] = "__NONFINITE__"; metrics["peakMemoryGB"] = .object(peak); stats["metrics"] = .object(metrics); root["stats"] = .object(stats)
        let text = String(decoding: try WireJSON.encode(root), as: UTF8.self)
        let corrupt = text.replacingOccurrences(of: "\"__NONFINITE__\"", with: literal)
        #expect(corrupt != text)
        #expect(throws: (any Error).self) {
            try CatalogResultValidation.validate(data: Data(corrupt.utf8), path: tablePath, folder: folder, manifest: ModelManifest.load(folder))
        }
    }
}
