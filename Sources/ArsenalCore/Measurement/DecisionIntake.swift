// SPDX-License-Identifier: MIT
import Foundation

extension ModelIntakeCommand {
    static func writeCaptureManifest(_ evidence: URL) throws {
        let walker = FileManager.default.enumerator(at: evidence, includingPropertiesForKeys: [.isRegularFileKey])!
        var files: [JSONValue] = []
        for case let file as URL in walker where ["safetensors", "json"].contains(file.pathExtension) {
            if file.lastPathComponent == "fixture-manifest.json" { continue }
            files.append(["path": .string(String(file.path.dropFirst(evidence.path.count + 1))), "sha256": .string(try FileDigest.sha256(file))])
        }
        try WireJSON.encode(["files": .array(files)]).write(to: evidence.appendingPathComponent("fixture-manifest.json"), options: .withoutOverwriting)
    }

    static var d1ComponentKeys: Set<String> {
        var keys = Set(["projector", "vision/embedding"])
        for i in 0..<27 { keys.insert("vision/layer-\(i)") }
        for c in 0..<2 {
            keys.insert("case-\(c)/embedding"); keys.insert("case-\(c)/readout")
            for i in 0..<30 { keys.insert("case-\(c)/layer-\(i)") }
        }
        return keys
    }
    static func decisionGate(_ component: JSONObject, reference: URL, candidate: URL) throws -> JSONObject {
        let a = try IntakeTensor.read(reference.appendingPathComponent("probabilities.safetensors"))
        let b = try IntakeTensor.read(candidate.appendingPathComponent("probabilities.safetensors"))
        let shapes = ["case-0/question-0": [2], "case-1/question-0": [2], "case-1/question-1": [2]]
        guard Set(a.keys) == Set(shapes.keys), Set(b.keys) == Set(shapes.keys),
            shapes.allSatisfy({ a[$0.key]?.shape == $0.value && b[$0.key]?.shape == $0.value }),
            a.values.allSatisfy({ $0.values.allSatisfy(\.isFinite) }), b.values.allSatisfy({ $0.values.allSatisfy(\.isFinite) })
        else { throw ManifestError.invalid("Decision probability cases/keys/count/finiteness mismatch") }
        let delta = a.map { key, value in zip(value.values, b[key]!.values).map { abs($0 - $1) }.max()! }.max()!
        let unchanged = a.allSatisfy { key, value in
            let argmax = { (values: [Double]) in values.indices.max { values[$0] < values[$1] }! }
            return argmax(value.values) == argmax(b[key]!.values)
        }
        var gate = component
        gate["chosen_answers_unchanged"] = .bool(unchanged)
        gate["max_probability_delta"] = .number(delta); gate["probability_delta_max"] = 0.017
        gate["component_gate_passed"] = .bool(gate["component_gate_passed"]?.bool == true && delta <= 0.017 && unchanged)
        return gate
    }

    static func writeDecisionStats(rows: [JSONValue], model: ModelManifest, folder: URL, artifact: BundledWorkerArtifact, run: String, result: URL) throws {
        try IntakeQualification.validateCompletion(rows: rows, model: model, evidence: folder.appendingPathComponent("gate/evidence/intake/" + run))
        try writeCaptureManifest(folder.appendingPathComponent("gate/evidence/intake/" + run))
        guard let runner = model.runners.first else { throw ManifestError.invalid("Missing decision runner") }
        let implementation = artifact.qualityImplementation(model: model, runner: runner, directory: folder)
        let captured = try WireJSON.decode(Data(contentsOf: result))
        let loadIdentity = try ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder)
        guard captured["run"]?.string == run, captured["model"]?.string == model.id,
            captured["rows"]?.array == rows, captured["implementation_identity"]?.string == implementation,
            captured["load_identity"]?.string == loadIdentity,
            captured["worker_sha256"]?.string == (try FileDigest.sha256(artifact.helpers.appendingPathComponent("arsenal-worker-native")))
        else { throw ManifestError.invalid("Decision receipt does not match exact run/build/load recipe") }
        let date = captured["date"]?.string ?? ISO8601DateFormatter().string(from: Date())
        let resultHash = try FileDigest.sha256(result)
        var outcomes: [JSONValue] = []
        for value in rows {
            guard let row = value.object, let dtype = row["precision"]?.string, let seconds = row["warm"]?["median_s"]?.number,
                let peak = row["peak_gb"]?.number, let gate = row["gate"]?.object
            else { throw ManifestError.invalid("Incomplete decision intake row") }
            let qualification = try IntakeQualification.validatedRow(row, model: model)
            let source = "results/intake/" + result.lastPathComponent + "; bounds SHA256 " + (gate["bounds_sha256"]?.string ?? "missing")
            let isD1 = model.id == "d1-3b"
            let provenance = StatProvenance(
                chip: HostChip.current, date: date,
                method:
                    "Warm served ABBA BAAB; refund decision" + (isD1 ? ", 33 tokens" : "")
                    + ". Separate workers for component probes and default-workload Load/decision lifetime footprint. Busy Mac; quick check.",
                baseline: isD1 ? "Pinned BF16 served d1; previous text-only peak 6.032 GB" : "Pinned " + model.basePrecision.uppercased() + " served " + model.id, source: source,
                revision: model.weights.revision, run: run + "-" + dtype, gpuCores: HostChip.gpuCores, defaultWorkload: true)
            var stats = PrecisionStatSet(kind: .decisions, dtype: dtype, runner: runner.name)
            stats.qualityCheckPassed = qualification["passed"]?.bool
            stats.qualityIdentity = PrecisionEvidence.qualityIdentity(model: model, dtype: dtype, runner: runner.name, implementation: implementation, loadIdentity: loadIdentity)
            guard stats.qualityIdentity != nil else { throw ManifestError.invalid("Missing built worker source-closure receipt") }
            stats.availability = StatValue(
                state: dtype == "int8" && stats.qualityCheckPassed == false ? .unavailable : .quickCheck,
                reason: dtype == "int8" && stats.qualityCheckPassed == false ? "Did not pass the quality check" : nil, provenance: provenance)
            stats.set(.millisecondsPerDecision, StatValue(state: .quickCheck, value: seconds * 1000, provenance: provenance))
            stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: peak, provenance: provenance))
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let mode = try ModelLoadConfiguration.defaultFast(model: model, runner: runner, folder: folder) ? "fast" : "exact"
            let data = try encoder.encode(PrecisionStatsResult(model: model.id, mode: mode, stats: stats))
            _ = try PrecisionStatsResult.decode(data)
            let tablePath = "results/intake-m5-max-" + dtype + "-native-" + mode + ".json"
            try data.write(to: folder.appendingPathComponent(tablePath), options: .atomic)
            let boundsPath = "gate/evidence/intake/" + run + "/" + (dtype == model.basePrecision ? "bounds.json" : "quantized-bounds.json")
            guard gate["bounds_sha256"]?.string == (try FileDigest.sha256(folder.appendingPathComponent(boundsPath))) else {
                throw ManifestError.invalid("Decision binding bounds differ from exact run")
            }
            outcomes.append(
                .object([
                    "result": .string(tablePath), "result_sha256": .string(FileDigest.sha256(data)), "dtype": .string(dtype),
                    "qualification": .object(qualification),
                    "passed": .bool(stats.qualityCheckPassed == true), "quality_identity": .string(stats.qualityIdentity!),
                    "capture_result": .string("results/intake/" + result.lastPathComponent), "capture_result_sha256": .string(resultHash),
                    "bounds_sha256": gate["bounds_sha256"]!, "recipe": .string(model.derive?[dtype] ?? "vendor:" + dtype)
                ]))
        }
        let binding: JSONObject = [
            "schema": 3, "qualification_policy": .string(IntakeQualification.policyID), "family": .string(runner.worker ?? model.kind), "run": .string(run),
            "implementation_identity": .string(implementation!), "load_identity": .string(loadIdentity),
            "worker_sha256": captured["worker_sha256"]!, "outcomes": .array(outcomes),
            "basis": "Written by intake completion from this exact saved-input run, worker, load recipe, bounds and emitted stats."
        ]
        let bindingData = try WireJSON.encode(binding)
        try bindingData.write(to: folder.appendingPathComponent("gate/evidence/intake/" + run + "/quality-binding.json"), options: .withoutOverwriting)
        try bindingData.write(to: folder.appendingPathComponent("gate/evidence/quality-binding-" + String(date.prefix(10)) + ".json"), options: .atomic)
    }
}
