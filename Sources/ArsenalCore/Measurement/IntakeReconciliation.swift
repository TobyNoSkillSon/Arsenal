// SPDX-License-Identifier: MIT
import Foundation

extension ModelIntakeCommand {
    /// Re-emit retained fixed-recipe evidence through the common boundary. No model execution,
    /// new metrics, or invented Load requests; missing identities require a real isolated run.
    static func reconcileQualification(_ args: [String]) throws {
        func option(_ key: String) -> String? { args.firstIndex(of: key).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil } }
        guard let modelPath = option("--model"), let bindingPath = option("--binding") else {
            throw ManifestError.invalid("Reconciliation needs --model and --binding")
        }
        let folder = URL(fileURLWithPath: modelPath), model = try ModelManifest.load(folder)
        guard model.kind != "voices" else { throw ManifestError.invalid("Searched voices retain their distinct held-out run") }
        let bindingURL = try SafePath.resolve(bindingPath, inside: folder)
        let oldBinding = try WireJSON.decode(Data(contentsOf: bindingURL))
        guard var outcomes = oldBinding["outcomes"]?.array, !outcomes.isEmpty,
            let resultPath = outcomes.first?["capture_result"]?.string,
            outcomes.allSatisfy({ $0["capture_result"]?.string == resultPath })
        else { throw ManifestError.invalid("Missing exact retained binding") }
        let oldResult = try SafePath.resolve(resultPath, inside: folder)
        var result = try WireJSON.decode(Data(contentsOf: oldResult))
        guard result["model"]?.string == model.id, let oldRun = result["run"]?.string,
            SafePath.isRelative(oldRun), !oldRun.contains("/"), let originalRows = result["rows"]?.array,
            outcomes.allSatisfy({ $0["capture_result_sha256"]?.string == (try? FileDigest.sha256(oldResult)) })
        else { throw ManifestError.invalid("Saved result differs from original binding") }
        let source = folder.appendingPathComponent("gate/evidence/intake/" + oldRun)
        let inventoryURL = source.appendingPathComponent("fixture-manifest.json")
        let inventory = try WireJSON.decode(Data(contentsOf: inventoryURL))
        guard let files = inventory["files"]?.array else { throw ManifestError.invalid("Missing captured inventory; rerun required") }
        func recorded(_ path: String) throws -> String {
            let matches = files.filter { $0["path"]?.string == path }
            guard matches.count == 1, let hash = matches[0]["sha256"]?.string else { throw ManifestError.invalid("Missing captured identity: \(path)") }
            return hash
        }
        // Verify every retained byte. Omitted raw tensors/diagnostics remain identified in the
        // original inventory; required bounds and final row metrics below must be present.
        for file in files {
            guard let path = file["path"]?.string, let hash = file["sha256"]?.string else { throw ManifestError.invalid("Invalid captured inventory") }
            let url = try SafePath.resolve(path, inside: source)
            if FileManager.default.fileExists(atPath: url.path), try FileDigest.sha256(url) != hash { throw ManifestError.invalid("Saved evidence hash differs: \(path)") }
        }
        let run = oldRun + "-qualification-" + UUID().uuidString.lowercased()
        let evidence = folder.appendingPathComponent("gate/evidence/intake/" + run)
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        var committed = false
        defer { if !committed { try? FileManager.default.removeItem(at: evidence) } }
        // JSON is sufficient to recheck metrics. No weights or raw tensors are duplicated.
        for file in files {
            let path = file["path"]!.string!
            let from = try SafePath.resolve(path, inside: source)
            if !path.hasSuffix(".json") || ["quality-binding.json", "receipt.json", "latest-pointer.json", "policy.json"].contains(path)
                || !FileManager.default.fileExists(atPath: from.path)
            {
                continue
            }
            let to = try SafePath.resolve(path, inside: evidence)
            try FileManager.default.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: from, to: to)
        }
        let inputHash = try recorded("fixture/inputs.safetensors")
        var rows: [JSONValue] = []
        for value in originalRows {
            guard var row = value.object, let dtype = row["precision"]?.string, var gate = row["gate"]?.object,
                let load = row["capture_load"]?.object ?? row["load"]?.object,
                load["requested_load"]?.object != nil, row["fixture_sha256"]?.string == inputHash
            else { throw ManifestError.invalid("Saved row has no exact input/ordinary Load identity; rerun required") }
            let boundsName = ["int8", "int4"].contains(dtype) ? "quantized-bounds.json" : "bounds.json"
            let bounds = evidence.appendingPathComponent(boundsName)
            guard gate["bounds_sha256"]?.string == (try recorded(boundsName)),
                try FileDigest.sha256(bounds) == gate["bounds_sha256"]?.string
            else { throw ManifestError.invalid("Original frozen bounds differ") }
            let identities = [
                "fixture_inputs_sha256": inputHash,
                "reference_outputs_sha256": try recorded("reference/outputs.safetensors"),
                "candidate_outputs_sha256": try recorded((dtype == model.basePrecision ? "reference-repeat" : dtype) + "/outputs.safetensors")
            ]
            for (key, hash) in identities {
                guard gate[key] == nil || gate[key] == .null || gate[key]?.string == hash else { throw ManifestError.invalid("Saved gate identity differs: \(key)") }
                gate[key] = .string(hash)
            }
            let planFile = evidence.appendingPathComponent("qualification-plan-" + (["int8", "int4"].contains(dtype) ? "quantized" : dtype) + ".json")
            let plan =
                FileManager.default.fileExists(atPath: planFile.path)
                ? try WireJSON.decode(Data(contentsOf: planFile))
                : try IntakeQualification.freezeRecorded(inputsSHA256: inputHash, selectionBounds: bounds, to: planFile)
            let qualification = try IntakeQualification.complete(plan: plan, selection: gate, heldout: nil, searched: false, ordinaryLoad: load)
            guard qualification["passed"] == row["qualified"] else { throw ManifestError.invalid("Reconciliation would change a saved verdict") }
            row["gate"] = .object(gate); row["selection_gate"] = .object(gate); row["qualification"] = .object(qualification)
            rows.append(.object(row))
        }
        result["rows"] = .array(rows); result["run"] = .string(run); result["capture_run"] = .string(oldRun)
        var policy = result["policy"]!.object!; policy["run"] = .string(run); result["policy"] = .object(policy)
        result["reconciliation"] = [
            "method":
                "Retained metrics through shared qualification; no model execution or new measurement. Original failures, dates, timings, memory and worker/load identities preserved.",
            "source_result": .string(resultPath), "source_result_sha256": .string(try FileDigest.sha256(oldResult)),
            "source_binding": .string("gate/evidence/intake/" + run + "/reconciliation-source-binding.json"),
            "source_binding_original_path": .string(bindingPath), "source_binding_sha256": .string(try FileDigest.sha256(bindingURL)),
            "source_inventory_sha256": .string(try FileDigest.sha256(inventoryURL))
        ]
        try Data(contentsOf: bindingURL).write(to: evidence.appendingPathComponent("reconciliation-source-binding.json"), options: .withoutOverwriting)
        let newPath = "results/intake/" + run + ".json", newResult = folder.appendingPathComponent(newPath)
        try WireJSON.encode(result).write(to: newResult, options: .withoutOverwriting)
        try WireJSON.encode(policy).write(to: evidence.appendingPathComponent("policy.json"))
        try WireJSON.encode(["run": .string(run), "capture_run": .string(oldRun), "results": .string(newPath)])
            .write(to: evidence.appendingPathComponent("receipt.json"))
        // Keep omitted raw hashes and replace only newly emitted JSON identities.
        var captured = Dictionary(uniqueKeysWithValues: files.map { ($0["path"]!.string!, $0["sha256"]!.string!) })
        let walker = FileManager.default.enumerator(at: evidence, includingPropertiesForKeys: nil)!
        for case let file as URL in walker where file.pathExtension == "json" {
            captured[String(file.path.dropFirst(evidence.path.count + 1))] = try FileDigest.sha256(file)
        }
        try WireJSON.encode(["files": .array(captured.keys.sorted().map { ["path": .string($0), "sha256": .string(captured[$0]!)] })])
            .write(to: evidence.appendingPathComponent("fixture-manifest.json"))
        try IntakeQualification.validateCompletion(rows: rows, model: model, evidence: evidence)
        var tables: [(URL, Data)] = []
        for index in outcomes.indices {
            var outcome = outcomes[index].object!
            let dtype = outcome["dtype"]!.string!, row = rows.first { $0["precision"]?.string == dtype }!
            let table = folder.appendingPathComponent(outcome["result"]!.string!)
            var document = try WireJSON.decode(Data(contentsOf: table)), stats = document["stats"]!.object!
            guard stats["qualityCheckPassed"] == row["qualified"], outcome["passed"] == row["qualified"] else { throw ManifestError.invalid("Saved table verdict differs") }
            for key in ["availability", "metrics"] {
                if key == "availability" {
                    stats[key] = try rebindProvenance(stats[key]!, oldRun: oldRun, run: run, path: newPath)
                } else {
                    stats[key] = .object(try stats[key]!.object!.mapValues { try rebindProvenance($0, oldRun: oldRun, run: run, path: newPath) })
                }
            }
            document["stats"] = .object(stats)
            let data = try WireJSON.encode(document); tables.append((table, data))
            outcome["result_sha256"] = .string(FileDigest.sha256(data)); outcome["capture_result"] = .string(newPath)
            outcome["capture_result_sha256"] = .string(try FileDigest.sha256(newResult)); outcome["qualification"] = row["qualification"]
            outcomes[index] = .object(outcome)
        }
        var binding = oldBinding
        binding["schema"] = 3; binding["qualification_policy"] = .string(IntakeQualification.policyID)
        binding["run"] = .string(run); binding["outcomes"] = .array(outcomes)
        binding["basis"] =
            "Re-emitted through shared fixed-recipe qualification from original frozen bounds and captured inventory; no new model execution or measurement. See result reconciliation provenance."
        let data = try WireJSON.encode(binding)
        for (file, data) in tables { try data.write(to: file, options: .atomic) }
        try data.write(to: evidence.appendingPathComponent("quality-binding.json"), options: .withoutOverwriting)
        let date = String(ISO8601DateFormatter().string(from: Date()).prefix(10))
        try data.write(to: folder.appendingPathComponent("gate/evidence/quality-binding-" + date + ".json"), options: .atomic)
        if model.kind == "sound" || model.kind == "enhancement" { try writeSoundLatestPointer(folder: folder, run: run) }
        committed = true
        print("\(model.id): re-emitted \(rows.count) unchanged tier verdicts; \(run)")
    }

    static func rebindProvenance(_ value: JSONValue, oldRun: String, run: String, path: String) throws -> JSONValue {
        guard var cell = value.object else { throw ManifestError.invalid("Invalid retained stat cell") }
        guard var provenance = cell["provenance"]?.object else {
            guard !["quickCheck", "measured"].contains(cell["state"]?.string ?? "") else { throw ManifestError.invalid("Recorded stat has no provenance") }
            return value
        }
        guard let originalRun = provenance["run"]?.string, originalRun.hasPrefix(oldRun + "-"), let source = provenance["source"]?.string else {
            throw ManifestError.invalid("Retained stat has no exact capture provenance")
        }
        provenance["run"] = .string(originalRun.replacingOccurrences(of: oldRun, with: run))
        let suffix = source.split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)
        provenance["source"] = .string(path + (suffix.count == 2 ? ";" + suffix[1] : ""))
        cell["provenance"] = .object(provenance); return .object(cell)
    }
}
