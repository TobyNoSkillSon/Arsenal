// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

extension ModelIntakeCommand {
    static func freezeComponentBounds(reference: URL, repeatCapture: URL, evidence: URL, half: String, componentKeys: Set<String>?, reused: Bool) throws {
        let floor = reused ? evidence.appendingPathComponent("reference-floor") : repeatCapture
        try IntakeComponentGate.freeze(reference: reference, repeatCapture: floor, to: evidence.appendingPathComponent("bounds.json"), half: half, expectedKeys: componentKeys)
        try IntakeComponentGate.freeze(
            reference: reference, repeatCapture: floor, to: evidence.appendingPathComponent("quantized-bounds.json"), half: half, quantized: true, expectedKeys: componentKeys)
    }

    static func reuseSoundReference(run: String?, model: ModelManifest, folder: URL, weights: URL, evidence: URL) throws {
        guard let run else { return }
        guard model.id == "stable-audio-3-small-sfx" || downloadedModels.contains(model.id), SafePath.isRelative(run), !run.contains("/") else {
            throw ManifestError.invalid("Invalid Small SFX reference run")
        }
        let source = folder.appendingPathComponent("gate/evidence/intake/" + run)
        let prior = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent(model.basePrecision + "-snapshot.json")))
        let current = try WireJSON.decode(Data(contentsOf: StoredNames.snapshotURL(in: weights)))
        func identities(_ snapshot: JSONObject) -> [JSONValue] {
            (snapshot["files"]?.array ?? []).filter { !vonModels.contains(model.id) || $0["path"]?.string != "manifest.json" }.sorted {
                ($0["path"]?.string ?? "") < ($1["path"]?.string ?? "")
            }
        }
        guard identities(prior) == identities(current), prior["model"] == current["model"], prior["revision"] == current["revision"] else {
            throw ManifestError.invalid("Saved reference baseline learned weights or sidecars differ")
        }
        for name in ["fixture", "reference"] {
            try FileManager.default.copyItem(at: source.appendingPathComponent(name), to: evidence.appendingPathComponent(name))
        }
        try FileManager.default.copyItem(at: source.appendingPathComponent("reference-repeat"), to: evidence.appendingPathComponent("reference-floor"))
        if downloadedModels.contains(model.id) {
            try FileManager.default.copyItem(at: source.appendingPathComponent("format-qualification"), to: evidence.appendingPathComponent("format-qualification"))
        }
        try WireJSON.encode([
            "source_run": .string(run),
            "basis":
                "Identical FP16 snapshot hashes; only Small SFX conversion receipt copying changed. Original immutable reference tensors/floor reused; the current FP16 implementation is replayed on those saved inputs."
        ])
        .write(to: evidence.appendingPathComponent("reused-reference.json"), options: .withoutOverwriting)
    }

    static func recordSoundCoverage(dtype: String, weights: URL, evidence: URL) throws {
        let files = try SafePath.files(in: weights).filter { $0.hasSuffix(".coverage.json") }
        guard files.count == 3 else { throw ManifestError.invalid("Expected Small SFX coverage for DiT, T5Gemma and SAME-S") }
        var receipts: JSONObject = [:]
        for relative in files {
            receipts[relative] = .object(try WireJSON.decode(Data(contentsOf: SafePath.resolve(relative, inside: weights))))
        }
        try WireJSON.encode(receipts).write(to: evidence.appendingPathComponent(dtype + "-component-coverage.json"), options: .withoutOverwriting)
    }

    static func writeCompletedStats(
        rows: [JSONValue], model: ModelManifest, folder: URL, artifact: BundledWorkerArtifact, run: String, result: URL, baseline: WorkerProcess, paths: ArsenalPaths
    ) async throws {
        if model.kind == "enhancement" || model.kind == "images" {
            try writeSoundStats(rows: rows, model: model, folder: folder, artifact: artifact, run: run, result: result)
        } else if model.kind == "sound" {
            let legacy = try await baseline.request(
                op: "generate_sound",
                params: [
                    "prompt": "rain on a tin roof", "duration": 1.5, "steps": 2, "seed": 20260927, "stream": false,
                    "out_dir": .string(paths.directory("jobs").appendingPathComponent(run + "/legacy-cast-sanity").path)
                ], timeout: 90)
            try WireJSON.encode(legacy).write(to: folder.appendingPathComponent("gate/evidence/intake/" + run + "/legacy-cast-sanity-reply.json"), options: .withoutOverwriting)
            try writeSoundStats(rows: rows, model: model, folder: folder, artifact: artifact, run: run, result: result)
        } else if model.kind == "decisions" {
            try writeDecisionStats(rows: rows, model: model, folder: folder, artifact: artifact, run: run, result: result)
        }
    }

    static func writeSoundStats(rows: [JSONValue], model: ModelManifest, folder: URL, artifact: BundledWorkerArtifact, run: String, result: URL) throws {
        try IntakeQualification.validateCompletion(rows: rows, model: model, evidence: folder.appendingPathComponent("gate/evidence/intake/" + run))
        try writeCaptureManifest(folder.appendingPathComponent("gate/evidence/intake/" + run))
        guard let runner = model.runners.first else { throw ManifestError.invalid("Missing sound runner") }
        let implementation = artifact.qualityImplementation(model: model, runner: runner, directory: folder)
        let captured = try WireJSON.decode(Data(contentsOf: result))
        let loadIdentity = try ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder)
        guard captured["run"]?.string == run, captured["model"]?.string == model.id,
            captured["rows"]?.array == rows, captured["implementation_identity"]?.string == implementation,
            captured["load_identity"]?.string == loadIdentity,
            captured["worker_sha256"]?.string == (try FileDigest.sha256(artifact.helpers.appendingPathComponent("arsenal-worker-native")))
        else { throw ManifestError.invalid("Sound receipt does not match exact run/build/load recipe") }
        let date = captured["date"]?.string ?? ISO8601DateFormatter().string(from: Date())
        let resultHash = try FileDigest.sha256(result)
        var outcomes: [JSONValue] = []
        for value in rows {
            guard let row = value.object, let dtype = row["precision"]?.string, let seconds = row["warm"]?["median_s"]?.number,
                let peak = row["peak_gb"]?.number, let gate = row["gate"]?.object
            else { throw ManifestError.invalid("Incomplete sound intake row") }
            let qualification = try IntakeQualification.validatedRow(row, model: model)
            if model.kind == "images", !model.precisions.contains(dtype) { continue }
            let source = "results/intake/" + result.lastPathComponent + "; bounds SHA256 " + (gate["bounds_sha256"]?.string ?? "missing")
            let provenance = StatProvenance(
                chip: HostChip.current, date: date,
                method:
                    model.kind == "images"
                    ? (model.id == "birefnet"
                        ? "Warm served remove_background ABBA BAAB, first fixed 512x384 image, default 1024x1024 preprocessing. "
                            + "Fresh ordinary Load/operation workers; no component captures. Busy Mac; quick check."
                        : "Warm served ABBA BAAB, default 1024x1024/four-step image. Fresh ordinary Load/generation workers; no component captures. Busy Mac; quick check.")
                    : model.kind == "enhancement"
                        ? "Warm served ABBA BAAB, three-second 24-kHz input, default enhancement settings. "
                            + "Fresh ordinary Load/enhance workers; no component captures. Busy Mac; quick check."
                        : "Warm served ABBA BAAB, five-second/eight-step sound. "
                            + "Lifetime physical footprint covers fresh ordinary Load/generation workers with no component captures. Busy Mac; quick check.",
                baseline: "Pinned \(model.basePrecision.uppercased()) served \(model.name); independent vendor qualification remains open", source: source,
                revision: model.weights.revision, run: run + "-" + dtype, gpuCores: HostChip.gpuCores, defaultWorkload: true)
            var stats = PrecisionStatSet(kind: model.kind == "images" ? .images : model.kind == "enhancement" ? .enhancement : .sound, dtype: dtype, runner: runner.name)
            stats.qualityCheckPassed = qualification["passed"]?.bool
            stats.qualityIdentity = PrecisionEvidence.qualityIdentity(model: model, dtype: dtype, runner: runner.name, implementation: implementation, loadIdentity: loadIdentity)
            guard stats.qualityIdentity != nil else { throw ManifestError.invalid("Missing built worker source-closure receipt") }
            stats.availability = StatValue(
                state: dtype == "int8" && stats.qualityCheckPassed == false ? .unavailable : .quickCheck,
                reason: dtype == "int8" && stats.qualityCheckPassed == false ? "Did not pass the quality check" : nil, provenance: provenance)
            stats.set(
                model.kind == "images" ? .secondsPerImage : .realTimeFactor,
                StatValue(state: .quickCheck, value: seconds / (model.kind == "images" ? 1 : model.kind == "enhancement" ? 3 : 5), provenance: provenance))
            stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: peak, provenance: provenance))
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let mode = try ModelLoadConfiguration.defaultFast(model: model, runner: runner, folder: folder) ? "fast" : "exact"
            let data = try encoder.encode(PrecisionStatsResult(model: model.id, mode: mode, stats: stats))
            _ = try PrecisionStatsResult.decode(data)
            let tablePath = "results/intake-m5-max-" + dtype + "-native-" + mode + ".json"
            try data.write(to: folder.appendingPathComponent(tablePath), options: .atomic)
            let boundsPath = "gate/evidence/intake/" + run + "/" + (dtype == model.basePrecision ? "bounds.json" : "quantized-bounds.json")
            guard gate["bounds_sha256"]?.string == (try FileDigest.sha256(folder.appendingPathComponent(boundsPath))) else {
                throw ManifestError.invalid("Sound binding bounds differ from exact run")
            }
            outcomes.append(
                .object([
                    "result": .string(tablePath), "result_sha256": .string(FileDigest.sha256(data)), "dtype": .string(dtype),
                    "qualification": .object(qualification),
                    "passed": stats.qualityCheckPassed.map(JSONValue.bool) ?? .null, "quality_identity": .string(stats.qualityIdentity!),
                    "scope": .string(
                        dtype == model.basePrecision
                            ? "Qualified 16-bit tier; saved-input reproducibility plus the recorded format acceptance basis"
                            : "Quantization versus the captured " + model.basePrecision.uppercased() + " baseline"),
                    "capture_result": .string("results/intake/" + result.lastPathComponent), "capture_result_sha256": .string(resultHash),
                    "bounds_sha256": gate["bounds_sha256"]!, "format_qualification_sha256": gate["format_qualification_sha256"] ?? .null,
                    "recipe": .string(model.derive?[dtype] ?? "vendor:" + dtype)
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
        try writeSoundLatestPointer(folder: folder, run: run)
    }
}

extension ModelIntakeCommand {
    static func prepareComponentGate(model: ModelManifest, path: String?, reference: URL, replay: URL, evidence: URL, half: String, keys: Set<String>?, reused: Bool) throws
        -> JSONObject
    {
        try freezeComponentBounds(reference: reference, repeatCapture: replay, evidence: evidence, half: half, componentKeys: keys, reused: reused)
        let gate = try IntakeComponentGate.evaluate(reference: reference, candidate: replay, bounds: evidence.appendingPathComponent("bounds.json"))
        if downloadedModels.contains(model.id) {
            let format = evidence.appendingPathComponent("format-qualification/format-gate.json")
            let qualification = try WireJSON.decode(Data(contentsOf: format))
            var combined = gate
            combined["format_qualification"] = .object(qualification)
            combined["format_qualification_sha256"] = .string(try FileDigest.sha256(format))
            let native = model.basePrecision
            var formatAccepted = qualification["component_gate_passed"]?.bool == true
            if native == "bf16" {
                let original = try IntakeTensor.read(evidence.appendingPathComponent("format-qualification/bf16-reference/outputs.safetensors"))
                let current = try IntakeTensor.read(reference.appendingPathComponent("outputs.safetensors"))
                formatAccepted =
                    qualification["component_gate_passed"]?.bool == false && Set(original.keys) == Set(current.keys)
                    && original.allSatisfy { key, value in current[key]?.shape == value.shape && current[key]?.values == value.values }
            }
            combined["selected_half"] = .string(native)
            combined["component_gate_passed"] = .bool(gate["component_gate_passed"]?.bool == true && formatAccepted)
            return combined
        }
        return try qualifySoundFormat(model: model, path: path, replay: replay, evidence: evidence, reproducibility: gate)
    }

    /// Freeze before running FP16; evaluate afterwards. Both operations are offline and load no model.
    static func soundFormatCommand(_ args: [String]) throws {
        guard let index = args.firstIndex(of: "--format-evidence"), args.indices.contains(index + 1) else {
            throw ManifestError.invalid("Sound format qualification needs --format-evidence")
        }
        let root = URL(fileURLWithPath: args[index + 1])
        let reference = root.appendingPathComponent("bf16-reference"), rounding = root.appendingPathComponent("bf16-rounding")
        let bounds = root.appendingPathComponent("bounds.json")
        if args.contains("--freeze-sound-format") {
            guard !FileManager.default.fileExists(atPath: root.appendingPathComponent("fp16-candidate").path) else {
                throw ManifestError.invalid("Freeze source calibration before executing any FP16 candidate")
            }
            try IntakeComponentGate.freezeFP32Source(
                reference: reference, roundingCapture: rounding, repeatCapture: root.appendingPathComponent("bf16-repeat"),
                to: bounds, expectedKeys: ["decoder", "seconds", "padding"])
            print("SAME-S: BF16 rounding error and FP16 activation-range bounds frozen")
        } else {
            let gate = try IntakeComponentGate.evaluate(reference: reference, candidate: root.appendingPathComponent("fp16-candidate"), bounds: bounds)
            try WireJSON.encode(gate).write(to: root.appendingPathComponent("format-gate.json"), options: .withoutOverwriting)
            print("SAME-S FP16 against BF16 rounding floor: \(gate["component_gate_passed"]?.bool == true ? "pass" : "fail (select BF16)")")
        }
    }

    static func qualifySoundFormat(model: ModelManifest, path: String?, replay: URL, evidence: URL, reproducibility: JSONObject) throws -> JSONObject {
        guard model.id == "stable-audio-3-small-sfx" else { return reproducibility }
        guard let path else { throw ManifestError.invalid("Small SFX intake needs saved --format-qualification evidence") }
        let source = URL(fileURLWithPath: path), target = evidence.appendingPathComponent("format-qualification")
        let receipt = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("source-integrity.json")))
        guard receipt["source_same_s_sha256"]?.string == "909928a8e6937c1ebe6ac4b729f0462bd3773704a11ea18278e42671dc69bfe4",
            receipt["source_dit_sha256"]?.string == "7e702d2640699a57fe436ca975fda16832040ba568c1e092c2ae826987558118",
            receipt["fixture_sha256"]?.string == (try FileDigest.sha256(evidence.appendingPathComponent("fixture/inputs.safetensors")))
        else { throw ManifestError.invalid("Sound format source or identical-input fixture does not match") }
        let frozen = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("bounds.json")))
        guard frozen["policy_kind"]?.string == IntakeComponentGate.fp32SourcePolicyID else {
            throw ManifestError.invalid("Small SFX requires the shared FP32-source qualification policy")
        }
        let gate = try IntakeComponentGate.evaluate(
            reference: source.appendingPathComponent("bf16-reference"), candidate: source.appendingPathComponent("fp16-candidate"),
            bounds: source.appendingPathComponent("bounds.json"))
        try IntakeComponentGate.validateSavedEvaluation(try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("format-gate.json"))), current: gate)
        // Tie the standalone source-cast probe back to this build's served component path.
        let candidate = try IntakeTensor.read(source.appendingPathComponent("fp16-candidate/outputs.safetensors"))
        let served = try IntakeTensor.read(replay.appendingPathComponent("outputs.safetensors"))
        for key in ["decoder", "seconds"] {
            guard let a = candidate[key], let b = served[key], a.shape == b.shape, a.values == b.values else {
                throw ManifestError.invalid("Served sound component differs from format-qualified output: " + key)
            }
        }
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        for file in ["bf16-reference", "bf16-rounding", "bf16-repeat", "fp16-candidate", "bounds.json", "format-gate.json", "source-integrity.json", "policy.json"] {
            try FileManager.default.copyItem(at: source.appendingPathComponent(file), to: target.appendingPathComponent(file))
        }
        if !FileManager.default.fileExists(atPath: target.appendingPathComponent("format-gate-source.json").path) {
            try FileManager.default.copyItem(at: target.appendingPathComponent("format-gate.json"), to: target.appendingPathComponent("format-gate-source.json"))
        }
        try WireJSON.encode(gate).write(to: target.appendingPathComponent("format-gate.json"), options: .atomic)
        var combined = reproducibility
        combined["format_qualification"] = .object(gate)
        combined["format_qualification_sha256"] = .string(try FileDigest.sha256(target.appendingPathComponent("format-gate.json")))
        combined["component_gate_passed"] = .bool(reproducibility["component_gate_passed"]?.bool == true && gate["component_gate_passed"]?.bool == true)
        guard combined["component_gate_passed"]?.bool == true else {
            throw ManifestError.invalid("FP16 source qualification failed; BF16 recipe required before quantized intake")
        }
        return combined
    }
}

extension ModelIntakeCommand {
    /// Repair only the discovery pointer from the exact completed binding; never rerun or rescore a model.
    static func soundPointerCommand(_ args: [String]) throws {
        func option(_ key: String) -> String? { args.firstIndex(of: key).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil } }
        guard let model = option("--model"), let run = option("--run") else { throw ManifestError.invalid("Sound pointer refresh needs --model and --run") }
        let folder = URL(fileURLWithPath: model)
        guard try ModelManifest.load(folder).kind == "sound" else { throw ManifestError.invalid("Expected a sound model") }
        try writeSoundLatestPointer(folder: folder, run: run)
        print("Sound intake pointer refreshed from verified completed binding: " + run)
    }

    static func writeSoundLatestPointer(folder: URL, run: String) throws {
        guard SafePath.isRelative(run), !run.contains("/") else { throw ManifestError.invalid("Invalid sound intake run") }
        let prefix = "gate/evidence/intake/" + run + "/", resultPath = "results/intake/" + run + ".json"
        let resultHash = try FileDigest.sha256(folder.appendingPathComponent(resultPath))
        let binding = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(prefix + "quality-binding.json")))
        let native = try ModelManifest.load(folder).basePrecision
        guard binding["run"]?.string == run, let outcomes = binding["outcomes"]?.array, outcomes.count == 3,
            Set(outcomes.compactMap { $0["dtype"]?.string }) == Set([native, "int8", "int4"])
        else { throw ManifestError.invalid("Incomplete sound outcome binding") }
        for row in outcomes {
            guard row["capture_result"]?.string == resultPath, row["capture_result_sha256"]?.string == resultHash,
                let result = row["result"]?.string, let dtype = row["dtype"]?.string,
                try FileDigest.sha256(SafePath.resolve(result, inside: folder)) == row["result_sha256"]?.string,
                try FileDigest.sha256(folder.appendingPathComponent(prefix + (dtype == native ? "bounds.json" : "quantized-bounds.json"))) == row["bounds_sha256"]?.string
            else { throw ManifestError.invalid("Sound pointer does not match the completed result, table stats or bounds") }
        }
        let bounds = prefix + "bounds.json", quant = prefix + "quantized-bounds.json"
        let data = try WireJSON.encode([
            "run": .string(run), "capture_run": .string(run), "results_path": .string(resultPath),
            "bounds_path": .string(bounds), "bounds_sha256": .string(try FileDigest.sha256(folder.appendingPathComponent(bounds))),
            "quantized_bounds_path": .string(quant), "quantized_bounds_sha256": .string(try FileDigest.sha256(folder.appendingPathComponent(quant)))
        ])
        try data.write(to: folder.appendingPathComponent(prefix + "latest-pointer.json"), options: .atomic)
        try data.write(to: folder.appendingPathComponent("gate/evidence/intake-latest.json"), options: .atomic)
    }
}
