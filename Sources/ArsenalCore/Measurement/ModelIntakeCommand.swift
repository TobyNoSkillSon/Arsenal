// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import Darwin

/// Agent-only offline qualification through the production Get/converter/worker interfaces.
public enum ModelIntakeCommand {
    public static func run(arguments: [String]) async -> Int32 {
        do {
            if arguments.contains("--download-source") {
                try await downloadIntakeSource(arguments)
            } else if arguments.contains("--refresh-sound-pointer") {
                try soundPointerCommand(arguments)
            } else if arguments.contains("--voice") {
                try await executeVoice(arguments)
            } else if arguments.contains("--freeze-sound-format") || arguments.contains("--evaluate-sound-format") {
                try soundFormatCommand(arguments)
            } else if arguments.contains("--reconcile-qualification") {
                try reconcileQualification(arguments)
            } else if arguments.contains("--rescore") {
                try rescore(arguments)
            } else {
                try await execute(arguments)
            }
            return 0
        } catch { FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8)); return 1 }
    }
    static func cloneDonor(model: ModelManifest, source: URL, lab: URL) throws {
        let donor = lab.appendingPathComponent("donor/" + model.weights.repo)
        try FileManager.default.createDirectory(at: donor, withIntermediateDirectories: true)
        // APFS clones are private lab inputs. Never write through the installed app's snapshot links.
        for relative in model.weights.files {
            let from = try SafePath.resolve(relative, inside: source).resolvingSymlinksInPath()
            let to = try SafePath.resolve(relative, inside: donor)
            try FileManager.default.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: to.path) {
                let process = Process(); process.executableURL = URL(fileURLWithPath: "/bin/cp")
                process.arguments = ["-c", from.path, to.path]; try process.run(); process.waitUntilExit()
                guard process.terminationStatus == 0 else { throw ManifestError.invalid("APFS lab clone failed") }
            }
            guard try FileDigest.sha256(from) == FileDigest.sha256(to) else { throw ManifestError.invalid("APFS donor clone digest differs: " + relative) }
        }
    }

    // Keep the intake lifecycle and its scoped worker cleanup together.
    // swiftlint:disable:next function_body_length
    static func execute(_ args: [String]) async throws {
        func option(_ key: String) -> String? { args.firstIndex(of: key).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil } }
        guard let folderPath = option("--model"), let sourcePath = option("--source"), let helperPath = option("--helpers"),
            let labPath = option("--support"), args.contains("--go")
        else {
            throw ManifestError.invalid("Intake needs --model, --source (existing pinned snapshot), --helpers, isolated --support and current-authority --go")
        }
        let folder = URL(fileURLWithPath: folderPath).standardizedFileURL, lab = URL(fileURLWithPath: labPath).standardizedFileURL
        guard lab.pathComponents.contains("tmp"), !lab.resolvingSymlinksInPath().path.contains("Library/Application Support") else {
            throw ManifestError.invalid("Intake support must be a lab tmp directory")
        }
        guard !FileManager.default.fileExists(atPath: lab.path) else { throw ManifestError.invalid("Use a fresh intake lab directory") }
        let original = try loadIntakeModel(folder)
        if original.id == "birefnet" { try await executeBiRefNet(args); return }
        let paths = ArsenalPaths(support: lab); try paths.create()
        defer {
            for name in ["donor", "weights", "cache"] { try? FileManager.default.removeItem(at: lab.appendingPathComponent(name)) }
        }
        let budget = min(60_000_000_000, UInt64(MemoryProbe().availableMB(loadedMB: 0) * 650_000))
        let memoryGuard = IntakeMemoryGuard(limit: budget); memoryGuard.start(); defer { memoryGuard.stop() }
        try cloneDonor(model: original, source: URL(fileURLWithPath: sourcePath), lab: lab)
        let artifact = BundledWorkerArtifact(helpers: URL(fileURLWithPath: helperPath))
        let converter = WorkerPrecisionConverter(artifacts: artifact, paths: paths)
        let fetcher = ModelFetcher(support: lab, converter: converter, localSources: [lab.appendingPathComponent("donor").path])
        let isImage = original.kind == "images", isDecision = original.kind == "decisions"
        let baselineTier = downloadedModels.contains(original.id) ? original.basePrecision : original.id == "d1-3b" ? "bf16" : isImage ? "bf16" : "fp16"
        let componentKeys = try decisionComponentKeys(original, source: URL(fileURLWithPath: sourcePath))
        let runID = "intake-" + ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-") + "-" + UUID().uuidString.lowercased()
        let evidence = folder.appendingPathComponent("gate/evidence/intake/" + runID)
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        let resultFile = folder.appendingPathComponent("results/intake/" + runID + ".json")
        try FileManager.default.createDirectory(at: resultFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fixtureDir = evidence.appendingPathComponent("fixture")
        let bounds = evidence.appendingPathComponent("bounds.json")
        let quantizedBounds = evidence.appendingPathComponent("quantized-bounds.json")
        let fixture: JSONObject =
            isImage
            ? ["prompt": "A red ceramic teapot on a wooden table, soft daylight, product photograph.", "width": 1024, "height": 1024, "steps": 4, "seed": 24301]
            : ["prompt": "Gentle rain on a window.", "duration": 5, "steps": 8, "seed": 42, "stream": false]
        let workload =
            isDecision
            ? defaultDecisionFixture
            : original.kind == "enhancement"
                ? try MeasurementWorkload.request(kind: .enhancement, model: original.id, batched: false, count: 1, directory: paths.directory("jobs")).params : fixture
        let policy = Self.policy(runID: runID, fixture: workload, isImage: isImage, model: original, componentKeys: componentKeys)
        try WireJSON.encode(policy).write(to: evidence.appendingPathComponent("policy.json"), options: .withoutOverwriting)
        var rows: [JSONValue] = []
        var snapshotAssets: [String: [String: URL]] = [:]
        var fp16Passed = false
        func snapshot(_ dtype: String) async throws -> URL {
            try memoryGuard.check(); let model = intakeModel(original: original, dtype: dtype, fp16Passed: fp16Passed)
            let weights = try await fetcher.get(model: model, modelDirectory: folder, precision: dtype, approvedDownloads: [:])
            _ = try await fetcher.verifyInstalled(model: model, precision: dtype, forceHash: true)
            try Data(contentsOf: StoredNames.snapshotURL(in: weights.weightsDirectory))
                .write(to: evidence.appendingPathComponent(dtype + "-snapshot.json"), options: .withoutOverwriting)
            snapshotAssets[dtype] = weights.assets
            if downloadedModels.contains(original.id) { try recordDownloadedCoverage(model: original, dtype: dtype, weights: weights.weightsDirectory, evidence: evidence) }
            if original.id == "stable-audio-3-small-sfx" { try recordSoundCoverage(dtype: dtype, weights: weights.weightsDirectory, evidence: evidence) }
            return weights.weightsDirectory
        }
        func start(_ dtype: String, weights: URL) async throws -> (WorkerProcess, JSONObject) {
            try await startWorker(
                dtype, weights: weights, original: original, artifact: artifact, folder: folder, paths: paths, memoryGuard: memoryGuard, isImage: isImage,
                assets: snapshotAssets[dtype] ?? [:])
        }
        func capture(_ process: WorkerProcess, name: String, create: Bool = false) async throws -> URL {
            let output = evidence.appendingPathComponent(name)
            let reused = (original.id == "stable-audio-3-small-sfx" || downloadedModels.contains(original.id)) && option("--reference-run") != nil && name == "reference"
            if !reused {
                _ = try await process.request(
                    op: "measurement_components", params: componentParams(fixture: fixtureDir, output: output, create: create, policy: policy), timeout: 120)
            }
            if !FileManager.default.fileExists(atPath: output.appendingPathComponent("fixture-identity.json").path) {
                let outputs = try IntakeTensor.read(output.appendingPathComponent("outputs.safetensors"))
                try WireJSON.encode([
                    "fixture_inputs_sha256": .string(try FileDigest.sha256(fixtureDir.appendingPathComponent("inputs.safetensors"))),
                    "component_count": .number(Double(outputs.count))
                ]).write(to: output.appendingPathComponent("fixture-identity.json"), options: .withoutOverwriting)
            }
            try memoryGuard.check(); return output
        }
        func generate(_ process: WorkerProcess, label: String) async throws -> (Double, JSONObject) {
            var request = workload
            request["out_dir"] = .string(paths.directory("jobs").appendingPathComponent(runID + "/" + label).path)
            let start = ProcessInfo.processInfo.systemUptime
            let reply = try await process.request(
                op: isDecision ? "decide" : isImage ? "generate" : original.kind == "enhancement" ? "enhance" : "generate_sound", params: request, timeout: 120)
            try memoryGuard.check(); return (ProcessInfo.processInfo.systemUptime - start, reply)
        }
        func persist() throws {
            try persistIntake(
                original: original, policy: policy, rows: rows, fp16Passed: fp16Passed,
                runID: runID, artifact: artifact, folder: folder, resources: memoryGuard.receipt(), resultFile: resultFile, evidence: evidence)
        }
        if args.contains("--derive-only") {
            for tier in isImage ? ["bf16", "fp16", "int8", "int4"] : [baselineTier, "int8", "int4"] { _ = try await snapshot(tier) }
            print("\(original.id): derived snapshots recorded at \(evidence.path); no model executed")
            await artifact.shutdown(); return
        }
        let baseWeights = try await snapshot(baselineTier)
        try reuseSoundReference(run: option("--reference-run"), model: original, folder: folder, weights: baseWeights, evidence: evidence)
        if downloadedModels.contains(original.id) && option("--reference-run") == nil {
            if let prior = option("--format-qualification") {
                try acceptDownloadedFormat(model: original, from: URL(fileURLWithPath: prior), evidence: evidence)
            } else {
                let fp16Trial = baselineTier == "fp16" ? baseWeights : try await snapshot("fp16")
                try await qualifyDownloadedSource(
                    model: original, source: URL(fileURLWithPath: sourcePath), evidence: evidence, lab: lab, artifact: artifact, paths: paths, keys: componentKeys!,
                    fp16: fp16Trial,
                    start: start)
            }
        }
        let (initialBaseline, captureBaselineLoad) = try await start(baselineTier, weights: baseWeights)
        var baseline = initialBaseline
        do {
            // First generation is cold and doubles as the one end-to-end sanity artifact. Never decides the precision gate.
            let (baseCold, baseSanity) = try await generate(baseline, label: baselineTier + "-sanity")
            let reference = try await capture(baseline, name: "reference", create: true)
            let repeatCapture = try await capture(baseline, name: "reference-repeat")
            var baseGate = try prepareComponentGate(
                model: original, path: option("--format-qualification"), reference: reference, replay: repeatCapture,
                evidence: evidence, half: baselineTier, keys: componentKeys, reused: option("--reference-run") != nil)
            if isDecision { baseGate = try decisionGate(baseGate, reference: reference, candidate: repeatCapture) }
            let basePlan = try IntakeQualification.freeze(
                selection: fixtureDir, selectionBounds: bounds, to: evidence.appendingPathComponent("qualification-plan-" + baselineTier + ".json"))
            let fixedPlan = try IntakeQualification.freeze(
                selection: fixtureDir, selectionBounds: quantizedBounds, to: evidence.appendingPathComponent("qualification-plan-quantized.json"))
            let baseQualification = try IntakeQualification.complete(
                plan: basePlan, selection: baseGate, heldout: nil, searched: false, ordinaryLoad: captureBaselineLoad)
            let fixtureHash = try FileDigest.sha256(fixtureDir.appendingPathComponent("inputs.safetensors"))
            // Lifetime high-water marks cannot be reset after capture. Every kind gets a fresh serving process.
            let (servingBaseline, baselineLoad) = try await freshServingWorker(capture: baseline) { try await start(baselineTier, weights: baseWeights) }
            baseline = servingBaseline
            var baselineSamples: [Double] = []
            var baseRow: JSONObject = [
                "precision": .string(baselineTier), "gate": .object(baseGate), "qualified": baseQualification["passed"]!,
                "qualification": .object(baseQualification), "selection_gate": .object(baseGate),
                "cold_seconds": .number(baseCold),
                "load": .object(baselineLoad), "capture_load": .object(captureBaselineLoad), "sanity_reply": .object(baseSanity), "fixture_sha256": .string(fixtureHash),
                "state": "quickCheck"
            ]
            if !isImage { fp16Passed = baselineTier == "fp16" && baseGate["component_gate_passed"]?.bool == true }
            for tier in (isImage ? ["fp16", "int8", "int4"] : ["int8", "int4"]) {
                let weights = try await snapshot(tier)
                let initialCandidate = try await start(tier, weights: weights)
                var candidate = initialCandidate.0
                let captureLoad = initialCandidate.1
                do {
                    let (cold, sanity) = try await generate(candidate, label: tier + "-sanity")
                    let captureDir = try await capture(candidate, name: tier)
                    guard try FileDigest.sha256(fixtureDir.appendingPathComponent("inputs.safetensors")) == fixtureHash else {
                        throw ManifestError.invalid("Immutable fixture changed")
                    }
                    var gate = try IntakeComponentGate.evaluate(reference: reference, candidate: captureDir, bounds: tier == "fp16" ? bounds : quantizedBounds)
                    if isDecision {
                        gate = try decisionGate(gate, reference: reference, candidate: captureDir)
                    }
                    let candidatePlan: JSONObject
                    if tier == "fp16" {
                        candidatePlan = try IntakeQualification.freeze(
                            selection: fixtureDir, selectionBounds: bounds, to: evidence.appendingPathComponent("qualification-plan-fp16.json"))
                    } else {
                        candidatePlan = fixedPlan
                    }
                    let qualification = try IntakeQualification.complete(
                        plan: candidatePlan, selection: gate, heldout: nil, searched: false, ordinaryLoad: captureLoad)
                    let (servingCandidate, loaded) = try await freshServingWorker(capture: candidate) { try await start(tier, weights: weights) }
                    candidate = servingCandidate
                    if tier == "fp16" { fp16Passed = gate["component_gate_passed"]?.bool == true }
                    // Warm fresh ordinary serving workers, then use balanced same-session windows.
                    _ = try await generate(baseline, label: tier + "-A-warm")
                    _ = try await generate(candidate, label: tier + "-B-warm")
                    var a: [Double] = [], b: [Double] = []
                    for (i, arm) in [0, 1, 1, 0, 1, 0, 0, 1].enumerated() {
                        let (seconds, _) = try await generate(arm == 0 ? baseline : candidate, label: tier + "-window-\(i)-\(arm)")
                        if arm == 0 { a.append(seconds) } else { b.append(seconds) }
                    }
                    baselineSamples += a
                    let startMedian = IntakeTimingSummary(samples: Array(a.prefix(2))).median
                    let endMedian = IntakeTimingSummary(samples: Array(a.suffix(2))).median
                    var row: JSONObject = [
                        "precision": .string(tier), "gate": .object(gate), "qualified": .bool(gate["component_gate_passed"]?.bool == true),
                        "peak_gb": .number(try await servingPeak(candidate)),
                        "serving_worker": .object(await servingWorkerReceipt(candidate)),
                        "cold_seconds": .number(cold), "load": .object(loaded), "capture_load": .object(captureLoad), "sanity_reply": .object(sanity),
                        "warm": IntakeTimingSummary(samples: b).json, "paired_baseline_warm": IntakeTimingSummary(samples: a).json,
                        "baseline_drift_fraction": .number(endMedian / startMedian - 1), "fixture_sha256": .string(fixtureHash), "state": "quickCheck"
                    ]
                    row["qualification"] = .object(qualification); row["selection_gate"] = .object(gate)
                    row["qualified"] = qualification["passed"]
                    rows.append(.object(row.merging(precisionOffering(dtype: tier, passed: gate["component_gate_passed"]?.bool)) { _, new in new })); try persist()
                    print(
                        "\(original.id) \(tier): component gate \(gate["component_gate_passed"]?.bool == true ? "pass" : "fail"); warm median \(IntakeTimingSummary(samples: b).median) s"
                    )
                    await candidate.unload()
                } catch { await candidate.unload(); throw error }
            }
            baseRow["serving_worker"] = .object(await servingWorkerReceipt(baseline))
            baseRow["peak_gb"] = .number(try await servingPeak(baseline)); baseRow["warm"] = IntakeTimingSummary(samples: baselineSamples).json
            rows.insert(.object(baseRow), at: 0)
            try persist()
            try await writeCompletedStats(rows: rows, model: original, folder: folder, artifact: artifact, run: runID, result: resultFile, baseline: baseline, paths: paths)
            await baseline.unload(); await artifact.shutdown()
            print("\(original.id): immutable evidence \(runID); independent reference qualification still open")
        } catch {
            await baseline.unload(); await artifact.shutdown()
            rows.append(["error": .string(error.localizedDescription)]); try? persist(); throw error
        }
    }

    private static func loadIntakeModel(_ folder: URL) throws -> ModelManifest {
        let original = try ModelManifest.load(folder)
        guard
            ["flux2-klein-4b", "stable-audio-3-medium", "stable-audio-3-small-sfx", "d1-3b", "birefnet"].contains(original.id) || layaModels.contains(original.id)
                || downloadedModels.contains(original.id)
        else {
            throw ManifestError.invalid("A new architecture needs an adapter and independent reference before intake")
        }
        return original
    }

    private static func intakeModel(original: ModelManifest, dtype: String, fp16Passed: Bool) -> ModelManifest {
        // Laya keeps its catalog recipe: the decisions worker quantizes at load from the FP16 view.
        if downloadedModels.contains(original.id), dtype == "fp16", original.basePrecision == "bf16" {
            // Make FP16 the trial's base so Get casts the pinned FP32 source directly, never the selected BF16 view.
            var trial = original
            trial.derive = (trial.derive ?? [:]).merging(["fp16": vonModels.contains(original.id) ? "cast:von-half-2" : "cast:re-use"]) { _, b in b }
            return trial
        }
        if layaModels.contains(original.id) || downloadedModels.contains(original.id) { return original }
        let isImage = original.kind == "images", isDecision = original.id == "d1-3b"
        var model = original
        if !isDecision && (dtype == "fp16" || !isImage || fp16Passed) { model.derive = (model.derive ?? [:]).merging(["fp16": "cast:" + original.id]) { _, b in b } }
        let soundRecipe = original.id == "stable-audio-3-small-sfx" ? "intake-v2:" : "intake-v1:"
        model.derive = (model.derive ?? [:]).merging([
            "int8": (isImage || isDecision) ? "mlx-affine-g64:" + original.id : soundRecipe + original.id,
            "int4": (isImage || isDecision) ? "mlx-affine-g64:" + original.id : soundRecipe + original.id
        ]) { _, b in b }
        return model
    }

    private static func persistIntake(
        original: ModelManifest, policy: JSONObject, rows: [JSONValue], fp16Passed: Bool,
        runID: String, artifact: BundledWorkerArtifact, folder: URL, resources: JSONObject, resultFile: URL, evidence: URL
    ) throws {
        guard let runner = original.runners.first,
            let implementation = artifact.qualityImplementation(model: original, runner: runner, directory: folder)
        else {
            throw ManifestError.invalid("Missing worker implementation receipt")
        }
        let document: JSONObject = [
            "implementation_identity": .string(implementation),
            "load_identity": .string(try ModelLoadConfiguration.identity(model: original, runner: runner, folder: folder)),
            "model": .string(original.id), "revision": .string(original.weights.revision), "run": .string(runID),
            "date": .string(ISO8601DateFormatter().string(from: Date())), "chip": .string(HostChip.current), "gpu_cores": .number(Double(HostChip.gpuCores)),
            "worker_sha256": .string(try FileDigest.sha256(artifact.helpers.appendingPathComponent("arsenal-worker-native"))),
            "policy": .object(policy), "rows": .array(rows), "fp16_selected": .bool(original.kind == "images" && fp16Passed), "fp16_component_passed": .bool(fp16Passed),
            "network_bytes": 0,
            "resource_receipt": .object(resources),
            "note": .string(
                "Qualification uses saved identical component inputs and the declared format/quantized-tier policy. "
                    + "Catalog recipe enablement is an explicit metadata step; independent vendor qualification stays separate.")
        ]
        try WireJSON.encode(document).write(to: resultFile, options: .atomic)
        try WireJSON.encode(["run": .string(runID), "results": .string("results/intake/" + runID + ".json"), "evidence": .string("gate/evidence/intake/" + runID)])
            .write(to: evidence.appendingPathComponent("receipt.json"), options: .atomic)
    }

    /// Re-evaluate saved identical-input captures without loading weights or starting a worker.
    private static func rescore(_ args: [String]) throws {
        func option(_ key: String) -> String? { args.firstIndex(of: key).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil } }
        guard let path = option("--model"), let run = option("--run"), SafePath.isRelative(run), !run.contains("/") else {
            throw ManifestError.invalid("Rescore needs --model and --run (saved intake ID)")
        }
        let folder = URL(fileURLWithPath: path), model = try ModelManifest.load(folder)
        let source = folder.appendingPathComponent("gate/evidence/intake/" + run)
        var result = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("results/intake/" + run + ".json")))
        let manifest = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("fixture-manifest.json")))
        guard let capturedFiles = manifest["files"]?.array, !capturedFiles.isEmpty else { throw ManifestError.invalid("Missing saved capture hashes") }
        for entry in capturedFiles {
            guard let path = entry["path"]?.string, let hash = entry["sha256"]?.string,
                try FileDigest.sha256(SafePath.resolve(path, inside: source)) == hash
            else { throw ManifestError.invalid("Saved capture hash mismatch") }
        }
        let id = run + "-rescore-" + UUID().uuidString.lowercased()
        let target = folder.appendingPathComponent("gate/evidence/intake/" + id)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        let half = model.basePrecision, reference = source.appendingPathComponent("reference"), again = source.appendingPathComponent("reference-repeat")
        let format = target.appendingPathComponent("bounds.json"), quant = target.appendingPathComponent("quantized-bounds.json")
        let savedPolicy = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("policy.json")))
        let keys =
            model.id == "d1-3b"
            ? d1ComponentKeys
            : model.id == "re-use" ? try downloadedComponentKeys(model, source: folder) : savedPolicy["component_keys"]?.array.map { Set($0.compactMap(\.string)) }
        try IntakeComponentGate.freeze(reference: reference, repeatCapture: again, to: format, half: half, expectedKeys: keys)
        try IntakeComponentGate.freeze(reference: reference, repeatCapture: again, to: quant, half: half, quantized: true, expectedKeys: keys)
        let basePlan = try IntakeQualification.freeze(
            selection: source.appendingPathComponent("fixture"), selectionBounds: format, to: target.appendingPathComponent("qualification-plan-" + half + ".json"))
        let quantPlan = try IntakeQualification.freeze(
            selection: source.appendingPathComponent("fixture"), selectionBounds: quant, to: target.appendingPathComponent("qualification-plan-quantized.json"))
        // A rescore is a complete qualification record even when it does not publish table stats.
        for name in ["fixture", "reference", "reference-repeat", "fp16", "int8", "int4", "format-qualification"] {
            let from = source.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: from.path) { try FileManager.default.copyItem(at: from, to: target.appendingPathComponent(name)) }
        }
        let rows = try result["rows"]?.array?.map { value -> JSONValue in
            guard var row = value.object, let dtype = row["precision"]?.string else { throw ManifestError.invalid("Malformed saved row") }
            let quantized = ["int8", "int4"].contains(dtype)
            let capture = source.appendingPathComponent(dtype == half ? "reference-repeat" : dtype)
            var gate = try IntakeComponentGate.evaluate(reference: reference, candidate: capture, bounds: quantized ? quant : format)
            if model.kind == "decisions" { gate = try decisionGate(gate, reference: reference, candidate: capture) }
            if dtype == half, let prior = row["gate"]?.object, let format = prior["format_qualification"] {
                gate["format_qualification"] = format; gate["format_qualification_sha256"] = prior["format_qualification_sha256"]
                gate["selected_half"] = prior["selected_half"]
                gate["component_gate_passed"] = .bool(gate["component_gate_passed"]?.bool == true && prior["component_gate_passed"]?.bool == true)
            }
            let plan: JSONObject
            if !quantized && dtype != half {
                plan = try IntakeQualification.freeze(
                    selection: source.appendingPathComponent("fixture"), selectionBounds: format, to: target.appendingPathComponent("qualification-plan-" + dtype + ".json"))
            } else {
                plan = quantized ? quantPlan : basePlan
            }
            if model.id == "birefnet" { row["fixture_sha256"] = row["gate"]?["fixture_inputs_sha256"] }
            guard row["fixture_sha256"] == plan["selection"]?["inputs_sha256"],
                gate["fixture_inputs_sha256"] == nil || gate["fixture_inputs_sha256"] == .null || gate["fixture_inputs_sha256"] == row["fixture_sha256"]
            else { throw ManifestError.invalid("Saved rescore input identity differs") }
            gate["fixture_inputs_sha256"] = row["fixture_sha256"]
            let qualification = try IntakeQualification.complete(
                plan: plan, selection: gate, heldout: nil, searched: false, ordinaryLoad: row["capture_load"]?.object ?? row["load"]?.object)
            row["gate"] = .object(gate); row["qualified"] = qualification["passed"]
            row["qualification"] = .object(qualification); row["selection_gate"] = .object(gate)
            row["offered"] = .bool(offered(dtype: dtype, passed: gate["component_gate_passed"]?.bool))
            row["precisionQualityWarnings"] = .object(Self.qualityWarnings(dtype: dtype, passed: gate["component_gate_passed"]?.bool))
            row["capture_run"] = .string(run)
            return .object(row)
        }
        guard let rows, rows.count == (model.kind == "images" && model.id != "birefnet" ? 4 : 3),
            model.id == "birefnet" || result["policy"]?["fixture"]?.object != nil
        else { throw ManifestError.invalid("Incomplete saved intake") }
        result["rows"] = .array(rows); result["run"] = .string(id); result["capture_run"] = .string(run)
        var rescored =
            model.id == "birefnet"
            ? savedPolicy
            : policy(
                runID: id, fixture: result["policy"]?["fixture"]?.object ?? [:], isImage: model.kind == "images", model: model, componentKeys: keys)
        rescored["run"] = .string(id)
        // The captured states are part of the saved fixture; a rescore never restates them.
        rescored["component_states"] = savedPolicy["component_states"]
        result["policy"] = .object(rescored)
        result["note"] = "Host-only rescore of saved identical-input tensors and activation traces. Original run and timings preserved."
        let resultPath = "results/intake/" + id + ".json"
        try WireJSON.encode(result).write(to: folder.appendingPathComponent(resultPath), options: .withoutOverwriting)
        try WireJSON.encode(["run": .string(id), "capture_run": .string(run), "results": .string(resultPath)])
            .write(to: target.appendingPathComponent("receipt.json"), options: .withoutOverwriting)
        try WireJSON.encode(rescored).write(to: target.appendingPathComponent("policy.json"), options: .withoutOverwriting)
        if args.contains("--write-stats") {
            guard let helpers = option("--helpers"), model.kind == "enhancement" || model.id == "birefnet" else {
                throw ManifestError.invalid("Saved stats completion needs --helpers and an enhancement or background-removal model")
            }
            let artifact = BundledWorkerArtifact(helpers: URL(fileURLWithPath: helpers))
            try writeSoundStats(rows: rows, model: model, folder: folder, artifact: artifact, run: id, result: folder.appendingPathComponent(resultPath))
        } else {
            try writeCaptureManifest(target)
        }
        print(resultPath)
    }

    static let defaultDecisionFixture: JSONObject = [
        "state": "The customer asks for a refund.", "questions": ["refund": ["type": "noul", "instructions": "Does the customer ask for a refund?"]]
    ]

    static let layaModels: Set<String> = ["laya-english", "laya-multilingual", "laya-typed-decisions"]

    /// A model is graded only on inputs in its supported languages: English-only Laya models get two English states;
    /// Multilingual's second state is German.
    static func layaStates(_ model: ModelManifest) -> [String] {
        let second =
            model.languages.contains("en") && model.languages.count == 1
            ? "The package has not arrived. Please check the delivery." : "Das Paket ist nicht angekommen. Bitte prüfen Sie die Lieferung."
        return ["The customer asks for a refund.", second]
    }

    static func componentParams(fixture: URL, output: URL, create: Bool, policy: JSONObject) -> JSONObject {
        var params: JSONObject = ["fixture_dir": .string(fixture.path), "output_dir": .string(output.path), "capture": .bool(create)]
        if let states = policy["component_states"] { params["states"] = states }
        return params
    }

    /// Exact component keys a decision intake must capture; nil for image/sound pilots (their five-component check).
    static func decisionComponentKeys(_ model: ModelManifest, source: URL) throws -> Set<String>? {
        if downloadedModels.contains(model.id) { return try downloadedComponentKeys(model, source: source) }
        if model.id == "d1-3b" { return d1ComponentKeys }
        if model.id == "stable-audio-3-small-sfx" { return Set(["prompt", "seconds", "denoiser-0", "denoiser-3", "denoiser-7", "decoder"]) }
        guard layaModels.contains(model.id) else { return nil }
        let encoder = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("encoder/config.json")))
        let agent = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("rl_agent_config.json")))
        guard let layers = encoder["num_hidden_layers"]?.number, layers > 0, let heads = agent["head_layers"]?.number, heads >= 0 else {
            throw ManifestError.invalid("Laya encoder/head configuration missing")
        }
        return layaComponentKeys(layers: Int(layers), headLayers: Int(heads))
    }
    /// Mirrors LayaModel.intakeComponentKeys in the decisions worker.
    static func layaComponentKeys(layers: Int, headLayers: Int) -> Set<String> {
        var keys = Set<String>()
        for c in 0..<2 {
            keys.insert("case-\(c)/embedding"); keys.insert("case-\(c)/scorer")
            for i in 0..<layers { keys.insert("case-\(c)/layer-\(i)") }
            for i in 0..<headLayers { keys.insert("case-\(c)/head-\(i)") }
        }
        return keys
    }

    static func precisionOffering(dtype: String, passed: Bool?) -> JSONObject {
        ["offered": .bool(offered(dtype: dtype, passed: passed)), "precisionQualityWarnings": .object(qualityWarnings(dtype: dtype, passed: passed))]
    }

    /// int4 is always offered; int8 unless its check failed; a 16-bit format candidate only when it passed.
    static func offered(dtype: String, passed: Bool?) -> Bool {
        dtype == "int4" || (dtype == "int8" ? passed != false : passed == true)
    }

    static func qualityWarnings(dtype: String, passed: Bool?) -> JSONObject {
        guard dtype == "int4", passed != true else { return [:] }
        return ["int4": .string(passed == false ? PrecisionStatSet.int4QualityWarning : PrecisionStatSet.int4UncheckedWarning)]
    }

    private static func policy(runID: String, fixture: JSONObject, isImage: Bool, model: ModelManifest, componentKeys: Set<String>? = nil) -> JSONObject {
        if model.id == "re-use" { return enhancementIntakePolicy(runID: runID, fixture: fixture, model: model, keys: componentKeys) }
        let isDecision = model.id == "d1-3b"
        if layaModels.contains(model.id) || vonModels.contains(model.id) {
            return [
                "component_keys": .array((componentKeys ?? []).sorted().map(JSONValue.string)),
                "component_states": .array(layaStates(model).map(JSONValue.string)),
                "run": .string(runID), "fixture": .object(fixture), "timing_order": "ABBA BAAB (four warm synchronized requests per arm); baseline resident across pairs",
                "format_gate": .string(
                    "Teacher-forced components on two unpadded one-row cases in the model's supported languages (yes/no, two-option choice; see "
                        + "component_states): embedding, every encoder layer, every head layer, scorer; three served-path decision probability vectors, "
                        + "max |dp| <= 0.017 and unchanged answers"),
                "quantized_tier_gate": .string(IntakeComponentGate.quantizedBasis),
                "reference_scope": .string(
                    "Pinned \(model.basePrecision.uppercased()) served \(model.id) on the default Load path. "
                        + (vonModels.contains(model.id)
                            ? "Von source-format calibration uses GPU BF16 nearest/truncation casts and a separate FP16 candidate; BF16 serving requires its frozen rejection. "
                            : "Laya retains its earlier qualification. ")
                        + "No new vendor qualification."
                ),
                "independent_reference_qualified": false, "machine_in_use": true
            ]
        }
        return [
            "run": .string(runID), "fixture": .object(fixture), "timing_order": "ABBA BAAB (four warm synchronized requests per arm); baseline resident across pairs",
            "format_gate": .string(
                isDecision
                    ? "93 teacher-forced components: both layouts, 30 language/27 vision blocks, embeddings, projector/readout; whole-decision max |dp| <= 0.017"
                    : (model.id == "stable-audio-3-small-sfx" ? "Six" : "Five")
                        + " teacher-forced components; A/A numerical floor + one reference-format ULP; internal activation range/finiteness/clamp checks"),
            "quantized_tier_gate": .string(IntakeComponentGate.quantizedBasis),
            "reference_scope": .string(
                isDecision
                    ? "Pinned BF16 served d1; historically independent-qualified runner. Quantization check against BF16; no new vendor qualification."
                    : isImage
                        ? "Original pinned BF16 served implementation. Independent mflux capture is constrained, not stock-qualified."
                        : model.id == "stable-audio-3-small-sfx"
                            ? "FP32-source SAME-S and learned conditioning qualify FP16 against GPU BF16 casts within their frozen BF16 rounding floors. "
                                + "No FP32 reference execution; independent vendor parity remains open."
                            : "Native FP16 weights, including SAME-L cast before execution. SAME-L accepted on finite/no-clamp evidence and "
                                + "retained original PCM (spectral relative L2 0.0157; waveform 0.0212); no FP32 reference execution. Independent vendor parity remains open."
            ),
            "independent_reference_qualified": false, "machine_in_use": true
        ]
    }

    static func defaultLoadParameters(
        model: ModelManifest, runner: RunnerSpec, folder: URL, weights: URL,
        assets: [String: URL], chip: String = HostChip.current
    ) throws -> JSONObject {
        let fast = try ModelLoadConfiguration.defaultFast(model: model, runner: runner, folder: folder)
        return [
            "model_dir": .string(folder.path), "weights_dir": .string(weights.path), "dtype": .string(model.basePrecision),
            "fast": .bool(fast),
            "config": .object(
                try ModelLoadConfiguration.config(
                    model: model, folder: folder, assets: assets, chip: chip, fast: fast, settings: ModelLoadConfiguration.defaults(model),
                    maximumContext: model.kind == "chat" ? ModelContext.maximum(snapshot: weights, model: model) : nil))
        ]
    }

    static func startWorker(
        _ dtype: String, weights: URL, original: ModelManifest, artifact: BundledWorkerArtifact, folder: URL, paths: ArsenalPaths, memoryGuard: IntakeMemoryGuard, isImage: Bool,
        assets: [String: URL]
    ) async throws -> (WorkerProcess, JSONObject) {
        guard let runner = original.runners.first else { throw ManifestError.invalid("Missing worker") }
        let process = WorkerProcess()
        do {
            try memoryGuard.check()
            let available = MemoryProbe().availableMB(loadedMB: 0)
            guard available > (isImage ? 25_000 : 12_000) else { throw ManifestError.invalid("Insufficient RAM for paired resident intake; no worker started") }
            let launch = try await artifact.launch(model: original, runner: runner, directory: folder, paths: paths)
            try await process.start(launch)
            var params = try defaultLoadParameters(model: original, runner: runner, folder: folder, weights: weights, assets: assets)
            params["dtype"] = .string(dtype)
            var loaded = try await process.request(op: "load", params: params, timeout: 120)
            // Keep the exact request alongside worker-reported selected paths and fallback reasons.
            loaded["requested_load"] = .object(params)
            return (process, loaded)
        } catch { await process.unload(); throw error }
    }

    static func freshServingWorker(capture: WorkerProcess, start: () async throws -> (WorkerProcess, JSONObject)) async throws -> (WorkerProcess, JSONObject) {
        await capture.unload()
        return try await start()
    }

    static func servingWorkerReceipt(_ process: WorkerProcess) async -> JSONObject {
        ["pid": .number(Double(await process.pid)), "measurement_components_requests": .number(Double(await process.componentCaptureRequests))]
    }

    static func servingPeak(_ process: WorkerProcess) async throws -> Double {
        guard await process.componentCaptureRequests == 0 else { throw ManifestError.invalid("Serving peak worker received component captures") }
        let status = try await process.request(op: "status", params: [:], timeout: 10)
        guard let mb = status["memory"]?["footprint_peak_mb"]?.number, mb > 0 else { throw ManifestError.invalid("Missing worker physical-footprint peak") }
        return mb / 1000
    }

}

/// Hard process-footprint budget and swap-growth stop; only this command's registered children are terminated.
final class IntakeMemoryGuard: @unchecked Sendable {
    private let lock = NSLock()
    private var reason: String?
    private var timer: DispatchSourceTimer?
    private let limit: UInt64
    private let started = ProcessInfo.processInfo.systemUptime
    private let deadline = ProcessInfo.processInfo.systemUptime + 590
    private var peakFootprint: UInt64 = 0
    private var peakSwap: UInt64 = 0
    private let initialSwap = IntakeMemoryGuard.swapBytes()
    init(limit: UInt64) { self.limit = limit }
    private static func swapBytes() -> UInt64 {
        var info = xsw_usage(); var length = MemoryLayout<xsw_usage>.size
        return sysctlbyname("vm.swapusage", &info, &length, nil, 0) == 0 ? info.xsu_used : 0
    }
    func start() {
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now(), repeating: .milliseconds(100))
        timer.setEventHandler { [weak self] in self?.sample() }; self.timer = timer; timer.resume()
    }
    func stop() { timer?.cancel(); timer = nil }
    private func sample() {
        let pids = WorkerPIDRegistry.snapshot()
        var bytes: UInt64 = 0
        for pid in pids {
            var info = rusage_info_v4()
            let status = withUnsafeMutablePointer(to: &info) { p in
                p.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V4, $0) }
            }
            if status == 0 { bytes += info.ri_phys_footprint }
        }
        let swap = Self.swapBytes()
        lock.withLock {
            peakFootprint = max(peakFootprint, bytes); peakSwap = max(peakSwap, swap)
        }
        if bytes > limit || swap > initialSwap || ProcessInfo.processInfo.systemUptime >= deadline {
            lock.withLock { reason = "Intake stopped: hard worker memory budget, any swap growth, or 590-second run ceiling exceeded" }
            for pid in pids { _ = kill(pid, SIGKILL) }
        }
    }
    func receipt() -> JSONObject {
        lock.withLock {
            [
                "elapsed_s": .number(ProcessInfo.processInfo.systemUptime - started), "worker_budget_bytes": .number(Double(limit)),
                "worker_peak_bytes": .number(Double(peakFootprint)), "initial_swap_bytes": .number(Double(initialSwap)),
                "peak_swap_bytes": .number(Double(max(initialSwap, peakSwap))), "stopped": .bool(reason != nil)
            ]
        }
    }
    func check() throws { if let message = lock.withLock({ reason }) { throw ManifestError.invalid(message) } }
}
