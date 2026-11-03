// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

extension ModelIntakeCommand {
    static let voiceModels: Set<String> = ["breeze-tts-2", "qwen3-tts-base", "qwen3-tts-voicedesign", "pocket-tts", "kokoro-82m"]

    static func voiceModel(_ original: ModelManifest, half: String) -> ModelManifest {
        var model = original
        model.vendor = nil
        model.derive = [
            half: "cast:" + model.id,
            "int8": "intake-v3:" + model.id + ":all", "int4": "intake-v3:" + model.id + ":all"
        ]
        return model
    }
    static func voiceRecipeMatrices(_ recipe: String, model: String) -> Set<String> {
        let prefix = "intake-v3:" + model + ":"
        guard recipe.hasPrefix(prefix) else { return [] }
        let suffix = String(recipe.dropFirst(prefix.count))
        return suffix == "all" ? [] : Set(suffix.split(separator: ",").map(String.init))
    }
    static func voiceComponent(_ key: String) -> String {
        var parts = key.split(separator: "-").map(String.init)
        while let last = parts.last, Int(last) != nil { parts.removeLast() }
        return parts.joined(separator: "-")
    }
    static func voiceFallbacks(_ gate: JSONObject) throws -> Set<String> {
        guard let components = gate["components"]?.object, !components.isEmpty,
            components.values.allSatisfy({ $0["passed"]?.bool != nil })
        else { throw ManifestError.invalid("Missing component verdicts") }
        return Set(components.filter { $0.value["passed"]?.bool == false }.map { voiceComponent($0.key) })
    }
    static func gateEvidence(gate: JSONObject, keeping: Set<String>) -> JSONObject {
        (gate["components"]?.object ?? [:]).filter { keeping.contains(voiceComponent($0.key)) }
    }
    static func voiceKeys(_ id: String) -> Set<String> {
        var keys: Set<String> = []
        if id == "kokoro-82m" {
            return [
                "kokoro-g2p-0", "kokoro-g2p-1", "kokoro-style", "kokoro-bert", "kokoro-projection", "kokoro-prosody-0", "kokoro-prosody-1", "kokoro-prosody-2", "kokoro-prosody-3",
                "kokoro-text", "kokoro-decoder"
            ]
        }
        if id == "pocket-tts" {
            keys = ["text-conditioner", "preset-conditioning", "mimi-encoder", "speaker-projection"]
            for i in 0..<3 { for name in ["latent-projection", "backbone", "eos", "flow", "mimi-transformer-upsampling", "mimi-waveform"] { keys.insert("\(name)-\(i)") } }
        } else {
            keys = ["text-embedding", "text-projection", "codec-decoder", "codec-encoder", "codec-encoder-transformer", "codec-downsample"]
            if id == "breeze-tts-2" {
                keys.formUnion(["text-encoder", "audio-embedding"])
                for i in 0..<3 {
                    keys.insert("backbone-\(i)"); keys.insert("head-\(i)")
                    for h in [0, 7, 14] { keys.insert("depth-\(i)-\(h)") }
                }
            } else {
                keys.insert("codec-embedding")
                if id == "qwen3-tts-base" { keys.insert("speaker-encoder") }
                for i in 0..<3 {
                    keys.insert("talker-\(i)")
                    for h in 0..<15 { keys.insert("predictor-\(i)-\(h)"); keys.insert("predictor-embedding-\(i)-\(h)") }
                }
            }
        }
        return keys
    }

    static func requireVoiceHalfCapture(_ output: URL, half: String) throws {
        let tensors = try IntakeTensor.read(output.appendingPathComponent("outputs.safetensors"))
        let expected = half == "fp16" ? "F16" : "BF16"
        guard let wrong = tensors.first(where: { $0.value.dtype != expected }) else { return }
        throw ManifestError.invalid("Voice component \(wrong.key) returned \(wrong.value.dtype); expected \(expected). Learned component streams must remain half.")
    }

    // A terminal frame can precede the worker's admission release by a few instructions.
    // A busy rejection executed no request, so retrying it cannot duplicate capture or generation.
    static func voiceWhenReady(_ request: () async throws -> JSONObject) async throws -> JSONObject {
        for attempt in 0..<25 {
            try Task.checkCancellation()
            do { return try await request() } catch let error as WorkerFailure where error.code == "busy" && attempt < 24 {
                try await Task.sleep(for: .milliseconds(10))
            }
        }
        throw ManifestError.invalid("Voice request admission did not settle")
    }

    static let voiceSelectionCases: [JSONValue] = [
        ["text": "Hello. This is a short voice test.", "prefix_lengths": [1, 2, 3], "code_frames": 3, "offset": 0, "preset": "alba"],
        [
            "text": "The parcel arrived before sunrise, packed in bright red paper.", "prefix_lengths": [2, 5, 10], "code_frames": 11, "offset": 1001, "audio_hz": 257,
            "preset": "fantine"
        ],
        [
            "text": "Measure each panel twice, then close the window near the clock.", "prefix_lengths": [3, 8, 13], "code_frames": 14, "offset": 2002, "audio_hz": 443,
            "preset": "jean"
        ]
    ]
    /// Saved component inputs isolate owners. Unchanged owners retain the known passing verdict;
    /// final stored selection and held-out captures always run the complete set.
    static func voiceSelectionTargets(keeping: Set<String>, knownPassing: Set<String>?, owners: JSONObject, all: Set<String>) -> Set<String> {
        guard let knownPassing else { return all }
        let changed = keeping.symmetricDifference(knownPassing)
        let mapped = changed.compactMap { owners[$0]?.string }
        guard mapped.count == changed.count else { return all }
        let affected = Set(mapped)
        let targets = Set(all.filter { affected.contains(voiceComponent($0)) })
        return targets.isEmpty ? all : targets
    }
    static func voiceSelectionCases(for model: String) -> [JSONValue] {
        var cases = voiceSelectionCases
        if model == "kokoro-82m" {
            // Exercise all style classes AND distinct BART source/decoder prefix lengths.
            for (index, offset) in [(1, 1010), (2, 2005)] {
                var row = cases[index].object!
                row["offset"] = .number(Double(offset))
                cases[index] = .object(row)
            }
        }
        return cases
    }
    static func voiceFixedSelection(seed: Set<String>, owners: JSONObject, failures: Set<String>) throws -> Set<String> {
        guard seed.allSatisfy({ owners[$0]?.string != nil }) else { throw ManifestError.invalid("Fixed selection contains an unknown matrix") }
        // Passing parts stay packed, even when the supplied conservative inventory retained them.
        return Set(seed.filter { failures.contains(owners[$0]!.string!) })
    }
    static func voiceCanPublishResult(rows: [JSONValue], failure: String?) -> Bool {
        failure == nil && rows.count == 3 && rows.first(where: { $0["precision"]?.string == "int8" })?["qualified"]?.bool == true
    }
    static func selectVoiceFP16(selectionPassed: Bool, heldoutPassed: Bool) throws -> Bool {
        guard !selectionPassed || heldoutPassed else {
            throw ManifestError.invalid("Selected FP16 failed held-out qualification; continue format selection on selection data and use a new untouched held-out set")
        }
        return selectionPassed
    }

    static func voiceRatios(_ samples: [Double]) -> JSONValue {
        [
            "samples": .array(samples.map(JSONValue.number)), "median": .number(IntakeTimingSummary(samples: samples).median),
            "min": .number(samples.min()!), "max": .number(samples.max()!), "unit": "compute s/audio s"
        ]
    }

    static func voiceArguments(option: (String) -> String?, go: Bool) throws -> (URL, URL, ModelManifest, String, String) {
        guard let folderPath = option("--model"), let sourcePath = option("--source"), let helperPath = option("--helpers"), let labPath = option("--support"),
            go
        else {
            throw ManifestError.invalid("Voice intake needs --model, --source, --helpers, fresh isolated --support and --go")
        }
        let folder = URL(fileURLWithPath: folderPath).standardizedFileURL, lab = URL(fileURLWithPath: labPath).standardizedFileURL
        let original = try ModelManifest.load(folder)
        guard voiceModels.contains(original.id), lab.pathComponents.contains("tmp"), !lab.resolvingSymlinksInPath().path.contains("Library/Application Support"),
            !FileManager.default.fileExists(atPath: lab.path)
        else {
            throw ManifestError.invalid("Voice intake requires a supported model and fresh tmp lab")
        }
        return (folder, lab, original, sourcePath, helperPath)
    }

    static func voicePolicy(run: String, workload: JSONObject, original: ModelManifest) -> JSONObject {
        return [
            "run": .string(run), "fixture": .object(workload), "architecture_component_keys": .array(voiceKeys(original.id).sorted().map(JSONValue.string)),
            "reference_scope": "BF16 cast of all FP32-source learned weights and presets, exponent-preserving constrained reference; no FP32 model execution.",
            "format_gate": .string(IntakeComponentGate.fp32SourceBasis),
            "quantized_tier_gate": .string(IntakeComponentGate.quantizedBasis), "independent_reference_qualified": false,
            "pocket_waveform_input_policy_id": "saved-unit-rms-floor-0p01-seanet-v1",
            "pocket_decoder_boundary":
                "mimi-transformer-upsampling = SEANet input after upsample/decoder transformer. mimi-waveform = separate SEANet output on saved input normalized to unit RMS (floor .01). Shared bounds rel L2 <=.05, cosine >=.999, finite/no new clamps, frozen before candidates. Original waveform failure remains historical; no perceptual claim.",
            "independent_reference_gap":
                "No retained vendor half-GPU component/PCM capture with matched inputs; no download or vendor environment installed. Internal precision qualification only.",
            "machine_in_use": true, "timing_order": "ABBA BAAB, production Load and speak, four synchronized warm requests per arm"
        ]
    }

    // One bounded intake transaction owns worker cleanup and immutable evidence.
    // swiftlint:disable:next function_body_length
    static func executeVoice(_ args: [String]) async throws {
        func option(_ key: String) -> String? { args.firstIndex(of: key).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil } }
        let (folder, lab, original, sourcePath, helperPath) = try voiceArguments(option: option, go: args.contains("--go"))
        let paths = ArsenalPaths(support: lab); try paths.create()
        defer { for name in ["donor", "weights", "cache", "rounding"] { try? FileManager.default.removeItem(at: lab.appendingPathComponent(name)) } }
        try cloneDonor(model: original, source: URL(fileURLWithPath: sourcePath), lab: lab)
        if let assetSource = option("--asset-source") {
            for asset in original.asset ?? [] {
                var donor = original; donor.weights = asset.weights
                try cloneDonor(model: donor, source: URL(fileURLWithPath: assetSource).appendingPathComponent(asset.repo), lab: lab)
            }
        }
        let guardrail = IntakeMemoryGuard(limit: min(60_000_000_000, UInt64(MemoryProbe().availableMB(loadedMB: 0) * 650_000)))
        guardrail.start(); defer { guardrail.stop() }
        let artifact = BundledWorkerArtifact(helpers: URL(fileURLWithPath: helperPath))
        let fetcher = ModelFetcher(support: lab, converter: WorkerPrecisionConverter(artifacts: artifact, paths: paths), localSources: [lab.appendingPathComponent("donor").path])
        let run = "intake-" + ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-") + "-" + UUID().uuidString.lowercased()
        let evidence = folder.appendingPathComponent("gate/evidence/intake/" + run)
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        let result = folder.appendingPathComponent("results/intake/" + run + ".json")
        try FileManager.default.createDirectory(at: result.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fixture = evidence.appendingPathComponent("fixture"), heldoutFixture = evidence.appendingPathComponent("heldout-fixture"),
            formatBounds = evidence.appendingPathComponent("bounds.json"),
            quantBounds = evidence.appendingPathComponent("quantized-bounds.json")
        try FileManager.default.createDirectory(at: evidence.appendingPathComponent("workload"), withIntermediateDirectories: true)
        let workload = try MeasurementWorkload.request(kind: .voices, model: original.id, batched: false, count: 1, directory: evidence.appendingPathComponent("workload")).params
        var policy = voicePolicy(run: run, workload: workload, original: original)
        let attempt = Int(option("--heldout-generation") ?? "0") ?? 0
        guard attempt >= 0 else { throw ManifestError.invalid("Invalid held-out generation") }
        // Plans and thresholds precede all candidate execution. Different generations are new
        // untouched cases; a held-out failure never enters the selection-removal closure.
        let selectionCases = voiceSelectionCases(for: original.id)
        let unseenTexts = [
            "Please set the small brass key beside the folded map before breakfast.",
            "A train crossed three valleys while the lighthouse keeper counted twelve ships.",
            "After lunch, carry six glass bowls upstairs and leave the curtains open.",
            "Winter clouds drift above the orchard; each green gate has a copper handle.",
            "The library closes early on Friday, but the narrow side entrance stays open.",
            "Seven wooden boats waited by the quay as a distant bell marked noon."
        ]
        let heldoutCases: [JSONValue] = [
            [
                "text": .string(
                    (attempt == 0 ? "At platform seven, bring the blue umbrella and ask for ticket" : unseenTexts[((attempt - 1) * 2) % unseenTexts.count]) + " "
                        + String(37 + attempt) + "."), "prefix_lengths": .array([4 + attempt % 3, 7 + attempt % 4, 11 + attempt % 3].map { .number(Double($0)) }),
                "code_frames": .number(Double(12 + attempt % 3)),
                "offset": .number(Double(31 + attempt * 19)), "preset": "marius"
            ],
            [
                "text": .string(
                    (attempt == 0 ? "Quiet rivers curve past nine old bridges; tomorrow the market opens at" : unseenTexts[((attempt - 1) * 2 + 1) % unseenTexts.count]) + " "
                        + String(6 + attempt) + "."), "prefix_lengths": .array([3 + attempt % 3, 6 + attempt % 4, 9 + attempt % 3].map { .number(Double($0)) }),
                "code_frames": .number(Double(10 + attempt % 3)), "offset": .number(Double(67 + attempt * 23)), "preset": "cosette"
            ]
        ]
        for (directory, cases) in [(fixture, selectionCases), (heldoutFixture, heldoutCases)] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try WireJSON.encode(["schema": 1, "cases": .array(cases)]).write(to: directory.appendingPathComponent("fixture-plan.json"), options: .withoutOverwriting)
        }
        func keys(_ count: Int) -> Set<String> { Set((0..<count).flatMap { i in voiceKeys(original.id).map { $0 + "-" + String(i) } }) }
        policy["selection_component_keys"] = .array(keys(selectionCases.count).sorted().map(JSONValue.string))
        policy["heldout_component_keys"] = .array(keys(heldoutCases.count).sorted().map(JSONValue.string))
        policy["selection_case_count"] = .number(Double(selectionCases.count))
        policy["heldout_case_count"] = .number(Double(heldoutCases.count))
        var selectionSeed: Set<String> = []
        if let seedPath = option("--selection-seed-coverage") {
            let seedURL = URL(fileURLWithPath: seedPath).standardizedFileURL.resolvingSymlinksInPath()
            guard seedURL.path.hasPrefix(folder.appendingPathComponent("gate/evidence/intake").resolvingSymlinksInPath().path + "/") else {
                throw ManifestError.invalid("Selection seed must be this model's retained selection coverage")
            }
            let seed = try WireJSON.decode(Data(contentsOf: seedURL))
            guard seed["rule"]?["id"]?.string?.hasPrefix("intake-v3:" + original.id + ":") == true,
                let matrices = seed["fallback_matrices"]?.array
            else { throw ManifestError.invalid("Missing selection-only seed inventory") }
            selectionSeed = Set(matrices.compactMap(\.string))
            policy["selection_seed_sha256"] = .string(try FileDigest.sha256(seedURL))
            policy["selection_seed_source"] = .string(seedURL.path.replacingOccurrences(of: folder.path + "/", with: ""))
        }
        let fixedSelection = args.contains("--fixed-int8-selection")
        guard !fixedSelection || option("--selection-seed-coverage") != nil else {
            throw ManifestError.invalid("Fixed int8 selection requires a model-local selection inventory")
        }
        policy["int8_selection_strategy"] = .string(fixedSelection ? "fixed prior selection candidate; no minimality claim" : "adaptive projection-group removal")
        try WireJSON.encode(policy).write(to: evidence.appendingPathComponent("policy.json"), options: .withoutOverwriting)
        let heldoutFormatBounds = evidence.appendingPathComponent("heldout-format-bounds.json")
        let heldoutQuantBounds = evidence.appendingPathComponent("heldout-quantized-bounds.json")
        var snapshotAssets: [String: URL] = [:]
        var selected = voiceModel(original, half: "bf16"), rows: [JSONValue] = [], workers: [WorkerProcess] = []
        func snapshot(_ dtype: String, model: ModelManifest, label: String? = nil, using customFetcher: ModelFetcher? = nil) async throws -> URL {
            let fetcher = customFetcher ?? fetcher
            try guardrail.check()
            let fetched = try await fetcher.get(model: model, modelDirectory: folder, precision: dtype, approvedDownloads: [:])
            snapshotAssets = fetched.assets
            _ = try await fetcher.verifyInstalled(model: model, precision: dtype, forceHash: true)
            try Data(contentsOf: StoredNames.snapshotURL(in: fetched.weightsDirectory)).write(
                to: evidence.appendingPathComponent((label ?? dtype) + "-snapshot.json"), options: .withoutOverwriting)
            let conversion = fetched.weightsDirectory.appendingPathComponent("conversion.json")
            if FileManager.default.fileExists(atPath: conversion.path) {
                try Data(contentsOf: conversion).write(to: evidence.appendingPathComponent((label ?? dtype) + "-coverage.json"))
            }
            return fetched.weightsDirectory
        }
        func start(_ dtype: String, weights: URL, model: ModelManifest) async throws -> (WorkerProcess, JSONObject) {
            let p = try await startWorker(
                dtype, weights: weights, original: model, artifact: artifact, folder: folder, paths: paths, memoryGuard: guardrail, isImage: false, assets: snapshotAssets)
            workers.append(p.0); return p
        }
        let referenceReplay = option("--reuse-reference").map { URL(fileURLWithPath: $0).standardizedFileURL }
        if let referenceReplay {
            guard referenceReplay.path.hasPrefix(folder.appendingPathComponent("gate/evidence/intake/").path),
                let runner = selected.runners.first
            else { throw ManifestError.invalid("Reference replay must belong to this model's retained intake") }
            let receipt = try WireJSON.decode(Data(contentsOf: referenceReplay.appendingPathComponent("incomplete-result.json")))
            guard receipt["model"]?.string == original.id, receipt["revision"]?.string == original.weights.revision,
                receipt["worker_sha256"]?.string == (try FileDigest.sha256(artifact.helpers.appendingPathComponent("arsenal-worker-native"))),
                receipt["implementation_identity"]?.string == artifact.qualityImplementation(model: selected, runner: runner, directory: folder),
                !FileManager.default.fileExists(atPath: referenceReplay.appendingPathComponent("int8-heldout").path)
            else { throw ManifestError.invalid("Reference replay requires exact worker/source revision and untouched int8 held-out cases") }
            for (name, target) in [("fixture", fixture), ("heldout-fixture", heldoutFixture)] {
                let source = referenceReplay.appendingPathComponent(name)
                guard try FileDigest.sha256(source.appendingPathComponent("fixture-plan.json")) == FileDigest.sha256(target.appendingPathComponent("fixture-plan.json")) else {
                    throw ManifestError.invalid("Reference replay inputs differ from newly frozen plan")
                }
                try FileManager.default.copyItem(at: source.appendingPathComponent("inputs.safetensors"), to: target.appendingPathComponent("inputs.safetensors"))
            }
            policy["reference_replay"] = .string(referenceReplay.lastPathComponent)
        }
        func capture(_ process: WorkerProcess, name: String, create: Bool = false, diagnostic: URL? = nil, components: Set<String>? = nil, heldout: Bool = false) async throws
            -> URL
        {
            let output = evidence.appendingPathComponent(name)
            if let referenceReplay, ["reference", "reference-repeat", "heldout-reference", "heldout-reference-repeat", "bf16-truncation", "heldout-bf16-truncation"].contains(name)
            {
                try FileManager.default.copyItem(at: referenceReplay.appendingPathComponent(name), to: output)
                try mergeVoiceAuxiliaryCoverage(capture: output, evidence: evidence, label: name == "reference" ? "bf16" : name)
                return output
            }
            var request: JSONObject = ["fixture_dir": .string((heldout ? heldoutFixture : fixture).path), "output_dir": .string(output.path), "capture": .bool(create)]
            if let diagnostic, let components {
                request["diagnostic_reference"] = .string(diagnostic.path)
                request["diagnostic_components"] = .array(components.sorted().map(JSONValue.string))
            }
            let reply = try await voiceWhenReady { try await process.request(op: "measurement_components", params: request, timeout: 120) }
            let usedFixture = heldout ? heldoutFixture : fixture
            try WireJSON.encode([
                "fixture_inputs_sha256": .string(try FileDigest.sha256(usedFixture.appendingPathComponent("inputs.safetensors"))),
                "component_count": reply["components"] ?? .null
            ]).write(to: output.appendingPathComponent("fixture-identity.json"), options: .withoutOverwriting)
            try mergeVoiceAuxiliaryCoverage(capture: output, evidence: evidence, label: name == "reference" ? "bf16" : name)
            try guardrail.check()
            try requireVoiceHalfCapture(output, half: name.hasPrefix("fp16") ? "fp16" : selected.basePrecision)
            return output
        }
        func generate(_ process: WorkerProcess, _ label: String) async throws -> (Double, JSONObject) {
            var request = workload
            request["out_dir"] = .string(evidence.appendingPathComponent("sanity/" + label).path)
            let before = ProcessInfo.processInfo.systemUptime
            let reply = try await voiceWhenReady { try await process.request(op: "speak", params: request, timeout: 100) }
            try guardrail.check()
            return (ProcessInfo.processInfo.systemUptime - before, reply)
        }
        func persist(_ failure: String? = nil) throws {
            guard let runner = selected.runners.first, let implementation = artifact.qualityImplementation(model: selected, runner: runner, directory: folder) else {
                throw ManifestError.invalid("Missing voice build identity")
            }
            var document: JSONObject = [
                "model": .string(original.id), "revision": .string(original.weights.revision), "run": .string(run),
                "date": .string(ISO8601DateFormatter().string(from: Date())), "chip": .string(HostChip.current), "gpu_cores": .number(Double(HostChip.gpuCores)),
                "worker_sha256": .string(try FileDigest.sha256(artifact.helpers.appendingPathComponent("arsenal-worker-native"))),
                "implementation_identity": .string(implementation),
                "load_identity": .string(try ModelLoadConfiguration.identity(model: selected, runner: runner, folder: folder)), "selected_half": .string(selected.basePrecision),
                "fp16_selected": .bool(selected.basePrecision == "fp16"),
                "note": "BF16-source constrained reference; same saved component inputs; end-to-end audio is diagnostic. Busy-machine quick checks on the default served workload.",
                "rows": .array(rows), "recipes": .object((selected.derive ?? [:]).mapValues(JSONValue.string)), "policy": .object(policy), "network_bytes": 0,
                "resource_receipt": .object(guardrail.receipt())
            ]
            document["worker_build"] = .string(option("--worker-build") ?? "unspecified")
            if !voiceCanPublishResult(rows: rows, failure: failure) {
                if let failure { document["error"] = .string(failure) }
                try WireJSON.encode(document).write(to: evidence.appendingPathComponent("incomplete-result.json"), options: .atomic)
                return
            }
            try WireJSON.encode(document).write(to: result, options: .atomic)
            try WireJSON.encode(["run": .string(run), "results": .string("results/intake/" + result.lastPathComponent), "evidence": .string("gate/evidence/intake/" + run)]).write(
                to: evidence.appendingPathComponent("receipt.json"), options: .atomic)
        }
        do {
            let baseWeights = try await snapshot("bf16", model: selected)
            var (baseline, baseLoad) = try await start("bf16", weights: baseWeights, model: selected)
            let reference = try await capture(baseline, name: "reference", create: true)
            let repeated = try await capture(baseline, name: "reference-repeat")
            let heldoutReference = try await capture(baseline, name: "heldout-reference", create: true, heldout: true)
            let heldoutRepeated = try await capture(baseline, name: "heldout-reference-repeat", heldout: true)
            let roundingPaths = ArsenalPaths(support: lab.appendingPathComponent("rounding")); try roundingPaths.create()
            let roundingFetcher = ModelFetcher(
                support: roundingPaths.support,
                converter: WorkerPrecisionConverter(artifacts: artifact, paths: roundingPaths), localSources: [lab.appendingPathComponent("donor").path])
            var roundingModel = selected
            roundingModel.derive?["bf16"] = "cast:" + original.id + ":bf16-truncation"
            let roundingWeights = try await snapshot("bf16", model: roundingModel, label: "bf16-truncation", using: roundingFetcher)
            let (roundingWorker, _) = try await start("bf16", weights: roundingWeights, model: roundingModel)
            let rounding = try await capture(roundingWorker, name: "bf16-truncation")
            let heldoutRounding = try await capture(roundingWorker, name: "heldout-bf16-truncation", heldout: true)
            await roundingWorker.unload()
            try IntakeComponentGate.freezeFP32Source(
                reference: heldoutReference, roundingCapture: heldoutRounding, repeatCapture: heldoutRepeated,
                to: heldoutFormatBounds, expectedKeys: keys(heldoutCases.count))
            try IntakeComponentGate.freezeFP32Source(
                reference: reference, roundingCapture: rounding, repeatCapture: repeated,
                to: formatBounds, expectedKeys: keys(selectionCases.count))
            let fixtureHash = try FileDigest.sha256(fixture.appendingPathComponent("inputs.safetensors"))
            let fpModel = voiceModel(original, half: "fp16")
            let fpWeights = try await snapshot("fp16", model: fpModel)
            let (fpWorker, fpLoad) = try await start("fp16", weights: fpWeights, model: fpModel)
            let fpCapture = try await capture(fpWorker, name: "fp16")
            let fpGate = try IntakeComponentGate.evaluate(reference: reference, candidate: fpCapture, bounds: formatBounds)
            let fpHeldoutCapture = try await capture(fpWorker, name: "fp16-heldout", heldout: true)
            let fpHeldoutGate = try IntakeComponentGate.evaluate(reference: heldoutReference, candidate: fpHeldoutCapture, bounds: heldoutFormatBounds)
            let formatQualified = fpGate["component_gate_passed"]?.bool == true && fpHeldoutGate["component_gate_passed"]?.bool == true
            try WireJSON.encode(["precision": "fp16", "gate": .object(fpGate), "heldout_gate": .object(fpHeldoutGate), "qualified": .bool(formatQualified)]).write(
                to: evidence.appendingPathComponent("format-trial.json"), options: .withoutOverwriting)
            let passed = try selectVoiceFP16(
                selectionPassed: fpGate["component_gate_passed"]?.bool == true,
                heldoutPassed: fpHeldoutGate["component_gate_passed"]?.bool == true)
            var selectedCapture = reference, selectedHeldout = heldoutReference
            if passed {
                selected = fpModel
                await baseline.unload(); baseline = fpWorker; baseLoad = fpLoad; selectedCapture = fpCapture; selectedHeldout = fpHeldoutCapture
            } else {
                await fpWorker.unload()
            }
            policy["selected_half"] = .string(selected.basePrecision)
            // Half candidate cannot qualify quantization by comparing to BF16 if FP16 was selected.
            let selectedRepeat = passed ? try await capture(baseline, name: "selected-repeat") : repeated
            try IntakeComponentGate.freeze(
                reference: selectedCapture, repeatCapture: selectedRepeat, to: quantBounds, half: selected.basePrecision, quantized: true, expectedKeys: keys(selectionCases.count))
            let selectedHeldoutRepeat = passed ? try await capture(baseline, name: "selected-heldout-repeat", heldout: true) : heldoutRepeated
            try IntakeComponentGate.freeze(
                reference: selectedHeldout, repeatCapture: selectedHeldoutRepeat, to: heldoutQuantBounds,
                half: selected.basePrecision, quantized: true, expectedKeys: keys(heldoutCases.count))
            let safetyL2 = Double(option("--selection-relative-l2") ?? "0.05") ?? 0.05
            let safetyCosine = Double(option("--selection-cosine") ?? "0.999") ?? 0.999
            let searchBounds = evidence.appendingPathComponent("selection-search-bounds.json")
            try IntakeComponentGate.freezeSelectionSafety(from: quantBounds, to: searchBounds, relativeL2: safetyL2, cosine: safetyCosine)
            let frozenQualification = try IntakeQualification.freeze(
                selection: fixture, heldout: heldoutFixture, selectionBounds: quantBounds,
                heldoutBounds: heldoutQuantBounds, to: evidence.appendingPathComponent("qualification-plan.json"))
            let baseHeldoutGate = try IntakeComponentGate.evaluate(reference: selectedHeldout, candidate: selectedHeldoutRepeat, bounds: heldoutQuantBounds)
            let baseQualification = try IntakeQualification.complete(
                plan: frozenQualification,
                selection: try IntakeComponentGate.evaluate(reference: selectedCapture, candidate: selectedRepeat, bounds: quantBounds),
                heldout: baseHeldoutGate, searched: false, ordinaryLoad: baseLoad)
            await baseline.unload()
            // Fresh workers exclude capture-only tensors from the default Load/operation lifetime peaks.
            let freshBase = try await start(selected.basePrecision, weights: passed ? fpWeights : baseWeights, model: selected)
            baseline = freshBase.0; baseLoad = freshBase.1
            let (cold, sanity) = try await generate(baseline, selected.basePrecision + "-cold")
            var allBaseline: [Double] = [], baseRTF: [Double] = []
            var baseRow: JSONObject = [
                "precision": .string(selected.basePrecision), "gate": .object(baseHeldoutGate), "selection_gate": baseQualification["selection_gate"]!,
                "qualification": .object(baseQualification),
                "qualified": true,
                "end_to_end": .object(try voiceSanity(reference: sanity, candidate: sanity)), "cold_seconds": .number(cold), "load": .object(baseLoad),
                "sanity_reply": .object(sanity), "fixture_sha256": .string(fixtureHash), "state": "quickCheck", "capture_load": .object(passed ? fpLoad : baseLoad),
                "serving_worker": .object(await servingWorkerReceipt(baseline))
            ]
            for tier in ["int8", "int4"] {
                var weights = try await snapshot(tier, model: selected, label: tier == "int8" ? "int8-all" : nil)
                let (probe, captureLoad) = try await start(tier, weights: weights, model: selected)
                let candidateCapture = try await capture(probe, name: tier == "int8" ? "int8-all" : tier)
                var gate = try IntakeComponentGate.evaluate(reference: selectedCapture, candidate: candidateCapture, bounds: quantBounds)
                var searched = false
                if tier == "int8" {
                    let searchGate = try IntakeComponentGate.evaluate(reference: selectedCapture, candidate: candidateCapture, bounds: searchBounds)
                    try WireJSON.encode(searchGate).write(to: evidence.appendingPathComponent("int8-all-gate.json"), options: .withoutOverwriting)
                    let failures = try voiceFallbacks(searchGate)
                    var matrices: Set<String> = []
                    if !failures.isEmpty || fixedSelection {
                        searched = true
                        let coverageFile = evidence.appendingPathComponent("int8-all-coverage.json")
                        let inventory = try WireJSON.decode(Data(contentsOf: coverageFile))
                        if fixedSelection {
                            matrices = try voiceFixedSelection(seed: selectionSeed, owners: inventory["matrix_components"]?.object ?? [:], failures: failures)
                            try WireJSON.encode([
                                "basis": "Conservative saved selection candidate; no removal/minimum claim. Full stored selection and untouched held-out gates must both pass.",
                                "source_sha256": policy["selection_seed_sha256"]!, "source": policy["selection_seed_source"]!,
                                "matrices": .array(matrices.sorted().map(JSONValue.string)), "minimality_claimed": false
                            ]).write(to: evidence.appendingPathComponent("selection-choice.json"), options: .withoutOverwriting)
                        } else {
                            let owners = inventory["matrix_components"]?.object ?? [:]
                            let patchBase: JSONObject = [
                                "half_dir": .string((passed ? fpWeights : baseWeights).path), "coverage": .string(coverageFile.path),
                                "components": .array(failures.sorted().map(JSONValue.string))
                            ]
                            func patch(_ keeping: Set<String>) async throws -> [[String]] {
                                var request = patchBase; request["keeping"] = .array(keeping.sorted().map(JSONValue.string))
                                let reply = try await voiceWhenReady { try await probe.request(op: "measurement_voice_patch", params: request, timeout: 120) }
                                try guardrail.check()
                                guard let groups = reply["matrix_groups"]?.array else { throw ManifestError.invalid("Missing matrix diagnosis groups") }
                                return groups.map { $0.array?.compactMap(\.string) ?? [] }
                            }
                            let groups = try await patch([]).map(Set.init).filter { group in
                                group.contains { failures.contains(owners[$0]?.string ?? "") }
                            }
                            matrices = Set(groups.flatMap { $0 })
                            let targeted = Set(keys(selectionCases.count).filter { failures.contains(voiceComponent($0)) })
                            var trials: [JSONValue] = [], index = 0
                            // Inputs/targets and the all-packed starting state are immutable. Reusing
                            // the identical keeper set avoids duplicate GPU trials in the final sweep.
                            var trialCache: [Set<String>: (passed: Bool, failures: Set<String>)] = [:]
                            var knownPassing: Set<String>?
                            var lastSelectionFailures: Set<String> = []
                            func trial(_ keeping: Set<String>, reason: String) async throws -> Bool {
                                if let cached = trialCache[keeping] {
                                    lastSelectionFailures = cached.failures
                                    if cached.passed { knownPassing = keeping }
                                    return cached.passed
                                }
                                _ = try await patch(keeping)
                                let name = "diagnosis/trial-" + String(index); index += 1
                                let evaluated = voiceSelectionTargets(keeping: keeping, knownPassing: knownPassing, owners: owners, all: targeted)
                                let beforeCapture = ProcessInfo.processInfo.systemUptime
                                let captured = try await capture(probe, name: name, diagnostic: selectedCapture, components: evaluated)
                                let afterCapture = ProcessInfo.processInfo.systemUptime
                                let measured = try IntakeComponentGate.evaluate(reference: selectedCapture, candidate: captured, bounds: searchBounds, diagnosticKeys: evaluated)
                                let afterGate = ProcessInfo.processInfo.systemUptime
                                let passed = measured["component_gate_passed"]?.bool == true
                                lastSelectionFailures = try voiceFallbacks(measured)
                                trialCache[keeping] = (passed, lastSelectionFailures)
                                if passed { knownPassing = keeping }
                                try WireJSON.encode(measured).write(to: evidence.appendingPathComponent(name + "/gate.json"), options: .withoutOverwriting)
                                trials.append([
                                    "trial": .string(name), "reason": .string(reason), "keeping": .array(keeping.sorted().map(JSONValue.string)),
                                    "evaluated_component_keys": .array(evaluated.sorted().map(JSONValue.string)), "capture_seconds": .number(afterCapture - beforeCapture),
                                    "gate_seconds": .number(afterGate - afterCapture), "passed": .bool(passed)
                                ])
                                try WireJSON.encode(["diagnostic_only": true, "components": .array(targeted.sorted().map(JSONValue.string)), "trials": .array(trials)])
                                    .write(to: evidence.appendingPathComponent("matrix-diagnosis.json"), options: .atomic)
                                return passed
                            }
                            let priorSelectors =
                                selectionSeed.isEmpty ? voiceRecipeMatrices(original.derive?["int8"] ?? "", model: original.id) : selectionSeed
                            let prior = Set(groups.filter { !$0.isDisjoint(with: priorSelectors) }.flatMap { $0 })
                            var seeded = false
                            if !prior.isEmpty {
                                var proposed = prior
                                seeded = try await trial(proposed, reason: "prior recipe is only a selection seed, never qualification")
                                while !seeded {
                                    let repair = Set(groups.filter { group in group.contains { lastSelectionFailures.contains(owners[$0]?.string ?? "") } }.flatMap { $0 })
                                    let expanded = proposed.union(repair)
                                    guard expanded != proposed else { break }
                                    proposed = expanded
                                    seeded = try await trial(proposed, reason: "repair only owners still failing on selection cases; held-out remains untouched")
                                }
                                if seeded { matrices = proposed }
                            }
                            if !seeded {
                                guard try await trial(matrices, reason: "all matrices of failing owners retained, other owners packed") else {
                                    throw ManifestError.invalid("Even full eligible matrix retention did not repair component; inspect diagnostic evidence")
                                }
                            }
                            func remove(_ candidates: [Set<String>]) async throws {
                                guard !candidates.isEmpty else { return }
                                let subset = Set(candidates.flatMap { $0 }).intersection(matrices)
                                guard !subset.isEmpty else { return }
                                if try await trial(matrices.subtracting(subset), reason: "re-quantize group of " + String(candidates.count) + " matrices") {
                                    matrices.subtract(subset)
                                } else if candidates.count > 1 {
                                    let middle = candidates.count / 2
                                    try await remove(Array(candidates[..<middle])); try await remove(Array(candidates[middle...]))
                                }
                            }
                            // Group removal also applies after repairing a prior seed; expanding an owner
                            // must not turn a short intake into one trial per matrix before any bulk removal.
                            try await remove(groups.filter { !$0.isDisjoint(with: matrices) }.sorted { $0.sorted().joined() < $1.sorted().joined() })
                            // Revisit singles after group removal: quantization interactions are not monotonic.
                            var changed = true
                            while changed {
                                changed = false
                                for group in groups where !group.isDisjoint(with: matrices) {
                                    if try await trial(matrices.subtracting(group), reason: "final atomic projection-group necessity with all other selected groups fixed") {
                                        matrices.subtract(group); changed = true
                                    }
                                }
                            }
                        }
                        await probe.unload()
                        selected.derive?["int8"] = "intake-v3:" + original.id + ":" + (matrices.isEmpty ? "all" : matrices.sorted().joined(separator: ","))
                        weights = try await snapshot(tier, model: selected)
                        let (mixed, _) = try await start(tier, weights: weights, model: selected)
                        let mixedCapture = try await capture(mixed, name: tier)
                        gate = try IntakeComponentGate.evaluate(reference: selectedCapture, candidate: mixedCapture, bounds: quantBounds)
                        await mixed.unload()
                    } else {
                        await probe.unload()
                        for suffix in ["-snapshot.json", "-coverage.json"] {
                            try FileManager.default.copyItem(at: evidence.appendingPathComponent("int8-all" + suffix), to: evidence.appendingPathComponent("int8" + suffix))
                        }
                    }
                    let coverageFile = evidence.appendingPathComponent("int8-coverage.json")
                    var coverage = try WireJSON.decode(Data(contentsOf: coverageFile))
                    coverage["fallback_evidence"] = .object(
                        gateEvidence(gate: try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("int8-all-gate.json"))), keeping: failures))
                    try WireJSON.encode(coverage).write(to: coverageFile, options: .atomic)
                }
                guard try FileDigest.sha256(fixture.appendingPathComponent("inputs.safetensors")) == fixtureHash else { throw ManifestError.invalid("Voice fixture changed") }
                await probe.unload()
                if tier == "int8", gate["component_gate_passed"]?.bool != true {
                    throw ManifestError.invalid("Stored int8 recipe failed selection replay; held-out remains untouched and tier remains pending")
                }
                // Only the final stored recipe sees held-out outputs. No removal closure can
                // read these results; a failure remains a failed qualification for this plan.
                let (qualificationWorker, qualificationLoad) = try await start(tier, weights: weights, model: selected)
                let heldoutCapture = try await capture(qualificationWorker, name: tier + "-heldout", heldout: true)
                let heldoutGate = try IntakeComponentGate.evaluate(reference: selectedHeldout, candidate: heldoutCapture, bounds: heldoutQuantBounds)
                for (name, directory) in [("selection", fixture), ("heldout", heldoutFixture)] {
                    guard frozenQualification[name]?["inputs_sha256"]?.string == (try FileDigest.sha256(directory.appendingPathComponent("inputs.safetensors"))),
                        frozenQualification[name]?["plan_sha256"]?.string == (try FileDigest.sha256(directory.appendingPathComponent("fixture-plan.json")))
                    else {
                        throw ManifestError.invalid("Frozen qualification inputs changed")
                    }
                }
                let qualification = try IntakeQualification.complete(
                    plan: frozenQualification, selection: gate, heldout: heldoutGate,
                    searched: searched, ordinaryLoad: qualificationLoad)
                await qualificationWorker.unload()
                let (candidate, loaded) = try await start(tier, weights: weights, model: selected)
                let (candidateCold, candidateSanity) = try await generate(candidate, tier + "-cold")
                _ = try await generate(baseline, tier + "-A-warm"); _ = try await generate(candidate, tier + "-B-warm")
                var a: [Double] = [], b: [Double] = [], rtf: [Double] = [], aRTF: [Double] = []
                for (i, arm) in [0, 1, 1, 0, 1, 0, 0, 1].enumerated() {
                    let (seconds, reply) = try await generate(arm == 0 ? baseline : candidate, "\(tier)-\(i)-\(arm)")
                    guard let ratio = reply["rtf"]?.number else { throw ManifestError.invalid("Missing served voice RTF") }
                    if arm == 0 { a.append(seconds); aRTF.append(ratio) } else { b.append(seconds); rtf.append(ratio) }
                }
                allBaseline += a; baseRTF += aRTF
                let row: JSONObject = [
                    "precision": .string(tier), "gate": .object(heldoutGate), "selection_gate": .object(gate), "qualification": .object(qualification),
                    "qualified": qualification["passed"]!,
                    "capture_load": .object(captureLoad), "serving_worker": .object(await servingWorkerReceipt(candidate)),
                    "peak_gb": .number(try await servingPeak(candidate)), "cold_seconds": .number(candidateCold), "load": .object(loaded),
                    "sanity_reply": .object(candidateSanity), "end_to_end": .object(try voiceSanity(reference: sanity, candidate: candidateSanity)),
                    "warm": IntakeTimingSummary(samples: b).json, "warm_rtf": voiceRatios(rtf),
                    "paired_baseline_warm": IntakeTimingSummary(samples: a).json, "paired_baseline_rtf": voiceRatios(aRTF),
                    "baseline_drift_fraction": .number(IntakeTimingSummary(samples: Array(a.suffix(2))).median / IntakeTimingSummary(samples: Array(a.prefix(2))).median - 1),
                    "fixture_sha256": .string(fixtureHash), "state": "quickCheck"
                ]
                rows.append(.object(row.merging(precisionOffering(dtype: tier, passed: qualification["passed"]?.bool)) { _, b in b }))
                try persist(); print("\(original.id) \(tier): selection \(gate["component_gate_passed"]!), heldout \(heldoutGate["component_gate_passed"]!)")
                await candidate.unload()
            }
            baseRow["peak_gb"] = .number(try await servingPeak(baseline)); baseRow["warm"] = IntakeTimingSummary(samples: allBaseline).json
            baseRow["warm_rtf"] = voiceRatios(baseRTF)
            rows.insert(.object(baseRow), at: 0)
            await baseline.unload()
            // A failed held-out int8 attempt is pending, never a newly unavailable table tier.
            // Retain its evidence, but do not publish a recipe/binding. Continue selection only
            // in a new transaction, with a new untouched held-out generation.
            guard rows.first(where: { $0["precision"]?.string == "int8" })?["qualified"]?.bool == true else {
                try persist(); throw ManifestError.invalid("Int8 held-out qualification failed; tier remains pending/offered. New recipe needs a new held-out generation.")
            }
            // Only a complete run changes catalog recipes. Historical vendor pins remain in git history.
            try enableVoiceRecipes(folder: folder, model: selected)
            try persist()
            try writeVoiceStats(rows: rows, model: selected, folder: folder, artifact: artifact, run: run, result: result)
            await artifact.shutdown()
            print("\(original.id): \(run), selected \(selected.basePrecision); independent vendor comparison unavailable")
        } catch {
            for worker in workers { await worker.unload() }
            await artifact.shutdown(); try? persist(error.localizedDescription); throw error
        }
    }

    static func enableVoiceRecipes(folder: URL, model: ModelManifest) throws {
        let file = folder.appendingPathComponent("model.toml")
        var text = try String(contentsOf: file, encoding: .utf8)
        let pattern = #"(?ms)^\[derive\]\n.*?(?=^\[|\z)"#
        let derive = "[derive]\n" + (model.derive ?? [:]).sorted { $0.key < $1.key }.map { "\($0.key) = \"\($0.value)\"" }.joined(separator: "\n") + "\n\n"
        text = text.replacingOccurrences(of: pattern, with: derive, options: .regularExpression)
        if model.id == "breeze-tts-2" { text = text.replacingOccurrences(of: #"(?ms)^\[vendor\]\n.*?(?=^\[weights\])"#, with: "", options: .regularExpression) }
        try text.write(to: file, atomically: true, encoding: .utf8)
        let reread = try ModelManifest.load(folder)
        guard reread.derive == model.derive, reread.basePrecision == model.basePrecision, reread.vendor == nil else { throw ManifestError.invalid("Voice recipe write mismatch") }
    }
}
