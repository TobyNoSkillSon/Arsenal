// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import CoreGraphics
import ImageIO

extension ModelIntakeCommand {
    static let biRefNetComponentKeys = Set(
        ["patch-0", "patch-1", "squeeze", "decoder"]
            + (0..<2).flatMap { c in (0..<4).flatMap { ["stage-\(c)-\($0)/out", "stage-\(c)-\($0)/down"] } })
    static let biRefNetMaskMAEBound = 0.02

    // Fixed all-matrix recipe: no fallback search and no held-out qualification claim.
    // swiftlint:disable:next function_body_length
    static func executeBiRefNet(_ args: [String]) async throws {
        func option(_ key: String) -> String? { args.firstIndex(of: key).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil } }
        guard let folderPath = option("--model"), let sourcePath = option("--source"), let helperPath = option("--helpers"),
            let supportPath = option("--support"), args.contains("--go")
        else { throw ManifestError.invalid("BiRefNet intake needs --model, --source, --helpers, fresh isolated --support and --go") }
        let folder = URL(fileURLWithPath: folderPath), lab = URL(fileURLWithPath: supportPath).standardizedFileURL
        guard lab.pathComponents.contains("tmp"), !lab.resolvingSymlinksInPath().path.contains("Library/Application Support"),
            !FileManager.default.fileExists(atPath: lab.path)
        else { throw ManifestError.invalid("Use a fresh lab tmp support directory") }
        let model = try ModelManifest.load(folder)
        guard model.id == "birefnet", model.derive == ["int8": "intake-v1:birefnet", "int4": "intake-v1:birefnet"] else {
            throw ManifestError.invalid("Expected the fixed BiRefNet all-matrix recipe")
        }
        let paths = ArsenalPaths(support: lab); try paths.create()
        defer { for name in ["donor", "weights", "cache"] { try? FileManager.default.removeItem(at: lab.appendingPathComponent(name)) } }
        let guarder = IntakeMemoryGuard(limit: min(60_000_000_000, UInt64(MemoryProbe().availableMB(loadedMB: 0) * 650_000)))
        guarder.start(); defer { guarder.stop() }
        try cloneDonor(model: model, source: URL(fileURLWithPath: sourcePath), lab: lab)
        let artifact = BundledWorkerArtifact(helpers: URL(fileURLWithPath: helperPath))
        let fetcher = ModelFetcher(support: lab, converter: WorkerPrecisionConverter(artifacts: artifact, paths: paths), localSources: [lab.appendingPathComponent("donor").path])
        let run = "intake-" + ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-") + "-" + UUID().uuidString.lowercased()
        let evidence = folder.appendingPathComponent("gate/evidence/intake/" + run)
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        let result = folder.appendingPathComponent("results/intake/" + run + ".json")
        try FileManager.default.createDirectory(at: result.deletingLastPathComponent(), withIntermediateDirectories: true)
        let images = try biRefNetImages(in: paths.directory("inbox"))
        let imagePins = try images.map { image -> JSONValue in ["name": .string(image.lastPathComponent), "sha256": .string(try FileDigest.sha256(image))] }
        let policy: JSONObject = [
            "run": .string(run), "component_keys": .array(biRefNetComponentKeys.sorted().map(JSONValue.string)),
            "quantized_tier_gate": .string(IntakeComponentGate.quantizedBasis), "searched": false,
            "mask_diagnostic": [
                "metric": "mean absolute difference of source-resolution uint8 masks / 255", "bound": .number(biRefNetMaskMAEBound),
                "each_image_must_pass": true, "affects_precision_gate": false, "fixtures": .array(imagePins)
            ],
            "reference_scope": "Original pinned FP16 Swift MLX worker, previously vendor-qualified; no new independent vendor run.",
            "timing_order": "ABBA BAAB; first fixed image, default 1024x1024 preprocessing; four warm replies per arm",
            "machine_in_use": true,
            "fixture_scope": "Three deterministic synthetic images: solid object, thin structure and partial transparency; not a labelled natural-image quality set."
        ]
        try WireJSON.encode(policy).write(to: evidence.appendingPathComponent("policy.json"), options: .withoutOverwriting)
        for image in images { try FileManager.default.copyItem(at: image, to: evidence.appendingPathComponent(image.lastPathComponent)) }
        let fixture = evidence.appendingPathComponent("fixture"), reference = evidence.appendingPathComponent("reference")
        var rows: [JSONValue] = [], workers: [WorkerProcess] = []
        func snapshot(_ tier: String) async throws -> URL {
            try guarder.check()
            let fetched = try await fetcher.get(model: model, modelDirectory: folder, precision: tier, approvedDownloads: [:])
            _ = try await fetcher.verifyInstalled(model: model, precision: tier, forceHash: true)
            let weights = fetched.weightsDirectory
            try Data(contentsOf: StoredNames.snapshotURL(in: weights)).write(to: evidence.appendingPathComponent(tier + "-snapshot.json"))
            if tier != "fp16" {
                try Data(contentsOf: weights.appendingPathComponent("component-coverage.json")).write(to: evidence.appendingPathComponent(tier + "-component-coverage.json"))
            }
            return weights
        }
        func start(_ tier: String, _ weights: URL) async throws -> (WorkerProcess, JSONObject) {
            let clock = ProcessInfo.processInfo.systemUptime
            let started = try await startWorker(
                tier, weights: weights, original: model, artifact: artifact, folder: folder, paths: paths, memoryGuard: guarder, isImage: true, assets: [:])
            workers.append(started.0)
            var load = started.1; load["cold_load_seconds"] = .number(ProcessInfo.processInfo.systemUptime - clock)
            return (started.0, load)
        }
        func capture(_ worker: WorkerProcess, _ name: String, create: Bool = false) async throws -> URL {
            let output = evidence.appendingPathComponent(name)
            _ = try await worker.request(
                op: "measurement_components", params: ["fixture_dir": .string(fixture.path), "output_dir": .string(output.path), "capture": .bool(create)], timeout: 120)
            try WireJSON.encode([
                "fixture_inputs_sha256": .string(try FileDigest.sha256(fixture.appendingPathComponent("inputs.safetensors"))),
                "component_count": .number(Double(biRefNetComponentKeys.count))
            ]).write(to: output.appendingPathComponent("fixture-identity.json"))
            try guarder.check(); return output
        }
        func serve(_ worker: WorkerProcess, _ label: String, image: Int = 0) async throws -> (Double, JSONObject) {
            let request: JSONObject = ["image": ["path": .string(images[image].path)], "out_dir": .string(paths.directory("jobs").appendingPathComponent(run + "/" + label).path)]
            let start = ProcessInfo.processInfo.systemUptime
            let reply = try await worker.request(op: "remove_background", params: request, timeout: 120)
            let elapsed = ProcessInfo.processInfo.systemUptime - start
            try guarder.check()
            guard let files = reply["files"]?.array, files.count == 2, Set(files.compactMap { $0["id"]?.string }) == ["foreground", "mask"] else {
                throw ManifestError.invalid("BiRefNet did not deliver both usable PNGs")
            }
            for file in files {
                guard let path = file["path"]?.string, try FileDigest.sha256(URL(fileURLWithPath: path)) == file["sha256"]?.string else {
                    throw ManifestError.invalid("BiRefNet output identity mismatch")
                }
            }
            return (elapsed, reply)
        }
        func persist() throws {
            guard let runner = model.runners.first, let implementation = artifact.qualityImplementation(model: model, runner: runner, directory: folder) else {
                throw ManifestError.invalid("Missing worker source closure")
            }
            try WireJSON.encode([
                "model": .string(model.id), "revision": .string(model.weights.revision), "run": .string(run),
                "date": .string(ISO8601DateFormatter().string(from: Date())), "chip": .string(HostChip.current), "gpu_cores": .number(Double(HostChip.gpuCores)),
                "worker_sha256": .string(try FileDigest.sha256(artifact.helpers.appendingPathComponent("arsenal-worker-native"))),
                "implementation_identity": .string(implementation), "load_identity": .string(try ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder)),
                "rows": .array(rows), "policy": .object(policy), "resource_receipt": .object(guarder.receipt()), "network_bytes": 0
            ]).write(to: result, options: .atomic)
        }
        do {
            let baseWeights = try await snapshot("fp16")
            let (captureBase, baseLoad) = try await start("fp16", baseWeights)
            _ = try await capture(captureBase, "reference", create: true)
            let replay = try await capture(captureBase, "reference-repeat")
            try freezeComponentBounds(reference: reference, repeatCapture: replay, evidence: evidence, half: "fp16", componentKeys: biRefNetComponentKeys, reused: false)
            let fixtureHash = try FileDigest.sha256(fixture.appendingPathComponent("inputs.safetensors"))
            let basePlan = try IntakeQualification.freeze(
                selection: fixture, selectionBounds: evidence.appendingPathComponent("bounds.json"), to: evidence.appendingPathComponent("qualification-plan-fp16.json"))
            let quantPlan = try IntakeQualification.freeze(
                selection: fixture, selectionBounds: evidence.appendingPathComponent("quantized-bounds.json"),
                to: evidence.appendingPathComponent("qualification-plan-quantized.json"))
            let baseGate = try IntakeComponentGate.evaluate(reference: reference, candidate: replay, bounds: evidence.appendingPathComponent("bounds.json"))
            let baseQualification = try IntakeQualification.complete(plan: basePlan, selection: baseGate, heldout: nil, searched: false, ordinaryLoad: baseLoad)
            guard baseQualification["passed"] == true else { throw ManifestError.invalid("FP16 reproducibility failed") }
            await captureBase.unload()
            let (base, loaded) = try await start("fp16", baseWeights)
            var baseSanity: [JSONObject] = [], baseSamples: [Double] = [], baseFirst = 0.0
            for i in images.indices {
                let first = try await serve(base, "fp16-sanity-\(i)", image: i)
                if i == 0 { baseFirst = first.0 }
                baseSanity.append(first.1)
            }
            for tier in ["int8", "int4"] {
                let weights = try await snapshot(tier)
                let (captureWorker, captureLoad) = try await start(tier, weights)
                let output = try await capture(captureWorker, tier)
                guard try FileDigest.sha256(fixture.appendingPathComponent("inputs.safetensors")) == fixtureHash else { throw ManifestError.invalid("Saved inputs changed") }
                let gate = try IntakeComponentGate.evaluate(reference: reference, candidate: output, bounds: evidence.appendingPathComponent("quantized-bounds.json"))
                let qualification = try IntakeQualification.complete(plan: quantPlan, selection: gate, heldout: nil, searched: false, ordinaryLoad: captureLoad)
                await captureWorker.unload()
                let (candidate, candidateLoad) = try await start(tier, weights)
                var sanity: [JSONValue] = [], firstSeconds = 0.0
                for i in images.indices {
                    let served = try await serve(candidate, tier + "-sanity-\(i)", image: i)
                    if i == 0 { firstSeconds = served.0 }
                    let reply = served.1
                    sanity.append(.object(try biRefNetMaskDiagnostic(reference: baseSanity[i], candidate: reply)))
                    try biRefNetSaveSanity(reply: reply, to: evidence, label: tier + "-\(i)")
                }
                _ = try await serve(base, tier + "-A-warm"); _ = try await serve(candidate, tier + "-B-warm")
                var a: [Double] = [], b: [Double] = []
                for (i, arm) in [0, 1, 1, 0, 1, 0, 0, 1].enumerated() {
                    let seconds = try await serve(arm == 0 ? base : candidate, tier + "-window-\(i)").0
                    if arm == 0 { a.append(seconds) } else { b.append(seconds) }
                }
                baseSamples += a
                var row: JSONObject = [
                    "precision": .string(tier), "gate": .object(gate), "selection_gate": .object(gate), "qualification": .object(qualification),
                    "qualified": qualification["passed"]!,
                    "load": .object(candidateLoad), "capture_load": .object(captureLoad), "serving_worker": .object(await servingWorkerReceipt(candidate)),
                    "peak_gb": .number(try await servingPeak(candidate)), "warm": IntakeTimingSummary(samples: b).json,
                    "paired_baseline_warm": IntakeTimingSummary(samples: a).json,
                    "baseline_drift_fraction": .number(IntakeTimingSummary(samples: Array(a.suffix(2))).median / IntakeTimingSummary(samples: Array(a.prefix(2))).median - 1),
                    "first_request_seconds": .number(firstSeconds), "end_to_end": .array(sanity), "state": "quickCheck"
                ]
                row.merge(precisionOffering(dtype: tier, passed: qualification["passed"]?.bool)) { _, new in new }
                rows.append(.object(row)); try persist()
                await candidate.unload()
                print("BiRefNet \(tier): component gate \(qualification["passed"] == true ? "pass" : "fail"); \(IntakeTimingSummary(samples: b).median * 1000) ms/image (busy Mac)")
            }
            for i in baseSanity.indices { try biRefNetSaveSanity(reply: baseSanity[i], to: evidence, label: "fp16-\(i)") }
            rows.insert(
                [
                    "precision": "fp16", "gate": .object(baseGate), "selection_gate": .object(baseGate), "qualification": .object(baseQualification), "qualified": true,
                    "load": .object(loaded), "capture_load": .object(baseLoad), "serving_worker": .object(await servingWorkerReceipt(base)),
                    "first_request_seconds": .number(baseFirst), "peak_gb": .number(try await servingPeak(base)), "warm": IntakeTimingSummary(samples: baseSamples).json,
                    "state": "quickCheck"
                ], at: 0)
            try persist()
            try writeSoundStats(rows: rows, model: model, folder: folder, artifact: artifact, run: run, result: result)
            for worker in workers { await worker.unload() }
            await artifact.shutdown()
            print("BiRefNet completed: " + run)
        } catch {
            for worker in workers { await worker.unload() }
            await artifact.shutdown(); try? persist(); throw error
        }
    }
}
