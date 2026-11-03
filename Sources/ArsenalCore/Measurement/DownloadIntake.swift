// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// The three explicitly authorized network intakes. All other intake remains local-source-only.
extension ModelIntakeCommand {
    static let downloadedModels: Set<String> = ["von-1.1", "von-1.2", "re-use"]
    static let vonModels: Set<String> = ["von-1.1", "von-1.2"]

    static func downloadIntakeSource(_ args: [String]) async throws {
        func option(_ key: String) -> String? { args.firstIndex(of: key).flatMap { args.indices.contains($0 + 1) ? args[$0 + 1] : nil } }
        guard args.contains("--go"), let path = option("--model"), let support = option("--support"), let helpers = option("--helpers") else {
            throw ManifestError.invalid("Source Get needs --model, --helpers, fresh lab --support and --go")
        }
        let folder = URL(fileURLWithPath: path), lab = URL(fileURLWithPath: support).standardizedFileURL
        let model = try ModelManifest.load(folder)
        guard downloadedModels.contains(model.id), model.weights.dtype == "fp32", lab.pathComponents.contains("tmp"),
            !lab.resolvingSymlinksInPath().path.contains("Library/Application Support"), !FileManager.default.fileExists(atPath: lab.path)
        else { throw ManifestError.invalid("Source Get is restricted to the three approved pinned models in a fresh tmp lab") }
        let paths = ArsenalPaths(support: lab); try paths.create()
        let artifact = BundledWorkerArtifact(helpers: URL(fileURLWithPath: helpers))
        let converter = CapturingIntakeConverter(base: WorkerPrecisionConverter(artifacts: artifact, paths: paths), destination: lab.appendingPathComponent("pinned-source"))
        let fetcher = ModelFetcher(support: lab, converter: converter, localSources: [])
        defer { for name in ["weights", "cache"] { try? FileManager.default.removeItem(at: lab.appendingPathComponent(name)) } }
        let files = try intakePinnedFiles(model.weights)
        do {
            _ = try await fetcher.get(
                model: model, modelDirectory: folder, precision: model.basePrecision, approvedDownloads: Dictionary(uniqueKeysWithValues: files.map { ($0.sha256, $0.bytes) }))
            let receipt: JSONObject = [
                "model": .string(model.id), "revision": .string(model.weights.revision), "download_bytes": .number(Double(files.reduce(0) { $0 + $1.bytes })),
                "files": .array(files.map { ["path": .string($0.path), "bytes": .number(Double($0.bytes)), "sha256": .string($0.sha256)] })
            ]
            try WireJSON.encode(receipt).write(to: lab.appendingPathComponent("download-receipt.json"), options: .withoutOverwriting)
            await artifact.shutdown(); print("Pinned source Get complete: " + lab.appendingPathComponent("pinned-source").path)
        } catch { await artifact.shutdown(); throw error }
    }

    static func downloadedComponentKeys(_ model: ModelManifest, source: URL) throws -> Set<String> {
        if model.id == "re-use" { return Set(["encoder", "magnitude", "phase"] + (0..<30).flatMap { ["block-\($0)/time", "block-\($0)/frequency"] }) }
        let config = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("config.json")))
        guard let layers = config["num_hidden_layers"]?.number, layers > 0 else { throw ManifestError.invalid("Von layer configuration missing") }
        return Set((0..<2).flatMap { c in ["case-\(c)/embedding", "case-\(c)/scorer"] + (0..<Int(layers)).map { "case-\(c)/layer-\($0)" } })
    }

    static func enhancementIntakePolicy(runID: String, fixture: JSONObject, model: ModelManifest, keys: Set<String>?) -> JSONObject {
        [
            "run": .string(runID), "fixture": .object(fixture), "component_keys": .array((keys ?? []).sorted().map(JSONValue.string)),
            "timing_order": "ABBA BAAB (four synchronized warm requests per arm); resident baseline across pairs",
            "format_gate": "63 teacher-forced outputs: spectral encoder, 30 time/frequency blocks, magnitude and phase decoders; source-format policy frozen before FP16",
            "quantized_tier_gate": .string(IntakeComponentGate.quantizedBasis),
            "reference_scope": .string(
                "Pinned FP32-only RE-USE source; GPU BF16 nearest/truncation calibration under the shared source-format policy. Selected " + model.basePrecision.uppercased()
                    + "; no FP32 learned parameters executed. No new independent vendor qualification."),
            "independent_reference_qualified": false, "machine_in_use": true
        ]
    }

    static func recordDownloadedCoverage(model: ModelManifest, dtype: String, weights: URL, evidence: URL) throws {
        let name = model.id == "re-use" ? "component-coverage.json" : ["fp16", "bf16"].contains(dtype) ? "manifest.json" : "arsenal-decisions-view.json"
        let legacy = name.replacingOccurrences(of: "arsenal-", with: "mobius-") // legacy-compat: views written before the rename
        let data = try Data(contentsOf: StoredNames.current(name, legacy: legacy, in: weights))
        try data.write(to: evidence.appendingPathComponent(dtype + "-component-coverage.json"), options: .withoutOverwriting)
    }

    static func acceptDownloadedFormat(model: ModelManifest, from source: URL, evidence: URL) throws {
        let policy = try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("policy.json")))
        let expected: Set<DerivedFile> = Set(try intakePinnedFiles(model.weights))
        let captured = try policy["source_files"]?.array?.map { value -> DerivedFile in
            guard let path = value["path"]?.string, let hash = value["sha256"]?.string, let bytes = value["bytes"]?.number else {
                throw ManifestError.invalid("Invalid source format pin")
            }
            return DerivedFile(path: path, bytes: Int64(bytes), sha256: hash)
        }
        let bounds = source.appendingPathComponent("bounds.json")
        guard policy["policy_kind"]?.string == IntakeComponentGate.fp32SourcePolicyID, let captured, Set(captured) == expected else {
            throw ManifestError.invalid("Source format pins or policy differ")
        }
        let gate = try IntakeComponentGate.evaluate(
            reference: source.appendingPathComponent("bf16-reference"), candidate: source.appendingPathComponent("fp16-candidate"), bounds: bounds)
        try IntakeComponentGate.validateSavedEvaluation(try WireJSON.decode(Data(contentsOf: source.appendingPathComponent("format-gate.json"))), current: gate)
        guard model.basePrecision == (gate["component_gate_passed"]?.bool == true ? "fp16" : "bf16")
        else { throw ManifestError.invalid("Selected half disagrees with frozen source-format result") }
        try FileManager.default.copyItem(at: source, to: evidence.appendingPathComponent("format-qualification"))
        let target = evidence.appendingPathComponent("format-qualification")
        if !FileManager.default.fileExists(atPath: target.appendingPathComponent("format-gate-source.json").path) {
            try FileManager.default.copyItem(at: target.appendingPathComponent("format-gate.json"), to: target.appendingPathComponent("format-gate-source.json"))
        }
        try WireJSON.encode(gate).write(to: target.appendingPathComponent("format-gate.json"), options: .atomic)
    }

    /// BF16 source-weight rounding establishes the frozen bound; FP16 is never its own format reference.
    static func qualifyDownloadedSource(
        model: ModelManifest, source: URL, evidence: URL, lab: URL, artifact: BundledWorkerArtifact, paths: ArsenalPaths,
        keys: Set<String>, fp16: URL, start: (String, URL) async throws -> (WorkerProcess, JSONObject)
    ) async throws {
        guard downloadedModels.contains(model.id), let runner = model.runners.first else { return }
        let root = evidence.appendingPathComponent("format-qualification"), fixture = root.appendingPathComponent("fixture")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let weightRoot = lab.appendingPathComponent("format-weights")
        defer { try? FileManager.default.removeItem(at: weightRoot) }
        var sources: [JSONValue] = []
        for file in try intakePinnedFiles(model.weights) {
            let from = try SafePath.resolve(file.path, inside: source)
            guard try FileDigest.sha256(from) == file.sha256, try FileDigest.bytes(from) == file.bytes else { throw ManifestError.invalid("Source pin differs before format cast") }
            sources.append(["path": .string(file.path), "sha256": .string(file.sha256), "bytes": .number(Double(file.bytes))])
        }
        try WireJSON.encode(["policy_kind": .string(IntakeComponentGate.fp32SourcePolicyID), "basis": .string(IntakeComponentGate.fp32SourceBasis), "source_files": .array(sources)]
        )
        .write(to: root.appendingPathComponent("policy.json"), options: .withoutOverwriting)
        for (label, truncate) in [("bf16-reference", false), ("bf16-rounding", true)] {
            let directory = weightRoot.appendingPathComponent(label)
            let converter = WorkerProcess()
            do {
                try await converter.start(artifact.launch(model: model, runner: runner, directory: URL(fileURLWithPath: "."), paths: paths))
                _ = try await converter.request(
                    op: "measurement_source", params: ["src_dir": .string(source.path), "dst_dir": .string(directory.path), "truncate": .bool(truncate)], timeout: 90)
                await converter.unload()
            } catch { await converter.unload(); throw error }
            let (worker, _) = try await start("bf16", directory)
            do {
                _ = try await worker.request(
                    op: "measurement_components",
                    params: ["fixture_dir": .string(fixture.path), "output_dir": .string(root.appendingPathComponent(label).path), "capture": .bool(!truncate)], timeout: 120)
                if !truncate {
                    _ = try await worker.request(
                        op: "measurement_components",
                        params: ["fixture_dir": .string(fixture.path), "output_dir": .string(root.appendingPathComponent("bf16-repeat").path), "capture": false], timeout: 120)
                }
                await worker.unload()
            } catch { await worker.unload(); throw error }
        }
        let bounds = root.appendingPathComponent("bounds.json")
        try IntakeComponentGate.freezeFP32Source(
            reference: root.appendingPathComponent("bf16-reference"), roundingCapture: root.appendingPathComponent("bf16-rounding"),
            repeatCapture: root.appendingPathComponent("bf16-repeat"), to: bounds, expectedKeys: keys)
        let (worker, _) = try await start("fp16", fp16)
        do {
            _ = try await worker.request(
                op: "measurement_components",
                params: ["fixture_dir": .string(fixture.path), "output_dir": .string(root.appendingPathComponent("fp16-candidate").path), "capture": false], timeout: 120)
            await worker.unload()
        } catch { await worker.unload(); throw error }
        let gate = try IntakeComponentGate.evaluate(
            reference: root.appendingPathComponent("bf16-reference"), candidate: root.appendingPathComponent("fp16-candidate"), bounds: bounds)
        try WireJSON.encode(gate).write(to: root.appendingPathComponent("format-gate.json"), options: .withoutOverwriting)
        let selected = gate["component_gate_passed"]?.bool == true ? "fp16" : "bf16"
        guard model.basePrecision == selected else {
            throw ManifestError.invalid("Selected half disagrees with frozen source-format result; \(selected.uppercased()) recipe required")
        }
    }
}

private struct CapturingIntakeConverter: PrecisionConverter {
    let base: WorkerPrecisionConverter
    let destination: URL
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile] {
        try await convert(
            model: model, modelDirectory: modelDirectory, sourceDirectory: sourceDirectory, assets: [:], destinationDirectory: destinationDirectory, precision: precision,
            recipe: recipe)
    }
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, assets: [String: URL], destinationDirectory: URL, precision: String, recipe: String) async throws
        -> [DerivedFile]
    {
        if precision == model.basePrecision, model.weights.dtype == "fp32" {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            for file in try intakePinnedFiles(model.weights) {
                let source = try SafePath.resolve(file.path, inside: sourceDirectory).resolvingSymlinksInPath(), target = try SafePath.resolve(file.path, inside: destination)
                guard try FileDigest.sha256(source) == file.sha256 else { throw ManifestError.invalid("Get source pin mismatch") }
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                let copy = Process(); copy.executableURL = URL(fileURLWithPath: "/bin/cp"); copy.arguments = ["-c", source.path, target.path]
                try copy.run(); copy.waitUntilExit()
                guard copy.terminationStatus == 0 else { throw ManifestError.invalid("Pinned-source clone failed") }
            }
        }
        return try await base.convert(
            model: model, modelDirectory: modelDirectory, sourceDirectory: sourceDirectory, assets: assets, destinationDirectory: destinationDirectory, precision: precision,
            recipe: recipe)
    }
}

private func intakePinnedFiles(_ source: WeightSpec) throws -> [DerivedFile] {
    try source.files.map { path in
        guard let sha = source.sha256?[path], let bytes = source.bytes?[path] else { throw ManifestError.invalid("Source needs pinned hash and size: " + path) }
        return DerivedFile(path: path, bytes: bytes, sha256: sha.lowercased())
    }
}
