// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

public protocol WorkerArtifact: Sendable {
    func qualityImplementation(model: ModelManifest, runner: RunnerSpec, directory: URL) -> String?
    func qualityImplementation(model: ModelManifest, runner: RunnerSpec, directory: URL, catalogFiles: [CatalogFile]) -> String?
    func validateEngine(model: ModelManifest, runner: RunnerSpec) throws
    func shutdown() async
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL, approvedPlan: GetPlan) async throws
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws
}
extension WorkerArtifact {
    public func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL, approvedPlan: GetPlan) async throws {
        try await prepare(model: model, runner: runner, directory: directory)
    }
    public func qualityImplementation(model: ModelManifest, runner: RunnerSpec, directory: URL) -> String? { nil }
    public func qualityImplementation(model: ModelManifest, runner: RunnerSpec, directory: URL, catalogFiles: [CatalogFile]) -> String? {
        qualityImplementation(model: model, runner: runner, directory: directory)
    }
    public func validateEngine(model: ModelManifest, runner: RunnerSpec) throws {}
    public func shutdown() async {}
}
public struct BundledWorkerArtifact: WorkerArtifact {
    public let helpers: URL
    /// The native workers' source-closure receipt. It lives in Contents/Resources: codesign rejects data files in Helpers.
    public let qualityIdentities: URL
    public init(helpers: URL? = nil, qualityIdentities: URL? = nil) {
        let helpers =
            helpers ?? Bundle.main.executableURL?.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Helpers")
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".build/debug")
        self.helpers = helpers
        self.qualityIdentities =
            qualityIdentities ?? helpers.deletingLastPathComponent().appendingPathComponent("Resources/quality-identities.json")
    }
    public func qualityImplementation(model: ModelManifest, runner: RunnerSpec, directory: URL) -> String? {
        if let entry = runner.entry, let file = try? SafePath.resolve(entry, inside: directory) {
            return try? FileDigest.sha256(file)
        }
        guard let data = try? Data(contentsOf: qualityIdentities),
            let identities = try? JSONDecoder().decode([String: String].self, from: data)
        else { return nil }
        return identities[runner.worker ?? model.kind]
    }
    public func qualityImplementation(model: ModelManifest, runner: RunnerSpec, directory: URL, catalogFiles: [CatalogFile]) -> String? {
        if let entry = runner.entry { return catalogFiles.first { $0.path == entry }?.sha256.lowercased() }
        return qualityImplementation(model: model, runner: runner, directory: directory)
    }
    public func validateEngine(model: ModelManifest, runner: RunnerSpec) throws {
        guard runner.language != "python" else { throw WorkerFailure(code: "unsupported_runner", message: "Python model runners are unsupported; choose a native runner.") }
        guard !(runner.language == "swift" && runner.entry != nil) else { return }
        let binary = helpers.appendingPathComponent("arsenal-worker-\(runner.worker ?? model.kind)")
        guard FileManager.default.isExecutableFile(atPath: binary.path) else {
            throw WorkerFailure(code: "worker_missing", message: "This model needs an engine that is not in this Arsenal build")
        }
    }
    public func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    public func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        try validateEngine(model: model, runner: runner)
        var environment = RunnerEnvironment.scrub(); environment["ARSENAL_SUPPORT_DIR"] = paths.support.path; environment["ARSENAL_CACHE_DIR"] = paths.cache.path
        for name in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"] { environment[name] = "1" }
        LogRetention.prune(paths.directory("logs"))
        let log = paths.directory("logs").appendingPathComponent("\(model.id)-\(UUID().uuidString).log")
        let binary =
            try runner.language == "swift" && runner.entry != nil
            ? SafePath.resolve(runner.entry!, inside: directory)
            : helpers.appendingPathComponent("arsenal-worker-\(runner.worker ?? model.kind)")
        guard FileManager.default.isExecutableFile(atPath: binary.path) else {
            throw WorkerFailure(code: "worker_missing", message: "The approved Swift runner executable is missing or not executable.")
        }
        return WorkerLaunch(executable: binary, kind: model.kind, runner: runner.name, environment: environment, log: log)

    }
}
public struct WorkerPrecisionConverter: PrecisionConverter {
    public let artifacts: any WorkerArtifact
    public let paths: ArsenalPaths
    public let runnerNames: [String: String]
    /// The Mac-wide memory stop the conversion worker takes part in (a simulated Mac in tests).
    public let macMemory: MacMemoryGuard
    public init(artifacts: any WorkerArtifact, paths: ArsenalPaths, runnerNames: [String: String] = [:], macMemory: MacMemoryGuard = .shared) {
        self.artifacts = artifacts; self.paths = paths; self.runnerNames = runnerNames; self.macMemory = macMemory
    }
    public func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile]
    {
        try await convert(
            model: model, modelDirectory: modelDirectory, sourceDirectory: sourceDirectory, assets: [:], destinationDirectory: destinationDirectory, precision: precision,
            recipe: recipe)
    }
    public func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, assets: [String: URL], destinationDirectory: URL, precision: String, recipe: String)
        async throws -> [DerivedFile]
    {
        guard
            let runner = (runnerNames[model.id] ?? model.selection?.runner(chip: HostChip.current)).flatMap({ name in model.runners.first { $0.name == name } })
                ?? model.runners.first
        else { throw WorkerFailure(code: "runner_missing", message: "No conversion runner is available.") }
        let worker = WorkerProcess(macMemory: macMemory)
        // Startup can throw on cancellation before hello. The caller must await the same
        // cleanup for startup and requests, rather than handing it to an unstructured task.
        do {
            let launch = try await artifacts.launch(model: model, runner: runner, directory: modelDirectory, paths: paths)
            try Task.checkCancellation()
            try await worker.start(launch)
            let timeout = Self.convertTimeout(bytes: Self.treeBytes([sourceDirectory] + assets.values))
            let result = try await worker.request(
                op: "convert",
                params: [
                    "model_dir": .string(modelDirectory.path), "src_dir": .string(sourceDirectory.path), "dst_dir": .string(destinationDirectory.path),
                    "dtype": .string(precision), "group_size": 64, "recipe": .string(recipe), "assets": .object(assets.mapValues { .string($0.path) })
                ], timeout: timeout,
                // Converting has no model figures: the Mac-wide stop applies, named for what was running.
                memoryCeiling: .unmetered("the \(precision.uppercased()) conversion of \(model.name)"))
            await worker.unload()
            guard let list = result["files"]?.array else { throw WorkerFailure(code: "conversion", message: "Converter returned no file list.") }
            let converted = try list.map { value in
                guard let object = value.object else { throw WireError.malformedFrame }
                var file = try JSONDecoder().decode(DerivedFile.self, from: WireJSON.encode(object))
                if file.path.hasPrefix("/") {
                    let url = URL(fileURLWithPath: file.path).standardizedFileURL
                    let prefix = destinationDirectory.standardizedFileURL.path + "/"
                    guard url.path.hasPrefix(prefix) else { throw WorkerFailure(code: "conversion_path", message: "Converter wrote outside its staging directory.") }
                    file.path = String(url.path.dropFirst(prefix.count))
                }
                _ = try SafePath.resolve(file.path, inside: destinationDirectory)
                return file
            }
            // Snapshot/provenance manifests belong to the store, never to a worker's copied source view.
            let reserved = StoredNames.reserved
            for file in converted where reserved.contains(file.path) { try FileManager.default.removeItem(at: SafePath.resolve(file.path, inside: destinationDirectory)) }
            return converted.filter { !reserved.contains($0.path) }
        } catch { await worker.unload(); throw error }
    }

    /// Conversion reads what it is given, so its deadline grows with those bytes: at least 600 s, plus 300 s per GiB.
    /// The per-GiB rate is a conservative placeholder until the 8–9B conversions are measured on this Mac.
    static let convertFloorSeconds: TimeInterval = 600
    static let convertSecondsPerGiB: Double = 300
    static func convertTimeout(bytes: Int64) -> TimeInterval {
        max(convertFloorSeconds, Double(bytes) / Double(1 << 30) * convertSecondsPerGiB)
    }
    /// Bytes of every file under the given paths (a file counts as itself; a missing path counts as nothing).
    static func treeBytes(_ urls: [URL]) -> Int64 {
        urls.reduce(Int64(0)) { total, url in
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else { return total }
            guard isDirectory.boolValue else { return total + ((try? FileDigest.bytes(url)) ?? 0) }
            guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else { return total }
            var sum = total
            for case let file as URL in walker where (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                sum += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
            return sum
        }
    }
}
