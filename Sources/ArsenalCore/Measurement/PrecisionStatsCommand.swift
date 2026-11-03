// SPDX-License-Identifier: MIT
import Foundation

actor LiveEnergyProbe {
    private var meter: IOReportEnergyMeter?
    func begin() throws { if meter == nil { meter = try IOReportEnergyMeter() }; try meter?.begin() }
    func end() throws -> CPUGPUEnergy {
        guard let meter else { throw ManifestError.invalid("Energy sampler unavailable") }
        return try meter.end()
    }
}

public enum PrecisionStatsCommand {
    public static func run(arguments: [String]) async -> Int32 {
        do { try await execute(arguments); return 0 } catch {
            FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8)); return 1
        }
    }
    static func execute(_ args: [String]) async throws {
        guard let command = args.first, ["seed", "list", "run"].contains(command) else { throw ManifestError.invalid("Expected seed, list or run") }
        func option(_ flag: String) -> String? { args.firstIndex(of: flag).flatMap { index in args.indices.contains(index + 1) ? args[index + 1] : nil } }
        guard let supportPath = option("--support") ?? ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] else {
            throw ManifestError.invalid("An explicit isolated or approved support directory is required")
        }
        let support = URL(fileURLWithPath: supportPath), queue = PendingBenchmarkQueue(support: support)
        if command == "list" {
            let rows = try queue.read()
            print("\(rows.filter { $0.state == "pending" }.count) pending / \(rows.count) total precision runs")
            return
        }
        guard let modelRoot = option("--models") else { throw ManifestError.invalid("Model folder root is required") }
        let models = URL(fileURLWithPath: modelRoot)
        if command == "seed" {
            var added = 0
            for folder in try FileManager.default.contentsOfDirectory(at: models, includingPropertiesForKeys: nil).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("model.toml").path) else { continue }
                let model = try ModelManifest.load(folder, kindsDirectory: models.deletingLastPathComponent().appendingPathComponent("kinds"))
                let runner = model.selection?.runner(chip: HostChip.current) ?? model.runners.first?.name ?? "Unavailable"
                let hasFastKernels = try KernelManifest.loadIfPresent(folder: folder).kernels.contains {
                    $0.mode == "fast" && ($0.chip == "*" || $0.chip == RunnerSelection.family(HostChip.current))
                }
                let mode = model.runners.first(where: { $0.name == runner })?.fastExact == true || hasFastKernels ? "fast" : "exact"
                added += try queue.introduce(model, runner: runner, mode: mode)
            }
            print("Queued \(added) precision runs; none executed")
            return
        }
        guard args.contains("--go") else { throw ManifestError.invalid("Queue held. Run requires the owner's current go via --go; nothing executed") }
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        let helpers =
            option("--helpers").map { URL(fileURLWithPath: $0) }
            ?? (executable.deletingLastPathComponent().lastPathComponent == "MacOS"
                ? executable.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Helpers") : executable.deletingLastPathComponent())
        let resources = option("--resources").map { URL(fileURLWithPath: $0) } ?? helpers.deletingLastPathComponent().appendingPathComponent("Resources")
        let sourceSupport = option("--source-support").map { URL(fileURLWithPath: $0) } ?? support
        let adapterHash = try FileDigest.sha256(executable)
        let workerHash = try FileDigest.sha256(helpers.appendingPathComponent("arsenal-worker-native"))
        let lease = try BenchmarkRunLease(support: support)
        defer { withExtendedLifetime(lease) {} }
        // Retire old mode/revision entries before taking the held drain snapshot.
        for request in try queue.read().filter({ $0.state == "pending" }) {
            guard let folder = try? SafePath.resolve(request.model, inside: models),
                let model = try? ModelManifest.load(
                    folder,
                    kindsDirectory: models.deletingLastPathComponent().appendingPathComponent("kinds"))
            else { continue }
            let kernels = try KernelManifest.loadIfPresent(folder: folder)
            let fast =
                model.runners.first { $0.name == request.runner }?.fastExact == true
                || kernels.kernels.contains {
                    $0.mode == "fast" && ($0.chip == "*" || $0.chip == RunnerSelection.family(HostChip.current))
                }
            try queue.introduce(model, runner: request.runner, mode: fast ? "fast" : "exact")
        }
        let pending = try queue.read().filter { $0.state == "pending" }
        let groups = Dictionary(grouping: pending) { [$0.model, $0.revision, $0.runner, $0.mode].joined(separator: ":") }
        var completed = 0
        for key in groups.keys.sorted() {
            guard let requests = groups[key], let first = requests.first else { continue }
            let folder: URL, manifest: ModelManifest
            do {
                folder = try SafePath.resolve(first.model, inside: models)
                manifest = try ModelManifest.load(folder, kindsDirectory: models.deletingLastPathComponent().appendingPathComponent("kinds"))
            } catch {
                for request in requests { try queue.recordFailure(id: request.id, reason: error.localizedDescription) }
                continue
            }
            guard manifest.weights.revision == first.revision else {
                for request in requests { try queue.recordFailure(id: request.id, reason: "Pinned model revision changed; introduce its current run before measuring") }
                continue
            }
            let scratch = support.appendingPathComponent("benchmarks/profile-" + UUID().uuidString)
            let paths = ArsenalPaths(support: scratch)
            var opened: RealModelsService?
            let service: RealModelsService
            do {
                try paths.create()
                var config = ArsenalConfiguration()
                config.runnerOverrides[first.model] = first.runner
                // No source pruning in a reference/candidate bracket, no saved settings or warm residents.
                try JSONEncoder().encode(config).write(to: scratch.appendingPathComponent("config.json"), options: .atomic)
                service = try await RealModelsService.open(
                    repository: resources, resources: resources, paths: paths,
                    artifacts: BundledWorkerArtifact(helpers: helpers), sourceRetentionThresholdBytes: .max)
                opened = service
                try await service.add(id: first.model)
                let served = try await service.measurementManifest(id: first.model)
                guard served.weights.revision == first.revision else { throw ManifestError.invalid("Bundled served revision differs from queued revision") }
                let row = await service.rows().first { $0.id == first.model }
                if let reason = ServingMeasurementPolicy.refusal(model: first.model, fast: row?.fast == true) { throw ManifestError.invalid(reason) }
                try await MeasurementProfile.importSource(model: served, sourceSupport: sourceSupport, paths: paths)
            } catch {
                await opened?.shutdown()
                for request in requests { try queue.recordFailure(id: request.id, reason: error.localizedDescription) }
                try? FileManager.default.removeItem(at: scratch)
                continue
            }
            let served = try await service.measurementManifest(id: first.model)
            let effectiveMode = await service.rows().first { $0.id == first.model }?.fast == true ? "fast" : "exact"
            guard effectiveMode == first.mode else {
                await service.shutdown()
                try queue.introduce(served, runner: first.runner, mode: effectiveMode)
                try? FileManager.default.removeItem(at: scratch)
                continue
            }
            let adapterDirectory = scratch.appendingPathComponent("benchmarks/output")
            let adapter = MeasurementAdapter(service: service, model: served, directory: adapterDirectory)
            let energy = LiveEnergyProbe()
            let probe = MeasurementProbe(capacityJoules: try? BatteryEnergy.liveCapacityJoules(), begin: { try await energy.begin() }, end: { try await energy.end() })
            let method = MeasurementMethod(
                chip: HostChip.current, controlled: true, source: "weights " + first.revision + "; drainer sha256 " + adapterHash + "; served worker sha256 " + workerHash,
                gpuCores: HostChip.gpuCores > 0 ? HostChip.gpuCores : nil)
            do {
                let results = try await PrecisionMeasurementHarness.run(
                    model: first.model, kind: first.kind, sourceDtype: first.sourceDtype, revision: first.revision,
                    runnerID: first.runner, mode: effectiveMode, runner: adapter, probe: probe, method: method, queue: queue, approved: true)
                for var result in results {
                    result.stats.qualityIdentity = qualityIdentity(result, model: manifest, helpers: helpers, folder: folder)
                    guard let request = requests.first(where: { $0.dtype == result.stats.dtype }) else { continue }
                    let speed: StatMetric =
                        first.kind == .chat ? .tokensPerSecond : first.kind == .decisions ? .millisecondsPerDecision : first.kind == .images ? .secondsPerImage : .realTimeFactor
                    if result.stats.value(speed).value == nil || result.stats.availability.reason?.hasPrefix("Own-16-bit timing bracket failed:") == true {
                        try queue.recordFailure(id: request.id, reason: result.stats.availability.reason ?? "No measurements produced")
                        continue
                    }
                    let resultName = "results/local/\(HostChip.current)-\(request.dtype)-\(request.runner)-\(request.mode)-\(UUID().uuidString).md"
                    let file = try SafePath.resolve(resultName, inside: folder)
                    try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try result.markdown(notes: "Pending queue; Mac in use. Runner and battery hardware still require independent verification.").write(to: file, options: .atomic)
                    try queue.complete(id: request.id, result: file.path); completed += 1
                }
                for request in requests where !results.contains(where: { $0.stats.dtype == request.dtype }) {
                    try queue.recordFailure(id: request.id, reason: "Original 16-bit reference failed; comparison deferred")
                }
            } catch {
                for request in requests { try queue.recordFailure(id: request.id, reason: error.localizedDescription) }
            }
            await adapter.shutdown()
            // Retain ordinary worker stderr before deleting cloned weights, derived snapshots, gates and jobs.
            let logs = (try? FileManager.default.contentsOfDirectory(at: paths.directory("logs"), includingPropertiesForKeys: nil)) ?? []
            for log in logs {
                let destination = support.appendingPathComponent("benchmarks/logs/" + log.lastPathComponent)
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? FileManager.default.copyItem(at: log, to: destination)
            }
            try? FileManager.default.removeItem(at: scratch)
        }
        let remaining = try queue.read().filter { $0.state == "pending" }
        print("\(completed) precision runs recorded; \(remaining.count) pending")
        if !remaining.isEmpty {
            let failures = Dictionary(grouping: remaining) { $0.model + "/" + $0.runner }
            let reasons = failures.keys.sorted().map { key in
                let entries = failures[key] ?? []
                let first = entries.first { $0.dtype == $0.sourceDtype } ?? entries.first
                return key + ": " + (first?.error ?? "Measurement remains pending")
            }
            throw ManifestError.invalid("Queue remains pending:\n" + reasons.joined(separator: "\n"))
        }
    }

    private static func qualityIdentity(_ result: PrecisionStatsResult, model: ModelManifest, helpers: URL, folder: URL) -> String? {
        PrecisionEvidence.qualityIdentity(
            model: model, dtype: result.stats.dtype, runner: result.stats.runner,
            implementation: model.runners.first(where: { $0.name == result.stats.runner }).flatMap {
                BundledWorkerArtifact(helpers: helpers).qualityImplementation(model: model, runner: $0, directory: folder)
            },
            loadIdentity: model.runners.first(where: { $0.name == result.stats.runner }).flatMap {
                try? ModelLoadConfiguration.identity(model: model, runner: $0, folder: folder, fast: result.mode == "fast")
            })
    }

}
