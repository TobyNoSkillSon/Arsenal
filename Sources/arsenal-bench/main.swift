// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalBench

@main struct ArsenalBenchCommand {
    static func main() async {
        do { try await execute(Array(CommandLine.arguments.dropFirst())) } catch {
            FileHandle.standardError.write(Data("arsenal-bench: \(error.localizedDescription.replacingOccurrences(of: "\n", with: " "))\n".utf8))
            exit(error.localizedDescription.contains("GPU gate closed") ? 75 : 1)
        }
    }
    static func execute(_ args: [String]) async throws {
        if args.isEmpty || args.contains("--help") {
            print(
                "arsenal-bench run <model> [--precision bf16|fp16|int8|int4] [--mode fast|exact] [--suite suite-v1] [--profile quick|full] [--stub|--quick-check] [--repo PATH] [--resources DIR] [--helpers DIR] [--out DIR] [--json]; validate FILE; self-test"
            )
            return
        }
        let repository = URL(fileURLWithPath: option("--repo", args) ?? FileManager.default.currentDirectoryPath)
        if args[0] == "validate", args.count == 2 {
            let data = try Data(contentsOf: URL(fileURLWithPath: args[1]))
            if args[1].hasSuffix(".md") { try ResearchLogValidation.validate(markdown: String(decoding: data, as: UTF8.self)) } else { try ResultValidation.validate(data: data) }
            print("Valid \(args[1])."); return
        }
        if args[0] == "self-test" {
            try await BenchmarkSelfTest.run(repository: repository)
            print("Benchmark checks passed: all four stub suites, contamination, closed gate, validation, badges and ranking."); return
        }
        if args[0] == "prepare-proof-catalog", let resources = option("--resources", args) {
            try await BenchmarkLocalProof.prepareCatalog(repository: repository, resources: URL(fileURLWithPath: resources), paths: ArsenalPaths())
            print("Prepared signed local proof catalog; model folders and weights are installed only by Get."); return
        }
        if args[0] == "get", args.count > 1 {
            guard BenchmarkLocalProof.allowedModels.contains(args[1]) else { throw BenchmarkError.invalid("Product proof Get is currently limited to Kokoro 82M.") }
            let paths = ArsenalPaths(), helpers = option("--helpers", args).map { URL(fileURLWithPath: $0) }
            let artifacts = BundledWorkerArtifact(helpers: helpers)
            let real = try await RealModelsService.open(
                repository: repository, resources: option("--resources", args).map { URL(fileURLWithPath: $0) }, paths: paths, artifacts: artifacts)
            do { try await real.get(id: args[1], precision: option("--precision", args) ?? "fp16") } catch { await real.shutdown(); throw error }
            await real.shutdown(); print("Installed \(args[1]) through RealModelsService Get."); return
        }
        guard args[0] == "run", args.count > 1, !args[1].hasPrefix("--") else { throw BenchmarkError.invalid("Use arsenal-bench --help.") }
        let flags: Set<String> = ["--stub", "--json", "--quick-check"]
        let options: Set<String> = ["--repo", "--resources", "--helpers", "--out", "--precision", "--mode", "--suite", "--profile"]
        var index = 2
        while index < args.count {
            if flags.contains(args[index]) {
                index += 1
            } else if options.contains(args[index]), index + 1 < args.count, !args[index + 1].hasPrefix("--") {
                index += 2
            } else {
                throw BenchmarkError.invalid("Unknown or missing benchmark option \(args[index]).")
            }
        }
        let request = BenchmarkRequest(
            id: args[1], precision: option("--precision", args) ?? "bf16", mode: option("--mode", args) ?? "fast",
            suite: option("--suite", args) ?? "suite-v1", profile: option("--profile", args) ?? "quick", quickCheck: args.contains("--quick-check"))
        try request.validate()
        let stub = args.contains("--stub")
        let output = URL(fileURLWithPath: option("--out", args) ?? ArsenalPaths().directory("jobs/bench").path)
        let service: any ModelsService
        var realService: RealModelsService?
        var configuration = BenchmarkConfiguration(repository: repository, outputDirectory: output, stub: stub)
        if stub {
            let fake = StubModelsService(); try await fake.get(id: request.id, precision: request.precision); service = fake
        } else {
            try BenchmarkGate.check(repository: repository, requireMeasurement: !request.quickCheck)
            let paths = ArsenalPaths()
            let helpers = option("--helpers", args).map { URL(fileURLWithPath: $0) }
            let artifacts = BundledWorkerArtifact(helpers: helpers)
            let real = try await RealModelsService.open(
                repository: repository, resources: option("--resources", args).map { URL(fileURLWithPath: $0) }, paths: paths, artifacts: artifacts)
            service = real; realService = real
            let manifest = try ModelManifest.load(paths.directory("models").appendingPathComponent(request.id))
            guard let row = await real.rows().first(where: { $0.id == request.id }),
                let runner = manifest.runners.first(where: { $0.name == row.runner })
            else { throw BenchmarkError.invalid("Installed model has no selected benchmark runner.") }
            let launch = try await artifacts.launch(model: manifest, runner: runner, directory: paths.directory("models").appendingPathComponent(request.id), paths: paths)
            configuration.workerURL = launch.executable
            configuration.context = manifest.context
            configuration.appBuild =
                (try? BenchCommand.output("/usr/bin/git", ["-C", repository.path, "rev-parse", "HEAD"]))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "development"
        }
        let realForDeadline = realService
        let boundedProof =
            request.quickCheck
            ? Task {
                do { try await Task.sleep(for: .seconds(60)); await realForDeadline?.shutdown() } catch {}
            } : nil
        defer { boundedProof?.cancel() }
        let result: ResultsFile
        do { result = try await BenchmarkHarness(service: service, configuration: configuration).run(request) } catch { await realService?.shutdown(); throw error }
        await realService?.shutdown()
        if args.contains("--json") {
            print(String(decoding: try JSONEncoder().encode(result), as: UTF8.self))
        } else {
            print(
                "\(result.model): \(result.stub ? "stub" : request.quickCheck ? "quick check" : "measured") \(result.profile) suite saved to \(output.appendingPathComponent(result.model).appendingPathComponent(result.chip + ".json").path)."
            )
        }
    }
    static func option(_ flag: String, _ args: [String]) -> String? {
        guard let index = args.firstIndex(of: flag), index + 1 < args.count else { return nil }
        return args[index + 1]
    }
}
