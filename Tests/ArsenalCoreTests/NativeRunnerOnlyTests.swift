// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct NativeRunnerOnlyTests {
    @Test func pythonDeclarationsFailManifestAndSchemaAndCannotLaunch() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = repo.appendingPathComponent("Tests/Fixtures/DropIn/fixture-layers")
        var model = try ModelManifest.load(folder)
        model.runner[0].language = "python"
        #expect(
            !RunnerSelector.eligible(
                model.runner[0], environment: RunnerSelectionEnvironment(chip: "M5 Max", macOS: "26.0", installedWorkers: [model.runner[0].worker ?? "stub"]), fast: false))
        model.runner[0].python = "3.13"
        model.runner[0].entry = "run.py"
        model.runner[0].lock = "requirements.lock"
        #expect(model.problems(folder: folder, kinds: []).contains { $0.contains("Python model runners are unsupported") })
        var parser = TOMLParser(try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
        var document = try parser.parse()
        guard case .array(var runners) = document["runner"], case .table(var runner) = runners[0] else { Issue.record("Missing fixture runner"); return }
        runner["language"] = .string("python"); runner["python"] = .string("3.13")
        runners[0] = .table(runner); document["runner"] = .array(runners)
        let schema = try Data(contentsOf: repo.appendingPathComponent("Resources/schemas/model.schema.json"))
        #expect(try !ModelSchemaValidation.validate(document: document, schemaData: schema).isEmpty)
        #expect(throws: WorkerFailure.self) { try BundledWorkerArtifact().validateEngine(model: model, runner: model.runner[0]) }
        let paths = ArsenalPaths(support: repo.appendingPathComponent("tmp/unsupported-python-" + UUID().uuidString))
        defer { try? FileManager.default.removeItem(at: paths.support) }
        await #expect(throws: WorkerFailure.self) {
            _ = try await BundledWorkerArtifact().launch(model: model, runner: model.runner[0], directory: folder, paths: paths)
        }
        #expect(!FileManager.default.fileExists(atPath: paths.support.appendingPathComponent("envs").path))
    }
}
