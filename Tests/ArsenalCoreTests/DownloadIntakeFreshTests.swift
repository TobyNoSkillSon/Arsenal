// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct DownloadIntakeFreshTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @Test(arguments: ["von-1.1", "von-1.2", "re-use"], [false, true])
    func finalBF16ManifestTrialsPinnedSourceAsFP16BeforeServing(id: String, fp16Passes: Bool) async throws {
        let root = repo.appendingPathComponent("tmp/intake-fixture-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = try makeModel(id, root: root), source = root.appendingPathComponent("source"), lab = root.appendingPathComponent("support")
        let model = try ModelManifest.load(folder)
        #expect(model.basePrecision == "bf16" && model.derive?["fp16"] == nil)
        let helpers = try makeHelpers(root: root, model: model, fp16Passes: fp16Passes)
        do {
            // No --reference-run or --format-qualification: exercise the ordinary fresh intake path.
            try await ModelIntakeCommand.execute([
                "--model", folder.path, "--source", source.path, "--helpers", helpers.path, "--support", lab.path, "--go"
            ])
            Issue.record("Fixture must stop before serving figures")
        } catch {
            #expect(error.localizedDescription.contains(fp16Passes ? "FP16 recipe required" : "Intake fixture reached selected BF16 serving baseline"))
        }
        let runs = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("gate/evidence/intake"), includingPropertiesForKeys: nil)
        let evidence = try #require(runs.first)
        #expect(runs.count == 1)
        let fp16 = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("fp16-snapshot.json")))
        let bf16 = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("bf16-snapshot.json")))
        #expect(fp16["precision"] == "fp16" && bf16["precision"] == "bf16")
        #expect(fp16["derived"]?["sourceDtype"] == "fp32")
        #expect(fp16["derived"]?["sourceHashes"] == bf16["derived"]?["sourceHashes"])
        let coverage = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("fp16-component-coverage.json")))
        #expect(coverage["dtype"] == "F16" && coverage["source_is_original"] == true)
        #expect(coverage["recipe"]?.string == (id == "re-use" ? "cast:re-use" : "cast:von-half-2"))
        let format = evidence.appendingPathComponent("format-qualification")
        let candidate = try IntakeTensor.read(format.appendingPathComponent("fp16-candidate/outputs.safetensors"))
        let reference = try IntakeTensor.read(format.appendingPathComponent("bf16-reference/outputs.safetensors"))
        #expect(candidate.count == (id == "re-use" ? 63 : 60))
        #expect(candidate.values.allSatisfy { $0.dtype == "F16" })
        #expect(reference.values.allSatisfy { $0.dtype == "BF16" })
        let gate = try WireJSON.decode(Data(contentsOf: format.appendingPathComponent("format-gate.json")))
        #expect(gate["component_gate_passed"]?.bool == fp16Passes)
        #expect(
            gate
                == (try IntakeComponentGate.evaluate(
                    reference: format.appendingPathComponent("bf16-reference"), candidate: format.appendingPathComponent("fp16-candidate"),
                    bounds: format.appendingPathComponent("bounds.json"))))
        let baseline = lab.appendingPathComponent("fixture-baseline-load.json")
        #expect(FileManager.default.fileExists(atPath: baseline.path) == !fp16Passes)
        if !fp16Passes {
            #expect(gate["components"]?.object?.values.allSatisfy { $0["cosine"] == -1 } == true)
            let loaded = try WireJSON.decode(Data(contentsOf: baseline))
            #expect(loaded["actual_dtype"] == "BF16" && loaded["requested"]?["dtype"] == "bf16")
            let reused = root.appendingPathComponent("accepted")
            try FileManager.default.createDirectory(at: reused, withIntermediateDirectories: true)
            try ModelIntakeCommand.acceptDownloadedFormat(model: model, from: format, evidence: reused)
            let second = root.appendingPathComponent("accepted-again")
            try FileManager.default.createDirectory(at: second, withIntermediateDirectories: true)
            try ModelIntakeCommand.acceptDownloadedFormat(model: model, from: reused.appendingPathComponent("format-qualification"), evidence: second)
            #expect(
                try Data(contentsOf: second.appendingPathComponent("format-qualification/format-gate-source.json"))
                    == Data(contentsOf: format.appendingPathComponent("format-gate.json")))
            #expect(
                try Data(contentsOf: format.appendingPathComponent("format-gate.json")) == Data(contentsOf: reused.appendingPathComponent("format-qualification/format-gate.json")))
        }
        let policy = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("policy.json")))
        #expect(policy["run"]?.string == evidence.lastPathComponent)
        #expect(policy["reference_scope"]?.string?.contains(id == "re-use" ? "Selected BF16" : "Pinned BF16 served " + id) == true)
        if id != "re-use" {
            #expect(policy["reference_scope"]?.string?.contains("GPU BF16 nearest/truncation") == true)
            #expect(policy["reference_scope"]?.string?.contains("separate FP16 candidate; BF16 serving requires its frozen rejection") == true)
        }
        #expect(!FileManager.default.fileExists(atPath: lab.appendingPathComponent("weights").path))
    }

    private func makeHelpers(root: URL, model: ModelManifest, fp16Passes: Bool) throws -> URL {
        let helpers = root.appendingPathComponent("helpers")
        try FileManager.default.createDirectory(at: helpers, withIntermediateDirectories: true)
        let binary = stubLaunch().executable.path.replacingOccurrences(of: "'", with: "'\"'\"'")
        let script = "#!/bin/sh\nexec '" + binary + "' --kind " + model.kind + " --intake-fixture" + (fp16Passes ? " --fp16-passes" : "") + "\n"
        let helper = helpers.appendingPathComponent("arsenal-worker-" + model.runners[0].worker!)
        try Data(script.utf8).write(to: helper)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helper.path)
        return helpers
    }

    private func makeModel(_ id: String, root: URL) throws -> URL {
        let original = shippedOrRetiredFolder(repo, id), folder = root.appendingPathComponent("models/" + id)
        let model = try ModelManifest.load(original), source = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        var parser = TOMLParser(try String(contentsOf: original.appendingPathComponent("model.toml"), encoding: .utf8))
        var table = try parser.parse()
        guard case .table(var weights) = table["weights"] else { throw ManifestError.invalid("Fixture needs source weights") }
        var hashes: [String: TOMLValue] = [:], sizes: [String: TOMLValue] = [:]
        for name in model.weights.files {
            let data = Data((name == "config.json" ? "{\"num_hidden_layers\":28}" : "fixture pinned source").utf8)
            try data.write(to: source.appendingPathComponent(name))
            hashes[name] = .string(FileDigest.sha256(data)); sizes[name] = .integer(Int64(data.count))
        }
        weights["sha256"] = .table(hashes); weights["bytes"] = .table(sizes); table["weights"] = .table(weights)
        for name in ["README.md", model.gate.recipe, model.bench.recipe, "log/fixture.md"] {
            let target = folder.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("Fixture".utf8).write(to: target)
        }
        for name in model.operations.flatMap({ [$0.input] + [$0.outputSchema].compactMap { $0 } }) where !name.hasPrefix("kinds/") {
            let target = folder.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: original.appendingPathComponent(name), to: target)
        }
        try Data(TOMLWriter.encode(table).utf8).write(to: folder.appendingPathComponent("model.toml"))
        try FileManager.default.copyItem(at: repo.appendingPathComponent("Resources/kinds"), to: root.appendingPathComponent("kinds"))
        return folder
    }
}
