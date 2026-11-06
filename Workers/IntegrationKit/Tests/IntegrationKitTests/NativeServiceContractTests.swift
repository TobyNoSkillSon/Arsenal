// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import Testing
import CryptoKit
import MLX
import ModelPrecision
@testable import ArsenalCore
@testable import IntegrationKit

private final class NativeTestBundleAnchor: NSObject {}

private struct NativeFixtureSource: CatalogSource {
    let files: [String: Data]
    func read(path: String, commit: String) async throws -> Data { try #require(files[path]) }
}
private struct NativeFixtureArtifact: WorkerArtifact {
    let executable: URL
    let metallib: String
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        WorkerLaunch(
            executable: executable, kind: model.kind,
            environment: ["ARSENAL_SUPPORT_DIR": paths.support.path, "ARSENAL_CACHE_DIR": paths.cache.path, "ARSENAL_MLX_METALLIB": metallib],
            log: paths.support.appendingPathComponent("fixture-worker.log"))
    }
}
@Suite(.serialized) struct NativeServiceContractTests {
    @Test func workerMetadataMatchesProductionArtifactPolicy() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("result.txt")
        try Data("abc".utf8).write(to: file)
        let metadata = try NativeMedia.file(file, root: root, mime: "text/plain", id: "result")
        #expect(NativeFiles.maxBytes == OperationArtifacts.maxArtifactBytes)
        #expect(metadata["bytes"] == 3)
        #expect(metadata["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        let operation = ModelOperation(name: "fixture", title: "fixture", inputSchema: [:], output: "file", stream: false)
        let validated = try OperationArtifacts.output(["files": [.object(metadata)]], operation: operation, directory: root)
        #expect(validated["files"]?.array?.first?["sha256"] == metadata["sha256"])
        let link = root.appendingPathComponent("linked.txt")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
        #expect(throws: WorkerFailure.self) { _ = try NativeMedia.file(link, root: root, mime: "text/plain") }
        #expect(throws: WorkerFailure.self) {
            _ = try OperationArtifacts.output(["files": [["id": "bad", "path": .string(link.path), "mime": "text/plain"]]], operation: operation, directory: root)
        }
    }
    @Test func terminalRepliesReleaseAdmissionBeforeBackToBackRequests() async throws {
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-wire-sequence-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = Bundle(for: NativeTestBundleAnchor.self).bundleURL.deletingLastPathComponent().appendingPathComponent("native-integration-fixture")
        let worker = WorkerProcess()
        do {
            try await worker.start(
                WorkerLaunch(
                    executable: fixture, kind: "images", environment: ["ARSENAL_MLX_METALLIB": try #require(ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"])],
                    log: root.appendingPathComponent("worker.log")))
            for _ in 0..<3000 {
                let status = try await worker.request(op: "status")
                #expect(status["loaded"] == false)
                do { _ = try await worker.request(op: "unsupported"); Issue.record("Unloaded operation succeeded") } catch let failure as WorkerFailure {
                    #expect(failure.code == "not_loaded")
                }
            }
            await worker.unload()
        } catch { await worker.unload(); throw error }
    }
    @Test(.heavy) func parserFixturesReachTheAdaptersServingParser() async throws {
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-parse-output-" + UUID().uuidString)
        let weights = root.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // The fixture tensors are written in this process; do not depend on an earlier suite having set the metallib.
        let metallib = try #require(ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"]); setTestMetallib(URL(fileURLWithPath: metallib))
        let writer = try TensorWriter(weights.appendingPathComponent("model.safetensors"))
        try writer.append("projection.weight", MLXArray.zeros([4, 64], dtype: .float16))
        try writer.append("projection.bias", MLXArray.zeros([4], dtype: .float16)); try writer.finish()
        try Data(#"{"model_type":"integration_fixture"}"#.utf8).write(to: weights.appendingPathComponent("config.json"))
        let fixture = Bundle(for: NativeTestBundleAnchor.self).bundleURL.deletingLastPathComponent().appendingPathComponent("native-integration-fixture")
        // Production launches scrub ARSENAL_NATIVE_LAB (RunnerEnvironment.scrub), and parse_output is a lab-only
        // operation, so this test spawns the fixture with the lab flag directly, as scripts/model-check.py does.
        var environment = RunnerEnvironment.scrub(["ARSENAL_MLX_METALLIB": metallib]); environment["ARSENAL_NATIVE_LAB"] = "1"
        let worker = try await LabFixtureWorker(executable: fixture, environment: environment, log: root.appendingPathComponent("worker.log"))
        do {
            do {
                _ = try await worker.request(op: "parse_output", params: ["operation": "project", "raw": "[1]"]); Issue.record("An unloaded worker parsed output")
            } catch let failure as WorkerFailure { #expect(failure.code == "invalid_request") }
            _ = try await worker.request(op: "load", params: ["model_dir": .string(root.path), "weights_dir": .string(weights.path), "dtype": "fp16"])
            let parsed = try await worker.request(op: "parse_output", params: ["operation": "project", "input": [:], "raw": "[0.5, 2]"])
            #expect(parsed["values"] == [0.5, 2])
            for (operation, raw, code) in [("project", "not json", NativeOutputParsingFailure.code), ("chat", "[1]", "invalid_request")] {
                do {
                    _ = try await worker.request(op: "parse_output", params: ["operation": .string(operation), "raw": .string(raw)]); Issue.record("Parsed \(raw)")
                } catch let failure as WorkerFailure { #expect(failure.code == code) }
            }
            await worker.unload()
        } catch { await worker.unload(); throw error }
    }
    @Test(.heavy, arguments: ["fp16", "bf16"]) func ordinaryGetLoadOperationAndFileValidation(servingHalf: String) async throws {
        let repo = URL(fileURLWithPath: String(#filePath).components(separatedBy: "/Workers/IntegrationKit/Tests/")[0])
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-native-service-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let metallib = try #require(ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"]); setTestMetallib(URL(fileURLWithPath: metallib))
        let fixture = Bundle(for: NativeTestBundleAnchor.self).bundleURL.deletingLastPathComponent().appendingPathComponent("native-integration-fixture")
        #expect(FileManager.default.isExecutableFile(atPath: fixture.path))
        let revision = String(repeating: "a", count: 40), source = root.appendingPathComponent("hf/test/fixture/resolve/" + revision)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let writer = try TensorWriter(source.appendingPathComponent("model.safetensors"))
        try writer.append("projection.weight", MLXArray.ones([4, 64], dtype: .bfloat16) * MLXArray(Float(0.125)).asType(.bfloat16))
        try writer.append("projection.bias", MLXArray([Float(0.25), 0.5, 0.75, 1]).asType(.bfloat16)); try writer.finish()
        try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
        let component = root.appendingPathComponent("hf/test/component/resolve/" + revision)
        try FileManager.default.createDirectory(at: component.appendingPathComponent("nested"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: component.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
        let componentWriter = try TensorWriter(component.appendingPathComponent("nested/component.safetensors"))
        try componentWriter.append("spatial.weight", MLXArray(Float(2))); try componentWriter.finish()
        try Data("{}".utf8).write(to: component.appendingPathComponent("tokenizer/config.json"))
        let server = try ChildProcess(
            executable: URL(fileURLWithPath: "/opt/homebrew/bin/python3"),
            arguments: [
                "-I", "-u", "-c",
                "import http.server,functools; s=http.server.HTTPServer(('127.0.0.1',0),functools.partial(http.server.SimpleHTTPRequestHandler,directory=\(String(reflecting: root.appendingPathComponent("hf").path)))); print(s.server_port,flush=True); s.serve_forever()"
            ],
            environment: RunnerEnvironment.scrub(), log: root.appendingPathComponent("server.log"))
        defer { server.signal(SIGKILL); _ = server.awaitExit(); server.closeInput(); server.finishOutput() }
        let portNumber = try await nativeFixturePort(server)
        var parser = TOMLParser(try String(contentsOf: repo.appendingPathComponent("models/birefnet/model.toml"), encoding: .utf8)); var table = try parser.parse()
        table["id"] = .string("native-fixture"); table["name"] = .string("Synthetic native fixture"); table["menu"] = nil; table["tools"] = .array([.string("operation")])
        table["architecture"] = .string("integration_fixture")
        table["precision"] = .table(["dtype": .string(servingHalf), "reason": .string("Synthetic BF16 exception contract fixture."), "evidence": .string("log/001-fixture.md")])
        table["weights"] = .table([
            "repo": .string("test/fixture"), "revision": .string(revision), "dtype": .string("bf16"), "files": .array([.string("model.safetensors"), .string("config.json")]),
            "sha256": .table(
                try Dictionary(uniqueKeysWithValues: ["model.safetensors", "config.json"].map { ($0, .string(try FileDigest.sha256(source.appendingPathComponent($0)))) })),
            "bytes": .table(
                try Dictionary(uniqueKeysWithValues: ["model.safetensors", "config.json"].map { ($0, .integer(try FileDigest.bytes(source.appendingPathComponent($0)))) }))
        ])
        if case .array(var runners) = table["runner"], case .table(var runner) = runners[0] {
            // A [precision] block belongs to a normalized native family; the fixture executable stands in for that family's worker.
            runner["worker"] = .string("geometry"); runners[0] = .table(runner); table["runner"] = .array(runners)
        }
        let recipe = servingHalf == "bf16" ? "fixture-components-bf16" : "fixture-components"
        table["derive"] = .table([servingHalf: .string("cast:" + recipe), "int8": .string(recipe), "int4": .string(recipe)])
        let componentFiles = ["nested/component.safetensors", "tokenizer/config.json"]
        table["asset"] = .array([
            .table([
                "id": .string("codec"), "repo": .string("test/component"), "repo_type": .string("model"), "revision": .string(revision), "licence": .string("MIT"),
                "files": .array(componentFiles.map(TOMLValue.string)),
                "sha256": .table(try Dictionary(uniqueKeysWithValues: componentFiles.map { ($0, .string(try FileDigest.sha256(component.appendingPathComponent($0)))) })),
                "bytes": .table(try Dictionary(uniqueKeysWithValues: componentFiles.map { ($0, .integer(try FileDigest.bytes(component.appendingPathComponent($0)))) }))
            ])
        ])
        func op(_ name: String, _ output: String) -> TOMLValue {
            .table([
                "name": .string(name), "title": .string(name), "input": .string("ops/" + name + ".input.json"), "output_schema": .string("ops/" + name + ".output.json"),
                "output": .string(output), "stream": .bool(false)
            ])
        }
        guard case .table(var imageOp) = op("project_image", "json") else { throw WireError.malformedFrame }
        imageOp["prepare_image"] = .table(["model": .string("cutout-fixture"), "operation": .string("remove_background"), "file_id": .string("foreground")])
        table["op"] = .array([op("project", "json"), op("upscale_image", "file"), .table(imageOp), op("remove_background", "file")])
        var files: [String: Data] = [
            "model.toml": Data(TOMLWriter.encode(table).utf8), "README.md": Data("Synthetic fixture".utf8),
            "log/001-fixture.md": Data("Synthetic half-precision graph, no trained weights.".utf8), "gate/gate.toml": Data("schema=1".utf8),
            "bench/bench.toml": Data("schema=1".utf8),
            "ops/project.input.json": Data(#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false}"#.utf8),
            "ops/project.output.json": Data(
                #"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"values":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}},"required":["values"],"additionalProperties":false}"#
                    .utf8),
            "ops/upscale_image.input.json": Data(
                #"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"omit_id":{"type":"boolean"}},"additionalProperties":false}"#.utf8),
            "ops/upscale_image.output.json": try Data(contentsOf: repo.appendingPathComponent("Resources/kinds/restoration/ops/upscale_image.output.json"))
        ]
        files["ops/project_image.input.json"] = try Data(contentsOf: repo.appendingPathComponent("Resources/kinds/geometry/ops/image_to_3d.input.json"))
        files["ops/project_image.output.json"] = Data(
            #"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"values":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"prepared_path":{"type":"string"}},"required":["values","prepared_path"],"additionalProperties":false}"#
                .utf8)
        files["ops/remove_background.input.json"] = try Data(contentsOf: repo.appendingPathComponent("models/birefnet/ops/remove_background.input.json"))
        files["ops/remove_background.output.json"] = files["ops/upscale_image.output.json"]
        var dependencyTable = table; dependencyTable["id"] = .string("cutout-fixture"); dependencyTable["op"] = .array([op("remove_background", "file")])
        var dependencyFiles = files; dependencyFiles["model.toml"] = Data(TOMLWriter.encode(dependencyTable).utf8)
        let dependencyFolder = root.appendingPathComponent("cutout-fixture")
        for (path, data) in dependencyFiles {
            let url = dependencyFolder.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true); try data.write(to: url)
        }
        let dependencyManifest = try ModelManifest.load(dependencyFolder)
        let folder = root.appendingPathComponent("native-fixture")
        for (path, data) in files {
            let url = folder.appendingPathComponent(path); try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
        }
        let manifest = try ModelManifest.load(folder)
        let catalog = CatalogDocument(
            sequence: 1, sourceCommit: revision,
            models: [
                CatalogModel(manifest: manifest, files: files.map { CatalogFile(path: $0.key, sha256: FileDigest.sha256($0.value), bytes: Int64($0.value.count)) }),
                CatalogModel(
                    manifest: dependencyManifest, files: dependencyFiles.map { CatalogFile(path: $0.key, sha256: FileDigest.sha256($0.value), bytes: Int64($0.value.count)) })
            ])
        var sourceFiles = Dictionary(uniqueKeysWithValues: files.map { ("models/native-fixture/" + $0.key, $0.value) })
        sourceFiles.merge(Dictionary(uniqueKeysWithValues: dependencyFiles.map { ("models/cutout-fixture/" + $0.key, $0.value) }), uniquingKeysWith: { a, _ in a })
        let data = try catalog.encoded(), key = Curve25519.Signing.PrivateKey()
        let service = try await RealModelsService.open(
            catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation,
            source: NativeFixtureSource(files: sourceFiles), kinds: KindSpec.defaults,
            paths: ArsenalPaths(support: root.appendingPathComponent("support")), artifacts: NativeFixtureArtifact(executable: fixture, metallib: metallib),
            client: HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:\(portNumber)")!))
        do {
            try await service.add(id: manifest.id)
            for precision in [servingHalf, "int8", "int4"] {
                try await service.get(id: manifest.id, precision: precision)
                let verifier = ModelFetcher(support: root.appendingPathComponent("support"))
                let snapshot = try #require(try await verifier.store.verifyInstalledSnapshot(model: manifest, precision: precision))
                #expect(snapshot.derived?.assetSources?["codec"]?.repo == "test/component")
                var changed = manifest; changed.asset?[0].revision = String(repeating: "b", count: 40)
                #expect(try await verifier.updateRequired(model: changed, precision: precision))
                do { _ = try await verifier.store.verifyInstalledSnapshot(model: changed, precision: precision); Issue.record("Stale component pins accepted") } catch {}
                let rawHash = try FileDigest.sha256(source.appendingPathComponent("model.safetensors"))
                if servingHalf == "fp16" {
                    #expect(try await verifier.store.hasBlob(sha256: rawHash, bytes: FileDigest.bytes(source.appendingPathComponent("model.safetensors"))) == false)
                }
                try await service.load(id: manifest.id, precision: precision, fast: nil)
                let result = try await service.run(id: manifest.id, op: "project", params: [:])
                #expect(result["values"]?.array?.compactMap(\.number) == [8.25, 8.5, 8.75, 9])
                let artifact = try await service.run(id: manifest.id, op: "upscale_image", params: [:])
                #expect(artifact["files"]?.array?.first?["id"] == "output")
                do { _ = try await service.run(id: manifest.id, op: "upscale_image", params: ["omit_id": true]); Issue.record("Service accepted a missing file ID") } catch let
                    failure as WorkerFailure
                { #expect(failure.code == "invalid_output") }
                try await service.unload(id: manifest.id)
            }
            let inbox = root.appendingPathComponent("support/inbox")
            try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
            let image = inbox.appendingPathComponent("input.png")
            try Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJ1sAAAAASUVORK5CYII=")!.write(to: image)
            let imageInput: JSONObject = ["image": ["path": .string(image.path)]]
            for added in [false, true] {
                if added { try await service.add(id: dependencyManifest.id) }
                do { _ = try await service.run(id: manifest.id, op: "project_image", params: imageInput); Issue.record("Missing dependency accepted") } catch let failure
                    as WorkerFailure
                { #expect(failure.code == "dependency_not_installed") }
            }
            try await service.get(id: dependencyManifest.id, precision: servingHalf)
            let result = try await service.run(id: manifest.id, op: "project_image", params: imageInput)
            #expect(result["values"]?.array?.compactMap(\.number) == [8.25, 8.5, 8.75, 9])
            let preparedPath = try #require(result["prepared_path"]?.string)
            #expect(preparedPath != image.path && !FileManager.default.fileExists(atPath: preparedPath))
            for name in ["opaque", "wrong-id"] {
                let invalidImage = inbox.appendingPathComponent(name + ".png"); try FileManager.default.copyItem(at: image, to: invalidImage)
                do {
                    _ = try await service.run(id: manifest.id, op: "project_image", params: ["image": ["path": .string(invalidImage.path)]])
                    Issue.record("Invalid dependency cutout accepted")
                } catch let failure as WorkerFailure { #expect(failure.code == "invalid_dependency_output") }
            }
            await service.shutdown()
        } catch { await service.shutdown(); throw error }
    }
}

/// A sequential wire client for a lab-flagged fixture worker, outside WorkerProcess's production environment scrub.
private final class LabFixtureWorker: @unchecked Sendable {
    private let child: ChildProcess
    private var nextID = 1
    init(executable: URL, environment: [String: String], log: URL) async throws {
        child = try ChildProcess(executable: executable, arguments: [], environment: environment, log: log)
        guard case .json(let hello)? = try await child.readPacket()?.frame, hello["pid"]?.number == Double(child.pid) else {
            await unload(); throw WorkerFailure(code: "handshake", message: "Fixture worker sent no hello.")
        }
    }
    func request(op: String, params: JSONObject = [:]) async throws -> JSONObject {
        let id = nextID; nextID += 1
        try child.write(.json(WorkerRequest(id: id, op: op, params: params).json), deadline: Date().addingTimeInterval(10))
        while let packet = try await child.readPacket() {
            guard case .json(let object) = packet.frame, object["id"]?.number == Double(id), object["event"] == nil else { continue }
            if object["ok"]?.bool == true { return object }
            let details = object["error"]?.object ?? [:]
            throw WorkerFailure(code: details["code"]?.string ?? "worker", message: details["message"]?.string ?? "Worker request failed.", details: details)
        }
        throw WorkerFailure(code: "worker_exit", message: "Fixture worker exited.")
    }
    func unload() async {
        child.closeInput()
        let child = self.child
        let killer = Task.detached {
            try? await Task.sleep(for: .seconds(5))
            if !Task.isCancelled { child.signal(SIGKILL) }
        }
        _ = await child.awaitExitAsync(); killer.cancel(); child.finishOutput()
    }
}
private func nativeFixturePort(_ server: ChildProcess) async throws -> Int {
    var port = Data()
    while let byte = try await server.readOutputAsync(count: 1), byte != Data([10]), !byte.isEmpty { port += byte }
    return try #require(Int(String(decoding: port, as: UTF8.self)))
}
