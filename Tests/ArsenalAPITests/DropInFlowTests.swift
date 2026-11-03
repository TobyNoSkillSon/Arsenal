// SPDX-License-Identifier: MIT
import Foundation
import Testing
import CryptoKit
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

private struct DropInSource: CatalogSource {
    func read(path: String, commit: String) async throws -> Data { throw ManifestError.invalid("empty fixture catalog") }
}
// Transport fixture admission must not depend on the busy Mac's instantaneous free pages.
private func dropInMemory(_ root: URL) throws -> MemoryProbe {
    let file = root.appendingPathComponent("memory.json")
    try WireJSON.encode(["available_mb": 128000]).write(to: file)
    return MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": file.path], totalMB: 128000)
}
struct DropInFlowTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @Test func fixtureRefreshApproveGetHTTPMCPIdleSwapAndRollback() async throws {
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-dropin-flow-" + UUID().uuidString)
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        try paths.create()
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = paths.support.appendingPathComponent("local-models/fixture-layers")
        try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("Tests/Fixtures/DropIn/fixture-layers"), to: folder)
        // Familiar operation names must remain generic outside their legacy kind adapters.
        let kindFile = folder.appendingPathComponent("kinds/local.layers.toml")
        let jsonOp =
            "\n[[op]]\nname = \"generate\"\ntitle = \"Generate JSON\"\ninput = \"ops/generate.input.json\"\noutput_schema = \"ops/generate.output.json\"\noutput = \"json\"\nstream = false\ndefault = true\n"
        try (String(contentsOf: kindFile, encoding: .utf8) + jsonOp).write(to: kindFile, atomically: true, encoding: .utf8)
        let jsonSchema: JSONObject = [
            "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "required": ["value"], "additionalProperties": false,
            "properties": ["value": ["type": "integer"]]
        ]
        try WireJSON.encode(jsonSchema).write(to: folder.appendingPathComponent("kinds/local.layers/ops/generate.input.json"))
        var outputJSONSchema = jsonSchema
        var outputProperties = outputJSONSchema["properties"]!.object!
        outputProperties["timing"] = ["type": "object"]; outputJSONSchema["properties"] = .object(outputProperties)
        try WireJSON.encode(outputJSONSchema).write(to: folder.appendingPathComponent("kinds/local.layers/ops/generate.output.json"))
        let manifest = folder.appendingPathComponent("model.toml")
        let swiftManifest = try String(contentsOf: manifest, encoding: .utf8)
        try swiftManifest.replacingOccurrences(of: "worker = \"stub\"", with: "entry = \"runners/swift/bin/fixture-layers\"")
            .write(to: manifest, atomically: true, encoding: .utf8)
        try buildSwift(folder, scratch: root.appendingPathComponent("swift-build"))
        let donor = root.appendingPathComponent("donor/fixture/layers")
        try FileManager.default.createDirectory(at: donor, withIntermediateDirectories: true)
        try Data([12, 34, 56]).write(to: donor.appendingPathComponent("base.bin"))
        var config = ArsenalConfiguration(); config.residency.manualIdleMinutes = 1; config.get = ["localSources": [.string(root.appendingPathComponent("donor").path)]]
        try config.write(paths.support.appendingPathComponent("config.json"))
        let document = CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: [])
        let data = try document.encoded(), key = Curve25519.Signing.PrivateKey()
        let service = try await RealModelsService.open(
            catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation,
            source: DropInSource(), kinds: [], paths: paths, artifacts: BundledWorkerArtifact(),
            memoryProbe: try dropInMemory(paths.support))
        let server = ArsenalAPIServer(service: service, supportDirectory: paths.support, port: 0)
        let port = try await server.start()
        defer { server.stop() }
        do {
            let router = APIRouter(service: service, support: paths.support); await router.setPort(port)
            func settings(_ body: JSONObject) async throws -> CallTool.Result { try await router.mcpSettings(body) }
            try await service.refresh()
            let before = try #require(await service.catalog(query: .init()).first?.row)
            #expect(before.kind == "local.layers" && before.tools == ["operation"])
            _ = try await settings(["action": "approve_code", "model": "fixture-layers", "content_digest": .string(before.contentDigest!)])
            try await service.add(id: "fixture-layers")
            let plan = try await service.getPlan(id: "fixture-layers", dtype: "bf16"); #expect(plan.totalDownloadBytes == 0)
            try await service.get(id: "fixture-layers", precision: "bf16", approvedPlan: plan)
            try await service.load(id: "fixture-layers")
            #expect(try await service.run(id: "fixture-layers", op: "generate", params: ["value": 3])["value"] == 3)
            let operations = await service.operations(id: "fixture-layers"), operation = try #require(operations.first { $0.name == "separate" })
            #expect(operation.outputSchema != nil && operation.policy?.timeoutS == 10 && operation.annotations?["primary"] == true)
            func http(_ body: JSONObject, accept: String = "application/json") async throws -> (Data, Int) {
                var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/fixture-layers/ops/separate")!)
                request.httpMethod = "POST"; request.timeoutInterval = 15; request.httpBody = try WireJSON.encode(body)
                request.setValue("application/json", forHTTPHeaderField: "Content-Type"); request.setValue(accept, forHTTPHeaderField: "Accept")
                let (data, response) = try await URLSession.shared.data(for: request)
                return (data, (response as? HTTPURLResponse)?.statusCode ?? 0)
            }
            let reply = try await http(["layers": 4]), result = try WireJSON.decode(reply.0)
            #expect(reply.1 == 200, "\(String(decoding: reply.0, as: UTF8.self))")
            #expect(result["files"]?.array?.count == 4 && result["metadata"]?["revision"] == 1)
            let file = try #require(result["files"]?.array?.first)
            #expect(file["depth_min"] == 0 && file["label"] == "Layer 0" && file["bytes"]?.number != nil)
            let (image, response) = try await URLSession.shared.data(from: URL(string: file["url"]!.string!)!)
            #expect((response as? HTTPURLResponse)?.statusCode == 200 && FileDigest.sha256(image) == file["sha256"]?.string)
            let entry = try await router.mcpEntry("fixture-layers")
            let job = MCPJob(kind: .run, model: "fixture-layers", name: entry.name, operation: operation.name, dtype: "fp16", requester: "dropin-test", environment: .live)
            let mcp = try await router.mcpRun(
                entry: entry, operation: operation, input: ["layers": 4], requester: "dropin-test", play: false, job: job, directory: try await router.newJob())
            #expect(mcp.text.hasPrefix("Created 4 files with fixture-layers") || mcp.text.hasPrefix("Created 4 images with fixture-layers"), "\(mcp.text)")
            let structured = mcp.json
            #expect(structured["files"]?.array?.count == 4)
            #expect(structured["files"]?.array?.first?["depth_max"] == 0.25)
            #expect(try await router.readMCPResource(structured["files"]!.array!.first!["uri"]!.string!).uri.hasPrefix("arsenal://outputs/"))
            let sse = try await http(["layers": 2], accept: "text/event-stream")
            #expect(sse.1 == 200 && String(decoding: sse.0, as: UTF8.self).contains("event: result"))
            for bad in ["duplicate", "missing", "mime"] { #expect(try await http(["layers": 4, "bad": .string(bad)]).1 == 502) }
            #expect(try await http(["layers": 0]).1 == 422)
            let originalPID = try #require(result["metadata"]?["pid"]?.number)
            try writeEvidence(folder); try await service.refresh()
            let evidenceDigest = try #require(await service.catalog(query: .init()).first?.row.contentDigest)
            _ = try await settings(["action": "approve_code", "model": "fixture-layers", "content_digest": .string(evidenceDigest)])
            let evidenceReply = try await http(["layers": 1])
            #expect(try WireJSON.decode(evidenceReply.0)["metadata"]?["pid"]?.number == originalPID)
            #expect(await service.rows().first?.revision?.active == before.contentDigest)
            let script = folder.appendingPathComponent("runners/swift/Sources/FixtureLayers/main.swift")
            // Active request keeps revision 1's schema while an approved revision is staged.
            try String(contentsOf: script, encoding: .utf8).replacingOccurrences(of: "let revision = 1", with: "let revision = 2").write(
                to: script, atomically: true, encoding: .utf8)
            let outputSchema = folder.appendingPathComponent("kinds/local.layers/ops/separate.output.json")
            var schema = try WireJSON.decode(Data(contentsOf: outputSchema))
            var properties = schema["properties"]!.object!, metadata = properties["metadata"]!.object!, metadataProperties = metadata["properties"]!.object!
            metadataProperties["revision"] = ["const": 2]; metadata["properties"] = .object(metadataProperties); properties["metadata"] = .object(metadata)
            schema["properties"] = .object(properties)
            try WireJSON.encode(schema).write(to: outputSchema)
            try buildSwift(folder, scratch: root.appendingPathComponent("swift-build"))
            let active = Task { try await service.run(id: "fixture-layers", op: "separate", params: ["layers": 4, "delay": 0.3]) }
            try await Task.sleep(for: .milliseconds(50))
            try await service.refresh()
            await #expect(throws: WorkerFailure.self) { try await service.approveLocalCode(id: "fixture-layers", contentDigest: before.contentDigest!) }
            // The owner's yes to the current digest alone would now approve it (DayPass26CoreTests); this flow
            // approves the next revision instead, after its weights change.
            // A changed weight pin cannot swap until Get installs the candidate.
            let newWeights = Data([65, 43, 21])
            try newWeights.write(to: donor.appendingPathComponent("base.bin"))
            let changedManifest = try String(contentsOf: manifest, encoding: .utf8)
                .replacingOccurrences(of: String(repeating: "a", count: 40), with: String(repeating: "b", count: 40))
                .replacingOccurrences(of: "651f3a7ba220e024b0d76f3df555f406db4912ae05d63f13af6b02d7dcc0fa53", with: FileDigest.sha256(newWeights))
            try changedManifest.write(to: manifest, atomically: true, encoding: .utf8)
            try writeEvidence(folder)
            try await service.refresh()
            #expect(await service.rows().first?.metrics["seconds"] == nil)
            #expect(await service.rows().first?.revision?.pending == nil)
            let candidate = try #require(await service.catalog(query: .init()).first?.row.contentDigest)
            _ = try await settings(["action": "approve_code", "model": "fixture-layers", "content_digest": .string(candidate)])
            #expect(try await active.value["metadata"]?["revision"] == 1)
            #expect(await service.rows().first?.revision?.active == before.contentDigest)
            #expect(await service.rejectedRevisionDigests["fixture-layers"] == nil)
            let revisionPlan = try await service.getPlan(id: "fixture-layers", dtype: "bf16")
            try await service.get(id: "fixture-layers", precision: "bf16", approvedPlan: revisionPlan)
            let arrival = Task { try await service.run(id: "fixture-layers", op: "separate", params: ["layers": 4]) }
            #expect(try await arrival.value["metadata"]?["revision"] == 2)
            try await waitSwap(service, active: candidate)
            #expect(try await service.run(id: "fixture-layers", op: "separate", params: ["layers": 4])["metadata"]?["revision"] == 2)
            #expect(await service.rows().first?.revision?.previous == before.contentDigest)
            #expect(await service.timers["fixture-layers"] != nil)
            _ = try await settings(["action": "rollback", "model": "fixture-layers"])
            #expect(try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json")).rejectedRevisions["fixture-layers"] == candidate)
            #expect(try await service.run(id: "fixture-layers", op: "separate", params: ["layers": 4])["metadata"]?["revision"] == 1)
            try await service.unload(id: "fixture-layers")
            #expect(try await service.run(id: "fixture-layers", op: "separate", params: ["layers": 4])["metadata"]?["revision"] == 1)
            // Failed approved load restores revision 1; same digest is not retried automatically.
            try String(contentsOf: script, encoding: .utf8).replacingOccurrences(of: "let badLoad = false", with: "let badLoad = true").write(
                to: script, atomically: true, encoding: .utf8)
            try buildSwift(folder, scratch: root.appendingPathComponent("swift-build"))
            try writeEvidence(folder); try await service.refresh()
            let broken = try #require(await service.catalog(query: .init()).first?.row.contentDigest)
            _ = try await settings(["action": "approve_code", "model": "fixture-layers", "content_digest": .string(broken)])
            for _ in 0..<100 {
                if await service.rows().first?.revision?.error != nil { break }; try await Task.sleep(for: .milliseconds(20))
            }
            #expect(await service.rows().first?.revision?.error != nil)
            #expect(try await service.run(id: "fixture-layers", op: "separate", params: ["layers": 4])["metadata"]?["revision"] == 1)
            // Explicit Unload during a real successful candidate load must stay unloaded.
            try String(contentsOf: script, encoding: .utf8).replacingOccurrences(of: "let badLoad = true", with: "let badLoad = false").write(
                to: script, atomically: true, encoding: .utf8)
            try buildSwift(folder, scratch: root.appendingPathComponent("swift-build"))
            try writeEvidence(folder); try await service.refresh()
            let repaired = try #require(await service.catalog(query: .init()).first?.row.contentDigest)
            _ = try await settings(["action": "approve_code", "model": "fixture-layers", "content_digest": .string(repaired)])
            try await Task.sleep(for: .milliseconds(30))
            try await service.unload(id: "fixture-layers")
            #expect(await service.residents["fixture-layers"] == nil)
            #expect(await service.rejectedRevisionDigests["fixture-layers"] == nil)
            #expect(try await router.readMCPResource("arsenal://integration/SKILL.md").uri == "arsenal://integration/SKILL.md")
            try await service.load(id: "fixture-layers")
            try await service.rollbackRevision(id: "fixture-layers")
            await service.shutdown()
            #expect(await service.supervisor.residentIDs().isEmpty)
            try await assertRestartRecovery(paths, data: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, rejected: repaired)
        } catch { await service.shutdown(); throw error }
    }
    private func assertRestartRecovery(_ paths: ArsenalPaths, data: Data, signature: Data, publicKey: Data, rejected: String) async throws {
        let restarted = try await RealModelsService.open(
            catalogData: data, signature: signature, publicKey: publicKey,
            source: DropInSource(), kinds: [], paths: paths, artifacts: BundledWorkerArtifact(),
            memoryProbe: try dropInMemory(paths.support))
        do {
            #expect(await restarted.rejectedRevisionDigests["fixture-layers"] == rejected)
            #expect(try await restarted.run(id: "fixture-layers", op: "separate", params: ["layers": 4])["metadata"]?["revision"] == 1)
            await restarted.shutdown()
        } catch { await restarted.shutdown(); throw error }
    }
    private func buildSwift(_ folder: URL, scratch: URL) throws {
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [folder.appendingPathComponent("runners/swift/build.sh").path]
        process.environment = ProcessInfo.processInfo.environment.merging(["ARSENAL_FIXTURE_BUILD": scratch.path]) { _, new in new }
        try process.run(); process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        guard process.terminationStatus == 0 else { throw ManifestError.invalid("Swift fixture build failed") }
        let files = try SafePath.files(in: folder).filter { $0.hasPrefix("runners/swift/") && $0 != "runners/swift/build-receipt.json" && !$0.hasSuffix("build.sh") }
        let hashes = try Dictionary(uniqueKeysWithValues: files.map { ($0, JSONValue.string(try FileDigest.sha256(SafePath.resolve($0, inside: folder)))) })
        try WireJSON.encode(["command": "fixture build.sh, release arm64", "toolchain": "CLT Swift", "sha256": .object(hashes)]).write(
            to: folder.appendingPathComponent("runners/swift/build-receipt.json"))
    }
    private func waitSwap(_ service: RealModelsService, active: String) async throws {
        for _ in 0..<100 {
            if await service.rows().first?.revision?.active == active { return }; try await Task.sleep(for: .milliseconds(20))
        }
        Issue.record("Automatic idle revision swap did not finish")
    }
    private func writeEvidence(_ folder: URL) throws {
        let local = try LocalModelFolders.discover(support: folder.deletingLastPathComponent().deletingLastPathComponent(), signedIDs: [], kinds: []).first!
        let directory = folder.appendingPathComponent("results/gate")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("retained deterministic fixture output evidence".utf8).write(to: directory.appendingPathComponent("actual.txt"))
        var result = ResultsFile(
            model: "fixture-layers", kind: "local.layers", chip: HostChip.current, osBuild: "test", appBuild: "test", runner: "swift",
            workerDigest: String(repeating: "a", count: 64), precision: "bf16", mode: "exact", suite: .init(id: "fixture", sha256: String(repeating: "a", count: 64)),
            profile: "quick", stub: false, gatePassed: true, conditions: .init(locked: false, idle: [], run: []), metrics: ["seconds": 0.01], cases: [])
        result.checks = .init(
            protocolStatus: .pass, parity: .pass, task: .pass, reference: ["identity": "deterministic-fixture-reference", "independent": true],
            implementationFingerprint: try ImplementationEvidence.fingerprint(local), folderDigest: local.contentDigest, fixtureIdentity: ImplementationEvidence.fixtures(local),
            cases: ["outputs": 4], evidence: ["results/gate/actual.txt"])
        result.checks?.speed = .pass
        result.checks?.evidenceHashes = ["results/gate/actual.txt": try FileDigest.sha256(directory.appendingPathComponent("actual.txt"))]
        try JSONEncoder().encode(result).write(to: directory.appendingPathComponent("result.json"))
    }
}
