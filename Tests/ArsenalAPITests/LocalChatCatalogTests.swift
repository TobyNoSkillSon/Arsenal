// SPDX-License-Identifier: MIT
import Foundation
import Testing
import CryptoKit
@testable import ArsenalCore
@testable import ArsenalAPI

private struct ChatBoundarySource: CatalogSource {
    let files: [String: Data]
    func read(path: String, commit: String) async throws -> Data {
        guard let data = files[path] else { throw ManifestError.invalid("Missing fixture catalog file") }
        return data
    }
}
private struct ChatBoundaryWorker: WorkerArtifact {
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        let binary = try #require(ProcessInfo.processInfo.environment["ARSENAL_TEST_STUB_WORKER"])
        return WorkerLaunch(
            executable: URL(fileURLWithPath: binary), arguments: ["--kind", model.kind], kind: model.kind, runner: runner.name,
            environment: ["PATH": "/usr/bin:/bin", "ARSENAL_SUPPORT_DIR": paths.support.path, "ARSENAL_CACHE_DIR": paths.cache.path],
            log: paths.support.appendingPathComponent("chat-fixture.log"))
    }
}

struct LocalChatCatalogTests {
    @Test func nonChatCatalogStillDiscoversAndDispatchesUserAddedChatOverHTTPAndMCP() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-local-chat-" + UUID().uuidString)
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        try paths.create()
        defer { try? FileManager.default.removeItem(at: root) }

        // A real non-chat catalog entry, with no signed chat descriptor to fall back to.
        let signedFolder = repo.appendingPathComponent("models/laya-english")
        let manifest = try ModelManifest.load(signedFolder)
        #expect(CatalogShipping.includes(manifest) && manifest.kind != "chat")
        var sourceFiles: [String: Data] = [:]
        let inventory = try SafePath.files(in: signedFolder).map { path -> CatalogFile in
            let data = try Data(contentsOf: signedFolder.appendingPathComponent(path))
            sourceFiles["models/" + manifest.id + "/" + path] = data
            return CatalogFile(path: path, sha256: FileDigest.sha256(data), bytes: Int64(data.count))
        }
        let sharedKinds = repo.appendingPathComponent("Resources/kinds")
        let kindFiles = try SafePath.files(in: sharedKinds).map { path -> CatalogFile in
            let data = try Data(contentsOf: sharedKinds.appendingPathComponent(path))
            sourceFiles["kinds/" + path] = data
            return CatalogFile(path: path, sha256: FileDigest.sha256(data), bytes: Int64(data.count))
        }
        let document = CatalogDocument(
            sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: [CatalogModel(manifest: manifest, files: inventory)], kindFiles: kindFiles)
        #expect(document.models.count == 1 && document.models.allSatisfy { $0.manifest.kind != "chat" })
        let data = try document.encoded(), key = Curve25519.Signing.PrivateKey()
        let memoryFile = paths.support.appendingPathComponent("memory.json")
        try WireJSON.encode(["available_mb": 128000]).write(to: memoryFile)
        var config = ArsenalConfiguration()
        config.get = ["localSources": [.string(root.appendingPathComponent("donor").path)]]
        try config.write(paths.support.appendingPathComponent("config.json"))
        let kindsDirectory = repo.appendingPathComponent("Resources/kinds")
        let service = try await RealModelsService.open(
            catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation,
            source: ChatBoundarySource(files: sourceFiles), kinds: KindSpec.load(directory: kindsDirectory), paths: paths, artifacts: ChatBoundaryWorker(),
            kindsDirectory: kindsDirectory,
            memoryProbe: MemoryProbe(environment: ["ARSENAL_TEST_MEMORY_FILE": memoryFile.path], totalMB: 128000))
        let server = ArsenalAPIServer(service: service, supportDirectory: paths.support, port: 0)
        do {
            #expect(await service.catalog(query: .init(kinds: ["chat"])).isEmpty)
            let folder = paths.support.appendingPathComponent("local-models/fixture-chat")
            try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: repo.appendingPathComponent("Tests/Fixtures/DropIn/fixture-layers"), to: folder)
            let descriptor = folder.appendingPathComponent("model.toml")
            var parser = TOMLParser(try String(contentsOf: descriptor, encoding: .utf8))
            var table = try parser.parse()
            table["id"] = .string("fixture-chat"); table["kind"] = .string("chat"); table["kind_file"] = nil
            table["tools"] = .array([.string("chat")])
            table["runner"] = .array([
                .table([
                    "name": .string("native"), "language": .string("swift"), "worker": .string("chat"), "protocol": .integer(1),
                    "chips": .array([.string("*")]), "min_macos": .string("26.0"), "fast_exact": .bool(false)
                ])
            ])
            table["selection"] = .table(["default": .string("native")])
            try TOMLWriter.encode(table).write(to: descriptor, atomically: true, encoding: .utf8)
            try WireJSON.encode(["chat": ["messages": [["role": "user", "content": "Hello"]], "max_tokens": 8, "stream": false]])
                .write(to: folder.appendingPathComponent("gate/runner-check.json"))
            _ = try ModelManifest.load(folder, kinds: ["chat"], kindsDirectory: kindsDirectory)
            let donor = root.appendingPathComponent("donor/fixture/layers")
            try FileManager.default.createDirectory(at: donor, withIntermediateDirectories: true)
            try Data([12, 34, 56]).write(to: donor.appendingPathComponent("base.bin"))
            try await service.refresh()
            let discovered = await service.catalog(query: .init(kinds: ["chat"]))
            #expect(discovered.map { $0.row.id } == ["fixture-chat"])
            try await service.add(id: "fixture-chat")
            let plan = try await service.getPlan(id: "fixture-chat", dtype: "bf16")
            #expect(plan.totalDownloadBytes == 0)
            try await service.get(id: "fixture-chat", precision: "bf16", approvedPlan: plan)

            let port = try await server.start()
            defer { server.stop() }
            let router = APIRouter(service: service, support: paths.support)
            await router.setPort(port)
            let (listed, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/v1/models?kind=chat")!)
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            #expect(try WireJSON.decode(listed)["data"]?.array?.compactMap { $0["id"]?.string } == ["fixture-chat"])
            let mcpList = try await router.callMCPTool(
                MCPNames.tool("find"), arguments: ["task": "chat", "on_mac": "installed", "format": "json"], requester: "Chat boundary", server: nil, progress: nil)
            let models = try WireJSON.decode(JSONEncoder().encode(try #require(mcpList.structuredContent)))
            let listedIDs = models["models"]?.array?.compactMap { $0["id"]?.string }
            #expect(listedIDs == ["fixture-chat"])
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"; request.timeoutInterval = 15
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let input: JSONObject = ["messages": [["role": "user", "content": "Hello"]], "max_tokens": 8, "stream": false]
            request.httpBody = try WireJSON.encode(input.merging(["model": "fixture-chat"]) { _, new in new })
            let (reply, status) = try await URLSession.shared.data(for: request)
            #expect((status as? HTTPURLResponse)?.statusCode == 200, "\(String(decoding: reply, as: UTF8.self))")
            #expect(try WireJSON.decode(reply)["choices"]?.array?.first?["message"]?["content"] == "Hello from Arsenal.")
            let arguments: JSONObject = ["model": "fixture-chat", "input": .object(input), "wait_seconds": 5]
            let mcpReply = try await router.callMCPTool(
                MCPNames.tool("run"), arguments: arguments, requester: "Chat boundary", server: nil, progress: nil)
            #expect(mcpReply.isError != true)
            #expect(mcpResultText(mcpReply).hasSuffix("\nHello from Arsenal."), "\(mcpResultText(mcpReply))")
            await service.shutdown()
            #expect(await service.supervisor.residentIDs().isEmpty)
        } catch { server.stop(); await service.shutdown(); throw error }
    }
}
