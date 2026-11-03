// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
import MCP
@testable import ArsenalAPI

private struct FaultingControlService: ModelsService {
    let error: any Error
    var allowForce = false
    func rows() async -> [ModelRow] { [ModelRow(id: "model", name: "Model", kind: "voices", states: ["bf16": .init(state: .ready)])] }
    func kinds() async -> [KindSpec] { KindSpec.defaults }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws { throw error }
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws { if !allowForce || !LoadMemoryContext.force { throw error } }
    func unload(id: String) async throws { throw error }
    func setKeepHot(id: String, on: Bool) async throws { throw error }
    func delete(id: String) async throws { throw error }
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        if !allowForce || !LoadMemoryContext.force { throw error }; return ["ok": true]
    }
    func runRefusal(id: String, op: String) async -> WorkerFailure? {
        allowForce && !LoadMemoryContext.force ? error as? WorkerFailure : nil
    }
    func operations(id: String) async -> [ModelOperation] {
        [
            ModelOperation(
                name: "speak", title: "Speak", inputSchema: ["type": "object", "properties": ["text": ["type": "string"]], "additionalProperties": false], output: "json",
                stream: false)
        ]
    }
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
}

@Test func controlIntegrityFailureIs409AndIncludesRepairInstruction() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-final-c/integrity-api-\(UUID().uuidString)")
    let server = ArsenalAPIServer(service: FaultingControlService(error: ManifestError.invalid("checksum or size mismatch: model.toml")), supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    let config = URLSessionConfiguration.ephemeral; config.connectionProxyDictionary = [:]
    let session = URLSession(configuration: config); defer { session.invalidateAndCancel() }
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/load")!)
    request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = Data(#"{"id":"model"}"#.utf8)
    let (body, response) = try await session.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 409)
    #expect(String(decoding: body, as: UTF8.self).contains("checksum or size mismatch: model.toml"))
    #expect(String(decoding: body, as: UTF8.self).contains("\(MCPNames.prefix)download(model: model) to repair"))
}

@Test func controlPrecisionFailurePreservesMessageAnd400Status() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-final-c/api-\(UUID().uuidString)")
    let stub = StubModelsService(rows: [ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])])
    let server = ArsenalAPIServer(service: stub, supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    let config = URLSessionConfiguration.ephemeral; config.connectionProxyDictionary = [:]
    let session = URLSession(configuration: config); defer { session.invalidateAndCancel() }
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/load")!)
    request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = Data(#"{"id":"voice","precision":"bogus"}"#.utf8)
    let (body, response) = try await session.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 400)
    #expect(String(decoding: body, as: UTF8.self).contains("dtype"))
}

@Test func typedServiceStatusesAndMessagesArePreserved() {
    for (code, status) in [
        ("not_installed", 404), ("not_found", 404), ("precision", 400), ("memory", 507), ("busy", 409), ("integrity", 409), ("update_available", 409), ("shutting_down", 503),
        ("timeout", 504)
    ] {
        let failure = serviceHTTPFailure(WorkerFailure(code: code, message: "Useful diagnosis."))
        #expect(failure.status == status)
        #expect(failure.message == "Useful diagnosis.")
    }
    #expect(serviceHTTPFailure(ManifestError.invalid("disk full; free space and retry Get")).status == 507)
    #expect(serviceHTTPFailure(ManifestError.invalid("weights are not installed; call \(MCPNames.prefix)download(model: model)")).status == 404)
    #expect(serviceHTTPFailure(ManifestError.invalid("catalog signature rejected")).status == 409)
}

@Test func memoryRefusalAndForceTravelThroughHTTPAndMCP() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-memory-api-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let refusal = LoadMemoryRefusal(model: "model", name: "Model", precision: "bf16", neededGB: 20.4, freeGB: 11.2)
    let service = FaultingControlService(error: refusal.failure, allowForce: true)
    let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    let config = URLSessionConfiguration.ephemeral; config.connectionProxyDictionary = [:]
    let session = URLSession(configuration: config); defer { session.invalidateAndCancel() }
    for path in ["/arsenal/v1/load", "/arsenal/v1/models/model/ops/speak"] {
        for force in [false, true] {
            var body: JSONObject = path.hasSuffix("load") ? ["id": "model"] : ["text": "Hello"]
            if force { body["force"] = true }
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)" + path)!)
            request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try WireJSON.encode(body)
            let (data, response) = try await session.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == (force ? 200 : 409))
            let object = try WireJSON.decode(data)
            if !force {
                #expect(object["code"] == "insufficient_memory")
                #expect(object["details"]?["needed_gb"] == 20.4 && object["details"]?["free_gb"] == 11.2)
                #expect(object["error"]?.string?.contains("force: true") == true)
            } else {
                #expect(object["ok"] == true)
            }
        }
    }
    let token = try String(contentsOf: root.appendingPathComponent("mcp-token"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    let endpoint = URL(string: "http://127.0.0.1:\(port)/mcp")!
    let transport = HTTPClientTransport(
        endpoint: endpoint,
        requestModifier: { request in
            var request = request; request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization"); return request
        })
    let client = Client(name: "Memory admission tests", version: "1")
    _ = try await client.connect(transport: transport)
    for name in [MCPNames.tool("load"), MCPNames.tool("run")] {
        for force in [false, true] {
            var arguments: [String: MCP.Value] = ["model": "model", "force": .bool(force)]
            if name == MCPNames.tool("run") { arguments["operation"] = "speak"; arguments["input"] = ["text": "Hello"]; arguments["wait_seconds"] = 1 }
            let reply = try await client.callTool(name: name, arguments: arguments)
            let text = reply.content.compactMap { content -> String? in
                if case .text(let text, _, _) = content { return text }; return nil
            }.joined()
            if force {
                #expect(reply.isError == false, "\(text)")
            } else {
                #expect(reply.isError == true)
                let verb = name == MCPNames.tool("load") ? "Not loaded" : "Not run"
                #expect(text.hasPrefix("[memory] \(verb): Model needs ~19 GB; 10 GB free under the "), "\(text)")
                #expect(text.hasSuffix("retry with force:true only if the user accepts macOS swap slowing the whole Mac."))
            }
        }
    }
    await client.disconnect()
    #expect(!LoadMemoryContext.force)
}
