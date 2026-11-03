// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

@Test func v3MemoryAndToolsRoutesPersistAndValidate() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-v3-api-" + UUID().uuidString)
    var row = ModelRow(id: "chat", name: "Chat", kind: "chat"); row.tools = ["chat"]
    row.states = ["bf16": .init(state: .ready)] // On the table: its weights are on this Mac.
    let service = StubModelsService(rows: [row], supportDirectory: root)
    let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    func send(_ method: String, _ path: String, _ body: JSONObject? = nil) async throws -> (Data, Int) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/" + path)!)
        request.httpMethod = method; request.timeoutInterval = 10
        if let body { request.httpBody = try WireJSON.encode(body); request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
    let initial = try await send("GET", "memory/settings")
    let initialBody = try WireJSON.decode(initial.0)
    #expect(initial.1 == 200 && initialBody["when_full"] == "refuse")
    let changed = try await send("POST", "memory/settings", ["when_full": "queue"])
    let changedBody = try WireJSON.decode(changed.0)
    #expect(changed.1 == 200 && changedBody["when_full"] == "queue")
    #expect(try await send("POST", "memory/settings", ["when_full": "evict"]).1 == 422)
    #expect(try await send("POST", "memory/settings", ["when_full": false]).1 == 422)
    #expect(try await send("POST", "models/chat/tools", ["on": false]).1 == 200)
    #expect(await service.rows().first?.toolsShown == false)
    #expect(try await send("POST", "models/chat/tools", ["on": "false"]).1 == 422)
    let reference = try await send("GET", "models/chat/reference?dtype=bf16&mode=fast")
    #expect(reference.1 == 200 && String(decoding: reference.0, as: UTF8.self) == "null")
    let spec = try await APIRouter(service: service, support: root).openAPI()
    #expect(spec["paths"]?["/arsenal/v1/memory/settings"]?["post"] != nil)
    #expect(spec["paths"]?["/arsenal/v1/models/{id}/tools"]?["post"] != nil)
    #expect(spec["paths"]?["/arsenal/v1/models/{id}/reference"]?["get"]?["parameters"]?.array?.count == 3)
    #expect(try ArsenalConfiguration.read(root.appendingPathComponent("config.json")).memory.whenFull == .queue)
}

@Test func v3InsufficientMemoryUses409AcrossDialects() throws {
    let error = WorkerFailure(code: "insufficient_memory", message: "Needs 4000 MB. Loaded: Chat (3000 MB).")
    #expect(serviceHTTPFailure(error).status == 409 && workerHTTPStatus(error) == 409)
    let control = try WireJSON.decode(errorBody(serviceHTTPFailure(error), path: "/arsenal/v1/load", requestID: "test"))
    #expect(control["code"] == "insufficient_memory" && control["error"] == .string(error.message))
    let standard = try WireJSON.decode(errorBody(serviceHTTPFailure(error), path: "/v1/chat/completions", requestID: "test"))
    #expect(standard["error"]?["code"] == "insufficient_memory")
    let refusal = LoadMemoryRefusal(model: "m", name: "M", precision: "bf16", neededGB: 20.4, freeGB: 11.2)
    for path in ["/providers/anthropic/v1/messages", "/providers/elevenlabs/v1/text-to-speech/voice", "/providers/replicate/v1/predictions", "/providers/typesafe/v1/judge"] {
        let body = try WireJSON.decode(errorBody(serviceHTTPFailure(refusal.failure), path: path, requestID: "test"))
        #expect(body["code"] == "insufficient_memory" && body["status"] == 409)
        #expect(body["details"]?["needed_gb"] == 20.4 && body["details"]?["free_gb"] == 11.2)
        #expect(body["details"]?["force_hint"]?.string?.contains("force: true") == true)
    }

}

private struct V3InvalidReferenceService: ModelsService {
    func reference(id: String, dtype: String?, mode: String?) async throws -> ModelReference? {
        throw WorkerFailure(code: "invalid_request", message: "Choose an offered dtype and fast or exact mode.")
    }
    func rows() async -> [ModelRow] { [] }
    func kinds() async -> [KindSpec] { [] }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws {}
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject { [:] }
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
}
@Test func v3InvalidReferenceRequestReturns422() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-v3-reference-" + UUID().uuidString)
    let server = ArsenalAPIServer(service: V3InvalidReferenceService(), supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    let (data, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/chat/reference?mode=unknown")!)
    #expect((response as? HTTPURLResponse)?.statusCode == 422)
    #expect(try WireJSON.decode(data)["code"] == "invalid_request")
}
