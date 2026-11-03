// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

private actor RerankEmbeddingService: ModelsService {
    let declaresEmbeddings: Bool
    init(_ declaresEmbeddings: Bool) { self.declaresEmbeddings = declaresEmbeddings }
    func rows() -> [ModelRow] {
        [ModelRow(id: "jina-fixture", name: "Jina fixture", kind: "embeddings", precisions: ["fp16"], states: ["fp16": .init(state: .loaded)], selectedPrecision: "fp16")]
    }
    func operations(id: String) -> [ModelOperation] {
        var properties: JSONObject = ["query": ["type": "string"], "documents": ["type": "array", "items": ["type": "string"]]]
        if declaresEmbeddings { properties["return_embeddings"] = ["type": "boolean"] }
        return [
            ModelOperation(
                name: "rerank", title: "Rank",
                inputSchema: ["type": "object", "properties": .object(properties), "required": ["query", "documents"], "additionalProperties": false], output: "json", stream: false
            )
        ]
    }
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        var result: JSONObject = ["index": 0, "relevance_score": 0.75]
        if params["return_embeddings"] == true { result["embedding"] = [0.25, 0.5] }
        return ["results": [.object(result)]]
    }
    func get(id: String, precision: String) async throws {}
    func load(id: String, precision: String?, fast: Bool?) async throws {}
    func unload(id: String) async throws {}
    func setPrecision(id: String, precision: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func runRaw(id: String, op: String, params: JSONObject) async throws -> Data { Data() }
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func updateMemory(_ settings: MemorySettings) async throws {}
    func updateSettings(_ settings: ResidencySettings) async throws {}
    func settings() async -> ResidencySettings { .init() }
    func kinds() async -> [KindSpec] { KindSpec.defaults }
    func cancelGet(id: String) async {}
    func catalogStatus() async -> CatalogStatus { .init(source: "fixture", verified: true) }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
}

struct RerankEmbeddingAliasTests {
    @Test(arguments: [true, false]) func aliasesPreserveDeclaredEmbeddingOption(declared: Bool) async throws {
        let support = URL(fileURLWithPath: "/private/tmp/rerank-alias-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let service = RerankEmbeddingService(declared)
        let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
        let port = try await server.start(); defer { server.stop() }
        for alias in ["/v1/rerank", "/v2/rerank", "/providers/cohere/v2/rerank"] {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)" + alias)!)
            request.httpMethod = "POST"; request.timeoutInterval = 10; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try WireJSON.encode(["model": "jina-fixture", "query": "q", "documents": ["d"], "return_embeddings": true])
            let (data, response) = try await URLSession.shared.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == (declared ? 200 : 422))
            if declared { #expect(try WireJSON.decode(data)["results"]?.array?.first?["embedding"] == [0.25, 0.5]) }
        }
    }
}
