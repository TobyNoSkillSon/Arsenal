// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

@Test func registryGivesSystemOnePriorityOverRootOpenAI() {
    let registry = DialectRegistry.standard
    #expect(registry.module(for: "/v1/systemone")?.id == "typesafe")
    #expect(registry.module(for: "/v1/judge")?.id == "typesafe")
    #expect(registry.module(for: "/v1/models")?.id == "openai")
    #expect(registry.module(for: "/providers/elevenlabs/v1/music")?.id == "elevenlabs")
    #expect(registry.modules.count == 6)
    #expect(registry.module(for: "/v1/embeddings")?.id == "capabilities")
    #expect(Set(registry.routes.map { $0.method + $0.path }).count == registry.routes.count)
}

@Test func catalogFacetsAreTypedAndValidated() throws {
    let query = try catalogQuery([
        "kind": "voices,sound", "q": "music", "dtype": "fp16", "commercial": "true", "fits_ram": "false", "sort": "size", "order": "desc", "max_size_bytes": "1000000"
    ])
    #expect(query.kinds == ["voices", "sound"] && query.text == "music")
    #expect(query.dtypes == ["fp16"] && query.commercial == true && query.fitsRAM == false)
    #expect(query.sort == .size && query.descending && query.maxSizeBytes == 1_000_000)
    for input in [["sort": "bogus"], ["order": "down"], ["commercial": "1"], ["max_size_bytes": "-1"]] {
        #expect(throws: HTTPFailure.self) { try catalogQuery(input) }
    }
}

@Test func openAPIRebasesSchemaLocalReferencesWithoutReadingFolders() async throws {
    let row = ModelRow(id: "declared", name: "Declared", kind: "voices", states: ["bf16": .init(state: .ready)])
    let op = ModelOperation(
        name: "test", title: "Test", inputSchema: ["type": "object", "$defs": ["Text": ["type": "string"]], "properties": ["text": ["$ref": "#/$defs/Text"]]], output: "json",
        stream: false)
    let stub = StubModelsService(rows: [row], operations: [row.id: [op]])
    let router = APIRouter(service: stub, support: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-api/openapi"))
    let document = try await router.openAPI()
    #expect(document["openapi"] == "3.1.0")
    #expect(document["components"]?["schemas"]?["declared_test"]?["properties"]?["text"]?["$ref"] == "#/components/schemas/declared_test/$defs/Text")
    #expect(
        document["paths"]?["/arsenal/v1/models/declared/ops/test"]?["post"]?["requestBody"]?["content"]?["application/json"]?["schema"]?["$ref"]
            == "#/components/schemas/declared_test")
}

@Test func universalRejectsSchemaBeforeStartingStream() async throws {
    let row = ModelRow(id: "declared", name: "Declared", kind: "chat", states: ["bf16": .init(state: .ready)])
    let op = ModelOperation(
        name: "chat", title: "Chat", inputSchema: ["type": "object", "required": ["messages"], "properties": ["messages": ["type": "array", "minItems": 1]]], output: "text-stream",
        stream: true)
    let server = ArsenalAPIServer(
        service: StubModelsService(rows: [row], operations: [row.id: [op]]), supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-api/schema-\(UUID().uuidString)"), port: 0)
    let port = try await server.start(); defer { server.stop() }
    for body: JSONObject in [[:], ["messages": []], ["messages": [["role": "user", "content": "test"]], "out_dir": "/tmp/escape"]] {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/declared/ops/chat")!)
        request.httpMethod = "POST"; request.httpBody = try WireJSON.encode(body); request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 422)
        #expect((response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Content-Type") == "application/json")
        #expect(!data.isEmpty)
    }
}

@Test func soundInputAndStereoWAVAreNotMislabelled() throws {
    let params = try replicateSoundParams(["prompt": "Piano", "op": "generate_music"])
    #expect(params["duration"] == nil)
    for value: JSONValue in ["5", -1, 0, 381, true, .null] {
        #expect(throws: HTTPFailure.self) { try replicateSoundParams(["prompt": "Piano", "duration": value]) }
    }
    let raw = Data(repeating: 0, count: 17640)
    let wav = PCM.wav(raw, rate: 44100, channels: 2)
    let read = try PCM.readWAV(wav)
    #expect(read.0 == raw && read.1 == 44100 && read.2 == 2)
}

@Test func digestBoundLocalApprovalRejectsStaleContent() async throws {
    let stub = StubModelsService(rows: [StubModelsService.localFixture])
    let server = ArsenalAPIServer(service: stub, supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-api/approve-\(UUID().uuidString)"), port: 0)
    let port = try await server.start(); defer { server.stop() }
    for (digest, status) in [(String(repeating: "b", count: 64), 409), (String(repeating: "a", count: 64), 200)] {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/stub-local/approve")!)
        request.httpMethod = "POST"; request.httpBody = try WireJSON.encode(["content_digest": .string(digest)])
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == status)
        if status == 409 { #expect(String(decoding: data, as: UTF8.self).contains("changed")) }
    }
    // Not downloaded, so not on the table: the catalog carries its approval.
    #expect(await stub.catalog().first { $0.id == "stub-local" }?.row.codeConsentRequired == false)
}

@Test func reasoningParametersAndBlocksStaySeparateFromAnswer() async throws {
    let router = APIRouter(service: StubModelsService(), support: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-api/reasoning"))
    let params = try await router.anthropicParams(["messages": [["role": "user", "content": "hi"]], "max_tokens": 32, "thinking": ["type": "enabled", "budget_tokens": 1024]])
    #expect(params["thinking"] == true)
    let content = try router.anthropicContent(["reasoning": "Think first.", "text": "Answer."])
    #expect(content.count == 2 && content[0]["type"] as? String == "thinking")
    #expect(content[0]["thinking"] as? String == "Think first.")
    #expect(content[1]["text"] as? String == "Answer.")
    #expect(try router.anthropicContent(["reasoning": "Think first.", "text": "Answer."], thinking: false).count == 1)
}
