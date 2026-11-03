// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

struct Fix31NegotiationTests {
    @Test func legacyChatAcceptSSEStillForcesTokensAndLegacyAudioStaysWAV() async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-fix31-neg-" + UUID().uuidString)
        let row = ModelRow(id: "legacy", name: "Legacy", kind: "voices", states: ["bf16": .init(state: .ready)])
        let operations = [
            ModelOperation(name: "chat", title: "Chat", inputSchema: ["type": "object", "properties": ["stream": ["type": "boolean"]]], output: "text-stream", stream: true),
            ModelOperation(name: "speak", title: "Speak", inputSchema: ["type": "object", "properties": ["stream": ["type": "boolean"]]], output: "audio", stream: true)
        ]
        let service = StubModelsService(rows: [row], operations: [row.id: operations])
        let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
        let port = try await server.start(); defer { server.stop(); try? FileManager.default.removeItem(at: support) }
        func request(_ op: String, _ accept: String, _ body: JSONObject) async throws -> (Data, HTTPURLResponse) {
            var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/legacy/ops/\(op)")!)
            req.httpMethod = "POST"; req.httpBody = try WireJSON.encode(body); req.timeoutInterval = 15
            req.setValue("application/json", forHTTPHeaderField: "Content-Type"); req.setValue(accept, forHTTPHeaderField: "Accept")
            let (data, response) = try await URLSession.shared.data(for: req); return (data, try #require(response as? HTTPURLResponse))
        }
        let chat = try await request("chat", "text/event-stream", [:])
        #expect(chat.1.statusCode == 200 && chat.1.mimeType == "text/event-stream")
        #expect(String(decoding: chat.0, as: UTF8.self).contains("event: token"))
        for accept in ["application/json", "application/json;q=1", "application/json, text/plain, */*", "text/event-stream", "text/event-stream;q=1"] {
            let audio = try await request("speak", accept, ["text": "hello"])
            #expect(audio.1.statusCode == 200 && audio.1.mimeType == "audio/wav")
            #expect(audio.0.starts(with: Data("RIFF".utf8)))
        }
        let stream = try await request("speak", "text/event-stream", ["text": "hello", "stream": true])
        #expect(stream.1.statusCode == 200 && stream.1.mimeType == "audio/pcm")
        #expect(!String(decoding: stream.0, as: UTF8.self).contains("event: error"))
    }
    @Test func openAPIPropagatesGenericOutputAndPolicy() async throws {
        // OpenAPI documents the operations of the user's models: ones with weights on this Mac.
        let row = ModelRow(id: "new", name: "New", kind: "local.new", states: ["bf16": .init(state: .ready)])
        let op = ModelOperation(
            name: "work", title: "Work", inputSchema: ["type": "object"], output: "json", stream: true,
            outputSchema: ["type": "object"], policy: OperationPolicy(timeoutS: 7, firstResponseTimeoutS: 3), annotations: ["primary": true], schemaVersion: 3)
        let service = StubModelsService(rows: [row], operations: [row.id: [op]])
        let api = APIRouter(service: service, support: URL(fileURLWithPath: "/private/tmp/fix31-openapi"))
        let post = try await api.openAPI()["paths"]?["/arsenal/v1/models/new/ops/work"]?["post"]
        #expect(post?["x-arsenal-output"] == "json")
        #expect(post?["x-arsenal-policy"]?["timeout_s"] == 7)
        #expect(post?["x-arsenal-output-schema"]?["type"] == "object")
        #expect(post?["x-arsenal-annotations"]?["primary"] == true)
    }
    @Test func acceptWeightsRequireAnExplicitPreference() {
        for header in ["application/json;q=1", "Application/JSON; charset=utf-8; q=0.8", "text/plain;q=0.2, application/json;q=0.9"] {
            #expect(OperationAccept(header).preferred == "application/json")
        }
        #expect(OperationAccept("application/json;q=0.1, text/event-stream;q=1").preferred == "text/event-stream")
        for header in ["application/json;q=0", "application/json;q=oops", "text/event-stream;q=1.1", "application/json, */*", "application/json, text/event-stream", ""] {
            #expect(OperationAccept(header).preferred == nil)
        }
    }
    @Test func schemaThreeUsesWeightedJSONAndSSEPreferences() async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-fix31-weighted-" + UUID().uuidString)
        let row = ModelRow(id: "generic", name: "Generic", kind: "local.generic", states: ["bf16": .init(state: .ready)])
        let schema: JSONObject = ["type": "object", "properties": ["stream": ["type": "boolean"]]]
        let operations = [
            ModelOperation(name: "chat", title: "Chat", inputSchema: schema, output: "text-stream", stream: true, schemaVersion: 3),
            ModelOperation(name: "speak", title: "Speak", inputSchema: schema, output: "audio", stream: true, schemaVersion: 3)
        ]
        let service = StubModelsService(rows: [row], operations: [row.id: operations])
        let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
        let port = try await server.start(); defer { server.stop(); try? FileManager.default.removeItem(at: support) }
        func request(_ op: String, _ accept: String) async throws -> (Data, HTTPURLResponse) {
            var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/generic/ops/\(op)")!)
            req.httpMethod = "POST"; req.httpBody = try WireJSON.encode(["text": "hello"]); req.timeoutInterval = 15
            req.setValue("application/json", forHTTPHeaderField: "Content-Type"); req.setValue(accept, forHTTPHeaderField: "Accept")
            let (data, response) = try await URLSession.shared.data(for: req); return (data, try #require(response as? HTTPURLResponse))
        }
        for op in ["chat", "speak"] {
            let json = try await request(op, "text/event-stream;q=0.2, application/json;q=1")
            #expect(json.1.statusCode == 200 && json.1.mimeType == "application/json")
            if op == "speak" { #expect(try WireJSON.decode(json.0)["files"]?.array?.first?["url"]?.string != nil) }
            let sse = try await request(op, "application/json;q=0.2, text/event-stream;q=1")
            #expect(sse.1.statusCode == 200 && sse.1.mimeType == "text/event-stream")
            #expect(String(decoding: sse.0, as: UTF8.self).contains("event: result"))
        }
        for accept in ["*/*", "application/json, text/plain, */*", "application/json;q=0"] {
            let legacy = try await request("speak", accept)
            #expect(legacy.1.mimeType == "audio/wav" && legacy.0.starts(with: Data("RIFF".utf8)))
        }
    }

}
