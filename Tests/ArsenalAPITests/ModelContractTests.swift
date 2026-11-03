// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

private let contractIDs = [
    "breeze-tts-2", "d1-3b", "flux2-klein-4b", "kokoro-82m", "laya-english", "laya-multilingual", "laya-typed-decisions", "pocket-tts",
    "qwen3-tts-base", "qwen3-tts-voicedesign", "qwen3.8-27b", "re-use", "stable-audio-3-medium", "stable-audio-3-small-sfx", "von-1.1", "von-1.2"
]
private var contractRepo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

/// Withdrawn models (non-open licences) live in retired-models/; their contract stays covered there.
private func contractFolder(_ id: String) -> URL {
    let shipped = contractRepo.appendingPathComponent("models/" + id)
    return FileManager.default.fileExists(atPath: shipped.path) ? shipped : contractRepo.appendingPathComponent("retired-models/" + id)
}

/// API adapter/schema coverage only. Production worker/engine coverage lives in NativeModelContractTests.
/// It loads no model and uses no production profile.
private actor ContractService: ModelsService {
    let stub: StubModelsService
    let declared: [ModelOperation]
    let model: ModelManifest
    var settingsMap = ModelSettings()
    var requests: [JSONObject] = []
    init(id: String, support: URL) throws {
        let folder = contractFolder(id)
        let loadedModel = try ModelManifest.load(folder)
        model = loadedModel
        let kind = try KindSpec.load(directory: contractRepo.appendingPathComponent("Resources/kinds")).first { $0.id == loadedModel.kind }
        declared = try model.resolvedOperations(kind: kind).map { op in
            let input = try WireJSON.decode(OperationSchema.read(path: op.input, folder: folder, kindsDirectory: contractRepo.appendingPathComponent("Resources/kinds")))
            return ModelOperation(name: op.name, title: op.title, inputSchema: input, output: op.output, stream: op.stream)
        }
        let metadata = try WireJSON.decode(JSONEncoder().encode(model))
        settingsMap.declare(id, metadata: metadata)
        let row = ModelRow(id: id, name: model.name, kind: model.kind, states: ["bf16": .init(state: .ready)])
        // Response fixture has no model-specific behaviour: validation and capture above it use the real contract.
        stub = StubModelsService(rows: [row], operations: [id: declared], supportDirectory: support)
    }
    func rows() async -> [ModelRow] { await stub.rows() }
    func kinds() async -> [KindSpec] { await stub.kinds() }
    func operations(id: String) async -> [ModelOperation] { declared }
    func catalog(query: CatalogQuery) async -> [CatalogEntry] { await stub.catalog(query: query) }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws {}
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        let input = try params["body"]?.string.map { try WireJSON.decode(Data($0.utf8)) } ?? params
        let operation = try #require(declared.first { $0.name == op })
        let mapped = settingsMap.request(id, params: input, schema: operation.inputSchema)
        try OperationSchemaValidator.validate(.object(mapped), schema: .object(operation.inputSchema))
        requests.append(mapped.merging(["model": .string(id), "op": .string(op)]) { _, new in new })
        if op == "clone_voice" { return ["voice_id": "voice-11111111-1111-1111-1111-111111111111"] }
        var fixture = mapped
        // The captured request retains the model's voice; generic response generation does not choose its identity.
        fixture["voice"] = nil
        return try await stub.run(id: id, op: op, params: fixture, sink: sink)
    }
}
private func contractInput(_ id: String, kind: String, reference: URL) -> (String, JSONObject) {
    switch kind {
    case "voices":
        var input: JSONObject = ["text": "Hello.", "seed": 42]
        if id == "kokoro-82m" { input["voice"] = "af_heart"; input["speed"] = 1.2 }
        if id == "pocket-tts" { input["voice"] = "alba" }
        if id == "qwen3-tts-base" { input["reference"] = ["path": .string(reference.path), "text": "Hello."]; input["language"] = "english" }
        if id == "qwen3-tts-voicedesign" || id == "breeze-tts-2" {
            input["instructions"] = "A calm narrator"; input["temperature"] = 0.7; input["max_tokens"] = 64
        }
        if id == "breeze-tts-2" { input["guidance"] = 3 }
        return ("speak", input)
    case "sound": return ("generate_sound", ["prompt": "Rain", "duration": 2, "steps": 8, "seed": 42, "guidance": 2, "negative_prompt": "Speech"])
    case "images": return ("generate", ["prompt": "A kettle", "width": 512, "height": 512, "steps": 4, "guidance": 1, "seed": 42])
    case "chat": return ("chat", ["messages": [["role": "user", "content": "Hello"]], "thinking": false, "max_tokens": 64, "seed": 42, "system_prompt": "Be brief."])
    case "enhancement": return ("enhance", ["audio": ["path": .string(reference.path)], "mode": "restore", "rate": 16000])
    default: return ("decide", ["state": "A refund request", "questions": ["refund": ["type": "noul", "instructions": "Does this ask for a refund?"]]])
    }
}

struct ModelContractTests {}

extension ModelContractTests {
    @Test func tableDiscoveryExcludesCatalogOnlyRowsAndIncludesPendingGet() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-list-contract-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        // The table holds exactly the models with weights on this Mac or a Get under way: "-pending" is downloading,
        // "-catalog" has no weights (the stub records every row as added, so membership alone does not list it).
        let rows = ["voices", "sound", "enhancement", "chat", "images", "decisions"].flatMap { kind in
            [
                ModelRow(id: kind + "-pending", name: kind, kind: kind, states: ["bf16": .init(state: .downloading, progress: 0.1)]),
                ModelRow(id: kind + "-catalog", name: kind, kind: kind)
            ]
        }
        let service = StubModelsService(rows: rows, supportDirectory: support)
        let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
        let port = try await server.start(); defer { server.stop() }
        for path in [
            "/v1/models", "/v1/models?kind=chat", "/arsenal/v1/models", "/providers/anthropic/v1/models", "/providers/elevenlabs/v1/models", "/providers/typesafe/v1/models",
            "/providers/replicate/v1/models"
        ] {
            let (data, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)" + path)!)
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            let text = String(decoding: data, as: UTF8.self)
            #expect(text.contains("-pending") && !text.contains("-catalog"), "\(path): \(text)")
        }
        let (_, listed) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/v1/models/chat-pending")!)
        #expect((listed as? HTTPURLResponse)?.statusCode == 200)
        let (_, hidden) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/v1/models/chat-catalog")!)
        #expect((hidden as? HTTPURLResponse)?.statusCode == 404)
        let (browse, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/catalog/browse?kind=chat")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 200 && String(decoding: browse, as: UTF8.self).contains("chat-catalog"))
        let router = APIRouter(service: service, support: support)
        for (place, count): (JSONValue?, Int) in [("table", 1), (nil, 2)] {
            var arguments: JSONObject = ["task": "chat", "format": "json"]
            if let place { arguments["on_mac"] = place }
            let reply = try await router.callMCPTool(MCPNames.tool("find"), arguments: arguments, requester: "Contract", server: nil, progress: nil)
            let result = try WireJSON.decode(JSONEncoder().encode(try #require(reply.structuredContent)))
            #expect(result["models"]?.array?.count == count)
        }
        let spec = try await router.openAPI()
        #expect(spec["paths"]?["/arsenal/v1/catalog/browse"]?["get"] != nil)
        #expect(MCPTools.all.contains { $0.name == MCPNames.tool("find") })
    }

    @Test func modelsRejectUnsupportedFieldsBeforeWorkerDispatch() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-schema-contract-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        for id in contractIDs {
            let service = try ContractService(id: id, support: support)
            let kind = await service.model.kind
            var (op, input) = contractInput(id, kind: kind, reference: support.appendingPathComponent("inbox/ref.wav"))
            if kind == "decisions" {
                input["questions"] = [:]
            } else if id == "flux2-klein-4b" {
                input["guidance"] = 2
            } else if kind == "chat" {
                input["tools"] = [["type": "function"]]
            } else if id == "breeze-tts-2" {
                input["reference"] = ["path": "/tmp/ref.wav"]
            } else if kind == "voices" {
                input["language"] = "polish"
            } else if kind == "sound" {
                input["duration"] = 381
            } else {
                input["rate"] = 12345
            }
            await #expect(throws: WorkerFailure.self) { _ = try await service.run(id: id, op: op, params: input, sink: nil) }
            #expect(await service.requests.isEmpty)
        }
    }

    @Test func speechAliasesSamplingAndSoundGuidanceSurviveAdapters() async throws {
        let router = APIRouter(service: StubModelsService(), support: contractRepo.appendingPathComponent("tmp/contracts/unused"))
        let input: JSONObject = [
            "input": "Hello", "description": "Narrator", "seed": 42, "max_tokens": 64, "top_p": 0.8, "top_k": 50, "temperature": 0.7, "repetition_penalty": 1.05, "guidance": 3
        ]
        let mapped = try await router.speechParams(input)
        for (key, value) in input where key != "input" { #expect(mapped[key] == value) }
        #expect(mapped["text"] == "Hello")
        let sound = try replicateSoundParams(["prompt": "Rain", "guidance": 2, "negative_prompt": "Speech", "steps": 8])
        #expect(sound["guidance"] == 2 && sound["negative_prompt"] == "Speech")
        #expect(throws: HTTPFailure.self) { try replicateSoundParams(["prompt": "Rain", "steps": 101]) }
    }
}

extension ModelContractTests {
    @Test(arguments: contractIDs) func secondaryDeclaredOperationsUseUniversalHTTPAndMCP(id: String) async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-secondary-contract-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let inbox = support.appendingPathComponent("inbox"); try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        // MCP hands models the real path of an inbox file (`/private/var/…`); HTTP forwards what it was given.
        let audio = URL(fileURLWithPath: APIRouter.mcpRealPath(inbox)).appendingPathComponent("reference.wav")
        try PCM.wav(Data(repeating: 0, count: 96), rate: 24000).write(to: audio)
        let service = try ContractService(id: id, support: support)
        let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
        let port = try await server.start(); defer { server.stop() }
        let router = APIRouter(service: service, support: support)
        let kind = await service.model.kind, primary = contractInput(id, kind: kind, reference: audio).0
        var checked = 0
        for op in await service.declared where op.name != primary {
            let input: JSONObject
            switch op.name {
            case "voices": input = [:]
            case "clone_voice":
                var clone: JSONObject = ["audio": ["path": .string(audio.path)], "name": "Narrator"]
                if id != "pocket-tts" { clone["transcript"] = "Hello." }
                input = clone
            case "design_voice": input = ["description": "Calm narrator", "preview_text": "Hello.", "seed": 42, "temperature": 0.7]
            case "generate_music": input = ["prompt": "Piano", "duration": 10, "guidance": 2, "negative_prompt": "Speech", "seed": 42]
            case "judge": input = ["items": ["A refund", "An invoice"], "questions": ["refund": ["type": "noul", "instructions": "Is this a refund?"]]]
            case "count_tokens": input = ["messages": [["role": "user", "content": "Hello"]], "thinking": false]
            default: Issue.record("Uncovered operation \(id)/\(op.name)"); continue
            }
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/\(id)/ops/\(op.name)")!)
            request.httpMethod = "POST"; request.timeoutInterval = 10; request.httpBody = try WireJSON.encode(input)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let (data, response) = try await URLSession.shared.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            if op.output == "audio" {
                #expect(try PCM.readWAV(data).0.count == 9600)
            } else {
                let output = try WireJSON.decode(data)
                switch op.name {
                case "voices": #expect(output["voices"]?.array?.count == 1)
                case "judge": #expect(output["results"]?.array?.count == 2)
                case "count_tokens": #expect(output["tokens"] == 8)
                default: #expect(output["voice_id"]?.string != nil)
                }
            }
            let reply = try await router.callMCPTool(
                MCPNames.tool("run"), arguments: ["model": .string(id), "operation": .string(op.name), "input": .object(input), "wait_seconds": 5], requester: "Contract",
                server: nil,
                progress: nil)
            #expect(reply.isError != true, "\(mcpResultText(reply))")
            let recorded = await service.requests.suffix(2)
            #expect(recorded.count == 2)
            for worker in recorded { for (key, value) in input { #expect(worker[key] == value, "\(id)/\(op.name) drops \(key)") } }
            checked += 1
        }
        let declaredCount = await service.declared.count
        #expect(checked == declaredCount - 1)
    }
}

extension ModelContractTests {
    @Test func stopByteLimitRejectsMultibyteBeforeDispatchForStringsAndArrays() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-stop-contract-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let service = try ContractService(id: "qwen3.8-27b", support: support)
        for array in [false, true] {
            let accepted = String(repeating: "🛑", count: 256), rejected = String(repeating: "🛑", count: 257)
            var input: JSONObject = ["messages": [["role": "user", "content": "Hello"]], "stop": array ? .array([.string(accepted)]) : .string(accepted)]
            _ = try await service.run(id: "qwen3.8-27b", op: "chat", params: input, sink: nil)
            input["stop"] = array ? .array([.string(rejected)]) : .string(rejected)
            await #expect(throws: WorkerFailure.self) { _ = try await service.run(id: "qwen3.8-27b", op: "chat", params: input, sink: nil) }
        }
        #expect(await service.requests.count == 2)
    }
    @Test func ineffectiveVoiceFieldsAreAbsentAndRejectedWhileQwenLanguagesAreCanonical() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-voice-fields-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        for id in ["kokoro-82m", "pocket-tts", "breeze-tts-2", "qwen3-tts-base", "qwen3-tts-voicedesign"] {
            let service = try ContractService(id: id, support: support)
            let operation = try #require(await service.declared.first { $0.name == "speak" })
            let properties = try #require(operation.inputSchema["properties"]?.object)
            let (_, input) = contractInput(id, kind: "voices", reference: support.appendingPathComponent("inbox/ref.wav"))
            let removed =
                id == "kokoro-82m" || id == "pocket-tts"
                ? ["language", "instructions", "description"]
                : id == "breeze-tts-2" ? ["language", "speed"] : id == "qwen3-tts-base" ? ["speed", "instructions", "description"] : ["speed"]
            for key in removed {
                #expect(properties[key] == nil)
                var invalid = input; invalid[key] = key == "speed" ? 1 : "ignored"
                await #expect(throws: WorkerFailure.self) { _ = try await service.run(id: id, op: "speak", params: invalid, sink: nil) }
            }
            if id.hasPrefix("qwen3-tts") {
                let rawLanguages = properties["language"]?["enum"]?.array?.compactMap { $0.string }
                let languages = try #require(rawLanguages)
                #expect(languages.count == 11 && !languages.contains("en") && languages.contains("english"))
            }
        }
        let pocket = try ContractService(id: "pocket-tts", support: support)
        let clone = try #require(await pocket.declared.first { $0.name == "clone_voice" })
        #expect(clone.inputSchema["properties"]?["transcript"] == nil)
        try OperationSchemaValidator.validate(["audio": ["path": "/tmp/ref.wav"], "name": "Narrator"], schema: .object(clone.inputSchema))
    }
}
