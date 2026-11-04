// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

let contractIDs = [
    "breeze-tts-2", "d1-3b", "flux2-klein-4b", "kokoro-82m", "laya-english", "laya-multilingual", "laya-typed-decisions", "pocket-tts",
    "qwen3-tts-base", "qwen3-tts-voicedesign", "qwen3.8-27b", "re-use", "stable-audio-3-medium", "stable-audio-3-small-sfx", "von-1.1", "von-1.2"
]
private var contractRepo: URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
}

/// Uses real folder schemas/defaults and production worker mapping, with capture engines instead of weights.
/// It loads no model and uses no production profile.
private actor ContractService: ModelsService {
    let stub: StubModelsService
    let declared: [ModelOperation]
    let model: ModelManifest
    let support: URL
    var settingsMap = ModelSettings()
    var requests: [JSONObject] = []
    init(id: String, support: URL) throws {
        self.support = support
        let folder =
            [contractRepo.appendingPathComponent("models/" + id), contractRepo.appendingPathComponent("retired-models/" + id)].first {
                FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path)
            } ?? contractRepo.appendingPathComponent("models/" + id)
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
        // Only discovery/lifecycle uses this stub. Inference runs production workers below.
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
        return try await engineBoundary(id: id, kind: model.kind, request: mapped.merging(["op": .string(op)]) { _, new in new }, support: support, sink: sink)
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
        if id.hasPrefix("qwen3-tts") || id == "breeze-tts-2" {
            input["max_tokens"] = 64; input["temperature"] = 0.7; input["top_p"] = 0.8; input["top_k"] = 50; input["repetition_penalty"] = 1.05
        }
        return ("speak", input)
    case "sound": return ("generate_sound", ["prompt": "Rain", "duration": 2, "steps": 8, "seed": 42, "guidance": 2, "negative_prompt": "Speech"])
    case "images": return ("generate", ["prompt": "A kettle", "width": 512, "height": 512, "steps": 4, "guidance": 1, "seed": 42])
    case "chat":
        return (
            "chat",
            [
                "messages": [["role": "user", "content": "Hello"]], "thinking": false, "max_tokens": 64, "seed": 42, "system_prompt": "Be brief.", "temperature": 0.7, "top_p": 0.8,
                "stop": ["END"]
            ]
        )
    case "enhancement": return ("enhance", ["audio": ["path": .string(reference.path)], "mode": "restore", "rate": 16000])
    default: return ("decide", ["state": "A refund request", "questions": ["refund": ["type": "noul", "instructions": "Does this ask for a refund?"]]])
    }
}

@Suite(.serialized) struct NativeModelContractTests {
    @Test func completeCatalogCoverage() throws {
        // Every model folder is either exercised by the native contract above or carries its own no-weights gate (smoked by model-check).
        let shippedFolders = try FileManager.default.contentsOfDirectory(at: contractRepo.appendingPathComponent("models"), includingPropertiesForKeys: nil)
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path) }
        // Withdrawn models keep their contract in retired-models/, so the coverage check still sees them.
        let withdrawn = contractIDs.filter { id in !shippedFolders.contains { $0.lastPathComponent == id } }.map { contractRepo.appendingPathComponent("retired-models/" + $0) }
        let folders = shippedFolders + withdrawn
        let ids = folders.map(\.lastPathComponent)
        let gated = Set(folders.filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("gate/no-weights.json").path) }.map(\.lastPathComponent))
        #expect(Set(contractIDs).isSubset(of: Set(ids)))
        // Fish Audio Drama 3 (22 Oct) has no random-weights load in the Voices family yet, so neither path fits; its port
        // is checked by the opt-in .heavy FishDrama3 parity tests on saved reference tensors. A known gap, listed here
        // so any other folder without coverage still fails.
        // FLUX 3 Image Klein (in the catalog since 2 Nov) is checked the same way: Flux3KleinTests' unit tests and its opt-in
        // .heavy per-component parity against mflux on a tiny random checkpoint.
        // The final picks of 2 Nov have no random-weights load in their families yet; each is checked by its family's
        // weight-free unit tests and opt-in .heavy parity against the maker (DiariZenTests, ElephantTests' ParityTests,
        // EmbeddingGemma2ParityTests, GIMMTests, LCOParityTests, MelBandRoformerTests, MossTranscribeTests,
        // SortformerV2Tests, VoxCPM2ParityTests). FlashVSR v1.1 (4 Nov) likewise: FlashVSRTests' geometry and stream-schedule
        // unit tests, and its opt-in .heavy FP16-against-BF16-release and one-frame-per-input checks.
        let heavyOnly: Set<String> = [
            "fish-drama-3", "flux3-image-klein", "diarizen-large-v2", "elephant-1", "embeddinggemma-2", "flashvsr-v1.1", "gimm-vfi-r", "lco-embedding-omni-7b",
            "melband-roformer-kim", "moss-transcribe-diarize", "sortformer-4spk-v2", "voxcpm2"
        ]
        // Kit models moved into the bundled worker (worker = "kit-<id>") and Qwen3 Embedding 0.6B came from a planned
        // folder with their folder runners' quick checks and parity logs; their no-weights gates are recorded by the first
        // smoke of a build that contains them. Listed by name so a new folder without coverage still fails.
        let noWeightsGatePending: Set<String> = [
            "astrobert", "audiobox-aesthetics", "chronos-2", "chronos-bolt-small", "clap-htsat-fused", "da3metric-large", "depth-anything-v2-small",
            "edgetam", "eomt-coco-panoptic-large-640", "esm2-t33-650m", "esm2-t6-8m", "esm3-sm-open-v1", "florence-2-base", "florence-2-large", "geneformer-v2-104m",
            "glm-ocr", "granite-embedding-small-english-r2", "granite-ttm-r3", "grounding-dino-base", "manga-ocr-base", "mitra-classifier-2",
            "molformer-xl-both-10pct", "owlv2-base-patch16-ensemble", "qwen3-embedding-0.6b", "real-esrgan-x4plus", "sam2.1-hiera-large",
            "sam2.1-hiera-tiny", "t0-alpha", "t0-gamma", "tiger-dnr", "tiger-speech", "timesfm-2.5-200m", "timestral-209m", "triposr", "ultra-50g", "uvdoc", "vitpose-base-simple",
            "vjepa2-vitl-fpc64-256", "voice-gender-ecapa"
        ]
        #expect(noWeightsGatePending.isDisjoint(with: gated), "a recorded no-weights gate leaves the pending list")
        let ungated = Set(ids).subtracting(contractIDs).subtracting(gated).subtracting(heavyOnly).subtracting(noWeightsGatePending)
        #expect(ungated.isEmpty, "folders without a contract or a no-weights gate: \(ungated.sorted())")
        #expect(!ids.isEmpty)
    }
    @Test(arguments: contractIDs) func primaryKindHTTPAndMCP(id: String) async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-model-contract-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let inbox = support.appendingPathComponent("inbox"); try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        // MCP hands models the real path of an inbox file (`/private/var/…`); HTTP forwards what it was given.
        let reference = URL(fileURLWithPath: APIRouter.mcpRealPath(inbox)).appendingPathComponent("reference.wav")
        try PCM.wav(Data(repeating: 0, count: 96), rate: 24000).write(to: reference)
        let service = try ContractService(id: id, support: support), kind = await service.model.kind
        let (op, input) = contractInput(id, kind: kind, reference: reference)
        let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
        let port = try await server.start(); defer { server.stop() }
        var body = input, path = "/arsenal/v1/models/\(id)/ops/\(op)"
        if kind == "voices" { path = "/v1/audio/speech"; body["input"] = body.removeValue(forKey: "text"); body["response_format"] = "wav" }
        if kind == "images" { path = "/v1/images/generations"; body["size"] = "512x512"; body["width"] = nil; body["height"] = nil }
        if kind == "chat" { path = "/v1/chat/completions" }
        if kind == "decisions" { path = "/v1/systemone" }
        if kind == "sound" {
            path = "/providers/elevenlabs/v1/sound-generation"; body["text"] = body.removeValue(forKey: "prompt"); body["duration_seconds"] = body.removeValue(forKey: "duration")
            body["model_id"] = .string(id)
        }
        if kind != "enhancement" && kind != "sound" { body["model"] = .string(id) }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)" + path)!); request.httpMethod = "POST"; request.timeoutInterval = 10
        if kind == "enhancement" {
            path = "/providers/elevenlabs/v1/audio-isolation"
            request.url = URL(string: "http://127.0.0.1:\(port)" + path)!
            var multipart = Data(
                "--contract\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\n\(id)\r\n--contract\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"reference.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
                    .utf8)
            multipart.append(try Data(contentsOf: reference)); multipart.append(Data("\r\n--contract--\r\n".utf8))
            request.httpBody = multipart; request.setValue("multipart/form-data; boundary=contract", forHTTPHeaderField: "Content-Type")
        } else {
            request.httpBody = try WireJSON.encode(body); request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200, "\(id): \(String(decoding: data, as: UTF8.self))")
        if ["voices", "sound", "enhancement"].contains(kind) {
            let (pcm, rate, channels) = try PCM.readWAV(data)
            #expect(pcm.count == Int(0.2 * Double(rate)) * channels * 2)
            #expect(rate == (kind == "sound" ? 44100 : 24000))
            #expect(Array(pcm.prefix(2)) == [0, 16])
        } else {
            let output = try WireJSON.decode(data)
            if kind == "images" { #expect(output["data"]?.array?.count == 1 && output["data"]?.array?.first?["b64_json"]?.string != nil) }
            if kind == "chat" {
                #expect(output["choices"]?.array?.first?["message"]?["content"] == "Hello from Arsenal.")
                #expect(output["usage"]?["prompt_tokens"] == 17 && output["usage"]?["completion_tokens"] == 4)
            }
            if kind == "decisions" { #expect(output["model"] == .string(id) && output["answers"]?["refund"]?["noul"] == 0.7696) }
        }
        let router = APIRouter(service: service, support: support)
        var arguments: JSONObject = ["model": .string(id), "operation": .string(op), "input": .object(input), "wait_seconds": 5, "format": "json"]
        if kind == "voices" { arguments["play"] = false }
        let reply = try await router.callMCPTool(MCPNames.tool("run"), arguments: arguments, requester: "Static model contract", server: nil, progress: nil)
        #expect(reply.isError != true, "\(id): \(mcpResultText(reply))")
        let mcpOutput = try WireJSON.decode(JSONEncoder().encode(try #require(reply.structuredContent)))
        if ["voices", "sound", "images", "enhancement"].contains(kind) {
            let files = try #require(mcpOutput["files"]?.array)
            #expect(files.count == 1 && mcpOutput["model"] == .string(id))
            let file = try #require(files.first)
            #expect(file["mime"] == .string(kind == "images" ? "image/png" : "audio/wav"))
            #expect(file["path"]?.string?.hasPrefix(support.appendingPathComponent("jobs").path) == true)
        } else if kind == "chat" {
            #expect(mcpOutput["text"] == "Hello from Arsenal.")
        } else {
            #expect(mcpOutput["answers"]?["refund"]?["noul"] == 0.7696)
        }

        let recorded = await service.requests
        #expect(recorded.count == 2)
        for (index, worker) in recorded.enumerated() {
            #expect(worker["model"] == .string(id) && worker["op"] == .string(op))
            for (key, value) in input {
                if kind == "enhancement" && index == 0 {
                    if key == "audio" { #expect(worker[key]?["path"]?.string?.hasPrefix(support.appendingPathComponent("jobs").path) == true) }
                } else {
                    #expect(worker[key] == value, "\(id) drops \(key)")
                }
            }
            #expect(worker["input"] == nil && worker["duration_seconds"] == nil)
        }
    }
}
