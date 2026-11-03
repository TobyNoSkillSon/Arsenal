// SPDX-License-Identifier: MIT
import Foundation
import Network
import Testing
import ArsenalCore
@testable import ArsenalAPI

private let invalidIntegers: [JSONValue] = [.number(1e100), .number(-1e100), .number(256.5), "256", true, .null, [], [:], .number(.infinity), .number(.nan)]

private func expectFailure(_ status: Int, _ operation: () async throws -> Void) async {
    do { try await operation(); Issue.record("Expected HTTP \(status)") } catch let error as HTTPFailure { #expect(error.status == status) } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test(arguments: invalidIntegers) func replicateRejectsInvalidDimensions(_ value: JSONValue) async {
    let router = APIRouter(service: StubModelsService(), support: URL(fileURLWithPath: "/tmp/arsenal-apifix/unit"))
    for key in ["width", "height"] {
        await expectFailure(422) { _ = try await router.replicateImageParams(["prompt": "test", key: value], dir: URL(fileURLWithPath: "/unused")) }
    }
}

@Test func imageNumericValidation() async throws {
    let router = APIRouter(service: StubModelsService(), support: URL(fileURLWithPath: "/tmp/arsenal-apifix/unit"))
    let directory = URL(fileURLWithPath: "/unused")
    let params = try await router.replicateImageParams(["prompt": "test", "width": 256, "height": 128, "num_outputs": 2], dir: directory)
    #expect(params["width"] == 256 && params["height"] == 128 && params["n"] == 2)
    for key in ["n", "num_outputs", "steps", "seed"] {
        for value in invalidIntegers {
            await expectFailure(422) { _ = try await router.replicateImageParams(["prompt": "test", key: value], dir: directory) }
        }
    }
    await expectFailure(422) { _ = try await router.replicateImageParams(["prompt": "test", "width": 65], dir: directory) }
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}

@Test func numericConversionsRejectInvalidWorkerValues() async throws {
    for value in invalidIntegers {
        await expectFailure(500) { _ = try tokenCount(value) }
        #expect(workerHTTPStatus(WorkerFailure(code: "unsupported", message: "Tools unsupported", details: ["status": value])) == 400)
    }
    await expectFailure(422) { _ = try jsonInteger(.number(Double(Int.max)), name: "value", range: 0...Int.max) }
    #expect(workerHTTPStatus(WorkerFailure(code: "unsupported", message: "Tools unsupported")) == 400)
    #expect(workerHTTPStatus(WorkerFailure(code: "context_length_exceeded", message: "Too long")) == 400)
    #expect(try chatUsage(["usage": ["prompt_tokens": 8, "completion_tokens": 6]])["total_tokens"] == 14)
    await expectFailure(500) { _ = try PCM.convert(Data([0, 0]), inputRate: Int.max, inputChannels: 1, outputRate: 24000) }
}

@Test func ignoredChatControlsAreRejected() async {
    let router = APIRouter(service: StubModelsService(), support: URL(fileURLWithPath: "/tmp/arsenal-apifix/unit"))
    let message: JSONValue = ["role": "user", "content": "hello"]
    await expectFailure(400) { _ = try await router.chatParams(["messages": [["role": "developer", "content": "test"]]]) }
    do { _ = try await router.chatParams(["messages": .array([message]), "response_format": ["type": "text"]]) } catch { Issue.record("Text format should be accepted: \(error)") }
    for controls: JSONObject in [
        ["tool_choice": "required"], ["tool_choice": ["type": "function", "function": ["name": "test"]]], ["parallel_tool_calls": true], ["parallel_tool_calls": false]
    ] {
        var request = controls; request["messages"] = .array([message])
        let body = request
        await expectFailure(400) { _ = try await router.chatParams(body) }
    }
    for controls: JSONObject in [["top_k": 10], ["tool_choice": ["type": "any"]], ["tool_choice": ["type": "auto", "disable_parallel_tool_use": true]]] {
        var request = controls; request["messages"] = .array([message]); request["max_tokens"] = 20
        let body = request
        await expectFailure(400) { _ = try await router.anthropicParams(body) }
    }
}

@Test func speechDoesNotCreateTransientFiles() async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-apifix/speech-\(UUID().uuidString)")
    let stub = StubModelsService(rows: [ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])])
    let router = APIRouter(service: stub, support: support)
    for _ in 0..<3 {
        let (pcm, rate, channels) = try await router.renderSpeech(id: "voice", params: ["text": "test", "voice": "af_heart", "stream": false])
        #expect(pcm.count == 9600 && rate == 24000 && channels == 1)
    }
    #expect(!FileManager.default.fileExists(atPath: support.path))
}

@Test func predictionCacheIsBoundedAndDurable() async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-apifix/predictions-\(UUID().uuidString)")
    let router = APIRouter(service: StubModelsService(), support: support)
    var ids: [String] = []
    for _ in 0..<(APIRouter.terminalPredictionCacheLimit + 10) {
        let id = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased(); ids.append(id)
        try await router.persist(
            PredictionRecord(
                id: id, model: "image", version: "version", input: ["prompt": "test"], status: "succeeded", paths: [], created: "now", precision: "bf16", runner: "stub", fast: true
            ))
    }
    #expect(await router.predictions.count == APIRouter.terminalPredictionCacheLimit)
    #expect(await router.predictions[ids[0]] == nil)
    #expect(try await router.prediction(ids[0])?.status == "succeeded")
    #expect(await router.predictions.count == APIRouter.terminalPredictionCacheLimit)
    #expect(try await router.prediction("../invalid") == nil)
    let restarted = APIRouter(service: StubModelsService(), support: support)
    try await restarted.recoverPredictions()
    #expect(await restarted.predictions.isEmpty)
    #expect(try await restarted.prediction(ids[0])?.status == "succeeded")
}

@Test func failedPredictionRemovesOnlyItsTransientDirectory() async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-apifix/cleanup-\(UUID().uuidString)")
    let router = APIRouter(service: UnsupportedChatService(), support: support)
    let directory = try await router.newJob()
    let durableDirectory = try await router.newJob()
    try Data("partial output".utf8).write(to: directory.appendingPathComponent("partial.png"))
    try Data("durable output".utf8).write(to: durableDirectory.appendingPathComponent("image.png"))
    let id = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    try await router.persist(
        PredictionRecord(
            id: id, model: "chat", version: "version", input: [:], status: "starting", paths: [], created: "now", precision: "bf16", runner: "stub", fast: true,
            outputDirectory: directory.path))
    await router.executePrediction(id: id, params: ["out_dir": .string(directory.path)])
    #expect(!FileManager.default.fileExists(atPath: directory.path))
    #expect(FileManager.default.fileExists(atPath: durableDirectory.appendingPathComponent("image.png").path))
    #expect(try await router.prediction(id)?.status == "failed")
    // Metadata records and outputs belonging to another request are never swept.
    await router.removeTransientJob(support.appendingPathComponent("jobs/replicate"))
    #expect(try await router.prediction(id)?.status == "failed")
    let url = try await router.outputURL(durableDirectory.appendingPathComponent("image.png").path)
    #expect(url.hasSuffix("/\(durableDirectory.lastPathComponent)/image.png"))
}

@Test func interruptedPredictionCleansItsDirectoryOnRestart() async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-apifix/restart-cleanup-\(UUID().uuidString)")
    let router = APIRouter(service: StubModelsService(), support: support)
    let directory = try await router.newJob()
    try Data("partial output".utf8).write(to: directory.appendingPathComponent("partial.png"))
    let id = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    try await router.persist(
        PredictionRecord(
            id: id, model: "image", version: "version", input: [:], status: "processing", paths: [], created: "now", precision: "bf16", runner: "stub", fast: true,
            outputDirectory: directory.path))
    let restarted = APIRouter(service: StubModelsService(), support: support)
    try await restarted.recoverPredictions()
    #expect(!FileManager.default.fileExists(atPath: directory.path))
    #expect(try await restarted.prediction(id)?.status == "failed")
    #expect(try await restarted.prediction(id)?.error?.contains("not regenerated") == true)
}

private extension APIRouter {
    func occupyPredictionSlotsForTest() async {
        // Completed no-op tasks stand in for occupied slots, without leaving
        // any running fixture or requiring 32 live inference operations.
        for index in 0..<Self.activePredictionLimit {
            let task = Task<Void, Never> {}; await task.value
            predictionTasks[String(index)] = task
        }
    }
}

@Test func predictionAdmissionIsBounded() async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-apifix/admission-\(UUID().uuidString)")
    let stub = StubModelsService(rows: [ModelRow(id: "image", name: "Image", kind: "images", states: ["bf16": .init(state: .ready)])])
    let router = APIRouter(service: stub, support: support)
    await router.occupyPredictionSlotsForTest()
    let request = HTTPRequest(
        method: "POST", target: "/providers/replicate/v1/predictions", headers: ["content-type": "application/json"],
        body: try WireJSON.encode(["version": "image", "input": ["prompt": "test"]]))
    let unused = HTTPConnection(NWConnection(host: .ipv4(.loopback), port: 1, using: .tcp), onEnd: {})
    await expectFailure(429) { try await router.replicate(request, unused) }
    let directories = try FileManager.default.contentsOfDirectory(atPath: support.appendingPathComponent("jobs").path)
    #expect(directories == ["replicate"])
}

// CPU service fixture emits the exact error code used by the real ChatContract.
// This covers listener -> router -> provider error mapping without loading MLX.
private struct UnsupportedChatService: ModelsService {
    var failureCode = "unsupported"
    var probe: CancellationProbe?
    func rows() async -> [ModelRow] { [ModelRow(id: "chat", name: "Chat", kind: "chat", states: ["bf16": .init(state: .ready)])] }
    func kinds() async -> [KindSpec] { [] }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws {}
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        if let probe {
            do {
                try await sink?(.json(["event": "token", "text": "hello"]))
                try await Task.sleep(for: .seconds(10))
            } catch { await probe.markCanceled(); throw error }
            return [:]
        }
        throw WorkerFailure(code: failureCode, message: "Tool calls are not supported by this chat runner yet.")
    }
    func runRaw(id: String, op: String, params: JSONObject) async throws -> Data { throw WorkerFailure(code: "unsupported", message: "Unsupported") }
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { MemorySnapshot() }
    func catalogStatus() async -> CatalogStatus { CatalogStatus() }
}

@Test(arguments: ["unsupported", "context_length_exceeded"]) func realChatUnsupportedCodeIsHTTP400(_ code: String) async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-apifix/http-\(UUID().uuidString)")
    let server = ArsenalAPIServer(service: UnsupportedChatService(failureCode: code), supportDirectory: support, port: 0)
    let port = try await server.start()
    defer { server.stop() }
    for path in ["/v1/chat/completions", "/providers/anthropic/v1/messages"] {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!); request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: JSONObject = ["model": "chat", "messages": [["role": "user", "content": "test"]], "max_tokens": 20]
        body["tools"] = path.contains("anthropic") ? [["name": "test", "input_schema": [:]]] : [["type": "function", "function": ["name": "test"]]]
        request.httpBody = try WireJSON.encode(body)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("Tool calls are not supported"))
    }
}

private actor CancellationProbe {
    var canceled = false
    func markCanceled() { canceled = true }
}

@Test func streamDisconnectCancelsModelsServiceRequest() async throws {
    let probe = CancellationProbe()
    let server = ArsenalAPIServer(
        service: UnsupportedChatService(probe: probe), supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-apifix/disconnect-\(UUID().uuidString)"), port: 0)
    let port = try await server.start()
    defer { server.stop() }
    let client = NWConnection(host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: UInt16(port))!, using: .tcp)
    client.start(queue: DispatchQueue(label: "apifix.disconnect-test"))
    let body = #"{"model":"chat","messages":[{"role":"user","content":"test"}],"stream":true}"#
    let request = Data("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:\(port)\r\nContent-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)".utf8)
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        client.send(
            content: request,
            completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
            })
    }
    let response: Data = try await withCheckedThrowingContinuation { continuation in
        client.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, _, error in
            if let data { continuation.resume(returning: data) } else { continuation.resume(throwing: error ?? CancellationError()) }
        }
    }
    #expect(String(decoding: response, as: UTF8.self).contains("HTTP/1.1 200"))
    client.cancel()
    for _ in 0..<100 where !(await probe.canceled) { try await Task.sleep(for: .milliseconds(10)) }
    #expect(await probe.canceled)
}
