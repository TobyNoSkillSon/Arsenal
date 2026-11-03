// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
import ArsenalWire
import WorkerTransport
import VireoCore
import SoundEngine
import ChatContract
import ChatEngine
import DecisionsProtocol
import VerdictEngine
import D1Engine
@testable import ArsenalAPI
@testable import VoicesWorker
@testable import SoundWorker
@testable import ChatWorker
@testable import D1Worker
@testable import FluxWorker

private typealias JSONObject = ArsenalWire.JSONObject
private typealias JSONValue = ArsenalWire.JSONValue

private final class SpeechCapture: WorkerSpeechModel {
    let id: String
    var sampleRate: Int { 24000 }
    var requests: [SpeechRequest] = []
    var speed: Float?
    init(_ id: String) { self.id = id }
    func setSpeed(_ speed: Float) throws { self.speed = speed }
    func unload() async {}
    func speak(_ request: SpeechRequest, control: GenerationControl, emit: @escaping ([Float]) async throws -> Void) async throws {
        requests.append(request)
        try await emit(Array(repeating: 0.125, count: 4800))
    }
}
private final class SoundCapture: SoundModel, EnhanceModel {
    let id: String
    var sampleRate: Int { id == "re-use" ? 24000 : 44100 }
    var maxSeconds: Double { id == "stable-audio-3-medium" ? 380 : 120 }
    var sound: SoundRequest?
    var enhancement: EnhanceRequest?
    var input: AudioClip?
    init(_ id: String) { self.id = id }
    func unload() async {}
    func generate(_ request: SoundRequest, control: GenerationControl) async throws -> AudioClip {
        sound = request
        return AudioClip(samples: Array(repeating: 0.125, count: sampleRate / 5), sampleRate: sampleRate)
    }
    func enhance(_ input: AudioClip, request: EnhanceRequest, control: GenerationControl) async throws -> AudioClip {
        self.input = input; enhancement = request
        return AudioClip(samples: Array(repeating: 0.125, count: sampleRate / 5), sampleRate: sampleRate)
    }
}
private final class ImageCapture: ImageEngine {
    var arguments: (String, Int, Int, Int, UInt64)?
    func image(prompt: String, width: Int, height: Int, steps: Int, seed: UInt64, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int) {
        arguments = (prompt, width, height, steps, seed)
        return (Array(repeating: 64, count: width * height * 3), width, height)
    }
}
private final class DecisionCapture: DecisionModel {
    let id: String
    var groups: [RequestGroup] = []
    init(_ id: String) { self.id = id }
    var contextLimit: Int { 8192 }
    var residentBytes: Int { 0 }
    func tokenCount(_ item: Item, _ questions: [VerdictEngine.Question]) throws -> Int { 17 }
    func predict(_ items: [Item], _ questions: [VerdictEngine.Question]) throws -> [ItemResult] {
        groups.append(RequestGroup(items: items, questions: questions))
        return items.map { _ in
            var a = Answer(); a.noul = 0.7696
            var answers = Answers()
            for q in questions { answers[q.id] = a }
            return .answers(answers)
        }
    }
}
private final class D1Capture: DecisionEngine {
    var requests: [D1Engine.Request] = []
    var convKernelOK: Bool { true }
    func loadPictures(_ urls: [URL], budget: inout PictureBudget) throws -> [Picture] { [] }
    func tokens(_ state: PyJSON?, _ questions: [D1Engine.Question]) throws -> Int { 17 }
    func imageTokens(_ state: PyJSON?, _ questions: [D1Engine.Question], _ pictures: [Picture]) throws -> Int { 17 }
    func run(_ requests: [D1Engine.Request], cancelled: () -> Bool) throws -> [Outcome] {
        self.requests = requests
        return requests.map { request in .success((probabilities: request.questions.map { _ in [0.7696, 0.2304] }, read: 17)) }
    }
}

/// Runs the actual production worker mapper with an injected engine. The returned wire frames/results are then
/// consumed by the ordinary HTTP/MCP adapters. No loaded weights or replacement request mapper.
func engineBoundary(id: String, kind: String, request: ArsenalWire.JSONObject, support: URL, sink: RunSink?) async throws -> ArsenalWire.JSONObject {
    let frames = support.appendingPathComponent("frames-" + UUID().uuidString)
    FileManager.default.createFile(atPath: frames.path, contents: nil)
    let out = try FileHandle(forWritingTo: frames)
    defer { try? out.close(); try? FileManager.default.removeItem(at: frames) }
    var r = request; r["id"] = 7
    var directResult: ArsenalWire.JSONObject?
    switch kind {
    case "voices":
        let engine = SpeechCapture(id)
        let runner = VoiceContractRunner(wire: WorkerTransport.WorkerChannel(output: out), support: support)
        runner.loaded = engine
        try SpeechOperation.check(r, modelID: id)
        directResult = try await runner.speak(r, control: GenerationControl())
        #expect(engine.requests.count == 1)
        let actual = try #require(engine.requests.first)
        #expect(actual.text == request["text"]?.string)
        #expect(actual.seed == 42)
        #expect(actual.language == (id.hasPrefix("qwen3-tts") ? "english" : nil))
        if id == "kokoro-82m" { #expect(actual.preset == "af_heart" && engine.speed == 1.2) }
        if id == "pocket-tts" { #expect(actual.preset == "alba") }
        if id == "qwen3-tts-base" { #expect(actual.reference?.samples.count == 48 && actual.referenceText == "Hello.") }
        if id.hasPrefix("qwen3-tts") || id == "breeze-tts-2" {
            #expect(actual.maxTokens == 64 && actual.temperature == 0.7)
            #expect(actual.topP == 0.8 && actual.topK == 50 && actual.repetitionPenalty == 1.05)
        }
        if id == "qwen3-tts-voicedesign" || id == "breeze-tts-2" { #expect(actual.description == "A calm narrator") }
        if id == "breeze-tts-2" { #expect(actual.guidance == 3) }
    case "sound", "enhancement":
        let engine = SoundCapture(id)
        let runner = SoundContractRunner(wire: WorkerTransport.WorkerChannel(output: out)); runner.loaded = engine
        if kind == "sound" {
            try await runner.generate(r, model: engine, control: GenerationControl())
        } else {
            try await runner.enhance(r, model: engine, control: GenerationControl())
        }
        if kind == "sound" {
            let actual = try #require(engine.sound)
            #expect(actual.prompt == "Rain" && actual.negativePrompt == "Speech")
            #expect(actual.kind == .sfx && actual.seconds == 2 && actual.steps == 8 && actual.seed == 42 && actual.guidance == 2)
        } else {
            #expect(engine.input?.samples.count == 48 && engine.input?.sampleRate == 24000)
            let actual = try #require(engine.enhancement)
            // The standard isolation endpoint chooses its default operation; MCP supplies restore/rate explicitly.
            #expect(actual.mode == (request["mode"]?.string ?? "denoise"))
            #expect(actual.options == (request["rate"] == nil ? [:] : ["rate": "16000"]))
        }
    case "images":
        let engine = ImageCapture()
        let jobs = Jobs(); #expect(jobs.admit(7))
        let runner = ImageContractRunner(wire: ImageContractChannel(output: out), jobs: jobs, support: support)
        runner.engine = engine
        let flux = try JSONDecoder().decode(ImageContractObject.self, from: WireJSON.encode(r))
        try runner.generate(flux, engine: engine)
        let actual = try #require(engine.arguments)
        #expect(actual.0 == "A kettle" && actual.1 == 512 && actual.2 == 512 && actual.3 == 4 && actual.4 == 42)
    case "chat":
        let params = try #require(JSONSerialization.jsonObject(with: WireJSON.encode(r)) as? [String: Any])
        var captured: ChatRequest?
        let result = try await ChatWorker.chat(params) { actual in
            captured = actual
            let fixture: ArsenalWire.JSONObject = [
                "text": "Hello from Arsenal.", "finish_reason": "stop", "usage": ["prompt_tokens": 17, "completion_tokens": 4],
                "ttft_s": 0.1, "tokens_per_s": 40, "timing": [:], "cached_tokens": 0, "generated_tokens": 4
            ]
            return try JSONDecoder().decode(ChatOutput.self, from: WireJSON.encode(fixture))
        }
        let actual = try #require(captured)
        #expect(actual.messages == [ChatMessage(role: "system", content: "Be brief."), ChatMessage(role: "user", content: "Hello")])
        #expect(actual.seed == 42 && actual.maxTokens == 64 && !actual.thinking)
        #expect(actual.temperature == 0.7 && actual.topP == 0.8 && actual.stops == ["END"])
        return try WireJSON.decode(JSONEncoder().encode(result))
    default:
        var body = request; body["id"] = nil; body["op"] = nil; body["model"] = .string(id)
        if id == "d1-3b" {
            let engine = D1Capture()
            let actualRunner = Backend(engine: engine)
            let py = try PyJSONParser.parse(WireJSON.encode(body))
            let fields = try actualRunner.decide(py, cancelled: { false })
            #expect(engine.requests.count == 1)
            let actual = try #require(engine.requests.first)
            #expect(actual.state?.string == "A refund request" && actual.questions.count == 1)
            #expect(actual.questions.first?.kind == .noul)
            #expect(actual.questions.first?.instructions.string == "Does this ask for a refund?")
            return try WireJSON.decode(Data(PyJSON.object(fields).dumps().utf8))
        }
        let engine = DecisionCapture(id)
        let spec = try #require(DecisionsCatalog.verdict.spec(id))
        let runner = DecisionsRunner(model: engine, spec: spec, bits: 16, recipe: .exact)
        guard case .reply(let result) = runner.decide(try WireJSON.encode(body)) else { throw EngineError("Wrong model") }
        #expect(result.status == 200 && engine.groups.count == 1)
        let actual = try #require(engine.groups.first)
        #expect(actual.items.map(\.text) == ["A refund request"] && actual.questions.count == 1)
        #expect(actual.questions.first?.id == "refund" && actual.questions.first?.kind == .noul)
        #expect(actual.questions.first?.instructions == "Does this ask for a refund?")
        return try WireJSON.decode(JSONSerialization.data(withJSONObject: result.body))
    }
    try out.close()
    let input = try FileHandle(forReadingFrom: frames); defer { try? input.close() }
    var result = directResult
    while let frame = try FrameIO.read(from: input) {
        if case .json(let value) = frame, value["ok"] == true { result = value } else { try await sink?(frame) }
    }
    return try #require(result)
}
