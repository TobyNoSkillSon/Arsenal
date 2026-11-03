// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalAPI
@testable import ArsenalCore

@Test func dialectAbsenceAllowsSavedDefaultsAndExplicitDimensionsWin() async throws {
    let router = APIRouter(service: StubModelsService(rows: []), support: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/api"))
    let dir = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/api/jobs")
    for params in [try await router.imageParams(["prompt": "test"], dir: dir), try await router.replicateImageParams(["prompt": "test"], dir: dir)] {
        #expect(params["width"] == nil && params["height"] == nil)
    }
    #expect(try await router.imageParams(["prompt": "test", "size": "768x512"], dir: dir)["width"] == 768)
    let partial = try await router.replicateImageParams(["prompt": "test", "width": 512], dir: dir)
    #expect(partial["width"] == 512 && partial["height"] == nil)
    var settings = ModelSettings()
    settings.declarations["sound"] = [["id": "duration", "type": "number", "default": 5, "applies": "request", "maps_to": "request.duration"]]
    try settings.set("sound", key: "duration", value: 12)
    for params in [try await router.musicParams(["prompt": "test"]).0, try replicateSoundParams(["prompt": "test"])] {
        #expect(params["duration"] == nil)
        #expect(settings.request("sound", params: params, schema: ["properties": ["duration": ["type": "number"]]])["duration"] == 12)
    }
    let planned = try await router.musicParams(["composition_plan": ["positive_global_styles": ["piano"], "sections": [["duration_ms": 7000]]]]).0
    #expect(planned["duration"] == 7)
    let explicit = try await router.musicParams(["prompt": "test", "music_length_ms": 9000]).0
    #expect(settings.request("sound", params: explicit)["duration"] == 9)
}

@Test func removedModelsWithDeletedWeightsDisappearFromProviderListingAndResolution() async throws {
    let row = ModelRow(id: "gone", name: "Gone", kind: "chat", states: ["bf16": .init(state: .ready)])
    let stub = StubModelsService(rows: [row])
    try await stub.remove(id: row.id, deleteWeights: true)
    let router = APIRouter(service: stub, support: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/removed"))
    #expect(await router.providerModelsAreEmpty())
    await #expect(throws: HTTPFailure.self) { _ = try await router.resolveRemovedModel(row.id) }
}
/// Remove keeps the downloaded files, so the model stays installed and usable off the table (agents use it), and the HTTP
/// listing and resolution include it.
@Test func removedModelKeepingItsWeightsStaysInstalledOffTheTable() async throws {
    let row = ModelRow(id: "kept", name: "Kept", kind: "chat", states: ["bf16": .init(state: .ready)])
    let stub = StubModelsService(rows: [row])
    try await stub.remove(id: row.id, deleteWeights: false)
    let router = APIRouter(service: stub, support: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/kept"))
    #expect(await router.installedIDs() == [row.id])
    #expect(try await router.resolveRemovedModel(row.id) == row.id)
}
extension APIRouter {
    func providerModelsAreEmpty() async -> Bool { await installedModels().isEmpty }
    func installedIDs() async -> [String] { await installedModels().compactMap { $0["id"] as? String } }
    func resolveRemovedModel(_ id: String) async throws -> String? { try await resolve(id, kind: "chat")["id"] as? String }
}

private actor RequestProbe {
    var calls: [(String, JSONObject)] = []
    func record(_ op: String, _ params: JSONObject) { calls.append((op, params)) }
    func last() -> JSONObject { calls.last?.1 ?? [:] }
}
private struct RecordingService: ModelsService {
    let kind: String
    let probe: RequestProbe
    var saved: JSONObject = [:]
    var voiceSupport: URL?
    func rows() async -> [ModelRow] { [ModelRow(id: "model", name: "Model", kind: kind, states: ["bf16": .init(state: .ready)])] }
    func kinds() async -> [KindSpec] { [] }
    func events() async -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) async throws {}
    func cancelGet(id: String) async {}
    func load(id: String, precision: String?, fast: Bool?) async throws {}
    func unload(id: String) async throws {}
    func setKeepHot(id: String, on: Bool) async throws {}
    func delete(id: String) async throws {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        // Exercise the same saved-setting merge as the actual service after dialect adaptation.
        var settings = ModelSettings()
        settings.declarations[id] = saved.map { key, value in
            ["id": .string(key), "type": value.string != nil ? "text" : "number", "default": value, "applies": "request", "maps_to": .string("request." + key)]
        }
        let merged = settings.request(id, params: params)
        await probe.record(op, merged)
        if op == "delete_voice" { try await deleteSavedVoice(id: id, voiceID: params["voice_id"]!.string!); return ["ok": true] }
        if op == "count_tokens" { return ["input_tokens": 8] }
        if op == "generate" {
            let directory = URL(fileURLWithPath: params["out_dir"]!.string!)
            let file = directory.appendingPathComponent("fixture.png")
            try Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j6ioAAAAASUVORK5CYII=")!.write(to: file)
            return ["images": [["path": .string(file.path)]]]
        }
        try await sink?(.json(["event": "audio_begin", "sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"]))
        try await sink?(.binary(stream: 1, bytes: Data(repeating: 0, count: 48)))
        return ["sample_rate": 24000, "channels": 1]
    }
    func menuActions() async -> [LoadedAction] { [] }
    func memory() async -> MemorySnapshot { .init() }
    func catalogStatus() async -> CatalogStatus { .init() }
    func deleteSavedVoice(id: String, voiceID: String) async throws {
        guard let voiceSupport else { throw WorkerFailure(code: "unknown_voice", message: "No saved voice") }
        try SavedVoiceStore.delete(model: id, voiceID: voiceID, support: voiceSupport)
    }
    func operations(id: String) async -> [ModelOperation] {
        [
            ModelOperation(
                name: "delete_voice", title: "Delete saved voice", inputSchema: ["type": "object", "required": ["voice_id"], "properties": ["voice_id": ["type": "string"]]],
                output: "json", stream: false), ModelOperation(name: "generate_sound", title: "Sound", inputSchema: [:], output: "audio", stream: true),
            ModelOperation(name: "generate_music", title: "Music", inputSchema: [:], output: "audio", stream: true)
        ]
    }
}
private func send(port: Int, path: String, body: JSONObject) async throws -> Int {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
    request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type"); request.httpBody = try WireJSON.encode(body)
    let (_, response) = try await URLSession.shared.data(for: request)
    return (response as? HTTPURLResponse)?.statusCode ?? 0
}
@Test func elevenLabsDurationReachesSavedSettingsThroughRealListener() async throws {
    let probe = RequestProbe()
    let server = ArsenalAPIServer(
        service: RecordingService(kind: "sound", probe: probe, saved: ["duration": 12]),
        supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/duration-\(UUID().uuidString)"), port: 0)
    let port = try await server.start(); defer { server.stop() }
    for (endpoint, body, expected): (String, JSONObject, JSONValue) in [
        ("sound-generation", ["text": "rain"], 12), ("sound-generation", ["text": "rain", "duration_seconds": 7], 7), ("music", ["prompt": "piano"], 12),
        ("music", ["prompt": "piano", "music_length_ms": 9000], 9)
    ] {
        #expect(try await send(port: port, path: "/providers/elevenlabs/v1/" + endpoint, body: body) == 200)
        #expect(await probe.last()["duration"] == expected)
    }
    #expect(try await send(port: port, path: "/providers/elevenlabs/v1/sound-generation", body: ["text": "rain", "duration_seconds": .null]) == 422)
}
@Test func anthropicTokenCountForwardsPromptFieldsNotSampling() async throws {
    let probe = RequestProbe()
    let server = ArsenalAPIServer(
        service: RecordingService(kind: "chat", probe: probe), supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/count-\(UUID().uuidString)"), port: 0)
    let port = try await server.start(); defer { server.stop() }
    for type in ["disabled", "enabled"] {
        #expect(
            try await send(
                port: port, path: "/providers/anthropic/v1/messages/count_tokens",
                body: [
                    "model": "model", "messages": [["role": "user", "content": "hi"]], "system": "system prompt", "thinking": ["type": .string(type)], "max_tokens": 20,
                    "temperature": 0.5, "tools": [["name": "lookup", "input_schema": ["type": "object"]]]
                ]) == 200)
        let params = await probe.last()
        #expect(params["thinking"] == .bool(type == "enabled") && params["tools"] != nil)
        #expect(params["messages"]?.array?.first?["role"] == "system")
        #expect(params["max_tokens"] == nil && params["temperature"] == nil && params["stream"] == nil)
    }
}
@Test(arguments: [("setting", 422), ("settings", 422), ("code_consent", 409), ("mode", 400)]) func settingAndConsentFailuresHaveUsefulHTTPStatuses(_ code: String, _ status: Int) {
    let failure = WorkerFailure(code: code, message: "Rejected")
    #expect(workerHTTPStatus(failure) == status)
    #expect(serviceHTTPFailure(failure).status == status)
}

@Test func openAIImageSizeReachesSavedResolutionThroughRealListener() async throws {
    let probe = RequestProbe()
    let service = RecordingService(kind: "images", probe: probe, saved: ["resolution": "512x512"])
    let server = ArsenalAPIServer(service: service, supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/images-\(UUID().uuidString)"), port: 0)
    let port = try await server.start(); defer { server.stop() }
    #expect(try await send(port: port, path: "/v1/images/generations", body: ["model": "model", "prompt": "test"]) == 200)
    let omitted = await probe.last()
    #expect(omitted["resolution"] == "512x512" && omitted["width"] == nil && omitted["height"] == nil)
    #expect(try await send(port: port, path: "/v1/images/generations", body: ["model": "model", "prompt": "test", "size": "768x512"]) == 200)
    #expect(await probe.last()["width"] == 768)
    let router = APIRouter(service: service, support: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/replicate-settings"))
    let directory = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/replicate-settings")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let params = try await router.replicateImageParams(["prompt": "test"], dir: directory)
    _ = try await service.run(id: "model", op: "generate", params: params, sink: nil)
    let replicate = await probe.last()
    #expect(replicate["resolution"] == "512x512" && replicate["width"] == nil && replicate["height"] == nil)
}

/// FLUX 3 Klein's worker answers `files` (schema 3), FLUX.2's `images`; the OpenAI images route read only `images` and
/// answered "Runner returned no images" for Klein (1 Nov).
@Test(arguments: ["images", "files"]) func openAIImagesReadEitherOutputShape(_ key: String) async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/shape-\(UUID().uuidString)")
    let router = APIRouter(service: StubModelsService(rows: []), support: support)
    let job = await router.jobsRoot.appendingPathComponent("job-1")
    try FileManager.default.createDirectory(at: job, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: support) }
    let png = Data([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 0])
    let file = job.appendingPathComponent("image-1.png")
    try png.write(to: file)
    let output = try await router.base64Images([key: .array([["path": .string(file.path)]])])
    #expect(output == [png.base64EncodedString()])
}
extension APIRouter {
    fileprivate func base64Images(_ result: JSONObject) throws -> [String?] { try imageOutput(result, format: "b64_json").map { $0["b64_json"] as? String } }
}

@Test func settingsPOSTIsAtomicAndApprovalSourceIsVisible() async throws {
    let row = StubModelsService.localFixture
    let stub = StubModelsService(
        rows: [row],
        metadata: [
            row.id: [
                "setting": [
                    ["id": "a", "type": "number", "default": 1, "min": 0, "max": 10, "applies": "request"],
                    ["id": "b", "type": "number", "default": 2, "min": 0, "max": 10, "applies": "request"]
                ]
            ]
        ])
    let server = ArsenalAPIServer(service: stub, supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/settings-\(UUID().uuidString)"), port: 0)
    let port = try await server.start(); defer { server.stop() }
    #expect(try await send(port: port, path: "/arsenal/v1/models/\(row.id)/settings", body: ["values": ["a": 7, "b": 99]]) == 422)
    #expect(try await stub.settings(id: row.id).map(\.value) == [1, 2])
    #expect(try await send(port: port, path: "/arsenal/v1/models/\(row.id)/settings", body: ["values": ["a": 7, "b": 9]]) == 200)
    for source in ["app", "unknown"] {
        #expect(
            try await send(port: port, path: "/arsenal/v1/models/\(row.id)/approve", body: ["content_digest": .string(row.contentDigest!), "approved_by": .string(source)]) == 422)
    }
    #expect(try await send(port: port, path: "/arsenal/v1/models/\(row.id)/approve", body: ["content_digest": .string(row.contentDigest!)]) == 200)
    // Not downloaded, so off the table: the catalog shows who approved its code.
    #expect(await stub.catalog().first { $0.id == row.id }?.row.approval?.by == .api)
    #expect(try await send(port: port, path: "/arsenal/v1/models/\(row.id)/approve", body: ["content_digest": .string(row.contentDigest!), "approved_by": "cli"]) == 200)
    #expect(await stub.catalog().first { $0.id == row.id }?.row.approval?.by == .cli)
}

@Test func voiceDeletionRoutesRemoveOnlySavedVoicesAndLeaveReceipt() async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/voices-\(UUID().uuidString)")
    let directory = support.appendingPathComponent("voices/model")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let service = RecordingService(kind: "voices", probe: RequestProbe(), voiceSupport: support)
    let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
    let port = try await server.start(); defer { server.stop() }
    for universal in [false, true] {
        let id = "voice-" + UUID().uuidString.lowercased()
        let reference = directory.appendingPathComponent(id + ".wav"), manifest = directory.appendingPathComponent(id + ".json")
        try Data([1, 2, 3]).write(to: reference)
        try WireJSON.encode(["id": .string(id), "reference": .string(reference.path)]).write(to: manifest)
        let code: Int
        if universal {
            code = try await send(port: port, path: "/arsenal/v1/models/model/ops/delete_voice", body: ["voice_id": .string(id)])
        } else {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/providers/elevenlabs/v1/voices/" + id)!)
            request.httpMethod = "DELETE"
            let (_, response) = try await URLSession.shared.data(for: request)
            code = (response as? HTTPURLResponse)?.statusCode ?? 0
        }
        #expect(code == 200)
        #expect(!FileManager.default.fileExists(atPath: reference.path) && !FileManager.default.fileExists(atPath: manifest.path))
    }
    var preset = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/providers/elevenlabs/v1/voices/af_heart")!); preset.httpMethod = "DELETE"
    let (_, response) = try await URLSession.shared.data(for: preset)
    #expect((response as? HTTPURLResponse)?.statusCode == 422)
    #expect(try await send(port: port, path: "/arsenal/v1/models/model/ops/delete_voice", body: ["voice_id": "af_heart"]) == 422)
    let missing = "voice-" + UUID().uuidString.lowercased()
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/providers/elevenlabs/v1/voices/" + missing)!); request.httpMethod = "DELETE"
    let (_, absent) = try await URLSession.shared.data(for: request)
    #expect((absent as? HTTPURLResponse)?.statusCode == 404)
}

/// Core review 6: HTTP settings take dtype, keep_hot, show_in_menu and runner as MCP's arsenal_settings does, with the
/// same checks; a refused key applies nothing.
@Test func httpSettingsTakeTheReservedKeysMCPTakes() async throws {
    let declaration: JSONObject = ["id": "speed", "type": "number", "title": "Speed", "default": 1, "min": 0.5, "max": 2, "applies": "request", "maps_to": "request.speed"]
    var row = ModelRow(id: "voice", name: "Voice", kind: "voices", precisions: ["fp16", "int8"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
    row.runner = "native"
    let stub = StubModelsService(rows: [row], metadata: ["voice": ["setting": [.object(declaration)]]])
    let server = ArsenalAPIServer(service: stub, supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/reserved-\(UUID().uuidString)"), port: 0)
    let port = try await server.start(); defer { server.stop() }
    let path = "/arsenal/v1/models/voice/settings"
    for refused: JSONObject in [["speed": 1.5, "dtype": "q3"], ["speed": 1.5, "keep_hot": "yes"], ["speed": 1.5, "show_in_menu": 1], ["speed": 1.5, "runner": "nope"]] {
        #expect(try await send(port: port, path: path, body: ["values": .object(refused)]) == 422, "\(refused)")
    }
    let untouched = try #require(await stub.rows().first)
    #expect(try await stub.settings(id: "voice").first?.value == 1 && untouched.selectedPrecision == "fp16" && !untouched.keepHot && untouched.toolsShown)
    #expect(try await send(port: port, path: path, body: ["values": ["speed": 1.5, "dtype": "int8", "keep_hot": true, "show_in_menu": false]]) == 200)
    let saved = try #require(await stub.rows().first)
    #expect(saved.selectedPrecision == "int8" && saved.keepHot && !saved.toolsShown)
    #expect(try await stub.settings(id: "voice").first?.value == 1.5)
}

/// Review of core review 6: a reserved key the model's state refuses after its own settings were written (a Get runs)
/// fails with what did apply, as MCP's answer says it.
@Test func httpSettingsSayWhatAppliedWhenAReservedKeyIsRefusedLater() async throws {
    let declaration: JSONObject = ["id": "speed", "type": "number", "title": "Speed", "default": 1, "min": 0.5, "max": 2, "applies": "request", "maps_to": "request.speed"]
    var row = ModelRow(
        id: "voice", name: "Voice", kind: "voices", precisions: ["fp16", "int8"], states: ["fp16": .init(state: .ready), "int8": .init(state: .downloading)],
        selectedPrecision: "fp16")
    row.runner = "native"
    let stub = StubModelsService(rows: [row], metadata: ["voice": ["setting": [.object(declaration)]]])
    let server = ArsenalAPIServer(service: stub, supportDirectory: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-d/partial-\(UUID().uuidString)"), port: 0)
    let port = try await server.start(); defer { server.stop() }
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models/voice/settings")!)
    request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try WireJSON.encode(["values": ["speed": 1.5, "dtype": "int8"]])
    let (data, response) = try await URLSession.shared.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 409)
    let body = try WireJSON.decode(data)
    #expect(body["error"]?.string?.hasPrefix("Applied speed; dtype was not: Wait for Get or Load") == true, "\(body)")
}
