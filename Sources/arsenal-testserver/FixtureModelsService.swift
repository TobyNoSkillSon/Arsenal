// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// Transport fixtures over the real StubModelsService lifecycle. No model weights, Python or GPU.
actor FixtureModelsService: ModelsService {
    let stub: StubModelsService
    var canceledRuns = 0
    var cancelledGets = 0
    var getTransfer: Task<Void, Error>?
    let golden: [GoldenDecisionFixture]
    struct GoldenDecisionFixture: Decodable, Sendable { let model: String; let requestJSON: String; let frameJSON: String }
    init() {
        let path = ProcessInfo.processInfo.environment["ARSENAL_DECISION_FIXTURES"]
        golden = path.flatMap { try? Data(contentsOf: URL(fileURLWithPath: $0)) }.flatMap { try? JSONDecoder().decode([GoldenDecisionFixture].self, from: $0) } ?? []
        var rows =
            ["voices", "decisions", "images", "chat", "sound", "enhancement", "audio", "embeddings", "perception", "restoration", "geometry", "language"].map { kind in
                ModelRow(
                    id: "stub-" + kind, name: kind == "chat" ? "Qwen3.8-27B (stub)" : "Stub " + kind, kind: kind, licence: "Apache-2.0",
                    states: ["bf16": PrecisionState(state: .ready)])
            } + [ModelRow(id: "stub-absent", name: "Absent", kind: "chat", downloadBytes: ["bf16": 0]), StubModelsService.localFixture]
        for index in rows.indices where rows[index].kind == "chat" {
            rows[index].metrics["context"] = .number(262144); rows[index].metrics["native_context"] = .number(262144)
        }
        for model in Set(golden.map(\.model)).sorted() {
            rows.append(ModelRow(id: model, name: model + " (golden fixture)", kind: "decisions", states: ["fp16": PrecisionState(state: .ready)], selectedPrecision: "fp16"))
        }
        stub = StubModelsService(
            rows: rows,
            metadata: [
                "stub-chat": [
                    "setting": [
                        [
                            "id": "context_limit", "title": "Context", "type": "choice", "default": 262144, "choices": [8192, 262144], "applies": "load",
                            "maps_to": "config.context_limit"
                        ], ["id": "thinking", "title": "Thinking", "type": "toggle", "default": true, "applies": "request", "maps_to": "request.thinking"]
                    ]
                ]
            ])
    }
    func rows() async -> [ModelRow] {
        var rows = await stub.rows()
        for i in rows.indices {
            rows[i].metrics["canceled_runs"] = .number(Double(canceledRuns)); rows[i].metrics["cancelled_gets"] = .number(Double(cancelledGets))
            rows[i].metrics["get_active"] = .bool(getTransfer != nil)
        }; return rows
    }
    func saveDesignedVoice(id: String, previewID: String, name: String) async throws -> JSONObject { ["voice_id": "cloned-stub"] }
    func approveLocalCode(id: String, contentDigest: String) async throws { try await stub.approveLocalCode(id: id, contentDigest: contentDigest) }
    func catalog(query: CatalogQuery) async -> [CatalogEntry] { await stub.catalog(query: query) }
    func getPlan(id: String, dtype: String) async throws -> GetPlan { try await (stub as any ModelsService).getPlan(id: id, dtype: dtype) }
    func refresh() async throws { try await stub.refresh() }
    func add(id: String) async throws { try await stub.add(id: id) }
    func remove(id: String, deleteWeights: Bool) async throws { try await stub.remove(id: id, deleteWeights: deleteWeights) }
    func settings(id: String) async throws -> [SettingState] { try await stub.settings(id: id) }
    func setSettings(id: String, values: JSONObject) async throws { try await stub.setSettings(id: id, values: values) }
    func approveLocalCode(id: String, contentDigest: String, approvedBy: ApprovalSource) async throws {
        try await stub.approveLocalCode(id: id, contentDigest: contentDigest, approvedBy: approvedBy)
    }
    func setSetting(id: String, key: String, value: JSONValue) async throws { try await stub.setSetting(id: id, key: key, value: value) }
    func applySettings(id: String) async throws { try await stub.applySettings(id: id) }
    func kinds() async -> [KindSpec] { await stub.kinds() }
    func events() async -> AsyncStream<ServiceEvent> { await stub.events() }
    func get(id: String, precision: String) async throws {
        if id != "stub-absent" { try await stub.get(id: id, precision: precision); return }
        let transfer = Task {
            try await Task.sleep(for: .seconds(30)); try await stub.get(id: id, precision: precision)
        }
        getTransfer = transfer
        defer { getTransfer = nil }
        try await transfer.value
    }
    func cancelGet(id: String) async {
        if id == "stub-absent", let transfer = getTransfer { transfer.cancel(); cancelledGets += 1 }
        await stub.cancelGet(id: id)
    }
    func load(id: String, precision: String?, fast: Bool?) async throws { try await stub.load(id: id, precision: precision, fast: fast) }
    func unload(id: String) async throws { try await stub.unload(id: id) }
    func setPrecision(id: String, precision: String) async throws { try await stub.setPrecision(id: id, precision: precision) }
    func setKeepHot(id: String, on: Bool) async throws { try await stub.setKeepHot(id: id, on: on) }
    func delete(id: String) async throws { try await stub.delete(id: id) }
    func onDiskModels() async throws -> [OnDiskModel] { try await stub.onDiskModels() }
    func deletePrecision(id: String, precision: String) async throws -> Int64 { try await stub.deletePrecision(id: id, precision: precision) }
    func deleteAllWeights(id: String) async throws -> Int64 { try await stub.deleteAllWeights(id: id) }
    func orphanBytes() async throws -> Int64 { try await stub.orphanBytes() }
    func deleteOrphans() async throws -> Int64 { try await stub.deleteOrphans() }
    func menuActions() async -> [LoadedAction] { await stub.menuActions() }
    func memory() async -> MemorySnapshot { await stub.memory() }
    func catalogStatus() async -> CatalogStatus { await stub.catalogStatus() }
    func runRaw(id: String, op: String, params: JSONObject) async throws -> Data {
        if let body = params["body"]?.string, let fixture = golden.first(where: { $0.model == id && Data($0.requestJSON.utf8) == Data(body.utf8) }) {
            try await stub.load(id: id, precision: nil, fast: nil)
            return Data(fixture.frameJSON.utf8)
        }
        var bodyParams = params
        if let body = params["body"]?.string, let parsed = try? WireJSON.decode(Data(body.utf8)) { bodyParams = parsed }
        if bodyParams["model"]?.string == "auto" {
            func target(_ state: JSONValue) -> String {
                let text = state.string ?? String(decoding: (try? JSONEncoder().encode(state)) ?? Data(), as: UTF8.self)
                let letters = text.unicodeScalars.filter { CharacterSet.letters.contains($0) }
                let ascii = letters.filter { $0.isASCII }.count
                return letters.isEmpty || Double(ascii) / Double(letters.count) >= 0.995 ? "laya-english" : "laya-multilingual"
            }
            if op == "decide", let state = bodyParams["state"], target(state) != id {
                return try WireJSON.encode(["ok": false, "error": ["code": "wrong_model", "model": .string(target(state))]])
            }
            if op == "judge", let items = bodyParams["items"]?.array {
                var routes: [String: [JSONValue]] = [:]
                for (index, item) in items.enumerated() { routes[target(item), default: []].append(.number(Double(index))) }
                if routes.keys.contains(where: { $0 != id }) {
                    return try WireJSON.encode(["ok": false, "error": ["code": "wrong_model", "routes": .object(routes.mapValues(JSONValue.array))]])
                }
            }
        }
        if bodyParams["raw_order_probe"]?.bool == true, let body = params["body"]?.string {
            return try WireJSON.encode(["ok": true, "model": .string(id), "request_bytes": .string(body)])
        }
        if let bits = bodyParams["bits"]?.number, bits != 0 && bits != 16 {
            return try WireJSON.encode([
                "ok": false, "error": ["code": "conflict", "status": 409, "body": ["detail": ["error_type": "conflict_error", "message": "Stub is loaded at 16-bit."]]]
            ])
        }
        return try WireJSON.encode(try await run(id: id, op: op, params: params, sink: nil))
    }
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject {
        var params = params
        if let body = params["body"]?.string, let parsed = try? WireJSON.decode(Data(body.utf8)) { params = parsed }
        do {
            if let declaration = await operations(id: id).first(where: { $0.name == op }) {
                try OperationSchemaValidator.validate(.object(params), schema: .object(declaration.inputSchema))
            }
            try await stub.load(id: id, precision: nil, fast: nil)
            if let capability = try await fixtureCapability(op, params: params, sink: sink) { return capability }
            switch op {
            case "voices": return ["ok": true, "voices": [["id": "af_heart", "name": "Heart", "language": "en", "gender": "female"]]]
            case "speak":
                guard (params["voice"]?.string ?? "af_heart") == "af_heart" else { throw WorkerFailure(code: "invalid_request", message: "Unknown voice; use af_heart.") }
                let pcm = Data(repeating: 0, count: 9600)
                try await sink?(.json(["event": "audio_begin", "stream": 1, "sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"]))
                for i in stride(from: 0, to: pcm.count, by: 1024) { try await sink?(.binary(stream: 1, bytes: Data(pcm[i..<min(pcm.count, i + 1024)]))) }
                if let out = params["out_dir"]?.string {
                    let file = URL(fileURLWithPath: out).appendingPathComponent("speech.wav")
                    try fixtureWAV(pcm, rate: 24000, channels: 1).write(to: file)
                    return ["ok": true, "path": .string(file.path), "duration_s": 0.2]
                }
                return ["ok": true, "duration_s": 0.2, "sample_rate": 24000, "channels": 1]
            case "generate_sound", "generate_music", "enhance":
                let prompt = params["prompt"]?.string ?? ""
                if prompt.contains("progress") {
                    try await sink?(.json(["event": "progress", "phase": "generating", "step": 7, "total": 9]))
                    try await Task.sleep(for: .milliseconds(30))
                    try await sink?(.json(["event": "progress", "phase": "generating", "step": 8, "total": 9]))
                }
                if prompt.contains("slow") { try await Task.sleep(for: .seconds(2)) }
                guard let out = params["out_dir"]?.string else { throw WorkerFailure(code: "invalid_request", message: "Missing output directory.") }
                if op == "enhance" {
                    guard let audio = params["audio"]?["path"]?.string, FileManager.default.fileExists(atPath: audio) else {
                        throw WorkerFailure(code: "invalid_request", message: "Missing input audio.")
                    }
                }
                let file = URL(fileURLWithPath: out).appendingPathComponent("audio.wav")
                let channels = op == "enhance" ? 1 : 2, rate = op == "enhance" ? 24000 : 44100
                try fixtureWAV(Data(repeating: 0, count: rate * channels * 2 / 5), rate: rate, channels: channels).write(to: file)
                if op == "generate_music" { return ["files": [["path": .string(file.path), "mime": "audio/wav"]], "seconds": 0.04] }
                return ["ok": true, "path": .string(file.path), "sample_rate": .number(Double(rate)), "channels": .number(Double(channels)), "seconds": 0.04]
            case "design_voice":
                guard let out = params["out_dir"]?.string else { throw WorkerFailure(code: "invalid_request", message: "Missing output directory.") }
                let file = URL(fileURLWithPath: out).appendingPathComponent("preview.wav")
                try fixtureWAV(Data(repeating: 0, count: 9600), rate: 24000, channels: 1).write(to: file)
                return ["voice_id": "designed-stub", "preview": ["path": .string(file.path)]]
            case "clone_voice": return ["voice_id": "cloned-stub"]
            case "generate":
                let prompt = params["prompt"]?.string ?? ""
                if prompt.contains("restart") {
                    try await Task.sleep(for: .seconds(10))
                } else if prompt.contains("slow") {
                    try await Task.sleep(for: .seconds(2))
                } else {
                    try await Task.sleep(for: .milliseconds(40))
                }
                guard let directory = params["out_dir"]?.string else { throw WorkerFailure(code: "invalid_request", message: "Missing output directory.") }
                let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j6ioAAAAASUVORK5CYII=")!
                let count = Int(params["n"]?.number ?? 1)
                var images: [JSONValue] = []
                for i in 0..<count {
                    let path = URL(fileURLWithPath: directory).appendingPathComponent("image-\(i).png").path
                    try png.write(to: URL(fileURLWithPath: path))
                    images.append(["path": .string(path), "bytes": .number(Double(png.count)), "width": 1, "height": 1])
                }
                return ["ok": true, "images": .array(images), "seed": params["seed"] ?? 1, "seconds": 0.04]
            case "count_tokens": return ["ok": true, "tokens": 8, "input_tokens": 8]
            case "chat":
                let messages = params["messages"]?.array ?? []
                let hasResult = messages.contains { $0["role"]?.string == "tool" }
                if let tools = params["tools"]?.array, !tools.isEmpty, !hasResult, params["tool_choice"]?.string != "none" {
                    let name = tools[0]["function"]?["name"]?.string ?? "get_weather"
                    return [
                        "ok": true, "tool_calls": [["id": "call_local", "type": "function", "function": ["name": .string(name), "arguments": "{\"city\":\"Łódź\"}"]]],
                        "finish_reason": "tool_calls", "usage": ["prompt_tokens": 8, "completion_tokens": 6]
                    ]
                }
                let slow = messages.contains { $0["content"]?.string?.contains("slow") == true }
                let text = params["response_format"]?["type"]?.string == "json_schema" ? "{\"name\":\"fixture\"}" : "Hello from Arsenal. Łódź ✨"
                let reasoning = params["thinking"]?.bool == true ? "I will answer briefly." : nil
                if reasoning != nil {
                    for part in ["I will ", "answer briefly."] { try await sink?(.json(["event": "reasoning", "text": .string(part)])) }
                }
                for chunk in slow ? Array(repeating: "waiting ", count: 100) : params["response_format"] == nil ? ["Hello ", "from Arsenal. ", "Łódź ", "✨"] : [text] {
                    if slow { try await Task.sleep(for: .milliseconds(100)) }
                    try await sink?(.json(["event": "token", "text": .string(chunk)]))
                }
                var result: JSONObject = ["ok": true, "text": .string(text), "finish_reason": "stop", "usage": ["prompt_tokens": 8, "completion_tokens": 6]]
                if let reasoning { result["reasoning"] = .string(reasoning) }
                result["timing"] = ["ttft_s": 0.02, "prefill_tps": 400, "decode_tps": 25, "max_itl_ms": 50, "peak_mb": 42]
                return result
            case "decide":
                return ["answers": .object(answers(params["questions"]?.object ?? [:], legacy: false)), "model": .string(id), "usage": ["input_tokens": 218, "output_tokens": 0]]
            case "judge":
                let questions = params["questions"]?.object ?? [:]
                let result: [JSONValue] = (params["items"]?.array ?? []).enumerated().map { index, item in
                    if item["image"] != nil { return ["index": .number(Double(index)), "ok": false, "error": "Decision models accept text only."] }
                    return ["index": .number(Double(index)), "ok": true, "answers": .object(answers(questions, legacy: true)), "model": .string(id), "ms": 1]
                }
                return ["results": .array(result)]
            default: throw WorkerFailure(code: "unsupported_op", message: "Unsupported stub operation \(op).")
            }
        } catch {
            if error is CancellationError || Task.isCancelled { canceledRuns += 1 }
            throw error
        }
    }
    private func answers(_ questions: JSONObject, legacy: Bool) -> JSONObject {
        questions.mapValues { question in
            let type = question["type"]?.string ?? "noul"
            var answer: JSONObject = [:]
            switch type {
            case "noul": answer = ["noul": 0.75]; if legacy { answer["confidence"] = 0.75 }
            case "choice":
                let keys = question["criteria"]?.object?.keys.sorted() ?? question["criteria"]?.array?.compactMap(\.string) ?? ["other"]
                let probabilities = Dictionary(
                    uniqueKeysWithValues: keys.enumerated().map { i, key in (key, JSONValue.number(keys.count == 1 ? 1 : i == 0 ? 0.75 : 0.25 / Double(keys.count - 1))) })
                answer = ["choice": .string(keys[0]), "confidence": 0.5, "probabilities": .object(probabilities)]
            case "score":
                let levels = question["criteria"]?.array ?? ["low", "high"]
                let p = 1.0 / Double(levels.count)
                answer = [
                    "score": .number(Double(levels.count - 1) / 2), "confidence": 0,
                    "probabilities": .object(Dictionary(uniqueKeysWithValues: levels.indices.map { (String($0), .number(p)) }))
                ]
                if !legacy { answer["legend"] = .object(Dictionary(uniqueKeysWithValues: levels.enumerated().map { (String($0.offset), $0.element) })) }
            default: break
            }
            if !legacy { answer["type"] = .string(type) }; return .object(answer)
        }
    }
}

private func fixtureWAV(_ data: Data, rate: Int, channels: Int) -> Data {
    var wav = Data("RIFF".utf8)
    func u32(_ value: Int) { var n = UInt32(value).littleEndian; withUnsafeBytes(of: &n) { wav.append(contentsOf: $0) } }
    func u16(_ value: Int) { var n = UInt16(value).littleEndian; withUnsafeBytes(of: &n) { wav.append(contentsOf: $0) } }
    u32(data.count + 36); wav += Data("WAVEfmt ".utf8); u32(16); u16(1); u16(channels); u32(rate); u32(rate * channels * 2); u16(channels * 2); u16(16); wav += Data("data".utf8)
    u32(data.count); wav += data
    return wav
}
