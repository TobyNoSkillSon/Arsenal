// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// The tools' only way into Arsenal: the app's own loopback HTTP API, used exactly as any other client uses it, named
// "Arsenal" (X-Arsenal-Client) so the activity view says who is asking. A tool never holds the service.

/// A failed API call, worded for the tool's error line.
struct ToolError: LocalizedError, Equatable {
    var message: String
    var status: Int?
    /// The service's error code (`code_consent`, `not_installed`, …) when the answer named one.
    var code: String?
    var details: JSONObject?
    var errorDescription: String? { message }
    init(_ message: String, status: Int? = nil, code: String? = nil) { self.message = message; self.status = status; self.code = code }

    /// The message an API error body carries (OpenAI `{error:{message}}`, control `{error}` or `{message}`).
    static func from(status: Int, body: Data) -> ToolError {
        let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        let error = object?["error"]
        let text =
            (error as? [String: Any])?["message"] as? String ?? error as? String ?? object?["message"] as? String
            ?? String(data: body.prefix(300), encoding: .utf8).flatMap { $0.isEmpty ? nil : $0 }
        let code = object?["code"] as? String ?? (error as? [String: Any])?["code"] as? String
        var result = ToolError(text ?? "Arsenal answered \(status).", status: status, code: code)
        if let encoded = object?["details"], let data = try? JSONSerialization.data(withJSONObject: encoded) {
            result.details = try? WireJSON.decode(data)
        }
        return result
    }
}

struct ArsenalClient: Sendable {
    let port: Int
    static let clientName = "Arsenal"
    var origin: String { "http://127.0.0.1:\(port)" }

    private static let session: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 600; c.timeoutIntervalForResource = 3600
        c.httpMaximumConnectionsPerHost = 8; c.urlCache = nil
        return URLSession(configuration: c)
    }()

    func request(_ method: String, _ path: String, json: Any? = nil, timeout: TimeInterval = 600, accept: String? = nil, headers: [String: String] = [:]) throws -> URLRequest {
        guard let url = URL(string: origin + path) else { throw ToolError("Bad API path \(path).") }
        var r = URLRequest(url: url, timeoutInterval: timeout)
        r.httpMethod = method
        r.setValue("127.0.0.1:\(port)", forHTTPHeaderField: "Host")
        r.setValue(Self.clientName, forHTTPHeaderField: "X-Arsenal-Client")
        if let accept { r.setValue(accept, forHTTPHeaderField: "Accept") }
        for (name, value) in headers { r.setValue(value, forHTTPHeaderField: name) }
        if LoadMemoryContext.force { r.setValue("true", forHTTPHeaderField: "X-Arsenal-Force-Load") }
        if let json {
            r.httpBody = try JSONSerialization.data(withJSONObject: json)
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return r
    }

    /// All tool requests share this refusal/presenter/retry hook. Force is transport metadata, never model input.
    private func withMemoryApproval<T>(_ request: URLRequest, send: (URLRequest) async throws -> T) async throws -> T {
        if MemoryLoadPresenter.ownsRequest { return try await send(request) }
        return try await MemoryLoadPresenter.perform { force in
            var request = request
            if force { request.setValue("true", forHTTPHeaderField: "X-Arsenal-Force-Load") }
            return try await send(request)
        }
    }

    /// One request; a non-2xx answer throws its message.
    func send(
        _ method: String, _ path: String, json: Any? = nil, timeout: TimeInterval = 600, accept: String? = nil, headers: [String: String] = [:]
    ) async throws -> (Data, HTTPURLResponse) {
        let (data, response): (Data, URLResponse)
        do {
            let built = try request(method, path, json: json.map(Self.plain), timeout: timeout, accept: accept, headers: headers)
            (data, response) = try await withMemoryApproval(built) { request in
                let (data, response) = try await Self.session.data(for: request)
                if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                    throw ToolError.from(status: http.statusCode, body: data)
                }
                return (data, response)
            }
        } catch let error as ToolError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            if (error as? URLError)?.code == .cancelled { throw CancellationError() }
            throw ToolError("Arsenal's API did not answer: \(error.localizedDescription)")
        }
        guard let http = response as? HTTPURLResponse else { throw ToolError("Arsenal's API sent no HTTP answer.") }
        guard (200..<300).contains(http.statusCode) else { throw ToolError.from(status: http.statusCode, body: data) }
        return (data, http)
    }

    /// A JSONObject body becomes plain Foundation values for JSONSerialization.
    private static func plain(_ value: Any) -> Any {
        guard let object = value as? JSONObject, let data = try? JSONEncoder().encode(object), let plain = try? JSONSerialization.jsonObject(with: data) else { return value }
        return plain
    }

    func object(_ path: String) async throws -> Any { try JSONSerialization.jsonObject(with: try await send("GET", path).0) }

    /// A model's settings as the table shows them (`context_limit`, `voice`, `resolution`, …).
    func settings(_ model: String) async throws -> [SettingState] {
        try JSONDecoder().decode([SettingState].self, from: try await send("GET", "/arsenal/v1/models/\(model)/settings").0)
    }
    func setSettings(_ model: String, _ values: [String: Any]) async throws {
        _ = try await send("POST", "/arsenal/v1/models/\(model)/settings", json: ["values": values, "apply": false])
    }

    /// C5: the best measured reference for this model, dtype and mode; nil when none exists.
    func reference(_ model: String, dtype: String, mode: String) async throws -> ToolReference? {
        let path = "/arsenal/v1/models/\(model)/reference?dtype=\(dtype)&mode=\(mode)"
        let data = try await send("GET", path).0
        guard let object = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) as? [String: Any] else { return nil }
        return ToolReference(object)
    }

    /// The input schema a model declares for an operation, from the API's OpenAPI document.
    func inputSchema(_ model: String, op: String) async throws -> [String: Any]? {
        guard let doc = try await object("/arsenal/v1/openapi.json") as? [String: Any] else { return nil }
        let post = ((doc["paths"] as? [String: Any])?["/arsenal/v1/models/\(model)/ops/\(op)"] as? [String: Any])?["post"] as? [String: Any]
        let schema = ((((post?["requestBody"] as? [String: Any])?["content"] as? [String: Any])?["application/json"] as? [String: Any])?["schema"] as? [String: Any])
        guard let ref = schema?["$ref"] as? String, ref.hasPrefix("#/components/schemas/") else { return schema }
        return ((doc["components"] as? [String: Any])?["schemas"] as? [String: Any])?[String(ref.dropFirst("#/components/schemas/".count))] as? [String: Any]
    }

    /// A model's operations as the service resolved them, from the API's OpenAPI document (one path per operation:
    /// title, input schema, output, stream; output schema, policy and annotations where the descriptor carries them).
    func operations(_ model: String) async throws -> [OperationDescriptor] {
        let data = try await send("GET", "/arsenal/v1/openapi.json").0
        guard case .object(let doc) = try JSONDecoder().decode(JSONValue.self, from: data) else { throw ToolError("Arsenal's API description is not an object.") }
        return Self.operations(model, openAPI: doc)
    }

    static func operations(_ model: String, openAPI doc: JSONObject) -> [OperationDescriptor] {
        let prefix = "/arsenal/v1/models/\(model)/ops/"
        let components = doc["components"]?["schemas"]?.object ?? [:]
        /// A component schema back in its own document's terms: the API rebased its local refs onto the components.
        func schema(_ value: JSONValue?) -> JSONObject? {
            guard var object = value?.object else { return nil }
            if let ref = object["$ref"]?.string, ref.hasPrefix("#/components/schemas/") {
                let id = String(ref.dropFirst("#/components/schemas/".count))
                guard let target = components[id]?.object else { return nil }
                func unbase(_ v: JSONValue) -> JSONValue {
                    switch v {
                    case .array(let items): return .array(items.map(unbase))
                    case .object(var o):
                        if let r = o["$ref"]?.string, r.hasPrefix("#/components/schemas/" + id) { o["$ref"] = .string("#" + r.dropFirst(("#/components/schemas/" + id).count)) }
                        return .object(o.mapValues(unbase))
                    default: return v
                    }
                }
                object = unbase(.object(target)).object ?? target
            }
            return object
        }
        var ops: [OperationDescriptor] = []
        for (path, item) in doc["paths"]?.object ?? [:] where path.hasPrefix(prefix) {
            guard let post = item["post"]?.object, let input = schema(post["requestBody"]?["content"]?["application/json"]?["schema"]) else { continue }
            let name = post["x-arsenal-op"]?.string ?? String(path.dropFirst(prefix.count))
            let content = post["responses"]?["200"]?["content"]?.object ?? [:]
            let output = post["x-arsenal-output"]?.string ?? Self.output(Set(content.keys))
            var outputSchema = schema(post["x-arsenal-output-schema"]) ?? schema(content["application/json"]?["schema"])
            if outputSchema == ["type": "object"] { outputSchema = nil }
            let policy = post["x-arsenal-policy"]?.object
            ops.append(
                OperationDescriptor(
                    name: name, title: post["summary"]?.string ?? OperationForm.title(name), inputSchema: input, outputSchema: outputSchema, output: output,
                    stream: post["x-arsenal-stream"]?.bool ?? false, annotations: post["x-arsenal-annotations"]?.object ?? [:],
                    timeout: (policy?["timeout_s"]?.number).flatMap { $0.isFinite && $0 > 0 ? $0 : nil }))
        }
        return ops.sorted { ($0.title.lowercased(), $0.name) < ($1.title.lowercased(), $1.name) }
    }

    /// The declared output from the response types the API lists for it.
    static func output(_ types: Set<String>) -> String {
        // File outputs first: a streamed file operation also lists event streams for its progress.
        if types.contains("image/png") { return "image" }
        if types.contains("audio/wav") { return "audio" }
        if types.contains("application/octet-stream") || types.contains("multipart/mixed") { return "file" }
        if types.contains("text/event-stream") { return "text-stream" }
        return "json"
    }

    /// Server-sent events from a GET or POST: each `data:` payload's bytes (comments and `[DONE]` skipped); `object`
    /// reads one as JSON.
    /// `requireDone`: the stream must end with `[DONE]`; ending without it throws (a reply cut off midway).
    func events(_ method: String, _ path: String, json: Any? = nil, requireDone: Bool = false) -> AsyncThrowingStream<Data, Error> {
        let built: URLRequest
        do { built = try request(method, path, json: json, accept: "text/event-stream") } catch { return AsyncThrowingStream { $0.finish(throwing: error) } }
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await withMemoryApproval(built) { request in
                        let (bytes, response) = try await Self.session.bytes(for: request)
                        guard let http = response as? HTTPURLResponse else { throw ToolError("Arsenal's API sent no HTTP answer.") }
                        guard (200..<300).contains(http.statusCode) else {
                            var body = Data()
                            for try await byte in bytes { body.append(byte); if body.count > 65_536 { break } }
                            throw ToolError.from(status: http.statusCode, body: body)
                        }
                        var done = false
                        for try await line in bytes.lines {
                            guard line.hasPrefix("data:") else { continue }
                            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                            if payload == "[DONE]" { done = true; break }
                            let data = Data(payload.utf8)
                            let error = ToolError.from(status: 409, body: data)
                            if error.code == "insufficient_memory", error.details != nil { throw error }
                            continuation.yield(data)
                        }
                        if requireDone, !done, !Task.isCancelled { throw ToolError("The reply stopped before it finished.") }
                    }
                    continuation.finish()
                } catch {
                    if error is CancellationError || Task.isCancelled || (error as? URLError)?.code == .cancelled { continuation.finish(throwing: CancellationError()); return }
                    continuation.finish(throwing: error is ToolError ? error : ToolError("Arsenal's API did not answer: \(error.localizedDescription)"))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// One event's JSON object (nil for anything else).
func eventObject(_ data: Data) -> [String: Any]? { (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] }

/// The workload a reference suite measured, so a run is compared only with the same kind of work.
enum ReferenceWorkload: Equatable {
    /// One-line prompts (the chat suite): comparable with a typed prompt of up to 256 tokens, never a preset.
    case shortPrompt
    /// Images of this size and step count.
    case image(width: Int, height: Int, steps: Int)
    static func of(_ suite: String) -> ReferenceWorkload? {
        switch suite {
        case "arsenal-chat-suite-v1", "mobius-chat-suite-v1": return .shortPrompt // legacy-compat: suite ids recorded before the rename
        case "arsenal-images-suite-v1", "mobius-images-suite-v1": return .image(width: 512, height: 512, steps: 4) // legacy-compat: suite ids recorded before the rename
        default: return nil
        }
    }
    var title: String {
        switch self {
        case .shortPrompt: return "short prompt"
        case .image(let w, let h, let steps): return "\(w) \u{00d7} \(h) \u{00b7} \(steps) steps"
        }
    }
}

/// C5's answer: the measured figures for this model and where they were measured.
struct ToolReference: Equatable {
    enum Match: String { case sameChip = "same_chip", sameFamily = "same_family", nearest }
    var match: Match
    var chip: String
    var date: String?
    var metrics: [String: Double]
    /// The benchmark suite the figures come from (`arsenal-chat-suite-v1`), which fixes what was measured.
    var suite: String?
    /// True when the figures come from a quick, uncontrolled run (no measured result exists).
    var quickCheck = false

    init?(_ object: [String: Any]) {
        guard let chip = object["chip"] as? String else { return nil }
        self.chip = Self.chipName(chip)
        match = (object["match"] as? String).flatMap(Match.init) ?? .nearest
        date = object["date"] as? String
        var metrics: [String: Double] = [:]
        for (key, value) in object["metrics"] as? [String: Any] ?? [:] { if let n = (value as? NSNumber)?.doubleValue, n.isFinite { metrics[key] = n } }
        self.metrics = metrics
        suite = (object["suite"] as? [String: Any])?["id"] as? String ?? object["suite"] as? String
        quickCheck = object["verification"] as? String == "quick check"
    }
    init(match: Match, chip: String, date: String?, metrics: [String: Double], suite: String? = nil, quickCheck: Bool = false) {
        self.match = match; self.chip = chip; self.date = date; self.metrics = metrics; self.suite = suite; self.quickCheck = quickCheck
    }

    /// What the reference was measured on, in the suite's own terms (bench/kinds/*/suite-v1.toml): the chat suite
    /// asks three one-line questions; the image suite makes 512 × 512 images in 4 steps.
    var workload: ReferenceWorkload? { suite.flatMap(ReferenceWorkload.of) }
    /// "M5 Max · short prompt · 2 Oct"; rough figures read "M5 Max · quick check · …".
    var baseline: String {
        ([chip] + (quickCheck ? ["quick check"] : []) + [workload?.title].compactMap { $0 } + [date.flatMap(Self.shortDate)].compactMap { $0 })
            .joined(separator: " \u{00b7} ")
    }
    static func shortDate(_ iso: String) -> String? {
        let parse = DateFormatter(); parse.locale = Locale(identifier: "en_US_POSIX"); parse.dateFormat = "yyyy-MM-dd"
        guard let date = parse.date(from: String(iso.prefix(10))) else { return nil }
        let out = DateFormatter(); out.locale = Locale(identifier: "en_GB"); out.dateFormat = "d MMM"
        return out.string(from: date)
    }

    /// "m5-max" → "M5 Max" (results files and C5 name chips in lowercase).
    static func chipName(_ id: String) -> String {
        id.split(separator: "-").map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined(separator: " ")
    }

    /// The first metric present under any of the names results files use for one figure.
    func value(_ names: [String]) -> Double? { names.lazy.compactMap { metrics[$0] }.first }

    /// Tooltip on the chip label: how close the reference is and when it was measured.
    var help: String {
        let how: String, verb = quickCheck ? "Quick check" : "Measured"
        switch match {
        case .sameChip: how = "\(verb) on this chip"
        case .sameFamily: how = "\(verb) on \(chip), the closest chip in this Mac's family with a result"
        case .nearest: how = "\(verb) on \(chip), the closest chip with a result"
        }
        return how + (date.map { ", \($0)." } ?? ".") + (quickCheck ? " Not a controlled measurement: rough figures." : "")
    }
}
