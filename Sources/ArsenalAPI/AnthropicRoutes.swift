// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func anthropic(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let path = String(request.path.dropFirst("/providers/anthropic".count))
        if request.method == "GET", path == "/v1/models" || path.hasPrefix("/v1/models/") {
            let models = await rows().filter { $0["kind"] as? String == "chat" }
            func model(_ row: [String: Any]) -> [String: Any] { ["id": row["id"] ?? "", "type": "model", "display_name": row["name"] ?? "", "created_at": "2026-10-02T00:00:00Z"] }
            if path == "/v1/models" {
                try await connection.respond(["data": models.map(model), "has_more": false, "first_id": models.first?["id"] ?? NSNull(), "last_id": models.last?["id"] ?? NSNull()])
            } else {
                try await connection.respond(model(try await resolve(String(path.dropFirst("/v1/models/".count)), kind: "chat")))
            }
            return
        }
        guard path == "/v1/messages" || path == "/v1/messages/count_tokens" else { throw HTTPFailure(404, "Anthropic endpoint not supported.") }
        let body = try object(request), model = try required(body, "model")
        _ = try await resolve(model, kind: "chat", operation: path.hasSuffix("count_tokens") ? "count_tokens" : "chat")
        let params = try anthropicParams(body)
        if path.hasSuffix("count_tokens") {
            let result = try await run(model, op: "count_tokens", params: anthropicCountParams(params))
            try await connection.respond(["input_tokens": try tokenCount(result["input_tokens"] ?? result["tokens"])]); return
        }
        _ = try jsonInteger(body["max_tokens"], name: "max_tokens", range: 1...8192)
        let warnings =
            body["thinking"]?["budget_tokens"] != nil ? ["X-Arsenal-Warnings": "budget_tokens ignored; local thinking is an on/off control without a cloud signature"] : [:]
        let id = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        if body["stream"]?.bool == true {
            let stream = AnthropicStream(connection: connection, model: model, id: id, thinking: params["thinking"]?.bool != false, headers: warnings)
            let result = try await run(
                model, op: "chat", params: params,
                sink: { frame in
                    if case .json(let value) = frame, let text = value["text"]?.string {
                        if value["event"]?.string == "token" { try await stream.text(text) } else if value["event"]?.string == "reasoning" { try await stream.reasoning(text) }
                    }
                })
            try await stream.complete(result)
        } else {
            let result = try await run(model, op: "chat", params: params)
            try await connection.respond(
                [
                    "id": id, "type": "message", "role": "assistant", "model": model, "content": try anthropicContent(result, thinking: params["thinking"]?.bool != false),
                    "stop_reason": anthropicStop(result), "stop_sequence": NSNull(),
                    "usage": ["input_tokens": try tokenCount(result["usage"]?["prompt_tokens"]), "output_tokens": try tokenCount(result["usage"]?["completion_tokens"])]
                ], headers: warnings)
        }
    }
    // Tokenization uses the same vendor prompt as generation, never its sampling controls.
    func anthropicCountParams(_ mapped: JSONObject) -> JSONObject {
        mapped.filter { ["messages", "thinking", "tools"].contains($0.key) }
    }
    func anthropicParams(_ body: JSONObject) throws -> JSONObject {
        try reject(body, ["metadata", "service_tier", "container", "context_management", "mcp_servers", "top_k", "tool_choice"])
        var messages: [JSONValue] = []
        func textContent(_ value: JSONValue) throws -> String {
            if let text = value.string { return text }
            guard let blocks = value.array, blocks.allSatisfy({ $0["type"]?.string == "text" && $0["text"]?.string != nil && $0["cache_control"] == nil }) else {
                throw HTTPFailure(400, "Only uncached text content is supported.")
            }
            return blocks.compactMap { $0["text"]?.string }.joined(separator: "\n")
        }
        if let system = body["system"] { messages.append(["role": "system", "content": .string(try textContent(system))]) }
        guard let original = body["messages"]?.array, !original.isEmpty else { throw HTTPFailure(422, "messages is required.") }
        for message in original {
            guard let role = message["role"]?.string, ["user", "assistant"].contains(role), let content = message["content"] else {
                throw HTTPFailure(422, "Invalid Anthropic message.")
            }
            if let text = content.string { messages.append(["role": .string(role), "content": .string(text)]); continue }
            guard let blocks = content.array else { throw HTTPFailure(422, "content must be text or blocks.") }
            var text: [String] = [], tools: [JSONValue] = [], media: [JSONValue] = []
            for block in blocks {
                switch block["type"]?.string {
                case "text": text.append(try textContent(.array([block])))
                case "image":
                    guard role == "user", block["cache_control"] == nil, let source = block["source"]?.object,
                        source["type"]?.string == "base64", let mime = source["media_type"]?.string, ["image/png", "image/jpeg"].contains(mime),
                        let data = source["data"]?.string, Data(base64Encoded: data) != nil
                    else { throw HTTPFailure(422, "Use an uncached base64 PNG/JPEG image block.") }
                    media.append(["type": "image_url", "image_url": ["url": .string("data:" + mime + ";base64," + data)]])
                case "thinking":
                    guard role == "assistant", block["thinking"]?.string != nil else { throw HTTPFailure(422, "Malformed thinking block.") }
                    // Local reasoning is not a signed cloud token or user text.
                    continue
                case "tool_use":
                    guard role == "assistant", let id = block["id"]?.string, let name = block["name"]?.string, let input = block["input"]?.object else {
                        throw HTTPFailure(422, "Malformed tool_use block.")
                    }
                    tools.append([
                        "id": .string(id), "type": "function",
                        "function": ["name": .string(name), "arguments": .string(String(decoding: try WireJSON.encode(input), as: UTF8.self))]
                    ])
                case "tool_result":
                    guard role == "user", let id = block["tool_use_id"]?.string else { throw HTTPFailure(422, "Malformed tool_result block.") }
                    messages.append(["role": "tool", "tool_call_id": .string(id), "content": .string(try textContent(block["content"] ?? ""))])
                default: throw HTTPFailure(400, "This model supports text and caller-executed tools only.")
                }
            }
            if !text.isEmpty || !tools.isEmpty || !media.isEmpty {
                var value: JSONObject = ["role": .string(role), "content": .string(text.joined(separator: "\n"))]
                if !media.isEmpty { value["content"] = .array(text.map { ["type": "text", "text": .string($0)] } + media) }
                if !tools.isEmpty { value["tool_calls"] = .array(tools) }; messages.append(.object(value))
            }
        }
        var mapped: JSONObject = ["messages": .array(messages), "stream": body["stream"] ?? false]
        for key in ["max_tokens", "temperature", "top_p"] { mapped[key] = body[key] }
        mapped["stop"] = body["stop_sequences"]
        if let thinking = body["thinking"] {
            guard let config = thinking.object, let type = config["type"]?.string, ["enabled", "disabled", "adaptive"].contains(type) else {
                throw HTTPFailure(422, "thinking must declare enabled, disabled or adaptive.")
            }
            if let budget = config["budget_tokens"] { _ = try jsonInteger(budget, name: "budget_tokens", range: 1...1_000_000) }
            mapped["thinking"] = .bool(type != "disabled")
        }
        if let tools = body["tools"]?.array {
            mapped["tools"] = .array(
                try tools.map { tool in
                    guard let name = tool["name"]?.string, tool["input_schema"]?.object != nil, tool["type"] == nil else {
                        throw HTTPFailure(400, "Only function tools with input_schema are supported.")
                    }
                    var function: JSONObject = ["name": .string(name), "parameters": tool["input_schema"]!]
                    function["description"] = tool["description"]
                    return ["type": "function", "function": .object(function)]
                })
        }
        return try chatParams(mapped)
    }
    nonisolated func anthropicContent(_ result: JSONObject, thinking: Bool = true) throws -> [[String: Any]] {
        var content: [[String: Any]] = []
        if thinking, let reasoning = result["reasoning"]?.string, !reasoning.isEmpty { content.append(["type": "thinking", "thinking": reasoning, "signature": ""]) }
        if let text = result["text"]?.string, !text.isEmpty { content.append(["type": "text", "text": text]) }
        for call in result["tool_calls"]?.array ?? [] {
            guard let name = call["function"]?["name"]?.string, let arguments = call["function"]?["arguments"]?.string,
                let object = try? JSONSerialization.jsonObject(with: Data(arguments.utf8)) as? [String: Any]
            else { throw HTTPFailure(500, "Invalid worker tool arguments.") }
            content.append(["type": "tool_use", "id": call["id"]?.string ?? "", "name": name, "input": object])
        }
        return content
    }
    func anthropicStop(_ result: JSONObject) -> String {
        switch result["finish_reason"]?.string {
        case "tool_calls": "tool_use"
        case "length": "max_tokens"
        default: "end_turn"
        }
    }
}

private actor AnthropicStream {
    let connection: HTTPConnection, model: String, id: String
    let thinking: Bool
    let headers: [String: String]
    var began = false, textStarted = false, textEmitted = false, reasoningStarted = false, reasoningEnded = false, reasoningEmitted = false
    init(connection: HTTPConnection, model: String, id: String, thinking: Bool, headers: [String: String]) {
        self.connection = connection; self.model = model; self.id = id; self.thinking = thinking; self.headers = headers
    }
    var textIndex: Int { reasoningStarted ? 1 : 0 }
    func reasoning(_ value: String) async throws {
        guard thinking else { return }
        guard !textStarted else { throw HTTPFailure(502, "Runner emitted reasoning after answer text.") }
        try await start()
        if !reasoningStarted {
            reasoningStarted = true
            try await connection.event(
                ["type": "content_block_start", "index": 0, "content_block": ["type": "thinking", "thinking": "", "signature": ""]], name: "content_block_start")
        }
        reasoningEmitted = true
        try await connection.event(["type": "content_block_delta", "index": 0, "delta": ["type": "thinking_delta", "thinking": value]], name: "content_block_delta")
    }
    func endReasoning() async throws {
        if reasoningStarted && !reasoningEnded { reasoningEnded = true; try await connection.event(["type": "content_block_stop", "index": 0], name: "content_block_stop") }
    }
    func start() async throws {
        guard !began else { return }; began = true
        try await connection.beginStream(headers: headers)
        try await connection.event(
            [
                "type": "message_start",
                "message": [
                    "id": id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": NSNull(), "stop_sequence": NSNull(),
                    "usage": ["input_tokens": 0, "output_tokens": 0]
                ]
            ], name: "message_start")
    }
    func text(_ value: String) async throws {
        try await start()
        try await endReasoning()
        if !textStarted {
            textStarted = true
            try await connection.event(["type": "content_block_start", "index": textIndex, "content_block": ["type": "text", "text": ""]], name: "content_block_start")
        }
        textEmitted = true
        try await connection.event(["type": "content_block_delta", "index": textIndex, "delta": ["type": "text_delta", "text": value]], name: "content_block_delta")
    }
    func complete(_ result: JSONObject) async throws {
        try await start()
        if !reasoningEmitted && !textStarted, let value = result["reasoning"]?.string, !value.isEmpty { try await reasoning(value) }
        try await endReasoning()
        if !textEmitted, let value = result["text"]?.string, !value.isEmpty { try await text(value) }
        var index = textIndex
        if textStarted { try await connection.event(["type": "content_block_stop", "index": textIndex], name: "content_block_stop"); index = textIndex + 1 }
        for call in result["tool_calls"]?.array ?? [] {
            let arguments = call["function"]?["arguments"]?.string ?? "{}"
            guard (try? JSONSerialization.jsonObject(with: Data(arguments.utf8))) is [String: Any] else { throw HTTPFailure(500, "Invalid tool arguments.") }
            try await connection.event(
                [
                    "type": "content_block_start", "index": index,
                    "content_block": ["type": "tool_use", "id": call["id"]?.string ?? "", "name": call["function"]?["name"]?.string ?? "", "input": [:]]
                ], name: "content_block_start")
            for char in arguments {
                try await connection.event(
                    ["type": "content_block_delta", "index": index, "delta": ["type": "input_json_delta", "partial_json": String(char)]], name: "content_block_delta")
            }
            try await connection.event(["type": "content_block_stop", "index": index], name: "content_block_stop"); index += 1
        }
        let reason = result["finish_reason"]?.string == "tool_calls" ? "tool_use" : result["finish_reason"]?.string == "length" ? "max_tokens" : "end_turn"
        try await connection.event(
            [
                "type": "message_delta", "delta": ["stop_reason": reason, "stop_sequence": NSNull()],
                "usage": ["output_tokens": try tokenCount(result["usage"]?["completion_tokens"])]
            ], name: "message_delta")
        try await connection.event(["type": "message_stop"], name: "message_stop")
    }
}
