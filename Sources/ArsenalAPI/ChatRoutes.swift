// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func chatParams(_ body: JSONObject) throws -> JSONObject {
        guard let messages = body["messages"]?.array, !messages.isEmpty else { throw HTTPFailure(422, "messages must be a nonempty array.") }
        for message in messages {
            if message["role"]?.string == "developer" { throw HTTPFailure(400, "The developer role is unsupported; use system.") }
            guard let role = message["role"]?.string, ["system", "user", "assistant", "tool"].contains(role) else { throw HTTPFailure(422, "Invalid message role.") }
            if let blocks = message["content"]?.array {
                for block in blocks {
                    switch block["type"]?.string {
                    case "text": guard block["text"]?.string != nil else { throw HTTPFailure(422, "Text part needs text.") }
                    case "image_url":
                        guard let url = block["image_url"]?["url"]?.string, url.hasPrefix("data:image/"), url.contains(";base64,") else {
                            throw HTTPFailure(422, "Use inline base64 image_url data; remote URLs are not fetched.")
                        }
                    case "input_audio":
                        guard let format = block["input_audio"]?["format"]?.string, ["wav", "mp3"].contains(format),
                            let data = block["input_audio"]?["data"]?.string, Data(base64Encoded: data) != nil
                        else { throw HTTPFailure(422, "Invalid input_audio part.") }
                    default: throw HTTPFailure(400, "Unsupported chat content part.")
                    }
                }
            } else if message["content"]?.string == nil && message["tool_calls"] == nil && message["content"] != .null {
                throw HTTPFailure(422, "Message content must be text.")
            }
        }
        if let format = body["response_format"], format != .object(["type": "text"]) {
            guard format["type"]?.string == "json_schema", format["json_schema"]?["schema"]?.object != nil,
                format["json_schema"]?["name"]?.string?.isEmpty == false
            else { throw HTTPFailure(422, "response_format requires a named json_schema.") }
        }
        try reject(
            body,
            [
                "logprobs", "top_logprobs", "audio", "modalities", "prediction", "web_search_options", "store", "frequency_penalty", "presence_penalty", "logit_bias",
                "tool_choice", "parallel_tool_calls"
            ])
        _ = try jsonInteger(body["n"], name: "n", range: 1...1, default: 1, status: 400)
        for key in ["max_tokens", "max_completion_tokens"] where body[key] != nil { _ = try jsonInteger(body[key], name: key, range: 1...8192) }
        if let seed = body["seed"] { _ = try jsonInteger(seed, name: "seed", range: 0...9_007_199_254_740_991) }
        if let thinking = body["thinking"], thinking.bool == nil { throw HTTPFailure(422, "thinking must be a boolean.") }
        var params: JSONObject = ["messages": .array(messages), "stream": body["stream"] ?? false]
        for key in ["max_tokens", "temperature", "top_p", "stop", "seed", "tools", "thinking", "system_prompt", "response_format"] { params[key] = body[key] }
        if let maximum = body["max_completion_tokens"] { params["max_tokens"] = maximum }
        return params
    }
    func validateStructuredChat(_ result: JSONObject, params: JSONObject) throws {
        guard let schema = params["response_format"]?["json_schema"]?["schema"]?.object else { return }
        guard let text = result["text"]?.string, let data = text.data(using: .utf8), let value = try? JSONDecoder().decode(JSONValue.self, from: data) else {
            throw HTTPFailure(502, "Model did not return valid JSON for the requested response schema.")
        }
        do { try OperationSchemaValidator.validate(value, schema: .object(schema)) } catch {
            throw HTTPFailure(502, "Model output failed the requested response schema: " + error.localizedDescription)
        }
    }

    func chat(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let body = try object(request), model = try required(body, "model")
        _ = try await resolve(model, kind: "chat", operation: "chat")
        let params = try chatParams(body)
        let id = "chatcmpl-" + UUID().uuidString.lowercased(), created = Int(Date().timeIntervalSince1970)
        if body["stream"]?.bool == true {
            let writer = ChatStream(connection: connection, id: id, model: model, created: created, thinking: params["thinking"]?.bool != false)
            let result = try await run(
                model, op: "chat", params: params,
                sink: { frame in
                    if case .json(let event) = frame, let text = event["text"]?.string {
                        if event["event"]?.string == "token" {
                            try await writer.text(text)
                        } else if event["event"]?.string == "reasoning", params["thinking"]?.bool != false {
                            try await writer.reasoning(text)
                        }
                    }
                })
            try validateStructuredChat(result, params: params)
            try await writer.complete(result, includeUsage: body["stream_options"]?["include_usage"]?.bool == true)
        } else {
            let result = try await run(model, op: "chat", params: params)
            try validateStructuredChat(result, params: params)
            var message: [String: Any] = ["role": "assistant", "content": result["text"]?.string ?? ""]
            if params["thinking"]?.bool != false, let reasoning = result["reasoning"]?.string { message["reasoning_content"] = reasoning }
            if let tools = result["tool_calls"] { message["tool_calls"] = plain(tools); if result["text"]?.string == nil { message["content"] = NSNull() } }
            var response: [String: Any] = [
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [["index": 0, "message": message, "finish_reason": result["finish_reason"]?.string ?? "stop"]], "usage": try chatUsage(result)
            ]
            if let timing = result["timing"] { response["arsenal_timing"] = plain(timing) }
            try await connection.respond(response)
        }
    }
}

private actor ChatStream {
    let connection: HTTPConnection, id: String, model: String, created: Int
    let thinking: Bool
    var began = false, emittedText = false, emittedReasoning = false
    init(connection: HTTPConnection, id: String, model: String, created: Int, thinking: Bool) {
        self.connection = connection; self.id = id; self.model = model; self.created = created; self.thinking = thinking
    }
    func start() async throws {
        guard !began else { return }; began = true
        try await connection.beginStream()
        try await delta(["role": "assistant", "content": ""])
    }
    func delta(_ value: [String: Any], finish: String? = nil) async throws {
        try await connection.event([
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [["index": 0, "delta": value, "finish_reason": finish as Any? ?? NSNull()]]
        ])
    }
    func text(_ text: String) async throws { try await start(); emittedText = true; try await delta(["content": text]) }
    func reasoning(_ text: String) async throws { try await start(); emittedReasoning = true; try await delta(["reasoning_content": text]) }
    func complete(_ result: JSONObject, includeUsage: Bool) async throws {
        try await start()
        if thinking && !emittedReasoning, let reasoning = result["reasoning"]?.string { try await self.reasoning(reasoning) }
        if !emittedText, let text = result["text"]?.string { try await delta(["content": text]) }
        if let tools = result["tool_calls"]?.array {
            for (index, tool) in tools.enumerated() {
                var value = tool.object.map(plain) ?? [:]; value["index"] = index
                let function = tool["function"]?.object ?? [:]
                value["function"] = ["name": function["name"]?.string ?? "", "arguments": ""]
                try await delta(["tool_calls": [value]])
                for character in function["arguments"]?.string ?? "{}" { try await delta(["tool_calls": [["index": index, "function": ["arguments": String(character)]]]]) }
            }
        }
        var final: [String: Any] = [
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [["index": 0, "delta": [:], "finish_reason": result["finish_reason"]?.string ?? "stop"]]
        ]
        if !includeUsage, let timing = result["timing"] { final["arsenal_timing"] = plain(timing) }
        try await connection.event(final)
        if includeUsage {
            var final: [String: Any] = ["id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": try chatUsage(result)]
            if let timing = result["timing"] { final["arsenal_timing"] = plain(timing) }
            try await connection.event(final)
        }
        try await connection.chunk(Data("data: [DONE]\n\n".utf8))
    }
}
