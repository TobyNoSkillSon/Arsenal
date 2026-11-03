// SPDX-License-Identifier: MIT
import Foundation
import CoreFoundation

public struct ChatFailure: Error, LocalizedError, Sendable {
    public let code: String
    public let message: String
    public init(_ message: String) { self.code = "invalid_request"; self.message = message }
    public init(_ code: String, _ message: String) { self.code = code; self.message = message }
    public var errorDescription: String? { message }
}

public struct ChatMessage: Codable, Sendable, Equatable {
    public let role: String
    public let content: String
    public init(role: String, content: String) { self.role = role; self.content = content }
}

public struct ChatRequest: Sendable {
    public let messages: [ChatMessage]
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let seed: UInt64?
    public let stops: [String]
    public let stream: Bool
    public let thinking: Bool

    public init(_ request: [String: Any]) throws {
        // Tool parsing/constrained decoding are not implemented. Never pretend a prompt is a schema guarantee.
        if let tools = request["tools"] {
            guard let tools = tools as? [Any] else { throw ChatFailure("tools must be an array.") }
            if !tools.isEmpty { throw ChatFailure("unsupported", "Tool calls are not supported by this chat runner yet.") }
        }
        if let format = request["response_format"] {
            guard let f = format as? [String: Any], let type = f["type"] as? String else { throw ChatFailure("response_format requires a type.") }
            if type != "text" { throw ChatFailure("unsupported", "JSON mode and constrained JSON schemas are not supported by this chat runner.") }
        }
        messages = try Self.messages(request["messages"], systemPrompt: request["system_prompt"])
        thinking = try Self.boolean(request, "thinking", default: false)
        maxTokens = try Self.integer(request, "max_tokens", default: 512, range: 1...8192)
        temperature = try Self.number(request, "temperature", default: 0.6, range: 0...2)
        topP = try Self.number(request, "top_p", default: 1, range: 0.000001...1)
        if let raw = request["seed"] {
            guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite,
                n.doubleValue >= 0, n.doubleValue <= 9_007_199_254_740_991, n.doubleValue.rounded() == n.doubleValue
            else {
                throw ChatFailure("seed must be an integer from 0 to 9007199254740991.")
            }
            seed = n.uint64Value
        } else {
            seed = nil
        }
        if let raw = request["stop"] {
            if let s = raw as? String {
                stops = [s]
            } else if let values = raw as? [String] {
                stops = values
            } else {
                throw ChatFailure("stop must be a string or a string array.")
            }
            guard stops.count <= 4, stops.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 1024 }) else {
                throw ChatFailure("Use up to four nonempty stop strings of at most 1024 bytes.")
            }
        } else {
            stops = []
        }
        stream = try Self.boolean(request, "stream", default: false)
    }

    public static func messages(_ raw: Any?, systemPrompt: Any?) throws -> [ChatMessage] {
        let messages = try parseMessages(raw)
        guard let systemPrompt else { return messages }
        guard let prompt = systemPrompt as? String else { throw ChatFailure("system_prompt must be a string.") }
        guard !prompt.isEmpty, !messages.contains(where: { $0.role == "system" }) else { return messages }
        return [ChatMessage(role: "system", content: prompt)] + messages
    }

    /// Canonical folder settings win over the v1 compatibility key.
    public static func contextLimit(_ config: [String: Any]) throws -> Int {
        if let raw = config["settings"] {
            guard let settings = raw as? [String: Any] else { throw ChatFailure("config.settings must be an object.") }
            if settings["context_limit"] != nil {
                return try integer(settings, "context_limit", default: 8192, range: 256...262144)
            }
        }
        return try integer(config, "context_limit", default: 8192, range: 256...262144)
    }

    public static func boolean(_ request: [String: Any], _ key: String, default fallback: Bool) throws -> Bool {
        guard let raw = request[key] else { return fallback }
        guard let n = raw as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else { throw ChatFailure("\(key) must be a boolean.") }
        return n.boolValue
    }

    public static func parseMessages(_ raw: Any?) throws -> [ChatMessage] {
        guard let values = raw as? [[String: Any]], !values.isEmpty else { throw ChatFailure("messages must be a nonempty array.") }
        return try values.map { value in
            guard let role = value["role"] as? String, ["system", "user", "assistant"].contains(role) else {
                throw ChatFailure("unsupported", "Only system, user, and assistant text messages are supported.")
            }
            guard value["tool_calls"] == nil else { throw ChatFailure("unsupported", "Tool-call history is not supported by this runner.") }
            let content: String
            if let text = value["content"] as? String {
                content = text
            } else if let blocks = value["content"] as? [[String: Any]] {
                content = try blocks.map { block in
                    guard block["type"] as? String == "text", let text = block["text"] as? String else {
                        throw ChatFailure("unsupported", "This chat model accepts text, not image or audio content.")
                    }
                    return text
                }.joined()
            } else {
                throw ChatFailure("Each message requires text content.")
            }
            return ChatMessage(role: role, content: content)
        }
    }
    public static func integer(_ request: [String: Any], _ key: String, default fallback: Int, range: ClosedRange<Int>) throws -> Int {
        guard let raw = request[key] else { return fallback }
        guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite,
            n.doubleValue.rounded() == n.doubleValue, n.doubleValue >= Double(range.lowerBound), n.doubleValue <= Double(range.upperBound)
        else {
            throw ChatFailure("\(key) must be an integer from \(range.lowerBound) to \(range.upperBound).")
        }
        return n.intValue
    }
    private static func number(_ request: [String: Any], _ key: String, default fallback: Float, range: ClosedRange<Float>) throws -> Float {
        guard let raw = request[key] else { return fallback }
        guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite, range.contains(n.floatValue) else {
            throw ChatFailure("\(key) is outside its supported range.")
        }
        return n.floatValue
    }
}
