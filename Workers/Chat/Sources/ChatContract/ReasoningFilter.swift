// SPDX-License-Identifier: MIT
import Foundation

public struct ChatDelta: Sendable, Equatable {
    public enum Event: String, Sendable { case token, reasoning }
    public let event: Event
    public let text: String
    public init(event: Event, text: String) { self.event = event; self.text = text }
}

/// The vendor template already opens <think>. Hold partial closing markers across detokenizer chunks.
public struct ReasoningFilter: Sendable {
    private var reasoning: Bool
    private var pending = ""
    private let marker = "</think>"
    public init(thinking: Bool) { reasoning = thinking }
    public mutating func append(_ text: String) -> [ChatDelta] {
        guard reasoning else { return text.isEmpty ? [] : [.init(event: .token, text: text)] }
        pending += text
        if let range = pending.range(of: marker) {
            let thought = String(pending[..<range.lowerBound])
            let answer = String(pending[range.upperBound...])
            pending = ""; reasoning = false
            return [.init(event: .reasoning, text: thought), .init(event: .token, text: answer)].filter { !$0.text.isEmpty }
        }
        var held = 0
        for n in 1..<marker.count where pending.hasSuffix(String(marker.prefix(n))) { held = n }
        let safe = String(pending.dropLast(held)); pending = String(pending.suffix(held))
        return safe.isEmpty ? [] : [.init(event: .reasoning, text: safe)]
    }
    public mutating func finish() -> [ChatDelta] {
        defer { pending = "" }
        return pending.isEmpty ? [] : [.init(event: reasoning ? .reasoning : .token, text: pending)]
    }
}
