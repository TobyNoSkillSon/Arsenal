// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// Splits generated text at a model's thinking markers into reasoning and answer, incrementally. Partial markers are
/// held back across chunks; whitespace that only separates the two parts is dropped.
/// Qwen3.5/Qwen3-VL: `<think>` … `</think>` (the template opens `<think>` when thinking is on).
/// Gemma 4: `<|channel>thought` … `<channel|>` (special tokens, so the text is decoded with them kept).
struct ThinkingSplitter {
    struct Part: Equatable { let reasoning: Bool; let text: String }
    let open: String
    let close: String
    let label: String
    private var inside: Bool
    private var pending = ""
    private var needLabel = false
    private var trimReasoning = true
    private var trimAnswer = false

    init(open: String, close: String, label: String = "", inside: Bool) {
        self.open = open; self.close = close; self.label = label; self.inside = inside
    }
    /// The splitter for a language architecture, or nil when the model has no thinking markers.
    static func make(architecture: String, thinking: Bool) -> ThinkingSplitter? {
        switch architecture {
        case "qwen3_5", "qwen3_vl": return ThinkingSplitter(open: "<think>", close: "</think>", inside: thinking)
        case "gemma4": return ThinkingSplitter(open: "<|channel>", close: "<channel|>", label: "thought", inside: false)
        default: return nil
        }
    }
    /// Gemma's markers are special tokens; its text is decoded with special tokens kept for this parse.
    static func keepsSpecialTokens(_ architecture: String) -> Bool { architecture == "gemma4" }

    mutating func append(_ text: String) -> [Part] {
        pending += text
        var parts: [Part] = []
        while !pending.isEmpty {
            if needLabel {
                if pending.count < label.count, label.hasPrefix(pending) { break }
                if pending.hasPrefix(label) { pending.removeFirst(label.count) }
                needLabel = false
                continue
            }
            let marker = inside ? close : open
            if let range = pending.range(of: marker) {
                emit(String(pending[..<range.lowerBound]), into: &parts)
                pending = String(pending[range.upperBound...])
                if inside { trimAnswer = true } else { needLabel = !label.isEmpty; trimReasoning = true }
                inside.toggle()
                continue
            }
            var held = 0
            for n in stride(from: min(marker.count - 1, pending.count), through: 1, by: -1) where pending.hasSuffix(String(marker.prefix(n))) {
                held = n
                break
            }
            emit(String(pending.dropLast(held)), into: &parts)
            pending = String(pending.suffix(held))
            break
        }
        return parts
    }
    mutating func finish() -> [Part] {
        var parts: [Part] = []
        if needLabel, label.hasPrefix(pending) { pending = "" }
        emit(pending, into: &parts)
        pending = ""
        return parts
    }
    private mutating func emit(_ text: String, into parts: inout [Part]) {
        var text = text
        if inside, trimReasoning { text = String(text.drop(while: \.isWhitespace)); if !text.isEmpty { trimReasoning = false } }
        if !inside, trimAnswer { text = String(text.drop(while: \.isWhitespace)); if !text.isEmpty { trimAnswer = false } }
        guard !text.isEmpty else { return }
        if let last = parts.last, last.reasoning == inside {
            parts[parts.count - 1] = Part(reasoning: inside, text: last.text + text)
        } else {
            parts.append(Part(reasoning: inside, text: text))
        }
    }
    /// Worker stream events: `reasoning` for thinking text, `token` for answer text (the API routes' contract).
    static func frames(_ parts: [Part]) -> [WireFrame] {
        parts.filter { !$0.text.isEmpty }.map { .json(["event": .string($0.reasoning ? "reasoning" : "token"), "text": .string($0.text)]) }
    }
    /// Whole-text split for a finished generation: the answer and the trimmed reasoning (nil when there was none).
    static func split(_ text: String, splitter: ThinkingSplitter) -> (answer: String, reasoning: String?) {
        var splitter = splitter
        let parts = splitter.append(text) + splitter.finish()
        let reasoning = parts.filter(\.reasoning).map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
        return (parts.filter { !$0.reasoning }.map(\.text).joined(), reasoning.isEmpty ? nil : reasoning)
    }
}
