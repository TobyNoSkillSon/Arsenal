// SPDX-License-Identifier: MIT
import Foundation

// The chat window's speed presets: prompts of about 1K…64K tokens of plain English notes with one question at the end.
// Each insertion opens with a fresh run line, so no part of it matches an earlier prompt and the model's prefix cache
// cannot shorten the prefill being measured. The text is made from a fixed word bank by a seeded generator: the same
// size reads the same apart from the run line, and nothing is downloaded or bundled.

struct PresetPrompt: Equatable, Identifiable {
    /// Nominal size in tokens (1024 … 65536).
    var tokens: Int
    var id: Int { tokens }
    var title: String { "\(tokens / 1024)K" }

    static let all: [PresetPrompt] = [1, 2, 4, 8, 16, 32, 64].map { PresetPrompt(tokens: $0 * 1024) }
    /// Room left for the reply when checking a preset against the loaded context.
    static let replyTokens = ChatSession.maxReplyTokens
    /// Seed estimate only; each complete preset request is resized using the model's real count.
    static let tokensPerWord = 1.3

    static let question = "\n\nQuestion: in two sentences, what are these notes mostly about?"

    /// The prompt for `words` words of notes, opening with the run's own line.
    static func text(words: Int, run: String) -> String {
        var generator = SeededWords(seed: 0x6D6F_6269_7573)
        var out = "Run \(run). Read the field notes below, then answer the question after them.\n\n"
        out.reserveCapacity(words * 7)
        var written = 0, sentence = 0
        while written < words {
            let length = min(words - written, 9 + generator.next(13))
            out += generator.sentence(length)
            written += length; sentence += 1
            out += sentence % 6 == 0 ? "\n\n" : " "
        }
        return out + question
    }
    /// The words for a nominal token size before any calibration.
    func words(ratio: Double = PresetPrompt.tokensPerWord) -> Int {
        // A count from the model below 0.5 or above 4 tokens per word is not English prose: use the estimate.
        let r = ratio.isFinite && (0.5...4).contains(ratio) ? ratio : Self.tokensPerWord
        return max(16, (Double(tokens) / r).roundedInt ?? tokens)
    }

    /// Count the complete, isolated request (including the typed question and the model's chat template).
    /// The estimate only seeds the search; never send it without a successful real count within 2%.
    @MainActor func sized(run: String, typed: String, contextLimit: Int?, count: (String) async throws -> Int) async throws -> (text: String, tokens: Int) {
        var low = 0, high: Int?, words = words()
        let tolerance = max(1, tokens / 50)
        for _ in 0..<24 {
            try Task.checkCancellation()
            let notes = Self.text(words: words, run: run)
            let text = typed.isEmpty ? notes : notes + "\n\n" + typed
            let actual = try await count(text)
            try Task.checkCancellation()
            guard actual > 0 else { throw ToolError("The model returned no prompt token count.") }
            if abs(actual - tokens) <= tolerance, contextLimit.map({ actual + Self.replyTokens <= $0 }) ?? true {
                return (text, actual)
            }
            if actual > tokens || contextLimit.map({ actual + Self.replyTokens > $0 }) == true {
                high = words
            } else {
                low = words + 1
            }
            if let high {
                guard low < high else { break }
                words = low + (high - low) / 2
            } else {
                words = min(tokens * 8, max(words + 1, Int(Double(words) * Double(tokens) / Double(actual))))
            }
        }
        throw ToolError("Could not size the \(title) prompt within 2% with room for the reply; shorten the typed question or choose a larger context.")
    }

    /// Why the preset cannot be sent to a model loaded with this context (with room for a reply); nil when it can.
    func unavailable(contextLimit: Int?, model: String) -> String? {
        guard let limit = contextLimit, tokens + Self.replyTokens > limit else { return nil }
        return "Longer than \(model)\u{2019}s \(FigureText.context(limit)) context"
    }
}

/// A tiny deterministic generator (SplitMix64) over a fixed bank of common words.
struct SeededWords {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next(_ bound: Int) -> Int {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Int((z ^ (z >> 31)) % UInt64(max(1, bound)))
    }
    mutating func sentence(_ words: Int) -> String {
        var parts: [String] = []
        parts.reserveCapacity(words)
        for i in 0..<words {
            let bank = i % 4 == 1 ? Self.nouns : i % 4 == 3 ? Self.verbs : Self.common
            parts.append(bank[next(bank.count)])
        }
        guard var first = parts.first else { return "" }
        first = first.prefix(1).uppercased() + first.dropFirst()
        parts[0] = first
        return parts.joined(separator: " ") + (next(5) == 0 ? "," : ".")
    }
    static let common = [
        "the", "a", "small", "quiet", "early", "north", "old", "new", "warm", "cold", "near", "far", "every", "some", "this", "that", "under", "over",
        "along", "beside", "after", "before", "with", "without", "through", "across", "around", "between", "behind", "above", "below", "inside", "again",
        "often", "rarely", "slowly", "quickly", "later", "soon", "still", "already", "nearly", "almost", "only", "also", "even", "then", "here", "there"
    ]
    static let nouns = [
        "river", "harbour", "village", "market", "orchard", "bridge", "ferry", "lighthouse", "garden", "valley", "station", "workshop", "library", "meadow",
        "forest", "island", "boat", "road", "mill", "school", "farm", "kitchen", "window", "letter", "lantern", "engine", "season", "harvest", "storm",
        "morning", "evening", "winter", "summer", "tide", "path", "hill", "field", "well", "chapel", "square", "fountain", "net", "sail", "map", "clock"
    ]
    static let verbs = [
        "opened", "closed", "carried", "followed", "repaired", "painted", "measured", "counted", "watched", "crossed", "gathered", "planted", "built",
        "visited", "named", "moved", "kept", "found", "lost", "shared", "sold", "bought", "cleaned", "lit", "rang", "turned", "waited", "returned",
        "recorded", "noticed", "described", "remembered", "changed", "joined", "left", "reached", "checked", "marked", "sorted", "mended"
    ]
}
