// SPDX-License-Identifier: MIT
import Foundation

/// The checkpoint's prompt contract, ported line for line from the maker's diffusers pipeline
/// (`modular_pipelines/minimax_music3/encoders.py`, `_clean_caption`, `_normalize_lyrics` and the tokenize step).
enum MiniMaxPrompt {
    static let audioEndToken = 151_670
    static let audioCFGToken = 151_654
    static let audioCodeOffset = 151_675
    static let semanticVocabulary = 16_384
    static let maxPromptTokens = 5_000
    static let maxAudioFrames = 9_000

    /// The text the tokenizer receives.
    static func text(caption: String, lyrics: String) -> String {
        "<|im_start|><|caption_start|>" + cleanCaption(caption) + "<|caption_end|><|lyrics_start|>" + normalizeLyrics(lyrics)
            + "<|lyrics_end|><|im_end|><|audio_start|>"
    }

    /// The classifier-free row: every token but the first and the two trailing structure tokens becomes the audio-CFG token.
    static func unconditional(_ ids: [Int]) -> [Int] {
        guard ids.count > 3 else { return ids }
        var row = ids
        for index in 1..<(ids.count - 2) { row[index] = audioCFGToken }
        return row
    }

    static func cleanCaption(_ caption: String) -> String {
        var text = rewriteSpecialTags(caption)
        var lines: [String] = []
        for raw in pythonSplitLines(text) {
            var line = replace(raw, #"^\s{0,3}#{1,6}\s+"#, with: "")
            line = replace(line, #"^\s*[*+-]\s+"#, with: "")
            line = replace(line, #"^\s*\*\s+"#, with: "")
            while line.contains("**") {
                let updated = replace(line, #"\*\*([^*]+)\*\*"#, with: "$1")
                if updated == line { break }
                line = updated
            }
            line = replace(line, #"(?<!\*)\*([^*\n]+)\*(?!\*)"#, with: "$1")
            lines.append(rstrip(line))
        }
        text = lines.joined(separator: "\n")
        text = replace(text, #"^\s*[-*_]{3,}\s*$"#, with: "", multiline: true)
        text = text.replacingOccurrences(of: "\u{2022} ", with: "").replacingOccurrences(of: "    ", with: "")
        return replace(text, #"\n{2,}"#, with: "\n")
    }

    static func normalizeLyrics(_ lyrics: String) -> String {
        guard let leading = try? NSRegularExpression(pattern: #"^[ \t]*((?:\[[^\]]+\][ \t]*)+)"#) else { return "[start]\n" + lyrics }
        var output: [String] = []
        for line in lyrics.components(separatedBy: "\n") {
            let ns = line as NSString
            if let match = leading.firstMatch(in: line, range: NSRange(location: 0, length: ns.length)) {
                output.append(ns.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines))
            } else {
                output.append(line)
            }
        }
        var text = output.joined(separator: "\n")
        text = text.replacingOccurrences(of: "] ", with: "]\n").replacingOccurrences(of: " [", with: "\n[").replacingOccurrences(of: " ^ ", with: "\n")
        text = mapMatches(text, #"\[([^\]]+)\]"#) { "[" + $0[1].lowercased() + "]" }
        return "[start]\n" + text
    }

    /// `<|bpm 120|>` → `bpm is 120`; a tag with one word keeps the word.
    static func rewriteSpecialTags(_ text: String) -> String {
        mapMatches(text, #"<\|([^|]*)\|>"#) { groups in
            let inner = groups[1].trimmingCharacters(in: .whitespacesAndNewlines)
            guard let split = inner.firstIndex(where: { $0.isWhitespace }) else { return inner }
            let head = String(inner[..<split])
            let tail = inner[split...].drop(while: { $0.isWhitespace })
            return head + " is " + tail
        }
    }

    /// Python's `str.splitlines()` boundaries.
    static func pythonSplitLines(_ text: String) -> [String] {
        let breaks: Set<Character> = ["\n", "\r", "\r\n", "\u{0B}", "\u{0C}", "\u{1C}", "\u{1D}", "\u{1E}", "\u{85}", "\u{2028}", "\u{2029}"]
        var lines: [String] = [], current = ""
        for character in text {
            if breaks.contains(character) {
                lines.append(current); current = ""
            } else {
                current.append(character)
            }
        }
        if !current.isEmpty { lines.append(current) }
        return lines
    }

    static func rstrip(_ text: String) -> String {
        var value = Substring(text)
        while let last = value.last, last.isWhitespace { value = value.dropLast() }
        return String(value)
    }

    static func replace(_ text: String, _ pattern: String, with template: String, multiline: Bool = false) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: multiline ? [.anchorsMatchLines] : []) else { return text }
        return regex.stringByReplacingMatches(in: text, range: NSRange(location: 0, length: (text as NSString).length), withTemplate: template)
    }

    static func mapMatches(_ text: String, _ pattern: String, _ transform: ([String]) -> String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let ns = text as NSString
        var result = "", cursor = 0
        for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            result += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            let groups = (0..<match.numberOfRanges).map { match.range(at: $0).location == NSNotFound ? "" : ns.substring(with: match.range(at: $0)) }
            result += transform(groups)
            cursor = match.range.location + match.range.length
        }
        return result + ns.substring(from: cursor)
    }
}
