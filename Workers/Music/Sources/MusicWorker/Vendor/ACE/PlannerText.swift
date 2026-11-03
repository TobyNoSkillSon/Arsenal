//
//  PlannerText.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The planner's text, both ways: the Qwen chat prompts upstream builds
//  (`build_formatted_prompt`, `build_formatted_prompt_with_cot`), the
//  reasoning block it writes back as YAML (`_format_metadata_as_cot`, which
//  is `yaml.dump(sort_keys=True, allow_unicode=True)` — so PyYAML's plain
//  and single-quoted line folding at 80 columns is reproduced here), and
//  the parse of the model's answer (`parse_lm_output`). Pure.
//

import Foundation

// What the planner decided: the reasoning block's fields.
public struct PlannedMetadata: Codable, Sendable, Equatable {
    public var bpm: Int?
    public var caption: String?
    public var duration: Int?
    public var keyscale: String?
    public var language: String?
    public var timesignature: String?

    public init(
        bpm: Int? = nil, caption: String? = nil, duration: Int? = nil, keyscale: String? = nil,
        language: String? = nil, timesignature: String? = nil
    ) {
        self.bpm = bpm
        self.caption = caption
        self.duration = duration
        self.keyscale = keyscale
        self.language = language
        self.timesignature = timesignature
    }
}

enum PlannerText {

    static let lmInstruction = "Generate audio semantic tokens based on the given conditions:"
    static let ditInstruction = "Fill the audio semantic mask based on the given conditions:"
    static let noUserInput = "NO USER INPUT"

    // MARK: Prompts

    // Qwen's chat template with a system and a user turn, assistant turn open.
    static func chat(system: String, user: String) -> String {
        "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
    }

    static var system: String { "# Instruction\n\(lmInstruction)\n\n" }

    static func user(caption: String, lyrics: String) -> String {
        "# Caption\n\(caption)\n\n# Lyric\n\(lyrics)\n"
    }

    // Whether a negative prompt says something (not empty and not the default).
    static func isMeaningful(negative: String) -> Bool {
        let trimmed = negative.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty && trimmed != noUserInput
    }

    // Phase 1 (reasoning) prompt.
    static func reasoningPrompt(caption: String, lyrics: String) -> String {
        chat(system: system, user: user(caption: caption, lyrics: lyrics))
    }

    // Phase 1's unconditional prompt (used only when the reasoning CFG scale is above 1).
    static func reasoningUnconditional(lyrics: String, negative: String) -> String {
        let body = isMeaningful(negative: negative) ? "# Caption\n\(negative)\n\n# Lyric\n\(lyrics)\n" : "# Lyric\n\(lyrics)\n"
        return chat(system: system, user: body)
    }

    // Phase 2 (codes) prompt: the reasoning block written in, assistant turn left open.
    static func codesPrompt(caption: String, lyrics: String, reasoning: String) -> String {
        chat(system: system, user: user(caption: caption, lyrics: lyrics)) + reasoning + "\n\n"
    }

    // Phase 2's unconditional prompt, as upstream trained its CFG dropout.
    static func codesUnconditional(negative: String) -> String {
        chat(system: system, user: isMeaningful(negative: negative) ? negative : noUserInput) + "<think>\n\n</think>\n\n"
    }

    // MARK: Reasoning block

    // `<think>\n<yaml>\n</think>` with the keys sorted, as upstream feeds it back.
    static func reasoningBlock(_ m: PlannedMetadata) -> String {
        var lines: [String] = []
        if let bpm = m.bpm { lines.append("bpm: \(bpm)") }
        if let caption = m.caption { lines.append(YAMLScalar.line(key: "caption", value: caption)) }
        if let duration = m.duration { lines.append("duration: \(duration)") }
        if let key = m.keyscale { lines.append(YAMLScalar.line(key: "keyscale", value: key)) }
        if let language = m.language { lines.append(YAMLScalar.line(key: "language", value: language)) }
        if var signature = m.timesignature {
            if signature.hasSuffix("/4") { signature = String(signature.split(separator: "/")[0]) }
            lines.append(Int(signature).map { "timesignature: \($0)" } ?? YAMLScalar.line(key: "timesignature", value: signature))
        }
        return "<think>\n\(lines.joined(separator: "\n"))\n</think>"
    }

    // MARK: Parsing

    // Upstream's `parse_lm_output`: the reasoning fields (YAML-ish, multi-line
    // values continued by indentation) and the audio codes.
    static func parse(_ output: String) -> (PlannedMetadata, [Int]) {
        var codes: [Int] = []
        var rest = output[...]
        while let open = rest.range(of: "<|audio_code_") {
            let tail = rest[open.upperBound...]
            guard let close = tail.range(of: "|>") else { break }
            if let value = Int(tail[..<close.lowerBound]) { codes.append(value) }
            rest = tail[close.upperBound...]
        }
        var reasoning: String
        if let start = output.range(of: "<think>"), let end = output.range(of: "</think>", range: start.upperBound..<output.endIndex) {
            reasoning = String(output[start.upperBound..<end.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            reasoning = output.components(separatedBy: "<|audio_code_").first?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        }
        if reasoning.isEmpty { reasoning = "" }
        var meta = PlannedMetadata()
        var key: String?
        var values: [String] = []
        func save() {
            defer { key = nil; values = [] }
            guard let key, !values.isEmpty else { return }
            let value = values.joined(separator: "\n")
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            switch key {
            case "bpm": meta.bpm = Int(trimmed)
            case "caption": meta.caption = collapse(value)
            case "duration": meta.duration = Int(trimmed)
            case "keyscale": meta.keyscale = trimmed
            case "language": meta.language = trimmed
            case "timesignature": meta.timesignature = trimmed
            default: break
            }
        }
        for line in reasoning.components(separatedBy: "\n") {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("<") { continue }
            if let first = line.first, !first.isWhitespace, let colon = line.firstIndex(of: ":") {
                save()
                key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
                let value = String(line[line.index(after: colon)...])
                if !value.trimmingCharacters(in: .whitespaces).isEmpty { values.append(value) }
            } else if line.hasPrefix(" ") || line.hasPrefix("\t"), key != nil {
                values.append(line)
            }
        }
        save()
        return (meta, codes)
    }

    // Upstream's `postprocess_caption`: YAML continuation lines joined with single spaces.
    static func collapse(_ caption: String) -> String {
        caption.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}

// PyYAML's emitter for one `key: value` line of a block mapping (default
// width 80, indent 2, unicode allowed): plain style when the scalar allows
// it, else single-quoted, with its word folding.
enum YAMLScalar {

    static let bestWidth = 80

    static func line(key: String, value: String) -> String {
        let prefix = "\(key):"
        let style = plainAllowed(value) ? Style.plain : (singleQuotedAllowed(value) ? .single : .double)
        switch style {
        case .plain: return prefix + plain(value, column: prefix.unicodeScalars.count)
        case .single: return prefix + " " + singleQuoted(value, column: prefix.unicodeScalars.count + 1)
        case .double: return prefix + " " + doubleQuoted(value)
        }
    }

    enum Style { case plain, single, double }

    // MARK: Analysis (PyYAML analyze_scalar + resolver)

    static func plainAllowed(_ s: String) -> Bool {
        guard !s.isEmpty else { return false }
        if resolvesToNonString(s) { return false }
        let a = analyse(s)
        return a.blockPlain
    }

    static func singleQuotedAllowed(_ s: String) -> Bool { analyse(s).single }

    struct Analysis { var blockPlain: Bool; var single: Bool }

    static func analyse(_ text: String) -> Analysis {
        let chars = Array(text.unicodeScalars)
        var block = false, lineBreaks = false, special = false
        var leadingSpace = false, leadingBreak = false, trailingSpace = false, trailingBreak = false
        var breakSpace = false, spaceBreak = false
        if text.hasPrefix("---") || text.hasPrefix("...") { block = true }
        let space: Set<Unicode.Scalar> = ["\0", " ", "\t", "\r", "\n", "\u{85}", "\u{2028}", "\u{2029}"]
        let breaks: Set<Unicode.Scalar> = ["\n", "\u{85}", "\u{2028}", "\u{2029}"]
        var preceded = true
        var followed = chars.count == 1 || space.contains(chars[1])
        var previousSpace = false, previousBreak = false
        for (index, ch) in chars.enumerated() {
            if index == 0 {
                if "#,[]{}&*!|>'\"%@`".unicodeScalars.contains(ch) { block = true }
                if (ch == "?" || ch == ":") && followed { block = true }
                if ch == "-" && followed { block = true }
            } else {
                if ch == ":" && followed { block = true }
                if ch == "#" && preceded { block = true }
            }
            if breaks.contains(ch) { lineBreaks = true }
            if !(ch == "\n" || (0x20...0x7E).contains(ch.value)) {
                let v = ch.value
                let printable =
                    v == 0x85 || (0xA0...0xD7FF).contains(v) || (0xE000...0xFFFD).contains(v)
                    || (0x10000..<0x10FFFF).contains(v)
                if !printable || v == 0xFEFF { special = true }
            }
            if ch == " " {
                if index == 0 { leadingSpace = true }
                if index == chars.count - 1 { trailingSpace = true }
                if previousBreak { breakSpace = true }
                previousSpace = true; previousBreak = false
            } else if breaks.contains(ch) {
                if index == 0 { leadingBreak = true }
                if index == chars.count - 1 { trailingBreak = true }
                if previousSpace { spaceBreak = true }
                previousSpace = false; previousBreak = true
            } else {
                previousSpace = false; previousBreak = false
            }
            preceded = space.contains(ch)
            followed = index + 2 >= chars.count || space.contains(chars[index + 2])
        }
        var blockPlain = true, single = true
        if leadingSpace || leadingBreak || trailingSpace || trailingBreak { blockPlain = false }
        if breakSpace { blockPlain = false; single = false }
        if spaceBreak || special { blockPlain = false; single = false }
        if lineBreaks { blockPlain = false }
        if block { blockPlain = false }
        return Analysis(blockPlain: blockPlain, single: single)
    }

    // YAML 1.1 implicit types PyYAML's resolver would read as something other than a string.
    static func resolvesToNonString(_ s: String) -> Bool {
        let bools = [
            "yes", "Yes", "YES", "no", "No", "NO", "true", "True", "TRUE", "false", "False", "FALSE",
            "on", "On", "ON", "off", "Off", "OFF"
        ]
        if bools.contains(s) || ["~", "null", "Null", "NULL", "="].contains(s) { return true }
        let patterns = [
            #"^[-+]?0b[0-1_]+$"#, #"^[-+]?0[0-7_]+$"#, #"^[-+]?(?:0|[1-9][0-9_]*)$"#, #"^[-+]?0x[0-9a-fA-F_]+$"#,
            #"^[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+$"#,
            #"^[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?$"#, #"^\.[0-9_]+(?:[eE][-+][0-9]+)?$"#,
            #"^[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*$"#, #"^[-+]?\.(?:inf|Inf|INF)$"#, #"^\.(?:nan|NaN|NAN)$"#,
            #"^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$"#,
            #"^[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?$"#
        ]
        return patterns.contains { s.range(of: $0, options: .regularExpression) != nil }
    }

    // MARK: Writers

    // `write_plain` after `key:` at `column`; returns the text including the leading space.
    static func plain(_ text: String, column start: Int) -> String {
        let chars = Array(text.unicodeScalars)
        var out = " "
        var column = start + 1
        var spaces = false
        var begin = 0, end = 0
        while end <= chars.count {
            let ch: Unicode.Scalar? = end < chars.count ? chars[end] : nil
            if spaces {
                if ch != " " {
                    if begin + 1 == end && column > bestWidth {
                        out += "\n  "
                        column = 2
                    } else {
                        let data = String(String.UnicodeScalarView(chars[begin..<end]))
                        out += data
                        column += end - begin
                    }
                    begin = end
                }
            } else if ch == nil || ch == " " {
                let data = String(String.UnicodeScalarView(chars[begin..<end]))
                out += data
                column += end - begin
                begin = end
            }
            if let ch { spaces = ch == " " }
            end += 1
        }
        return out
    }

    // `write_single_quoted` starting at `column` (the column of the opening quote).
    static func singleQuoted(_ text: String, column start: Int) -> String {
        let chars = Array(text.unicodeScalars)
        var out = "'"
        var column = start + 1
        var spaces = false
        var begin = 0, end = 0
        while end <= chars.count {
            let ch: Unicode.Scalar? = end < chars.count ? chars[end] : nil
            if spaces {
                if ch == nil || ch != " " {
                    if begin + 1 == end && column > bestWidth && begin != 0 && end != chars.count {
                        out += "\n  "
                        column = 2
                    } else {
                        out += String(String.UnicodeScalarView(chars[begin..<end]))
                        column += end - begin
                    }
                    begin = end
                }
            } else if ch == nil || ch == " " || ch == "'" {
                if begin < end {
                    out += String(String.UnicodeScalarView(chars[begin..<end]))
                    column += end - begin
                    begin = end
                }
            }
            if ch == "'" {
                out += "''"
                column += 2
                begin = end + 1
            }
            if let ch { spaces = ch == " " }
            end += 1
        }
        return out + "'"
    }

    // Rare: control characters. JSON escaping is close enough to PyYAML's double-quoted style here.
    static func doubleQuoted(_ text: String) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: [text], options: [.withoutEscapingSlashes])) ?? Data()
        let json = String(decoding: data, as: UTF8.self)
        return String(json.dropFirst().dropLast())
    }
}
