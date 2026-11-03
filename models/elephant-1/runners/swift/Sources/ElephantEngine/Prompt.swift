// SPDX-License-Identifier: MIT
// The maker's prompt (OneJev qev/prompt.py @81ce62f with the checkpoint's serve.patch, Apache-2.0; PROMPT_VERSION
// qev-labels-v2, default "tagged" style), rendered through the checkpoint's Qwen3.5 chat template as qev renders it
// (system + one user turn, add_generation_prompt, enable_thinking=false).
import Foundation

public enum QuestionKind: String, Sendable { case noul, choice, score }

/// One typed question after validation (qev/schema.py): instructions and option descriptions are text, an object, an
/// array or null; choice options keep the order they were sent in.
public struct Question: Sendable {
    public let kind: QuestionKind
    public let instructions: PyJSON?
    /// choice: (name, description) in request order; score: ("", level); noul: the true/false descriptions.
    public let options: [(String, PyJSON?)]
    public init(kind: QuestionKind, instructions: PyJSON?, options: [(String, PyJSON?)]) {
        self.kind = kind; self.instructions = instructions; self.options = options
    }
}

public enum MediaKind: String, Sendable { case image, video }

/// One branch: the suffix text and the answer labels its slots map to (qev RenderedQuestion).
public struct RenderedQuestion: Sendable {
    public let kind: QuestionKind
    public let labels: [String]
    public let descriptions: [PyJSON?]
    public let suffix: String
}

public enum Prompt {
    public static let version = "qev-labels-v2"
    public static let system =
        "Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning."
    static let letters = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map(String.init)
    static let notOneToken: Set<String> = [
        "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"
    ]
    /// A..Z, then two-letter labels without the 25 that are not one token, 256 in all (qev SLOT_LABELS).
    public static let slotLabels: [String] = {
        let pairs = letters.flatMap { a in letters.map { a + $0 } }.filter { !notOneToken.contains($0) }
        return letters + pairs.prefix(256 - letters.count)
    }()
    public static let maxChoiceOptions = 255
    public static let maxScoreLevels = 10
    static let noulInstructions = "Is the statement true, or is the answer to the question yes?"
    static let noulTrue = "the statement is true / the answer is yes"
    static let noulFalse = "the statement is false / the answer is no"
    public static let noneOption = "none of the options listed here"
    public static let noneDescription = "The correct option is not among the options listed in this question."
    public static let prefixMark = "QEV_PREFIX_MARK"

    /// `render_entry`: text stripped of surrounding whitespace, structured JSON as `json.dumps(indent=2)`, null as "".
    public static func entry(_ value: PyJSON?) -> String {
        guard let value, !value.isNull else { return "" }
        if let text = value.string { return pythonStrip(text) }
        return value.dumps(indent: 2)
    }

    /// `render_state`: a string as is, anything else as `json.dumps(ensure_ascii=False, indent=2)`.
    public static func state(_ value: PyJSON?) -> String {
        guard let value, !value.isNull else { return "" }
        if let text = value.string { return text }
        return value.dumps(indent: 2)
    }

    public static func stateBlock(_ value: PyJSON?) -> String { "<state>\n\(state(value))\n</state>\n\n" }

    /// `render_question` with an optional slot permutation.
    public static func render(_ q: Question, order: [Int]? = nil) throws -> RenderedQuestion {
        var labels: [String], names: [String], descriptions: [PyJSON?]
        let instructions: String
        switch q.kind {
        case .noul:
            instructions = nonEmpty(entry(q.instructions)) ?? noulInstructions
            let t: PyJSON? = q.options.first { $0.0 == "true" }?.1 ?? nil
            let f: PyJSON? = q.options.first { $0.0 == "false" }?.1 ?? nil
            // qev: a criterion given as null falls back to the default wording, like an absent one; "" stays "".
            let yes = t.map { entry($0) } ?? noulTrue
            let no = f.map { entry($0) } ?? noulFalse
            labels = ["yes", "no"]; names = ["yes", "no"]; descriptions = [.string(yes), .string(no)]
        case .choice:
            instructions = nonEmpty(entry(q.instructions)) ?? "Which option applies to the state?"
            labels = q.options.map(\.0); names = labels; descriptions = q.options.map(\.1)
        case .score:
            instructions = nonEmpty(entry(q.instructions)) ?? "Which level describes the state?"
            labels = q.options.indices.map(String.init); names = q.options.indices.map { "level \($0)" }; descriptions = q.options.map(\.1)
        }
        guard labels.count <= slotLabels.count else { throw PromptError("\(labels.count) options exceed the \(slotLabels.count) slots of prompt \(version)") }
        if let order {
            labels = order.map { labels[$0] }; names = order.map { names[$0] }; descriptions = order.map { descriptions[$0] }
        }
        let header = q.kind == .score ? "Rate the state:" : "Question:"
        let lines = zip(slotLabels, zip(names, descriptions)).map { letter, pair -> String in
            let text = entry(pair.1)
            return text.isEmpty ? "\(letter). \(pair.0)" : "\(letter). \(pair.0): \(text)"
        }
        let shown = Array(slotLabels.prefix(labels.count))
        let answer = shown.count <= letters.count ? "Answer with one letter: " : "Answer with one label: "
        let suffix = "\(header) \(instructions)\n\nOptions:\n" + lines.joined(separator: "\n") + "\n\n" + answer + shown.joined(separator: ", ") + "."
        return RenderedQuestion(kind: q.kind, labels: labels, descriptions: descriptions, suffix: suffix)
    }

    /// The checkpoint's chat template for [system, user] with add_generation_prompt and enable_thinking=false. The
    /// template trims each message's content (Jinja `trim`, Python `str.strip`).
    public static func chat(state: PyJSON?, suffix: String, images: Int = 0) throws -> String {
        try chat(state: state, suffix: suffix, media: Array(repeating: .image, count: images))
    }

    public static func chat(state: PyJSON?, suffix: String, media: [MediaKind]) throws -> String {
        var user = stateBlock(state) + suffix
        if !media.isEmpty { user = try visionUser(user, media: media) }
        return "<|im_start|>system\n" + pythonStrip(system) + "<|im_end|>\n<|im_start|>user\n" + pythonStrip(user)
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// The chat text before the question (qev `_prefix_and_suffix_ids`): the prompt rendered with the mark as its
    /// suffix, cut at the mark. The mark is the user turn's last text and the template adds only fixed tokens after
    /// it, so the last occurrence is the boundary. The maker cuts at the first (`str.index`), which a state holding the
    /// literal mark would move into the state; Arsenal cuts at the last, the same place for every other state.
    public static func prefixText(state: PyJSON?, media: [MediaKind] = []) throws -> String {
        let marked = try chat(state: state, suffix: prefixMark, media: media)
        guard let cut = marked.range(of: prefixMark, options: [.backwards, .literal]) else { throw PromptError("Prefix mark lost in the chat template.") }
        return String(marked[..<cut.lowerBound])
    }

    public static let visionBlock = "<|vision_start|><|image_pad|><|vision_end|>"
    public static let videoBlock = "<|vision_start|><|video_pad|><|vision_end|>"

    /// The maker's placeholder rule (qev/media.py content_parts, train/mm.py): `<image:N>` / `<video:N>` (N from 1) in the
    /// user text become the media items' vision blocks; every item is referenced exactly once, in order, by its own
    /// kind. Without media the text is left as written.
    public static func visionUser(_ text: String, images: Int) throws -> String {
        try visionUser(text, media: Array(repeating: .image, count: images))
    }

    /// `media` gives each item's kind in list order; `<image:N>` / `<video:N>` must name an item of that kind.
    public static func visionUser(_ text: String, media: [MediaKind]) throws -> String {
        var out = "", seen: [Int] = []
        var rest = Substring(text)
        while let open = rest.range(of: "<") {
            out += rest[..<open.lowerBound]
            let after = rest[open.upperBound...]
            if let m = placeholder(after) {
                guard let n = m.number, n >= 1, n <= media.count, media[n - 1].rawValue == m.kind else {
                    throw PromptError("placeholder <\(m.kind):\(m.number.map(String.init) ?? "…")> does not match a media item of \(media.count)")
                }
                out += media[n - 1] == .image ? visionBlock : videoBlock
                seen.append(n)
                rest = after[m.end...]
            } else {
                out += "<"
                rest = after
            }
        }
        out += rest
        guard seen == Array(1...media.count) else {
            throw PromptError("media must be referenced once each, in order, by <image:N> / <video:N>; saw \(seen) for \(media.count) items")
        }
        return out
    }

    /// `(image|video):(\d+)>` at the start of `s`, with Python's `\d` (any Unicode decimal digit) and `int()`: the
    /// number is nil when it does not fit an Int (Python still parses it, then finds no such media item).
    static func placeholder(_ s: Substring) -> (kind: String, number: Int?, end: Substring.Index)? {
        for kind in ["image", "video"] where s.hasPrefix(kind + ":") {
            let scalars = s.unicodeScalars
            var i = scalars.index(scalars.startIndex, offsetBy: kind.count + 1)
            let start = i
            var value: Int? = 0
            while i < scalars.endIndex, scalars[i].properties.numericType == .decimal, let digit = scalars[i].properties.numericValue {
                if let v = value {
                    let (m, o1) = v.multipliedReportingOverflow(by: 10)
                    let (sum, o2) = m.addingReportingOverflow(Int(digit))
                    value = o1 || o2 ? nil : sum
                }
                i = scalars.index(after: i)
            }
            guard i > start, i < scalars.endIndex, scalars[i] == ">" else { return nil }
            return (kind, value, scalars.index(after: i))
        }
        return nil
    }

    /// True when a question's own text holds a media placeholder (the maker refuses vision tokens in a question).
    public static func hasPlaceholder(_ text: String) -> Bool {
        var rest = Substring(text)
        while let open = rest.range(of: "<") {
            if placeholder(rest[open.upperBound...]) != nil { return true }
            rest = rest[open.upperBound...]
        }
        return false
    }

    static func nonEmpty(_ s: String) -> String? { s.isEmpty ? nil : s }

    /// Python's `str.strip()` with no argument: Unicode whitespace (str.isspace) at both ends.
    public static func pythonStrip(_ s: String) -> String {
        let scalars = Array(s.unicodeScalars)
        var start = 0, end = scalars.count
        while start < end, isPythonSpace(scalars[start]) { start += 1 }
        while end > start, isPythonSpace(scalars[end - 1]) { end -= 1 }
        var out = String.UnicodeScalarView()
        out.append(contentsOf: scalars[start..<end])
        return String(out)
    }

    /// `str.isspace` for one code point: bidirectional class WS/B/S or category Zs (CPython's _PyUnicode_IsWhitespace).
    static func isPythonSpace(_ c: Unicode.Scalar) -> Bool {
        switch c.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000: return true
        default: return false
        }
    }
}

public struct PromptError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}
