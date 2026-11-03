// SPDX-License-Identifier: LicenseRef-LFM-Open-1.0
// Modified by Arsenal: a Swift port of LiquidAI d1-3B's `prompt.py` and `api.py` (checkpoint revision
// 051bcc464b01b9f92942b364d9586b0ef5912432; licence in ../../LICENSE-LFM-Open-1.0). Defaults kept: no system turn, the
// `json_only` state style, option style `desc`, no lead. Python's str(), repr and json.dumps are reproduced in PyJSON.
import Foundation

public enum QuestionKind: String, Sendable { case noul, choice, score }

/// One question in the Decision Index's JSON (`type`, `instructions`, `criteria`), as `prompt.as_question` reads it.
public struct Question: Sendable {
    public let kind: QuestionKind
    public let instructions: PyJSON
    /// Choice: label -> description (null allowed). Noul: optional `true`/`false` descriptions.
    public let options: [(String, PyJSON)]
    /// Noul's raw criteria (`if q.criteria:`); nil when absent or null.
    public let noulCriteria: PyJSON?
    /// Score: ordered level descriptions.
    public let levels: [PyJSON]

    public init(kind: QuestionKind, instructions: PyJSON, options: [(String, PyJSON)] = [], noulCriteria: PyJSON? = nil, levels: [PyJSON] = []) {
        self.kind = kind
        self.instructions = instructions
        self.options = options
        self.noulCriteria = noulCriteria
        self.levels = levels
    }
}

public enum Prompt {
    public static let imStart = "<|im_start|>"
    public static let imEnd = "<|im_end|>"

    // MARK: state and question text

    /// `state_block(state, "json_only")`.
    public static func stateBlock(_ state: PyJSON) -> String {
        if case .string(let s) = state { return s + "\n\n" }
        return state.dumps(indent: 2) + "\n\n"
    }

    /// `prefix_text`: BOS, no system turn, the user turn with the pictures' markup, then the state (if any).
    public static func prefix(_ state: PyJSON?, bos: String, images: String = "") -> String {
        let body = state.map { stateBlock($0) + "\nQUESTION:\n" } ?? ""
        return bos + imStart + "user\n" + images + body
    }

    /// `suffix_text`: the question and the assistant header, up to the answer slot.
    public static func suffix(_ q: Question, codes: [String]) -> String {
        questionBlock(q, codes: codes) + imEnd + "\n" + imStart + "assistant\n"
    }

    static func questionBlock(_ q: Question, codes: [String]) -> String {
        switch q.kind {
        case .choice:
            let lines = q.options.enumerated().map { i, option in
                let text = option.1.truthy ? option.1.pyStr() : option.0.replacingOccurrences(of: "_", with: " ")
                return codes[i] + " " + text
            }
            return q.instructions.pyStr() + "\n\nOptions:\n" + lines.joined(separator: "\n") + "\n\nReply with the option code only."
        case .noul:
            var extra = ""
            if let c = q.noulCriteria, c.truthy {
                extra = "\nYes: " + (c["true"]?.pyStr() ?? "None") + "\nNo: " + (c["false"]?.pyStr() ?? "None")
            }
            return q.instructions.pyStr() + extra + "\n\nReply with yes or no only."
        case .score:
            let legend = q.levels.enumerated().map { "\($0.offset) " + $0.element.pyStr() }.joined(separator: "\n")
            return q.instructions.pyStr() + "\n\n" + legend + "\n\nReply with a single digit 0-\(q.levels.count - 1) only."
        }
    }

    /// `option_codes`: native letters when the labels already are single letters, else A..Z, else 00...
    static func optionCodes(_ labels: [String]) -> [String] {
        let labs = labels.map(pyStrip)
        if !labs.isEmpty && labs.allSatisfy({ $0.unicodeScalars.count == 1 && isAlpha($0.unicodeScalars.first!) }) { return labs }
        if labs.count <= 26 { return (0..<labs.count).map { String(UnicodeScalar(65 + $0)!) } }
        return (0..<labs.count).map { String(format: "%02d", $0) }
    }

    static func isAlpha(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }

    /// Python's `str.strip()`: whitespace by `str.isspace` (Unicode White_Space plus the separators U+001C-U+001F).
    static func pyStrip(_ s: String) -> String {
        let space = { (c: Unicode.Scalar) in c.properties.isWhitespace || (0x1C...0x1F).contains(c.value) }
        let scalars = Array(s.unicodeScalars)
        guard let first = scalars.firstIndex(where: { !space($0) }), let last = scalars.lastIndex(where: { !space($0) }) else { return "" }
        var out = ""
        out.unicodeScalars.append(contentsOf: scalars[first...last])
        return out
    }

    /// `_FALLBACK_POOL`.
    static let fallbackPool: [String] = {
        var pool = (65...90).map { String(UnicodeScalar($0)!) }
        pool += (0..<100).map { String(format: "%02d", $0) }
        pool += (97...122).map { String(UnicodeScalar($0)!) }
        pool += (0..<200).map { "#\($0)" }
        for a in 65...90 { for b in 65...90 { pool.append(String(UnicodeScalar(a)!) + String(UnicodeScalar(b)!)) } }
        return pool
    }()
}

/// The tokenizer-dependent parts of the prompt (option aliases, readout ids), memoised like the vendor's lru_caches.
public final class Verbalizer: @unchecked Sendable {
    public let tokenizer: LFMTokenizer
    private let lock = NSLock()
    private var aliasCache: [[Data]: [(String, Int)]] = [:]
    private var singleCache: [Data: Int?] = [:]

    public init(tokenizer: LFMTokenizer) { self.tokenizer = tokenizer }

    private func single(_ text: String) -> Int? {
        lock.lock()
        if let hit = singleCache[Data(text.utf8)] {
            lock.unlock()
            return hit
        }
        lock.unlock()
        let ids = tokenizer.encode(text)
        let value = ids.count == 1 ? ids[0] : nil
        lock.lock()
        singleCache[Data(text.utf8)] = value
        lock.unlock()
        return value
    }

    /// `aliases`: every label a distinct single-token code, [(code, token id)].
    public func aliases(_ labels: [String]) throws -> [(String, Int)] {
        let codes = Prompt.optionCodes(labels)
        let key = codes.map { Data($0.utf8) }
        lock.lock()
        if let hit = aliasCache[key] {
            lock.unlock()
            return hit
        }
        lock.unlock()
        var used = Set<Int>()
        var out: [(String, Int)] = []
        func take(_ raw: String) -> Bool {
            guard let id = single(raw), !used.contains(id) else { return false }
            out.append((raw, id))
            used.insert(id)
            return true
        }
        for code in codes where !take(code) {
            guard Prompt.fallbackPool.contains(where: { take($0) }) else {
                throw JSONError("no single-token alias left for \(codes.count) options")
            }
        }
        lock.lock()
        aliasCache[key] = out
        lock.unlock()
        return out
    }

    /// `_ids`: distinct single-token ids of `texts`, in order.
    private func ids(_ texts: [String]) -> [Int] {
        var out: [Int] = []
        for t in texts { if let id = single(t), !out.contains(id) { out.append(id) } }
        return out
    }

    public func codes(_ q: Question) throws -> [String] {
        q.kind == .choice ? try aliases(q.options.map(\.0)).map(\.0) : []
    }

    public func suffix(_ q: Question) throws -> String { Prompt.suffix(q, codes: try codes(q)) }

    /// `readout_ids`: token ids per option, max-pooled.
    public func readoutIDs(_ q: Question) throws -> [[Int]] {
        switch q.kind {
        case .noul:
            let yes = ids(["yes", "Yes", "YES"])
            let no = ids(["no", "No", "NO"])
            guard !yes.isEmpty, !no.isEmpty else { throw JSONError("tokenizer has no single-token yes/no") }
            return [yes, no]
        case .score:
            let groups = (0..<q.levels.count).map { ids([String($0)]) }
            guard groups.allSatisfy({ !$0.isEmpty }) else {
                throw JSONError("score with \(q.levels.count) levels needs single-token digits; the primitive is defined for 2 to 10")
            }
            return groups
        case .choice:
            let groups = try aliases(q.options.map(\.0)).map { code, id in [id] + ids([" " + code]).filter { $0 != id } }
            guard !groups.isEmpty else { throw JSONError("choice with no options") }
            return groups
        }
    }
}

public enum Readout {
    /// `readout`: a softmax over each option's best token logit (logits, not log-probabilities: the shift cancels).
    public static func probabilities(_ groups: [[Int]], _ logit: (Int) -> Double) -> [Double] {
        let scores = groups.map { $0.map(logit).max()! }
        let m = scores.max()!
        let exps = scores.map { Foundation.exp($0 - m) }
        let total = exps.reduce(0, +)
        return exps.map { $0 / total }
    }

    /// `api.answer`: a noul's P(yes); a choice's pick and its probabilities; a score's expected level.
    public static func answer(_ q: Question, _ probs: [Double]) -> PyJSON {
        func number(_ d: Double) -> PyJSON { .scalar(PyJSON.floatRepr(d)) }
        if q.kind == .noul { return .object([("type", .string("noul")), ("noul", number(probs[0]))]) }
        var best = 0
        for i in probs.indices where probs[i] > probs[best] { best = i }
        if q.kind == .choice {
            let names = q.options.map(\.0)
            return .object([
                ("type", .string("choice")), ("choice", .string(names[best])), ("confidence", number(probs[best])),
                ("probabilities", .object(zip(names, probs).map { ($0, number($1)) }))
            ])
        }
        let expected = probs.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
        return .object([
            ("type", .string("score")), ("score", number(expected)), ("confidence", number(probs[best])),
            ("probabilities", .object(probs.enumerated().map { (String($0.offset), number($0.element)) })),
            ("legend", .object(q.levels.enumerated().map { (String($0.offset), $0.element) }))
        ])
    }
}
