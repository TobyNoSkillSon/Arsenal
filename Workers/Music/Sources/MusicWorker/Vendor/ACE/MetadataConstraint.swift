//
//  MetadataConstraint.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  Upstream's MetadataConstrainedLogitsProcessor (acestep/constrained_
//  logits_processor.py at ca1e85f), the parts the text-to-music path runs:
//  a finite-state machine that walks the planner through
//
//      <think>
//      bpm: 30–300
//      caption: free text (no audio codes, no backticks, ≤ 512 tokens)
//      duration: 10–600
//      keyscale: A–G, optional ♯/♭/#/b, major or minor
//      language: one of the 51 codes
//      timesignature: 2, 3, 4 or 6
//      </think>
//
//  by masking every token the format does not allow, injecting the values
//  the caller fixed, and then, in the codes phase, allowing only audio-code
//  tokens until exactly duration × 5 have been written. Genres are skipped,
//  as upstream skips them. The token-level rules (longest single-token
//  prefix for fixed strings, prefix trees built from "field: value"
//  tokenisations, greedy first language token, caption-ending detection)
//  are upstream's, kept in the same order.
//

import Foundation
import MLX

// The planner's decoding states.
enum MetadataState: Equatable, Sendable {
    case thinkTag, newlineAfterThink
    case bpmName, bpmValue
    case captionName, captionValue
    case durationName, durationValue
    case keyscaleName, keyscaleValue
    case languageName, languageValue
    case timesigName, timesigValue
    case thinkEndTag, codesGeneration, completed
}

// The token facts the state machine needs, computed once per tokenizer.
struct MetadataVocabulary {
    let vocabularySize: Int
    let newline: Int
    let backtick: Int?
    let eos: Int
    // Every `<|audio_code_N|>` with N in 0…63999.
    let audioCodes: [Int]
    // token id → code value, for every valid audio-code token.
    let codeValue: [Int: Int]
    let bpmTree: [[Int]: Set<Int>]
    let durationTree: [[Int]: Set<Int>]
    let timesigTree: [[Int]: Set<Int>]
    let keyscaleTree: [[Int]: Set<Int>]
    let languageTree: [[Int]: Set<Int>]
    // `[1, V]` additive: -inf on every audio code (the caption may not write codes).
    let audioCodeBlock: MLXArray
    // `encode` and `decode` without special tokens.
    let encode: @Sendable (String) -> [Int]
    let decode: @Sendable ([Int]) -> String
    // The longest single-token prefix of each fixed string's remainder, memoised by remainder.
    private let fixedCache = FixedStringCache()

    init(
        vocabularySize: Int, eos: Int, audioCodeIDs: [String: Int],
        encode: @escaping @Sendable (String) -> [Int], decode: @escaping @Sendable ([Int]) -> String
    ) {
        self.vocabularySize = vocabularySize
        self.eos = eos
        self.encode = encode
        self.decode = decode
        newline = encode("\n").last ?? 198
        backtick = encode("`").last
        var codes: [Int] = []
        var values: [Int: Int] = [:]
        for (text, id) in audioCodeIDs where id < vocabularySize && text.hasSuffix("|>") {
            let digits = text.dropFirst("<|audio_code_".count).dropLast(2)
            guard let value = Int(digits), (0...MetadataLimits.maxAudioCode).contains(value) else { continue }
            codes.append(id)
            values[id] = value
        }
        audioCodes = codes.sorted()
        var block = [Float](repeating: 0, count: vocabularySize)
        for id in codes { block[id] = -.infinity }
        audioCodeBlock = MLXArray(block, [1, vocabularySize])
        codeValue = values
        let newline = self.newline
        func tree(_ values: [String], field: String, keyscale: Bool = false) -> [[Int]: Set<Int>] {
            let context = encode("\(field):")
            var tree: [[Int]: Set<Int>] = [:]
            for value in values {
                let ids = encode("\(field): \(value)")
                guard ids.count >= context.count, Array(ids.prefix(context.count)) == context else { continue }
                let tokens = Array(ids.dropFirst(context.count))
                guard !tokens.isEmpty || !keyscale else { continue }
                if keyscale {
                    let first = decode([tokens[0]]).trimmingCharacters(in: .whitespaces).uppercased().first
                    guard let first, "ABCDEFG".contains(first) else { continue }
                }
                for i in 0...tokens.count {
                    let prefix = Array(tokens.prefix(i))
                    tree[prefix, default: []].insert(i < tokens.count ? tokens[i] : newline)
                }
            }
            return tree
        }
        bpmTree = tree((MetadataLimits.bpm).map(String.init), field: "bpm")
        durationTree = tree((MetadataLimits.duration).map(String.init), field: "duration")
        timesigTree = tree(MetadataLimits.timeSignatures.map(String.init), field: "timesignature")
        keyscaleTree = tree(MetadataLimits.keyscales, field: "keyscale", keyscale: true)
        languageTree = tree(MetadataLimits.languages, field: "language")
    }

    // The token for the longest prefix of `remaining` that is a single token, else
    // every first token that starts a prefix, longest first — upstream's rule.
    func fixedTokens(_ remaining: String) -> [Int] {
        if let cached = fixedCache.get(remaining) { return cached }
        let chars = Array(remaining)
        var result: [Int] = []
        for end in stride(from: chars.count, through: 1, by: -1) {
            let ids = encode(String(chars[0..<end]))
            if ids.count == 1 { result = ids; break }
        }
        if result.isEmpty {
            var best: [Int: Int] = [:]
            for end in 1..<min(chars.count + 1, 20) {
                let prefix = String(chars[0..<end])
                guard let first = encode(prefix).first else { continue }
                let decoded = decode([first]).drop { $0 == " " }.lowercased()
                let wanted = prefix.drop { $0 == " " }.lowercased()
                if decoded.hasPrefix(wanted) || wanted.hasPrefix(decoded) { best[first] = max(best[first] ?? 0, end) }
            }
            result = best.sorted { $0.value > $1.value }.map(\.key)
        }
        fixedCache.set(remaining, result)
        return result
    }
}

// A lock-guarded memo for fixed-string lookups (the vocabulary is shared, read-mostly).
final class FixedStringCache {
    private var store: [String: [Int]] = [:]
    private let lock = NSLock()
    func get(_ key: String) -> [Int]? { lock.withLock { store[key] } }
    func set(_ key: String, _ value: [Int]) { lock.withLock { store[key] = value } }
}

// The ranges and vocabularies the planner's fields are held to.
public enum MetadataLimits {
    public static let bpm = 30...300
    public static let duration = 10...600
    public static let timeSignatures = [2, 3, 4, 6]
    public static let maxAudioCode = 63_999
    public static let notes = ["A", "B", "C", "D", "E", "F", "G"]
    public static let accidentals = ["", "#", "b", "♯", "♭"]
    public static let modes = ["major", "minor"]
    public static var keyscales: [String] {
        notes.flatMap { n in accidentals.flatMap { a in modes.map { "\(n)\(a) \($0)" } } }
    }
    public static let languages = [
        "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
        "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
        "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
        "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
        "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
        "unknown"
    ]
}

// One planner run's constraint state.
final class MetadataConstraint {

    // Fields the caller fixed; each is injected rather than generated.
    struct Fixed: Equatable {
        var bpm: String?
        var caption: String?
        var duration: String?
        var keyscale: String?
        var language: String?
        var timesignature: String?

        func value(_ field: String) -> String? {
            switch field {
            case "bpm": bpm
            case "caption": caption
            case "duration": duration
            case "keyscale": keyscale
            case "language": language
            case "timesignature": timesignature
            default: nil
            }
        }
    }

    let vocabulary: MetadataVocabulary
    let enabled: Bool
    let fixed: Fixed
    let skipCaption: Bool
    let skipLanguage: Bool
    let stopAtReasoning: Bool
    // Audio codes required before EOS is allowed (codes phase).
    let targetCodes: Int?

    private(set) var state: MetadataState = .thinkTag
    private var position = 0
    private var accumulated: [Int] = []
    private var accumulatedText = ""
    private var queue: [Int] = []
    private var queueField: String?
    private var captionAfterNewline = false
    private var captionTokens = 0
    private var captionEnding = false
    private var pendingField = ""
    private(set) var codesCount = 0

    init(
        vocabulary: MetadataVocabulary, enabled: Bool, fixed: Fixed, skipCaption: Bool, skipLanguage: Bool,
        stopAtReasoning: Bool, targetDuration: Double?
    ) {
        self.vocabulary = vocabulary
        self.enabled = enabled
        self.fixed = fixed
        self.skipCaption = skipCaption
        self.skipLanguage = skipLanguage
        self.stopAtReasoning = stopAtReasoning
        targetCodes = targetDuration.flatMap { $0 > 0 ? Int($0 * 5) : nil }
    }

    // The codes phase starts inside the codes, its prompt already holding `</think>`.
    func beginCodes() {
        state = .codesGeneration
        codesCount = 0
    }

    // MARK: Order

    private static let fixedStrings: [MetadataState: String] = [
        .thinkTag: "<think>", .newlineAfterThink: "\n", .bpmName: "bpm:", .captionName: "caption:",
        .durationName: "duration:", .keyscaleName: "keyscale:", .languageName: "language:",
        .timesigName: "timesignature:", .thinkEndTag: "</think>"
    ]

    private func nextField(after field: String) -> MetadataState {
        let order = ["bpm", "caption", "duration", "keyscale", "language", "timesignature"]
        guard let index = order.firstIndex(of: field) else { return .thinkEndTag }
        for next in order[(index + 1)...] {
            if next == "caption" && skipCaption { continue }
            if next == "language" && skipLanguage { continue }
            return Self.nameState(next)
        }
        return .thinkEndTag
    }

    private static func nameState(_ field: String) -> MetadataState {
        switch field {
        case "bpm": .bpmName
        case "caption": .captionName
        case "duration": .durationName
        case "keyscale": .keyscaleName
        case "language": .languageName
        default: .timesigName
        }
    }

    private func next(of current: MetadataState) -> MetadataState? {
        switch current {
        case .thinkTag: .newlineAfterThink
        case .newlineAfterThink: .bpmName
        case .bpmName: .bpmValue
        case .bpmValue: nextField(after: "bpm")
        case .captionName: skipCaption ? nil : .captionValue
        case .captionValue: skipCaption ? nil : nextField(after: "caption")
        case .durationName: .durationValue
        case .durationValue: nextField(after: "duration")
        case .keyscaleName: .keyscaleValue
        case .keyscaleValue: nextField(after: "keyscale")
        case .languageName: skipLanguage ? nil : .languageValue
        case .languageValue: skipLanguage ? nil : nextField(after: "language")
        case .timesigName: .timesigValue
        case .timesigValue: .thinkEndTag
        case .thinkEndTag: .codesGeneration
        case .codesGeneration: .completed
        case .completed: nil
        }
    }

    private func transition() {
        guard let next = next(of: state) else { return }
        state = next
        position = 0
        accumulated = []
        accumulatedText = ""
        captionAfterNewline = false
        captionTokens = 0
        captionEnding = false
        pendingField = ""
    }

    // MARK: Masking

    // What upstream calls the processor on the logits row `[1, V]` (float32).
    // Only the reads the rules need (an argmax, a few gathered scores) leave the GPU.
    func constrain(_ logits: MLXArray, allowedCodes: MLXArray) -> MLXArray {
        guard enabled, state != .completed else { return logits }
        if state == .codesGeneration { return codesMask(logits, allowedMask: allowedCodes) }
        return process(logits)
    }

    // The codes phase: audio codes and EOS only; EOS blocked until the count, then forced.
    func codesMask(_ logits: MLXArray, allowedMask: MLXArray) -> MLXArray {
        var out = logits + allowedMask
        if let target = targetCodes {
            let eos = vocabulary.eos
            if codesCount < target {
                out[0..., eos] = MLXArray(-Float.infinity)
            } else {
                out = whitelist(out, [eos])
            }
        }
        return out
    }

    // swiftlint:disable:next inclusive_language
    private func whitelist(_ logits: MLXArray, _ allowed: [Int]) -> MLXArray {
        let out = MLX.full(logits.shape, values: MLXArray(-Float.infinity), dtype: logits.dtype)
        guard !allowed.isEmpty else { return out }
        let index = MLXArray(allowed.map(Int32.init))
        out[0..., index] = logits[0..., index]
        return out
    }

    private func beginInjection(_ field: String, _ logits: MLXArray) -> MLXArray? {
        guard let value = fixed.value(field), queue.isEmpty else { return nil }
        let tokens = vocabulary.encode(" \(value)\n")
        guard let first = tokens.first else { return nil }
        queue = tokens
        queueField = field
        return whitelist(logits, [first])
    }

    private func process(_ logits: MLXArray) -> MLXArray {
        if let next = queue.first { return whitelist(logits, [next]) }
        if let fixedString = Self.fixedStrings[state] {
            let remaining = String(Array(fixedString).dropFirst(position))
            let allowed = remaining.isEmpty ? [] : vocabulary.fixedTokens(remaining)
            if !allowed.isEmpty {
                if state == .thinkEndTag && stopAtReasoning && fixedString.count - position <= 10 {
                    return whitelist(logits, [vocabulary.eos])
                }
                return whitelist(logits, allowed)
            }
            if state == .thinkEndTag && stopAtReasoning { return whitelist(logits, [vocabulary.eos]) }
            transition()
            if Self.fixedStrings[state] != nil { return logits }
            return process(MLX.zeros(like: logits))
        }
        switch state {
        case .bpmValue:
            if accumulated.isEmpty, let injected = beginInjection("bpm", logits) { return injected }
            return whitelist(logits, Array(vocabulary.bpmTree[accumulated] ?? []))
        case .durationValue:
            if accumulated.isEmpty, let injected = beginInjection("duration", logits) { return injected }
            return whitelist(logits, Array(vocabulary.durationTree[accumulated] ?? []))
        case .timesigValue:
            if accumulated.isEmpty, let injected = beginInjection("timesignature", logits) { return injected }
            return completeOrContinue(logits, tree: vocabulary.timesigTree, forceNewlineWhenEmpty: false)
        case .keyscaleValue:
            if accumulated.isEmpty, let injected = beginInjection("keyscale", logits) { return injected }
            return completeOrContinue(logits, tree: vocabulary.keyscaleTree, forceNewlineWhenEmpty: true)
        case .languageValue:
            if accumulated.isEmpty, let injected = beginInjection("language", logits) { return injected }
            if accumulated.isEmpty {
                let candidates = Array(vocabulary.languageTree[[]] ?? [])
                guard !candidates.isEmpty else { return whitelist(logits, [vocabulary.newline]) }
                let scores = logits[0..., MLXArray(candidates.map(Int32.init))].reshaped(-1).asArray(Float.self)
                var best = 0
                for i in scores.indices where scores[i] > scores[best] { best = i }
                return whitelist(logits, [candidates[best]])
            }
            return completeOrContinue(logits, tree: vocabulary.languageTree, forceNewlineWhenEmpty: true)
        case .captionValue:
            if accumulatedText.isEmpty, let injected = beginInjection("caption", logits) { return injected }
            return caption(logits)
        default:
            return logits
        }
    }

    // keyscale / language / timesignature: newline only once complete, else the continuations.
    private func completeOrContinue(_ logits: MLXArray, tree: [[Int]: Set<Int>], forceNewlineWhenEmpty: Bool) -> MLXArray {
        let options = tree[accumulated] ?? []
        if options.contains(vocabulary.newline) { return whitelist(logits, [vocabulary.newline]) }
        if !options.isEmpty || !forceNewlineWhenEmpty { return whitelist(logits, Array(options)) }
        return whitelist(logits, [vocabulary.newline])
    }

    private func caption(_ logits: MLXArray) -> MLXArray {
        if captionAfterNewline {
            let top = MLX.argMax(logits, axis: -1).item(Int.self)
            let text = vocabulary.decode([top])
            captionAfterNewline = false
            if let first = text.first, first != " " && first != "\t" {
                captionEnding = true
                pendingField = ""
                return logits
            }
        }
        if captionEnding { return logits }
        var out = logits + vocabulary.audioCodeBlock
        if let backtick = vocabulary.backtick { out[0..., backtick] = MLXArray(-Float.infinity) }
        if captionTokens >= 512 { out = whitelist(out, [vocabulary.newline]) }
        return out
    }

    // MARK: Advancing

    // Upstream's `update_state` after a token is chosen.
    func update(with token: Int) {
        guard enabled, state != .completed else { return }
        if state == .codesGeneration {
            codesCount += 1
            return
        }
        if !queue.isEmpty {
            queue.removeFirst()
            if queue.isEmpty, let field = queueField {
                queueField = nil
                state = nextField(after: field)
                position = 0
                accumulated = []
                accumulatedText = ""
            }
            return
        }
        let text = vocabulary.decode([token])
        if let fixedString = Self.fixedStrings[state] {
            position += text.count
            if position >= fixedString.count {
                if state == .thinkEndTag && stopAtReasoning {
                    state = .completed
                    position = 0
                } else {
                    transition()
                }
            }
            return
        }
        switch state {
        case .bpmValue, .durationValue, .timesigValue, .keyscaleValue, .languageValue:
            if token == vocabulary.newline {
                transition()
            } else {
                accumulated.append(token)
                accumulatedText += text
            }
        case .captionValue:
            captionTokens += 1
            accumulatedText += text
            captionAfterNewline = text.contains("\n")
            if captionEnding {
                pendingField += text
                if text.contains(":") {
                    let name = pendingField.trimmingCharacters(in: .whitespacesAndNewlines)
                        .trimmingCharacters(in: CharacterSet(charactersIn: ":")).trimmingCharacters(in: .whitespaces)
                        .lowercased()
                    let target: MetadataState? =
                        switch name {
                        case "duration": .durationValue
                        case "keyscale": .keyscaleValue
                        case "language": .languageValue
                        case "timesignature": .timesigValue
                        default: nil
                        }
                    if let target {
                        state = target
                        position = 0
                        accumulated = []
                        accumulatedText = ""
                        captionEnding = false
                        pendingField = ""
                    } else {
                        captionEnding = false
                        pendingField = ""
                        transition()
                    }
                }
            }
        default:
            break
        }
    }
}
