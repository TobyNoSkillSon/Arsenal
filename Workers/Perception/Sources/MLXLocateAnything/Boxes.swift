// SPDX-License-Identifier: MIT
import Foundation

public enum LocateAnythingOutputError: Error, Equatable, LocalizedError {
    case incompleteGeneration
    case malformedGrammar
    case repetition

    public var errorDescription: String? {
        switch self {
        case .incompleteGeneration: return "LocateAnything generation reached its token limit before termination; increase max_tokens."
        case .malformedGrammar: return "LocateAnything returned malformed grounding output; expected complete reference and box tokens."
        case .repetition: return "LocateAnything generation stopped because its output was repeating (finish_reason repetition)."
        }
    }
}

/// Arsenal: stops a looping generation instead of decoding to max_tokens. The output's tail is a block of p ≤ 128 tokens
/// repeated more than ⌊4·(1 + 3/p)⌋ times (Chandra's `detect_repeat_token` rule, in tokens) over at least `span`
/// tokens, or the last `span` tokens use at most 3 distinct tokens (the Qwen3-ASR backstop). Distinct boxes differ in
/// their coordinate tokens, so a long detection list never repeats; the same box written again and again does.
public struct LocateAnythingRepetitionGuard {
    public static let maxPeriod = 128
    public let span: Int
    private var tokens: [Int] = []
    private var run = Array(repeating: 0, count: maxPeriod + 1)
    private var window: [Int: Int] = [:]

    public init(span: Int = 256) { self.span = span }

    /// Appends generated tokens; true once the output loops.
    public mutating func append(_ new: [Int]) -> Bool {
        var looping = false
        for token in new {
            tokens.append(token)
            let count = tokens.count
            for p in stride(from: min(Self.maxPeriod, count - 1), through: 1, by: -1) {
                run[p] = tokens[count - 1 - p] == token ? run[p] + 1 : 0
                let tail = run[p] + p
                if tail >= span, tail >= p * (Int(4 * (1 + 3 / Double(p))) + 1) { looping = true }
            }
            window[token, default: 0] += 1
            if count > span {
                let old = tokens[count - 1 - span]
                if window[old] == 1 { window[old] = nil } else { window[old]! -= 1 }
            }
            if count >= span, window.count <= 3 { looping = true }
        }
        return looping
    }
}

public enum LocateAnythingBoxes {
    /// Generation must end explicitly, including when its final box happens to be complete at the cap.
    public static func completed(_ tokens: [Int], config: LocateAnythingConfiguration) throws -> [Int] {
        guard let end = tokens.last, config.eosTokenIds.contains(end) else {
            throw LocateAnythingOutputError.incompleteGeneration
        }
        let body = Array(tokens.dropLast())
        guard !body.contains(where: config.eosTokenIds.contains) else { throw LocateAnythingOutputError.malformedGrammar }
        return body
    }

    /// Parse complete reference/box blocks. An explicit `<box>None</box>` is a negative result;
    /// missing delimiters, stray tokens and partial coordinates are errors, never empty detections.
    public static func parse(_ tokens: [Int], config: LocateAnythingConfiguration, imageWidth: Int, imageHeight: Int, decode: ([Int]) -> String) throws -> [DetectedBox] {
        var boxes: [DetectedBox] = []
        var label: String?
        var pendingReference = false
        var sawBox = false
        var index = 0
        while index < tokens.count {
            let token = tokens[index]
            if token == config.textConfig.nullTokenId {
                index += 1
                continue
            }
            if token == config.refStartTokenId {
                guard !pendingReference, let end = tokens[(index + 1)...].firstIndex(of: config.refEndTokenId), end > index + 1 else {
                    throw LocateAnythingOutputError.malformedGrammar
                }
                let text = Array(tokens[(index + 1)..<end])
                let special = [config.refStartTokenId, config.boxStartTokenId, config.boxEndTokenId, config.textConfig.nullTokenId]
                guard !text.contains(where: { special.contains($0) || (config.coordStartTokenId...config.coordEndTokenId).contains($0) }) else {
                    throw LocateAnythingOutputError.malformedGrammar
                }
                label = decode(text)
                pendingReference = true
                index = end + 1
                continue
            }
            guard token == config.boxStartTokenId,
                let end = tokens[(index + 1)...].firstIndex(of: config.boxEndTokenId)
            else {
                throw LocateAnythingOutputError.malformedGrammar
            }
            let content = Array(tokens[(index + 1)..<end])
            if content != [config.noneTokenId] {
                guard [2, 4].contains(content.count), content.allSatisfy({ (config.coordStartTokenId...config.coordEndTokenId).contains($0) }) else {
                    throw LocateAnythingOutputError.malformedGrammar
                }
                let values = content.map { Float($0 - config.coordStartTokenId) / 1000 }
                let pixels = values.enumerated().map { $0.element * Float($0.offset.isMultiple(of: 2) ? imageWidth : imageHeight) }
                boxes.append(DetectedBox(label: label, box: values, boxPixels: pixels, isPoint: content.count == 2))
            }
            sawBox = true
            pendingReference = false
            index = end + 1
        }
        guard sawBox, !pendingReference else { throw LocateAnythingOutputError.malformedGrammar }
        return boxes
    }
}
