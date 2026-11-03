// SPDX-License-Identifier: Apache-2.0
// Streaming state/prefix algorithm adapted from NetEase Confucius4-R2T2
// 26d55a54ce5670cff9947a167d8ed95d569fd4d9, r2t2/r2t2_asr.py (streaming_transcribe_no_reset).
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
import Tokenizers

final class ConfuciusSession {
    /// The maker's example.py feeds the first chunk with its 160 ms lookahead (`--lookahead_ms` default), 320 ms at the default chunk size.
    static let lookaheadSamples = 2560
    let chunkSamples: Int
    let rollback: Int
    let prompt: String
    let forcedLanguage: String?
    var pending: [Float] = []
    var window: [Float] = []
    var entries: [(samples: Int, text: String)] = []
    var committed = ""
    var hypothesis = ""
    var language: String?
    var totalSamples = 0
    var decodedChunks = 0
    /// Adaptive generation budget of example.py `run_streaming`: the first chunk gets (step + lookahead) / 1280 tokens,
    /// later chunks step / 1280, growing by one per chunk whose transcript did not grow, capped at min(32, max(4, 2 * step / 1280)).
    /// The example's Chinese doubling branch reads a list it never fills, so it never runs and is not ported.
    var maxNewTokens: Int
    let firstMaxNewTokens: Int
    let baseMaxNewTokens: Int
    let maxNewTokensCap: Int
    private var longestText = 0
    private var evictedText = 0
    init(chunkMS: Int, rollback: Int, prompt: String, language: String?) {
        self.chunkSamples = chunkMS * 16; self.rollback = rollback; self.prompt = prompt; self.forcedLanguage = language
        firstMaxNewTokens = max(1, (chunkMS * 16 + Self.lookaheadSamples) / 1280)
        baseMaxNewTokens = max(1, chunkMS * 16 / 1280)
        maxNewTokensCap = min(32, max(4, 2 * (chunkMS * 16 / 1280)))
        maxNewTokens = firstMaxNewTokens
    }
    var nextChunkSamples: Int { decodedChunks == 0 ? chunkSamples + Self.lookaheadSamples : chunkSamples }
    func push(_ samples: [Float], model: Qwen3ASRModel, tokenLimit: Int, cancellation: NativeCancellation) throws {
        guard totalSamples + samples.count <= 16000 * 600 else { throw audioFailure("A transcription stream is bounded to ten minutes.") }
        totalSamples += samples.count; pending += samples
        while pending.count >= nextChunkSamples {
            let size = nextChunkSamples
            let chunk = Array(pending.prefix(size)); pending.removeFirst(size)
            try decode(chunk, final: false, model: model, tokenLimit: min(tokenLimit, maxNewTokens), cancellation: cancellation)
            adaptTokenBudget()
        }
    }
    func finish(model: Qwen3ASRModel, tokenLimit: Int, cancellation: NativeCancellation) throws {
        if !pending.isEmpty {
            let tail = pending; pending.removeAll()
            try decode(tail, final: true, model: model, tokenLimit: min(tokenLimit, firstMaxNewTokens), cancellation: cancellation)
        } else {
            commitHypothesis()
        }
    }
    func adaptTokenBudget() {
        // example.py compares the whole transcript; the rolling window evicts text, so count what left it too.
        let length = evictedText + hypothesis.count
        if length > longestText {
            longestText = length; maxNewTokens = baseMaxNewTokens
        } else {
            maxNewTokens += 1
        }
        maxNewTokens = min(maxNewTokensCap, maxNewTokens)
    }
    func commitHypothesis() {
        let prefix = entries.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
        let text = hypothesis.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix(prefix) {
            let delta = String(text.dropFirst(prefix.count))
            committed += delta
            if !entries.isEmpty { entries[entries.count - 1].text += delta }
        }
    }
    /// The finalized part of one decode. Rollback counts tokens of the transcript after `<asr_text>` only, so the
    /// language header is never cut into and committed (the maker's `has_tag` guard: an empty transcript rolls back nothing).
    static func fixedTranscript(_ text: String, rollback: Int, encode: (String) -> [Int], decode: ([Int]) -> String) -> String {
        var transcript = text
        if let tag = transcript.range(of: "<asr_text>", options: .backwards) { transcript = String(transcript[tag.upperBound...]) }
        transcript = String(transcript.prefix { $0 != "|" })
        let ids = encode(transcript)
        var drop = rollback
        while true {
            let end = max(0, ids.count - drop)
            if end == 0 { return "" }
            let fixed = decode(Array(ids.prefix(end)))
            if !fixed.contains("\u{fffd}") { return fixed.trimmingCharacters(in: .whitespacesAndNewlines) }
            drop += 1
        }
    }
    /// Takes one decoded chunk (the prefix plus the generated text) into the transcript; auto-language output commits
    /// nothing until it has `<asr_text>`.
    func accept(
        raw: String, prefixText: String, samples: Int, final: Bool, parse: (String) -> (language: String?, text: String),
        encode: (String) -> [Int], decode: ([Int]) -> String
    ) {
        var raw = raw
        var parsed = parse(raw)
        if forcedLanguage == "Chinese" || parsed.language == "Chinese" {
            raw = raw.replacingOccurrences(of: "(?<=[\\u4e00-\\u9fff])\\s+(?=[\\u4e00-\\u9fff])", with: "", options: .regularExpression)
            parsed = parse(raw)
        }
        guard forcedLanguage != nil || raw.contains("<asr_text>") else {
            // Auto language: nothing is committed before the model has written its language header and delimiter.
            language = nil; hypothesis = ""; entries.append((samples, "")); return
        }
        language = parsed.language == "None" ? nil : parsed.language
        var text = parsed.text
        if let tag = text.range(of: "<asr_text>", options: .backwards) { text = String(text[tag.upperBound...]) }
        hypothesis = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let fixed = Self.fixedTranscript(hypothesis, rollback: final ? 0 : rollback, encode: encode, decode: decode)
        let old = prefixText.trimmingCharacters(in: .whitespacesAndNewlines)
        let delta = fixed.hasPrefix(old) ? String(fixed.dropFirst(old.count)) : ""
        entries.append((samples, delta)); committed += delta
    }
    private func decode(_ chunk: [Float], final: Bool, model: Qwen3ASRModel, tokenLimit: Int, cancellation: NativeCancellation) throws {
        guard let tokenizer = model.tokenizer else { throw audioFailure("The local tokenizer is unavailable.") }
        try cancellation.check()
        decodedChunks += 1
        window += chunk
        if window.count > 16000 * 16 {
            // Trim complete audio/text chunks together. Upstream's fixed 160-ms count is wrong for other chunk sizes.
            var removed = 0
            while !entries.isEmpty && removed < 16000 * 8 {
                let entry = entries.removeFirst()
                removed += entry.samples; evictedText += entry.text.count
            }
            if removed > 0 { window.removeFirst(min(removed, window.count)) }
        }
        let prefixText = entries.map(\.text).joined()
        let prefix = forcedLanguage == nil && language != nil ? "language \(language!)<asr_text>" + prefixText : prefixText
        let generated = try qwenDecode(
            model, audio: MLXArray(window), context: prompt, language: forcedLanguage, prefix: prefix, tokenLimit: tokenLimit, cancellation: cancellation)
        var raw = (prefix + confuciusPunctuation(generated)).replacingOccurrences(of: "\u{fffd}", with: "")
        raw = String(raw.prefix { $0 != "|" })
        accept(
            raw: raw, prefixText: prefixText, samples: chunk.count, final: final,
            parse: { model.parseGeneratedChunk($0, forcedLanguage: self.forcedLanguage) },
            encode: { tokenizer.encode(text: $0) }, decode: { tokenizer.decode(tokens: $0) })
    }
}

func confuciusPunctuation(_ input: String) -> String {
    let toChinese: [Character: Character] = [",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："]
    let toEnglish = Dictionary(uniqueKeysWithValues: toChinese.map { ($0.value, $0.key) })
    var result = "", previous: Unicode.Scalar?
    for character in input {
        var value = character
        if let previous {
            if (0x4e00...0x9fff).contains(previous.value) {
                value = toChinese[character] ?? character
            } else if previous.isASCII && (CharacterSet.alphanumerics.contains(previous) || previous == "\"" || previous == "'") {
                value = toEnglish[character] ?? character
            }
        }
        result.append(value)
        if !character.isWhitespace { previous = character.unicodeScalars.last }
    }
    return result
}

func qwenDecode(_ model: Qwen3ASRModel, audio: MLXArray, context: String, language: String?, prefix: String = "", tokenLimit: Int, cancellation: NativeCancellation) throws
    -> String
{
    guard let tokenizer = model.tokenizer else { throw audioFailure("ASR tokenizer is not loaded.") }
    let (features, mask, count) = model.preprocessAudio(audio)
    let prompt = model.buildPromptText(numAudioTokens: count, context: context, language: language) + prefix
    let ids = MLXArray(tokenizer.encode(text: prompt).map(Int32.init)).reshaped(1, -1)
    let cache = model.makeCache()
    var logits = model(inputIds: ids, inputFeatures: features, featureAttentionMask: mask, cache: cache)
    var tokens: [Int] = []
    for _ in 0..<tokenLimit {
        try cancellation.check()
        let token = logits[0, -1].argMax().item(Int.self)
        if [151645, 151643].contains(token) { break }
        tokens.append(token)
        if tokens.count >= 24 && Set(tokens.suffix(24)).count <= 3 { break }
        logits = model(inputIds: MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
    }
    return tokenizer.decode(tokens: tokens)
}
