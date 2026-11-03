// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXLMCommon
import MLXNN
import ArsenalWire
import Tokenizers

extension Unicode.Scalar {
    /// `"0" <= ch <= "9"`, as the maker's parser tests digits.
    fileprivate var isASCIIDigit: Bool { ("0"..."9").contains(self) }
}

/// One `[start][Sxx]text[end]` segment of the model's transcript.
struct MossSegment: Equatable {
    let start: Double
    let end: Double
    let speaker: String
    let text: String
}

/// The maker's streaming transcript parser (transcript_parser.py `TranscriptStreamParser`, strip and skip-empty on):
/// a state machine over code points (Python iterates code points, so this iterates unicode scalars, not grapheme
/// clusters) that emits a segment once its end timestamp is confirmed by the next start or by close. The maker's quirks
/// are kept on purpose, so its transcripts parse identically: a timestamp run past 32 characters spills into the text
/// with its 33rd character twice. Times are capped at 86,400 s (the output schemas' bound).
struct MossTranscriptParser {
    static let maxSeconds = 86_400.0
    private enum State { case seekStart, readStart, expectSpeakerOpen, readSpeaker, readText, readEnd, afterEnd }
    private var state = State.seekStart
    private var token: [Unicode.Scalar] = [], text: [Unicode.Scalar] = [], pendingAfterEnd: [Unicode.Scalar] = []
    private var start: Double?, end: Double?, endToken: [Unicode.Scalar] = [], speaker: String?

    static func parse(_ transcript: String) -> [MossSegment] {
        var parser = MossTranscriptParser()
        return parser.feed(transcript) + parser.close()
    }

    private mutating func reset() {
        state = .seekStart; token = []; text = []; pendingAfterEnd = []; start = nil; end = nil; endToken = []; speaker = nil
    }
    private static func timestamp(_ chars: [Unicode.Scalar]) -> Double? {
        guard !chars.isEmpty, chars.filter({ $0 == "." }).count <= 1, chars.contains(where: \.isASCIIDigit),
            chars.allSatisfy({ $0.isASCIIDigit || $0 == "." })
        else { return nil }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: chars)
        return Double(String(view))
    }
    private static func isTimestampChar(_ c: Unicode.Scalar) -> Bool { c.isASCIIDigit || c == "." }

    mutating func feed(_ chunk: String) -> [MossSegment] {
        var out: [MossSegment] = []
        for c in chunk.unicodeScalars {
            switch state {
            case .seekStart:
                if c == "[" { token = []; state = .readStart }
            case .readStart:
                if c == "]" {
                    guard let value = Self.timestamp(token) else { reset(); continue }
                    start = value; state = .expectSpeakerOpen; token = []
                    continue
                }
                if Self.isTimestampChar(c) {
                    token.append(c)
                    if token.count <= 32 { continue }
                }
                reset()
                if c == "[" { state = .readStart }
            case .expectSpeakerOpen:
                if c == "[" { token = []; state = .readSpeaker } else if !c.properties.isWhitespace { reset() }
            case .readSpeaker:
                if c == "]" {
                    guard token.count >= 2, token[0] == "S", token.dropFirst().allSatisfy(\.isASCIIDigit) else { reset(); continue }
                    var view = String.UnicodeScalarView()
                    view.append(contentsOf: token)
                    speaker = String(view); text = []; state = .readText; token = []
                    continue
                }
                if c == "S" || c.isASCIIDigit {
                    token.append(c)
                    if token.count <= 16 { continue }
                }
                reset()
                if c == "[" { state = .readStart }
            case .readText:
                if c == "[" { token = []; state = .readEnd } else { text.append(c) }
            case .readEnd:
                if c == "]" {
                    if let value = Self.timestamp(token), let start, value >= start {
                        end = value; endToken = token; pendingAfterEnd = []; state = .afterEnd
                    } else {
                        text += ["["] + token + ["]"]; state = .readText
                    }
                    token = []
                    continue
                }
                if Self.isTimestampChar(c) {
                    token.append(c)
                    if token.count <= 32 { continue }
                }
                // As the maker: the token already holds this character when it overflowed, and it is appended again.
                text += ["["] + token + [c]; token = []; state = .readText
            case .afterEnd:
                if c == "[" {
                    if let segment = emitSegment() { out.append(segment) }
                    token = []; state = .readStart
                } else if c.properties.isWhitespace {
                    pendingAfterEnd.append(c)
                } else {
                    text += ["["] + endToken + ["]"] + pendingAfterEnd + [c]
                    pendingAfterEnd = []; end = nil; endToken = []; state = .readText
                }
            }
        }
        return out
    }

    mutating func close() -> [MossSegment] {
        defer { reset() }
        if state == .afterEnd, let segment = emitSegment() { return [segment] }
        return []
    }

    private mutating func emitSegment() -> MossSegment? {
        defer { token = []; text = []; pendingAfterEnd = []; start = nil; end = nil; endToken = []; speaker = nil; state = .seekStart }
        guard let start, let end, let speaker else { return nil }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: text)
        let body = String(view).trimmingCharacters(in: .whitespacesAndNewlines)
        return body.isEmpty
            ? nil : MossSegment(start: min(start, Self.maxSeconds), end: min(end, Self.maxSeconds), speaker: speaker, text: body)
    }
}

/// Incremental streaming text: tokens are decoded only since the last hand-over, and handed over once their text holds
/// no incomplete UTF-8 sequence (the decoder writes U+FFFD for a split character). The parser keeps its own state, so any
/// cut point is safe.
struct MossStreamDecoder {
    private var pending: [Int] = []
    mutating func push(_ token: Int, every: Int = 8, decode: ([Int]) -> String) -> String? {
        pending.append(token)
        guard pending.count >= every else { return nil }
        let text = decode(pending)
        guard !text.unicodeScalars.contains("\u{FFFD}") || pending.count >= 64 else { return nil }
        pending = []
        return text
    }
    /// Everything not yet handed over.
    mutating func finish(decode: ([Int]) -> String) -> String {
        defer { pending = [] }
        return pending.isEmpty ? "" : decode(pending)
    }
}

final class MossTranscribeAdapter: AudioAdapter {
    static let maxSeconds = 90 * 60
    static let prefillChunk = 1024
    override var architecture: String { "moss_transcribe_diarize" }
    override var operations: Set<String> { ["transcribe", "diarize"] }
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var digits: [Character: Int] = [:]

    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let config = try MossConfig.load(context.weights)
        let tokenizer = try qwenTokenizer(context.weights)
        var digits: [Character: Int] = [:]
        for digit in "0123456789" {
            let ids = tokenizer.encode(text: String(digit), addSpecialTokens: false)
            guard ids.count == 1 else { throw audioFailure("The tokenizer splits a digit.") }
            digits[digit] = ids[0]
        }
        self.tokenizer = tokenizer
        self.digits = digits
        try prepare(
            MossTranscriber(config), context, cancellation: cancellation,
            map: { key, value in
                if key == "lm_head.weight" { return nil } // tied to the token embedding
                if key.hasPrefix("model.whisper_encoder.conv"), key.hasSuffix(".weight"), value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
                return (key, value)
            })
    }

    override func unload() {
        tokenizer = nil
        super.unload()
    }

    /// Log-mel `[frames, mels]` of one 30 s chunk, padded with zeros as the maker pads it.
    static func features(_ chunk: [Float], config: MossConfig) -> MLXArray {
        let padded = chunk.count < config.chunkSamples ? chunk + [Float](repeating: 0, count: config.chunkSamples - chunk.count) : chunk
        return smartTurnLogMelSpectrogram(padded, sampleRate: 16000, nFft: config.nFFT, hopLength: config.hop, nMels: config.melBins)
    }

    /// Whisper → 4× merge → VQ adaptor: `[tokens, hidden]` for the whole recording, chunk by chunk.
    func audioEmbeddings(_ samples: [Float], model: MossModel, cancellation: NativeCancellation) throws -> (MLXArray, Int) {
        let config = model.config
        var parts: [MLXArray] = [], tokens = 0
        for start in stride(from: 0, to: samples.count, by: config.chunkSamples) {
            try cancellation.check()
            let chunk = Array(samples[start..<min(samples.count, start + config.chunkSamples)])
            let count = MossProcessor.tokenCount(samples: chunk.count, config: config)
            let encoded = try model.whisper(Self.features(chunk, config: config).expandedDimensions(axis: 0), check: { try cancellation.check() })
            let frames = encoded[0, ..<(count * config.mergeSize)]
            eval(frames)
            parts.append(frames)
            tokens += count
            Memory.clearCache()
        }
        let merged = concatenated(parts, axis: 0).reshaped(tokens, config.width * config.mergeSize)
        let adapted = model.adaptor(merged)
        eval(adapted)
        return (adapted, tokens)
    }

    /// Greedy generation as the maker's `generate_transcription` (do_sample False), returning the decoded transcript.
    func generate(
        _ samples: [Float], instruction: String, maxNewTokens: Int, cancellation: NativeCancellation, stream: Bool = false,
        onText: (String) throws -> Void = { _ in }
    ) throws -> (transcript: String, tokens: [Int], promptTokens: Int, truncated: Bool) {
        guard let root = graph as? MossTranscriber, let tokenizer else { throw audioFailure("MOSS-Transcribe-Diarize is not loaded.") }
        let model = root.model, config = model.config
        let (audio, audioTokens) = try audioEmbeddings(samples, model: model, cancellation: cancellation)
        let ids = MossProcessor.prompt(instruction: instruction, tokens: audioTokens, tokenizer: tokenizer, config: config, digits: digits)
        let positions = ids.indices.filter { ids[$0] == config.audioTokenID }
        guard positions.count == audioTokens else { throw audioFailure("Audio tokens and placeholders differ.") }
        var embeds = model.language.embedTokens(MLXArray(ids.map(Int32.init)))
        embeds[MLXArray(positions.map(Int32.init))] = audio.asType(embeds.dtype)
        let cache: [KVCache] = (0..<config.text.numHiddenLayers).map { _ in KVCacheSimple() }
        var hidden = embeds
        for start in stride(from: 0, to: ids.count, by: Self.prefillChunk) {
            try cancellation.check()
            let piece = embeds[start..<min(ids.count, start + Self.prefillChunk)].expandedDimensions(axis: 0)
            hidden = model.language(inputsEmbeds: piece, cache: cache)
            eval(hidden)
        }
        func next(_ h: MLXArray) -> Int { model.language.embedTokens.asLinear(h[0, -1]).argMax().item(Int.self) }
        var token = next(hidden), generated: [Int] = []
        var streamed = MossStreamDecoder()
        func decode(_ tokens: [Int]) -> String { tokenizer.decode(tokens: tokens, skipSpecialTokens: true) }
        while generated.count < maxNewTokens {
            try cancellation.check()
            // The maker's generation_config stops on <|im_end|> only.
            if token == MossProcessor.imEnd { break }
            generated.append(token)
            if stream, let text = streamed.push(token, decode: decode) { try onText(text) }
            hidden = model.language(inputsEmbeds: model.language.embedTokens(MLXArray([Int32(token)])).expandedDimensions(axis: 0), cache: cache)
            token = next(hidden)
        }
        if stream {
            let rest = streamed.finish(decode: decode)
            if !rest.isEmpty { try onText(rest) }
        }
        let transcript = tokenizer.decode(tokens: generated, skipSpecialTokens: true).trimmingCharacters(in: .whitespacesAndNewlines)
        Memory.clearCache()
        // The budget ran out before the end token: the transcript stops early, and the reply says so.
        return (transcript, generated, ids.count, token != MossProcessor.imEnd)
    }

    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operations.contains(operation) else { throw audioFailure("Unknown operation.") }
        try audioKeys(input, allowed: operation == "transcribe" ? ["audio", "prompt", "hotwords", "max_new_tokens", "stream"] : ["audio", "prompt", "hotwords", "max_new_tokens"])
        let prompt = try input["prompt"].map { value -> String in
            guard let text = value.string, text.count <= 4096 else { throw audioFailure("prompt must be a string of at most 4096 characters.") }
            return text
        }
        let hotwords =
            try input["hotwords"].map { value -> [String] in
                let words = value.array?.compactMap(\.string) ?? []
                guard let list = value.array, list.count <= 100, words.count == list.count, words.allSatisfy({ !$0.isEmpty && $0.count <= 100 })
                else { throw audioFailure("hotwords must be a list of at most 100 nonempty strings of at most 100 characters.") }
                return words
            } ?? []
        let maxNew = try audioInteger(input, key: "max_new_tokens", fallback: 5120, range: 1...65536)
        let pcm = try audioInput(input, rate: 16000, seconds: Self.maxSeconds)
        guard !pcm.samples.isEmpty else { throw audioFailure("Audio is empty.") }
        let stream = operation == "transcribe" && input["stream"]?.bool == true
        var streamParser = MossTranscriptParser()
        let result = try generate(
            pcm.samples, instruction: MossProcessor.instruction(prompt: prompt, hotwords: hotwords), maxNewTokens: maxNew, cancellation: cancellation,
            stream: stream,
            onText: { text in
                for segment in streamParser.feed(text) {
                    try emit(.json(["event": "token", "text": .string(segment.text), "speaker": .string(segment.speaker)]))
                }
            })
        if stream {
            // The last segment is confirmed only when the stream closes.
            for segment in streamParser.close() {
                try emit(.json(["event": "token", "text": .string(segment.text), "speaker": .string(segment.speaker)]))
            }
        }
        try cancellation.check()
        return try Self.reply(
            operation, transcript: result.transcript, truncated: result.truncated, customPrompt: MossProcessor.ownPrompt(prompt) != nil,
            duration: Double(pcm.samples.count) / 16000)
    }

    /// The reply for a finished transcript. `truncated` is true when max_new_tokens ran out before the model's end token,
    /// so the transcript stops early.
    static func reply(_ operation: String, transcript: String, truncated: Bool, customPrompt: Bool, duration: Double) throws -> JSONObject {
        let segments = MossTranscriptParser.parse(transcript)
        // Under the default instruction a transcript without timestamps (music, noise) is a valid, empty diarization;
        // only a custom instruction that asked for something else is refused.
        if operation == "diarize", customPrompt, segments.isEmpty, !transcript.isEmpty {
            throw audioFailure(
                "The model wrote no timestamped segments for this instruction; diarize needs the default instruction or one that asks for timestamps and speaker labels.")
        }
        var speakers: [String] = []
        for segment in segments where !speakers.contains(segment.speaker) { speakers.append(segment.speaker) }
        let segmentValues: [ArsenalWire.JSONValue] = segments.map {
            .object(["start": .number($0.start), "end": .number($0.end), "speaker": .string($0.speaker), "text": .string($0.text)])
        }
        if operation == "diarize" {
            let rttm = segments.map { String(format: "SPEAKER audio 1 %.3f %.3f <NA> <NA> %@ <NA> <NA>", $0.start, $0.end - $0.start, $0.speaker) }
            return [
                "segments": .array(segmentValues), "speakers": .array(speakers.map(ArsenalWire.JSONValue.string)),
                "rttm": .string(rttm.isEmpty ? "" : rttm.joined(separator: "\n") + "\n"), "transcript": .string(transcript),
                "truncated": .bool(truncated)
            ]
        }
        // Plain text: segments in order; a transcript without segment markup (a custom prompt) is returned as written.
        let text = segments.isEmpty ? transcript : segments.map(\.text).joined(separator: "\n")
        return [
            "text": .string(text), "duration": .number(duration), "segments": .array(segmentValues),
            "speakers": .array(speakers.map(ArsenalWire.JSONValue.string)), "transcript": .string(transcript), "truncated": .bool(truncated)
        ]
    }
}
