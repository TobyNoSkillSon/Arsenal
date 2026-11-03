import Testing
@testable import AudioWorker
@Suite struct ConfigurationTests {
    @Test func smartTurnDefaults() {
        let config = SmartTurnConfig()
        #expect(config.encoderConfig.dModel == 384)
        #expect(config.processorConfig.maxAudioSeconds == 8)
    }
    @Test func chunkBoundaryFlushKeepsRollbackWords() {
        let state = ConfuciusSession(chunkMS: 160, rollback: 5, prompt: "", language: "English")
        state.entries = [(2560, "hello")]
        state.committed = "Earlier. hello"
        state.hypothesis = "hello world"
        state.commitHypothesis()
        #expect(state.committed == "Earlier. hello world")
        state.commitHypothesis()
        #expect(state.committed == "Earlier. hello world")
    }
    /// A character-level stand-in tokenizer: one token per character, so rollback counts are visible in the text.
    private static func encode(_ text: String) -> [Int] { text.unicodeScalars.map { Int($0.value) } }
    private static func decode(_ ids: [Int]) -> String { String(String.UnicodeScalarView(ids.compactMap { Unicode.Scalar(UInt32($0)) })) }
    private static func parse(_ raw: String) -> (language: String?, text: String) {
        guard raw.hasPrefix("language "), let tag = raw.range(of: "<asr_text>") else { return (nil, raw.trimmingCharacters(in: .whitespaces)) }
        return (String(raw[raw.index(raw.startIndex, offsetBy: 9)..<tag.lowerBound]), String(raw[tag.upperBound...]).trimmingCharacters(in: .whitespaces))
    }
    @Test func rollbackNeverReachesTheLanguageHeader() {
        #expect(ConfuciusSession.fixedTranscript("hello world", rollback: 1, encode: Self.encode, decode: Self.decode) == "hello worl")
        // Rolling back more tokens than the transcript has commits nothing, never part of "language English".
        #expect(ConfuciusSession.fixedTranscript("language English<asr_text>hi", rollback: 5, encode: Self.encode, decode: Self.decode) == "")
        #expect(ConfuciusSession.fixedTranscript("language English<asr_text>", rollback: 0, encode: Self.encode, decode: Self.decode) == "")
        let state = ConfuciusSession(chunkMS: 160, rollback: 5, prompt: "", language: nil)
        state.accept(raw: "language English<asr_text>hi", prefixText: "", samples: 5120, final: false, parse: Self.parse, encode: Self.encode, decode: Self.decode)
        #expect(state.committed == "")
        #expect(state.language == "English")
        #expect(state.hypothesis == "hi")
        state.accept(
            raw: "language English<asr_text>hi there friend", prefixText: "", samples: 2560, final: false, parse: Self.parse, encode: Self.encode,
            decode: Self.decode)
        #expect(state.committed == "hi there f")
    }
    @Test func autoLanguageCommitsNothingBeforeTheDelimiter() {
        let state = ConfuciusSession(chunkMS: 160, rollback: 1, prompt: "", language: nil)
        state.accept(raw: "language Engl", prefixText: "", samples: 5120, final: true, parse: Self.parse, encode: Self.encode, decode: Self.decode)
        #expect(state.committed == "")
        #expect(state.hypothesis == "")
        #expect(state.language == nil)
        #expect(state.entries.count == 1)
        // A forced language has no header to wait for.
        let forced = ConfuciusSession(chunkMS: 160, rollback: 1, prompt: "", language: "English")
        forced.accept(raw: "hello", prefixText: "", samples: 5120, final: false, parse: Self.parse, encode: Self.encode, decode: Self.decode)
        #expect(forced.committed == "hell")
    }
    @Test func streamingBudgetFollowsTheMakerExample() {
        // example.py at its defaults: 160 ms chunks plus a 160 ms first-chunk lookahead, 4 tokens first, then 2..4.
        let state = ConfuciusSession(chunkMS: 160, rollback: 1, prompt: "", language: nil)
        #expect(state.nextChunkSamples == 5120)
        #expect([state.firstMaxNewTokens, state.baseMaxNewTokens, state.maxNewTokensCap, state.maxNewTokens] == [4, 2, 4, 4])
        state.decodedChunks = 1
        #expect(state.nextChunkSamples == 2560)
        state.hypothesis = "hi"; state.adaptTokenBudget()
        #expect(state.maxNewTokens == 2)
        state.adaptTokenBudget(); state.adaptTokenBudget(); state.adaptTokenBudget()
        #expect(state.maxNewTokens == 4)
        state.hypothesis = "hi there"; state.adaptTokenBudget()
        #expect(state.maxNewTokens == 2)
        let long = ConfuciusSession(chunkMS: 2000, rollback: 1, prompt: "", language: nil)
        #expect([long.firstMaxNewTokens, long.baseMaxNewTokens, long.maxNewTokensCap] == [27, 25, 32])
    }
    @Test func streamingIsDeclaredByTheFolder() {
        let manifest = "name = \"transcribe_session\"\n[[op]]\nname = \"transcribe\"\n[[op]]\nname = \"transcribe_session\"\n"
        #expect(manifestDeclaresOperation(manifest, "transcribe_session"))
        #expect(!manifestDeclaresOperation("name = \"transcribe_session\"\n[[op]]\nname = \"transcribe\"\n", "transcribe_session"))
    }
    @Test func transcriptionLanguagesAreBCP47Codes() {
        #expect(audioLanguageCode("English") == "en" && audioLanguageCode("cantonese") == "yue" && audioLanguageCode("en") == "en")
        #expect(audioLanguageCode("None") == nil && audioLanguageCode(" ") == nil)
        #expect(dominantAudioLanguage(["English", "Chinese", "Chinese", nil]) == "zh")
        #expect(dominantAudioLanguage(["English", "Chinese"]) == "en", "a tie keeps the earliest")
    }
    @Test func cohereSplitsLongAudioAtTheQuietestWindowLikeTheFeatureExtractor() {
        let rate = 16000
        #expect(CohereAdapter.chunks(Array(repeating: 0.5, count: 32 * rate), rate: rate, maximumSeconds: 35) == [0..<(32 * rate)])
        // 80 s of steady sound with one silent 0.1 s window at 33.0 s: the first cut lands there; the second, with no quieter
        // window in its last five seconds, at the first window searched (30 s after the first cut).
        var samples = [Float](repeating: 0.5, count: 80 * rate)
        for index in (33 * rate)..<(33 * rate + 1600) { samples[index] = 0 }
        #expect(CohereAdapter.chunks(samples, rate: rate, maximumSeconds: 35) == [0..<528_000, 528_000..<1_008_000, 1_008_000..<1_280_000])
    }
    @Test func punctuationFollowsThePrecedingLanguage() {
        #expect(confuciusPunctuation("你好, how are you？") == "你好， how are you?")
    }

    @Test func invalidIntegersFailBeforeConversion() {
        #expect(throws: (any Error).self) { try audioInteger(["steps": .number(1e100)], key: "steps", fallback: 16, range: 2...100) }
        #expect(throws: (any Error).self) { try audioInteger(["steps": .number(2.5)], key: "steps", fallback: 16, range: 2...100) }
        #expect(throws: (any Error).self) { try audioInteger(["steps": .bool(true)], key: "steps", fallback: 16, range: 2...100) }
    }

}
