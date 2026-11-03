import Foundation
import MLXAudioCore

/// Multilingual TextProcessor for Kokoro TTS (Vireo subset).
///
/// - English (US and UK voices): MisakiTextProcessor (gold/silver lexicons + BART fallback network)
/// - Spanish, French, Italian, Portuguese: IPA lexicon lookup (gruut/espeak dictionaries)
/// - Other Kokoro languages (Japanese, Mandarin, Hindi) need the neural byT5 phonemizer, which Vireo does not ship:
///   they fail with `LexiconError.unsupportedLanguage` instead of speaking garbage.
///
/// Resources come only from the pinned local snapshot directories the helper downloaded (no Hugging Face lookups).
/// Lexicons load lazily per language and stay cached; `prepare(for:)` loads one ahead of time.
public final class KokoroMultilingualProcessor: TextProcessor, @unchecked Sendable {
    private let englishProcessor: MisakiTextProcessor
    private let lexiconDirectory: URL?
    private var lexiconCache: [String: [String: String]] = [:]
    private let lock = NSLock()

    /// Kokoro voice prefix → language code mapping (every Kokoro voice family, supported or not).
    public static let voiceLanguageMap: [Character: String] = [
        "a": "en-us",
        "b": "en-gb",
        "e": "es",
        "f": "fr",
        "h": "hi",
        "i": "it",
        "j": "ja",
        "p": "pt",
        "z": "cmn",
    ]

    /// Language codes that use MisakiTextProcessor (English G2P).
    private static let englishCodes: Set<String> = ["en", "en-us", "en-gb"]

    /// Language code → TSV filename mapping (the Kokoro languages Vireo phonemizes with a lexicon).
    private static let langFileMap: [String: String] = [
        "es": "es_lexicon.tsv",
        "fr": "fr_lexicon.tsv",
        "it": "it_lexicon.tsv",
        "pt": "pt_lexicon.tsv",
        "pt-br": "pt_lexicon.tsv",
    ]

    /// Whether this processor can phonemize text for a Kokoro voice (by its language prefix).
    public static func supportsVoice(_ voice: String) -> Bool {
        guard let lang = languageForVoice(voice) else { return false }
        return englishCodes.contains(lang) || langFileMap[lang] != nil
    }

    public init(englishDirectory: URL?, britishDirectory: URL? = nil, lexiconDirectory: URL?, precision: BARTPrecision = BARTPrecision()) {
        self.englishProcessor = MisakiTextProcessor(resourceDirectory: englishDirectory, britishDirectory: britishDirectory, precision: precision)
        self.lexiconDirectory = lexiconDirectory
    }

    func intakeModel() throws -> BARTModel { try englishProcessor.intakeModel() }

    /// Infer language code from Kokoro voice name prefix.
    public static func languageForVoice(_ voice: String) -> String? {
        guard let first = voice.first else { return nil }
        return voiceLanguageMap[first]
    }

    public func process(text: String, language: String?) throws -> String {
        let lang = language?.lowercased() ?? "en-us"

        if Self.englishCodes.contains(lang) || lang.hasPrefix("en") {
            return try englishProcessor.process(text: text, language: language)
        }

        let lexicon = try loadLexicon(for: lang)
        return phonemize(text: text, lexicon: lexicon)
    }

    /// English G2P init (lexicon JSON + BART fallback network) for US or UK English, ahead of the first request.
    public func warmEnglish(british: Bool) throws {
        _ = try englishProcessor.process(text: "a", language: british ? "en-gb" : "en-us")
    }

    // MARK: - Lexicon Loading

    private func loadLexicon(for lang: String) throws -> [String: String] {
        if let cached = lock.withLock({ lexiconCache[lang] }) { return cached }

        guard let filename = Self.langFileMap[lang] else {
            throw LexiconError.unsupportedLanguage(lang)
        }
        guard let dir = lexiconDirectory else { throw LexiconError.lexiconNotDownloaded("kokoro-ipa-lexicons") }
        let tsvURL = dir.appendingPathComponent(filename)

        guard FileManager.default.fileExists(atPath: tsvURL.path) else {
            throw LexiconError.lexiconNotFound(filename)
        }

        let content = try String(contentsOf: tsvURL, encoding: .utf8)
        var dict = [String: String]()
        dict.reserveCapacity(100_000)

        for line in content.split(separator: "\n") {
            guard let tabIdx = line.firstIndex(of: "\t") else { continue }
            let word = String(line[line.startIndex..<tabIdx]).lowercased()
            let phonemes = String(line[line.index(after: tabIdx)...])
            let ipa = phonemes.split(separator: " ").joined()
            dict[word] = ipa
        }

        lock.withLock { lexiconCache[lang] = dict }
        return dict
    }

    public func prepare(for language: String) async throws {
        let lang = language.lowercased()

        if Self.englishCodes.contains(lang) || lang.hasPrefix("en") {
            try await englishProcessor.prepare()
            return
        }
        _ = try loadLexicon(for: lang)
    }

    // MARK: - Phonemization

    func phonemize(text: String, lexicon: [String: String]) -> String {
        let normalized = text.lowercased()
        var result = [String]()
        var currentWord = ""

        for ch in normalized {
            if ch.isLetter || ch == "'" || ch == "-" || ch == "\u{0301}" {
                currentWord.append(ch)
            } else {
                if !currentWord.isEmpty {
                    let ipa = lookupWord(currentWord, lexicon: lexicon)
                    result.append(ipa)
                    currentWord = ""
                }
                if ch == "," || ch == "." || ch == "!" || ch == "?" || ch == ":" || ch == ";" {
                    result.append(String(ch))
                }
            }
        }

        if !currentWord.isEmpty {
            let ipa = lookupWord(currentWord, lexicon: lexicon)
            result.append(ipa)
        }

        return result.joined(separator: " ")
    }

    func lookupWord(_ word: String, lexicon: [String: String]) -> String {
        if let ipa = lexicon[word] {
            return ipa
        }

        let stripped = word.decomposedStringWithCanonicalMapping
            .unicodeScalars
            .filter { !("\u{0300}"..."\u{036F}").contains($0) }
            .map { String($0) }
            .joined()

        if stripped != word, let ipa = lexicon[stripped] {
            return ipa
        }

        return word
    }
}

// MARK: - Errors

public enum LexiconError: Error, LocalizedError {
    case unsupportedLanguage(String)
    case lexiconNotFound(String)
    case lexiconNotDownloaded(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedLanguage(let lang):
            return "Kokoro in Vireo speaks English, Spanish, French, Italian and Portuguese; not \(lang)"
        case .lexiconNotFound(let file):
            return "Kokoro lexicon file not found: \(file)"
        case .lexiconNotDownloaded(let repo):
            return "Kokoro lexicons not downloaded (\(repo))"
        }
    }
}
