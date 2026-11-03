// SPDX-License-Identifier: MIT
// The library tokenizer (swift-transformers, the stock path and the fallback after an optimized-path failure) must
// produce the same ids as Verdict's fast tokenizers (Exact and Fast) on every text Verdict's fixture gate uses.
// Requalifies the swift-transformers pin whenever it moves (log: models/laya-english/log/004-native-link.md).
// Needs the dev weights (ARSENAL_DECISIONS_WEIGHTS) and Verdict's fixtures; skipped when either is absent. CPU only.
import Foundation
import Testing
import Hub
import Tokenizers
import VerdictEngine

struct TokenizerAgreementTests {
    static let weights = URL(
        fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_DECISIONS_WEIGHTS"]
            ?? NSHomeDirectory() + "/Library/Application Support/Arsenal-dev/weights/decisions")
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Verdict/lab/fixtures")

    /// (model, tokenizer directory) as each loader reads it.
    static let cases: [(String, String)] = [
        ("laya-english", "laya-english/fp16/tokenizer"), ("laya-multilingual", "laya-multilingual/fp16/tokenizer"),
        ("laya-typed-decisions", "laya-typed-decisions/fp16/tokenizer"), ("von-1.2", "von-1.2/fp32"), ("von-1.1", "von-1.1/fp32")
    ]

    /// swift-transformers 1.1.9 encodes the empty string as [235248] (▁) with the Metaspace pre-tokenizer; the fast
    /// tokenizer and HF tokenizers 0.22.1 give []. Affects only an empty text on the stock path (diagnosis, or the
    /// fallback after an optimized-path failure) of Laya Multilingual; recorded in models/laya-english/log/004-native-link.md.
    static let knownFastDivergence: [String: Set<String>] = ["laya-multilingual": [""]]

    /// Every rendered input, instruction and criterion of the three Laya fixture gates (English, multilingual, typed).
    static func texts() throws -> [String] {
        var out: [String] = ["", " ", "a", "Héllo wörld — 日本語のテキスト 🙂", "  leading and trailing  "]
        for model in ["laya-english", "laya-multilingual", "laya-typed-decisions"] {
            let url = fixtures.appendingPathComponent("\(model)-0bit.json.gz")
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/gunzip")
            process.arguments = ["-c", url.path]
            let pipe = Pipe()
            process.standardOutput = pipe
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            for item in object["items"] as? [[String: Any]] ?? [] { if let text = item["rendered_input"] as? String { out.append(text) } }
            for case let question as [String: Any] in (object["questions"] as? [String: Any] ?? [:]).values {
                if let text = question["instructions"] as? String { out.append(text) }
                if let criteria = question["criteria"] as? [String: String] { out += Array(criteria.keys) + Array(criteria.values) }
                if let criteria = question["criteria"] as? [String] { out += criteria }
            }
        }
        return out
    }

    @Test(arguments: cases.map(\.0))
    func libraryMatchesFastTokenizer(_ model: String) throws {
        let dir = Self.weights.appendingPathComponent(Self.cases.first { $0.0 == model }!.1)
        let tokenData = dir.appendingPathComponent("tokenizer.json"), configData = dir.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: tokenData.path), FileManager.default.fileExists(atPath: Self.fixtures.path) else { return }
        let token = try Data(contentsOf: tokenData), config = try Data(contentsOf: configData)
        let fast: (String, Bool) -> [Int]
        if model.hasPrefix("von") {
            let bpe = try #require(FastByteBPE(data: token, ignoringTruncationAndPadding: true))
            fast = { bpe.encode($0, addSpecialTokens: $1) }
        } else if let bpe = FastByteBPE(data: token) {
            fast = { bpe.encode($0, addSpecialTokens: $1) }
        } else {
            let bpe = try #require(FastMetaspaceBPE(data: token))
            fast = { bpe.encode($0, addSpecialTokens: $1) }
        }
        let decoder = JSONDecoder()
        let library = try AutoTokenizer.from(tokenizerConfig: decoder.decode(Config.self, from: config), tokenizerData: decoder.decode(Config.self, from: token))
        let texts = try Self.texts()
        #expect(texts.count > 600)
        var mismatches: [String] = []
        for text in texts where !(Self.knownFastDivergence[model]?.contains(text) ?? false) {
            for special in [false, true] where fast(text, special) != library.encode(text: text, addSpecialTokens: special) {
                mismatches.append(
                    "\(special ? "+" : "-")special \(text.prefix(60).unicodeScalars.map { String($0.value, radix: 16) }) fast \(fast(text, special).prefix(8)) library \(library.encode(text: text, addSpecialTokens: special).prefix(8))"
                )
            }
        }
        #expect(mismatches.isEmpty, "\(model): \(mismatches.count) of \(texts.count * 2) differ, e.g. \(mismatches.prefix(3))")
    }
}
