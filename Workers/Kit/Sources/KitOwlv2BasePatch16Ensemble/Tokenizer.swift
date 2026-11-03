// SPDX-License-Identifier: MIT
// CLIP's byte-level BPE tokenizer (transformers tokenization_clip.py, CLIPTokenizer without ftfy), from vocab.json and
// merges.txt: NFC, whitespace folded, lowercased, split by CLIP's pattern, each piece byte-mapped and merged by rank,
// "</w>" ending the last symbol. Returns <|startoftext|> ids <|endoftext|>, cut to `maxLength` tokens (OWLv2: 16).
// Copied from the clipseg-rd64-refined runner; only the context length became a parameter.
import Foundation

final class CLIPTokenizer {
    private let vocab: [String: Int]
    private let ranks: [String: Int]
    private let byteMap: [Character]
    private let pattern = try! NSRegularExpression(
        pattern: #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#,
        options: [.caseInsensitive])
    private var cache: [String: [String]] = [:]
    let bos: Int, eos: Int

    init(_ dir: URL) throws {
        guard let vocab = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("vocab.json"))) as? [String: Int]
        else { throw RunnerError.failed("vocab.json is not a token table.") }
        self.vocab = vocab
        let lines = try String(contentsOf: dir.appendingPathComponent("merges.txt"), encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)
        var ranks: [String: Int] = [:]
        // tokenization_clip.py: merges[1 : 49152 - 256 - 2 + 1]
        for (rank, line) in lines.dropFirst().prefix(49152 - 256 - 2).enumerated() where !line.isEmpty { ranks[String(line)] = rank }
        self.ranks = ranks
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var chars = bytes
        var n = 0
        for b in 0..<256 where !bytes.contains(b) { bytes.append(b); chars.append(256 + n); n += 1 }
        var map = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bytes, chars) { map[b] = Character(UnicodeScalar(c)!) }
        byteMap = map
        guard let bos = vocab["<|startoftext|>"], let eos = vocab["<|endoftext|>"] else { throw RunnerError.failed("vocab.json lacks CLIP's special tokens.") }
        self.bos = bos; self.eos = eos
    }

    func encode(_ text: String, maxLength: Int = 77) -> [Int] {
        var cleaned = text.precomposedStringWithCanonicalMapping
        cleaned = String(String.UnicodeScalarView(cleaned.unicodeScalars.compactMap { scalar -> UnicodeScalar? in
            if scalar.value == 0 || scalar.value == 0xFFFD { return nil }
            if scalar.properties.isWhitespace { return " " }
            if scalar.properties.generalCategory == .control || scalar.properties.generalCategory == .format { return nil }
            return scalar
        }))
        cleaned = cleaned.split(whereSeparator: { $0 == " " }).joined(separator: " ").lowercased()
        var ids: [Int] = []
        let range = NSRange(cleaned.startIndex..., in: cleaned)
        for match in pattern.matches(in: cleaned, range: range) {
            guard let r = Range(match.range, in: cleaned) else { continue }
            let piece = String(String(cleaned[r]).utf8.map { byteMap[Int($0)] })
            for symbol in bpe(piece) { if let id = vocab[symbol] { ids.append(id) } }
        }
        return [bos] + ids.prefix(maxLength - 2) + [eos]
    }

    private func bpe(_ token: String) -> [String] {
        if let hit = cache[token] { return hit }
        var word = token.map { String($0) }
        word[word.count - 1] += "</w>"
        while word.count > 1 {
            var best: (rank: Int, index: Int)? = nil
            for i in 0..<(word.count - 1) {
                if let rank = ranks[word[i] + " " + word[i + 1]], best == nil || rank < best!.rank { best = (rank, i) }
            }
            guard let pair = best else { break }
            let first = word[pair.index], second = word[pair.index + 1]
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    merged.append(first + second); i += 2
                } else {
                    merged.append(word[i]); i += 1
                }
            }
            word = merged
        }
        cache[token] = word
        return word
    }
}
