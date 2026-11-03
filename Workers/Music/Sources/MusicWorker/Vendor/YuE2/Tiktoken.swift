//
//  Tiktoken.swift
//  YuE2Music
//
//  Created by David Sherlock on 2026.
//
//  YuE2's frozen text and ABC tokenizer: Qwen's 151,643-rank byte BPE
//  (`qwen.tiktoken`), Qwen's split pattern, NFC first — `encode_ordinary`,
//  so special tokens in text are just text; the protocol adds their ids.
//  Byte-pair merging by lowest rank, as tiktoken merges.
//

import Foundation

struct Tiktoken {
    private let ranks: [Data: Int]
    private let bytesOf: [Int: Data]
    private let specials: [Int: String]
    private let pattern: NSRegularExpression

    static let splitPattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    init(file: URL) throws(YuE2Error) {
        let text: String
        do { text = try String(contentsOf: file, encoding: .utf8) } catch { throw .modelMissing("qwen.tiktoken unreadable") }
        var ranks: [Data: Int] = [:]
        var bytesOf: [Int: Data] = [:]
        ranks.reserveCapacity(151_643)
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard parts.count == 2, let bytes = Data(base64Encoded: String(parts[0])), let rank = Int(parts[1]) else { continue }
            ranks[bytes] = rank
            bytesOf[rank] = bytes
        }
        guard ranks.count == 151_643 else { throw .modelMissing("qwen.tiktoken has \(ranks.count) ranks, not 151,643") }
        self.ranks = ranks
        self.bytesOf = bytesOf
        var specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        specials += (0..<200).map { "<extra_\($0)>" }
        specials[204] = "<abc>"
        specials[205] = "</abc>"
        self.specials = Dictionary(uniqueKeysWithValues: specials.enumerated().map { (ranks.count + $0.offset, $0.element) })
        do { pattern = try NSRegularExpression(pattern: Self.splitPattern) } catch { throw .modelMissing("split pattern: \(error)") }
    }

    // Lab-only byte vocabulary; the ordinary BPE/token protocol and full ID space remain in use.
    init(randomFixture: Bool) throws {
        guard randomFixture else { throw YuE2Error.modelMissing("Invalid fixture tokenizer") }
        let tokens = (0..<YuE2Protocol.eod).map { id in (id, id < 256 ? Data([UInt8(id)]) : Data(" token\(id)".utf8)) }
        ranks = Dictionary(uniqueKeysWithValues: tokens.map { ($0.1, $0.0) })
        bytesOf = Dictionary(uniqueKeysWithValues: tokens)
        specials = [:]
        pattern = try NSRegularExpression(pattern: Self.splitPattern)
    }

    // Ordinary ids for `text` (NFC-normalised first).
    func encode(_ text: String) -> [Int] {
        let normalised = text.precomposedStringWithCanonicalMapping
        let ns = normalised as NSString
        var ids: [Int] = []
        for match in pattern.matches(in: normalised, range: NSRange(location: 0, length: ns.length)) {
            let piece = Data(ns.substring(with: match.range).utf8)
            if let rank = ranks[piece] { ids.append(rank); continue }
            ids += merge(piece)
        }
        return ids
    }

    // tiktoken's byte-pair merge: repeatedly join the adjacent pair whose bytes rank lowest.
    private func merge(_ piece: Data) -> [Int] {
        var parts: [Data] = piece.map { Data([$0]) }
        while parts.count > 1 {
            var best: (rank: Int, index: Int)?
            for i in 0..<(parts.count - 1) {
                if let rank = ranks[parts[i] + parts[i + 1]], best == nil || rank < best!.rank { best = (rank, i) }
            }
            guard let best else { break }
            // swiftlint:disable:next shorthand_operator
            parts[best.index] = parts[best.index] + parts[best.index + 1]
            parts.remove(at: best.index + 1)
        }
        return parts.compactMap { ranks[$0] }
    }

    // Text for ids; special and out-of-range ids as tiktoken would (specials by name).
    func decode(_ ids: [Int]) -> String {
        var data = Data()
        for id in ids {
            if let bytes = bytesOf[id] { data += bytes } else if let special = specials[id] { data += Data(special.utf8) }
        }
        return String(decoding: data, as: UTF8.self)
    }
}
