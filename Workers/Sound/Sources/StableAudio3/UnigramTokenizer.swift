import Foundation

/// SentencePiece **BPE** encoder for the T5Gemma tokenizer bundled with SA3
/// (`TOKENIZER_MODEL` in `t5gemma_f16.npz`: model_type BPE, identity normalizer,
/// no dummy prefix, whitespace escaped to U+2581, byte fallback, 245 user-defined
/// pieces such as "▁▁" and "\n\n").
///
/// Replaces speech-swift's `UnigramTokenizer`, which split on spaces, merged over
/// grapheme clusters and keyed pieces by Swift `String` — Swift string equality is
/// canonical equivalence, so e.g. ";" (U+003B) and ";" (U+037E) collided and runs
/// of spaces tokenized differently from `sentencepiece`. This version mirrors
/// `bpe_model.cc`: user-defined pieces are atomic, the rest starts as single
/// Unicode scalars, the highest-scoring adjacent merge (leftmost on ties) is applied
/// until none remains, and leftover symbols fall back to `<0xHH>` byte pieces.
/// Verified against `sentencepiece.Encode` on the prompts in
/// Tests/StableAudio3Tests/Fixtures/tokens.json.
public final class UnigramTokenizer: @unchecked Sendable {   // name kept from upstream
    public let model: SentencePieceModel
    let pieceID: [[UInt8]: Int]          // byte-exact lookup
    let mergeable: Set<Int>              // normal + user-defined pieces
    let userDefined: [[UInt8]]           // longest first
    let byteID: [Int]                    // 256 entries (-1 if absent)
    public let unkId: Int

    public init(model: SentencePieceModel, unkId: Int = 3) {
        self.model = model
        self.unkId = unkId
        var map: [[UInt8]: Int] = [:]
        var merge = Set<Int>()
        var user: [[UInt8]] = []
        var bytes = [Int](repeating: -1, count: 256)
        for (i, p) in model.pieces.enumerated() {
            let b = Array(p.text.utf8)
            switch p.type {
            case SentencePieceModel.PieceType.normal.rawValue:
                map[b] = i; merge.insert(i)
            case SentencePieceModel.PieceType.userDefined.rawValue:
                map[b] = i; merge.insert(i); user.append(b)
            case SentencePieceModel.PieceType.byte.rawValue:
                // "<0xHH>"
                if p.text.count == 6, p.text.hasPrefix("<0x"), let v = UInt8(p.text.dropFirst(3).prefix(2), radix: 16) {
                    bytes[Int(v)] = i
                }
            default: break
            }
        }
        self.pieceID = map
        self.mergeable = merge
        self.userDefined = user.sorted { $0.count > $1.count }
        self.byteID = bytes
    }

    public func encodeAsIds(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        // Identity normalizer + whitespace escape (" " → U+2581); nothing else.
        let normalized = Array(text.unicodeScalars.map { $0 == " " ? "\u{2581}" : String($0) }.joined().utf8)

        // Initial symbols: user-defined pieces as atoms, else one scalar each.
        var symbols: [[UInt8]] = []
        var i = 0
        while i < normalized.count {
            if let u = userDefined.first(where: { u in
                i + u.count <= normalized.count && normalized[i ..< i + u.count].elementsEqual(u)
            }) {
                symbols.append(u); i += u.count; continue
            }
            let len = Self.utf8Length(normalized[i])
            symbols.append(Array(normalized[i ..< min(normalized.count, i + len)]))
            i += len
        }

        // Greedy BPE: best score first, leftmost on ties.
        while symbols.count >= 2 {
            var best = -1
            var bestScore = -Float.infinity
            for j in 0 ..< symbols.count - 1 {
                guard let id = pieceID[symbols[j] + symbols[j + 1]], mergeable.contains(id) else { continue }
                let s = model.pieces[id].score
                if s > bestScore { bestScore = s; best = j }
            }
            if best < 0 { break }
            symbols.replaceSubrange(best ... best + 1, with: [symbols[best] + symbols[best + 1]])
        }

        var ids: [Int] = []
        for s in symbols {
            if let id = pieceID[s] { ids.append(id); continue }
            for b in s { ids.append(byteID[Int(b)] >= 0 ? byteID[Int(b)] : unkId) }
        }
        return ids
    }

    static func utf8Length(_ lead: UInt8) -> Int {
        switch lead {
        case 0 ..< 0x80: return 1
        case 0xC0 ..< 0xE0: return 2
        case 0xE0 ..< 0xF0: return 3
        case 0xF0 ..< 0xF8: return 4
        default: return 1
        }
    }
}
