// SPDX-License-Identifier: MIT
import Foundation

/// The checkpoint's `tokenizer.json` (LFM2: byte-level BPE, `ignore_merges`, one Split regex, 124 added tokens), as
/// HF `tokenizers` 0.22 encodes it with `add_special_tokens=False`. Anything else in the file is refused at load, so a
/// different tokenizer revision fails loudly instead of tokenizing differently.
public final class LFMTokenizer: @unchecked Sendable {
    private let vocab: [[UInt8]: Int]
    private let merges: [UInt64: (rank: Int, id: Int)]
    private let byteIDs: [Int] // byte -> id of its byte-level character
    private let added: [(scalars: [Unicode.Scalar], id: Int)] // longest first per first scalar
    private let addedByFirst: [UInt32: [Int]] // first scalar -> indexes into `added`
    private let addedIDs: [String: Int]
    private let lock = NSLock()
    private var cache: [[UInt8]: [Int]] = [:]

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
            model["byte_fallback"] as? Bool == false, model["ignore_merges"] as? Bool == true,
            model["continuing_subword_prefix"] is NSNull || model["continuing_subword_prefix"] == nil,
            model["end_of_word_suffix"] is NSNull || model["end_of_word_suffix"] == nil,
            root["normalizer"] is NSNull,
            let pre = root["pre_tokenizer"] as? [String: Any], pre["type"] as? String == "Sequence",
            let steps = pre["pretokenizers"] as? [[String: Any]], steps.count == 2,
            steps[0]["type"] as? String == "Split", steps[0]["behavior"] as? String == "Isolated", steps[0]["invert"] as? Bool == false,
            (steps[0]["pattern"] as? [String: Any])?["Regex"] as? String == Self.pattern,
            steps[1]["type"] as? String == "ByteLevel", steps[1]["add_prefix_space"] as? Bool == false,
            steps[1]["use_regex"] as? Bool == false,
            let rawVocab = model["vocab"] as? [String: Int], let rawMerges = model["merges"] as? [[String]],
            let addedRows = root["added_tokens"] as? [[String: Any]]
        else { throw JSONError("tokenizer.json is not the LFM2 byte-level BPE this runner implements") }
        var vocab = [[UInt8]: Int](minimumCapacity: rawVocab.count)
        for (word, id) in rawVocab { vocab[Array(word.utf8)] = id }
        guard vocab.count == rawVocab.count else { throw JSONError("tokenizer.json vocab has keys equal only up to Unicode normalisation") }
        let toByte = Self.byteLevelChars()
        var byteIDs = [Int](repeating: -1, count: 256)
        for b in 0..<256 {
            guard let id = vocab[Array(String(toByte[b]).utf8)] else { throw JSONError("tokenizer.json lacks byte \(b)") }
            byteIDs[b] = id
        }
        var merges = [UInt64: (rank: Int, id: Int)](minimumCapacity: rawMerges.count)
        for (rank, pair) in rawMerges.enumerated() {
            guard pair.count == 2, let a = vocab[Array(pair[0].utf8)], let b = vocab[Array(pair[1].utf8)],
                let m = vocab[Array((pair[0] + pair[1]).utf8)]
            else { throw JSONError("tokenizer.json merge \(rank) is not in the vocab") }
            let key = UInt64(a) << 32 | UInt64(b)
            if merges[key] == nil { merges[key] = (rank, m) }
        }
        var added: [(scalars: [Unicode.Scalar], id: Int)] = []
        var addedIDs: [String: Int] = [:]
        for row in addedRows {
            guard let content = row["content"] as? String, let id = row["id"] as? Int, row["single_word"] as? Bool == false,
                row["lstrip"] as? Bool == false, row["rstrip"] as? Bool == false, !content.isEmpty
            else { throw JSONError("tokenizer.json has an added token this runner does not implement") }
            added.append((Array(content.unicodeScalars), id))
            addedIDs[content] = id
        }
        added.sort { $0.scalars.count > $1.scalars.count }
        var byFirst: [UInt32: [Int]] = [:]
        for (i, a) in added.enumerated() { byFirst[a.scalars[0].value, default: []].append(i) }
        self.vocab = vocab
        self.merges = merges
        self.byteIDs = byteIDs
        self.added = added
        self.addedByFirst = byFirst
        self.addedIDs = addedIDs
    }

    static let pattern = #"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s"#

    public func tokenID(_ token: String) -> Int? { addedIDs[token] ?? vocab[Array(token.utf8)] }

    /// `tokenizer.encode(text, add_special_tokens=False).ids`.
    public func encode(_ text: String) -> [Int] {
        let s = Array(text.unicodeScalars)
        var out: [Int] = []
        out.reserveCapacity(s.count / 3)
        var start = 0
        var i = 0
        // Added tokens split the text first: leftmost, then longest.
        while i < s.count {
            if let candidates = addedByFirst[s[i].value],
                let hit = candidates.first(where: { c in
                    let a = added[c].scalars
                    return i + a.count <= s.count && s[i..<(i + a.count)].elementsEqual(a)
                })
            {
                if start < i { ordinary(s[start..<i], into: &out) }
                out.append(added[hit].id)
                i += added[hit].scalars.count
                start = i
            } else {
                i += 1
            }
        }
        if start < s.count { ordinary(s[start..<s.count], into: &out) }
        return out
    }

    // MARK: - pre-tokenizer: the Split regex, hand-written (leftmost-first alternation, as Rust's regex)

    private static func isL(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }
    private static func isN(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }
    private static func isS(_ c: Unicode.Scalar) -> Bool { c.properties.isWhitespace }
    private static func isCRLF(_ c: Unicode.Scalar) -> Bool { c == "\r" || c == "\n" }

    /// Length of the regex match starting at `i` (always >= 1).
    static func piece(_ s: ArraySlice<Unicode.Scalar>, _ i: Int) -> Int {
        let n = s.endIndex
        let c = s[i]
        // '(?i:[sdmt]|ll|ve|re)
        if c == "'" && i + 1 < n {
            let a = s[i + 1]
            if ["s", "S", "\u{17F}", "d", "D", "m", "M", "t", "T"].contains(a) { return 2 }
            if i + 2 < n {
                let b = s[i + 2]
                let pairs: [(Set<Unicode.Scalar>, Set<Unicode.Scalar>)] = [(["l", "L"], ["l", "L"]), (["v", "V"], ["e", "E"]), (["r", "R"], ["e", "E"])]
                if pairs.contains(where: { $0.0.contains(a) && $0.1.contains(b) }) { return 3 }
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        var j = i
        if !isCRLF(c) && !isL(c) && !isN(c) && i + 1 < n && isL(s[i + 1]) { j = i + 1 }
        if isL(s[j]) {
            var k = j
            while k < n && isL(s[k]) { k += 1 }
            return k - i
        }
        // \p{N}{1,3}
        if isN(c) {
            var k = i
            while k < n && k - i < 3 && isN(s[k]) { k += 1 }
            return k - i
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let other = { (x: Unicode.Scalar) in !isS(x) && !isL(x) && !isN(x) }
        var o = i
        if c == " " && i + 1 < n && other(s[i + 1]) { o = i + 1 }
        if other(s[o]) {
            var k = o
            while k < n && other(s[k]) { k += 1 }
            while k < n && isCRLF(s[k]) { k += 1 }
            return k - i
        }
        // \s*[\r\n]  |  \s+(?!\S)  |  \s
        if isS(c) {
            var e = i
            while e < n && isS(s[e]) { e += 1 }
            if let last = (i..<e).last(where: { isCRLF(s[$0]) }) { return last + 1 - i }
            if e == n { return e - i }
            if e - i >= 2 { return e - 1 - i }
            return 1
        }
        return 1 // unreachable: every scalar is a letter, number, space or other
    }

    private func ordinary(_ s: ArraySlice<Unicode.Scalar>, into out: inout [Int]) {
        var i = s.startIndex
        while i < s.endIndex {
            let length = Self.piece(s, i)
            var bytes: [UInt8] = []
            for c in s[i..<(i + length)] { bytes.append(contentsOf: String(c).utf8) }
            out.append(contentsOf: word(bytes))
            i += length
        }
    }

    // MARK: - byte-level BPE

    private func word(_ bytes: [UInt8]) -> [Int] {
        lock.lock()
        if let hit = cache[bytes] {
            lock.unlock()
            return hit
        }
        lock.unlock()
        let mapped = Self.mapped(bytes)
        let ids: [Int]
        if let whole = vocab[mapped] {
            ids = [whole] // ignore_merges: a word that is a token is that token
        } else {
            ids = bpe(bytes.map { byteIDs[Int($0)] })
        }
        if bytes.count <= 64 {
            lock.lock()
            if cache.count > 65_536 { cache.removeAll(keepingCapacity: true) }
            cache[bytes] = ids
            lock.unlock()
        }
        return ids
    }

    /// A linked list of symbols plus a min-heap of adjacent merges. Rank first, then original left index:
    /// this preserves the old scan's leftmost tie-break without shifting or rescanning the surviving symbols.
    private func bpe(_ input: [Int]) -> [Int] {
        guard input.count > 1 else { return input }
        struct Merge {
            let rank: Int
            let left: Int
            let right: Int
            let leftID: Int
            let rightID: Int
            let id: Int
            func precedes(_ other: Merge) -> Bool {
                rank < other.rank || (rank == other.rank && left < other.left)
            }
        }
        var symbols = input
        var previous = input.indices.map { $0 - 1 }
        var next = input.indices.map { $0 + 1 < input.count ? $0 + 1 : -1 }
        var alive = [Bool](repeating: true, count: input.count)
        var heap: [Merge] = []
        heap.reserveCapacity(input.count)
        func push(_ left: Int) {
            guard left >= 0, alive[left], next[left] >= 0 else { return }
            let right = next[left]
            guard let m = merges[UInt64(symbols[left]) << 32 | UInt64(symbols[right])] else { return }
            heap.append(Merge(rank: m.rank, left: left, right: right, leftID: symbols[left], rightID: symbols[right], id: m.id))
            var i = heap.count - 1
            while i > 0 {
                let parent = (i - 1) / 2
                if !heap[i].precedes(heap[parent]) { break }
                heap.swapAt(i, parent)
                i = parent
            }
        }
        func pop() -> Merge {
            let result = heap[0]
            let last = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = last
                var i = 0
                while 2 * i + 1 < heap.count {
                    var child = 2 * i + 1
                    if child + 1 < heap.count && heap[child + 1].precedes(heap[child]) { child += 1 }
                    if !heap[child].precedes(heap[i]) { break }
                    heap.swapAt(i, child)
                    i = child
                }
            }
            return result
        }
        for i in input.indices { push(i) }
        while !heap.isEmpty {
            let m = pop()
            // Entries touching a merged symbol are stale; only its two new neighbours are queued again.
            guard alive[m.left], alive[m.right], next[m.left] == m.right,
                symbols[m.left] == m.leftID, symbols[m.right] == m.rightID
            else { continue }
            symbols[m.left] = m.id
            alive[m.right] = false
            next[m.left] = next[m.right]
            if next[m.right] >= 0 { previous[next[m.right]] = m.left }
            push(previous[m.left])
            push(m.left)
        }
        return input.indices.compactMap { alive[$0] ? symbols[$0] : nil }
    }

    private static let table: [[UInt8]] = byteLevelChars().map { Array(String($0).utf8) }

    private static func mapped(_ bytes: [UInt8]) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(bytes.count * 2)
        for b in bytes { out.append(contentsOf: table[Int(b)]) }
        return out
    }

    /// GPT-2's `bytes_to_unicode`.
    static func byteLevelChars() -> [Unicode.Scalar] {
        var printable = Array(33...126) + Array(161...172) + Array(174...255)
        var chars = printable
        var n = 0
        for b in 0..<256 where !printable.contains(b) {
            printable.append(b)
            chars.append(256 + n)
            n += 1
        }
        var out = [Unicode.Scalar](repeating: " ", count: 256)
        for (b, c) in zip(printable, chars) { out[b] = Unicode.Scalar(UInt32(c))! }
        return out
    }
}
