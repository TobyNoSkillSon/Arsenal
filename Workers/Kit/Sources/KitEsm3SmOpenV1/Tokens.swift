// SPDX-License-Identifier: MIT
// ESM3 tokenizers, ported from esm/tokenization/* and esm/utils/function/* (esm commit 8f0834c). Foundation only: no
// MLX, no weights, so Tests/KitModels/esm3-sm-open-v1 compiles and checks this file on its own.
import Foundation

/// Token ids and vocabularies (esm/utils/constants/esm3.py).
enum ESM3Vocab {
    static let sequence = [
        "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W",
        "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>",
    ]
    static let sequenceBOS = 0, sequencePAD = 1, sequenceEOS = 2, sequenceUNK = 3, sequenceChainBreak = 31, sequenceMask = 32
    /// EsmSequenceTokenizer.all_special_ids: <unk>, <cls>, <pad>, <mask>, <eos>, "|".
    static let sequenceSpecial: Set<Int> = [0, 1, 2, 3, 31, 32]
    static let structureMask = 4096, structureEOS = 4097, structureBOS = 4098, structurePAD = 4099, structureChainBreak = 4100
    static let structureCodes = 4096
    static let ss8 = ["<pad>", "<motif>", "<unk>", "G", "H", "I", "T", "E", "B", "S", "C"]
    static let sasaBoundaries: [Double] = [0.8, 4.0, 9.6, 16.4, 24.5, 32.9, 42.0, 51.5, 61.2, 70.9, 81.6, 93.3, 107.2, 125.4, 151.4]
    static let functionDepth = 8, functionVocab = 260, functionNone = 3, functionLSHOffset = 4
    static let residueSlots = 16, residueVocab = 1478
    /// The 20 standard amino acids, in token order (ids 4 ... 23).
    static let standardAminoAcids = Array(sequence[4 ... 23])
    static let sequenceIndex: [String: Int] = Dictionary(uniqueKeysWithValues: sequence.enumerated().map { ($1, $0) })
    static let ss8Index: [Character: Int] = Dictionary(uniqueKeysWithValues: ss8.enumerated().compactMap { $1.count == 1 ? (Character($1), $0) : nil })

    /// Sampling vocabularies (generation.py `_sample_per_prompt`): every token minus the tokenizer's special tokens.
    static let sequenceValid = Array(4 ... 30)
    static let structureValid = Array(0 ..< 4096)
    static let ss8Valid = Array(3 ... 10)
    static let sasaValid = Array(3 ... 18)
}

/// A request that cannot be tokenized (unknown letters, labels out of range): the caller can fix it.
struct TokenError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

enum SequenceTokens {
    /// `<cls>` residues `<eos>`; "_" is a masked residue (encoding.tokenize_sequence). Whitespace is ignored.
    static func encode(_ text: String) throws -> [Int] {
        var ids = [ESM3Vocab.sequenceBOS]
        for character in text where !character.isWhitespace {
            if character == "_" { ids.append(ESM3Vocab.sequenceMask); continue }
            if character == "|" { throw TokenError("Multi-chain complexes (\"|\" chain breaks) are not covered by this runner; send one chain.") }
            guard let id = ESM3Vocab.sequenceIndex[String(character)], (4 ... 30).contains(id) else {
                throw TokenError("Sequence may hold only uppercase one-letter amino-acid codes (\(ESM3Vocab.sequence[4 ... 28].joined())) and _ for a masked residue; found \"\(character)\".")
            }
            ids.append(id)
        }
        ids.append(ESM3Vocab.sequenceEOS)
        guard ids.count > 2 else { throw TokenError("Sequence is empty.") }
        return ids
    }

    /// decoding.decode_sequence: residue letters, "_" for a mask; <cls>/<eos> are dropped.
    static func decode(_ ids: [Int]) -> String {
        ids.compactMap { id -> String? in
            switch id {
            case ESM3Vocab.sequenceBOS, ESM3Vocab.sequenceEOS: return nil
            case ESM3Vocab.sequenceMask: return "_"
            case 0 ..< ESM3Vocab.sequence.count: return ESM3Vocab.sequence[id]
            default: return "X"
            }
        }.joined()
    }
}

enum SS8Tokens {
    /// One <pad> at each end; "_" (and <pad>) is the mask, which is also <pad> (ss_tokenizer.py).
    static func encode(_ text: String, length: Int) throws -> [Int] {
        let letters = Array(text.filter { !$0.isWhitespace })
        guard letters.count == length else { throw TokenError("secondary_structure has \(letters.count) letters; the protein has \(length) residues.") }
        var ids = [0]
        for letter in letters {
            if letter == "_" { ids.append(0); continue }
            guard let id = ESM3Vocab.ss8Index[letter], id >= 3 else {
                throw TokenError("secondary_structure may hold only the 8 DSSP classes G H I T E B S C and _ for unknown; found \"\(letter)\".")
            }
            ids.append(id)
        }
        ids.append(0)
        return ids
    }

    static func decode(_ ids: [Int]) -> String {
        ids.dropFirst().dropLast().map { $0 >= 3 && $0 < ESM3Vocab.ss8.count ? ESM3Vocab.ss8[$0] : "_" }.joined()
    }
}

enum SASATokens {
    /// SASADiscretizingTokenizer.encode: a value goes to bucket `torch.bucketize(v, boundaries)` (first boundary >= v);
    /// nil is the mask (<pad>). One <pad> at each end.
    static func encode(_ values: [Double?], length: Int) throws -> [Int] {
        guard values.count == length else { throw TokenError("sasa has \(values.count) values; the protein has \(length) residues.") }
        var ids = [0]
        for value in values {
            guard let value else { ids.append(0); continue }
            guard value.isFinite, value >= 0 else { throw TokenError("sasa values must be finite and non-negative (null for unknown).") }
            ids.append(3 + bucket(value))
        }
        ids.append(0)
        return ids
    }

    static func bucket(_ value: Double) -> Int { ESM3Vocab.sasaBoundaries.firstIndex { value <= $0 } ?? ESM3Vocab.sasaBoundaries.count }

    /// Range midpoints of tokens 3 ... 18 (the last range ends at twice the last boundary).
    static let midpoints: [Double] = {
        let edges = [0.0] + ESM3Vocab.sasaBoundaries + [ESM3Vocab.sasaBoundaries.last! * 2]
        return (0 ..< 16).map { (edges[$0] + edges[$0 + 1]) / 2 }
    }()

    /// decode_float: a token's range midpoint, nil for special tokens.
    static func decode(_ ids: [Int]) -> [Double?] { ids.dropFirst().dropLast().map { $0 >= 3 && $0 <= 18 ? midpoints[$0 - 3] : nil } }

    /// sampling.sample_sasa_logits: the expected value over tokens 3 ... 17 with the 15 midpoints of [0] + boundaries;
    /// infinity where the most likely token is the open-ended last range.
    static func expected(probabilities p: [Double]) -> Double {
        let edges = [0.0] + ESM3Vocab.sasaBoundaries
        let centers = (0 ..< 15).map { (edges[$0] + edges[$0 + 1]) / 2 }
        let best = p.indices.max { p[$0] < p[$1] } ?? 0
        if best == 18 { return .infinity }
        return (0 ..< 15).reduce(0) { $0 + p[3 + $1] * centers[$1] }
    }
}

/// A labelled residue range, 1-indexed and inclusive, as esm.utils.types.FunctionAnnotation.
struct Annotation: Equatable {
    var label: String
    var start: Int
    var end: Int
}

enum Annotations {
    /// misc.merge_ranges over [start, end + 1): sorted by start, joined when the gap is at most `gap`.
    static func mergeRanges(_ ranges: [Range<Int>], gap: Int) -> [Range<Int>] {
        var merged: [Range<Int>] = []
        for r in ranges.sorted(by: { $0.lowerBound < $1.lowerBound }) {
            if let last = merged.last, last.upperBound + gap >= r.lowerBound {
                merged[merged.count - 1] = last.lowerBound ..< Swift.max(last.upperBound, r.upperBound)
            } else {
                merged.append(r)
            }
        }
        return merged
    }

    /// misc.merge_annotations: per label (in first-seen order), merge its ranges.
    static func merge(_ annotations: [Annotation], gap: Int) -> [Annotation] {
        var order: [String] = [], ranges: [String: [Range<Int>]] = [:]
        for a in annotations {
            if ranges[a.label] == nil { order.append(a.label) }
            ranges[a.label, default: []].append(a.start ..< (a.end + 1))
        }
        return order.flatMap { label in mergeRanges(ranges[label]!, gap: gap).map { Annotation(label: label, start: $0.lowerBound, end: $0.upperBound - 1) } }
    }

    static func dropShort(_ annotations: [Annotation], minLength: Int) -> [Annotation] {
        annotations.filter { $0.end - $0.start + 1 >= minLength }
    }
}

/// TF-IDF over the keyword vocabulary (esm/utils/function/tfidf.py), sparse as index -> value.
struct TFIDF {
    let vocabulary: [String]
    let index: [String: Int]
    let idf: [Double]

    init(vocabulary: [String], idf: [Double]) throws {
        guard vocabulary.count == idf.count else { throw TokenError("IDF size \(idf.count) differs from vocabulary size \(vocabulary.count).") }
        self.vocabulary = vocabulary
        self.idf = idf
        var index: [String: Int] = [:]
        for (i, term) in vocabulary.enumerated() where index[term] == nil { index[term] = i }
        self.index = index
    }

    /// (1 + log tf) * idf over the known terms, L2-normalised. Unknown terms are ignored.
    func encode(_ terms: [String]) -> [Int: Double] {
        var counts: [Int: Int] = [:]
        for term in terms { if let i = index[term] { counts[i, default: 0] += 1 } }
        var values: [Int: Double] = [:]
        for (i, tf) in counts { values[i] = (1 + Foundation.log(Double(tf))) * idf[i] }
        let norm = values.values.reduce(0) { $0 + $1 * $1 }.squareRoot()
        guard norm > 0 else { return [:] }
        return values.mapValues { $0 / norm }
    }
}

/// The function-annotation tokenizer (InterProQuantizedTokenizer): annotation ranges -> per-residue sets of keywords
/// and InterPro ids -> a TF-IDF vector -> 8 locality-sensitive hashes of 8 bits each.
struct FunctionTokenizer {
    let tfidf: TFIDF
    /// interpro2keywords: InterPro id -> its keywords. Its sorted keys are the supported InterPro labels.
    let interproKeywords: [String: [String]]
    /// One [8 x vocabulary] table of hyperplanes per depth position (hyperplanes_8bit_58641.npz, "0" ... "7").
    let hyperplanes: [[Double]]
    /// Residue-annotation labels (label_clean, ranked by count) and the raw description -> label map.
    let residueLabels: [String]
    let residueLabelSet: Set<String>

    init(tfidf: TFIDF, interproKeywords: [String: [String]], hyperplanes: [[Double]], residueLabels: [String]) throws {
        guard hyperplanes.count == ESM3Vocab.functionDepth, hyperplanes.allSatisfy({ $0.count == 8 * tfidf.vocabulary.count }) else {
            throw TokenError("LSH hyperplanes must be 8 tables of 8 x \(tfidf.vocabulary.count).")
        }
        self.tfidf = tfidf
        self.interproKeywords = interproKeywords
        self.hyperplanes = hyperplanes
        self.residueLabels = residueLabels
        residueLabelSet = Set(residueLabels)
    }

    enum Kind { case function, residue }

    /// encode_decode.encode_function_annotations: which tracks a label feeds (it may feed both), or an error.
    func kinds(of label: String) throws -> [Kind] {
        var kinds: [Kind] = []
        if let ipr = Self.interproID(label), interproKeywords[ipr] != nil { kinds.append(.function) }
        if tfidf.index[label] != nil { kinds.append(.function) }
        if residueLabelSet.contains(label) { kinds.append(.residue) }
        guard !kinds.isEmpty else {
            throw TokenError("Unknown function label \"\(label)\": use a function keyword from the keyword vocabulary, an InterPro id (IPR…) the model knows, or a residue annotation label.")
        }
        return kinds
    }

    static func interproID(_ label: String) -> String? {
        guard let range = label.range(of: #"IPR\d+"#, options: .regularExpression) else { return nil }
        return String(label[range])
    }

    /// The LSH hash of a set of labels (`_function_text_hash`), nil when it has no keyword weight (<unk>).
    func hash(_ labels: Set<String>) throws -> [Int]? {
        var keywords: [String] = [], interpro: [String] = []
        for label in labels.sorted() {
            if let ipr = Self.interproID(label), interproKeywords[ipr] != nil {
                interpro.append(ipr)
            } else if tfidf.index[label] != nil {
                keywords.append(label)
            } else {
                throw TokenError("Unsupported function label \"\(label)\".")
            }
        }
        var vector = tfidf.encode(keywords)
        for ipr in interpro {
            for (i, v) in tfidf.encode(interproKeywords[ipr] ?? []) { vector[i] = Swift.max(vector[i] ?? 0, v) }
        }
        let sum = vector.values.reduce(0, +)
        guard sum != 0, !sum.isNaN else { return nil }
        let size = tfidf.vocabulary.count
        return hyperplanes.map { table in
            var token = 0
            for bit in 0 ..< 8 {
                var similarity = 0.0
                for (i, v) in vector { similarity += table[bit * size + i] * v }
                if similarity >= 0 { token |= 1 << bit }
            }
            return token
        }
    }

    /// Function tokens [L + 2][8] for annotations over a sequence of `length` residues (pad rows at both ends).
    /// Annotations only of residue labels give the residue track's tokens; the maker's tokenizer then leaves that track
    /// all <pad> (its sample has no `interpro_site_residues`), so residue labels are accepted and change nothing.
    func encode(_ annotations: [Annotation], length: Int) throws -> [[Int]] {
        var positional = [Set<String>](repeating: [], count: length)
        var functionAnnotations: [Annotation] = []
        for a in annotations {
            guard 1 <= a.start, a.start <= a.end, a.end <= length else {
                throw TokenError("Annotation \"\(a.label)\" range \(a.start)-\(a.end) is outside 1-\(length).")
            }
            if try kinds(of: a.label).contains(.function) { functionAnnotations.append(a) }
        }
        let pad = [Int](repeating: 0, count: ESM3Vocab.functionDepth)
        guard !functionAnnotations.isEmpty else { return [[Int]](repeating: pad, count: length + 2) }
        for a in functionAnnotations { for i in (a.start - 1) ..< a.end { positional[i].insert(a.label) } }
        var cache: [Set<String>: [Int]?] = [:]
        var rows = [pad]
        for labels in positional {
            if labels.isEmpty { rows.append([Int](repeating: ESM3Vocab.functionNone, count: ESM3Vocab.functionDepth)); continue }
            let hash: [Int]?
            if let cached = cache[labels] { hash = cached } else { hash = try self.hash(labels); cache[labels] = hash }
            if let hash { rows.append(hash.map { $0 + ESM3Vocab.functionLSHOffset }) } else { rows.append([Int](repeating: 2, count: ESM3Vocab.functionDepth)) }
        }
        rows.append(pad)
        return rows
    }
}

/// Reading the tokenizer's data files from the converted snapshot.
enum TokenData {
    static func lines(_ url: URL) throws -> [String] {
        try String(contentsOf: url, encoding: .utf8).trimmingCharacters(in: .newlines).components(separatedBy: "\n")
    }

    /// A minimal CSV row split: commas outside double quotes; "" inside quotes is a quote.
    static func csvRow(_ line: String) -> [String] {
        var fields: [String] = [], field = "", quoted = false
        var chars = Array(line)[...]
        while let c = chars.popFirst() {
            if quoted {
                if c == "\"" {
                    if chars.first == "\"" { field.append("\""); chars.removeFirst() } else { quoted = false }
                } else { field.append(c) }
            } else if c == "\"" {
                quoted = true
            } else if c == "," {
                fields.append(field); field = ""
            } else { field.append(c) }
        }
        fields.append(field)
        return fields
    }

    /// interpro_29026_to_keywords_58641.csv: interpro_id,keywords (comma-separated inside quotes).
    static func interproKeywords(_ url: URL) throws -> [String: [String]] {
        let rows = try lines(url)
        guard rows.first?.trimmingCharacters(in: .whitespaces) == "interpro_id,keywords" else { throw TokenError("\(url.lastPathComponent): unexpected header.") }
        var out: [String: [String]] = [:]
        for row in rows.dropFirst() where !row.isEmpty {
            let fields = csvRow(row)
            guard fields.count >= 2 else { continue }
            out[fields[0]] = fields[1].isEmpty ? [] : fields[1].components(separatedBy: ",")
        }
        return out
    }

    /// The residue-annotation CSV: label_clean ranked by summed count, descending, ties in first-seen order
    /// (pandas groupby sorts keys, then a stable sort by count), and the raw label -> label_clean map.
    static func residueLabels(_ url: URL) throws -> (labels: [String], clean: [String: String]) {
        let rows = try lines(url)
        guard let header = rows.first.map(csvRow), let labelColumn = header.firstIndex(of: "label"),
            let countColumn = header.firstIndex(of: "count"), let cleanColumn = header.firstIndex(of: "label_clean")
        else { throw TokenError("\(url.lastPathComponent): unexpected header.") }
        var totals: [String: Int] = [:], clean: [String: String] = [:]
        for row in rows.dropFirst() where !row.isEmpty {
            let fields = csvRow(row)
            guard fields.count > Swift.max(labelColumn, countColumn, cleanColumn) else { continue }
            totals[fields[cleanColumn], default: 0] += Int(fields[countColumn]) ?? 0
            clean[fields[labelColumn]] = fields[cleanColumn]
        }
        let sortedKeys = totals.keys.sorted()
        let labels = sortedKeys.enumerated().sorted { a, b in
            totals[a.element]! != totals[b.element]! ? totals[a.element]! > totals[b.element]! : a.offset < b.offset
        }.map(\.element)
        return (labels, clean)
    }

    /// entry_list_safety_29026.list (tab-separated ENTRY_AC, ENTRY_TYPE, ENTRY_NAME): ids sorted, and id -> name.
    static func interproEntries(_ url: URL) throws -> (ids: [String], names: [String: String]) {
        let rows = try lines(url)
        guard rows.first?.hasPrefix("ENTRY_AC\tENTRY_TYPE\tENTRY_NAME") == true else { throw TokenError("\(url.lastPathComponent): unexpected header.") }
        var names: [String: String] = [:]
        for row in rows.dropFirst() where !row.isEmpty {
            let fields = row.components(separatedBy: "\t")
            guard fields.count >= 3 else { continue }
            names[fields[0]] = fields[2]
        }
        return (names.keys.sorted(), names)
    }
}

/// A float64 or float32 array from a safetensors file, read on the host without MLX (the converted .npy/.npz data).
enum HostTensors {
    static func read(_ url: URL) throws -> [String: (shape: [Int], values: [Double])] {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 8 else { throw TokenError("\(url.lastPathComponent) is not a safetensors file.") }
        let length = data.prefix(8).enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        guard length > 0, 8 + length <= data.count,
            let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< (8 + length))) as? [String: Any]
        else { throw TokenError("\(url.lastPathComponent) has no safetensors header.") }
        var out: [String: (shape: [Int], values: [Double])] = [:]
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String, let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
            else { throw TokenError("\(url.lastPathComponent): bad entry \(name).") }
            let start = 8 + length + offsets[0], end = 8 + length + offsets[1]
            guard start <= end, end <= data.count else { throw TokenError("\(url.lastPathComponent): \(name) is out of bounds.") }
            let slice = data.subdata(in: start ..< end)
            let values: [Double]
            switch dtype {
            case "F64": values = slice.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
            case "F32": values = slice.withUnsafeBytes { $0.bindMemory(to: Float.self).map(Double.init) }
            default: throw TokenError("\(url.lastPathComponent): \(name) is \(dtype), not a float array.")
            }
            guard values.count == shape.reduce(1, *) else { throw TokenError("\(url.lastPathComponent): \(name) has the wrong size.") }
            out[name] = (shape, values)
        }
        return out
    }
}

/// The unmasking schedule of iterative generation (noise_schedules.py, generation.py).
enum Schedule {
    static let names = ["cosine", "linear"]

    /// Fraction of the positions still masked after `step` (0-based) of `steps`.
    static func maskedFraction(_ schedule: String, step: Int, steps: Int) -> Double {
        let t = Double(step + 1) / Double(steps)
        switch schedule {
        case "linear": return 1 - t
        default: return Double(cos(Float(t) * Float.pi * 0.5))  // float32, as torch.cos on a float32 tensor
        }
    }

    /// How many positions to unmask now: still masked minus int(fraction * total + 0.1), as the maker computes in float32.
    static func toSample(_ schedule: String, step: Int, steps: Int, stillMasked: Int, total: Int) -> Int {
        let remaining = Int(Float(maskedFraction(schedule, step: step, steps: steps)) * Float(total) + 0.1)
        return Swift.max(0, stillMasked - remaining)
    }

    /// _get_annealed_temperature: (max(T - step / (steps - 1), 0.001))^2.
    static func annealed(_ temperature: Double, step: Int, steps: Int) -> Double {
        let ratio = Double(step) / Double(Swift.max(1, steps - 1))
        return Foundation.pow(Swift.max(temperature - ratio, 0.001), 2)
    }
}

/// Lab capture file names (ARSENAL_ESM3_CAPTURE_DIR): letters, digits, dot, dash and underscore only, never a path.
enum CaptureName {
    static func sanitized(_ name: String) -> String? {
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        guard !name.isEmpty, name.count <= 64, !name.hasPrefix("."), !name.contains(".."), name.allSatisfy(allowed.contains) else { return nil }
        return name
    }
}

/// A small deterministic generator (SplitMix64) for sampling and the random unmasking order, seeded per request.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Limits that keep a request inside what Arsenal carries and times: a reply travels in one frame of at most 16 MiB
/// (at most 26 bytes per number as JSON: 24 for the longest float with its comma, 2 of brackets), and local.protein ops time out after 900 s. Checked before the GPU runs.
enum RequestLimits {
    /// Numbers in one reply's large arrays (per-residue embeddings, logits, PAE): under 16 MiB as JSON in the worst case.
    static let replyNumbers = 640_000
    /// Model passes in one generate request, summed over its tracks, so it finishes inside the 900 s timeout.
    static let generationPasses = 1024
    /// Logit columns per residue, by return_logits track name.
    static let logitWidths = ["sequence": 64, "structure": 4096, "secondary_structure": 11, "sasa": 19, "function": 8 * 260, "residue_annotations": 1478]

    static func reply(_ numbers: Int, _ what: String) throws {
        guard numbers > replyNumbers else { return }
        throw TokenError("\(what) would return \(numbers) numbers, over the reply limit of \(replyNumbers) (16 MiB as JSON); ask for a shorter protein or less output.")
    }

    static func logits(_ tracks: [String], residues: Int) throws {
        var numbers = 0
        for name in tracks {
            guard let width = logitWidths[name] else { throw TokenError("return_logits names an unknown track \(name).") }
            numbers += width * residues
        }
        try reply(numbers, "return_logits for \(residues) residues")
    }

    /// `planned` holds each track's passes: num_steps, capped at that track's masked positions.
    static func passes(_ planned: [Int]) throws {
        let total = planned.reduce(0, +)
        guard total > generationPasses else { return }
        throw TokenError("generate runs at most \(generationPasses) model passes per request (num_steps per track, up to its masked positions, summed over tracks); this one needs \(total). Use fewer steps or tracks.")
    }
}

/// score_variants' notation: A23G (wild-type letter, 1-indexed position, mutant letter), several joined with ":".
enum Variants {
    typealias Mutation = (position: Int, wild: Int, mutant: Int)

    /// The mutations of one variant against `sequence` (token ids with <cls> and <eos>). An empty variant or an empty
    /// piece ("", ":", "A23G:") is refused, never scored as 0.
    static func parse(_ text: String, sequence: [Int]) throws -> [Mutation] {
        let length = sequence.count - 2
        let format = "Variant \"\(text)\" is not like A23G (wild-type letter, 1-indexed position, mutant letter; join several with :)."
        return try text.split(separator: ":", omittingEmptySubsequences: false).map { part -> Mutation in
            let chars = Array(part.trimmingCharacters(in: .whitespaces))
            guard chars.count >= 3, let position = Int(String(chars[1 ..< (chars.count - 1)])),
                let wild = ESM3Vocab.sequenceIndex[String(chars[0])], let mutant = ESM3Vocab.sequenceIndex[String(chars[chars.count - 1])],
                (4 ... 23).contains(wild), (4 ... 23).contains(mutant)
            else { throw TokenError(format) }
            guard position >= 1, position <= length else { throw TokenError("Variant \"\(text)\": position \(position) is outside 1-\(length).") }
            guard sequence[position] == wild else {
                throw TokenError("Variant \"\(text)\": the sequence has \(ESM3Vocab.sequence[sequence[position]]) at \(position), not \(chars[0]).")
            }
            return (position, wild, mutant)
        }
    }
}
