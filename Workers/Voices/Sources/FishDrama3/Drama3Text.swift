// SPDX-License-Identifier: MIT
// Text rules of the Drama 3 prompt and splitting long text into chunks. Reference: mlx-audio dad2087f
// `fish_drama3/chunking.py`. Works on Unicode scalars, as Python strings do, so cut points and byte counts match.
//
// Drama 3 is generated chunk by chunk: about `chunkBytes` of UTF-8 text with at most `chunkSpeakers` speakers each.
// Cuts prefer sentence ends, then clause marks, then spaces. Special tokens, paired tags such as <whisper>…</whisper>
// and [bracketed] cues are never cut, and every piece of a speaker's turn keeps that speaker's tag.
import Foundation

public enum Drama3Text {
    public static let chunkBytes = 300
    public static let chunkSpeakers = 3
    /// The tokenizer has <|speaker:0|> to <|speaker:15|>.
    public static let maxSpeakerID = 15

    /// Written only by the prompt builder; caller text containing them would inject conversation structure or audio rows.
    static let structuralTokens = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"]
    /// Unicode White_Space (Python's str.strip() would also strip U+001C…U+001F).
    static let whiteSpace: Set<UInt32> = Set(Array(0x9..<0xE) + [0x20, 0x85, 0xA0, 0x1680] + Array(0x2000..<0x200B) + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000])
    static let sentenceEnds: Set<Character> = Set(".!?;。！？；…\n")
    static let clauseMarks: Set<Character> = Set(",:，：、")
    static let spaces: Set<Character> = [" "]

    // Literal patterns, covered by the tests: compiling them cannot fail at run time.
    // swiftlint:disable:next force_try
    static let speakerTag = try! NSRegularExpression(pattern: #"<\|speaker:([0-9]+)\|>"#)
    // swiftlint:disable:next force_try
    static let uncuttable = try! NSRegularExpression(
        pattern: #"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]"#, options: [.dotMatchesLineSeparators])

    // MARK: - Unicode-scalar string helpers

    static func isWhite(_ s: Unicode.Scalar) -> Bool { whiteSpace.contains(s.value) }
    static func strip(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        guard let first = scalars.firstIndex(where: { !isWhite($0) }), let last = scalars.lastIndex(where: { !isWhite($0) }) else { return "" }
        return String(String.UnicodeScalarView(scalars[first...last]))
    }
    static func lstrip(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        guard let first = scalars.firstIndex(where: { !isWhite($0) }) else { return "" }
        return String(String.UnicodeScalarView(scalars[first...]))
    }
    static func bytes(_ text: String) -> Int { text.utf8.count }
    static func nsRange(_ text: String) -> NSRange { NSRange(text.startIndex..<text.endIndex, in: text) }
    /// The match of `speakerTag` anchored at the start of `text`.
    static func speakerTagPrefix(_ text: String) -> NSTextCheckingResult? {
        speakerTag.firstMatch(in: text, options: [.anchored], range: nsRange(text))
    }

    // MARK: - Validation and tags

    /// Rejects empty text, structural tokens and speaker tags outside 0…15.
    public static func validate(_ text: String, location: String) throws {
        guard !strip(text).isEmpty else { throw Drama3Error("\(location): text is empty") }
        for token in structuralTokens where text.contains(token) { throw Drama3Error("\(location): text must not contain \(token)") }
        let ns = text as NSString
        var search = NSRange(location: 0, length: ns.length)
        while true {
            let found = ns.range(of: "<|speaker", options: [], range: search)
            if found.location == NSNotFound { break }
            let tag = speakerTag.firstMatch(in: text, options: [.anchored], range: NSRange(location: found.location, length: ns.length - found.location))
            let number = tag.map { ns.substring(with: $0.range(at: 1)) } ?? ""
            guard !number.isEmpty, let value = Int(number), String(value) == number, value <= maxSpeakerID else {
                throw Drama3Error("\(location): speaker tags run from <|speaker:0|> to <|speaker:\(maxSpeakerID)|>")
            }
            search = NSRange(location: found.location + 1, length: ns.length - found.location - 1)
        }
    }

    /// Text that starts (after white space) with a speaker tag; speaker 0 by default.
    public static func withSpeakerTag(_ text: String) -> String {
        speakerTagPrefix(lstrip(text)) != nil ? text : "<|speaker:0|> " + text
    }

    /// A reference transcript whose tag directly follows "Text: " in the prompt.
    public static func referenceTranscript(_ text: String) -> String {
        let tagged = withSpeakerTag(text)
        let body = lstrip(tagged)
        let leading = String(String.UnicodeScalarView(Array(tagged.unicodeScalars).prefix(tagged.unicodeScalars.count - body.unicodeScalars.count)))
        guard let tag = speakerTagPrefix(body) else { return tagged }
        let ns = body as NSString
        return ns.substring(with: tag.range) + leading + ns.substring(from: tag.range.location + tag.range.length)
    }

    // MARK: - Chunking

    static func isDigit(_ c: Character) -> Bool {
        c.unicodeScalars.count == 1 && [.decimal, .digit].contains(c.unicodeScalars.first!.properties.numericType)
    }

    /// Cut after every mark, except a period between two digits (3.14). Iterates Unicode scalars like Python.
    static func cutAfter(_ text: String, marks: Set<Character>) -> [String] {
        let chars = text.unicodeScalars.map { Character($0) }
        var parts: [String] = [], start = 0
        for (i, char) in chars.enumerated() {
            let decimal = char == "." && i > 0 && i < chars.count - 1 && isDigit(chars[i - 1]) && isDigit(chars[i + 1])
            if marks.contains(char), !decimal {
                parts.append(join(chars[start...i])); start = i + 1
            }
        }
        if start < chars.count { parts.append(join(chars[start...])) }
        return parts
    }
    static func join(_ chars: ArraySlice<Character>) -> String {
        var s = String.UnicodeScalarView()
        for c in chars { s.append(contentsOf: c.unicodeScalars) }
        return String(s)
    }

    /// Pieces of at most `limit` bytes, cut at the first kind of mark that helps.
    static func fit(_ text: String, limit: Int, marks: ArraySlice<Set<Character>>) -> [String] {
        if bytes(text) <= limit { return [text] }
        if let first = marks.first {
            return cutAfter(text, marks: first).flatMap { fit($0, limit: limit, marks: marks.dropFirst()) }
        }
        var pieces: [String] = [], current = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            var candidate = current; candidate.append(scalar)
            if !current.isEmpty, String(candidate).utf8.count > limit {
                pieces.append(String(current)); current = String.UnicodeScalarView()
            }
            current.append(scalar)
        }
        return pieces + [String(current)]
    }

    static func freePieces(_ text: String, limit: Int) -> [String] {
        cutAfter(text, marks: sentenceEnds).flatMap { fit($0, limit: limit, marks: [clauseMarks, spaces][...]) }
    }

    static func pieces(_ text: String, limit: Int) -> [String] {
        let ns = text as NSString
        var pieces: [String] = [], last = 0
        for match in uncuttable.matches(in: text, range: nsRange(text)) {
            pieces += freePieces(ns.substring(with: NSRange(location: last, length: match.range.location - last)), limit: limit)
            pieces.append(ns.substring(with: match.range))
            last = match.range.location + match.range.length
        }
        return pieces + freePieces(ns.substring(from: last), limit: limit)
    }

    /// Pieces packed into chunks of about `limit` bytes. An uncuttable span longer than `limit` becomes a chunk of its own.
    public static func splitByLength(_ text: String, limit: Int = chunkBytes) -> [String] {
        if strip(text).isEmpty { return [] }
        if bytes(text) <= limit { return [text] }
        var chunks: [String] = [], current = ""
        for piece in pieces(text, limit: limit) {
            if !current.isEmpty, bytes(current + piece) > limit {
                chunks.append(current); current = ""
            }
            current += piece
        }
        chunks.append(current)
        return chunks.map(strip).filter { !$0.isEmpty }
    }

    /// Generation chunks in order. Tagged turns are split per speaker, each piece re-tagged, then regrouped so a chunk stays
    /// within `limit` and `maxSpeakers`.
    public static func split(_ text: String, limit: Int = chunkBytes, maxSpeakers: Int = chunkSpeakers) -> [String] {
        let ns = text as NSString
        let tags = speakerTag.matches(in: text, range: nsRange(text))
        if tags.isEmpty { return splitByLength(text, limit: limit) }
        // re.split with a capture group: [before, id, body, id, body, …].
        var turns: [(String, String)] = [("0", ns.substring(to: tags[0].range.location))]
        for (index, tag) in tags.enumerated() {
            let start = tag.range.location + tag.range.length
            let end = index + 1 < tags.count ? tags[index + 1].range.location : ns.length
            turns.append((ns.substring(with: tag.range(at: 1)), ns.substring(with: NSRange(location: start, length: end - start))))
        }
        let tagged = turns.flatMap { speaker, body in splitByLength(body, limit: limit).map { (speaker, "<|speaker:\(speaker)|> " + $0) } }
        var chunks: [[String]] = [], speakers: Set<String> = [], size = 0
        for (speaker, piece) in tagged {
            let newSpeaker = !speakers.contains(speaker)
            if !chunks.isEmpty, (newSpeaker && speakers.count >= maxSpeakers) || size + bytes(piece) > limit {
                chunks.append([]); speakers = []; size = 0
            }
            if chunks.isEmpty { chunks.append([]) }
            chunks[chunks.count - 1].append(piece)
            speakers.insert(speaker); size += bytes(piece)
        }
        return chunks.map { $0.joined(separator: "\n") }
    }
}
