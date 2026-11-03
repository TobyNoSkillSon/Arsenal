// SPDX-License-Identifier: MIT
// The reference-bracket-v1 prompt as frame rows. Reference: mlx-audio dad2087f `fish_drama3/drama3.py` `build_frames`.
// Host-side only (no MLX), so the layout is unit-testable without a GPU.
import Foundation

/// What the prompt builder needs from the text tokenizer.
public protocol Drama3Tokenizing {
    /// IDs of one segment, tokenized on its own with no added BOS/EOS (`tokenizers.Tokenizer.encode(segment).ids`).
    func encode(_ text: String) -> [Int]
    func id(of token: String) -> Int?
}

/// Codec codes of one clip: `rows[codebook][frame]`, `codebooks` rows of equal length.
public struct Drama3Codes: Equatable, Sendable {
    public var rows: [[Int32]]
    public init(rows: [[Int32]]) { self.rows = rows }
    public var frames: Int { rows.first?.count ?? 0 }

    func check(codebooks: Int, vocab: Int, location: String) throws {
        guard rows.count == codebooks, rows.allSatisfy({ $0.count == frames }) else {
            throw Drama3Error("\(location): codes must be [\(codebooks), frames]; got \(rows.count) rows")
        }
        guard frames > 0 else { throw Drama3Error("\(location): codes are empty") }
        guard rows.allSatisfy({ $0.allSatisfy { $0 >= 0 && Int($0) < vocab } }) else { throw Drama3Error("\(location): codes must be in [0, \(vocab))") }
    }
}

/// Frame rows [count, width] (row-major) and how many leading rows form the system block.
public struct Drama3Prompt: Equatable, Sendable {
    public var values: [Int32]
    public var width: Int
    public var systemRows: Int
    public var count: Int { values.count / width }
    public func row(_ index: Int) -> ArraySlice<Int32> { values[(index * width)..<((index + 1) * width)] }
}

public enum Drama3PromptBuilder {
    enum Segment {
        case text(String)
        case codes(Drama3Codes)
    }

    /// Text rows are [token, 0, …]; audio rows are [c0 + semanticBegin, c0…c9]. Every string is tokenized on its own:
    /// tokenizing the joined prompt changes the ids at the seams. The system block depends only on the references, so its
    /// cache state is reused across chunks and calls.
    public static func build(
        references: [(String, Drama3Codes)], history: [(String, Drama3Codes)], target: String, tokenizer: Drama3Tokenizing, codebooks: Int,
        vocab: Int, semanticBegin: Int
    ) throws -> Drama3Prompt {
        try Drama3Text.validate(target, location: "text")
        var segments: [Segment] = [.text("<|im_start|>system\n")]
        if references.isEmpty {
            segments.append(.text("convert the provided text to speech"))
        } else {
            segments.append(.text("convert the provided text to speech reference to the following:"))
            for (index, (transcript, codes)) in references.enumerated() {
                let location = "reference \(index + 1)"
                try Drama3Text.validate(transcript, location: location)
                try codes.check(codebooks: codebooks, vocab: vocab, location: location)
                segments += [.text("\n\n[Reference \(index + 1)]\nText: \(Drama3Text.referenceTranscript(transcript))\nSpeech: "), .codes(codes)]
            }
        }
        segments.append(.text("<|im_end|>\n"))
        let systemSegments = segments.count
        for (index, (text, codes)) in history.enumerated() {
            try Drama3Text.validate(text, location: "history \(index + 1)")
            try codes.check(codebooks: codebooks, vocab: vocab, location: "history \(index + 1)")
            segments += [
                .text("<|im_start|>user\n"), .text(Drama3Text.withSpeakerTag(text)), .text("<|im_end|>\n"), .text("<|im_start|>assistant\n<|voice|>"),
                .codes(codes), .text("<|im_end|>\n")
            ]
        }
        segments += [.text("<|im_start|>user\n"), .text(Drama3Text.withSpeakerTag(target)), .text("<|im_end|>\n"), .text("<|im_start|>assistant\n<|voice|>")]

        let width = 1 + codebooks
        var values: [Int32] = [], systemRows = 0
        for (index, segment) in segments.enumerated() {
            var rows = 0
            switch segment {
            case .text(let text):
                for id in tokenizer.encode(text) {
                    values.append(Int32(id)); values += repeatElement(0, count: codebooks); rows += 1
                }
            case .codes(let codes):
                for frame in 0..<codes.frames {
                    values.append(codes.rows[0][frame] + Int32(semanticBegin))
                    for book in 0..<codebooks { values.append(codes.rows[book][frame]) }
                    rows += 1
                }
            }
            if index < systemSegments { systemRows += rows }
        }
        return Drama3Prompt(values: values, width: width, systemRows: systemRows)
    }
}
