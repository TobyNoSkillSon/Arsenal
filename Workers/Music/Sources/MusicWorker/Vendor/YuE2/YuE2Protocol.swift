//
//  YuE2Protocol.swift
//  YuE2Music
//
//  Created by David Sherlock on 2026.
//
//  YuE2's checkpoint-native prompt protocol (yue2/protocol.py, yue2_infer
//  0.1.5): token ids, the three planning instructions, the positive and
//  negative prefixes, and how a long song is cut into acoustic chunks.
//

import Foundation

enum YuE2Protocol {
    static let eod = 151_643
    static let abcStart = 151_847, abcEnd = 151_848
    static let musicStart = 151_851, musicEnd = 151_852
    static let codecOffset = 151_853, codecSize = 32_768
    static let vocabularySize = 184_704
    static let context = 24_576
    // Latent frames (and semantic tokens) per second: 48 kHz / 1920.
    static let framesPerSecond = 25.0

    static func instruction(_ cot: YuE2Planning) -> String {
        switch cot {
        case .off: "Generate music with codec tokens from the given conditions."
        case .melody: "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
        case .full: "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
        }
    }

    static func text(style: String, lyrics: String, cot: YuE2Planning) -> String {
        "\(instruction(cot))\n[Tags]\n\(style)\n[Lyrics]\n\(lyrics)\n"
    }

    // The positive prefix: before the score (to plan it), or around a known score.
    static func prefix(style: String, lyrics: String, cot: YuE2Planning, abcIDs: [Int]?, tokenizer: Tiktoken) -> [Int] {
        let base = [eod] + tokenizer.encode(text(style: style, lyrics: lyrics, cot: cot))
        if cot == .off { return base + [abcStart, abcEnd, musicStart] }
        guard let abcIDs else { return base + [abcStart] }
        return base + [abcStart] + abcIDs + [abcEnd, musicStart]
    }

    // The guidance's negative prefix: the instruction alone, with the same score.
    static func negative(cot: YuE2Planning, abcIDs: [Int], tokenizer: Tiktoken) -> [Int] {
        let base = [eod] + tokenizer.encode(instruction(cot))
        if cot == .off { return base + [musicStart] }
        return base + [abcStart] + abcIDs + [abcEnd, musicStart]
    }

    // Frame ranges, each fitting the context beside its prefix.
    static func chunks(frames: Int, prefixTokens: Int) -> [Range<Int>] {
        let size = min((context - prefixTokens - 3) / 2, context)
        guard frames > 0, size > 0 else { return [] }
        return stride(from: 0, to: frames, by: size).map { $0..<min($0 + size, frames) }
    }
}
