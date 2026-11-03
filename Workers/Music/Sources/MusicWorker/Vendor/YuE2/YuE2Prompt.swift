//
//  YuE2Prompt.swift
//  YuE2Music
//
//  Created by David Sherlock on 2026.
//
//  A ``SongRequest`` in YuE2's terms: the caption becomes the `[Tags]`
//  style line (with the tempo and key named, as YuE2's own examples write
//  them), the lyrics keep their section tags in YuE2's capitalised form, an
//  instrumental becomes sections without words, and a duration caps the
//  25 Hz song tokens.
//

import Foundation

enum YuE2Prompt {

    // The `[Tags]` line.
    static func style(_ r: SongRequest) -> String {
        var parts = [r.caption.trimmingCharacters(in: .whitespacesAndNewlines)].filter { !$0.isEmpty }
        let lower = r.caption.lowercased()
        if let bpm = r.bpm, !lower.contains("bpm") { parts.append("\(bpm) BPM") }
        if let key = r.key, !key.isEmpty, !lower.contains(key.lowercased()) { parts.append(key) }
        if r.instrumental, !lower.contains("instrumental") { parts.append("instrumental") }
        return parts.joined(separator: ", ")
    }

    // The `[Lyrics]` block: tags capitalised (`[verse]` → `[Verse]`); an instrumental is wordless sections.
    static func lyrics(_ r: SongRequest) -> String {
        if r.instrumental || r.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Verse]\n\n[Chorus]\n\n[Outro]"
        }
        return capitaliseTags(r.lyrics.replacingOccurrences(of: "\r\n", with: "\n"))
    }

    // `[pre-chorus]` → `[Pre-Chorus]`; anything not a bracketed tag is untouched.
    static func capitaliseTags(_ text: String) -> String {
        text.split(separator: "\n", omittingEmptySubsequences: false).map { line -> String in
            let t = line.trimmingCharacters(in: .whitespaces)
            guard t.hasPrefix("["), t.hasSuffix("]"), t.count > 2 else { return String(line) }
            let inner = t.dropFirst().dropLast()
            let words = inner.split(separator: " ", omittingEmptySubsequences: false).map { word in
                word.split(separator: "-", omittingEmptySubsequences: false)
                    .map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined(separator: "-")
            }
            return "[" + words.joined(separator: " ") + "]"
        }.joined(separator: "\n")
    }

    // The song tokens' sampling; a duration caps them too (25 a second), whichever cap is lower wins.
    static func semanticSettings(_ o: YuE2Options, duration: Double?) -> YuE2SamplingSettings {
        var s = YuE2SamplingSettings(
            temperature: Float(o.semanticTemperature), topP: Float(o.semanticTopP), topK: o.semanticTopK,
            repetitionPenalty: Float(o.semanticRepetitionPenalty), penaltyWindow: o.semanticPenaltyWindow,
            minTokens: o.semanticMinTokens, maxTokens: o.semanticMaxTokens)
        if let duration {
            s.maxTokens = min(s.maxTokens, max(1, Int((duration * YuE2Protocol.framesPerSecond).rounded())))
            s.minTokens = min(s.minTokens, s.maxTokens)
        }
        return s
    }

    static func abcSettings(_ o: YuE2Options) -> YuE2SamplingSettings {
        YuE2SamplingSettings(
            temperature: Float(o.abcTemperature), topP: Float(o.abcTopP), topK: o.abcTopK,
            repetitionPenalty: Float(o.abcRepetitionPenalty), penaltyWindow: o.abcPenaltyWindow,
            minTokens: o.abcMinTokens, maxTokens: o.abcMaxTokens)
    }

    // Upstream's `Sampling.__post_init__` and request checks.
    static func validate(_ o: YuE2Options, _ s: [YuE2SamplingSettings]) throws(YuE2Error) {
        for x in s {
            guard (0...5).contains(x.temperature), x.topP > 0, x.topP <= 1, x.topK >= 1 else {
                throw .invalidRequest("YuE2 sampling: temperature 0–5, top-p in (0, 1], top-k ≥ 1")
            }
            guard x.repetitionPenalty > 0, (1...100).contains(x.penaltyWindow) else {
                throw .invalidRequest("YuE2 sampling: repetition penalty > 0, penalty window 1–100")
            }
            guard x.minTokens >= 0, x.minTokens <= x.maxTokens, x.maxTokens >= 1 else {
                throw .invalidRequest("YuE2 sampling: 0 ≤ min tokens ≤ max tokens")
            }
        }
        if let score = o.score {
            guard o.planning != .off, !score.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw .invalidRequest("a YuE2 score needs planning full or melody, and some ABC")
            }
        }
        if let g = o.guidance, !(0...20).contains(g) { throw .invalidRequest("YuE2 guidance must be 0–20") }
        guard o.odeSteps >= 1 else { throw .invalidRequest("YuE2 ODE steps must be at least 1") }
        guard o.decoderTileFrames >= 16 else { throw .invalidRequest("YuE2 decoder tiles must be at least 16 frames") }
    }
}
