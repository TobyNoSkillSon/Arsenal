// SPDX-License-Identifier: MIT
import Foundation

/// Stops a generation that is looping, so a degenerate decode ends with what it produced and `finish_reason`
/// `repetition` instead of spending the whole `max_tokens` budget.
///
/// Two rules, checked on each new token of the output (the prompt is not counted):
/// - Periodic tail, after Chandra's `detect_repeat_token` (datalab-to/chandra `chandra/model/util.py`, which the maker
///   uses to reject looping pages): the output ends in a block of `p` ≤ 128 tokens repeated more than
///   ⌊4·(1 + 3/p)⌋ times. Chandra measures characters over a 500-character window after generation; here the block and
///   its copies are tokens, and the repeated tail must also cover at least `span` tokens, so a legitimate run (a dot
///   leader, a row of empty table cells, a separator line) is never cut while it is short.
/// - No progress, after the stock Qwen3-ASR loop's backstop (last 24 tokens, at most 3 distinct): the last `span`
///   tokens use at most 3 distinct tokens.
///
/// `span` is per operation (`span(operation:)`), sized so that no real output of that task reaches it.
struct RepetitionGuard {
    static let maxPeriod = 128
    let span: Int
    private(set) var count = 0
    private var tokens: [Int] = []
    /// `run[p]`: how many trailing tokens equal the token `p` positions before them.
    private var run: [Int]
    private var window: [Int: Int] = [:]

    init(span: Int) {
        precondition(span >= 8)
        self.span = span
        run = Array(repeating: 0, count: Self.maxPeriod + 1)
    }

    /// The repeated span per operation, in tokens. Grounding answers are about ten tokens; translations and chat-style
    /// answers are prose; documents and extractions legitimately repeat structure (empty cells, rows, list items).
    static func span(operation: String) -> Int {
        switch operation {
        case "ground_screen": return 48
        case "translate", "describe", "tag": return 256
        default: return 512
        }
    }

    /// The smallest number of copies Chandra's rule needs for a block of `period` tokens (`repeat_count > max_repeats`).
    static func copies(period: Int) -> Int { Int(4 * (1 + 3 / Double(period))) + 1 }

    /// Appends one generated token; true when the output now loops.
    mutating func append(_ token: Int) -> Bool {
        tokens.append(token)
        count += 1
        var looping = false
        for p in stride(from: min(Self.maxPeriod, count - 1), through: 1, by: -1) {
            run[p] = tokens[count - 1 - p] == token ? run[p] + 1 : 0
            let tail = run[p] + p
            if tail >= span, tail >= p * Self.copies(period: p) { looping = true }
        }
        window[token, default: 0] += 1
        if count > span {
            let old = tokens[count - 1 - span]
            if window[old] == 1 { window[old] = nil } else { window[old]! -= 1 }
        }
        if count >= span, window.count <= 3 { looping = true }
        return looping
    }
}
