// SPDX-License-Identifier: MIT
import Foundation

/// Incremental detokenization for streamed text. A multi-byte character split across tokens decodes to a trailing
/// U+FFFD until its last byte arrives; that tail is held back so each delta ends on a complete character. Decoding
/// restarts after each completed line, so a long answer does not re-decode its whole history at every token.
struct StreamDetokenizer {
    private var segmentStart = 0
    private var emitted = 0

    /// The newly completed text after `tokens` (all generated tokens so far).
    mutating func next(_ tokens: [Int], decode: ([Int]) -> String) -> String {
        guard segmentStart <= tokens.count else { return "" }
        let scalars = Array(decode(Array(tokens[segmentStart...])).unicodeScalars)
        var complete = scalars.count
        while complete > 0, scalars[complete - 1] == "\u{FFFD}" { complete -= 1 }
        var delta = String.UnicodeScalarView()
        if complete > emitted { delta.append(contentsOf: scalars[emitted..<complete]) }
        emitted = max(emitted, complete)
        if complete == scalars.count, scalars.last == "\n" { segmentStart = tokens.count; emitted = 0 }
        return String(delta)
    }
}
