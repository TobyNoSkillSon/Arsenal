// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ArsenalWire

/// Element-level post-processing from PaddleX `paddlex/inference/pipelines/paddleocr_vl`
/// (pipeline.py and uilts.py at c852371c701d83cd21955b2963834825920ffebb).
enum PaddleOutput {
    /// uilts.py truncate_repetitive_content on Unicode scalars (Python string indexing): a long single line ending in a
    /// repeated phrase keeps its prefix, a line that is one unit repeated keeps the unit, and a block that is mostly one
    /// repeated line keeps that line. The pipeline uses min_count 5000 for tables and 50 otherwise.
    static func truncateRepetitive(_ content: String, minCount: Int, lineThreshold: Int = 10, charThreshold: Int = 10, minLength: Int = 10) -> String {
        guard content.unicodeScalars.count >= minCount else { return content }
        let stripped = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stripped.isEmpty else { return content }
        let scalars = Array(stripped.unicodeScalars)
        let singleLine = !scalars.contains("\n")
        if singleLine, scalars.count > 100, let suffix = repeatingSuffix(scalars, minLength: 8, minRepeats: 5),
            Double(suffix.unit * suffix.count) > Double(scalars.count) * 0.5
        {
            return string(suffix.prefix)
        }
        if singleLine, scalars.count > minLength, let unit = shortestRepeatingUnit(scalars), scalars.count / unit.count >= charThreshold {
            return string(unit)
        }
        let lines = content.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        guard lines.count >= lineThreshold else { return content }
        var counts: [String: Int] = [:], order: [String] = []
        for line in lines {
            if counts[line] == nil { order.append(line) }
            counts[line, default: 0] += 1
        }
        // Counter.most_common(1): the highest count, first seen on ties.
        var common = order[0]
        for line in order where counts[line]! > counts[common]! { common = line }
        let count = counts[common]!
        return count >= lineThreshold && Double(count) / Double(lines.count) >= 0.8 ? common : content
    }
    /// find_repeating_suffix: the longest unit (from len/min_repeats down to min_len) the text ends with min_repeats times.
    private static func repeatingSuffix(_ s: [Unicode.Scalar], minLength: Int, minRepeats: Int) -> (prefix: ArraySlice<Unicode.Scalar>, unit: Int, count: Int)? {
        let n = s.count
        guard n / minRepeats >= minLength else { return nil }
        for size in stride(from: n / minRepeats, through: minLength, by: -1) {
            let unit = s[(n - size)...]
            guard (1..<minRepeats).allSatisfy({ s[(n - ($0 + 1) * size)..<(n - $0 * size)].elementsEqual(unit) }) else { continue }
            var count = 0, end = n
            while end - size >= 0, s[(end - size)..<end].elementsEqual(unit) { end -= size; count += 1 }
            return (s[..<(n - count * size)], size, count)
        }
        return nil
    }
    /// find_shortest_repeating_substring.
    private static func shortestRepeatingUnit(_ s: [Unicode.Scalar]) -> ArraySlice<Unicode.Scalar>? {
        let n = s.count
        guard n >= 2 else { return nil }
        for size in 1...(n / 2) where n % size == 0 {
            if stride(from: size, to: n, by: size).allSatisfy({ s[$0..<($0 + size)].elementsEqual(s[0..<size]) }) { return s[0..<size] }
        }
        return nil
    }
    private static func string<S: Sequence>(_ scalars: S) -> String where S.Element == Unicode.Scalar {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }
    /// pipeline.py: `\( \)` and `\[ \]` delimiters become Markdown `$` and `$$`.
    static func latex(_ text: String) -> String {
        guard (text.contains("\\(") && text.contains("\\)")) || (text.contains("\\[") && text.contains("\\]")) else { return text }
        return text.replacingOccurrences(of: "$", with: "")
            .replacingOccurrences(of: "\\(", with: " $ ").replacingOccurrences(of: "\\)", with: " $")
            .replacingOccurrences(of: "\\[\\[", with: "\\[").replacingOccurrences(of: "\\]\\]", with: "\\]")
            .replacingOccurrences(of: "\\[", with: " $$ ").replacingOccurrences(of: "\\]", with: " $$ ")
    }
    /// uilts.py post_process_for_spotting: each text is followed by four (x, y) points as `<|LOC_n|>` tokens on a
    /// 0–1000 grid. Decoding skips the special LOC_BEGIN/LOC_END markers, so the reference's grouping fallback applies.
    static func spotting(_ text: String, size: CGSize) throws -> (markdown: String, blocks: [JSONValue]) {
        let source = text as NSString
        let tokens = try NSRegularExpression(pattern: #"<\|LOC_(\d+)\|>"#).matches(in: text, range: NSRange(location: 0, length: source.length))
        let markers = try NSRegularExpression(pattern: #"<\|[A-Z_]+\|>"#)
        var blocks: [JSONValue] = [], texts: [String] = [], last = 0, index = 0
        while index + 7 < tokens.count {
            let group = Array(tokens[index..<(index + 8)])
            let values = group.map { Double(source.substring(with: $0.range(at: 1))) ?? -1 }
            guard values.allSatisfy({ $0 >= 0 && $0 <= 1000 }) else { throw invalid() }
            let span = source.substring(with: NSRange(location: last, length: group[0].range.location - last))
            let label = markers.stringByReplacingMatches(in: span, range: NSRange(location: 0, length: (span as NSString).length), withTemplate: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let points = stride(from: 0, to: 8, by: 2).map { CGPoint(x: CGFloat(values[$0]) * size.width / 1000, y: CGFloat(values[$0 + 1]) * size.height / 1000) }
            let xs = points.map(\.x), ys = points.map(\.y)
            blocks.append([
                "category": "text", "text": .string(label), "coordinate_system": "pixels",
                "box": ["xmin": .number(xs.min()!), "ymin": .number(ys.min()!), "xmax": .number(xs.max()!), "ymax": .number(ys.max()!)],
                "polygon": .array(points.map { ["x": .number($0.x), "y": .number($0.y)] })
            ])
            texts.append(label)
            last = NSMaxRange(group[7].range); index += 8
        }
        guard !blocks.isEmpty || text.isEmpty else { throw invalid() }
        return (texts.joined(separator: "\n\n"), blocks)
    }
    private static func invalid() -> WorkerFailure {
        WorkerFailure(code: "invalid_model_output", message: "PaddleOCR spotting did not return text with four 0–1000 location points.")
    }
}
