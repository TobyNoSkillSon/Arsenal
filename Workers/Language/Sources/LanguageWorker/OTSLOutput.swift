// SPDX-License-Identifier: MIT
import Foundation

/// OTSL grid reconstruction, including rectangular row/column spans.
enum OTSLOutput {
    struct Cell { let text: String; let row: Int; let column: Int; let rows: Int; let columns: Int }
    /// TeleOCR card convert_otsl_to_html: an existing `<table>` passes through; rows are padded to the widest row;
    /// text without cells converts to an empty string.
    static func html(_ text: String) throws -> String {
        if text.hasPrefix("<table"), text.hasSuffix("</table>") { return text }
        return try grid(text) ?? ""
    }
    /// PaddleX convert_otsl_to_html: otsl_pad_to_sqr_v2 first; the pipeline keeps the model text when no cells result.
    static func paddle(_ text: String) throws -> String {
        let html = try grid(padToSquare(text)) ?? ""
        return html.isEmpty ? text : html
    }
    /// PaddleX otsl_pad_to_sqr_v2: each row's cells (a tag and its text; text before a row's first tag is dropped)
    /// are padded with `<ecel>` or trimmed to the width that minimizes the total length deviation, never below the
    /// last `<fcel>` of any row.
    static func padToSquare(_ text: String) -> String {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.contains("<nl>") else { return value + "<nl>" }
        let tags = "(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)"
        guard let regex = try? NSRegularExpression(pattern: tags + ".*?(?=" + tags + "|$)", options: [.dotMatchesLineSeparators]) else { return value }
        var rows: [(cells: [String], minimum: Int)] = []
        for line in value.components(separatedBy: "<nl>") where !line.isEmpty {
            let source = line as NSString
            let cells = regex.matches(in: line, range: NSRange(location: 0, length: source.length)).map { source.substring(with: $0.range) }
            guard !cells.isEmpty else { continue }
            rows.append((cells, (cells.lastIndex { $0.hasPrefix("<fcel>") }).map { $0 + 1 } ?? 0))
        }
        guard let low = rows.map(\.minimum).max(), let longest = rows.map(\.cells.count).max() else { return "<nl>" }
        var width = max(low, longest), cost = Int.max
        for candidate in low...max(low, longest) {
            let total = rows.reduce(0) { $0 + abs($1.cells.count - candidate) }
            if total < cost { cost = total; width = candidate }
        }
        return rows.map { row in
            (row.cells.count > width ? Array(row.cells.prefix(width)) : row.cells + Array(repeating: "<ecel>", count: width - row.cells.count)).joined()
        }.joined(separator: "<nl>") + "<nl>"
    }
    /// Grid HTML, or nil when the text holds no cells.
    private static func grid(_ text: String) throws -> String? {
        let pattern = #"<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>"#
        let regex = try NSRegularExpression(pattern: pattern)
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        guard !matches.isEmpty else { return nil }
        let string = text as NSString
        var tokens: [(String, String)] = []
        for (index, match) in matches.enumerated() {
            let end = NSMaxRange(match.range), next = index + 1 < matches.count ? matches[index + 1].range.location : string.length
            tokens.append((string.substring(with: match.range), string.substring(with: NSRange(location: end, length: next - end)).trimmingCharacters(in: .whitespacesAndNewlines)))
        }
        var grid: [[(String, String)]] = [[]]
        for token in tokens {
            if token.0 == "<nl>" { if !grid.last!.isEmpty { grid.append([]) } } else { grid[grid.count - 1].append(token) }
        }
        grid = grid.filter { !$0.isEmpty }
        guard let width = grid.map(\.count).max() else { return nil }
        guard grid.count <= 1024, width <= 1024 else { return text }
        for row in grid.indices { grid[row] += Array(repeating: ("<ecel>", ""), count: width - grid[row].count) }
        var cells: [Cell] = []
        for row in grid.indices {
            for column in 0..<width where ["<fcel>", "<ecel>"].contains(grid[row][column].0) {
                var columns = 1, rows = 1
                while column + columns < width, ["<lcel>", "<xcel>"].contains(grid[row][column + columns].0) { columns += 1 }
                while row + rows < grid.count, ["<ucel>", "<xcel>"].contains(grid[row + rows][column].0) { rows += 1 }
                cells.append(Cell(text: grid[row][column].1, row: row, column: column, rows: rows, columns: columns))
            }
        }
        var result = "<table>"
        for row in grid.indices {
            result += "<tr>"
            for cell in cells where cell.row == row {
                let attrs = (cell.rows > 1 ? " rowspan=\"\(cell.rows)\"" : "") + (cell.columns > 1 ? " colspan=\"\(cell.columns)\"" : "")
                let escaped = cell.text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
                    .replacingOccurrences(of: "\"", with: "&quot;").replacingOccurrences(of: "'", with: "&#x27;")
                result += "<td\(attrs)>\(escaped)</td>"
            }
            result += "</tr>"
        }
        return result + "</table>"
    }
}
