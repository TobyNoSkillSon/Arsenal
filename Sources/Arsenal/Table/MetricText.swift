// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// One figure cell: the value, an optional small second line (an image's resolution) and whether it is drawn lighter
/// (an estimate from another chip, marked `~`, or a quick check). Missing figures are `—`.
struct MetricText: Equatable {
    var value: String
    var detail: String?
    var estimate: Bool

    static let missing = "\u{2014}"
    /// Columns that come from the model's manifest, not from a benchmark.
    static let facts: Set<String> = ["context"]
    /// The Peak RAM column every kind ends with.
    static let peakRAM = KindColumn(id: "peak_ram", title: "Peak RAM")

    /// The kind's columns as the table shows them: the kind's own, then Peak RAM (once).
    static func columns(_ kind: KindSpec) -> [KindColumn] {
        kind.columns.contains { ["peak_ram", "peak_ram_mb"].contains($0.id) } ? kind.columns : kind.columns + [peakRAM]
    }

    /// A figure from the row's metrics: a string is shown as given (the results file formats it, e.g.
    /// "1.0 s · 512×512", whose part after " · " becomes the small line); a number is formatted by the column's unit.
    init(_ row: ModelRow, _ column: KindColumn) {
        let quick = row.isQuickCheck(column.id) || (column.id == Self.peakRAM.id && row.isQuickCheck("peak_ram_mb"))
        estimate = row.badge == .otherChip || quick
        detail = nil
        var raw: JSONValue? = row.metrics[column.id]
        if raw == nil, column.id == Self.peakRAM.id { raw = row.metrics["peak_ram_mb"] ?? row.peakRAMMB.map { .number($0) } }
        var text: String?
        switch raw {
        case .string(let s) where !s.isEmpty && s != Self.missing:
            let parts = s.components(separatedBy: " \u{00b7} ")
            text = parts[0]
            if parts.count > 1 { detail = parts.dropFirst().joined(separator: " \u{00b7} ") }
        case .number(let v) where v.isFinite:
            // `unit` is optional in some KindColumn versions; bind it once.
            let unit: String? = column.unit
            text = column.id == "peak_ram_mb" || (column.id == Self.peakRAM.id && unit != "GB") ? Self.megabytes(v) : Self.format(v, unit: unit, id: column.id)
        default: text = nil
        }
        if detail == nil, let resolution = row.metrics["resolution"]?.string, text != nil { detail = resolution }
        // Unverified figures are not shown (nothing measured them); another chip's are marked as estimates.
        // Unverified figures are not shown (nothing measured them); a manifest fact (a chat model's context) is not a
        // measurement and always shows.
        if row.badge == .unverified && !quick && !Self.facts.contains(column.id) { text = nil; detail = nil }
        let marked = row.badge == .otherChip
        value = text.map { marked ? "~" + $0 : $0 } ?? Self.missing
        if text == nil { estimate = false }
    }

    /// Units the bundled kinds declare whose column shows the bare number.
    static let unitless: Set<String> = ["\u{00d7} real time"]

    static func megabytes(_ mb: Double) -> String { mb >= 1000 ? MemoryDisplay.text(decimalGB: mb / 1000) : String(format: "%.0f MB", mb) }

    static func format(_ v: Double, unit: String?, id: String) -> String {
        switch unit {
        case "×": return String(format: v >= 10 ? "%.0f×" : "%.1f×", v)
        case "%": return String(format: "%.1f%%", v <= 1 ? v * 100 : v)
        case "ms": return String(format: "%.0f ms", v)
        case "GB": return MemoryDisplay.text(decimalGB: v)
        case "tokens/s": return String(format: v >= 100 ? "%.0f" : "%.1f", v)
        case "tokens": return v >= 1024 ? "\(Int(v / 1024))K" : String(format: "%.0f", v)
        case "s": return v < 1 ? String(format: "%.0f ms", v * 1000) : String(format: v < 10 ? "%.1f s" : "%.0f s", v)
        default:
            if id == "context" { return v >= 1024 ? "\(Int(v / 1024))K" : String(format: "%.0f", v) }
            let number = String(format: v >= 100 ? "%.0f" : "%.1f", v)
            // A kind's own unit (a local kind's "px", "fps") follows the number; the bundled speed column keeps its
            // familiar bare figure.
            guard let unit, !unit.isEmpty, !unitless.contains(unit), unit.count <= 12 else { return number }
            return number + " " + unit
        }
    }
}
