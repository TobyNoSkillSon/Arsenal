// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// The figure columns every kind shows (the same columns for every kind): Speed and Peak memory. What each
/// cell reads and prints is TableFigures.
enum TableStats {
    static let columns = TableFigures.columns.map { KindColumn(id: $0.rawValue, title: $0.title) }
    static func columns(_ kind: KindSpec) -> [KindColumn] { columns }
    /// The row's stat set for a dtype, only when it belongs to the row's kind, that dtype and the chosen runner.
    static func set(_ row: ModelRow, precision: String) -> PrecisionStatSet? {
        guard let kind = ModelStatKind(rawValue: row.kind) else { return nil }
        // int4 that failed its quality check is offered with a warning (the int4 policy), never greyed as unavailable.
        return row.precisionStats[precision].flatMap { $0.kind == kind && $0.dtype == precision && $0.runner == row.runner ? offered($0) : nil }
    }
    /// The int4 policy for the table: an int4 an older result marked unavailable for quality is offered with its
    /// warning, and its figures, recorded as unavailable with it, read as not measured yet.
    static func offered(_ set: PrecisionStatSet) -> PrecisionStatSet {
        var result = set.applyingInt4Policy
        guard set.availability.state == .unavailable, result.availability.state != .unavailable else { return result }
        for (key, value) in result.metrics where value.state == .unavailable { result.metrics[key] = StatValue() }
        return result
    }
    static func headerHelp(_ column: String) -> String {
        FigureColumn(rawValue: column).map(TableFigures.headerHelp) ?? "Figures for the chosen precision."
    }
    /// A group's own column hover: the kind's unit.
    static func help(kind: KindSpec, column: String) -> String {
        let local = TableFigures.localSpeedColumn(kind.columns.map(\.id)).flatMap { id in kind.columns.first { $0.id == id } }
        return FigureColumn(rawValue: column).map { TableFigures.headerHelp($0, kind: ModelStatKind(rawValue: KindPresentation.parent(kind.id)), local: local) } ?? ""
    }
}
