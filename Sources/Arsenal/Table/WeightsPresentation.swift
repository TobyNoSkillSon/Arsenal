// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// What a row shows of its weights on this Mac: the Precision column's lines, one per precision on disk (16-bit, then
// 8-bit, then 4-bit), each with its size and a trash that deletes that copy, and the model's own Delete (every copy and
// cache). A precision that is not on disk has no line. A delete the store would refuse shows as a disabled trash with its
// reason as the tooltip. No AppKit: unit-tested directly.

/// One precision on disk: "16-bit", its size and a trash.
struct WeightCopy: Equatable {
    let dtype: String
    /// The bit width the copy stores: "16-bit", "4-bit" (the dtype's own name when no width is known).
    let label: String
    /// "846 MB": the precision's own size, as the catalog and Arsenal's MCP report it (`ModelManifest.sizesByDtype`), not
    /// the store's byte count, which counts a file shared by hardlink once per precision.
    let size: String
    /// The reason its trash is off, nil when one click deletes the copy. While off, the reason is the trash's tooltip.
    let blocked: String?
    /// This copy's delete is running.
    let removing: Bool
    /// The trash's tooltip: the reason it is off, else what the click does.
    var help: String { blocked ?? "Delete the \(label) copy from this Mac" }
    var enabled: Bool { blocked == nil && !removing }
}

/// A row's weights: the copies on disk, and whether the model's own Delete can run now.
struct WeightsPresentation: Equatable {
    /// Most precise first.
    var copies: [WeightCopy]
    /// Every byte on disk for the model (copies and partial downloads): what its Delete frees.
    var bytes: Int64
    /// The model's Delete: nil when it can run, else the reason (the trash's tooltip).
    var deleteBlocked: String?
    /// The model's whole delete is running.
    var removing: Bool

    /// Weights the row holds: copies on disk, or bytes in caches.
    var hasWeights: Bool { bytes > 0 || !copies.isEmpty }

    static let empty = WeightsPresentation(copies: [], bytes: 0, deleteBlocked: nil, removing: false)

    /// A row's weights from the store's on-disk listing and the table's jobs in flight: `working` (Get, Load, Unload by
    /// model id) and `removing` (deletes: a model's id, a copy's `copyKey`). `sizes`: each precision's size from the
    /// catalog (`CatalogEntry.sizesByDtype`); a precision without one shows the store's own count.
    static func make(
        _ row: ModelRow, disk: OnDiskModel?, sizes: [String: Int64] = [:], working: Set<String>, removing: Set<String>
    ) -> WeightsPresentation {
        let deletingWhole = removing.contains(row.id)
        let copies = (disk?.precisions ?? [])
            .sorted { a, b in
                let x = PrecisionColumn.bits(a.dtype, formats: row.formats) ?? 0, y = PrecisionColumn.bits(b.dtype, formats: row.formats) ?? 0
                return x != y ? x > y : a.dtype < b.dtype
            }
            .map { copy in
                let busy = deletingWhole || removing.contains(copyKey(row.id, copy.dtype))
                return WeightCopy(
                    dtype: copy.dtype, label: bitLabel(copy.dtype, formats: row.formats), size: formatBytes(size(copy, sizes)),
                    blocked: copyBlock(
                        row, copy.dtype, inUse: copy.inUse, dependents: copy.dependents.map { bitLabel($0, formats: row.formats) },
                        working: working, removing: busy),
                    removing: busy)
            }
        let all = disk?.bytes ?? 0
        return WeightsPresentation(
            copies: copies, bytes: all,
            deleteBlocked: wholeBlock(row, disk: disk, working: working, removing: removing), removing: deletingWhole || copies.contains(where: \.removing))
    }

    /// The reason one copy's delete is refused now, nil when it can run. `dependents` are the labels of the copies built from it.
    static func copyBlock(_ row: ModelRow, _ dtype: String, inUse: Bool, dependents: [String] = [], working: Set<String>, removing: Bool) -> String? {
        if removing { return "Deleting\u{2026}" }
        if inUse || row.states[dtype]?.state == .loaded { return unloadFirst }
        if !dependents.isEmpty {
            return "Delete the \(dependents.joined(separator: " and ")) \(dependents.count == 1 ? "copy" : "copies") first: it is built from this one."
        }
        if let busy = busyReason(row) { return busy }
        if working.contains(row.id) { return currentAction }
        return nil
    }
    /// The reason the model's whole delete is refused now, nil when it can run.
    static func wholeBlock(_ row: ModelRow, disk: OnDiskModel?, working: Set<String>, removing: Set<String>) -> String? {
        if removing.contains(row.id) || (disk?.precisions ?? []).contains(where: { removing.contains(copyKey(row.id, $0.dtype)) }) { return "Deleting\u{2026}" }
        if (disk?.precisions ?? []).contains(where: \.inUse) || row.residentDType != nil { return unloadFirst }
        if let busy = busyReason(row) { return busy }
        if working.contains(row.id) { return currentAction }
        return nil
    }

    /// A copy's size as the row shows it: the catalog's figure for that precision when it has one, else the store's count.
    static func size(_ copy: OnDiskPrecision, _ sizes: [String: Int64]) -> Int64 {
        if let listed = sizes[copy.dtype], listed > 0 { return listed }
        return copy.bytes
    }

    static let unloadFirst = "Unload to delete"
    static let currentAction = "Wait for the current action to finish"

    /// Why a download, conversion or load of this model holds its files, in the words the table uses; nil when none runs.
    static func busyReason(_ row: ModelRow) -> String? {
        let states = row.states.values.map(\.state)
        if states.contains(.downloading) { return "Wait for the download to finish" }
        if states.contains(.deriving) { return "Wait for the conversion to finish" }
        if states.contains(.loading) { return "Wait for the load to finish" }
        return nil
    }

    /// The bit width a precision stores, as "16-bit" or "4-bit"; its own name when the width is unknown.
    static func bitLabel(_ dtype: String, formats: [ModelFormat]) -> String {
        guard let bits = PrecisionColumn.bits(dtype, formats: formats) else { return dtype }
        return "\(Int(bits))-bit"
    }

    /// Key of one copy's delete in `removing`.
    static func copyKey(_ id: String, _ dtype: String) -> String { id + "/" + dtype }
    /// Key of the unlisted weights' delete in `removing`.
    static let orphanKey = "orphans"
}

/// One line of a row's Precision column ("16-bit above 16-bit, 4-bit above 4-bit, and we just show the
/// megabytes and then delete. If it's there; if it's not there, we don't even show it."): a precision on disk, its size,
/// whether it is the chosen one (gold) or the one that runs (the white edge on the loaded row), and its trash.
struct PrecisionLine: Equatable {
    let copy: WeightCopy
    /// The precision column it chooses ("16", "4"); nil for a format the model no longer offers (a leftover int8, a
    /// withdrawn tier), whose line keeps its size and trash but cannot be chosen.
    let column: String?
    /// The chosen precision: gold.
    let selected: Bool
    /// The precision that runs now, on a loaded row: the white edge.
    let loaded: Bool
    /// The model offers this precision and it can run here: a choice, whether or not the row is busy now.
    let offered: Bool
    /// A click on the line chooses it now (offered and the row not locked).
    let choosable: Bool
    /// int4's quality warning: the orange triangle after the label; its words lead the line's hover.
    let warned: Bool
    /// The line's hover (the part left of the trash).
    let help: String

    var dtype: String { copy.dtype }

    /// The lines are a choice only when two or more of them are offered: then each offered line is a pill, the chosen one
    /// gold and the running one edged in white. A row with one offered precision has nothing to choose, so its lines are
    /// plain text (twenty gold pills down a column said nothing).
    static func showsChoice(_ lines: [PrecisionLine]) -> Bool { lines.count { $0.offered } >= 2 }

    /// Lines in the column's fixed order: 16-bit, then 8-bit, then 4-bit (`WeightsPresentation.copies`), only for the
    /// precisions on disk.
    static func make(_ row: ModelRow, presentation p: RowPresentation, weights: WeightsPresentation) -> [PrecisionLine] {
        let off = ModelHelp.unavailable(row), warnings = ModelHelp.warnings(row)
        let selected = PrecisionColumn.of(p.selected, formats: row.formats)
        return weights.copies.map { copy in
            // The column this copy chooses: its bit width's, when the row's dtype for that column is this copy.
            let column = PrecisionColumn.of(copy.dtype, formats: row.formats).flatMap { PrecisionColumn.dtype(row, $0) == copy.dtype ? $0 : nil }
            guard let column else {
                return PrecisionLine(
                    copy: copy, column: nil, selected: false, loaded: p.hot && p.loadedPrecision == copy.dtype, offered: false, choosable: false,
                    warned: false,
                    help: "\(copy.dtype) \u{00b7} \(copy.size). \(retiredLine)")
            }
            let help = off[column] ?? ModelHelp.precisionTooltip(row, column: column, presentation: p)
            return PrecisionLine(
                copy: copy, column: column, selected: selected == column, loaded: p.hot && p.loadedPrecision == copy.dtype,
                offered: off[column] == nil, choosable: off[column] == nil && !p.locked, warned: off[column] == nil && warnings[column] != nil, help: help)
        }
    }
    /// The hover of a leftover copy the model no longer offers (an int8, Only 16-bit and int4; or a tier the
    /// catalog withdrew).
    static let retiredLine = "No longer offered for this model; its files stay until you delete them."
}

/// The group of rows that no catalog entry lists: weights on disk only. Its rows carry this kind (see `unlistedRow`).
enum Unlisted {
    static let kind = "unlisted"
    static let title = "Not in catalog"

    /// The row for a model the catalog does not list: its name is its id, its copies are its precisions on disk, and it
    /// is never offered again (the catalog-removed presentation: Remove, its weights deleted with it).
    static func row(_ disk: OnDiskModel) -> ModelRow {
        var states: [String: PrecisionState] = [:]
        for copy in disk.precisions { states[copy.dtype] = PrecisionState(state: copy.inUse ? .loaded : .ready, phase: "catalogRemoved") }
        let first = disk.precisions.first?.dtype ?? ""
        return ModelRow(
            id: disk.id, name: disk.id, kind: kind, precisions: disk.precisions.map(\.dtype), states: states, selectedPrecision: first,
            fast: false, runner: "", verification: "No longer in the catalog")
    }

    /// The table's rows: the service's rows that are on the table (or have weights on disk, which the table lists too), then
    /// the catalog's own row for every catalog model with weights on disk that is not on the table, then one unlisted row per
    /// model with weights on disk that neither knows. A catalog model is never shown as "Not in catalog" (25 Oct: with an
    /// empty table every installed model was listed there, with a Remove that deleted its weights).
    static func rows(_ all: [ModelRow], disk: [String: OnDiskModel], catalog: [ModelRow] = []) -> [ModelRow] {
        let listed = all.filter { $0.onTable || disk[$0.id] != nil }
        var known = Set(all.map(\.id))
        let installed = catalog.filter { !known.contains($0.id) && disk[$0.id] != nil }
        known.formUnion(catalog.map(\.id))
        let unknown = disk.values
            .filter { !known.contains($0.id) && ($0.bytes > 0 || !$0.precisions.isEmpty) }
            .sorted { $0.id < $1.id }
            .map(row)
        return listed + installed + unknown
    }
}
