// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// The table's tooltips as plain strings (AppKit tooltips show them; the render harness writes them out for review).
enum ModelHelp {
    /// The Model cell: licence with its commercial-use hint, what it does, size and how it runs; the name only when the
    /// cell truncates it.
    static func text(_ row: ModelRow, presentation p: RowPresentation, name: Bool = false) -> String {
        var lines = name ? [row.name] : []
        let origin = [row.publisher.isEmpty ? nil : row.publisher, row.released.map(String.init)].compactMap { $0 }.joined(separator: ", ")
        let licence = row.licence.isEmpty ? nil : licenceLine(LicenceName.short(row.licence, name: nil), commercial: row.commercial)
        let first = [origin.isEmpty ? nil : origin, licence].compactMap { $0 }.joined(separator: " \u{00b7} ")
        if !first.isEmpty { lines.append(first) }
        if !row.summary.isEmpty { lines.append(row.summary) }
        var facts: [String] = []
        if !row.params.isEmpty { facts.append("\(ParamsText.display(row.params, name: row.name)) parameters") }
        if let native = row.precisions.first(where: { PrecisionColumn.of($0, formats: row.formats) == "16" }) { facts.append("native \(native)") }
        if let context = row.metrics["native_context"]?.number ?? row.metrics["context"]?.number {
            facts.append("max " + MetricText.format(context, unit: "tokens", id: "context") + " tokens")
        }
        if !facts.isEmpty { lines.append(facts.joined(separator: " \u{00b7} ")) }
        lines.append(runnerTooltip(row))
        if let note = row.retentionNote, !note.isEmpty { lines.append(note) }
        if let leftover = withdrawn(row) { lines.append(leftover) }
        if let warning = row.settingsWarning, !warning.isEmpty { lines.append(warning) }
        return lines.joined(separator: "\n")
    }

    /// Precisions still on this Mac that the catalog no longer offers (`ModelRow.withdrawnTiers`), in plain words: the
    /// Model hover's note, and the settings card's Remove hover.
    static func withdrawn(_ row: ModelRow) -> String? {
        guard !row.withdrawnTiers.isEmpty else { return nil }
        // Widths come from the format itself: an 8-bit copy has no Precision column, but it still names its width.
        func bits(_ dtype: String) -> Double { PrecisionColumn.bits(dtype, formats: row.formats) ?? 0 }
        let tiers = row.withdrawnTiers.sorted { bits($0.key) != bits($1.key) ? bits($0.key) > bits($1.key) : $0.key < $1.key }
        let described = tiers.map { (name: WeightsPresentation.bitLabel($0.key, formats: row.formats), size: formatBytes($0.value)) }
        let offered = PrecisionColumn.all.filter { PrecisionColumn.dtype(row, $0) != nil }.map { "\($0)-bit" }
        let runs = "this model now runs at \(offered.isEmpty ? "no precision" : offered.joined(separator: " and ")) only"
        let leftover: String
        if described.count == 1 {
            leftover = "An old \(described[0].name) copy (\(described[0].size)) is still on this Mac"
        } else {
            leftover = "Old copies are still on this Mac: " + described.map { "\($0.name) (\($0.size))" }.joined(separator: ", ")
        }
        return "\(leftover) \u{2014} \(runs)."
    }
    /// The settings card's Delete: what it deletes and what stays.
    static func withdrawnHelp(_ row: ModelRow) -> String {
        (withdrawn(row) ?? "") + "\nDeletes the old copy from this Mac. The precisions the model offers stay."
    }

    /// The settings card's Revoke: whose approval goes, and what happens next.
    static func revokeHelp(_ row: ModelRow) -> String {
        let digest = row.approval.map { " of contents \($0.digest.prefix(8))" } ?? ""
        return "Forgets your approval\(digest) and unloads \(row.name). Get or Load asks again before its code runs."
    }
    /// The Local badge (table status line and browser Status cell): unsigned, and who approved which contents when.
    static func local(_ row: ModelRow) -> String {
        var lines = ["A folder in your local models folder, not reviewed or signed by Arsenal."]
        if let a = row.approval {
            let at = (try? Date(a.at, strategy: .iso8601)).map { $0.formatted(date: .abbreviated, time: .shortened) } ?? a.at
            let line = "Approved by \(a.by.rawValue) \u{00b7} \(at) \u{00b7} contents \(a.digest.prefix(8))"
            lines.append(row.needsCodeConsent ? line + "; the folder has changed since, so Get or Load asks again." : line)
        } else if row.needsCodeConsent {
            lines.append("Its code is not approved yet: Get or Load asks first.")
        }
        return lines.joined(separator: "\n")
    }

    /// A model's checks on request: each claim separately (protocol shape, parity with its reference, the task), what was
    /// compared and how many cases, which implementation they tested and where the evidence is.
    static func checks(_ c: ModelChecksStatus) -> String {
        func word(_ s: CheckStatus) -> String { s == .pass ? "passed" : s == .fail ? "failed" : "not run" }
        var lines = ["Protocol \(word(c.protocolStatus)) \u{00b7} Parity \(word(c.parity)) \u{00b7} Task \(word(c.task)) \u{00b7} Speed \(word(c.speed))"]
        if c.stale { lines.append("Stale: the implementation or fixtures changed after these checks, so they no longer describe what runs.") }
        if let reference = c.reference, !reference.isEmpty {
            let name = reference["name"]?.string ?? reference["id"]?.string ?? reference["identity"]?.string
            if let name { lines.append("Reference: \(name)") }
        }
        if !c.cases.isEmpty { lines.append("Cases: " + c.cases.keys.sorted().map { "\($0) \(c.cases[$0]!)" }.joined(separator: ", ")) }
        if !c.implementationFingerprint.isEmpty { lines.append("Tested implementation \(c.implementationFingerprint.prefix(12))") }
        if !c.evidence.isEmpty { lines.append("Evidence: " + c.evidence.prefix(4).joined(separator: ", ")) }
        lines.append("Quick check, not a controlled measurement.")
        return lines.joined(separator: "\n")
    }

    /// The licence with its commercial-use hint, from the manifest's `licence.commercial`.
    /// The table's row does not carry the licence's conditions, so "not plainly commercial" reads as limited (none
    /// or conditional); the browser passes `conditions` and says which.
    static func licenceLine(_ licence: String, commercial: Bool, conditions: String?? = nil) -> String {
        let use: String
        switch (commercial, conditions) {
        case (true, _): use = "commercial use allowed"
        case (false, .none): use = "commercial use limited, see licence"
        case (false, .some(.some)): use = "commercial use limited"
        case (false, .some(.none)): use = "non-commercial use only"
        }
        return "\(licence), " + use
    }

    /// The table's runner line: `Runner: Native (Swift · MLX)`, named as the settings name it; a chosen runner
    /// names the default it replaces.
    static func runnerTooltip(_ row: ModelRow) -> String {
        if row.runner == "Unavailable" { return "No runner for this Mac" }
        var line = "Runner: " + TableController.runnerTitle(row.runner, language: row.runnerLanguage)
        if let fallback = row.defaultRunner, fallback != row.runner { line += ", chosen; default \(fallback)" }
        return line
    }

    static func runnerLine(_ runner: String, language: String) -> String {
        if runner == "Unavailable" { return "No runner for this Mac" }
        return "Runs in Arsenal's native worker (Swift, MLX)"
    }

    /// The precision columns the model does not offer, with their reason (greyed cells). Every reason reads "<State>.
    /// <Sentence>." so the lines match wherever they come from.
    static func unavailable(_ row: ModelRow) -> [String: String] {
        var off: [String: String] = [:]
        for column in PrecisionColumn.all {
            guard let dtype = PrecisionColumn.dtype(row, column) else {
                off[column] = column == "16" ? "Not offered. No 16-bit weights are listed for this model." : notOffered(row, column: column)
                continue
            }
            // int4 that failed its check stays offered (with its warning); anything else unavailable is greyed.
            if let current = TableStats.set(row, precision: dtype)?.applyingInt4Policy.availability, current.state == .unavailable {
                off[column] = "Unavailable. " + sentence(current.reason ?? "Its quality check did not pass")
                continue
            }
            // A format no runner here can run: greyed with the service's reason.
            if let reason = row.dtypeUnavailableReason(dtype) { off[column] = "Unavailable. " + sentence(reason) }
        }
        return off
    }
    /// Why a catalog model has no 8- or 4-bit cell: under a billion parameters the smaller versions would save less than
    /// 1 GB; a larger model ships at 16-bit unless the owner asked for a quantized tier, or lost one to its quality check, and
    /// the row cannot tell which. A local folder just declares none.
    static func notOffered(_ row: ModelRow, column: String) -> String {
        if row.local { return "Not offered. This folder declares no \(column)-bit format." }
        guard let params = ParamsText.value(row.params) else { return "Not offered. This model has no \(column)-bit format." }
        if params < 1e9 { return "Not offered. Under 1B parameters, \(column)-bit would save less than 1 GB, so this model runs at 16-bit only." }
        return "Not offered. This model runs at 16-bit only."
    }
    /// A reason as a sentence: capital first letter, one full stop.
    static func sentence(_ text: String) -> String {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = t.first else { return "" }
        let body = first.uppercased() + t.dropFirst()
        return body.hasSuffix(".") ? body : body + "."
    }

    /// A precision's quality check: passed, did not pass, or nil when it has not run. int4 follows the service's warning
    /// (`precisionQualityWarnings`), so the status hover and the int4 cell always say the same.
    static func quality(_ row: ModelRow, _ dtype: String) -> Bool? {
        if let effective = row.precisionDtypes[dtype], effective != dtype, ["fp16", "bf16"].contains(effective) { return quality(row, effective) }
        if dtype == "int4", row.precisions.contains("int4") {
            guard let warning = row.precisionQualityWarnings["int4"] else { return true }
            return warning == PrecisionStatSet.int4QualityWarning ? false : nil
        }
        guard let set = row.precisionStats[dtype], set.runner == row.runner, set.dtype == dtype, set.kind.rawValue == row.kind else { return nil }
        return set.applyingInt4Policy.qualityCheckPassed
    }
    /// One precision's quality check as a sentence.
    static func qualityLine(_ row: ModelRow, _ dtype: String) -> String {
        switch quality(row, dtype) {
        case true?: return "\(dtype) passed its quality check."
        case false?: return "\(dtype) did not pass its quality check."
        case nil: return "\(dtype) has no quality check yet."
        }
    }
    /// The quality check of each offered 8- and 4-bit precision (a greyed one says why in its own hover).
    static func quantizedQualityLines(_ row: ModelRow) -> [String] {
        let off = unavailable(row)
        return PrecisionColumn.all.filter { $0 != "16" && off[$0] == nil }.compactMap { PrecisionColumn.dtype(row, $0) }.map { qualityLine(row, $0) }
    }

    /// The precision cells with a quality warning, by column: int4 where its quality check did not pass or has not run
    /// (the service's `precisionQualityWarnings`). The cell draws the triangle; the text is the first line of its hover.
    static func warnings(_ row: ModelRow) -> [String: String] {
        var out: [String: String] = [:]
        for (dtype, text) in row.precisionQualityWarnings where !text.isEmpty {
            if let column = PrecisionColumn.all.first(where: { PrecisionColumn.dtype(row, $0) == dtype }) { out[column] = text }
        }
        return out
    }

    /// A precision cell's whole hover as the control sets it: the reason a cell cannot be chosen, else its warning, what
    /// it is, and what is busy while the row is locked.
    static func precisionTooltip(_ row: ModelRow, column: String, presentation p: RowPresentation) -> String {
        unavailable(row)[column]
            ?? TierControl.cellTooltip(
                warning: warnings(row)[column], help: precision(row, column: column, presentation: p), enabled: !p.locked, locked: lockLine(row))
    }

    /// Why a busy row's precision cells are locked, in plain words: which precision is downloading, being made or
    /// loading, and that the choice opens again when it finishes.
    static func lockLine(_ row: ModelRow) -> String {
        if row.catalogRemoved { return "Locked: this model is no longer in the catalog." }
        guard let busy = row.precisions.first(where: { [.downloading, .deriving, .loading].contains(row.states[$0]?.state) }) else {
            return TierControl.inUseHelp
        }
        let doing: String
        switch row.states[busy]?.state {
        case .downloading: doing = row.states[busy]?.phase == RowPresentation.verifyingLocalCopy ? "the copy of \(busy) on this Mac is checked" : "\(busy) downloads"
        case .deriving: doing = "\(busy) is made on this Mac"
        default: doing = "\(busy) loads"
        }
        return "Locked while \(doing); you can choose again when it finishes."
    }

    /// Where a dtype's weights come from, in one phrase (a folder's format declaration says, else the bit width does).
    static func origin(_ dtype: String, native: String, format: ModelFormat?) -> String {
        switch format?.source {
        case .vendor where dtype != native: return "\(dtype), a smaller version from the publisher"
        case .runnerNative: return "\(dtype), made from \(native) as it loads"
        case .derived: return "\(dtype), a smaller version made on this Mac from \(native)"
        default:
            let published = PrecisionColumn.of(dtype, formats: format.map { [$0] } ?? []) == "16"
            return published ? "\(dtype), as published" : "\(dtype), a smaller version made on this Mac from \(native)"
        }
    }

    /// A precision Arsenal removed to save space (source retention: the 16-bit copy, once a smaller version made from it
    /// was chosen and loaded): not on this Mac, still offered, and Get brings it back.
    static func removed(_ row: ModelRow, _ dtype: String) -> Bool {
        let s = row.states[dtype]
        return (s?.state ?? .notInstalled) == .notInstalled && s?.phase == PrecisionState.sourceRemovedPhase
    }
    /// The precision columns whose weights were removed to save space (the cell's download arrow).
    static func removed(_ row: ModelRow) -> Set<String> {
        Set(PrecisionColumn.all.filter { column in PrecisionColumn.dtype(row, column).map { removed(row, $0) } ?? false })
    }
    /// The 16-bit copy a smaller precision is made from was removed to save space, and this one is not here yet.
    static func sourceRemoved(_ row: ModelRow, _ dtype: String) -> Bool {
        guard let base = TableFigures.base(row), base != dtype, removed(row, base) else { return false }
        return ![.ready, .loaded, .loading].contains(row.states[dtype]?.state ?? .notInstalled)
    }
    static let removedLine = "Deleted from this Mac to save space after a smaller version was made"

    /// An offered precision cell: what it is and whether it is on this Mac.
    static func precision(_ row: ModelRow, column: String, presentation p: RowPresentation) -> String {
        guard let dtype = PrecisionColumn.dtype(row, column) else { return "" }
        let native = row.precisions.first { PrecisionColumn.of($0, formats: row.formats) == "16" } ?? "bf16"
        let what = row.precisionNotes[dtype] ?? Self.origin(dtype, native: native, format: row.formats.first { $0.id == dtype })
        let gone = removed(row, dtype)
        let state: String
        switch row.states[dtype]?.state ?? .notInstalled {
        case .loaded: state = "\nLoaded"
        case .ready: state = "\nOn this Mac"
        case .updateAvailable: state = "\nUpdate available"
        case .downloading: state = row.states[dtype]?.phase == RowPresentation.verifyingLocalCopy ? "\nChecking the copy on this Mac" : "\nDownloading"
        case .deriving: state = "\nBeing made now"
        case .loading: state = "\nLoading"
        case .failed: state = "\n" + (row.states[dtype]?.error ?? "Failed")
        // Not here yet: what Get does follows (its download, or the copy on this Mac it uses). Removed to save space
        // says so: it is neither "not offered" (greyed) nor never got.
        case .notInstalled: state = gone ? "\n" + removedLine : ""
        }
        // What a Get of this cell still fetches (the service's figure; a derived cell whose 16-bit source was removed
        // to save space downloads it again).
        if let bytes = row.downloadBytes[dtype], bytes > 0, ![.ready, .loaded].contains(row.states[dtype]?.state ?? .notInstalled) {
            if gone { return what + state + "\nGet downloads it again (\(formatBytes(bytes)))" }
            if sourceRemoved(row, dtype) { return what + state + "\nGet downloads the deleted 16-bit copy again (\(formatBytes(bytes)))" }
            return what + state + "\nGet downloads \(formatBytes(bytes))"
        }
        if row.downloadBytes[dtype] == 0, let local = row.localCopyBytes[dtype], local > 0, row.states[dtype]?.state ?? .notInstalled == .notInstalled {
            return what + state + "\nGet uses the copy on this Mac"
        }
        if gone { return what + state + "\nGet brings it back" }
        let pending = p.loadedPrecision.map { $0 != dtype && p.selected == dtype } ?? false
        return what + state + (pending ? "\n\(p.loadedPrecision ?? "") stays loaded until Reload" : "")
    }

    /// A pending row's Reload: what it applies, e.g. "Context 128K, int8".
    static func reloadHelp(_ changes: [String]) -> String {
        "Reload applies: " + changes.joined(separator: ", ") + ". Until then the loaded settings serve requests."
    }

    /// The action button's tooltip: what a click does that its title does not say (the precision, where the weights
    /// come from); while busy, what is running.
    /// `changes`: what a pending row's Reload applies; `loadChoices`: load-time settings away from their defaults.
    static func action(_ row: ModelRow, presentation p: RowPresentation, changes: [String] = [], loadChoices: [String] = []) -> String {
        if let gate = p.gate, p.busyText == nil { return gate.gaveUp ? "Tries the download again" : "Retries now" }
        if p.updateAvailable, p.action == .get, p.busyText == nil {
            let transfer = (p.downloadBytes ?? 0) > 0 ? " Downloads \(formatBytes(p.downloadBytes!))." : ""
            return "Updates \(RowPresentation.bits(p.selected, row)) before it can load." + transfer
        }
        if p.action == .reload, p.pending, p.busyText == nil, !changes.isEmpty { return reloadHelp(changes) }
        if p.busyText != nil {
            // The busy precision's state (another precision may be the one shown).
            let busy = row.precisions.first { [.downloading, .deriving, .loading].contains(row.states[$0]?.state) } ?? p.selected
            switch row.states[busy]?.state {
            case .downloading:
                return row.states[busy]?.phase == RowPresentation.verifyingLocalCopy ? "Checking the copy on this Mac" : "Downloading \(RowPresentation.bits(busy, row))"
            case .deriving: return "Making \(RowPresentation.bits(busy, row))"
            default: return "Loading \(RowPresentation.bits(busy, row))"
            }
        }
        let derived = PrecisionColumn.of(p.selected, formats: row.formats) != "16"
        let does: String
        switch p.action {
        case .get where p.getUsesLocalCopy:
            let local = formatBytes(p.localCopyBytes ?? 0)
            does =
                derived
                ? "Adds the 16-bit copy already on this Mac (\(local)), then makes \(RowPresentation.bits(p.selected, row)) from it. Nothing to download."
                : "Adds the copy already on this Mac (\(local)). Nothing to download."
        case .get where p.hot:
            let again = sourceRemoved(row, p.selected) ? " again (deleted to save space)" : ""
            does =
                (p.downloadBytes ?? 0 > 0
                    ? "Downloads the 16-bit weights\(again), then makes \(RowPresentation.bits(p.selected, row))."
                    : "Makes \(RowPresentation.bits(p.selected, row)) from the 16-bit weights on this Mac.")
                + " The loaded model keeps running until Reload."
        case .get where removed(row, p.selected):
            does = "Downloads \(RowPresentation.bits(p.selected, row)) again from Hugging Face; it was deleted from this Mac to save space."
        case .get where derived && sourceRemoved(row, p.selected):
            does = "Downloads the 16-bit weights again from Hugging Face (deleted to save space), then makes \(RowPresentation.bits(p.selected, row)) on this Mac."
        case .get:
            does =
                derived
                ? "Downloads the 16-bit weights from Hugging Face, then makes \(RowPresentation.bits(p.selected, row)) on this Mac."
                : "Downloads \(RowPresentation.bits(p.selected, row)) from Hugging Face."
        case .load where !loadChoices.isEmpty: does = "Loads \(RowPresentation.bits(p.selected, row)) \u{00b7} " + loadChoices.joined(separator: ", ")
        case .load: does = "Loads \(RowPresentation.bits(p.selected, row))"
        case .unload: does = "Frees its memory; the weights stay on this Mac."
        case .reload: does = "Reloads at \(RowPresentation.bits(p.selected, row))"
        case .remove:
            let weighed = row.states.values.contains { [.ready, .loaded].contains($0.state) }
            does = weighed ? "Deletes its weights from this Mac and removes it from the table." : "Removes it from the table."
        }
        // The last Get failed (its reason is the line under the name): the button runs the same Get again.
        if p.failed, p.action == .get { return "Tries again. " + does }
        return does
    }
}

/// Figure column headings and cells.
enum ColumnHelp {
    static let known: [String: String] = [
        "rtf": "Seconds of speech made per second. Higher is faster.",
        "first_audio": "Time from the request to the first playable audio. Lower is better.",
        "first_audio_s": "Time from the request to the first playable audio. Lower is better.",
        "roundtrip_wer": "Round-trip word error: the speech is transcribed back by a fixed recogniser and compared with the text. Lower is better.",
        "round_trip_wer": "Round-trip word error: the speech is transcribed back by a fixed recogniser and compared with the text. Lower is better.",
        "wer": "Round-trip word error: the speech is transcribed back by a fixed recogniser and compared with the text. Lower is better.",
        "accuracy": "Share of the fixed decision set answered correctly. Higher is better.",
        "latency": "Time for one decision. Lower is better.",
        "latency_ms": "Time for one decision. Lower is better.",
        "seconds_per_image": "Seconds to make one image at the stated resolution. Lower is better.",
        "seconds": "Seconds to make one image at the stated resolution. Lower is better.",
        "context": "Longest prompt plus answer the model accepts, in tokens.",
        "tokens_per_s": "Answer tokens generated per second. Higher is faster.",
        "tokens_per_second": "Answer tokens generated per second. Higher is faster.",
        "first_token": "Time from the request to the first answer token. Lower is better.",
        "ttft_s": "Time from the request to the first answer token. Lower is better.",
        "peak_ram": "Most memory the model used while running its benchmark. Lower is better."
    ]
    static func text(_ c: KindColumn) -> String { known[c.id] ?? c.title }

    /// A figure cell: where the number comes from.
    static func cell(_ row: ModelRow, _ c: KindColumn, chip: String?) -> String {
        let here = chip ?? "this Mac"
        if row.isQuickCheck(c.id) { return text(c) + "\n" + QuickCheck.help }
        if c.id == "context", let tokens = row.metrics["context"]?.number {
            // What Arsenal runs (the manifest's context), then the vendor's maximum when the service knows a larger one.
            let runs = "Arsenal runs up to \(MetricText.format(tokens, unit: "tokens", id: "context")) tokens"
            guard let native = row.metrics["native_context"]?.number, native > tokens else { return text(c) + "\n" + runs + "." }
            return text(c) + "\n" + runs + "; the model supports \(MetricText.format(native, unit: "tokens", id: "context"))."
        }
        if MetricText.facts.contains(c.id) { return text(c) + "\nFrom the model's manifest." }
        // Figures come only from the benchmark suite; a dash means it has not run here (an ad-hoc run never fills it).
        if row.metrics[c.id] == nil || row.metrics[c.id] == .null { return text(c) + "\nNot measured on \(here) yet." }
        switch row.badge {
        case .measured: return text(c) + "\nMeasured on \(here) with the kind's frozen benchmark suite."
        case .otherChip: return text(c) + "\nEstimate: measured on another chip, not on \(here)."
        case .unverified: return text(c) + "\nNot measured yet."
        }
    }
}
