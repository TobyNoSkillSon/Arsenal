// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import Synchronization

// What one row shows, from the service's ModelRow and an optimistic preference while it is saved.
// No AppKit/SwiftUI: unit-tested directly.

/// The row's one-word action.
enum RowActionKind: Equatable {
    case get, load, unload, remove
    /// The row is loaded with something else than what is chosen (a dtype cell, the Fast/Exact switch, a load-time
    /// setting): load it in place. The row stays loaded (blue); the button is drawn prominent while this is pending.
    case reload
    var title: String {
        switch self {
        case .get: return "Get"
        case .load: return "Load"
        case .unload: return "Unload"
        case .reload: return "Reload"
        case .remove: return "Remove"
        }
    }
}

/// The line under the model's name.
struct StatusLine: Equatable {
    /// `plain`: what the model does (its purpose), in the secondary text colour.
    enum Tone: Equatable { case measured, estimate, quiet, busy, error, pending, plain }
    var text: String
    var tone: Tone
    var help: String
}

/// Precision columns: 16 and 4-or-fewer bits. Any format label maps to the column of its bit width, so a folder's own
/// formats (mxfp4, oq4, a ~3-bit EXL3-style one) sit in the grid without the app knowing them (v2.2 item 5). 8-bit
/// formats (int8) have no column: Only 16-bit and int4 are allowed (21 Oct).
enum PrecisionColumn {
    static let all = ["16", "4"]
    static func of(_ dtype: String, formats: [ModelFormat] = []) -> String? {
        // fp32 is only ever a source to cast from, never a choice.
        guard let bits = bits(dtype, formats: formats), bits <= 16, !isEightBit(dtype, formats: formats) else { return nil }
        return bits >= 12 ? "16" : "4"
    }
    /// A format of 6 to 11 bits: int8 and its kin, which the table does not offer for any model.
    static func isEightBit(_ dtype: String, formats: [ModelFormat] = []) -> Bool {
        guard let bits = bits(dtype, formats: formats) else { return false }
        return bits >= 6 && bits < 12
    }
    /// The precision a row shows and runs for a saved or requested one. An 8-bit one (a preference saved before int8 was
    /// dropped) falls back to the row's 16-bit format, so no Get, Load or Reload can name it.
    static func offered(_ dtype: String, _ row: ModelRow) -> String {
        guard isEightBit(dtype, formats: row.formats), let sixteen = self.dtype(row, "16") else { return dtype }
        return sixteen
    }
    /// Bits per weight of a dtype: the folder's declared `bits_per_weight` whenever the row or entry lists the format;
    /// without one, only a label that is a known family prefix followed by its width (int4, q3_k, mxfp4, fp8, bf16). Any
    /// other label (exl3-5bpw) has no width until its folder declares one: digits are never read out of a name.
    static func bits(_ dtype: String, formats: [ModelFormat] = []) -> Double? {
        if let declared = formats.first(where: { $0.id == dtype })?.bitsPerWeight, declared.isFinite, declared > 0 { return declared }
        let d = dtype.lowercased()
        switch d {
        case "bf16", "fp16", "f16", "float16", "bfloat16": return 16
        case "fp32", "f32", "float32": return 32
        default: break
        }
        if let known = labelBits.withLock({ $0[d] }) { return known }
        let parsed = labelWidth(d)
        labelBits.withLock { $0[d] = parsed }
        return parsed
    }
    /// A label's width, read once per label: the table asks for every row's formats many times per render, and a Swift
    /// Regex match costs tens of microseconds (80% of a whole-table render in a debug build, 6 Nov 2026: the menu lagged).
    private static let labelBits = Mutex<[String: Double?]>([:])
    #if DEBUG
        /// How many times each label was read with the Regex (tests: once).
        static let labelReads = Mutex<[String: Int]>([:])
    #endif
    private static func labelWidth(_ d: String) -> Double? {
        #if DEBUG
            labelReads.withLock { $0[d, default: 0] += 1 }
        #endif
        guard let match = d.wholeMatch(of: /(?:u?int|q|fp|mxfp|nvfp|bf|f)(\d+(?:\.\d+)?)(?:_[a-z0-9]+)*/), let n = Double(match.1), n > 0, n <= 32 else {
            return nil
        }
        return n
    }
    /// The dtype a row offers in a column, e.g. "bf16" for 16 when the model is bf16.
    /// Two formats of one width (int4 and mxfp4): the selected one, else the first listed.
    static func dtype(_ row: ModelRow, _ column: String) -> String? {
        of(row.selectedPrecision, formats: row.formats) == column && row.precisions.contains(row.selectedPrecision)
            ? row.selectedPrecision : row.precisions.first { of($0, formats: row.formats) == column }
    }
    /// The column label: the real format the model runs in that column (fp16 or bf16, int8, int4; a folder's own format by
    /// its name). A column the model does not offer names the usual format, greyed.
    static func label(_ row: ModelRow, _ column: String) -> String {
        let selected = dtype(row, column) ?? (column == "16" ? "fp16" : "int\(column)")
        return row.precisionDtypes[selected] ?? selected
    }
}

struct RowPresentation: Equatable {
    var selected: String
    var fast: Bool
    var loadedPrecision: String?
    var action: RowActionKind
    var updateAvailable: Bool
    /// The shown precision's last Get failed: the button reads Try Again (it runs Get again).
    var failed = false
    /// The shown precision's Get waits for access to a gated Hugging Face model: the button stays Get (it retries now)
    /// and, until Arsenal stops retrying, a countdown to the automatic retry sits beside it.
    var gate: GateWait?
    /// When the automatic retry runs (the countdown's end), or nil.
    var countdown: Date? { busyText == nil ? gate?.retryAt : nil }
    /// A model leaving the catalog whose weights are on this Mac: its action deletes them, so it reads Delete (Remove
    /// only takes a model off the list).
    var removeDeletes = false
    /// A download's percentage or "…" while deriving/loading: replaces the action's title; no click.
    var busyText: String?
    /// A download runs: the cancel glyph shows.
    var cancellable: Bool
    /// Fraction for the row's progress fill while downloading.
    var progress: Double?
    /// The row's state as before: a transient state (getting, making, loading, failed, update, next load) or, at rest,
    /// its verification (badge, quick check, checks, Local).
    var status: StatusLine
    /// The line under the name: a transient state while it applies, else what the model does; its hover keeps the verification.
    var line: StatusLine
    /// Precision cells and the switch take no click (busy).
    var locked: Bool
    /// Bytes a Get still fetches for the shown precision (nil: unknown; 0: nothing to download, it is made here).
    var downloadBytes: Int64?
    /// Bytes of the shown precision's files a copy already on this Mac covers (Get checks and imports them).
    var localCopyBytes: Int64?
    var hot: Bool { loadedPrecision != nil }
    /// Loaded, and something chosen differs from what is loaded (dtype, Fast/Exact, a load-time setting): Reload pending.
    var pending: Bool
    /// Get covered by a local copy: nothing to download.
    var getUsesLocalCopy: Bool { action == .get && (downloadBytes ?? 0) == 0 && (localCopyBytes ?? 0) > 0 }
    /// The button's title: Get carries the download size before anything starts ("Get · 4.5 GB"); a Get with nothing
    /// to download (a copy on this Mac covers it) is plain "Get", its tooltip saying so.
    var actionTitle: String {
        // A failed Get: Try Again, with the size when the slot has room for it (`RowAction.shownTitle` drops it otherwise).
        if failed, action == .get, busyText == nil {
            guard let bytes = downloadBytes, bytes > 0 else { return "Try Again" }
            return "Try Again \u{00b7} \(formatBytes(bytes))"
        }
        if action == .remove { return removeDeletes ? "Delete" : "Remove" }
        if updateAvailable, action == .get {
            let bytes = downloadBytes ?? 0
            return bytes > 0 ? "Update \u{00b7} \(formatBytes(bytes))" : "Update"
        }
        if getUsesLocalCopy { return "Get" }
        guard action == .get, let bytes = downloadBytes, bytes > 0 else { return action.title }
        return "Get \u{00b7} \(formatBytes(bytes))"
    }

    /// A precision as a person reads it: "16-bit", "8-bit", "4-bit" (the format's own name when its width is unknown).
    static func bits(_ dtype: String, _ row: ModelRow) -> String { WeightsPresentation.bitLabel(dtype, formats: row.formats) }

    /// Optional optimistic choices before the service publishes the saved row.
    init(_ row: ModelRow, previewPrecision: String? = nil, previewFast: Bool? = nil, chip: String? = nil) {
        let loaded = row.residentDType
        let selected = PrecisionColumn.offered(
            previewPrecision.flatMap { row.precisions.contains($0) ? $0 : nil }
                ?? (row.precisions.contains(row.selectedPrecision) ? row.selectedPrecision : row.precisions.first ?? row.selectedPrecision), row)
        // No Fast path on this chip: Exact, whatever was saved.
        let loadedFast = row.fastAvailable && row.fast
        let fast = row.fastAvailable ? previewFast ?? row.fast : false
        self.selected = selected; self.fast = fast; loadedPrecision = loaded
        let state = row.states[selected] ?? PrecisionState()
        updateAvailable = state.state == .updateAvailable
        // Any precision of the row being fetched, derived or loaded locks the row's choices until it settles.
        let busyPrecision = row.precisions.first { [.downloading, .deriving, .loading].contains(row.states[$0]?.state) }
        locked = busyPrecision != nil
        busyText = nil; cancellable = false; progress = nil
        // Against the loaded configuration, never against the previous saved choice.
        pending = loaded != nil && (selected != loaded || fast != loadedFast || row.needsReload)
        switch state.state {
        case .downloading:
            action = .get
            busyText = state.progress.map { "\(Int(($0 * 100).rounded(.down)))%" } ?? "…"
            cancellable = true; progress = state.progress
        case .deriving, .loading:
            action = state.state == .loading ? (pending ? .reload : .load) : .get
            busyText = "…"
        case .loaded:
            action = pending ? .reload : .unload
        case .ready:
            action = pending ? .reload : .load
        case .notInstalled, .updateAvailable, .failed:
            action = state.phase == "localFolderErrorLoaded" ? .unload : .get
            gate = state.phase == PrecisionState.gatedPhase && action == .get ? state.gate : nil
            failed = state.state == .failed && action == .get && gate == nil
        }
        // A precision other than the busy one is shown while another is busy: show the busy one's progress anyway.
        if let busyPrecision, busyPrecision != selected, let other = row.states[busyPrecision] {
            busyText = other.state == .downloading ? (other.progress.map { "\(Int(($0 * 100).rounded(.down)))%" } ?? "…") : "…"
            cancellable = other.state == .downloading; progress = other.state == .downloading ? other.progress : nil
        }
        status = Self.status(row, selected: busyPrecision ?? selected, chip: chip)
        if let loaded, loaded != selected, busyPrecision == nil, status.tone != .error {
            status = StatusLine(text: Self.bits(selected, row) + " on next load", tone: .pending, help: Self.bits(loaded, row) + " stays loaded until Reload.")
        }
        downloadBytes = Self.downloadBytes(row, selected)
        localCopyBytes = row.localCopyBytes[selected]
        if row.catalogRemoved {
            action = .remove; pending = false; busyText = nil; cancellable = false; progress = nil
            locked = true
            let weighed = row.states.values.contains { [.ready, .loaded].contains($0.state) }
            failed = false; gate = nil; removeDeletes = weighed
            status = StatusLine(
                text: row.verification, tone: .quiet,
                help: weighed ? "Deletes its weights from this Mac and removes it from the table." : "Remove from the table.")
        }
        line = Self.line(row, status: status, resting: Self.resting(row, chip: chip))
    }

    /// What Get still fetches for a precision: the service's figure (`downloadBytes`); without one, the 16-bit download
    /// (`sizeBytes`) unless it is already on this Mac (int8/int4 are made here from it).
    static func downloadBytes(_ row: ModelRow, _ precision: String) -> Int64? {
        if let bytes = row.downloadBytes[precision] { return bytes }
        let source = row.precisions.first { PrecisionColumn.of($0, formats: row.formats) == "16" } ?? precision
        let have: Set<ModelState> = [.ready, .loaded, .loading]
        if have.contains(row.states[precision]?.state ?? .notInstalled) || have.contains(row.states[source]?.state ?? .notInstalled) { return 0 }
        return row.sizeBytes
    }

    /// `PrecisionState.phase` while a Get hash-checks a copy already on this Mac instead of downloading.
    static let verifyingLocalCopy = "verifyingLocalCopy"

    /// Transient states (failure, download, derive, load) replace the badge; otherwise the badge.
    static func status(_ row: ModelRow, selected: String, chip: String?) -> StatusLine {
        let s = row.states[selected] ?? PrecisionState()
        switch s.state {
        case .updateAvailable:
            return StatusLine(text: "Update available", tone: .quiet, help: "Update before loading this precision.")
        case .failed where s.phase == PrecisionState.gatedPhase && s.gate != nil:
            return gateLine(s.gate!)
        case .failed:
            // The line says it in plain words under its lead ("Get failed: the connection stalled"), else its first
            // sentence, over up to two lines; the hover keeps the whole reason, as it came.
            let message = s.error ?? "Failed"
            return StatusLine(text: ErrorWording.reword(message)?.text ?? ErrorWording.lead(message), tone: .error, help: message)
        case .downloading:
            if s.phase == Self.verifyingLocalCopy {
                // The bytes go to the tooltip: with them the line no longer fits the name column; the button shows %.
                let bytes = s.bytes.flatMap { done in s.totalBytes.map { " (\(formatBytes(done)) of \(formatBytes($0)))" } } ?? ""
                return StatusLine(text: "Checking local copy", tone: .busy, help: "Checking the hashes of the copy on this Mac\(bytes); nothing is downloaded.")
            }
            var text = "Downloading"
            if let done = s.bytes, let total = s.totalBytes, total > 0 {
                // "6.6 of 16.0 GB": one unit when both share it.
                let a = formatBytes(done), b = formatBytes(total)
                let unit = b.split(separator: " ").last.map(String.init) ?? ""
                text += " \u{00b7} \(a.hasSuffix(" " + unit) ? String(a.dropLast(unit.count + 1)) : a) of \(b)"
            } else if let p = s.progress {
                text += " \u{00b7} \(Int(p * 100))%"
            }
            return StatusLine(text: text, tone: .busy, help: "From Hugging Face; Get resumes after a cancel.")
        case .deriving:
            return StatusLine(text: "Making \(bits(selected, row))\u{2026}", tone: .busy, help: "Made once, on this Mac, from the 16-bit weights.")
        case .loading:
            return StatusLine(text: "Loading \(bits(selected, row))\u{2026}", tone: .busy, help: "")
        default:
            return resting(row, chip: chip)
        }
    }

    /// A Get waiting for a gated model: what to do, in a few words; the hover has the rest.
    static func gateLine(_ g: GateWait) -> StatusLine {
        let accept = "Accept the model\u{2019}s terms on its Hugging Face page (\(g.page)), signed in to the account the token belongs to."
        let rejected = g.tokenRejected == true ? "Hugging Face refused the saved token (expired or revoked). " : ""
        let token =
            rejected + "Hugging Face shares this model only with a signed-in account. Create a read token on the page Arsenal opened, add it in "
            + "Settings \u{203a} Hugging Face Token\u{2026}, then accept the model\u{2019}s terms on its page (\(g.page))."
        let manual = g.mode == "manual" ? " Its authors approve each request, which can take days." : ""
        let retry = " Arsenal retries every 30 s, \(g.limit) times; Get retries now."
        if g.gaveUp {
            return StatusLine(
                text: "No access on Hugging Face", tone: .error,
                help: "Stopped after \(g.limit) tries. " + (g.hasToken ? accept : token) + manual + " Then Get.")
        }
        if !g.hasToken {
            return StatusLine(
                text: g.tokenRejected == true ? "Replace the token in Settings" : "Add a token in Settings", tone: .pending, help: token + manual + retry)
        }
        let tries = g.tries > 0 ? " Tried \(g.tries) of \(g.limit)." : ""
        return StatusLine(text: g.tries > 0 ? "Still waiting for access" : "Waiting for access", tone: .pending, help: accept + manual + retry + tries)
    }

    /// What a row says once nothing is happening to it: its revision state, checks, Local, quick check or badge.
    static func resting(_ row: ModelRow, chip: String?) -> StatusLine {
        if let line = revisionLine(row) { return line }
        if let checks = row.checks { return checksLine(row, checks) }
        if row.local { return StatusLine(text: "Local", tone: .estimate, help: ModelHelp.local(row)) }
        if row.isQuickCheck, row.badge != .measured { return StatusLine(text: QuickCheck.title, tone: .busy, help: QuickCheck.help) }
        return badgeLine(row, chip: chip)
    }

    /// The line under the name. A state that took the line before keeps it while it applies: getting, making, loading,
    /// a failure, a precision waiting for its next load, a model leaving the catalog, a new revision, failed checks.
    /// Otherwise the model's purpose, with the verification it replaced (and a quiet state such as an update waiting
    /// for its Get, which its button says) in its hover.
    static func line(_ row: ModelRow, status: StatusLine, resting: StatusLine) -> StatusLine {
        let purpose = row.purpose.trimmingCharacters(in: .whitespacesAndNewlines)
        // A quiet state (an update waiting for its Get) never took the line; its button says it.
        let shown = row.catalogRemoved || [.busy, .error, .pending].contains(status.tone)
        let transient = row.catalogRemoved || (status != resting && shown) || status.tone == .error || revisionLine(row) != nil
        guard !transient, !purpose.isEmpty else { return status }
        let notes = (status == resting ? [] : [status]) + [resting]
        let help = notes.map { $0.text.prefix(1).uppercased() + $0.text.dropFirst() + "\n" + $0.help }.joined(separator: "\n\n")
        return StatusLine(text: purpose, tone: .plain, help: help)
    }

    /// A new revision of a loaded model: approved and waiting for the model's next idle moment, switching now, or
    /// refused at the switch (the previous one kept running). Nothing to say once it runs.
    static func revisionLine(_ row: ModelRow) -> StatusLine? {
        guard let r = row.revision else { return nil }
        if r.swapping { return StatusLine(text: "Updating\u{2026}", tone: .busy, help: "Requests wait a moment.") }
        if let error = r.error, !error.isEmpty {
            return StatusLine(text: "Update failed", tone: .error, help: error + "\nThe previous revision keeps running.")
        }
        if r.pending != nil {
            return StatusLine(text: "Update ready", tone: .busy, help: "Switches when nothing runs on it; the current revision stays for rollback.")
        }
        return nil
    }

    /// A model with retained checks shows their compact state ("Quick check passed", "Failed", "Stale", "Not checked";
    /// a local folder keeps "Local ·" in front); the evidence is the tooltip.
    static func checksLine(_ row: ModelRow, _ checks: ModelChecksStatus) -> StatusLine {
        let label = checks.label
        let tone: StatusLine.Tone = label == "Failed" ? .error : label == "Stale" ? .estimate : label == "Not checked" ? .quiet : .busy
        var help = ModelHelp.checks(checks)
        if row.local { help += "\n\n" + ModelHelp.local(row) }
        return StatusLine(text: (row.local ? "Local \u{00b7} " : "") + label, tone: tone, help: help)
    }

    /// The quality-gate state alone: where each figure was measured is that figure's own hover, so this line never
    /// says it (a row can pass no gate yet and still have figures measured on another Mac).
    /// The model's badge with each offered precision's own quality check in its hover, so the status never says "no
    /// quality check" while a precision's hover says its check did not pass (round-2 review, finding 4).
    static func badgeLine(_ row: ModelRow, chip: String?) -> StatusLine {
        let here = chip.map { "this Mac's chip (\($0))" } ?? "this Mac's chip"
        let base = row.precisions.first { PrecisionColumn.of($0, formats: row.formats) == "16" } ?? row.precisions.first ?? "The 16-bit format"
        var lines: [String]
        let tone: StatusLine.Tone
        switch row.badge {
        case .measured: lines = ["\(base) passed its quality check on \(here)."]; tone = .measured
        case .otherChip: lines = ["\(base) passed its quality check on another chip; not checked on \(here) yet."]; tone = .estimate
        case .unverified: lines = [ModelHelp.qualityLine(row, base)]; tone = .quiet
        }
        lines += ModelHelp.quantizedQualityLines(row)
        if row.badge == .unverified { lines.append(hasFigures(row) ? "Each figure's hover says where it was measured." : "No figures yet.") }
        return StatusLine(text: row.badge.rawValue, tone: tone, help: lines.joined(separator: " "))
    }
    /// The row shows any figure (a number, not a triangle) at any of its precisions.
    static func hasFigures(_ row: ModelRow) -> Bool {
        let folder = row.metrics.values.contains { $0 != .null }
        guard let kind = ModelStatKind(rawValue: row.kind) else { return folder || !row.folderFigures.isEmpty }
        if kind.isOperation, folder { return true }
        return row.precisions.contains { p in
            TableStats.set(row, precision: p).map { set in
                [TableFigures.speedMetric(set.kind), TableFigures.memoryMetric].contains { TableFigures.number(set.value($0)) != nil }
            } ?? false
        }
    }
}

/// Bytes as the table shows them: 412 MB, 1.3 GB. A few bytes read 1 KB, never "0 KB" for something that is there.
func formatBytes(_ bytes: Int64) -> String {
    let b = Double(bytes)
    if b >= 1e9 { return String(format: "%.1f GB", b / 1e9) }
    if b >= 1e6 { return String(format: "%.0f MB", b / 1e6) }
    if b > 0 && b < 500 { return "1 KB" }
    return String(format: "%.0f KB", max(0, b) / 1e3)
}
