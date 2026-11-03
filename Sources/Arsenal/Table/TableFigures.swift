// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// The table's two figure columns, the same for every kind: Speed, one figure in the kind's own unit with
// its change against the model's 16-bit format beneath, and Peak memory in GB, following the chosen precision. No
// AppKit/SwiftUI: unit-tested directly.
//
// - A figure is drawn at full contrast whatever produced it (a quick check, a controlled measurement); how it was
//   produced is its hover. Peak memory shows only where measured: a derived estimate is a missing figure.
// - A missing figure is no figure at all: the cell shows a yellow triangle and the hover says why.
// - A speed measured on another chip than this Mac's stays the number, with a small triangle and that chip's name
//   beneath it as its reference; it gets no delta, because a change measured elsewhere says nothing about this Mac.
//   Peak memory depends on the weights, not the chip: it shows as it is, the chip named in its hover.

enum FigureColumn: String, CaseIterable, Equatable {
    case speed, memory
    var title: String { self == .speed ? "Speed" : "Peak memory" }
}

/// One figure cell. `value == nil` is the missing state (the triangle).
struct TableFigure: Equatable {
    struct Delta: Equatable {
        enum Tone: Equatable { case better, worse, same }
        var text: String
        var tone: Tone
    }
    var value: String?
    /// The chip the figure was measured on, when it is not this Mac's ("M5 Max"); the cell's second line.
    var reference: String?
    /// Against the model's 16-bit format, both measured on this Mac; the cell's second line.
    var delta: Delta?
    var help: String
    var missing: Bool { value == nil }
}

enum TableFigures {
    static let columns = FigureColumn.allCases
    static let notMeasured = "Not measured yet."
    static let quickCheckHelp = "A short uncontrolled run: rough figures."

    /// The one speed figure a kind shows.
    static func speedMetric(_ kind: ModelStatKind) -> StatMetric {
        switch kind {
        case .decisions: return .millisecondsPerDecision
        case .chat: return .tokensPerSecond
        case .images: return .secondsPerImage
        case .voices, .sound, .enhancement: return .realTimeFactor
        case .audio, .embeddings, .perception, .restoration, .geometry: return .secondsPerRequest
        }
    }
    /// Figures that go to the speed cell's hover only.
    static func speedExtras(_ kind: ModelStatKind) -> [StatMetric] {
        switch kind {
        case .decisions: return [.batchedDecisionsPerSecond]
        case .chat: return [.firstTokenSeconds]
        case .voices: return [.firstAudioSeconds]
        case .images, .sound, .enhancement, .audio, .embeddings, .perception, .restoration, .geometry: return []
        }
    }
    static let memoryMetric = StatMetric.peakMemoryGB

    /// The column heading's hover, per kind (the group header's slots carry it). A kind Arsenal has no stat kind for (a
    /// folder's own kind) names the unit its folder declares for its speed column.
    static func headerHelp(_ column: FigureColumn, kind: ModelStatKind?, local: KindColumn? = nil) -> String {
        switch column {
        case .memory: return "Most memory the model uses while it runs, at the chosen precision, in GB."
        case .speed:
            switch kind {
            case .decisions: return "Time for one decision. Lower is faster."
            case .chat: return "Answer tokens per second. Higher is faster."
            case .images: return "Seconds to make one image. Lower is faster."
            case .voices: return "Seconds of speech made per second. Higher is faster."
            case .sound, .enhancement: return "Seconds of audio made per second. Higher is faster."
            case .audio, .embeddings, .perception, .restoration, .geometry: return "Time for one request of the model's own task. Lower is faster."
            case nil: return localSpeedHelp(local)
            }
        }
    }
    /// A folder kind's speed heading: its declared unit in words and which way is faster, from the unit alone.
    static func localSpeedHelp(_ column: KindColumn?) -> String {
        // No declared column: the speed is the quick check's one warm request (FolderReferenceFigures).
        guard let column else { return headerHelp(.speed, kind: .audio) }
        let unit = column.unit.trimmingCharacters(in: .whitespaces)
        let measured = "as each model's folder measures it with its own benchmark."
        switch unit.lowercased() {
        case "s", "sec", "seconds": return "Seconds per run, \(measured) Lower is faster."
        case "ms": return "Milliseconds per run, \(measured) Lower is faster."
        case "×", "x": return "Times faster than real time, \(measured) Higher is faster."
        case "": return "The folder's own speed figure, \(measured)"
        default:
            return unit.hasSuffix("/s") ? "\(unit), \(measured) Higher is faster." : "In \(unit), \(measured)"
        }
    }
    /// The column a folder kind's speed cell reads: `speed` when declared, else the first that is not memory or a fact.
    static func localSpeedColumn(_ ids: [String]) -> String? {
        ids.contains("speed") ? "speed" : ids.first { !["peak_ram", "peak_ram_mb", "resolution"].contains($0) && !MetricText.facts.contains($0) }
    }
    /// The shared heading's hover (above every group).
    static func headerHelp(_ column: FigureColumn) -> String {
        column == .speed
            ? "In each kind's own unit; the line beneath compares the chosen precision with the model's 16-bit format on this Mac."
            : headerHelp(.memory, kind: nil)
    }

    // MARK: Chips

    /// "m5-max", "Apple M5 Max" or "M5 Max" → "m5-max".
    static func slug(_ chip: String) -> String { ArsenalCore.HostChip.slug(chip) }
    /// "m5-max" → "M5 Max".
    static func chipName(_ chip: String) -> String {
        slug(chip).split(separator: "-").map { part in
            part.first.map { String($0).uppercased() + part.dropFirst() } ?? ""
        }.joined(separator: " ")
    }
    /// Measured on this Mac's configuration: the same chip, and the same GPU core count when the result records one
    /// (StatProvenance.isHost, with chip names compared in any of their spellings). Without a known host chip nothing
    /// counts as measured here.
    static func measuredHere(_ provenance: StatProvenance?, host: String?, gpuCores: Int = ArsenalCore.HostChip.gpuCores) -> Bool {
        guard let provenance, let host, !host.isEmpty else { return false }
        return slug(provenance.chip) == slug(host) && (provenance.gpuCores == nil || provenance.gpuCores == gpuCores)
    }
    /// "M5 Max", or "M5 Max · 40 GPU cores" when the result records them (hovers).
    static func hardware(_ provenance: StatProvenance) -> String {
        chipName(provenance.chip) + (provenance.gpuCores.map { " · \($0) GPU cores" } ?? "")
    }

    // MARK: Values

    /// States that carry a number the table shows: measured, quick check, and derived (memory estimated from the
    /// 16-bit measurement and each format's weight bytes).
    static func isFigure(_ state: StatState) -> Bool { [.measured, .quickCheck, .derived].contains(state) }
    static func isDerived(_ state: StatState) -> Bool { state == .derived }

    static func number(_ stat: StatValue) -> Double? {
        guard isFigure(stat.state), let v = stat.value, v.isFinite, v >= 0 else { return nil }
        return v
    }

    /// The figure as the cell prints it, in the kind's unit: "18 ms", "42 tok/s", "3.8 s", "47×", "15.9 GB".
    static func format(_ metric: StatMetric, _ v: Double) -> String? {
        switch metric {
        case .millisecondsPerDecision: return v < 10 ? String(format: "%.1f ms", v) : String(format: "%.0f ms", v)
        case .tokensPerSecond: return v < 10 ? String(format: "%.1f tok/s", v) : String(format: "%.0f tok/s", v)
        case .secondsPerImage: return v < 10 ? String(format: "%.1f s", v) : String(format: "%.0f s", v)
        case .secondsPerRequest:
            return v < 1 ? String(format: "%.0f ms", max(v * 1000, 1)) : v < 10 ? String(format: "%.1f s", v) : String(format: "%.0f s", v)
        case .realTimeFactor:
            // Compute seconds per audio second; shown as audio seconds per compute second.
            guard v > 0 else { return nil }
            let x = 1 / v
            return x < 10 ? String(format: "%.1f×", x) : String(format: "%.0f×", x)
        case .peakMemoryGB, .loadPeakMemoryGB: return ModelTableFigure.memoryText(v)
        case .batchedDecisionsPerSecond: return String(format: "%.0f decisions/s batched", v)
        case .firstTokenSeconds: return String(format: "first token %.2f s", v)
        case .firstAudioSeconds: return String(format: "first audio %.2f s", v)
        default: return String(format: "%.2f", v)
        }
    }
    /// Unit words for the hover's first line ("18 ms per decision").
    static func spelled(_ metric: StatMetric, _ v: Double) -> String {
        let shown = format(metric, v) ?? "—"
        switch metric {
        case .millisecondsPerDecision: return shown + " per decision"
        case .tokensPerSecond: return shown.replacingOccurrences(of: "tok/s", with: "answer tokens per second")
        case .secondsPerImage: return shown + " per image"
        case .secondsPerRequest: return shown + " per request"
        case .realTimeFactor: return shown + " faster than real time"
        case .peakMemoryGB, .loadPeakMemoryGB: return shown + " at most"
        default: return shown
        }
    }
    /// Higher is better for this metric (speeds as rates); lower for times.
    static func higherIsFaster(_ metric: StatMetric) -> Bool { [.tokensPerSecond, .batchedDecisionsPerSecond].contains(metric) }
    /// A speed as "how many per second", higher always faster (a real-time factor's reciprocal, a time's reciprocal).
    static func rate(_ metric: StatMetric, _ v: Double) -> Double? {
        guard v > 0 else { return nil }
        return higherIsFaster(metric) ? v : 1 / v
    }

    /// Vella's speed delta: "35% faster" / "20% slower"; at 2× or above "2.4× faster"; under 1 % "same".
    static func speedDelta(_ metric: StatMetric, value: Double, base: Double) -> TableFigure.Delta? {
        guard let v = rate(metric, value), let b = rate(metric, base), b > 0 else { return nil }
        let change = v / b - 1
        if abs(change) < 0.01 { return .init(text: "same", tone: .same) }
        if v / b >= 2 { return .init(text: String(format: "%.1f× faster", v / b), tone: .better) }
        return .init(text: String(format: "%.0f%%", abs(change) * 100) + (change > 0 ? " faster" : " slower"), tone: change > 0 ? .better : .worse)
    }

    // MARK: Cells

    /// The 16-bit dtype the row offers (fp16 or bf16).
    static func base(_ row: ModelRow) -> String? { row.precisions.first { PrecisionColumn.of($0, formats: row.formats) == "16" } }

    /// `shown`: the group the row shows in when it is not its kind's own (TableController.shownKinds).
    static func figure(_ row: ModelRow, precision: String, column: FigureColumn, host: String?, modePreviewChanged: Bool = false, shown: String? = nil) -> TableFigure {
        guard let kind = ModelStatKind(rawValue: row.kind) else { return local(row, precision: precision, column: column, host: host, shown: shown) }
        let set = TableStats.set(row, precision: precision)
        if set?.availability.state != .unavailable, let reason = row.dtypeUnavailableReason(precision) {
            return TableFigure(value: nil, help: "\(precision) is unavailable. " + ModelHelp.sentence(reason))
        }
        if modePreviewChanged {
            return TableFigure(value: nil, help: "The pending execution path has not been measured. Figures return after Reload.")
        }
        let base = base(row)
        let stated = figure(kind: kind, precision: precision, set: set, base: base.map { ($0, TableStats.set(row, precision: $0)) }, column: column, host: host)
        // Operation kinds read only their folder's results before they had stat sets; such a figure still stands.
        if kind.isOperation, stated.missing, set?.availability.state != .unavailable {
            let folder = local(row, precision: precision, column: column)
            if !folder.missing { return folder }
        }
        return stated
    }

    /// One precision's figure from its stat set (the table's row, or a runner in Get more models); `base` is the
    /// model's 16-bit dtype and its set, for the delta.
    static func figure(
        kind: ModelStatKind, precision: String, set: PrecisionStatSet?, base: (dtype: String, set: PrecisionStatSet?)?, column: FigureColumn, host: String?
    ) -> TableFigure {
        // A precision that cannot run here has no figures; its reason is the precision cell's hover and this one's.
        if set?.availability.state == .unavailable {
            return TableFigure(value: nil, help: "\(precision) is unavailable. " + ModelHelp.sentence(set?.availability.reason ?? "Its quality check did not pass"))
        }
        let metric = column == .speed ? speedMetric(kind) : memoryMetric
        let stat = (column == .memory ? set?.peakMemory.stat : set?.value(metric)) ?? StatValue()
        // Peak memory shows only where it was measured: a derived estimate is no figure (its triangle says not measured).
        guard let v = number(stat), let shown = format(metric, v), !(column == .memory && isDerived(stat.state)) else {
            let why = stat.state == .unavailable ? "Unavailable. " + ModelHelp.sentence(stat.reason ?? "No reason reported") : notMeasured
            return TableFigure(value: nil, help: "\(precision) · \(why)")
        }
        let here = measuredHere(stat.provenance, host: host)
        var lines = ["\(precision) · " + spelled(metric, v)]
        if column == .speed {
            for extra in speedExtras(kind) {
                if let x = set.map({ $0.value(extra) }).flatMap(number), let text = format(extra, x) { lines.append(text) }
            }
        }
        var figure = TableFigure(value: shown, help: "")
        lines += attribution(stat, column: column, host: host, figure: &figure)
        // Speed: against the 16-bit format, both measured on this Mac.
        if column == .speed, here, !isDerived(stat.state), let base, base.dtype != precision, let baseStat = base.set?.value(metric),
            !isDerived(baseStat.state), measuredHere(baseStat.provenance, host: host), let b = number(baseStat),
            let delta = speedDelta(metric, value: v, base: b)
        {
            figure.delta = delta
            let was = "\(base.dtype) (\(format(metric, b) ?? "—"))"
            lines.append(delta.tone == .same ? "Same speed as \(was)" : delta.text.prefix(1).uppercased() + delta.text.dropFirst() + " than \(was)")
        }
        figure.help = lines.joined(separator: "\n")
        return figure
    }

    /// How a derived figure was worked out (the data layer's arithmetic).
    static func basis(_ stat: StatValue) -> String? { stat.basis }

    /// Where a figure comes from, as its hover's lines after the value: another chip's speed names that chip as its
    /// reference (the cell's second line), then how and on what it was measured, then its basis.
    static func attribution(_ stat: StatValue, column: FigureColumn, host: String?, figure: inout TableFigure) -> [String] {
        var lines: [String] = []
        if let provenance = stat.provenance {
            let chip = chipName(provenance.chip)
            // Speed depends on the chip; peak memory on the weights, so another chip's memory figure stands as it is.
            if !measuredHere(provenance, host: host), column == .speed {
                figure.reference = chip
                lines.append("Not measured on your Mac; measured on \(chip).")
            }
            let how = stat.state == .quickCheck ? "Quick check" : isDerived(stat.state) ? "Estimate" : "Measured"
            lines.append("\(how) on \(hardware(provenance)) · \(date(provenance.date))")
            if stat.state == .quickCheck { lines.append(quickCheckHelp) }
        }
        // A derived figure's arithmetic; a quick check's operation and input.
        if let basis = basis(stat), !basis.isEmpty { lines.append(basis) }
        return lines
    }

    /// A folder kind's figures: its declared speed column and its peak memory, from the folder's results, attributed
    /// as every other figure is. A folder's results carry no chip name, only the service's badge (measured with this
    /// Mac's chip, or another chip), so no chip is named: the hover says "this Mac" or "another chip", never the host
    /// chip by assumption.
    ///
    /// A folder kind that declares no figure of its own reads its quick checks (`row.folderFigures`: Arsenal's own
    /// Resources or the folder's records, never a file added to the approved folder), shown as the built-in operation
    /// kinds show theirs: peak memory, and one warm request's time as the speed, the reference chip named when it is not
    /// this Mac's. In a group whose speed is not one request's time (a kit reader shown in Documents, whose speed is
    /// answer tokens per second) the speed stays a triangle that says so.
    static func local(_ row: ModelRow, precision: String, column: FigureColumn, host: String? = nil, shown: String? = nil) -> TableFigure {
        let own = declared(row, precision: precision, column: column)
        guard own.missing, let entry = row.folderFigures[precision] else { return own }
        if column == .speed, let group = shown, let kind = ModelStatKind(rawValue: KindPresentation.parent(group)), speedMetric(kind) != .secondsPerRequest {
            let title = KindPresentation.title(group) ?? group.capitalized
            return TableFigure(value: nil, help: "\(precision) · Speed in \(title) is \(unitWords(speedMetric(kind))), which one request's time cannot give.")
        }
        let metric: StatMetric = column == .memory ? .peakMemoryGB : .secondsPerRequest
        let stat = column == .memory ? entry.peakMemoryGB : entry.secondsPerRequest
        guard let v = number(stat), let text = format(metric, v) else { return own }
        var figure = TableFigure(value: text, help: "")
        let lines = ["\(precision) · " + spelled(metric, v)] + attribution(stat, column: column, host: host, figure: &figure)
        figure.help = lines.joined(separator: "\n")
        return figure
    }
    /// A speed unit in words, for a hover.
    static func unitWords(_ metric: StatMetric) -> String {
        switch metric {
        case .tokensPerSecond: return "answer tokens per second"
        case .realTimeFactor: return "seconds of audio made per second"
        case .millisecondsPerDecision: return "the time for one decision"
        case .secondsPerImage: return "seconds per image"
        default: return metric.unit
        }
    }
    /// A folder kind's own declared figure (its results' columns).
    static func declared(_ row: ModelRow, precision: String, column: FigureColumn) -> TableFigure {
        let id: String?
        switch column {
        case .memory: id = MetricText.peakRAM.id
        case .speed: id = localSpeedColumn(row.metrics.keys.sorted())
        }
        guard let id else { return TableFigure(value: nil, help: "\(precision) · \(notMeasured)") }
        let text = MetricText(row, KindColumn(id: id, title: column.title, unit: column == .memory ? "GB" : ""))
        guard text.value != MetricText.missing else { return TableFigure(value: nil, help: "\(precision) · \(notMeasured)") }
        let value = text.value.hasPrefix("~") ? String(text.value.dropFirst()) : text.value
        // The folder's own words for the figure (its workload, e.g. "38 s · 720p, 5 s clip") lead the hover.
        let full = row.metrics[id]?.string ?? value
        var lines = ["\(precision) · " + (column == .memory ? value + " at most" : full)]
        var figure = TableFigure(value: value, help: "")
        switch row.badge {
        case .measured: lines.append("Measured on this Mac with the folder's own benchmark.")
        case .otherChip:
            // Speed depends on the chip; peak memory on the weights (as for the built-in kinds).
            if column == .speed {
                figure.reference = "Other chip"
                lines.append("Not measured on your Mac; measured on another chip.")
            } else {
                lines.append("Measured on another chip with the folder's own benchmark.")
            }
        case .unverified: lines.append("From the folder's results, not a qualified measurement.")
        }
        if row.isQuickCheck(id) { lines.append(quickCheckHelp) }
        figure.help = lines.joined(separator: "\n")
        return figure
    }

    /// "2026-10-09T…" → "9 Oct 2026"; anything else as given.
    static func date(_ text: String) -> String {
        let iso = ISO8601DateFormatter()
        let fractional = ISO8601DateFormatter(); fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let day = ISO8601DateFormatter(); day.formatOptions = [.withFullDate]
        guard let d = iso.date(from: text) ?? fractional.date(from: text) ?? day.date(from: text) else { return text }
        let f = DateFormatter(); f.locale = Locale(identifier: "en_GB"); f.dateFormat = "d MMM yyyy"; f.timeZone = TimeZone(identifier: "UTC")
        return f.string(from: d)
    }
}
