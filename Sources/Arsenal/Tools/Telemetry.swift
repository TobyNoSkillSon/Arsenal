// SPDX-License-Identifier: MIT
import ArsenalCore
import SwiftUI

// The figures above a tool's work area (chat, image): one fixed-width column per figure, this run's value, the measured
// reference beneath it and, under that, what the reference was measured on (chip, prompt or image size, date). Only the
// service's own figures (C4 timing, C3 activity) are shown, never a client estimate. Text only, no charts; the grid keeps
// its place and widths from the first frame.

/// What a tool window knows about its model.
struct ToolModel: Equatable {
    var id: String
    var name: String
    var kind: String
    /// The dtype it is loaded with (the selected one when not loaded).
    var dtype: String
    /// "fast" or "exact", as C5 and results files name the mode.
    var mode: String
    var contextLimit: Int?
    var memoryNotice: String?

    init(id: String, name: String, kind: String, dtype: String, mode: String, contextLimit: Int? = nil) {
        self.id = id; self.name = name; self.kind = kind; self.dtype = dtype; self.mode = mode; self.contextLimit = contextLimit
    }
    init(_ row: ModelRow) {
        self.init(
            id: row.id, name: row.name, kind: row.kind, dtype: row.residentDType ?? row.selectedPrecision, mode: row.fastAvailable && row.fast ? "fast" : "exact")
        memoryNotice = row.loadMemoryNote
    }
    /// The window's top line: the precision, and "Fast" only while the fast path runs (Arsenal has no Exact/Fast
    /// switch, so the ordinary path is not named).
    var detail: String { mode == "fast" ? "\(precision) \u{00b7} Fast" : precision }
    /// The precision as the Models table says it: "16-bit", "8-bit", "4-bit" (the dtype's own name when unknown).
    var precision: String { Self.bits(dtype) }
    static func bits(_ dtype: String) -> String {
        let d = dtype.lowercased()
        if ["fp16", "bf16", "float16", "bfloat16", "f16", "16"].contains(d) { return "16-bit" }
        if ["int8", "q8", "8bit", "8"].contains(d) || d.hasPrefix("int8") { return "8-bit" }
        if ["int4", "q4", "4bit", "4"].contains(d) || d.hasPrefix("int4") { return "4-bit" }
        return dtype
    }
    /// The Details strip's Precision column: how this model runs ("16-bit · Fast").
    var precisionFigure: Figure {
        Figure(title: "Precision", value: detail, reference: nil, help: "The precision this model runs at, set in Models.", width: 112)
    }
}

/// One figure: its title (with its definition as tooltip), this run's value (nil before anything ran), the reference
/// value, and its column's fixed width.
struct Figure: Equatable, Identifiable {
    var title: String
    var value: String?
    var reference: String?
    var help: String
    var width: CGFloat
    var id: String { title }
}

/// The reference line's state: loading, none measured, one found, or the API could not say.
enum ReferenceState: Equatable {
    case loading, none, found(ToolReference), failed(String)
}

/// What the reference row says beside its values: compared (with its baseline), or why there is nothing to compare.
enum ReferenceLine: Equatable {
    case compared(ToolReference)
    case unavailable(String)
}

enum FigureText {
    static func seconds(_ s: Double?, decimals: Int = 2) -> String? {
        guard let s, s.isFinite, s >= 0, s < 1e9 else { return nil }
        return s < 10 ? String(format: "%.\(decimals)f s", s) : String(format: "%.1f s", s)
    }
    static func rate(_ r: Double?, unit: String = "tok/s") -> String? {
        guard let r, r.isFinite, r > 0 else { return nil }
        if r < 100 { return String(format: "%.1f", r) + " " + unit }
        return r.roundedInt.map { "\($0.grouped) \(unit)" }
    }
    static func gigabytes(mb: Double?) -> String? {
        guard let mb, mb.isFinite, mb > 0, mb < 1e12 else { return nil }
        return String(format: "%.1f GB", MemoryDisplay.gib(decimalGB: mb / 1000))
    }
    static func milliseconds(_ ms: Double?) -> String? {
        guard let ms, ms >= 0, let whole = ms.roundedInt else { return nil }
        return "\(whole.grouped) ms"
    }
    /// "8,214 in → 96 out"; an unknown side is a dash, both unknown is nil.
    static func tokens(in prompt: Int?, out reply: Int?) -> String? {
        guard prompt != nil || reply != nil else { return nil }
        return "\(prompt?.grouped ?? "\u{2014}") in \u{2192} \(reply?.grouped ?? "\u{2014}") out"
    }
    /// 32768 → "32K".
    static func context(_ tokens: Int) -> String { tokens >= 1024 && tokens % 1024 == 0 ? "\(tokens / 1024)K" : tokens.grouped }
}

/// Numbers large, their units small: "8,214 in → 96 out" reads as two numbers.
struct FigureValue: View {
    let text: String
    let size: CGFloat
    let color: Color
    let unitColor: Color
    var body: some View {
        let words = text.split(separator: " ", omittingEmptySubsequences: false).map(String.init)
        words.enumerated().reduce(Text("")) { line, item in
            let (i, word) = item
            let number = word.contains(where: \.isNumber) || word == "\u{2014}"
            let piece = Text((i == 0 ? "" : " ") + word)
                .font(number ? .system(size: size, weight: .medium).monospacedDigit() : .system(size: size * 0.8, weight: .regular))
                .foregroundColor(number ? color : unitColor)
            return line + piece
        }
        .lineLimit(1)
    }
}

/// The figures grid: titles, this run, the reference and the baseline it was measured on. Fixed column widths and
/// monospaced digits, so a changing value never moves its neighbours; values change in place.
struct TelemetryGrid: View {
    @Environment(\.palette) private var p
    let figures: [Figure]
    let reference: ReferenceLine?
    /// False where no reference can exist yet (sound): the Reference and baseline rows are left out, not drawn empty.
    var showsReference = true

    static let labelWidth: CGFloat = 80
    static let spacing: CGFloat = 12

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: Self.spacing, verticalSpacing: 3) {
            GridRow {
                Color.clear.frame(width: Self.labelWidth, height: 1)
                ForEach(figures) { f in
                    Text(f.title).font(.system(size: 11, weight: .medium)).foregroundStyle(p.t3.swiftUI).lineLimit(1)
                        // The whole column heading carries the hover, not only the glyphs of its title.
                        .frame(width: f.width, alignment: .leading).contentShape(Rectangle()).windowHelp(f.help)
                }
            }
            GridRow {
                Text("This run").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).frame(width: Self.labelWidth, alignment: .leading)
                ForEach(figures) { f in
                    FigureValue(text: f.value ?? "\u{2014}", size: 15, color: (f.value == nil ? p.t4 : p.t1).swiftUI, unitColor: p.t2.swiftUI)
                        .frame(width: f.width, alignment: .leading)
                }
            }
            if showsReference { referenceRows }
        }
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder private var referenceRows: some View {
        GridRow {
            Text("Reference").font(.system(size: 12)).foregroundStyle(p.t3.swiftUI).frame(width: Self.labelWidth, alignment: .leading)
            switch reference {
            case .compared:
                ForEach(figures) { f in
                    FigureValue(text: f.reference ?? "\u{2014}", size: 12, color: (f.reference == nil ? p.t4 : p.t2).swiftUI, unitColor: p.t3.swiftUI)
                        .frame(width: f.width, alignment: .leading)
                }
            case .unavailable(let why):
                Text(why).font(.system(size: 12)).foregroundStyle(p.t3.swiftUI).lineLimit(1).gridCellColumns(max(1, figures.count))
            case nil:
                Text(" ").font(.system(size: 12)).gridCellColumns(max(1, figures.count))
            }
        }
        GridRow {
            Color.clear.frame(width: Self.labelWidth, height: 1)
            Text(baseline).font(.system(size: 11)).foregroundStyle(p.t3.swiftUI).lineLimit(1).gridCellColumns(max(1, figures.count))
                .contentShape(Rectangle()).windowHelp(baselineHelp)
        }
    }

    private var baseline: String {
        if case .compared(let r) = reference { return r.baseline } else { return " " }
    }
    private var baselineHelp: String? {
        if case .compared(let r) = reference { return r.help } else { return nil }
    }
}

extension Double {
    /// The nearest whole number when it is representable (finite, within ±1e15); nil otherwise. Figures come from
    /// workers and the API: a finite 1e20 must read as unknown, never trap.
    var roundedInt: Int? {
        guard isFinite, abs(self) < 1e15 else { return nil }
        return Int(rounded())
    }
}

extension Int {
    /// 8214 → "8,214": the app's English number style, whatever the Mac's region (a "." grouping reads as a decimal).
    var grouped: String { formatted(.number.locale(Locale(identifier: "en_US"))) }
}
