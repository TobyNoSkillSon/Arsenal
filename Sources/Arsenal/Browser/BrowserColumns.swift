// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation

// Column geometry: fixed chevron/icon/action columns, nine elastic columns with a
// weight, min and max, filled proportionally ("water-filling"), a hiding order when the minimums do not fit, and a flex
// gap before the action column so Add/Remove stay on the right edge. Plain values: unit-tested.

enum BrowserColumns {
    struct Spec: Equatable { var weight: CGFloat, min: CGFloat, max: CGFloat }
    /// List padding: inside the first and last columns, so a row view spans the whole width (its card insets by 8).
    static let pad: CGFloat = 16, gap: CGFloat = 12
    static let fixed: [BrowserColumn: CGFloat] = [.chevron: 12 + pad, .icon: 18, .action: 76 + pad]
    static let elastic: [BrowserColumn] = [.model, .kind, .params, .download, .ram, .dtype, .licence, .languages, .status]
    /// The weights give each column its one-line width at the default window (1440 pt, sidebar open): Kind fits
    /// "Upscale & Smooth", Languages "100+ languages", Download "On this Mac". Status is empty for nearly every model,
    /// so it gets the least (its two words wrap). The minimums are the wrapping fallback of a narrower window.
    static let specs: [BrowserColumn: Spec] = [
        .model: Spec(weight: 0.179, min: 150, max: 420), .kind: Spec(weight: 0.129, min: 84, max: 130), .params: Spec(weight: 0.053, min: 48, max: 72),
        .download: Spec(weight: 0.086, min: 78, max: 104), .ram: Spec(weight: 0.092, min: 84, max: 104), .dtype: Spec(weight: 0.165, min: 150, max: 176),
        .licence: Spec(weight: 0.114, min: 100, max: 210), .languages: Spec(weight: 0.12, min: 72, max: 150), .status: Spec(weight: 0.062, min: 56, max: 120)
    ]
    /// Hidden first when the minimums do not fit; their facts stay in the expanded row.
    static let hideOrder: [BrowserColumn] = [.kind, .languages, .dtype, .status]
    static var defaultWeights: [BrowserColumn: CGFloat] { specs.mapValues(\.weight) }
    static let weightsKey = "BrowserColumnWeights"

    struct Layout: Equatable {
        var widths: [BrowserColumn: CGFloat]
        var hidden: Set<BrowserColumn>
        /// The empty space before the action column once every elastic column is at its max.
        var flex: CGFloat
        /// Width shared by the elastic columns.
        var elasticWidth: CGFloat
        /// The table width this layout was solved for.
        var tableWidth: CGFloat
    }

    /// Every column (spacing included) for a table `width` wide. Each visible column takes `width + gap`.
    static func solve(width: CGFloat, weights: [BrowserColumn: CGFloat] = defaultWeights) -> Layout {
        var visible = elastic
        let fixedSum = fixed.values.reduce(0, +)
        // Visible columns: chevron, icon, the elastic ones, flex, action.
        func available(_ n: Int) -> CGFloat { width - fixedSum - CGFloat(n + 4) * gap }
        var hidden: Set<BrowserColumn> = []
        for c in hideOrder where visible.map({ specs[$0]!.min }).reduce(0, +) > available(visible.count) {
            visible.removeAll { $0 == c }; hidden.insert(c)
        }
        let total = available(visible.count)
        var widths: [BrowserColumn: CGFloat] = [:]
        var free = visible, remaining = total
        while !free.isEmpty {
            let sum = free.map { weights[$0] ?? specs[$0]!.weight }.reduce(0, +)
            var clamped = false
            for c in free {
                let s = specs[c]!, w = remaining * (weights[c] ?? s.weight) / max(sum, 0.0001)
                if w < s.min || w > s.max {
                    let v = w < s.min ? s.min : s.max
                    widths[c] = v; remaining -= v; free.removeAll { $0 == c }; clamped = true; break
                }
            }
            if !clamped { for c in free { widths[c] = remaining * (weights[c] ?? specs[c]!.weight) / sum }; free = [] }
        }
        for (c, w) in fixed { widths[c] = w }
        let used = visible.map { widths[$0]! }.reduce(0, +)
        return Layout(widths: widths, hidden: hidden, flex: max(0, (total - used).rounded(.down)), elasticWidth: total, tableWidth: width)
    }

    /// The user dragged a column to `width`: that column's weight becomes its share, the others keep their ratios.
    static func resized(_ c: BrowserColumn, to width: CGFloat, in layout: Layout, weights: [BrowserColumn: CGFloat]) -> [BrowserColumn: CGFloat] {
        guard let s = specs[c], layout.elasticWidth > 0 else { return weights }
        let target = min(max(width, s.min), s.max)
        let others = elastic.filter { $0 != c }
        let rest = others.map { weights[$0] ?? specs[$0]!.weight }.reduce(0, +)
        func with(_ share: CGFloat) -> [BrowserColumn: CGFloat] {
            var next = weights
            next[c] = share
            for o in others { next[o] = (weights[o] ?? specs[o]!.weight) / max(rest, 0.0001) * max(0, 1 - share) }
            return next
        }
        // Clamped neighbours change what a share is worth: refine until the column lands on the dragged width.
        var share = min(0.95, target / layout.elasticWidth)
        for _ in 0..<6 {
            let got = solve(width: layout.tableWidth, weights: with(share)).widths[c] ?? target
            guard got > 0, abs(got - target) > 0.5 else { break }
            share = min(0.95, max(0.001, share * target / got))
        }
        return with(share)
    }

    static func savedWeights(_ defaults: UserDefaults = .standard) -> [BrowserColumn: CGFloat] {
        guard let raw = defaults.dictionary(forKey: weightsKey) as? [String: Double] else { return defaultWeights }
        var w = defaultWeights
        for (k, v) in raw { if let c = BrowserColumn(rawValue: k), specs[c] != nil, v.isFinite, v > 0 { w[c] = v } }
        return w
    }
    static func save(_ weights: [BrowserColumn: CGFloat], _ defaults: UserDefaults = .standard) {
        defaults.set(Dictionary(uniqueKeysWithValues: weights.map { ($0.key.rawValue, Double($0.value)) }), forKey: weightsKey)
    }
}
