// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// A runner's figures at the displayed precision, the same two the Models table shows (Speed, Peak memory) and formatted
/// the same way (TableFigures): full contrast, another chip's figure named beneath, a missing one as the yellow triangle.
enum BrowserRunnerStats {
    static func rows(kind: String, dtype: String, runner: CatalogRunner, base: String? = nil, host: String? = HostChip.current) -> [ModelDetail.RunnerStat] {
        guard let kind = ModelStatKind(rawValue: kind) else { return [] }
        func set(_ d: String) -> PrecisionStatSet? { runner.stats[d].flatMap { $0.kind == kind && $0.dtype == d && $0.runner == runner.name ? $0 : nil } }
        return TableFigures.columns.map { column in
            let f = TableFigures.figure(kind: kind, precision: dtype, set: set(dtype), base: base.map { ($0, set($0)) }, column: column, host: host)
            return ModelDetail.RunnerStat(title: column.title, value: f.value ?? "", help: f.help, missing: f.missing, reference: f.reference, delta: f.delta)
        }
    }

    static func feature(_ feature: ArsenalCore.RunnerFeature) -> ModelDetail.RunnerFeature {
        let symbol: String, title: String
        switch feature {
        case .batching: symbol = "square.stack.3d.up"; title = "Batches requests"
        case .sequentialOnly: symbol = "arrow.right"; title = "One request at a time"
        case .streaming: symbol = "waveform.path"; title = "Streams its output"
        }
        return ModelDetail.RunnerFeature(id: feature.rawValue, symbol: symbol, help: feature.description, title: title)
    }
}
