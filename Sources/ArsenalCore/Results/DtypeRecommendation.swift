// SPDX-License-Identifier: MIT
import Foundation

/// Recommend only comparable, qualified timings; without measured evidence the vendor dtype remains selected.
public enum DtypeRecommendation {
    public static func best(_ results: [ResultsFile], kind: String, descriptor: KindSpec? = nil) -> String? {
        let key: String, maximize: Bool
        switch kind {
        case "chat": key = "decode_tokens_per_s"; maximize = true
        case "voices", "sound", "enhancement": key = "speed_x_realtime"; maximize = true
        case "images": key = "seconds_per_image"; maximize = false
        case "decisions": key = "latency_s"; maximize = false
        default:
            guard let metric = descriptor?.comparisonMetric, let direction = descriptor?.comparisonDirection else { return nil }
            key = metric; maximize = direction == "higher"
        }
        let eligible = results.filter { $0.eligibleMeasurement && $0.kind == kind && $0.metrics[key].flatMap({ $0 }).map({ $0.isFinite && $0 > 0 }) == true }
        guard let latest = eligible.sorted(by: ResultsFile.newestFirst).first else { return nil }
        let comparable = eligible.filter { $0.suite == latest.suite && $0.profile == latest.profile }
        return comparable.sorted { a, b in
            let left = a.metrics[key]!!, right = b.metrics[key]!!
            if left == right { return a.recordedAt == b.recordedAt ? a.precision < b.precision : ResultsFile.newestFirst(a, b) }
            return maximize ? left > right : left < right
        }.first?.precision
    }
}
