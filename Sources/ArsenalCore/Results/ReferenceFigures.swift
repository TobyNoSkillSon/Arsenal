// SPDX-License-Identifier: MIT
import Foundation

public enum ReferenceMatch: String, Codable, Sendable { case sameChip = "same_chip", sameFamily = "same_family", nearest }
public struct ModelReference: Codable, Sendable {
    public var match: ReferenceMatch
    public var chip: String
    public var metrics: [String: Double?]
    public var date: String
    public var dtype: String
    public var mode: String
    public var runner: String
    public var suite: BenchmarkSuiteIdentity
    /// "Measured", or "quick check" when no measured result exists and rough figures stand in.
    public var verification: String
}
public enum ReferenceFigures {
    /// Stable lookup order, independent of the host and of which results happen to be present.
    public static let chipOrder = [
        "m1", "m1-pro", "m1-max", "m1-ultra", "m2", "m2-pro", "m2-max", "m2-ultra", "m3", "m3-pro", "m3-max", "m3-ultra",
        "m4", "m4-pro", "m4-max", "m4-ultra", "m5", "m5-pro", "m5-max", "m5-ultra"
    ]
    /// The closest measured result; when no measured one exists, the closest quick check, labelled as such, so the
    /// user sees rough figures rather than none.
    public static func best(_ results: [ResultsFile], chip: String, dtype: String, mode: String) -> ModelReference? {
        let usable = results.filter {
            $0.recordedAt != nil && $0.precision == dtype && $0.mode == mode && ($0.chip.lowercased() == chip.lowercased() || chipOrder.contains($0.chip.lowercased()))
        }
        return closest(usable.filter(\.eligibleMeasurement), chip: chip) ?? closest(usable.filter(\.quickCheck), chip: chip)
    }
    private static func closest(_ eligible: [ResultsFile], chip: String) -> ModelReference? {
        let chip = chip.lowercased()
        func newest(_ values: [ResultsFile]) -> ResultsFile? {
            values.sorted { a, b in a.recordedAt == b.recordedAt ? a.runner < b.runner : ResultsFile.newestFirst(a, b) }.first
        }
        func reference(_ result: ResultsFile, _ match: ReferenceMatch) -> ModelReference {
            .init(
                match: match, chip: result.chip, metrics: result.metrics, date: result.date, dtype: result.precision, mode: result.mode, runner: result.runner, suite: result.suite,
                verification: result.eligibleMeasurement ? "Measured" : result.verificationLabel)
        }
        if let result = newest(eligible.filter { $0.chip.lowercased() == chip }) { return reference(result, .sameChip) }
        // An unlisted host uses the newest end of the fixed lineup as its nearest known reference.
        let position = chipOrder.firstIndex(of: chip) ?? chipOrder.count
        let family = chipOrder.contains(chip) ? eligible.filter { RunnerSelection.family($0.chip) == RunnerSelection.family(chip) } : []
        let candidates = family.isEmpty ? eligible : family
        let ordered = candidates.sorted { a, b in
            let left = chipOrder.firstIndex(of: a.chip.lowercased())!, right = chipOrder.firstIndex(of: b.chip.lowercased())!
            let leftDistance = abs(left - position), rightDistance = abs(right - position)
            if leftDistance != rightDistance { return leftDistance < rightDistance }
            if left != right { return left < right }
            return a.recordedAt == b.recordedAt ? a.runner < b.runner : ResultsFile.newestFirst(a, b)
        }
        return ordered.first.map { reference($0, family.isEmpty ? .nearest : .sameFamily) }
    }
}
