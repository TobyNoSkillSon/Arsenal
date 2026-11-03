// SPDX-License-Identifier: MIT
import Foundation

/// A quick check's own record, kept in the model folder (`results/quick-checks/<run>.json`) beside the table results it
/// supplies: the measuring script's rows, one per precision, as recorded. Its figures are quick checks, never Measured:
/// a busy Mac, one Load and two requests, no interleaved 16-bit brackets. Table results bind to it by run and value,
/// for peak memory and the kind's speed figure where that is one request's time (secondsPerRequest, secondsPerImage).
public enum QuickCheckEvidence {
    public static let schema = "quick-check-1"
    public static let directory = "results/quick-checks/"
    /// Metrics a quick-check record can supply. Peak memory: the worker's macOS phys_footprint_peak in MiB, stored in
    /// GB (10^9 bytes). Request time: one warm request's wall seconds.
    public static let metrics: [StatMetric] = [.peakMemoryGB, .secondsPerRequest, .secondsPerImage]
    public static func gigabytes(mebibytes: Double) -> Double { mebibytes * 1_048_576 / 1e9 }
}
