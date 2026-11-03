// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import ArsenalWire

/// What one request on a worker is expected to use, for the two memory stops the supervisor samples every half second
/// while the request runs ("Do not exceed hundred gigabytes of RAM", after a TeleOCR int4 generation grew
/// one worker to tens of gigabytes and the Mac into 15 GB of swap). A stop kills the worker and fails its requests with
/// `memory` (HTTP 507) once it has exited.
///
/// - The Mac-wide limit (`MacMemoryGuard`) is the main stop: it re-reads the whole Mac's memory in use on every tick and,
///   past 100/128 of RAM, stops the running worker that has grown furthest past its expected peak plus 1 GB. Legitimate
///   large requests run as long as the Mac has room, whatever their expected peak says.
/// - A runaway backstop per worker, only where it cannot cut a legitimate large input: four times a *measured request
///   peak* and at least 16 GB above it. A measured peak is the only figure that describes a request (Load peaks and
///   forecasts do not), and 16 GB of activations or cache past it exceeds what a small model's bounded context can
///   hold; for models whose peak is above 5 GB the 4× bound lies at or beyond the Mac-wide limit anyway.
///
/// A forced (Load Anyway) resident accepted swap at Load, so it takes no part in the Mac-wide stop; it keeps its earlier
/// expected-peak bound (twice the expected peak, at least 4 GB above it) as its only stop, and none without a known peak.
public struct WorkerMemoryCeiling: Sendable, Equatable {
    /// What the error message calls the worker: a model's name, or "the INT8 conversion of …".
    public var name: String
    /// Megabytes; nil when the model declares no peak, forecast or resident footprint.
    public var expectedMB: Double?
    public var precision: String?
    /// Load Anyway: outside the Mac-wide stop.
    public var forced: Bool
    /// Megabytes of footprint at which this worker is a runaway regardless of the Mac; nil means none.
    public var runawayMB: Double?
    /// True when `runawayMB` derives from a controlled measurement of the request peak (the message says "measured"); a
    /// quick-check or estimated table peak arms the same stop and is called the expected peak.
    public var measuredPeak: Bool
    public init(name: String, expectedMB: Double? = nil, precision: String? = nil, forced: Bool = false, runawayMB: Double? = nil, measuredPeak: Bool = false) {
        self.name = name; self.expectedMB = expectedMB; self.precision = precision; self.forced = forced; self.runawayMB = runawayMB
        self.measuredPeak = measuredPeak
    }
    /// From the requirement Load admission used (table peak, Load peak, forecast or resident footprint).
    public init(name: String, requirement: LoadMemoryRequirement?, precision: String?, forced: Bool) {
        let expected = requirement?.neededGB.flatMap { $0.isFinite && $0 > 0 ? $0 * 1000 : nil }
        let armed = requirement?.basis == .peak && expected != nil
        let runaway: Double? =
            forced ? expected.map { max($0 * 2, $0 + 4000) } : armed ? expected.map { max($0 * Self.runawayFactor, $0 + Self.runawayMarginMB) } : nil
        // A quick-check or estimated table peak arms the stop as before but is never called "measured".
        let measured = armed && !forced && requirement?.peakMeasured != false
        self.init(name: name, expectedMB: expected, precision: precision, forced: forced, runawayMB: runaway, measuredPeak: measured)
    }
    /// A worker with no model figures (Get-time conversion, intake): the Mac-wide stop only.
    public static func unmetered(_ name: String) -> Self { Self(name: name) }
    public static let runawayFactor = 4.0
    public static let runawayMarginMB = 16_000.0
    public var runawayLimitBytes: UInt64? { runawayMB.map { UInt64(max(0, $0) * 1e6) } }

    static func gb(_ bytes: UInt64) -> String { MemoryDisplay.text(decimalGB: Double(bytes) / 1e9) }
    static func number(_ bytes: UInt64) -> JSONValue { .number((Double(bytes) / 1e7).rounded() / 100) }
    var expectedText: String? {
        expectedMB.map { mb in MemoryDisplay.text(decimalGB: mb / 1000) + (precision.map { " at \($0.uppercased())" } ?? "") }
    }
    /// The backstop's stop. Says only what is true at that moment: how far the worker grew and against which figure.
    func runawayFailure(footprintBytes: UInt64, limitBytes: UInt64) -> WorkerFailure {
        let basis: String
        if forced {
            basis = expectedText.map { " (it is expected to need about \($0) and was loaded with Load Anyway)" } ?? ""
        } else {
            basis = expectedText.map { measuredPeak ? " (its measured peak is \($0))" : " (its expected peak is \($0))" } ?? ""
        }
        var details: JSONObject = [
            "reason": "runaway", "footprint_gb": Self.number(footprintBytes), "ceiling_gb": Self.number(limitBytes), "status": 507
        ]
        if let expectedMB { details["expected_gb"] = .number((expectedMB / 10).rounded() / 100) }
        return WorkerFailure(
            code: "memory",
            message: "Arsenal stopped \(name): it grew to \(Self.gb(footprintBytes)), past its \(Self.gb(limitBytes)) runaway limit\(basis). The model was unloaded; try again.",
            details: details)
    }
    /// The Mac-wide stop. Names the Mac's state and this worker's growth during the request; it does not guess whether
    /// the input or the output made it grow.
    func macFailure(footprintBytes: UInt64, startBytes: UInt64, inUseBytes: UInt64, limitBytes: UInt64) -> WorkerFailure {
        let expected = expectedText.map { " (it is expected to need about \($0))" } ?? ""
        var details: JSONObject = [
            "reason": "mac_memory_limit", "footprint_gb": Self.number(footprintBytes), "start_gb": Self.number(startBytes),
            "mac_in_use_gb": Self.number(inUseBytes), "mac_limit_gb": Self.number(limitBytes), "status": 507
        ]
        if let expectedMB { details["expected_gb"] = .number((expectedMB / 10).rounded() / 100) }
        return WorkerFailure(
            code: "memory",
            message:
                "Arsenal stopped \(name): the Mac had \(Self.gb(inUseBytes)) of memory in use, past its \(Self.gb(limitBytes)) limit (of this Mac's \(Self.gb(UInt64(Double(limitBytes) / MacMemoryGuard.safeInUseFraction)))), "
                + "and \(name) had grown from \(Self.gb(startBytes)) to \(Self.gb(footprintBytes)) during this request\(expected). "
                + "The model was unloaded; try again when more memory is free.",
            details: details)
    }
    /// Footprint of a worker and every process it started.
    static func footprint(pid: Int32) -> UInt64 {
        guard pid > 0 else { return 0 }
        var total: UInt64 = 0
        for member in ServedFootprintPeak.processTree(roots: [pid]) {
            var info = rusage_info_v4()
            let status = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(member, RUSAGE_INFO_V4, $0) }
            }
            if status == 0 { total += info.ri_phys_footprint }
        }
        return total
    }
}
