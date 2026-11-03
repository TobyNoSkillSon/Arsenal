// SPDX-License-Identifier: MIT
import Foundation

/// Version 2 lives in results/*.md with +++ TOML front matter. Version 1 JSON stays readable.
public struct PrecisionStatsResult: Codable, Sendable, Equatable {
    public var schema: Int = 2
    public var model: String
    public var mode: String
    public var stats: PrecisionStatSet
    // In-memory legacy adapter metadata. The original ResultsFile remains the persisted evidence;
    // this coverage is deliberately excluded from the public schema-2 format.
    var qualityEvidence: RecordedQualityEvidence?
    var qualityIdentityMismatch = false
    enum CodingKeys: String, CodingKey { case schema, model, mode, stats }

    public init(model: String, mode: String, stats: PrecisionStatSet) { self.model = model; self.mode = mode; self.stats = stats }
    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= 1_048_576 else { throw ManifestError.invalid("Stat result exceeds 1 MiB") }
        let text = String(decoding: data, as: UTF8.self)
        let result: Self
        if text.hasPrefix("+++\n"), let end = text.range(of: "\n+++", range: text.index(text.startIndex, offsetBy: 4)..<text.endIndex) {
            result = try TOMLParser.decode(Self.self, from: String(text[text.index(text.startIndex, offsetBy: 4)..<end.lowerBound]))
        } else {
            result = try JSONDecoder().decode(Self.self, from: data)
        }
        guard result.schema == 2, ModelManifest.validID(result.model), ["fast", "exact"].contains(result.mode) else { throw ManifestError.invalid("Invalid stat result identity") }
        try result.stats.validate(); return result
    }
    public func markdown(notes: String = "") throws -> Data {
        try stats.validate()
        let json = try WireJSON.decode(JSONEncoder().encode(self))
        func toml(_ value: JSONValue) throws -> TOMLValue {
            switch value {
            case .string(let s): return .string(s)
            case .number(let n): return .decimal(n)
            case .bool(let b): return .bool(b)
            case .object(let o): return .table(try o.filter { $0.value != .null }.mapValues(toml))
            case .array(let a): return .array(try a.map(toml))
            case .null: throw ManifestError.invalid("TOML cannot encode null")
            }
        }
        let table = try json.mapValues(toml)
        return Data(("+++\n" + TOMLWriter.encode(table) + "+++\n" + notes).utf8)
    }
}

public enum PrecisionStatsReader {
    /// Current runner support takes precedence over historical results, including a previously passed gate.
    public static func sets(
        kind: String, baseDtype: String, runner: String, supported: [String], results: [PrecisionStatsResult], chip: String = HostChip.current, gpuCores: Int = HostChip.gpuCores
    ) -> [String:
        PrecisionStatSet]
    {
        guard let kind = ModelStatKind(rawValue: kind) else { return [:] }
        func instant(_ result: PrecisionStatsResult) -> Date {
            let date = result.stats.availability.provenance?.date ?? ""
            let formatter = ISO8601DateFormatter(); formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            return formatter.date(from: date) ?? ISO8601DateFormatter().date(from: date) ?? .distantPast
        }
        return Dictionary(
            uniqueKeysWithValues: [baseDtype, "int8", "int4"].map { dtype in
                guard supported.contains(dtype) else {
                    return (dtype, PrecisionStatSet(kind: kind, dtype: dtype, runner: runner, unavailableReason: "This runner does not support this precision."))
                }
                let matching = results.filter { $0.stats.runner == runner && $0.stats.dtype == dtype && $0.stats.kind == kind }
                let ordered = matching.sorted {
                    let defaultA = $0.stats.availability.provenance?.defaultWorkload == true
                    let defaultB = $1.stats.availability.provenance?.defaultWorkload == true
                    if defaultA != defaultB { return defaultA }
                    let a = $0.stats.availability.provenance?.isHost(chip: chip, gpuCores: gpuCores) == true
                    let b = $1.stats.availability.provenance?.isHost(chip: chip, gpuCores: gpuCores) == true
                    if a != b { return a }
                    return instant($0) > instant($1)
                }
                // Once default-workload evidence exists, never fill its missing figures from a different workload.
                let figures =
                    ordered.contains { $0.stats.availability.provenance?.defaultWorkload == true }
                    ? ordered.filter { $0.stats.availability.provenance?.defaultWorkload == true } : ordered
                guard
                    var value = figures.first(where: { result in
                        result.stats.metrics.values.contains { [.quickCheck, .measured].contains($0.state) }
                    })?.stats ?? figures.first?.stats
                else {
                    // No result: unchecked, and every tier stays offered (an unchecked int8 has no mark).
                    return (dtype, PrecisionStatSet(kind: kind, dtype: dtype, runner: runner))
                }
                // Per metric: this Mac first, then a controlled measurement over a quick check, then the order above (newest).
                for metric in kind.metrics {
                    let candidates = figures.enumerated().filter {
                        [.measured, .quickCheck].contains($0.element.stats.value(metric).state) && $0.element.stats.value(metric).value != nil
                    }
                    let best = candidates.min { a, b in
                        let hostA = a.element.stats.availability.provenance?.isHost(chip: chip, gpuCores: gpuCores) == true
                        let hostB = b.element.stats.availability.provenance?.isHost(chip: chip, gpuCores: gpuCores) == true
                        if hostA != hostB { return hostA }
                        let measuredA = a.element.stats.value(metric).state == .measured, measuredB = b.element.stats.value(metric).state == .measured
                        if measuredA != measuredB { return measuredA }
                        return a.offset < b.offset
                    }
                    if let best { value.set(metric, best.element.stats.value(metric)) }
                }
                // Quality applies across chips; timing retains the host-chip preference above.
                let failure = matching.sorted { instant($0) > instant($1) }.first { failed in
                    guard failed.stats.hasLegacyQualityFailure || failed.stats.qualityCheckPassed == false else { return false }
                    return !matching.contains { newer in
                        guard instant(newer) > instant(failed), newer.mode == failed.mode else { return false }
                        // Current schema-2 outcomes have already been scoped to this recipe/revision by the service.
                        if failed.qualityEvidence == nil, newer.qualityEvidence == nil,
                            failed.stats.qualityCheckPassed != nil, newer.stats.qualityCheckPassed == true,
                            let revision = failed.stats.availability.provenance?.revision,
                            newer.stats.availability.provenance?.revision == revision
                        {
                            return true
                        }
                        // Legacy suites clear a failure only when they cover every recorded case (batch included).
                        guard let pass = newer.qualityEvidence, let rejected = failed.qualityEvidence else { return false }
                        return pass.completePass && pass.suiteID == rejected.suiteID
                            && Set(rejected.caseIDs).isSubset(of: Set(pass.caseIDs))
                            && Set(rejected.answerCaseIDs).isSubset(of: Set(pass.passingAnswerIDs))
                    }
                }
                let runtimeFailure = ordered.first {
                    $0.stats.availability.state == .unavailable && !$0.stats.hasLegacyQualityFailure
                }
                let availabilitySource = runtimeFailure ?? failure ?? figures.first(where: { $0.stats.availability.state != .unavailable })
                if let availabilitySource {
                    value.availability = availabilitySource.stats.availability
                    value.qualityOnlyUnavailable = availabilitySource.stats.qualityOnlyUnavailable
                }
                value.qualityCheckPassed = failure != nil ? false : matching.sorted { instant($0) > instant($1) }.first?.stats.qualityCheckPassed
                // int8 is greyed only when its recorded check failed. Unchecked, or a pass from another recipe/implementation
                // (scoped to unchecked by the service), stays offered without a mark; int4 is always offered (its warning).
                if dtype == "int8", value.qualityCheckPassed == false, runtimeFailure == nil {
                    value.availability = StatValue(state: .unavailable, reason: "Did not pass the quality check", provenance: value.availability.provenance)
                    value.qualityOnlyUnavailable = nil
                }
                return (dtype, value.applyingInt4Policy)
            })
    }
    public static func legacy(_ result: ResultsFile, source: String? = nil) -> PrecisionStatsResult? {
        let qualityFailure = result.recordedQualityFailure
        let qualityEvidence = result.recordedQualityEvidence
        guard let kind = ModelStatKind(rawValue: result.kind), result.recordedAt != nil, !result.stub,
            result.quickCheck || result.eligibleMeasurement || qualityFailure != nil || qualityEvidence.completePass
        else { return nil }
        var stats = PrecisionStatSet(kind: kind, dtype: result.precision, runner: result.runner)
        let state: StatState = result.quickCheck ? .quickCheck : result.eligibleMeasurement ? .measured : .notMeasured
        let provenance = StatProvenance(
            chip: result.chip, date: result.date, method: "Legacy \(result.profile) suite; \(result.verificationLabel)" + (result.workload.map { "; " + $0 } ?? ""), source: source,
            gpuCores: result.gpu_cores)
        stats.availability = qualityFailure.map(StatValue.unavailable) ?? StatValue(state: state, provenance: provenance)
        stats.availability.provenance = provenance
        stats.qualityOnlyUnavailable = qualityFailure != nil ? true : nil
        stats.qualityCheckPassed = qualityFailure != nil ? false : qualityEvidence.completePass ? true : nil
        func mapped() -> PrecisionStatsResult {
            var value = PrecisionStatsResult(model: result.model, mode: result.mode, stats: stats)
            value.qualityEvidence = qualityEvidence
            return value
        }
        // A full run without controlled timing supplies only its quality conclusion, never measured figures.
        guard result.quickCheck || result.eligibleMeasurement else {
            return mapped()
        }
        var mapping: [(StatMetric, String, Double)] = [(.peakMemoryGB, "peak_ram_mb", 0.001)]
        switch kind {
        case .decisions: mapping += [(.millisecondsPerDecision, "latency_s", 1000), (.batchedDecisionsPerSecond, "batch_questions_per_s", 1)]
        case .chat:
            // Legacy peak_ram_mb covers the whole run, not peak at load. Do not relabel it.
            mapping += [(.firstTokenSeconds, "first_token_s", 1), (.tokensPerSecond, "decode_tokens_per_s", 1)]
        case .images: mapping += [(.secondsPerImage, "seconds_per_image", 1)]
        case .voices, .sound, .enhancement:
            if let x = result.metrics["speed_x_realtime"] ?? nil, x > 0 {
                stats.set(.realTimeFactor, StatValue(state: state, value: 1 / x, provenance: provenance))
            }
            if kind == .voices { mapping += [(.firstAudioSeconds, "first_audio_s", 1)] }
        case .audio, .embeddings, .perception, .restoration, .geometry: break
        }
        for (metric, key, scale) in mapping where kind.metrics.contains(metric) {
            if let n = result.metrics[key] ?? nil, n.isFinite && n >= 0 { stats.set(metric, StatValue(state: state, value: n * scale, provenance: provenance)) }
        }
        // accuracy/WER/reference_deviation are not the requested own-16-bit quality metrics.
        return mapped()
    }
}
