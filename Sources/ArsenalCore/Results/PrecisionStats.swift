// SPDX-License-Identifier: MIT
import Foundation

/// Metric identities carry their units. Never reinterpret task accuracy as agreement to 16-bit.
public enum StatMetric: String, Codable, Sendable, CaseIterable {
    case agreementPercent, nextTokenAgreementPercent, nextTokenKL, psnrDB, spectralDistance
    case millisecondsPerDecision, batchedDecisionsPerSecond, firstTokenSeconds, tokensPerSecond
    case secondsPerImage, realTimeFactor, firstAudioSeconds, peakMemoryGB, loadPeakMemoryGB, maximumContextTokens, secondsPerRequest
    case batteryPercentPer1000Decisions, batteryPercentPer100000Tokens, batteryPercentPer100Images
    case batteryPercentPerHourSpeech, batteryPercentPerHourAudio

    public var unit: String {
        switch self {
        case .agreementPercent, .nextTokenAgreementPercent: return "%"
        case .nextTokenKL: return "nats"
        case .psnrDB: return "dB"
        case .spectralDistance: return "relative L2 magnitude spectrum"
        case .millisecondsPerDecision: return "ms/decision"
        case .batchedDecisionsPerSecond: return "decisions/s (batched)"
        case .firstTokenSeconds, .firstAudioSeconds: return "s"
        case .tokensPerSecond: return "tokens/s"
        case .secondsPerImage: return "s/image"
        case .secondsPerRequest: return "s/request"
        case .realTimeFactor: return "compute s/audio s"
        case .peakMemoryGB, .loadPeakMemoryGB: return "GB (10^9 bytes)"
        case .maximumContextTokens: return "tokens"
        case .batteryPercentPer1000Decisions: return "% battery/1,000 decisions"
        case .batteryPercentPer100000Tokens: return "% battery/100k tokens"
        case .batteryPercentPer100Images: return "% battery/100 images"
        case .batteryPercentPerHourSpeech: return "% battery/hour of speech"
        case .batteryPercentPerHourAudio: return "% battery/hour of audio"
        }
    }
}

/// Operation kinds (transcription, embeddings, perception, restoration, 3D) have no served measurement workload: their
/// figures are peak memory and the time of one warm request of the model's own operation, recorded by quick checks.
public enum ModelStatKind: String, Codable, Sendable, CaseIterable {
    case decisions, chat, images, voices, sound, enhancement
    case audio, embeddings, perception, restoration, geometry
    public static let operationKinds: [Self] = [.audio, .embeddings, .perception, .restoration, .geometry]
    /// One figure set for every model whose kind has no generative speed unit.
    public var isOperation: Bool { Self.operationKinds.contains(self) }
    /// Kinds the precision measurement harness can drive (MeasurementWorkload). Operation kinds are filled by quick checks.
    public static var measurable: [Self] { allCases.filter { !$0.isOperation } }
    static func noWorkload(_ kind: Self) -> ManifestError { ManifestError.invalid("No served measurement workload for \(kind.rawValue) models") }
    public var metrics: [StatMetric] {
        switch self {
        case .audio, .embeddings, .perception, .restoration, .geometry: return [.secondsPerRequest, .peakMemoryGB]
        case .decisions: return [.agreementPercent, .millisecondsPerDecision, .batchedDecisionsPerSecond, .peakMemoryGB, .batteryPercentPer1000Decisions]
        case .chat:
            return [
                .nextTokenAgreementPercent, .nextTokenKL, .firstTokenSeconds, .tokensPerSecond, .peakMemoryGB, .loadPeakMemoryGB, .maximumContextTokens,
                .batteryPercentPer100000Tokens
            ]
        case .images: return [.psnrDB, .secondsPerImage, .peakMemoryGB, .batteryPercentPer100Images]
        case .voices: return [.spectralDistance, .realTimeFactor, .firstAudioSeconds, .peakMemoryGB, .batteryPercentPerHourSpeech]
        case .sound, .enhancement: return [.spectralDistance, .realTimeFactor, .peakMemoryGB, .batteryPercentPerHourAudio]
        }
    }
}

public enum StatState: String, Codable, Sendable { case measured, quickCheck, derived, declared, notMeasured, unavailable }
public struct StatProvenance: Codable, Sendable, Equatable {
    public var chip: String
    public var gpuCores: Int?
    public var date: String
    public var method: String
    public var baseline: String?
    public var source: String?
    public var revision: String?
    public var run: String?
    public var defaultWorkload: Bool?
    public init(
        chip: String, date: String, method: String, baseline: String? = nil, source: String? = nil, revision: String? = nil, run: String? = nil, gpuCores: Int? = nil,
        defaultWorkload: Bool? = nil
    ) {
        self.defaultWorkload = defaultWorkload
        self.gpuCores = gpuCores
        self.chip = chip; self.date = date; self.method = method; self.baseline = baseline; self.source = source; self.revision = revision; self.run = run
    }
}

/// State is explicit for every cell; absent values never become zero or a placeholder number.
public struct StatValue: Codable, Sendable, Equatable {
    public var state: StatState
    public var value: Double?
    public var reason: String?
    public var basis: String?
    public var provenance: StatProvenance?
    public init(state: StatState = .notMeasured, value: Double? = nil, reason: String? = nil, provenance: StatProvenance? = nil, basis: String? = nil) {
        self.basis = basis
        self.state = state; self.value = value; self.reason = reason; self.provenance = provenance
    }
    public static func unavailable(_ reason: String) -> Self { Self(state: .unavailable, reason: reason) }
    public var label: String {
        switch state {
        case .notMeasured: return "not measured"
        case .unavailable: return "unavailable: " + (reason ?? "reason missing")
        case .quickCheck: return "quick check"
        case .derived: return "estimate"
        case .declared: return "model config"
        case .measured: return "measured"
        }
    }
}

/// Keys are StatMetric raw values for ordinary JSON/TOML objects. Access with value(_:) in Swift.
/// dtype is the actual fp16/bf16/int8/int4 running, not a second 16-bit variant.
public struct PrecisionStatSet: Codable, Sendable, Equatable {
    public var kind: ModelStatKind
    public var dtype: String
    public var runner: String
    public var availability: StatValue
    /// Numerical quality outcome, independent of whether the precision can run. Nil means unchecked.
    public var qualityCheckPassed: Bool?
    public var qualityOnlyUnavailable: Bool?
    public var qualityIdentity: String?
    public var metrics: [String: StatValue]
    public init(kind: ModelStatKind, dtype: String, runner: String, unavailableReason: String? = nil) {
        self.kind = kind; self.dtype = dtype; self.runner = runner
        availability = unavailableReason.map(StatValue.unavailable) ?? StatValue()
        metrics = Dictionary(uniqueKeysWithValues: kind.metrics.map { ($0.rawValue, unavailableReason.map(StatValue.unavailable) ?? StatValue()) })
    }
    enum CodingKeys: String, CodingKey { case kind, dtype, runner, availability, qualityCheckPassed, qualityOnlyUnavailable, qualityIdentity, metrics }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = try c.decode(ModelStatKind.self, forKey: .kind)
        dtype = try c.decode(String.self, forKey: .dtype)
        runner = try c.decode(String.self, forKey: .runner)
        availability = try c.decode(StatValue.self, forKey: .availability)
        qualityIdentity = try c.decodeIfPresent(String.self, forKey: .qualityIdentity)
        qualityOnlyUnavailable = try c.decodeIfPresent(Bool.self, forKey: .qualityOnlyUnavailable)
        qualityCheckPassed = try c.decodeIfPresent(Bool.self, forKey: .qualityCheckPassed)
        metrics = try c.decode([String: StatValue].self, forKey: .metrics)
        // Older schema-2 chat results reported load peak only. Preserve it without inventing a run peak.
        if kind == .chat, metrics[StatMetric.peakMemoryGB.rawValue] == nil {
            metrics[StatMetric.peakMemoryGB.rawValue] = StatValue()
        }
    }
    public func value(_ metric: StatMetric) -> StatValue { metrics[metric.rawValue] ?? StatValue() }
    public mutating func set(_ metric: StatMetric, _ value: StatValue) { metrics[metric.rawValue] = value }
    /// Old results used availability for quality failures. Preserve those files as evidence,
    /// but offer int4 with a separate quality outcome when reading them.
    static let legacyQualityFailureReasons = ["Did not pass the quality check", "Quality gate failed", "Quality failed"]
    var hasLegacyQualityFailure: Bool {
        availability.state == .unavailable && (qualityOnlyUnavailable == true || Self.legacyQualityFailureReasons.contains(availability.reason ?? ""))
    }
    public var applyingInt4Policy: Self {
        var result = self
        guard dtype == "int4", availability.state == .unavailable,
            hasLegacyQualityFailure
        else { return result }
        result.qualityCheckPassed = false
        result.availability = StatValue(state: availability.provenance == nil ? .notMeasured : .quickCheck, provenance: availability.provenance)
        return result
    }
    public var selectionUnavailableReason: String? {
        let current = applyingInt4Policy.availability
        return current.state == .unavailable ? current.reason : nil
    }
    /// int4's warning, the table's int4 hover and the API's `precisionQualityWarnings["int4"]` (copy: table-vella r2).
    public static let int4QualityWarning = "int4 may lose quality from 4-bit quantization. Its quality check did not pass."
    public static let int4UncheckedWarning = "int4 may lose quality from 4-bit quantization. It has not been quality-checked yet."
    public func validate() throws {
        guard ["fp16", "bf16", "int8", "int4"].contains(dtype), !runner.isEmpty,
            Set(metrics.keys) == Set(kind.metrics.map(\.rawValue))
        else { throw ManifestError.invalid("Invalid per-kind stat set or dtype") }
        for (stat, isAvailability) in metrics.values.map({ ($0, false) }) + [(availability, true)] {
            if [.measured, .quickCheck, .derived, .declared].contains(stat.state) {
                // Availability is a gate state, with no numeric value.
                guard let provenance = stat.provenance, !provenance.chip.isEmpty, !provenance.method.isEmpty,
                    ISO8601DateFormatter().date(from: provenance.date) != nil || Self.fractionalDate(provenance.date) != nil,
                    isAvailability || stat.value.map({ $0.isFinite && $0 >= 0 }) == true
                else {
                    throw ManifestError.invalid("A numeric stat needs a finite value and provenance")
                }
            } else {
                guard stat.value == nil else { throw ManifestError.invalid("An unmeasured/unavailable stat cannot contain a number") }
            }
            if stat.state == .derived && (isAvailability || stat.basis?.isEmpty != false) {
                throw ManifestError.invalid("Estimated stats need a basis and cannot be a gate")
            }
            if let cores = stat.provenance?.gpuCores, cores <= 0 { throw ManifestError.invalid("Invalid GPU core count") }
            if stat.state == .unavailable && (stat.reason?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true) {
                throw ManifestError.invalid("Unavailable stats need a reason")
            }
        }
    }
    private static func fractionalDate(_ text: String) -> Date? {
        let formatter = ISO8601DateFormatter(); formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: text)
    }

}

public enum RunnerFeature: String, Codable, Sendable, CaseIterable {
    case batching, sequentialOnly, streaming
    /// Declared capabilities of the bundled family worker, independent of unmeasured speed.
    public static func native(worker: String?) -> [Self] {
        switch worker {
        case "decisions", "d1": return [.batching]
        case "chat", "voices": return [.sequentialOnly, .streaming]
        case "sound", "enhancement", "flux": return [.sequentialOnly]
        default: return []
        }
    }
    public var description: String {
        switch self {
        case .batching: return "Processes several requests in one batch."
        case .sequentialOnly: return "Processes one request at a time."
        case .streaming: return "Returns output as it is generated."
        }
    }
}
