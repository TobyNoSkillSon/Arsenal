// SPDX-License-Identifier: MIT
import Foundation

public struct ConditionSample: Codable, Sendable {
    public var at: String
    public var phase: String
    public var other_cpu_pct: Double?
    public var other_gpu_pct: Double?
    public var compositing_gpu_pct: Double?
    public var valid: Bool
    public var note: String?

    public init(at: String, phase: String, cpu: Double?, gpu: Double?, compositing: Double?, note: String? = nil) {
        self.at = at; self.phase = phase; other_cpu_pct = cpu; other_gpu_pct = gpu
        compositing_gpu_pct = compositing; self.note = note
        valid =
            cpu.map { $0.isFinite && $0 >= 0 && $0 < 250 } == true && gpu.map { $0.isFinite && $0 >= 0 && $0 < 10 } == true
            && compositing.map { $0.isFinite && $0 >= 0 && $0 <= 15 } == true
    }

    enum CodingKeys: String, CodingKey { case at, phase, other_cpu_pct, other_gpu_pct, compositing_gpu_pct, valid, note }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(at, forKey: .at); try c.encode(phase, forKey: .phase)
        try c.encode(other_cpu_pct, forKey: .other_cpu_pct); try c.encode(other_gpu_pct, forKey: .other_gpu_pct)
        try c.encode(compositing_gpu_pct, forKey: .compositing_gpu_pct); try c.encode(valid, forKey: .valid)
        try c.encodeIfPresent(note, forKey: .note)
    }
}

public struct BenchmarkConditions: Codable, Sendable {
    public var measurement_lock: Bool
    public var contaminated: Bool
    public var contended: Bool
    public var cpu_limit_pct: Double = 250
    public var gpu_limit_pct: Double = 10
    public var sample_interval_s: Double = 3
    public var idle_samples: [ConditionSample]
    public var run_samples: [ConditionSample]
    public var energy: String = "not_measured"
    public var note: String?

    public init(locked: Bool, idle: [ConditionSample], run: [ConditionSample], note: String? = nil) {
        measurement_lock = locked; idle_samples = idle; run_samples = run; self.note = note
        contaminated = idle.contains { !$0.valid } || run.contains { !$0.valid }
        contended = contaminated
    }
}

public struct BenchmarkSuiteIdentity: Codable, Sendable, Equatable {
    public var id: String
    public var version: Int
    public var sha256: String
    public init(id: String, version: Int = 1, sha256: String) { self.id = id; self.version = version; self.sha256 = sha256 }
}

public struct BenchmarkCaseResult: Codable, Sendable {
    public var id: String
    public var wall_s: Double
    public var metrics: [String: Double?]
    public var output: String?
    public init(id: String, wall: Double, metrics: [String: Double?], output: String? = nil) {
        self.id = id; wall_s = wall; self.metrics = metrics; self.output = output
    }
}

public struct ResultsFile: Codable, Sendable {
    public var checks: ModelChecksStatus?
    public var schema: Int = 1
    public var model: String
    public var kind: String
    public var chip: String
    public var gpu_cores: Int?
    public var workload: String?
    public var os_build: String
    public var app_build: String
    public var runner: String
    public var worker_digest: String
    public var precision: String
    public var mode: String
    public var suite: BenchmarkSuiteIdentity
    public var profile: String
    public var date: String
    public var stub: Bool
    public var gate_passed: Bool
    public var conditions: BenchmarkConditions
    public var metrics: [String: Double?]
    public var cases: [BenchmarkCaseResult]

    public init(
        model: String, kind: String, chip: String, osBuild: String, appBuild: String,
        runner: String, workerDigest: String, precision: String, mode: String,
        suite: BenchmarkSuiteIdentity, profile: String, stub: Bool, gatePassed: Bool,
        conditions: BenchmarkConditions, metrics: [String: Double?], cases: [BenchmarkCaseResult]
    ) {
        self.model = model; self.kind = kind; self.chip = chip; os_build = osBuild; app_build = appBuild
        self.runner = runner; worker_digest = workerDigest; self.precision = precision; self.mode = mode
        self.suite = suite; self.profile = profile; date = ISO8601DateFormatter().string(from: Date())
        self.stub = stub; gate_passed = gatePassed; self.conditions = conditions; self.metrics = metrics; self.cases = cases
    }

    /// Parse instants rather than ordering ISO8601 spellings (offsets and fractions vary).
    public var recordedAt: Date? {
        // Foundation accepts trailing junk and normalizes impossible days; reject both before parsing.
        let syntax = #"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:?[0-9]{2})\z"#
        guard date.range(of: syntax, options: .regularExpression) != nil else { return nil }
        let fields = date.prefix(19).split { "-T:".contains($0) }.compactMap { Int($0) }
        guard fields.count == 6, fields[0] > 0, (1...12).contains(fields[1]), (0...23).contains(fields[3]),
            (0...59).contains(fields[4]), (0...59).contains(fields[5])
        else { return nil }
        var calendar = Calendar(identifier: .iso8601)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        guard let month = calendar.date(from: DateComponents(year: fields[0], month: fields[1], day: 1)),
            let days = calendar.range(of: .day, in: .month, for: month), days.contains(fields[2])
        else { return nil }
        if !date.hasSuffix("Z") {
            guard let zone = date.split(whereSeparator: { $0 == "+" || $0 == "-" }).last else { return nil }
            let digits = zone.filter(\.isNumber)
            guard digits.count == 4, let hours = Int(digits.prefix(2)), let minutes = Int(digits.suffix(2)), hours <= 23, minutes <= 59 else { return nil }
        }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: date)
            ?? {
                formatter.formatOptions = [.withInternetDateTime]
                return formatter.date(from: date)
            }()
    }

    /// Invalid dates sort first for evidence gates: an undated failure cannot be hidden by a pass.
    public static func newestFirst(_ left: ResultsFile, _ right: ResultsFile) -> Bool {
        switch (left.recordedAt, right.recordedAt) {
        case (nil, nil): return false
        case (nil, _): return true
        case (_, nil): return false
        case let (left?, right?): return left > right
        }
    }

    public var eligibleMeasurement: Bool {
        recordedAt != nil && !stub && gate_passed && conditions.measurement_lock && !conditions.contaminated && conditions.idle_samples.count >= 3
            && !conditions.run_samples.isEmpty
            && (conditions.idle_samples + conditions.run_samples).allSatisfy(\.valid)
    }

    /// A quick, uncontrolled run of the suite (`profile` quick, no measurement lock): rough figures that are shown,
    /// labelled "quick check", but never ranked or Measured. Classified by its fields, not by its note's wording.
    public var quickCheck: Bool { !stub && profile == "quick" && !conditions.measurement_lock }

    /// Legacy gate_passed also meant "not qualified": the suite CLI left it false by default,
    /// and image gates included unfinished reference qualification. Neither records a quality failure.
    /// Keep explicit parity/task failures, and explain a rejected gate with its recorded answer cases.
    public var recordedQualityFailure: String? {
        if checks?.stale == false, checks?.parity == .fail { return "The recorded reference parity check failed." }
        if checks?.stale == false, checks?.task == .fail { return "The recorded task quality check failed." }
        guard !gate_passed else { return nil }
        let answerKey: String
        switch kind {
        case "decisions": answerKey = "accuracy"
        case "chat": answerKey = "quality_accuracy"
        default: return nil
        }
        func mismatch(_ metrics: [String: Double?]) -> Bool {
            guard let value = metrics[answerKey] ?? nil else { return false }
            return value.isFinite && value >= 0 && value < 1
        }
        let failed = cases.filter { mismatch($0.metrics) }.map(\.id)
        if !failed.isEmpty { return "Recorded answer mismatch: " + failed.joined(separator: ", ") + "." }
        if mismatch(metrics) { return "The recorded answer accuracy was below 100%." }
        return nil
    }

    /// Complete passing coverage, independent of whether timings qualified for measurement.
    var recordedQualityEvidence: RecordedQualityEvidence {
        let currentChecks = checks.flatMap { $0.stale ? nil : $0 }
        let answerKey = kind == "decisions" ? "accuracy" : kind == "chat" ? "quality_accuracy" : nil
        let qualityFailure = recordedQualityFailure
        let answersPass =
            answerKey.map { key in
                let answerCases = cases.filter { $0.metrics.keys.contains(key) }
                return metrics[key] == 1 && !answerCases.isEmpty && answerCases.allSatisfy { $0.metrics[key] == 1 }
            } ?? false
        let checksPass = currentChecks?.parity == .pass && currentChecks?.task == .pass
        return RecordedQualityEvidence(
            completePass: qualityFailure == nil && (checksPass || answersPass), suiteID: suite.id, caseIDs: cases.map(\.id),
            answerCaseIDs: answerKey.map { key in cases.filter { $0.metrics.keys.contains(key) }.map(\.id) } ?? [],
            passingAnswerIDs: answerKey.map { key in cases.filter { $0.metrics[key] == 1 }.map(\.id) } ?? [])
    }

    public var verificationLabel: String {
        if stub { return "stub" }
        if quickCheck { return "quick check" }
        if conditions.contaminated { return "contaminated" }
        if eligibleMeasurement { return "Measured" }
        return "Unverified"
    }

    public func tableMetrics(columns declaredColumns: [KindColumn]? = nil) -> [String: String] {
        func value(_ key: String, _ format: String, scale: Double = 1) -> String {
            guard let wrapped = metrics[key], let v = wrapped, v.isFinite else { return "—" }
            return String(format: format, v * scale)
        }
        var columns = ["peak_ram": value("peak_ram_mb", "%.2f GB", scale: 0.001)]
        switch kind {
        case "voices":
            columns["speed"] = value("speed_x_realtime", "%.1f×")
            columns["first_audio"] = value("first_audio_s", "%.0f ms", scale: 1000)
            columns["roundtrip_wer"] = value("round_trip_wer", "%.1f%%", scale: 100)
        case "decisions":
            columns["accuracy"] = value("accuracy", "%.1f%%", scale: 100)
            columns["latency"] = value("latency_s", "%.0f ms", scale: 1000)
        case "images": columns["seconds_per_image"] = value("seconds_per_image", "%.1f s · 512×512")
        case "chat":
            columns["context"] = value("context", "%.0f")
            columns["tokens_per_s"] = value("decode_tokens_per_s", "%.1f tok/s")
            columns["first_token"] = value("first_token_s", "%.0f ms", scale: 1000)
        default:
            for column in declaredColumns ?? [] {
                columns[column.id] = value(column.id, "%.3g" + (column.unit.isEmpty ? "" : " " + column.unit))
            }
        }
        return columns
    }
}

public enum MeasurementBadge: String, Codable, Sendable {
    case measured = "Measured"
    case anotherChip = "Measured on another chip"
    case unverified = "Unverified"

    public static func derive(results: [ResultsFile], chip: String) -> Self {
        let qualified = results.filter(\.eligibleMeasurement)
        if qualified.contains(where: { $0.chip == chip }) { return .measured }
        return qualified.isEmpty ? .unverified : .anotherChip
    }
}

/// Coverage is carried through the legacy adapter, rather than inferred from a timing availability badge.
struct RecordedQualityEvidence: Codable, Sendable, Equatable {
    var completePass: Bool
    var suiteID: String
    /// Every recorded case, timing-only ones included.
    var caseIDs: [String]
    /// Cases that carry an answer, and those whose answer was right.
    var answerCaseIDs: [String]
    var passingAnswerIDs: [String]
}
