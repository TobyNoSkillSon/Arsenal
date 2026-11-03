// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

public enum BenchmarkError: Error, LocalizedError, Sendable {
    case invalid(String)
    case conditions([ConditionSample])
    public var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        case .conditions(let samples):
            guard let sample = samples.last else { return "Benchmark window contaminated; not started." }
            func number(_ value: Double?) -> String { value.map { String(format: "%.1f%%", $0) } ?? "unavailable" }
            return
                "Benchmark window contaminated: other CPU \(number(sample.other_cpu_pct)), GPU \(number(sample.other_gpu_pct)), compositing \(number(sample.compositing_gpu_pct)); not started."
        }
    }
}

public enum ResultValidation {
    @discardableResult public static func validate(data: Data) throws -> ResultsFile {
        let root = try WireJSON.decode(data)
        let required: Set<String> = [
            "schema", "model", "kind", "chip", "os_build", "app_build", "runner", "worker_digest", "precision", "mode", "suite", "profile", "date", "stub", "gate_passed",
            "conditions", "metrics", "cases"
        ]
        let optional: Set<String> = ["gpu_cores", "workload"]
        let keys = Set(root.keys)
        guard required.isSubset(of: keys), keys.subtracting(required).isSubset(of: optional) else {
            throw BenchmarkError.invalid("Benchmark result has missing or unknown top-level fields.")
        }
        let r = try JSONDecoder().decode(ResultsFile.self, from: data)
        guard r.schema == 1, ["voices", "decisions", "images", "chat"].contains(r.kind),
            ["bf16", "fp16", "int8", "int4"].contains(r.precision), ["fast", "exact"].contains(r.mode),
            ["quick", "full"].contains(r.profile), r.suite.version == 1,
            matches(r.model, "^[a-z0-9][a-z0-9.-]*$"), matches(r.chip, "^[a-z0-9]+(?:-[a-z0-9]+)*$"),
            matches(r.worker_digest, "^[a-f0-9]{64}$"), matches(r.suite.sha256, "^[a-f0-9]{64}$"),
            !r.os_build.isEmpty, !r.app_build.isEmpty, !r.runner.isEmpty,
            ["arsenal-\(r.kind)-suite-v1", "mobius-\(r.kind)-suite-v1"].contains(r.suite.id), // legacy-compat: suite ids recorded before the rename
            ISO8601DateFormatter().date(from: r.date) != nil,
            !r.cases.isEmpty, r.conditions.cpu_limit_pct == 250, r.conditions.gpu_limit_pct == 10,
            r.conditions.sample_interval_s == 3, r.conditions.energy == "not_measured",
            r.conditions.idle_samples.count >= 3, !r.conditions.run_samples.isEmpty,
            r.metrics.keys.contains("peak_ram_mb"),
            r.conditions.idle_samples.allSatisfy({ $0.phase == "idle" }),
            r.conditions.run_samples.allSatisfy({ $0.phase == "run" })
        else { throw BenchmarkError.invalid("Benchmark result has invalid identity or missing conditions.") }
        let metricKeys: Set<String> = [
            "speed_x_realtime", "first_audio_s", "round_trip_wer", "accuracy", "latency_s", "batch_questions_per_s", "seconds_per_image", "reference_deviation",
            "prefill_tokens_per_s", "decode_tokens_per_s", "first_token_s", "quality_accuracy", "peak_ram_mb", "context"
        ]
        guard Set(r.metrics.keys).isSubset(of: metricKeys) else { throw BenchmarkError.invalid("Benchmark result contains an unknown metric.") }
        for sample in r.conditions.idle_samples + r.conditions.run_samples {
            let recomputed = ConditionSample(
                at: sample.at, phase: sample.phase, cpu: sample.other_cpu_pct,
                gpu: sample.other_gpu_pct, compositing: sample.compositing_gpu_pct)
            guard sample.valid == recomputed.valid, ["idle", "run"].contains(sample.phase) else {
                throw BenchmarkError.invalid("Benchmark condition validity disagrees with its samples.")
            }
        }
        let contaminated = (r.conditions.idle_samples + r.conditions.run_samples).contains { !$0.valid }
        guard r.conditions.contaminated == contaminated, r.conditions.contended == contaminated else {
            throw BenchmarkError.invalid("Benchmark contamination flag disagrees with its samples.")
        }
        let values = Array(r.metrics.values) + r.cases.flatMap { Array($0.metrics.values) }
        guard values.compactMap({ $0 }).allSatisfy({ $0.isFinite && $0 >= 0 }),
            r.cases.allSatisfy({ $0.wall_s.isFinite && $0.wall_s >= 0 })
        else {
            throw BenchmarkError.invalid("Benchmark metrics must be finite and nonnegative or null.")
        }
        for key in ["accuracy", "quality_accuracy", "round_trip_wer"] {
            if let v = r.metrics[key] ?? nil, v > 1 { throw BenchmarkError.invalid("Benchmark \(key) must be a fraction between zero and one.") }
        }
        return r
    }
    static func matches(_ value: String, _ pattern: String) -> Bool { value.range(of: pattern, options: .regularExpression) != nil }
}

public enum ResearchLogValidation {
    /// TOML (+++) or the portable YAML subset (---): one top-level field per line.
    public static func validate(markdown: String) throws {
        let lines = markdown.components(separatedBy: .newlines)
        guard let fence = lines.first, ["+++", "---"].contains(fence),
            let end = lines.dropFirst().firstIndex(of: fence)
        else {
            throw BenchmarkError.invalid("Research log needs TOML or YAML front matter.")
        }
        let separator: Character = fence == "+++" ? "=" : ":"
        var fields: [String: String] = [:]
        for line in lines[1..<end] {
            guard !line.trimmingCharacters(in: .whitespaces).hasPrefix("#"), let split = line.firstIndex(of: separator) else { continue }
            let key = line[..<split].trimmingCharacters(in: .whitespaces)
            guard fields[key] == nil else { throw BenchmarkError.invalid("Research log repeats \(key).") }
            fields[key] = line[line.index(after: split)...].trimmingCharacters(in: .whitespaces)
        }
        for key in ["hypothesis", "change", "measurement", "verdict", "numbers", "links"] {
            guard let value = fields[key], !["", "\"\"", "''", "null"].contains(value) else {
                throw BenchmarkError.invalid("Research log is missing \(key).")
            }
        }
        var parsed: [String: TOMLValue]
        if fence == "+++" {
            var parser = TOMLParser(lines[1..<end].joined(separator: "\n")); parsed = try parser.parse()
        } else {
            // The supported YAML subset is scalar fields plus TOML-compatible inline arrays.
            var text = ""
            for (key, raw) in fields {
                if raw.hasPrefix("[") || raw.hasPrefix("\"") || raw.hasPrefix("'") {
                    text += "\(key) = \(raw)\n"
                } else {
                    text += "\(key) = \(String(decoding: try JSONEncoder().encode(raw), as: UTF8.self))\n"
                }
            }
            var parser = TOMLParser(text); parsed = try parser.parse()
        }
        for key in ["hypothesis", "change", "measurement"] {
            guard case .string(let value) = parsed[key], !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw BenchmarkError.invalid("Research log \(key) must be nonempty text.")
            }
        }
        guard case .string(let verdict) = parsed["verdict"], case .array(let links) = parsed["links"],
            links.allSatisfy({
                if case .string = $0 { return true }; return false
            })
        else {
            throw BenchmarkError.invalid("Research log needs a text verdict and an inline array of string links.")
        }
        switch parsed["numbers"] {
        case .table, .array, .string: break
        default: throw BenchmarkError.invalid("Research log numbers must be an object, array, or text explaining no measurement.")
        }
        guard ["kept", "rejected", "inconclusive"].contains(verdict),
            lines.dropFirst(end + 1).joined().trimmingCharacters(in: .whitespacesAndNewlines).count > 0
        else {
            throw BenchmarkError.invalid("Research log needs a valid verdict, inline links array, and narrative.")
        }
    }
}

public enum BenchmarkRanking {
    /// Never compares different suite hashes, profiles, precision or mode.
    public static func build(results: [ResultsFile], chip: String) -> [String: [String]] {
        var rankings: [String: [String]] = [:]
        let groups = Dictionary(grouping: results.filter { $0.chip == chip && $0.eligibleMeasurement }) { $0.model }
        for (model, rows) in groups {
            let latest = rows.sorted(by: ResultsFile.newestFirst).first!
            let comparable = rows.filter { $0.suite == latest.suite && $0.profile == latest.profile && $0.precision == latest.precision && $0.mode == latest.mode }
            let byRunner = Dictionary(grouping: comparable, by: \.runner).compactMapValues { $0.sorted(by: ResultsFile.newestFirst).first }
            rankings[model] = byRunner.values.sorted {
                let a = score($0), b = score($1)
                return a == b ? $0.runner < $1.runner : a < b
            }.map(\.runner)
        }
        return rankings
    }
    private static func score(_ r: ResultsFile) -> Double {
        guard let key = ["voices": "speed_x_realtime", "decisions": "latency_s", "images": "seconds_per_image", "chat": "decode_tokens_per_s"][r.kind] else { return .infinity }
        guard let value = r.metrics[key] ?? nil, value > 0 else { return .infinity }
        return ["voices", "chat"].contains(r.kind) ? 1 / value : value
    }
}
