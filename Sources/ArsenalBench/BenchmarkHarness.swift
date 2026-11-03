// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import ArsenalCore

public struct BenchmarkRequest: Codable, Sendable {
    public var id: String
    public var precision: String
    public var mode: String
    public var suite: String
    public var profile: String
    public var quickCheck: Bool
    enum CodingKeys: String, CodingKey { case id, precision, mode, suite, profile; case quickCheck = "quick_check" }
    public init(id: String, precision: String = "bf16", mode: String = "fast", suite: String = "suite-v1", profile: String = "quick", quickCheck: Bool = false) {
        self.id = id; self.precision = precision; self.mode = mode; self.suite = suite; self.profile = profile
        self.quickCheck = quickCheck
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id); precision = try c.decodeIfPresent(String.self, forKey: .precision) ?? "bf16"
        mode = try c.decodeIfPresent(String.self, forKey: .mode) ?? "fast"; suite = try c.decodeIfPresent(String.self, forKey: .suite) ?? "suite-v1"
        profile = try c.decodeIfPresent(String.self, forKey: .profile) ?? "quick"
        quickCheck = try c.decodeIfPresent(Bool.self, forKey: .quickCheck) ?? false
    }
    public func validate() throws {
        guard ResultValidation.matches(id, "^[a-z0-9][a-z0-9.-]*$"), ["bf16", "fp16", "int8", "int4"].contains(precision),
            ["fast", "exact"].contains(mode), suite == "suite-v1", ["quick", "full"].contains(profile)
        else {
            throw BenchmarkError.invalid("Use a model id, bf16|fp16|int8|int4, fast|exact, suite-v1, and quick|full.")
        }
        if quickCheck && profile != "quick" { throw BenchmarkError.invalid("A quick check must use the quick profile.") }
    }
}

public struct BenchmarkConfiguration: Sendable {
    public var repository: URL
    public var outputDirectory: URL
    public var appBuild: String
    public var workerURL: URL?
    public var stub: Bool
    public var gatePassed: Bool
    public var context: Int?
    public init(
        repository: URL, outputDirectory: URL, appBuild: String = "development", workerURL: URL? = nil,
        stub: Bool = false, gatePassed: Bool = false, context: Int? = nil
    ) {
        self.repository = repository; self.outputDirectory = outputDirectory; self.appBuild = appBuild
        self.workerURL = workerURL; self.stub = stub; self.gatePassed = gatePassed; self.context = context
    }
}

private struct SuiteCase: Codable, Sendable {
    var id: String; var op: String; var params: JSONObject; var labels: JSONObject?; var expected: String?
}
private actor RunObservations {
    var conditions: [ConditionSample] = []
    var peakRAM: Double?
    func append(_ sample: ConditionSample, memory: Double?) {
        conditions.append(sample)
        if let memory, memory > 0 { peakRAM = max(peakRAM ?? 0, memory) }
    }
    func memory(_ memory: Double?) { if let memory, memory > 0 { peakRAM = max(peakRAM ?? 0, memory) } }
    func snapshot() -> ([ConditionSample], Double?) { (conditions, peakRAM) }
}
private actor StreamObservations {
    let start = Date()
    var first: Double?
    var text = ""
    func receive(_ frame: WireFrame) {
        switch frame {
        case .json(let value):
            if value["event"]?.string == "token", let token = value["text"]?.string, !token.isEmpty {
                first = first ?? Date().timeIntervalSince(start); text += token
            }
        case .binary(_, let bytes): if !bytes.isEmpty { first = first ?? Date().timeIntervalSince(start) }
        }
    }
    func values() -> (Double?, String) { (first, text) }
}

public struct BenchmarkHarness: Sendable {
    public var service: any ModelsService
    public var configuration: BenchmarkConfiguration
    public var probe: ConditionProbe
    public init(service: any ModelsService, configuration: BenchmarkConfiguration, probe: ConditionProbe? = nil) {
        self.service = service; self.configuration = configuration
        self.probe = probe ?? (configuration.stub ? .synthetic : .live)
    }

    public func run(_ request: BenchmarkRequest) async throws -> ResultsFile {
        try request.validate()
        if configuration.stub && !(service is StubModelsService) {
            throw BenchmarkError.invalid("Stub benchmarks require StubModelsService; real services cannot bypass the GPU gate.")
        }
        if !configuration.stub && configuration.workerURL == nil {
            throw BenchmarkError.invalid("Real benchmark needs a worker artifact path to record its digest.")
        }
        if !configuration.stub {
            if request.quickCheck && !BenchmarkLocalProof.allowedModels.contains(request.id) {
                throw BenchmarkError.invalid("Quick checks are currently limited to Kokoro 82M.")
            }
            try BenchmarkGate.check(repository: configuration.repository, requireMeasurement: !request.quickCheck)
        }
        guard let row = await service.rows().first(where: { $0.id == request.id }) else { throw BenchmarkError.invalid("Unknown benchmark model \(request.id).") }
        let folder = configuration.repository.appendingPathComponent("Tests/Fixtures/Bench/kinds/\(row.kind)")
        let definition = try Data(contentsOf: folder.appendingPathComponent("suite-v1.toml"))
        let fixtureData = try Data(contentsOf: folder.appendingPathComponent("fixtures-v1.json"))
        let cases = try JSONDecoder().decode([SuiteCase].self, from: fixtureData)
        let suite = BenchmarkSuiteIdentity(id: "arsenal-\(row.kind)-suite-v1", sha256: Self.digest(definition + fixtureData))
        guard !cases.isEmpty else { throw BenchmarkError.invalid("Benchmark suite is empty.") }
        // Idle comes before load: another loaded model must not be hidden in the measured footprint.
        let idle: [ConditionSample]
        do { idle = try await probe.requireIdle(refuseContended: !request.quickCheck) } catch let error as BenchmarkError {
            if case .conditions(let samples) = error {
                let directory = configuration.outputDirectory.appendingPathComponent(request.id)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try JSONEncoder().encode(samples).write(to: directory.appendingPathComponent("idle-refusal.json"), options: .atomic)
            }
            throw error
        }
        try await service.load(id: request.id, precision: request.precision, fast: request.mode == "fast")
        do {
            let result = try await loadedRun(request, row: row, cases: cases, suite: suite, idle: idle)
            try await service.unload(id: request.id)
            return result
        } catch {
            try? await service.unload(id: request.id)
            throw error
        }
    }

    private func loadedRun(
        _ request: BenchmarkRequest, row: ModelRow, cases: [SuiteCase],
        suite: BenchmarkSuiteIdentity, idle: [ConditionSample]
    ) async throws -> ResultsFile {
        let output = configuration.outputDirectory.appendingPathComponent(request.id)
        if configuration.stub && output.path.contains("/models/") {
            throw BenchmarkError.invalid("Stub results may not be written into model folders.")
        }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var selectedVoice: String?
        if row.kind == "voices" {
            let available = try await service.run(id: request.id, op: "voices")["voices"]?.array ?? []
            let ids = available.compactMap { $0["id"]?.string }
            if !ids.isEmpty { selectedVoice = ids.contains("af_heart") ? "af_heart" : ids[0] }
        }
        func params(_ entry: SuiteCase) -> JSONObject {
            var p = entry.params
            if entry.op == "generate" || entry.op == "speak" { p["out_dir"] = .string(output.path) }
            if let selectedVoice { p["voice"] = .string(selectedVoice) }
            return p
        }
        // Warm-up uses exactly the first suite case; no samples or metrics from this call enter the result.
        _ = try await service.run(id: request.id, op: cases[0].op, params: params(cases[0]))
        let observations = RunObservations(), probe = probe, service = service
        let sampler = Task {
            while !Task.isCancelled {
                do {
                    let sample = try await probe.sample("run")
                    let memory = await service.memory().models[request.id]?.footprintMB
                    await observations.append(sample, memory: memory)
                    if configuration.stub { try await Task.sleep(for: .milliseconds(10)) }
                } catch {
                    if !Task.isCancelled {
                        await observations.append(
                            ConditionSample(
                                at: ISO8601DateFormatter().string(from: Date()), phase: "run",
                                cpu: nil, gpu: nil, compositing: nil, note: error.localizedDescription), memory: nil)
                    }
                    break
                }
            }
        }
        var results: [BenchmarkCaseResult] = []
        do {
            for repetition in 0..<(request.profile == "full" ? 10 : 1) {
                for entry in cases {
                    if !configuration.stub { try BenchmarkGate.check(repository: configuration.repository, requireMeasurement: !request.quickCheck) }
                    let meter = StreamObservations(), start = Date()
                    let reply = try await service.run(id: request.id, op: entry.op, params: params(entry), sink: { frame in await meter.receive(frame) })
                    let wall = Date().timeIntervalSince(start), stream = await meter.values()
                    results.append(Self.caseResult(entry, reply: reply, wall: wall, stream: stream, repetition: repetition, voice: selectedVoice))
                    await observations.memory(await service.memory().models[request.id]?.footprintMB)
                    let contended = await observations.snapshot().0.contains(where: { !$0.valid })
                    if !request.quickCheck && contended { break }
                }
                let contended = await observations.snapshot().0.contains(where: { !$0.valid })
                if !request.quickCheck && contended { break }
            }
            let contended = await observations.snapshot().0.contains(where: { !$0.valid })
            if row.kind == "decisions", request.quickCheck || !contended {
                let start = Date()
                let questions = cases.reduce(into: JSONObject()) { value, entry in
                    value.merge(entry.params["questions"]?.object ?? [:], uniquingKeysWith: { first, _ in first })
                }
                let reply = try await service.run(
                    id: request.id, op: "judge", params: ["items": .array(cases.map { $0.params["state"] ?? .null }), "questions": .object(questions)])
                let seconds = Date().timeIntervalSince(start)
                let count = Double(cases.count * questions.count)
                results.append(.init(id: "batch", wall: seconds, metrics: ["batch_questions_per_s": count / max(seconds, 1e-9)], output: reply["error"]?.string))
            }
            // Let an in-flight three-second sample finish so even very short workloads have recorded telemetry.
            if !configuration.stub { try await Task.sleep(for: .seconds(3.5)) }
            sampler.cancel(); await sampler.value
        } catch { sampler.cancel(); await sampler.value; throw error }
        var (during, peak) = await observations.snapshot()
        if during.isEmpty { during = [try await probe.sample("run")] }
        var metrics = Self.aggregate(results, kind: row.kind)
        metrics["peak_ram_mb"] = .some(peak)
        if row.kind == "chat" { metrics["context"] = .some(configuration.context.map(Double.init)) }
        let conditions = BenchmarkConditions(
            locked: !configuration.stub && !request.quickCheck, idle: idle, run: during,
            note: configuration.stub
                ? "Synthetic stub; not a model measurement."
                : (request.quickCheck ? "Quick check under gpulock; gpulock quick proof; not controlled measurement, never ranked or Measured. " : "")
                    + "Speed and quality only; energy not measured. RAM is sampled worker footprint, a lower bound on transient peak.")
        let workerDigest: String
        if let worker = configuration.workerURL {
            workerDigest = Self.digest(try Data(contentsOf: worker))
        } else if configuration.stub {
            workerDigest = Self.digest(Data("StubModelsService".utf8))
        } else {
            throw BenchmarkError.invalid("Real benchmark needs a worker artifact path to record its digest.")
        }
        let result = ResultsFile(
            model: request.id, kind: row.kind, chip: try Self.chip(),
            osBuild: try BenchCommand.output("/usr/bin/sw_vers", ["-buildVersion"]).trimmingCharacters(in: .whitespacesAndNewlines),
            appBuild: configuration.appBuild, runner: row.runner, workerDigest: workerDigest,
            precision: request.precision, mode: request.mode, suite: suite, profile: request.profile,
            stub: configuration.stub, gatePassed: !configuration.stub && configuration.gatePassed,
            conditions: conditions, metrics: metrics, cases: results)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(result); try ResultValidation.validate(data: data)
        try data.write(to: output.appendingPathComponent("\(result.chip).json"), options: .atomic)
        return result
    }

    private static func caseResult(
        _ entry: SuiteCase, reply: JSONObject, wall: Double,
        stream: (Double?, String), repetition: Int, voice: String?
    ) -> BenchmarkCaseResult {
        var m: [String: Double?] = [:]
        var output: String?
        switch entry.op {
        case "speak":
            let seconds = reply["gen_s"]?.number ?? wall
            m["speed_x_realtime"] = .some(reply["duration_s"]?.number.map { $0 / max(seconds, 1e-9) })
            m["first_audio_s"] = .some(reply["first_audio_s"]?.number ?? stream.0)
            m["_audio_s"] = .some(reply["duration_s"]?.number)
            m["_gen_s"] = seconds
            output = voice.map { "voice=\($0)" }
        case "decide":
            let answers = reply["answers"]?.object ?? [:]
            var correct = 0.0
            for (key, expected) in entry.labels ?? [:] {
                let answer = answers[key]
                if let flag = expected.bool, let number = answer?["noul"]?.number, (number >= 0.5) == flag {
                    correct += 1
                } else if let label = expected.string, answer?["choice"]?.string == label {
                    correct += 1
                } else if let label = expected.number, let score = answer?["score"]?.number, score.rounded() == label {
                    correct += 1
                }
            }
            m["accuracy"] = correct / Double(max(1, entry.labels?.count ?? 0))
            m["latency_s"] = wall / Double(max(1, entry.params["questions"]?.object?.count ?? 1))
        case "generate":
            let count = reply["images"]?.array?.count ?? 0
            m["seconds_per_image"] = .some(count > 0 ? wall / Double(count) : nil)
            output = reply["images"]?.array?.first?["path"]?.string
        case "chat":
            let text = reply["text"]?.string ?? stream.1
            output = text
            m["first_token_s"] = .some(reply["ttft_s"]?.number ?? stream.0)
            m["decode_tokens_per_s"] = .some(reply["tokens_per_s"]?.number)
            if let seconds = reply["prefill_s"]?.number, seconds > 0 {
                m["prefill_tokens_per_s"] = .some(reply["usage"]?["prompt_tokens"]?.number.map { $0 / seconds })
            } else {
                m["prefill_tokens_per_s"] = .some(nil)
            }
            m["quality_accuracy"] = normalize(text) == normalize(entry.expected ?? "") ? 1 : 0
        default: break
        }
        return .init(id: "\(entry.id)-\(repetition)", wall: wall, metrics: m, output: output)
    }
    public static func aggregate(_ results: [BenchmarkCaseResult], kind: String) -> [String: Double?] {
        let keys = Set(results.flatMap { $0.metrics.keys }.filter { !$0.hasPrefix("_") })
        var metrics: [String: Double?] = [:]
        for key in keys {
            let numbers = results.compactMap { $0.metrics[key] ?? nil }.sorted()
            if numbers.isEmpty {
                metrics[key] = .some(nil)
            } else if ["accuracy", "quality_accuracy"].contains(key) {
                metrics[key] = numbers.reduce(0, +) / Double(numbers.count)
            } else {
                let mid = numbers.count / 2; metrics[key] = numbers.count % 2 == 0 ? (numbers[mid - 1] + numbers[mid]) / 2 : numbers[mid]
            }
        }
        if kind == "voices" {
            metrics["round_trip_wer"] = .some(nil)
            let audio = results.compactMap { $0.metrics["_audio_s"] ?? nil }
            let generation = results.compactMap { $0.metrics["_gen_s"] ?? nil }.reduce(0, +)
            metrics["speed_x_realtime"] = .some(audio.count == results.count && generation > 0 ? audio.reduce(0, +) / generation : nil)
        }
        if kind == "images" { metrics["reference_deviation"] = .some(nil) }
        return metrics
    }
    static func normalize(_ text: String) -> String { text.lowercased().filter { $0.isLetter || $0.isNumber } }
    public static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    public static func chip() throws -> String {
        HostChip.current
    }
}
