// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

actor FixtureMeasurementRunner: PrecisionMeasurementRunner {
    nonisolated let fixture: Bool
    var prepares: [(String, String)] = []
    var calls = 0
    var dtype = "bf16"
    var downloads: Int64 = 0
    var failingBracket = false
    var audioMismatch = false
    var memorySkip = false
    func skipInt8ForMemory() { memorySkip = true }
    func changeAudioCount() { audioMismatch = true }
    func failBrackets() { failingBracket = true }
    init(fixture: Bool = true, downloads: Int64 = 0) { self.fixture = fixture; self.downloads = downloads }
    func prepare(dtype: String, sourceDtype: String) throws -> MeasurementPreparation {
        prepares.append((dtype, sourceDtype)); self.dtype = dtype; calls += 1
        if memorySkip && dtype == "int8" { throw MeasurementMemorySkip(needMB: 112_674) }
        if failingBracket && dtype == sourceDtype && prepares.count > 1 { throw ManifestError.invalid("fixture baseline bracket failed") }
        return MeasurementPreparation(sourceDtype: sourceDtype, dtype: dtype, downloadedBytes: downloads, loadPeakBytes: 2_000_000_000)
    }
    func quality(kind: ModelStatKind) -> MeasurementOutput {
        calls += 1
        switch kind {
        case .chat: return MeasurementOutput(fixtureID: "fixed-32-token-text-v1", logits: [[1, 0], [0, 1]])
        case .decisions: return MeasurementOutput(fixtureID: "fixed-decisions-v1", decisions: dtype == "int4" ? [1, 2] : [1, 1])
        case .images: return MeasurementOutput(fixtureID: "fixed-1024-default-seed-v1", samples: Array(repeating: 0.5, count: 1024 * 1024 * 3), width: 1024, height: 1024)
        case .voices, .sound, .enhancement:
            return MeasurementOutput(
                fixtureID: "fixed-audio-seed-v1", samples: (0..<(audioMismatch && dtype == "int4" ? 512 : 256)).map { sin(Double($0) * 0.1) }, sampleRate: 24000)
        case .audio, .embeddings, .perception, .restoration, .geometry: return MeasurementOutput(fixtureID: "no-served-workload")
        }
    }
    func warmup(kind: ModelStatKind, steps: Int) { calls += 1 }
    func window(kind: ModelStatKind, steps: Int, batched: Bool) -> MeasurementWindow {
        calls += 1
        return MeasurementWindow(elapsedSeconds: 2, completedUnits: kind == .images ? 1 : 32, firstOutputSeconds: 0.1, peakBytes: 3_000_000_000, batchSize: batched ? 8 : 1)
    }
    func gate(kind: ModelStatKind, dtype: String, metrics: [String: Double]) -> MeasurementGate {
        let failed = dtype == "int4" && kind == .decisions
        return MeasurementGate(passed: !failed, policy: "Fixture argmax equality", unavailableReason: failed ? "Changed fixture answer" : nil)
    }
    func unload() { calls += 1 }
}

struct MeasurementTests {
    func queue() -> PendingBenchmarkQueue { PendingBenchmarkQueue(support: FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-stats-" + UUID().uuidString)) }
    @Test func chatWindowUsesSeededOrdinarySamplingDefaults() throws {
        let request = try MeasurementWorkload.request(kind: .chat, model: "fixture", batched: false, count: 1, directory: URL(fileURLWithPath: "/unused"))
        #expect(request.op == "chat" && request.params["seed"]?.number == 0)
        #expect(request.params["temperature"] == nil && request.params["top_p"] == nil)
        #expect(request.params["max_tokens"]?.number == 32)
        #expect(MeasurementWorkload.identity(kind: .chat) == "arsenal-served-chat-v3-seed0-defaults")
    }
    /// The benchmark's enhancement request must pass every shipped enhancement model's own `enhance` schema (AudioSR once refused `stream`).
    @Test func enhancementWorkloadMatchesEveryShippedEnhancementSchema() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-workload-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let request = try MeasurementWorkload.request(kind: .enhancement, model: "fixture", batched: false, count: 1, directory: directory)
        let params = try JSONSerialization.jsonObject(with: WireJSON.encode(request.params))
        var checked: [String] = []
        for id in ["audiosr-basic", "re-use"] {
            let folder = shippedOrRetiredFolder(repo, id)
            let model = try ModelManifest.load(folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
            #expect(model.kind == "enhancement")
            let op = try #require(model.op?.first { $0.name == request.op }, "\(id)")
            let schema = try Data(contentsOf: folder.appendingPathComponent(op.input))
            #expect(try ModelSchemaValidation.validate(value: params, schemaData: schema).isEmpty, "\(id)")
            checked.append(id)
        }
        #expect(checked.count == 2)
    }
    @Test func readyFixtureGetPlanHasNoDownloadThroughEitherCallPath() async throws {
        let stub = StubModelsService(rows: [ModelRow(id: "fixture", name: "Fixture", kind: "voices", states: ["bf16": PrecisionState(state: .ready)])])
        let service: any ModelsService = stub
        #expect(try await stub.getPlan(id: "fixture", dtype: "bf16").totalDownloadBytes == 0)
        #expect(try await service.getPlan(id: "fixture", dtype: "bf16").requiresDownload == false)
    }
    @Test func competingQueueDrainerIsRefusedAndReleaseAllowsNext() throws {
        let queue = queue(), support = queue.file.deletingLastPathComponent().deletingLastPathComponent()
        defer { try? FileManager.default.removeItem(at: support) }
        var lease: BenchmarkRunLease? = try BenchmarkRunLease(support: support)
        #expect(throws: (any Error).self) { try BenchmarkRunLease(support: support) }
        withExtendedLifetime(lease) {}
        lease = nil
        let next = try BenchmarkRunLease(support: support)
        withExtendedLifetime(next) {}
    }
    @Test func noGoQueuesBeforeAnyRunnerCall() async throws {
        let runner = FixtureMeasurementRunner(fixture: false), queue = queue()
        defer { try? FileManager.default.removeItem(at: queue.file.deletingLastPathComponent().deletingLastPathComponent()) }
        let result = try await PrecisionMeasurementHarness.run(
            model: "fixture", kind: .chat, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
            runner: runner, probe: .unavailable, method: .init(chip: "fixture", source: "fixture"), queue: queue)
        #expect(result.isEmpty)
        #expect(await runner.calls == 0)
        #expect(try queue.read().map(\.dtype) == ["bf16", "int8", "int4"])
        _ = try await PrecisionMeasurementHarness.run(
            model: "fixture", kind: .chat, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
            runner: runner, probe: .unavailable, method: .init(chip: "fixture", source: "fixture"), queue: queue)
        #expect(try queue.read().count == 3)
    }
    @Test func allKindsAllPrecisionsUseOwn16BitAndNeverEarnMeasured() async throws {
        for kind in ModelStatKind.measurable {
            let runner = FixtureMeasurementRunner(), queue = queue()
            defer { try? FileManager.default.removeItem(at: queue.file.deletingLastPathComponent().deletingLastPathComponent()) }
            let probe = MeasurementProbe(capacityJoules: 360000, begin: {}, end: { CPUGPUEnergy(cpuJoules: 2, gpuJoules: 8) })
            let results = try await PrecisionMeasurementHarness.run(
                model: "fixture", kind: kind, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
                runner: runner, probe: probe, method: .init(chip: "fixture", controlled: true, source: "fixture"), queue: queue)
            #expect(results.count == 3)
            #expect(await runner.prepares.map { $0.1 } == Array(repeating: "bf16", count: 5))
            #expect(results.allSatisfy { !$0.stats.metrics.values.contains { $0.state == .measured } })
            for result in results { try result.stats.validate() }
            if kind == .decisions { #expect(results.last?.stats.availability.reason == "Changed fixture answer") }
            if kind == .chat {
                #expect(results.map { $0.stats.value(.loadPeakMemoryGB).value } == [2, 2, 2])
                #expect(results.map { $0.stats.peakMemory.value } == [3, 3, 3])
                #expect(results.first?.stats.peakMemory.stat.provenance?.method.contains("operation peak") == true)
            }
        }
    }
    @Test func producedHardwareSurvivesSerializationAndReaderSelection() async throws {
        var captured: [PrecisionStatsResult] = []
        for (cores, date) in [(Int?(40), "2026-10-13T12:00:00Z"), (Int?(32), "2026-10-14T12:00:00Z"), (nil, "2026-10-15T12:00:00Z")] {
            let queue = queue()
            defer { try? FileManager.default.removeItem(at: queue.file.deletingLastPathComponent().deletingLastPathComponent()) }
            let results = try await PrecisionMeasurementHarness.run(
                model: "fixture", kind: .chat, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
                runner: FixtureMeasurementRunner(), probe: .unavailable,
                method: .init(chip: "m5-max", date: date, source: "fixture", gpuCores: cores), queue: queue)
            #expect(results.count == 3)
            for result in results {
                let decoded = try PrecisionStatsResult.decode(result.markdown())
                #expect(decoded.stats.availability.provenance?.gpuCores == cores)
                let peak = decoded.stats.peakMemory
                #expect(peak.value == 3 && peak.stat.provenance?.gpuCores == cores)
                #expect(peak.stat.provenance?.isHost(chip: "m5-max", gpuCores: 40) == (cores == 40))
                #expect(peak.stat.provenance?.isHost(chip: "m5-max", gpuCores: 32) == (cores == 32))
                #expect(peak.stat.provenance?.isHost(chip: "m5-max", gpuCores: 0) == false)
                captured.append(decoded)
            }
        }
        #expect(captured.count == 9)
        for cores in [32, 40] {
            let sets = PrecisionStatsReader.sets(
                kind: "chat", baseDtype: "bf16", runner: "native", supported: ["bf16", "int8", "int4"],
                results: captured, chip: "m5-max", gpuCores: cores)
            #expect(sets.count == 3)
            #expect(sets.values.allSatisfy { $0.peakMemory.stat.provenance?.gpuCores == cores && $0.speed.stat.provenance?.gpuCores == cores })
        }
    }
    @Test func memorySkipLeavesCandidatePendingAndContinuesLaterPrecisions() async throws {
        let runner = FixtureMeasurementRunner(), queue = queue()
        defer { try? FileManager.default.removeItem(at: queue.file.deletingLastPathComponent().deletingLastPathComponent()) }
        await runner.skipInt8ForMemory()
        let results = try await PrecisionMeasurementHarness.run(
            model: "fixture", kind: .chat, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
            runner: runner, probe: .unavailable, method: .init(chip: "fixture", source: "fixture"), queue: queue)
        #expect(results.count == 3)
        #expect(results.first { $0.stats.dtype == "int8" }?.stats.availability.reason == "skipped: not enough free memory (needs ~112.7 GB)")
        #expect(results.first { $0.stats.dtype == "int4" }?.stats.value(.tokensPerSecond).value == 16)
        #expect(await runner.prepares.map { $0.0 } == ["bf16", "int8", "bf16", "int4", "bf16"])
    }
    @Test func failedBaselineBracketPreventsMeasuredDefaultAndFeaturesAreDeclared() async throws {
        let runner = FixtureMeasurementRunner(), queue = queue()
        await runner.failBrackets()
        let results = try await PrecisionMeasurementHarness.run(
            model: "fixture", kind: .chat, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
            runner: runner, probe: .unavailable, method: .init(chip: "fixture", source: "fixture"), queue: queue)
        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.stats.availability.state == .unavailable && $0.stats.availability.reason?.contains("bracket failed") == true })
        #expect(RunnerFeature.native(worker: "decisions") == [.batching])
        #expect(RunnerFeature.native(worker: "chat") == [.sequentialOnly, .streaming])
        #expect(RunnerFeature.native(worker: "unknown").isEmpty)
    }
    @Test func differentAudioDurationIsUnavailableButItsSpeedAndEnergyAreStillMeasured() async throws {
        let runner = FixtureMeasurementRunner(), queue = queue()
        await runner.changeAudioCount()
        let probe = MeasurementProbe(capacityJoules: 360000, begin: {}, end: { CPUGPUEnergy(cpuJoules: 2, gpuJoules: 8) })
        let results = try await PrecisionMeasurementHarness.run(
            model: "fixture", kind: .voices, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
            runner: runner, probe: probe, method: .init(chip: "fixture", source: "fixture"), queue: queue)
        let candidate = try #require(results.last?.stats)
        #expect(candidate.availability.state == .unavailable && candidate.availability.reason?.contains("sample counts differ") == true)
        #expect(candidate.value(.spectralDistance).state == .unavailable)
        #expect(candidate.value(.realTimeFactor).value != nil && candidate.value(.batteryPercentPerHourSpeech).value != nil)
        #expect(candidate.value(.firstAudioSeconds).value == 0.1)
    }
    @Test func receiptsRejectDownloadsAndEnergyMissingStaysUnavailable() async throws {
        let runner = FixtureMeasurementRunner(downloads: 1), queue = queue()
        let results = try await PrecisionMeasurementHarness.run(
            model: "fixture", kind: .chat, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
            runner: runner, probe: .unavailable, method: .init(chip: "fixture", source: "fixture"), queue: queue)
        #expect(results.first?.stats.availability.state == .unavailable)
        #expect(await runner.prepares.count == 1)
        let good = FixtureMeasurementRunner()
        let sampled = try await PrecisionMeasurementHarness.run(
            model: "fixture", kind: .chat, sourceDtype: "bf16", revision: "pinned", runnerID: "native", mode: "exact",
            runner: good, probe: .unavailable, method: .init(chip: "fixture", source: "fixture"), queue: queue)
        #expect(sampled.first?.stats.value(.batteryPercentPer100000Tokens).state == .unavailable)
    }
    @Test func qualityMathChecksCountsStableLogitsAndKnownValues() throws {
        #expect(try StatsQuality.agreement(reference: [0, 1], candidate: [0, 0]) == 50)
        #expect(throws: (any Error).self) { try StatsQuality.agreement(reference: [0], candidate: []) }
        let identity = try StatsQuality.chat(reference: [[10000, 9999]], candidate: [[10000, 9999]])
        #expect(identity.agreement == 100 && abs(identity.kl) < 1e-12)
        let changed = try StatsQuality.chat(reference: [[1, 0]], candidate: [[0, 1]])
        #expect(changed.agreement == 0 && changed.kl > 0)
        #expect(try StatsQuality.psnr(reference: [0, 0], candidate: [0.1, 0.1]) == 20)
        #expect(try StatsQuality.psnr(reference: [0, 0], candidate: [0, 0]) == 120)
        let sine = (0..<256).map { sin(Double($0) * 0.1) }
        #expect(try StatsQuality.spectralDistance(reference: sine, candidate: sine) == 0)
        let scaled = try StatsQuality.spectralDistance(reference: sine, candidate: sine.map { $0 * 2 })
        #expect(abs(scaled - 1) < 1e-10)
        #expect(throws: (any Error).self) { try StatsQuality.chat(reference: [[0, 1]], candidate: [[0]]) }
    }
    @Test func batteryMathAndCounterSelectionNeverDoubleCount() throws {
        #expect(try BatteryEnergy.capacityJoules(milliampHours: 10000, millivolts: 10000) == 360000)
        #expect(try BatteryEnergy.percent(joules: 360, capacityJoules: 360000, completedUnits: 100, statedUnits: 1000) == 1)
        let counters = try IOReportEnergyMeter.cpuGPU(["CPU Energy": 3, "ECPU Energy": 1, "PCPU Energy": 2, "GPU Energy": 4, "ANE Energy": 9])
        #expect(counters.joules == 7)
        #expect(try IOReportEnergyMeter.cpuGPU(["ECPU Energy": 1, "PCPU Energy": 2, "GPU Energy": 4]).joules == 7)
        #expect(throws: (any Error).self) { try IOReportEnergyMeter.cpuGPU(["CPU Energy": 3]) }
        #expect(IOReportEnergyMeter.jouleScale("mJ") == 0.001)
        #expect(throws: (any Error).self) { try BatteryEnergy.percent(joules: 1, capacityJoules: 0, completedUnits: 1, statedUnits: 100) }
    }
    @Test func queueCompletionFailureAndSeedMatrix() throws {
        let queue = queue()
        defer { try? FileManager.default.removeItem(at: queue.file.deletingLastPathComponent().deletingLastPathComponent()) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = repo.appendingPathComponent("models")
        let folders = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        var expected = 0
        for folder in folders {
            let model = try ModelManifest.load(folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
            try queue.introduce(model, runner: model.runners.first?.name ?? "native", mode: "exact")
            // Every precision of a measured kind (decisions, chat, images, voices, sound, enhancement) is one pending entry; operation kinds none.
            if let kind = ModelStatKind(rawValue: model.kind), !kind.isOperation { expected += model.precisions.count }
        }
        #expect(try queue.read().count == expected, "one pending entry per precision of each measured model")
        let entry = try #require(try queue.read().first)
        try queue.complete(id: entry.id, result: "results/fixture.md")
        #expect(try queue.read().first?.state == "complete")
        try queue.recordFailure(id: entry.id, reason: "fixture failure")
        #expect(try queue.read().first?.state == "pending")
        #expect(try queue.read().first?.error == "fixture failure")
    }
    @Test func contextFromConfigWinsWithoutReadingWeights() throws {
        #expect(ModelContext.maximum(config: Data(#"{"text_config":{"max_position_embeddings":262144},"max_position_embeddings":8192}"#.utf8), fallback: 32768) == 262144)
        #expect(ModelContext.maximum(config: Data(#"{"max_position_embeddings":131072}"#.utf8), fallback: 32768) == 131072)
        #expect(ModelContext.maximum(config: Data(#"{"max_position_embeddings":1.5}"#.utf8), fallback: 262144) == 262144)
        let load = ModelContext.loadConfig(["context_limit": 8192, "settings": ["context_limit": 32768]], maximum: 262144)
        #expect(load["context_limit"] == 262144 && load["settings"]?["context_limit"] == 262144)
    }
}
