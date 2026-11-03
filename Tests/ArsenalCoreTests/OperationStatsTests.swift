// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Operation kinds (transcription, embeddings, perception, restoration, 3D) carry peak memory and one request's time,
/// filled by quick checks: the night of 22 Oct's files in the model folders decode, stay current and reach the table.
struct OperationStatsTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func quick(_ model: String, _ dtype: String) throws -> PrecisionStatsResult {
        try PrecisionStatsResult.decode(Data(contentsOf: repo.appendingPathComponent("models/\(model)/results/quick-m5-max-\(dtype)-native-exact.json")))
    }

    @Test func operationKindsCarryRequestTimeAndPeakOnly() throws {
        #expect(ModelStatKind.operationKinds.count == 5)
        for kind in ModelStatKind.operationKinds {
            #expect(kind.isOperation && kind.metrics == [.secondsPerRequest, .peakMemoryGB])
            #expect(!ModelStatKind.measurable.contains(kind))
            #expect(PrecisionStatSet(kind: kind, dtype: "fp16", runner: "native").speed.metric == .secondsPerRequest)
        }
        #expect(ModelStatKind.measurable == [.decisions, .chat, .images, .voices, .sound, .enhancement])
        // A cell of another kind's metric is not part of an operation set.
        var set = PrecisionStatSet(kind: .audio, dtype: "fp16", runner: "native")
        set.set(.realTimeFactor, StatValue())
        #expect(throws: (any Error).self) { try set.validate() }
    }

    @Test func nightQuickCheckDecodesAsCurrentQuickCheckFigures() throws {
        let manifest = try ModelManifest.load(repo.appendingPathComponent("models/parakeet-v3-ultra"))
        let result = try quick("parakeet-v3-ultra", "fp16")
        #expect(result.stats.kind == .audio && result.stats.kind.rawValue == manifest.kind)
        #expect(Set(result.stats.metrics.keys) == ["secondsPerRequest", "peakMemoryGB"])
        #expect(PrecisionEvidence.current(result.stats, revision: manifest.weights.revision))
        #expect(!PrecisionEvidence.current(result.stats, revision: String(repeating: "0", count: 40)))
        // Never "measured": a busy-Mac quick check.
        #expect(([result.stats.availability] + Array(result.stats.metrics.values)).allSatisfy { $0.state == .quickCheck })
        let sets = PrecisionStatsReader.sets(
            kind: manifest.kind, baseDtype: manifest.basePrecision, runner: "native", supported: manifest.precisions, results: [result], chip: "m5-max", gpuCores: 40)
        let fp16 = try #require(sets["fp16"])
        #expect(abs((fp16.peakMemory.value ?? 0) - QuickCheckEvidence.gigabytes(mebibytes: 1740.8)) < 1e-12)
        #expect(fp16.peakMemory.stat.state == .quickCheck)
        #expect(fp16.speed.metric == .secondsPerRequest && abs((fp16.speed.value ?? 0) - 0.095) < 1e-12)
        #expect(fp16.speed.stat.basis == "One warm Transcribe request on a 3.3 s speech-and-music clip.")
        #expect(fp16.speed.stat.provenance?.method.contains("not the controlled interleaved (ABBA) protocol") == true)
        #expect(fp16.qualityCheckPassed == nil)
        // The 16-bit format stays: no own-16-bit quality metric to recommend another.
        #expect(PrecisionRecommendation.choose(sets, baseDtype: "fp16", chip: "m5-max").pending)
    }

    @Test func chatKindOperationModelGetsPeakMemoryOnly() throws {
        let result = try quick("teleocr", "fp16")
        #expect(result.stats.kind == .chat)
        #expect(result.stats.value(.peakMemoryGB).state == .quickCheck)
        #expect(result.stats.value(.tokensPerSecond).state == .notMeasured && result.stats.speed.value == nil)
    }

    @Test func operationKindsHaveNoServedMeasurement() async throws {
        let queue = PendingBenchmarkQueue(support: FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-operation-" + UUID().uuidString))
        defer { try? FileManager.default.removeItem(at: queue.file.deletingLastPathComponent().deletingLastPathComponent()) }
        let manifest = try ModelManifest.load(repo.appendingPathComponent("models/parakeet-v3-ultra"))
        #expect(try queue.introduce(manifest, runner: "native", mode: "exact") == 0)
        #expect(try queue.read().isEmpty)
        #expect(throws: (any Error).self) {
            try MeasurementWorkload.request(kind: .perception, model: "fixture", batched: false, count: 1, directory: URL(fileURLWithPath: "/unused"))
        }
        await #expect(throws: (any Error).self) {
            try await PrecisionMeasurementHarness.run(
                model: "fixture", kind: .audio, sourceDtype: "fp16", revision: "pinned", runnerID: "native", mode: "exact",
                runner: FixtureMeasurementRunner(), probe: .unavailable, method: .init(chip: "fixture", source: "fixture"), queue: queue, approved: true)
        }
    }

    /// The six original kinds' table peaks arm the runaway stop as before, whatever their state (their quick checks came
    /// from intake runs). An operation kind's quick-check or estimated peak (one short input) sizes the Load but never
    /// arms it; its measured peak does.
    @Test func operationQuickCheckPeaksNeverArmTheRunawayStop() {
        for kind in ModelStatKind.operationKinds {
            for state in [StatState.quickCheck, .derived] {
                let (requirement, stop) = ceiling(kind, state)
                #expect(requirement.neededGB == 2.09 && requirement.basis == .unmeasuredPeak, "\(kind) \(state)")
                #expect(stop.expectedMB == 2090 && stop.runawayMB == nil && !stop.measuredPeak, "\(kind) \(state)")
            }
            let (requirement, measured) = ceiling(kind, .measured)
            #expect(requirement.basis == .peak && measured.measuredPeak && measured.runawayMB == 2090 + WorkerMemoryCeiling.runawayMarginMB)
        }
        for kind in ModelStatKind.measurable {
            for state in [StatState.quickCheck, .derived, .measured] {
                let (requirement, stop) = ceiling(kind, state)
                #expect(requirement.basis == .peak && stop.runawayMB == 2090 + WorkerMemoryCeiling.runawayMarginMB, "\(kind) \(state)")
            }
        }
    }

    /// The runaway stop calls its figure the measured peak only when it was measured; otherwise the expected peak.
    @Test func runawayStopNamesAQuickCheckPeakExpected() {
        let quick = ceiling(.chat, .quickCheck).1, measured = ceiling(.chat, .measured).1
        let quickMessage = quick.runawayFailure(footprintBytes: 20_000_000_000, limitBytes: 18_090_000_000).message
        #expect(!quick.measuredPeak && quickMessage.contains("(its expected peak is 1.95 GB at FP16)") && !quickMessage.contains("measured"))
        #expect(measured.measuredPeak)
        #expect(measured.runawayFailure(footprintBytes: 20_000_000_000, limitBytes: 18_090_000_000).message.contains("(its measured peak is 1.95 GB at FP16)"))
    }

    private func ceiling(_ kind: ModelStatKind, _ state: StatState) -> (LoadMemoryRequirement, WorkerMemoryCeiling) {
        let provenance = StatProvenance(chip: "m5-max", date: "2026-10-22T02:54:56Z", method: "fixture", revision: "pinned", run: "fixture-fp16", gpuCores: 40)
        var row = ModelRow(id: "fixture", name: "Fixture", kind: kind.rawValue, precisions: ["fp16"])
        row.runner = "native"
        var set = PrecisionStatSet(kind: kind, dtype: "fp16", runner: "native")
        set.set(.peakMemoryGB, StatValue(state: state, value: 2.09, provenance: provenance, basis: state == .derived ? "fixture" : nil))
        row.precisionStats["fp16"] = set
        let requirement = LoadMemoryRequirement(row: row, precision: "fp16", forecast: nil)
        return (requirement, WorkerMemoryCeiling(name: "Fixture", requirement: requirement, precision: "fp16", forced: false))
    }

    /// Per figure: this Mac first; on the same Mac a controlled measurement beats a newer quick check.
    @Test func readerPrefersMeasuredOverQuickCheckOnTheSameMac() {
        func result(_ state: StatState, _ value: Double, date: String, chip: String = "m5-max", cores: Int = 40) -> PrecisionStatsResult {
            let provenance = StatProvenance(chip: chip, date: date, method: "fixture", revision: "pinned", run: "run-\(date)", gpuCores: cores)
            var stats = PrecisionStatSet(kind: .audio, dtype: "fp16", runner: "native")
            stats.availability = StatValue(state: state, provenance: provenance)
            stats.set(.peakMemoryGB, StatValue(state: state, value: value, provenance: provenance))
            return PrecisionStatsResult(model: "fixture", mode: "exact", stats: stats)
        }
        func peak(_ results: [PrecisionStatsResult]) -> StatValue? {
            PrecisionStatsReader.sets(kind: "audio", baseDtype: "fp16", runner: "native", supported: ["fp16"], results: results, chip: "m5-max", gpuCores: 40)["fp16"]?
                .value(.peakMemoryGB)
        }
        let measured = result(.measured, 2.5, date: "2026-10-20T00:00:00Z"), quick = result(.quickCheck, 2.1, date: "2026-10-22T00:00:00Z")
        #expect(peak([quick, measured])?.value == 2.5 && peak([measured, quick])?.state == .measured)
        // Another Mac's measurement does not displace this Mac's quick check.
        let elsewhere = result(.measured, 3.0, date: "2026-10-21T00:00:00Z", chip: "m3-pro", cores: 18)
        #expect(peak([elsewhere, quick])?.value == 2.1)
    }
}
