// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import Testing
@testable import ArsenalCore

struct StatsRevisionTests {
    private func stats(_ dtype: String, speed: Double) -> PrecisionStatSet {
        var set = PrecisionStatSet(kind: .chat, dtype: dtype, runner: "native")
        let provenance = StatProvenance(
            chip: HostChip.current, date: "2026-10-09T00:00:00Z", method: "ordinary chat window", baseline: "own bf16", source: "fixture",
            revision: String(repeating: "a", count: 40), run: "same-run")
        set.availability = StatValue(state: .measured, provenance: provenance)
        set.set(.tokensPerSecond, StatValue(state: .measured, value: speed, provenance: provenance))
        set.set(.nextTokenAgreementPercent, StatValue(state: .measured, value: 100, provenance: provenance))
        return set
    }
    @Test(arguments: ["revision", "run", "date", "source", "chip"])
    func mismatchedQualityCannotSelectInt8(field: String) {
        let base = stats("bf16", speed: 20)
        var quant = stats("int8", speed: 30), quality = quant.value(.nextTokenAgreementPercent)
        switch field {
        case "revision": quality.provenance?.revision = "OLD"
        case "run": quality.provenance?.run = "another-run"
        case "date": quality.provenance?.date = "2026-10-08T00:00:00Z"
        case "source": quality.provenance?.source = "another-worker"
        default: quality.provenance?.chip = "another-chip"
        }
        quant.set(.nextTokenAgreementPercent, quality)
        let result = PrecisionRecommendation.choose(["bf16": base, "int8": quant], baseDtype: "bf16", chip: HostChip.current)
        #expect(result.dtype == "bf16")
        #expect(!PrecisionEvidence.current(quant, revision: String(repeating: "a", count: 40)))
    }
    @Test func noRevisionOrRunCannotRank() {
        var base = stats("bf16", speed: 20), quant = stats("int8", speed: 30)
        base.availability.provenance?.revision = nil; quant.availability.provenance?.run = nil
        #expect(PrecisionRecommendation.choose(["bf16": base, "int8": quant], baseDtype: "bf16", chip: HostChip.current).pending)
    }
    @Test func signedSchema2WithStaleQualityIsNotShown() async throws {
        var set = stats("bf16", speed: 42), quality = set.value(.nextTokenAgreementPercent)
        quality.provenance?.revision = "OLD"; set.set(.nextTokenAgreementPercent, quality)
        let result = PrecisionStatsResult(model: "fixture-chat", mode: "fast", stats: set)
        let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], extraFiles: ["results/mixed.md": result.markdown()])
        defer { fixture.stop() }
        #expect(await catalogRow(fixture.service)?.precisionStats["bf16"]?.value(.tokensPerSecond).state == .notMeasured)
        #expect(await fixture.service.catalog().first?.runners.first?.stats["bf16"]?.value(.nextTokenAgreementPercent).state == .notMeasured)
        await fixture.service.shutdown()
    }
    @Test func savedExactIsRetiredOnOpenWithoutChangingDtype() async throws {
        var configuration = ArsenalConfiguration()
        configuration.selections["fixture-chat"] = ModelPreferences(precision: "int8", fast: false)
        let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], configuration: configuration)
        defer { fixture.stop() }
        let saved = try ArsenalConfiguration.read(fixture.paths.support.appendingPathComponent("config.json"))
        #expect(saved.selections["fixture-chat"]?.fast == true && saved.selections["fixture-chat"]?.precision == "int8")
        #expect(await catalogRow(fixture.service)?.fast == true)
        await fixture.service.shutdown()
    }
    @Test func ordinaryAdapterUsesTheSameGetLoadSettingsAndRunner() async throws {
        let fixture = try await makeLocalCatalogFixture(
            kinds: ["chat"], filePayloads: ["config.json": Data("{\"max_position_embeddings\":262144}".utf8)],
            artifacts: ProtocolFixtureArtifact(scenario: "normal"), context: 32768, nativeContext: 262144)
        defer { fixture.stop() }
        try await fixture.service.get(id: "fixture-chat", precision: "bf16")
        let model = try await fixture.service.measurementManifest(id: "fixture-chat")
        let directory = fixture.paths.directory("benchmarks").appendingPathComponent("output")
        let adapter = MeasurementAdapter(service: fixture.service, model: model, directory: directory, fixture: true)
        let preparation = try await adapter.prepare(dtype: "bf16", sourceDtype: "bf16")
        #expect(preparation.servedPath && !preparation.inMemory && preparation.downloadedBytes == 0)
        let load = try await fixture.service.run(id: "fixture-chat", op: "inspect_load")
        #expect(load["config"]?["context_limit"] == 262144)
        let row = try #require(await fixture.service.rows().first)
        #expect(row.residentRunner == row.runner && row.residentDType == "bf16" && row.fast && row.state == .loaded)
        await adapter.shutdown()
    }
    @Test func audioIdentityRejectsSpectrumCollisions() throws {
        let samples = (0..<256).map { sin(Double($0) * 0.1) }
        let base = MeasurementOutput(fixtureID: "audio", samples: samples, sampleRate: 24000)
        let flipped = MeasurementOutput(fixtureID: "audio", samples: samples.map { -$0 }, sampleRate: 24000)
        var edge = base; edge.samples?[255] += 1
        #expect(try StatsQuality.spectralDistance(reference: samples, candidate: flipped.samples!) == 0)
        #expect(try StatsQuality.spectralDistance(reference: samples, candidate: edge.samples!) == 0)
        #expect(StatsQuality.audioIdentical(reference: base, candidate: base))
        #expect(!StatsQuality.audioIdentical(reference: base, candidate: flipped))
        #expect(!StatsQuality.audioIdentical(reference: base, candidate: edge))
    }
    @Test func countersRejectIncompleteAndResetClusters() throws {
        let cases: [[String: Double]] = [
            ["ECPU Energy": 2, "GPU Energy": 1], ["ECPU Energy": -1, "PCPU Energy": 3, "GPU Energy": 1],
            ["CPU Energy": 2, "ECPU Energy": -1, "GPU Energy": 1], ["CPU Energy": 2, "ECPU Energy": 1, "GPU Energy": 1], ["Mystery CPU Energy": 2, "GPU Energy": 1]
        ]
        for counters in cases {
            #expect(throws: (any Error).self) { try IOReportEnergyMeter.cpuGPU(counters) }
        }
        #expect(try IOReportEnergyMeter.cpuGPU(["ECPU Energy": 1, "P0CPU Energy": 2, "P1CPU Energy": 3, "GPU Energy": 4]).joules == 10)
    }
    @Test func footprintIncludesOrdinaryQualificationChildren() throws {
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/bin/sleep"); child.arguments = ["10"]
        try child.run()
        defer { if child.isRunning { child.terminate(); child.waitUntilExit() } }
        #expect(ServedFootprintPeak.processTree(roots: [getpid()]).contains(child.processIdentifier))
    }
    @Test func fp32ServingPathsAreRefusedBeforeLoad() {
        #expect(ServingMeasurementPolicy.refusal(model: "re-use", fast: true)?.contains("FP32") == true)
        #expect(ServingMeasurementPolicy.refusal(model: "stable-audio-3-medium", fast: false)?.contains("FP32") == true)
        #expect(ServingMeasurementPolicy.refusal(model: "stable-audio-3-medium", fast: true)?.contains("FP32") == true)
        #expect(ServingMeasurementPolicy.refusal(model: "qwen3.8-27b", fast: false) == nil)
    }
    @Test func profileImportsOnlyClonesAndLeavesSourceUnchanged() async throws {
        let fixture = try await makeLocalCatalogFixture(kinds: ["chat"])
        defer { fixture.stop() }
        try await fixture.service.get(id: "fixture-chat", precision: "bf16")
        try await fixture.service.get(id: "fixture-chat", precision: "int4")
        let model = try await fixture.service.measurementManifest(id: "fixture-chat")
        let paths = ArsenalPaths(support: fixture.paths.support.appendingPathComponent("isolated")); try paths.create()
        let store = BlobStore(support: fixture.paths.support)
        let source = try await store.snapshotURL(model: model.id, revision: model.weights.revision, precision: model.basePrecision)
        let before = try FileDigest.sha256(source.appendingPathComponent("arsenal-snapshot.json"))
        try await MeasurementProfile.importSource(model: model, sourceSupport: fixture.paths.support, paths: paths)
        let imported = try await ModelFetcher(support: paths.support).verifyInstalled(model: model, precision: model.basePrecision)
        #expect(try Data(contentsOf: imported.weightsDirectory.appendingPathComponent("base.bin")) == Data([12, 34, 56]))
        #expect(try FileDigest.sha256(source.appendingPathComponent("arsenal-snapshot.json")) == before)
        #expect(imported.weightsDirectory.path.hasPrefix(paths.support.path))
        let quantized = try await ModelFetcher(support: paths.support).verifyInstalled(model: model, precision: "int4")
        #expect(quantized.weightsDirectory.path.hasPrefix(paths.support.path))
        await fixture.service.shutdown()
    }
    @Test func queueRetiresStaleModeAndRevisionWithoutLosingReceipts() async throws {
        let fixture = try await makeLocalCatalogFixture(kinds: ["chat"])
        defer { fixture.stop() }
        var model = try await fixture.service.measurementManifest(id: "fixture-chat")
        let queue = PendingBenchmarkQueue(support: fixture.paths.support)
        try queue.introduce(model, runner: "native", mode: "exact")
        try queue.introduce(model, runner: "native", mode: "fast")
        let current = try queue.read().filter { $0.state == "pending" }
        #expect(current.count == 3 && current.allSatisfy { $0.mode == "fast" })
        model.weights.revision = String(repeating: "b", count: 40)
        try queue.introduce(model, runner: "native", mode: "fast")
        #expect(try queue.read().filter { $0.state == "pending" }.allSatisfy { $0.revision == model.weights.revision })
        #expect(try queue.read().filter { $0.state == "superseded" }.count == 6)
        await fixture.service.shutdown()
    }
}
