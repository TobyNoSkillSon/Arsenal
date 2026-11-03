// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct ModelsTableDataTests {
    let p = StatProvenance(chip: "m5-max", date: "2026-10-15T12:00:00Z", method: "fixed workload", gpuCores: 40)
    func set(_ dtype: String, kind: ModelStatKind = .images, peak: Double? = nil) -> PrecisionStatSet {
        var result = PrecisionStatSet(kind: kind, dtype: dtype, runner: "native")
        result.availability = StatValue(state: .quickCheck, provenance: p)
        if let peak { result.set(.peakMemoryGB, StatValue(state: .quickCheck, value: peak, provenance: p)) }
        return result
    }
    @Test func layaCatalogDoesNotBorrowItsBasePeakForOtherPrecisions() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fixture = try await makeLocalCatalogFixture(kinds: ["decisions"])
        defer { fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent()) }
        let model = try ModelManifest.load(repo.appendingPathComponent("models/laya-english"))
        let data = try Data(contentsOf: repo.appendingPathComponent("models/laya-english/results/m5-max.json"))
        var row = ModelRow(id: model.id, name: model.name, kind: model.kind, precisions: model.precisions)
        row.precisionStats = await fixture.service.realCatalogStats(model, files: ["results/m5-max.json": data])
        #expect(row.precisionStats["fp16"]?.peakMemory.value == 1.754)
        // Laya ships 16-bit only since 26 Oct: its withdrawn tiers borrow nothing either, measured or not.
        for dtype in ["int8", "int4"] {
            #expect(row.precisionStats[dtype]?.peakMemory.value == nil)
            #expect([.notMeasured, .unavailable].contains(row.precisionStats[dtype]?.value(.peakMemoryGB).state ?? .notMeasured))
        }
        await fixture.service.shutdown()
    }
    @Test func speedHasSeparateUnitsAndUnavailableTiersHaveNoFigures() {
        for (kind, metric, value, unit) in [
            (ModelStatKind.decisions, StatMetric.millisecondsPerDecision, 8.0, "ms/decision"),
            (.chat, .tokensPerSecond, 30.0, "tokens/s"), (.images, .secondsPerImage, 4.0, "s/image"),
            (.voices, .realTimeFactor, 0.25, "× real time"), (.sound, .realTimeFactor, 0.25, "× real time"),
            (.enhancement, .realTimeFactor, 0.25, "× real time")
        ] {
            var stats = set("fp16", kind: kind)
            stats.set(metric, StatValue(state: .quickCheck, value: value, provenance: p))
            #expect(stats.speed.value == (metric == .realTimeFactor ? 4 : value))
            #expect(stats.speed.unit == unit && stats.speed.stat.provenance == p)
            stats.availability = .unavailable("Unsupported")
            #expect(stats.speed.value == nil)
        }
    }
    @Test func chipAttributionNeverClaimsAnotherCoreConfiguration() {
        #expect(p.chipName == "M5 Max" && p.hardwareDescription == "M5 Max · 40 GPU cores")
        #expect(p.isHost(chip: "m5-max", gpuCores: 40))
        #expect(!p.isHost(chip: "m5-max", gpuCores: 32))
        #expect(!p.isHost(chip: "m4-max", gpuCores: 40))
        #expect(!p.isHost(chip: "m5-max", gpuCores: 0))
        var unknown = p; unknown.gpuCores = nil
        #expect(!unknown.isHost(chip: "m5-max", gpuCores: 40))
        #expect(!unknown.isHost(chip: "m5-max", gpuCores: 0))
    }
    @Test func partialLocalEvidenceRetainsReferenceChipPerFigure() throws {
        var local = set("fp16"); local.availability.provenance?.chip = "m4-max"
        let localP = StatProvenance(chip: "m4-max", date: p.date, method: p.method)
        local.set(.secondsPerImage, StatValue(state: .measured, value: 8, provenance: localP))
        let other = set("fp16", peak: 12)
        let sets = PrecisionStatsReader.sets(
            kind: "images", baseDtype: "fp16", runner: "native", supported: ["fp16"],
            results: [
                .init(model: "fixture", mode: "fast", stats: other), .init(model: "fixture", mode: "fast", stats: local)
            ], chip: "m4-max")
        let chosen = try #require(sets["fp16"])
        #expect(chosen.speed.value == 8 && chosen.speed.stat.provenance?.chip == "m4-max")
        #expect(chosen.peakMemory.value == 12 && chosen.peakMemory.stat.provenance?.chip == "m5-max")
        #expect(chosen.peakMemory.stat.provenance?.gpuCores == 40)
    }
    @Test func oldChatSchemaKeepsLoadPeakSeparateFromRunPeak() throws {
        var stats = set("bf16", kind: .chat)
        stats.set(.loadPeakMemoryGB, StatValue(state: .measured, value: 10, provenance: p))
        var object = try WireJSON.decode(JSONEncoder().encode(PrecisionStatsResult(model: "fixture", mode: "fast", stats: stats)))
        var inner = try #require(object["stats"]?.object), metrics = try #require(inner["metrics"]?.object)
        metrics.removeValue(forKey: "peakMemoryGB"); inner["metrics"] = .object(metrics); object["stats"] = .object(inner)
        let decoded = try PrecisionStatsResult.decode(WireJSON.encode(object))
        #expect(decoded.stats.peakMemory.value == nil)
        #expect(decoded.stats.value(.loadPeakMemoryGB).value == 10)
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "chat")
        row.precisionStats["bf16"] = decoded.stats
        let check = LoadMemoryAssessment(row: row, precision: "bf16", forecast: nil, freeMB: 9000)
        #expect(check.neededGB == 10 && !check.fits && check.basis == .loadPeak)
        #expect(row.precisionStats["bf16"]?.peakMemory.value == nil)
    }
    @Test func shippedCatalogHasSixteenNonChatModelsButChatFolderStillLoads() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folders = try FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("models"), includingPropertiesForKeys: nil).filter {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path)
        }
        let models = try folders.map { try ModelManifest.load($0) }
        let shipped = models.filter(CatalogShipping.includes)
        #expect(shipped.count >= 16 && shipped.allSatisfy { $0.kind != "chat" || CatalogShipping.includes($0) })
        #expect(models.first { $0.id == "qwen3.8-27b" }?.operations.contains { $0.name == "chat" } == true)
        // Their memory payload declares the 16-bit copy; the quantized tiers were withdrawn on 26 Oct (under 1 GB at 16-bit).
        for id in ["laya-english", "laya-multilingual", "laya-typed-decisions", "kokoro-82m"] {
            let model = try #require(models.first { $0.id == id })
            #expect(model.memory?.residentBytes[model.basePrecision] != nil && model.memory?.residentBytes["int8"] == nil, "\(id)")
        }
    }
    @Test func invalidDerivedEvidenceCannotMasqueradeAsMeasurement() throws {
        var stats = set("fp16")
        stats.set(.peakMemoryGB, StatValue(state: .derived, value: 3, provenance: p))
        #expect(throws: (any Error).self) { try stats.validate() }
        stats.set(.peakMemoryGB, StatValue(state: .derived, value: 3, provenance: p, basis: "Weight substitution"))
        try stats.validate()
        #expect(PrecisionRecommendation.choose(["fp16": stats], baseDtype: "fp16", chip: "m5-max").pending)
    }
    @Test func matchingGpuConfigurationWinsOverNewerSameNameReference() throws {
        var exact = set("fp16", peak: 10), other = set("fp16", peak: 12)
        exact.availability.provenance?.date = "2026-10-14T12:00:00Z"
        other.availability.provenance?.gpuCores = 32
        other.metrics[StatMetric.peakMemoryGB.rawValue]?.provenance?.gpuCores = 32
        let selected = PrecisionStatsReader.sets(
            kind: "images", baseDtype: "fp16", runner: "native", supported: ["fp16"],
            results: [
                .init(model: "fixture", mode: "fast", stats: other), .init(model: "fixture", mode: "fast", stats: exact)
            ], chip: "m5-max", gpuCores: 40)
        #expect(selected["fp16"]?.peakMemory.value == 10)
    }
    @Test func legacyFluxWorkloadAndChatPeakAreAttributed() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try JSONDecoder().decode(ResultsFile.self, from: Data(contentsOf: repo.appendingPathComponent("models/flux2-klein-4b/results/m5-max.json")))
        let mapped = try #require(PrecisionStatsReader.legacy(source))
        #expect(mapped.stats.peakMemory.stat.provenance?.method.contains("512×512") == true)
        #expect(mapped.stats.speed.stat.provenance?.gpuCores == 40)
        var chat = source; chat.kind = "chat"; chat.model = "fixture-chat"
        let peak = try #require(PrecisionStatsReader.legacy(chat)).stats
        #expect(peak.value(.peakMemoryGB).value == source.metrics["peak_ram_mb"].flatMap { $0 }.map { $0 / 1000 })
        #expect(peak.value(.loadPeakMemoryGB).value == nil)
    }

}

private extension RealModelsService {
    func realCatalogStats(_ model: ModelManifest, files: [String: Data]) -> [String: PrecisionStatSet] {
        cachePrecisionEvidence(model, files: files)
        return statsForRunner(model, runner: "native")
    }
}
