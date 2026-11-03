// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct PrecisionStatsTests {
    @Test func allKindsHaveExplicitMissingCells() throws {
        for kind in ModelStatKind.allCases {
            let sets = PrecisionStatsReader.sets(kind: kind.rawValue, baseDtype: "bf16", runner: "native", supported: ["bf16", "int8"], results: [])
            #expect(sets.count == 3)
            #expect(sets["int4"]?.availability.state == .unavailable)
            #expect(sets["bf16"]?.metrics.values.allSatisfy { $0.state == .notMeasured && $0.value == nil } == true)
            try sets.values.forEach { try $0.validate() }
        }
    }
    @Test func failedInt4QualityWarnsWithoutBlockingAdmission() {
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "sound", precisions: ["fp16", "int4"])
        row.runner = "native"
        row.precisionStats["int4"] = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native", unavailableReason: "Did not pass the quality check")
        #expect(row.dtypeUnavailableReason("int4") == nil)
        #expect(row.precisionQualityWarnings["int4"] == PrecisionStatSet.int4QualityWarning)
        #expect(row.dtypeUnavailableReason("int8") == nil)
        row.runner = "another"
        #expect(row.dtypeUnavailableReason("int4") == nil)
        #expect(row.precisionQualityWarnings["int4"] == PrecisionStatSet.int4UncheckedWarning)
    }
    @Test func frontMatterRoundTripAndInvalidState() throws {
        var stats = PrecisionStatSet(kind: .chat, dtype: "int8", runner: "native")
        let provenance = StatProvenance(
            chip: "m5-max", date: "2026-10-09T12:00:00Z", method: "32 fixed teacher-forced tokens", baseline: "own bf16", revision: String(repeating: "a", count: 40),
            run: "fixture-run")
        stats.set(.nextTokenAgreementPercent, StatValue(state: .quickCheck, value: 96, provenance: provenance))
        let result = PrecisionStatsResult(model: "fixture-chat", mode: "exact", stats: stats)
        #expect(try PrecisionStatsResult.decode(result.markdown(notes: "Retained notes.")) == result)
        stats.set(.tokensPerSecond, .unavailable("Runner has no decoding operation."))
        try stats.validate()
        stats.set(.tokensPerSecond, StatValue(state: .notMeasured, value: 100))
        #expect(throws: (any Error).self) { try stats.validate() }
    }
    @Test func legacyAccuracyDoesNotBecomeAgreement() throws {
        let conditions = BenchmarkConditions(locked: false, idle: [], run: [])
        let legacy = ResultsFile(
            model: "fixture", kind: "decisions", chip: "m5-max", osBuild: "fixture", appBuild: "fixture", runner: "native",
            workerDigest: "fixture", precision: "int8", mode: "exact", suite: .init(id: "fixture", sha256: "fixture"), profile: "quick", stub: false,
            gatePassed: false, conditions: conditions, metrics: ["accuracy": 0.98, "latency_s": 0.12, "peak_ram_mb": 1500], cases: [])
        let mapped = try #require(PrecisionStatsReader.legacy(legacy))
        #expect(mapped.stats.value(.agreementPercent).state == .notMeasured)
        #expect(mapped.stats.value(.millisecondsPerDecision).value == 120)
        #expect(mapped.stats.value(.millisecondsPerDecision).state == .quickCheck)
        #expect(mapped.stats.availability.state == .unavailable)
    }
    @Test func rowAndRunnerTransportRoundTrip() throws {
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "images")
        row.precisionStats = PrecisionStatsReader.sets(kind: "images", baseDtype: "bf16", runner: "native", supported: ["bf16"], results: [])
        #expect(try JSONDecoder().decode(ModelRow.self, from: JSONEncoder().encode(row)) == row)
        var runner = CatalogRunner(name: "native", language: "swift", engine: "MLX")
        runner.stats = row.precisionStats; runner.features = [.sequentialOnly]
        #expect(try JSONDecoder().decode(CatalogRunner.self, from: JSONEncoder().encode(runner)) == runner)
    }
    @Test func realServiceReadsFrontMatterIntoRowsAndRunner() async throws {
        var stats = PrecisionStatSet(kind: .chat, dtype: "bf16", runner: "native")
        let provenance = StatProvenance(
            chip: HostChip.current, date: "2026-10-09T12:00:00Z", method: "fixture only", revision: String(repeating: "a", count: 40), run: "fixture-run")
        stats.availability = StatValue(state: .quickCheck, provenance: provenance)
        stats.set(.tokensPerSecond, StatValue(state: .quickCheck, value: 42, provenance: provenance))
        let result = PrecisionStatsResult(model: "fixture-chat", mode: "fast", stats: stats)
        let fixture = try await makeLocalCatalogFixture(kinds: ["chat"], extraFiles: ["results/stats.md": result.markdown()])
        defer { fixture.stop() }
        let row = try #require(await catalogRow(fixture.service))
        #expect(row.precisionStats["bf16"]?.value(.tokensPerSecond).value == 42)
        let entry = try #require(await fixture.service.catalog(query: .init()).first)
        #expect(entry.runners.first?.stats["bf16"]?.value(.tokensPerSecond).state == .quickCheck)
    }

    @Test func localResultsAreVisibleAndDoNotWeakenSignedFolderChecks() async throws {
        let fixture = try await makeLocalCatalogFixture(kinds: ["chat"])
        defer { fixture.stop() }
        let folder = fixture.paths.directory("models").appendingPathComponent("fixture-chat")
        let installer = ModelFolderInstaller(support: fixture.paths.support)
        _ = try await installer.accept(data: fixture.catalogData, signature: fixture.signature, publicKey: fixture.publicKey)
        _ = try await installer.install(id: "fixture-chat", source: fixture.source)
        let model = try await installer.verify(installedFolder: folder)
        var stats = PrecisionStatSet(kind: .chat, dtype: "bf16", runner: "native")
        let source = "weights " + fixture.revision + "; adapter sha256 " + String(repeating: "0", count: 64)
        let p = StatProvenance(
            chip: HostChip.current, date: "2026-10-09T12:00:00Z", method: "fixture", baseline: "own bf16", source: source, revision: fixture.revision, run: "fixture-run")
        stats.availability = StatValue(state: .quickCheck, provenance: p)
        stats.set(.tokensPerSecond, StatValue(state: .quickCheck, value: 43, provenance: p))
        let result = PrecisionStatsResult(model: model.id, mode: "fast", stats: stats)
        let file = folder.appendingPathComponent("results/local/fixture.md")
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try result.markdown().write(to: file)
        _ = try await installer.verify(installedFolder: folder)
        #expect(LocalPrecisionResults.read(folder: folder, model: model).count == 1)
        #expect(await catalogRow(fixture.service)?.precisionStats["bf16"]?.value(.tokensPerSecond).value == 43)
        // A stale revision stays historical; it cannot supply this model's default.
        var stale = result; stale.stats.availability.provenance?.revision = "old"
        try stale.markdown().write(to: file)
        #expect(LocalPrecisionResults.read(folder: folder, model: model).isEmpty)
        try Data("arbitrary code".utf8).write(to: file.deletingLastPathComponent().appendingPathComponent("run.swift"))
        await #expect(throws: (any Error).self) { try await installer.verify(installedFolder: folder) }
        try FileManager.default.removeItem(at: file.deletingLastPathComponent().appendingPathComponent("run.swift"))
        try Data("invalid front matter".utf8).write(to: file)
        await #expect(throws: (any Error).self) { try await installer.verify(installedFolder: folder) }
        await fixture.service.shutdown()
    }

    @Test func typedRecommendationRequiresComparableControlledEvidence() {
        func set(_ dtype: String, _ speed: Double, state: StatState = .measured) -> PrecisionStatSet {
            var stats = PrecisionStatSet(kind: .chat, dtype: dtype, runner: "native")
            let p = StatProvenance(
                chip: "m5-max", date: "2026-10-09T12:00:00Z", method: "fixed 32-token window at 128 prompt tokens", baseline: "own bf16",
                revision: String(repeating: "a", count: 40), run: "fixture-run")
            stats.availability = StatValue(state: state, provenance: p)
            stats.set(.tokensPerSecond, StatValue(state: state, value: speed, provenance: p))
            stats.set(.nextTokenAgreementPercent, StatValue(state: state, value: 100, provenance: p))
            return stats
        }
        let sets = ["bf16": set("bf16", 20), "int8": set("int8", 30), "int4": set("int4", 60, state: .quickCheck)]
        #expect(PrecisionRecommendation.choose(sets, baseDtype: "bf16", chip: "m5-max").dtype == "int8")
        #expect(!PrecisionRecommendation.choose(sets, baseDtype: "bf16", chip: "m5-max").pending)
        #expect(PrecisionRecommendation.choose(sets, baseDtype: "bf16", chip: "m4-max").pending)
        var failed = sets; failed["int8"]?.availability = .unavailable("Quality gate failed")
        #expect(PrecisionRecommendation.choose(failed, baseDtype: "bf16", chip: "m5-max").dtype == "bf16")
    }

    @Test func removedInt8SupportOverridesHistoricalPassingResult() throws {
        var stats = PrecisionStatSet(kind: .chat, dtype: "int8", runner: "native")
        let p = StatProvenance(
            chip: "m5-max", date: "2026-10-08T12:00:00Z", method: "fixture", baseline: "own bf16", revision: String(repeating: "a", count: 40), run: "fixture-run")
        stats.qualityCheckPassed = true
        stats.availability = StatValue(state: .measured, provenance: p)
        stats.set(.tokensPerSecond, StatValue(state: .measured, value: 60, provenance: p))
        let history = [PrecisionStatsResult(model: "fixture", mode: "fast", stats: stats)]
        let before = PrecisionStatsReader.sets(
            kind: "chat", baseDtype: "bf16", runner: "native", supported: ["bf16", "int8"], results: history, chip: "m5-max")
        #expect(before["int8"]?.availability.state == .measured)
        #expect(before["int8"]?.value(.tokensPerSecond).value == 60)

        // The revised manifest removes int8 but retains the old results file.
        let after = PrecisionStatsReader.sets(kind: "chat", baseDtype: "bf16", runner: "native", supported: ["bf16"], results: history, chip: "m5-max")
        let unavailable = try #require(after["int8"])
        #expect(unavailable.availability.state == .unavailable)
        #expect(unavailable.availability.reason == "This runner does not support this precision.")
        #expect(unavailable.metrics.values.allSatisfy { $0.state == .unavailable && $0.value == nil })
        try unavailable.validate()

        // A newly selected runner cannot inherit the previous runner's passed gate either.
        let other = PrecisionStatsReader.sets(
            kind: "chat", baseDtype: "bf16", runner: "future", supported: ["bf16", "int8"], results: history, chip: "m5-max")
        #expect(other["int8"]?.availability.state == .notMeasured)
        #expect(other["int8"]?.selectionUnavailableReason == nil)
        #expect(other["int8"]?.value(.tokensPerSecond).value == nil)
        #expect(other["int8"]?.runner == "future")
    }

    @Test func readerPrefersHostAndNewestFailureByParsedInstant() {
        func result(_ chip: String, _ date: String, failed: Bool) -> PrecisionStatsResult {
            var set = PrecisionStatSet(kind: .chat, dtype: "bf16", runner: "native")
            set.availability = failed ? .unavailable("Quality failed") : StatValue(state: .measured)
            set.availability.provenance = StatProvenance(chip: chip, date: date, method: "fixture")
            return .init(model: "fixture", mode: "exact", stats: set)
        }
        let results = [
            result("m4-max", "2026-10-09T12:00:00Z", failed: false), result("m5-max", "2026-10-09T10:00:00Z", failed: false),
            result("m5-max", "2026-10-09T09:00:01-01:00", failed: true)
        ]
        let sets = PrecisionStatsReader.sets(kind: "chat", baseDtype: "bf16", runner: "native", supported: ["bf16"], results: results, chip: "m5-max")
        #expect(sets["bf16"]?.availability.state == .unavailable)
        let unsupported = PrecisionStatsReader.sets(kind: "chat", baseDtype: "bf16", runner: "native", supported: [], results: results, chip: "m5-max")
        #expect(unsupported["bf16"]?.availability.reason == "This runner does not support this precision.")
    }

}
