// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// A quick check as d1-3B's results file records one: quick profile, no measurement lock, contended samples, gate
/// passed, a note in its own words.
func quickResult(chip: String, mode: String = "exact", note: String = "suite-v1 quick recipe, in-process; not a controlled measurement.") -> ResultsFile {
    let busy = ConditionSample(at: "2026-10-08T06:00:00Z", phase: "run", cpu: 400, gpu: 30, compositing: 5)
    var result = ResultsFile(
        model: "fixture-decisions", kind: "decisions", chip: chip, osBuild: "test", appBuild: "test", runner: "native",
        workerDigest: String(repeating: "a", count: 64), precision: "bf16", mode: mode,
        suite: .init(id: "arsenal-decisions-suite-v1", sha256: String(repeating: "b", count: 64)), profile: "quick", stub: false, gatePassed: true,
        conditions: .init(locked: false, idle: [busy, busy, busy], run: [busy], note: note),
        metrics: ["accuracy": 0.75, "latency_s": 0.0198, "batch_questions_per_s": 274, "peak_ram_mb": 6439], cases: [])
    result.date = "2026-10-08T06:20:00Z"
    return result
}

@Test func quickCheckIsClassifiedByItsFieldsNotItsNote() {
    let quick = quickResult(chip: "m5-max")
    #expect(quick.quickCheck && quick.verificationLabel == "quick check" && !quick.eligibleMeasurement)
    #expect(MeasurementBadge.derive(results: [quick], chip: "m5-max") == .unverified)
    var locked = quick; locked.conditions = .init(locked: true, idle: [], run: [])
    #expect(!locked.quickCheck)
    var full = quick; full.profile = "full"
    #expect(!full.quickCheck && full.verificationLabel == "contaminated")
    var stub = quick; stub.stub = true
    #expect(!stub.quickCheck && stub.verificationLabel == "stub")
}

@Test func referenceFallsBackToALabelledQuickCheckOnlyWithoutAMeasurement() throws {
    let quick = quickResult(chip: "m5-max", mode: "fast")
    let rough = try #require(ReferenceFigures.best([quick], chip: "m5-max", dtype: "bf16", mode: "fast"))
    #expect(rough.verification == "quick check" && rough.match == .sameChip && rough.metrics["latency_s"] == 0.0198)
    let measured = v3MeasuredResult(model: "fixture-decisions", chip: "m4-max")
    #expect(ReferenceFigures.best([quick, measured], chip: "m5-max", dtype: "bf16", mode: "fast")?.verification == "Measured")
    #expect(ReferenceFigures.best([quick], chip: "m5-max", dtype: "bf16", mode: "exact") == nil)
    let object = try WireJSON.decode(JSONEncoder().encode(rough))
    #expect(object["verification"] == "quick check")
}

/// A single-mode runner (`fast_exact = false`) runs Exact even when the stored preference says Fast; its exact quick
/// check must reach the row as figures labelled "quick check", context excluded from the rough figures.
@Test func singleModeRunnerRowShowsItsQuickCheck() async throws {
    let result = quickResult(chip: HostChip.current)
    let data = try JSONEncoder().encode(result)
    let fixture = try await makeLocalCatalogFixture(
        kinds: ["decisions"], context: 32768,
        extraTable: [
            "runner": .array([
                .table([
                    "name": .string("native"), "language": .string("swift"), "worker": .string("stub"), "protocol": .integer(1),
                    "chips": .array([.string("*")]), "min_macos": .string("26.0"), "fast_exact": .bool(false)
                ])
            ])
        ], extraFiles: ["results/\(HostChip.current).json": data])
    defer { fixture.stop() }
    let row = try #require(await catalogRow(fixture.service, "fixture-decisions"))
    #expect(row.fast == false)
    #expect(row.verification == "quick check")
    #expect(row.metrics["accuracy"] == "75.0%" && row.metrics["latency"] == "20 ms" && row.metrics["peak_ram"] == "6.44 GB")
    #expect(row.metrics["context"] == .number(32768))
    #expect(row.quickMetrics["accuracy"] == true && row.quickMetrics["context"] == nil)
    #expect(row.peakRAMMB == 6439)
    let reference = try #require(try await fixture.service.reference(id: "fixture-decisions"))
    #expect(reference.verification == "quick check" && reference.mode == "exact")
}
