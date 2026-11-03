// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

func v3MeasuredResult(model: String = "fixture-chat", chip: String, dtype: String = "bf16", mode: String = "fast", speed: Double = 42) -> ResultsFile {
    let sample = ConditionSample(at: "2026-10-06T00:00:00Z", phase: "test", cpu: 0, gpu: 0, compositing: 0)
    var result = ResultsFile(
        model: model, kind: "chat", chip: chip, osBuild: "test", appBuild: "test", runner: "native", workerDigest: String(repeating: "a", count: 64),
        precision: dtype, mode: mode, suite: .init(id: "chat-v1", sha256: String(repeating: "b", count: 64)), profile: "full", stub: false, gatePassed: true,
        conditions: .init(locked: true, idle: [sample, sample, sample], run: [sample]), metrics: ["decode_tokens_per_s": speed, "unknown": nil], cases: [])
    result.date = "2026-10-06T00:00:00Z"
    return result
}

@Test func v3ReferenceLookupPrefersChipThenFamilyThenNearest() {
    let same = v3MeasuredResult(chip: "m5-max"), family = v3MeasuredResult(chip: "m5"), nearest = v3MeasuredResult(chip: "m4-ultra")
    #expect(ReferenceFigures.best([nearest, family, same], chip: "m5-max", dtype: "bf16", mode: "fast")?.match == .sameChip)
    #expect(ReferenceFigures.best([nearest, family], chip: "m5-max", dtype: "bf16", mode: "fast")?.match == .sameFamily)
    #expect(ReferenceFigures.best([nearest], chip: "m5-max", dtype: "bf16", mode: "fast")?.match == .nearest)
    #expect(ReferenceFigures.best([nearest], chip: "m5-max", dtype: "bf16", mode: "fast")?.chip == "m4-ultra")
    #expect(ReferenceFigures.chipOrder.count == 20 && Set(ReferenceFigures.chipOrder).count == 20)
}

@Test func v3ReferenceLookupFiltersDtypeModeAndUnmeasuredResults() throws {
    var same = v3MeasuredResult(chip: "m5-max"), old = same
    old.date = "2026-10-05T00:00:00Z"
    #expect(ReferenceFigures.best([old, same], chip: "m5-max", dtype: "bf16", mode: "fast")?.date == same.date)
    #expect(ReferenceFigures.best([same], chip: "m5-max", dtype: "int4", mode: "fast") == nil)
    #expect(ReferenceFigures.best([same], chip: "m5-max", dtype: "bf16", mode: "exact") == nil)
    same.stub = true
    #expect(ReferenceFigures.best([same], chip: "m5-max", dtype: "bf16", mode: "fast") == nil)
    same.stub = false; same.conditions.contaminated = true
    #expect(ReferenceFigures.best([same], chip: "m5-max", dtype: "bf16", mode: "fast") == nil)
    #expect(ReferenceFigures.best([old], chip: "unknown-chip", dtype: "bf16", mode: "fast")?.match == .nearest)
    let olderChip = v3MeasuredResult(chip: "m1")
    #expect(ReferenceFigures.best([olderChip, old], chip: "m6-max", dtype: "bf16", mode: "fast")?.chip == "m5-max")
    let object = try WireJSON.decode(JSONEncoder().encode(ReferenceFigures.best([old], chip: "m5-max", dtype: "bf16", mode: "fast")))
    #expect(object["match"] == "same_chip" && object["metrics"]?["unknown"] == .null)
}

@Test func v3ReferenceTiesUseFixedChipOrder() {
    let pro = v3MeasuredResult(chip: "m5-pro"), ultra = v3MeasuredResult(chip: "m5-ultra")
    #expect(ReferenceFigures.best([ultra, pro], chip: "m5-max", dtype: "bf16", mode: "fast")?.chip == "m5-pro")
}

@Test func v3ToolsManifestRowsCatalogAndPreferenceRoundTrip() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], extraTable: ["tools": .array([.string("chat")])]); defer { f.stop() }
    // Recorded by Add (it joins the table with its download), it starts out of the menu; Show in menu turns it on and off.
    let row = try #require(await catalogRow(f.service, "fixture-chat"))
    #expect(row.tools == ["chat"] && !row.toolsShown && row.activity == "chat")
    try await f.service.setToolsShown(id: row.id, on: true)
    #expect(await catalogRow(f.service, row.id)?.toolsShown == true)
    try await f.service.setToolsShown(id: row.id, on: false)
    #expect(await catalogRow(f.service, row.id)?.toolsShown == false)
    let entry = try #require(await f.service.catalog(query: .init()).first)
    let object = try WireJSON.decode(JSONEncoder().encode(entry))
    #expect(object["tools"] == ["chat"] && object["toolsShown"] == false && object["row"]?["toolsShown"] == false)
    try await f.service.get(id: row.id, precision: "bf16")
    try await f.service.load(id: row.id, precision: "bf16", fast: false)
    #expect(await f.service.rows().first?.toolsShown == false)
    #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).selections[row.id]?.toolsShown == false)
    await f.service.shutdown()
}

@Test func v3FolderToolsValidationAndOldPreferencesDefaults() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let folder = root.appendingPathComponent("models/qwen3.8-27b")
    var model = try ModelManifest.load(folder, kindsDirectory: root.appendingPathComponent("kinds"))
    #expect(model.tools == ["chat"] && model.activitySection == "chat")
    model.tools = ["chat", "chat", "invalid"]; model.activity = "../bad"
    let errors = model.problems(folder: folder)
    #expect(errors.contains { $0.contains("tools") } && errors.contains { $0.contains("activity") })
    let old = try JSONDecoder().decode(ModelPreferences.self, from: Data(#"{"precision":"bf16","fast":false,"keepHot":true}"#.utf8))
    #expect(old.toolsShown && !old.fast && old.keepHot)
}

func typedChatMeasurements() throws -> [String: Data] {
    try Dictionary(
        uniqueKeysWithValues: [("bf16", 40.0), ("int4", 60.0)].map { dtype, speed in
            var stats = PrecisionStatSet(kind: .chat, dtype: dtype, runner: "native")
            let p = StatProvenance(
                chip: HostChip.current, date: "2026-10-09T00:00:00Z", method: "fixed 32-token fixture", baseline: "own bf16", revision: String(repeating: "a", count: 40),
                run: "fixture-run")
            stats.availability = StatValue(state: .measured, provenance: p)
            stats.set(.tokensPerSecond, StatValue(state: .measured, value: speed, provenance: p))
            stats.set(.nextTokenAgreementPercent, StatValue(state: .measured, value: 100, provenance: p))
            return ("results/typed-" + dtype + ".md", try PrecisionStatsResult(model: "fixture-chat", mode: "fast", stats: stats).markdown())
        })
}

@Test func v3AddSelectsMeasuredRecommendedDtypeAndFast() async throws {
    let result = v3MeasuredResult(chip: HostChip.current, dtype: "int4", speed: 60)
    let f = try await makeLocalCatalogFixture(
        kinds: ["chat"], addToTable: false, extraFiles: try typedChatMeasurements().merging(["results/measured.json": JSONEncoder().encode(result)]) { a, _ in a })
    defer { f.stop() }
    try await f.service.add(id: "fixture-chat")
    let row = try #require(await catalogRow(f.service, "fixture-chat"))
    #expect(row.selectedPrecision == "int4" && row.fast)
    #expect(row.recommendedPrecision == nil && !row.recommendationPending)
    #expect(try await f.service.reference(id: "fixture-chat", dtype: "int4", mode: "fast")?.chip == HostChip.current)
    await f.service.shutdown()
}

@Test func v3DtypeRecommendationCannotCompareDifferentSuites() {
    var base = v3MeasuredResult(chip: "m5", speed: 40), int4 = v3MeasuredResult(chip: "m5", dtype: "int4", speed: 60)
    #expect(DtypeRecommendation.best([base, int4], kind: "chat") == "int4")
    base.suite = .init(id: "new-suite", sha256: String(repeating: "c", count: 64)); base.date = "2026-10-07T00:00:00Z"
    #expect(DtypeRecommendation.best([base, int4], kind: "chat") == "bf16")
    int4.gate_passed = false
    #expect(DtypeRecommendation.best([int4], kind: "chat") == nil)
}

@Test func v3AddPreservesExplicitDtypeAndAutomaticallyRetiresExactAcrossRemoveAndReadd() async throws {
    let result = v3MeasuredResult(chip: HostChip.current, dtype: "int4", speed: 60)
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], extraFiles: ["results/measured.json": try JSONEncoder().encode(result)])
    defer { f.stop() }
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    try await f.service.load(id: "fixture-chat", precision: "bf16", fast: false)
    try await f.service.remove(id: "fixture-chat", deleteWeights: false)
    try await f.service.add(id: "fixture-chat")
    let row = try #require(await f.service.rows().first)
    #expect(row.selectedPrecision == "bf16" && row.fast)
    let saved = try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json"))
    #expect(saved.selections[row.id]?.precision == "bf16" && saved.selections[row.id]?.fast == true)
    await f.service.shutdown()
}

@Test func v3SettingsBeforeAddDoNotChooseDtype() async throws {
    let result = v3MeasuredResult(chip: HostChip.current, dtype: "int4", speed: 60)
    let f = try await makeLocalCatalogFixture(
        kinds: ["chat"], addToTable: false, extraFiles: try typedChatMeasurements().merging(["results/measured.json": JSONEncoder().encode(result)]) { a, _ in a })
    defer { f.stop() }
    try await f.service.setToolsShown(id: "fixture-chat", on: false)
    try await f.service.setKeepHot(id: "fixture-chat", on: true)
    let file = f.paths.support.appendingPathComponent("config.json")
    #expect(try ArsenalConfiguration.read(file).selections["fixture-chat"]?.explicitSelection == false)
    try await f.service.add(id: "fixture-chat")
    let row = try #require(await catalogRow(f.service, "fixture-chat"))
    #expect(row.selectedPrecision == "int4" && row.fast)
    #expect(row.recommendedPrecision == nil && !row.recommendationPending)
    #expect(!row.toolsShown && row.keepHot)
    #expect(try ArsenalConfiguration.read(file).selections[row.id]?.explicitSelection == true)
    await f.service.shutdown()
}

@Test(arguments: ["2026-10-08T10:00:00.500Z", "2026-10-08T09:00:01-01:00"])
func resultSelectorsCompareParsedInstants(newest: String) throws {
    var old = v3MeasuredResult(chip: "m5-max", dtype: "int4", speed: 60), new = old
    old.date = "2026-10-08T10:00:00Z"
    new.date = newest; new.precision = "bf16"
    new.suite = .init(id: "new-suite", sha256: "new")
    #expect(try #require(new.recordedAt) > #require(old.recordedAt))
    #expect(ReferenceFigures.best([old, new], chip: "m5-max", dtype: "bf16", mode: "fast")?.date == new.date)
    var sameDtype = old; sameDtype.precision = "bf16"
    #expect(ReferenceFigures.best([sameDtype, new], chip: "m5-max", dtype: "bf16", mode: "fast")?.date == new.date)
    #expect(DtypeRecommendation.best([old, new], kind: "chat") == "bf16")
    var invalid = new; invalid.date = "invalid-date"
    #expect(invalid.recordedAt == nil && !invalid.eligibleMeasurement)
    #expect(ReferenceFigures.best([invalid], chip: "m5-max", dtype: "bf16", mode: "fast") == nil)
    invalid.profile = "quick"; invalid.conditions.measurement_lock = false
    #expect(ReferenceFigures.best([invalid], chip: "m5-max", dtype: "bf16", mode: "fast") == nil)
    #expect(DtypeRecommendation.best([invalid], kind: "chat") == nil)
}

@Test(arguments: [
    "invalid-date", "", "2026-10-08T10:00:00Zjunk", "2026-10-08T10:00:00Z\n", "2026-02-30T10:00:00Z", "2026-13-01T10:00:00Z", "2026-10-08T24:00:00Z",
    "2026-10-08T10:00:00+24:00", "2026-10-08T10:00:00+02:60", "2026-10-08T10:00:00"
])
func invalidResultTimestampsFailClosed(date: String) {
    var result = v3MeasuredResult(chip: "m5-max"); result.date = date
    #expect(result.recordedAt == nil && !result.eligibleMeasurement)
}

@Test(arguments: ["2026-02-28T12:00:00+02:00", "2026-02-28T12:00:00+0200", "2026-02-28T09:00:00-01:00", "2026-02-28T10:00:00.000Z"])
func equivalentResultTimestampsParseToTheSameInstant(date: String) {
    var result = v3MeasuredResult(chip: "m5-max"), utc = result
    result.date = date; utc.date = "2026-02-28T10:00:00Z"
    #expect(result.recordedAt != nil && result.recordedAt == utc.recordedAt)
}
