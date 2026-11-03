// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct LegacyAvailabilityTests {
    private func correctQuickCheck() -> ResultsFile {
        var result = quickResult(chip: HostChip.current)
        result.runner = "native"
        result.mode = "fast"
        result.gate_passed = false
        result.metrics["accuracy"] = 1
        result.cases = [.init(id: "yes-0", wall: 0.01, metrics: ["accuracy": 1])]
        return result
    }

    @Test func contendedCorrectAnswersRemainAvailableAndLabelledQuick() throws {
        let result = correctQuickCheck()
        #expect(result.conditions.contended && !result.gate_passed)
        #expect(result.recordedQualityFailure == nil)
        let mapped = try #require(PrecisionStatsReader.legacy(result))
        #expect(mapped.stats.availability.state == .quickCheck)
        #expect(mapped.stats.availability.reason == nil)
        #expect(mapped.stats.value(.millisecondsPerDecision).state == .quickCheck)
        #expect(mapped.stats.value(.millisecondsPerDecision).value == 19.8)
        #expect(mapped.stats.value(.agreementPercent).state == .notMeasured)
        #expect(!result.eligibleMeasurement)
        try mapped.stats.validate()
    }

    @Test(arguments: ["decisions", "chat"])
    func answerMismatchStaysUnavailableWithCaseReason(kind: String) throws {
        var result = correctQuickCheck()
        result.kind = kind
        let key = kind == "chat" ? "quality_accuracy" : "accuracy"
        // Aggregate success must not hide a failed case, even on a busy Mac.
        result.metrics[key] = 1
        result.cases.append(.init(id: "wrong-0", wall: 0.01, metrics: [key: 0]))
        let mapped = try #require(PrecisionStatsReader.legacy(result))
        #expect(mapped.stats.availability.state == .unavailable)
        #expect(mapped.stats.availability.reason == "Recorded answer mismatch: wrong-0.")
        try mapped.stats.validate()
    }

    @Test(arguments: ["parity", "task"])
    func explicitQualityFailuresSurviveContentionAndFullRunFiltering(check: String) throws {
        var result = correctQuickCheck()
        result.profile = "full"
        result.checks = ModelChecksStatus(parity: check == "parity" ? .fail : .pass, task: check == "task" ? .fail : .pass)
        let mapped = try #require(PrecisionStatsReader.legacy(result))
        #expect(mapped.stats.availability.state == .unavailable)
        #expect(mapped.stats.availability.reason?.contains(check == "parity" ? "reference parity" : "task quality") == true)
        #expect(mapped.stats.value(.millisecondsPerDecision).state == .notMeasured)
        try mapped.stats.validate()
    }

    @Test func missingQualityEvidenceAndFailedSpeedCheckAreNotQualityFailure() throws {
        var result = correctQuickCheck()
        result.kind = "voices"
        result.metrics = ["round_trip_wer": nil, "speed_x_realtime": 10]
        result.cases = []
        result.checks = ModelChecksStatus()
        result.checks?.speed = .fail
        let mapped = try #require(PrecisionStatsReader.legacy(result))
        #expect(mapped.stats.availability.state == .quickCheck)
        #expect(mapped.stats.value(.realTimeFactor).state == .quickCheck)
        #expect(mapped.stats.value(.spectralDistance).state == .notMeasured)
    }

    private func mappedSet(_ results: [ResultsFile]) throws -> PrecisionStatSet {
        let mapped = try results.map { try #require(PrecisionStatsReader.legacy($0)) }
        return try #require(
            PrecisionStatsReader.sets(
                kind: results[0].kind, baseDtype: "bf16", runner: "native", supported: ["bf16"], results: mapped)["bf16"])
    }

    @Test(arguments: ["parity", "task"])
    func differentSuiteCannotClearRecordedCheckFailure(check: String) throws {
        var failed = correctQuickCheck()
        failed.kind = "voices"
        failed.cases = [.init(id: "short-0", wall: 0.01, metrics: [:])]
        failed.checks = .init(parity: check == "parity" ? .fail : .pass, task: check == "task" ? .fail : .pass)
        var passed = failed
        passed.date = "2026-10-08T07:20:00Z"
        passed.checks = .init(parity: .pass, task: .pass)
        passed.suite.id = "different-suite"
        let blocked = try mappedSet([failed, passed])
        #expect(blocked.availability.state == .unavailable)
        #expect(blocked.availability.reason == failed.recordedQualityFailure)
        passed.suite.id = failed.suite.id
        #expect(try mappedSet([failed, passed]).availability.state == .quickCheck)
        passed.cases = []
        #expect(try mappedSet([failed, passed]).availability.state == .unavailable)
    }

    @Test func newerSinglesCannotClearRecordedBatchFailure() throws {
        var failed = correctQuickCheck()
        failed.cases.append(.init(id: "batch", wall: 0.01, metrics: ["accuracy": 0]))
        var passed = correctQuickCheck()
        passed.date = "2026-10-08T07:20:00Z"
        let blocked = try mappedSet([failed, passed])
        #expect(blocked.availability.state == .unavailable)
        #expect(blocked.availability.reason == "Recorded answer mismatch: batch.")
        passed.cases.append(.init(id: "batch", wall: 0.01, metrics: ["accuracy": 1]))
        #expect(try mappedSet([failed, passed]).availability.state == .quickCheck)
    }

    @Test(arguments: ["decisions", "chat"])
    func newerTimingOnlyBatchCannotClearRecordedBatchFailure(kind: String) throws {
        let key = kind == "chat" ? "quality_accuracy" : "accuracy"
        var failed = correctQuickCheck()
        failed.kind = kind
        failed.metrics[key] = 1
        failed.cases[0].metrics = [key: 1]
        failed.cases.append(.init(id: "batch", wall: 0.01, metrics: [key: 0]))
        var newer = failed
        newer.date = "2026-10-08T07:20:00Z"
        newer.cases[1].metrics = ["batch_questions_per_s": 274]
        let blocked = try mappedSet([failed, newer])
        #expect(blocked.availability.state == .unavailable)
        #expect(blocked.availability.reason == "Recorded answer mismatch: batch.")
        newer.cases[1].metrics[key] = 1
        #expect(try mappedSet([failed, newer]).availability.state == .quickCheck)
    }

    @Test(arguments: [("decisions", "parity"), ("decisions", "task"), ("chat", "parity"), ("chat", "task")])
    func newerPassOmittingTimingOnlyCaseCannotClearCheckFailure(kind: String, check: String) throws {
        let key = kind == "chat" ? "quality_accuracy" : "accuracy"
        var failed = correctQuickCheck()
        failed.kind = kind
        failed.metrics[key] = 1
        failed.cases[0].metrics = [key: 1]
        failed.cases.append(.init(id: "batch", wall: 0.01, metrics: ["batch_questions_per_s": 274]))
        failed.checks = .init(parity: check == "parity" ? .fail : .pass, task: check == "task" ? .fail : .pass)
        var newer = failed
        newer.date = "2026-10-08T07:20:00Z"
        newer.checks = .init(parity: .pass, task: .pass)
        newer.cases.removeLast()
        #expect(try mappedSet([failed, newer]).availability.state == .unavailable)
        newer.cases.append(.init(id: "batch", wall: 0.01, metrics: ["batch_questions_per_s": 280]))
        #expect(try mappedSet([failed, newer]).availability.state == .quickCheck)
    }

    @Test func realLayaPassingAnswersClearEarlierFailureWithTimingOnlyBatch() throws {
        let passed = try realResult("laya-multilingual")
        try validateLegacyCases(passed)
        let batch = try #require(passed.cases.first { $0.id == "batch" })
        #expect(Set(batch.metrics.keys) == Set(["batch_questions_per_s"]))
        var failed = passed
        failed.date = "2026-10-01T08:59:49Z"
        let yes = try #require(failed.cases.firstIndex { $0.id == "yes-0" })
        failed.cases[yes].metrics["accuracy"] = 0
        func selected(_ results: [ResultsFile]) throws -> PrecisionStatSet {
            let mapped = try results.map { try #require(PrecisionStatsReader.legacy($0)) }
            return try #require(
                PrecisionStatsReader.sets(
                    kind: passed.kind, baseDtype: passed.precision, runner: passed.runner, supported: [passed.precision], results: mapped)[passed.precision])
        }
        #expect(try selected([failed]).availability.state == .unavailable)
        #expect(failed.recordedQualityFailure == "Recorded answer mismatch: yes-0.")
        let recovered = try selected([failed, passed])
        #expect(recovered.availability.state == .quickCheck)
        #expect(recovered.availability.reason == nil)
        #expect(recovered.availability.provenance?.date == passed.date)
        try recovered.validate()
        // A batch carrying an answer metric must still pass, unlike a timing-only batch.
        for accuracy in [0.0, 0.5, 2.0] {
            var wrongBatch = passed
            let index = try #require(wrongBatch.cases.firstIndex { $0.id == "batch" })
            wrongBatch.cases[index].metrics["accuracy"] = accuracy
            #expect(try selected([failed, wrongBatch]).availability.state == .unavailable)
        }
    }

    @Test func timingOnlyCheckCannotClearEarlierAnswerFailure() throws {
        var failed = correctQuickCheck()
        failed.cases[0].metrics["accuracy"] = 0
        var timing = correctQuickCheck()
        timing.date = "2026-10-08T07:20:00Z"
        timing.cases = []
        timing.metrics["accuracy"] = nil
        timing.metrics["latency_s"] = 0.03
        let set = try mappedSet([failed, timing])
        #expect(set.availability.state == .unavailable)
        #expect(set.availability.reason == "Recorded answer mismatch: yes-0.")
        #expect(set.availability.provenance?.date == failed.date)
        #expect(set.value(.millisecondsPerDecision).value == 30)
        #expect(set.value(.millisecondsPerDecision).provenance?.date == timing.date)
        try set.validate()
    }

    @Test func newerCompleteAnswersSupersedeFailureButPartialAnswersDoNot() throws {
        var failed = correctQuickCheck()
        failed.cases.append(.init(id: "no-0", wall: 0.01, metrics: ["accuracy": 0]))
        var partial = correctQuickCheck()
        partial.date = "2026-10-08T07:20:00Z"
        #expect(try mappedSet([failed, partial]).availability.state == .unavailable)
        var passed = partial
        passed.date = "2026-10-08T08:20:00Z"
        passed.cases.append(.init(id: "no-0", wall: 0.01, metrics: ["accuracy": 1]))
        #expect(try mappedSet([failed, partial, passed]).availability.state == .quickCheck)
        var qualityOnly = passed
        qualityOnly.profile = "full"
        let quality = try #require(PrecisionStatsReader.legacy(qualityOnly))
        #expect(quality.stats.availability.state == .notMeasured)
        #expect(quality.stats.metrics.values.allSatisfy { $0.state == .notMeasured && $0.value == nil })
        let recovered = try mappedSet([failed, qualityOnly])
        #expect(recovered.availability.state == .notMeasured && recovered.availability.reason == nil)
        #expect(recovered.value(.millisecondsPerDecision).provenance?.date == failed.date)
        try recovered.validate()
        // A different suite or a tied timestamp cannot erase this suite's recorded rejection.
        var otherSuite = passed
        otherSuite.suite.id = "different suite"
        #expect(try mappedSet([failed, otherSuite]).availability.state == .unavailable)
        var sameSuite = passed
        sameSuite.suite.version += 1
        sameSuite.suite.sha256 = "updated suite contents"
        #expect(try mappedSet([failed, sameSuite]).availability.state == .quickCheck)
        var tied = passed
        tied.date = failed.date
        #expect(try mappedSet([failed, tied]).availability.state == .unavailable)
    }

    @Test func newerV2MetricsCannotClearFailureWithoutSuiteAndCaseEvidence() throws {
        var failed = correctQuickCheck()
        failed.cases[0].metrics["accuracy"] = 0
        let failure = try #require(PrecisionStatsReader.legacy(failed))
        var stats = PrecisionStatSet(kind: .decisions, dtype: "bf16", runner: "native")
        let identity = StatProvenance(
            chip: HostChip.current, date: "2026-10-08T07:20:00Z", method: "Runner quality gate: fixed reference",
            source: "fixture", revision: "current", run: "current")
        stats.availability = StatValue(state: .quickCheck, provenance: identity)
        stats.set(.millisecondsPerDecision, StatValue(state: .quickCheck, value: 30, provenance: identity))
        func selected(_ stats: PrecisionStatSet) throws -> PrecisionStatSet {
            let newer = PrecisionStatsResult(model: failed.model, mode: failed.mode, stats: stats)
            return try #require(
                PrecisionStatsReader.sets(
                    kind: "decisions", baseDtype: "bf16", runner: "native", supported: ["bf16"], results: [failure, newer])["bf16"])
        }
        #expect(try selected(stats).availability.state == .unavailable)
        stats.set(.agreementPercent, StatValue(state: .quickCheck, value: 100, provenance: identity))
        #expect(try selected(stats).availability.state == .unavailable)
    }

    @Test func legacyAdapterMetadataDoesNotChangePublicResultSchema() throws {
        let mapped = try #require(PrecisionStatsReader.legacy(correctQuickCheck()))
        let encoded = try WireJSON.decode(JSONEncoder().encode(mapped))
        #expect(Set(encoded.keys) == Set(["schema", "model", "mode", "stats"]))
        #expect(try PrecisionStatsResult.decode(mapped.markdown()).stats == mapped.stats)
    }

    @Test func staleFailedCheckCannotBlockCurrentPassingAnswers() throws {
        var result = correctQuickCheck()
        result.gate_passed = true
        result.checks = .init(parity: .fail, task: .pass, stale: true)
        let mapped = try #require(PrecisionStatsReader.legacy(result))
        #expect(mapped.stats.availability.state == .quickCheck)
        #expect(mapped.stats.availability.reason == nil)
    }

    @Test(arguments: ["fingerprint", "fixtures", "retained", "stale", "current"])
    func localAvailabilityUsesCompatibleChecksAndRetainsHistory(change: String) async throws {
        let fixture = try await makeLocalCatalogFixture(kinds: ["decisions"])
        defer {
            fixture.stop()
            try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent())
        }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = fixture.paths.support.appendingPathComponent("local-models/fixture-layers")
        try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: root.appendingPathComponent("Tests/Fixtures/DropIn/fixture-layers"), to: folder)
        let evidence = folder.appendingPathComponent("results/retained.txt")
        try FileManager.default.createDirectory(at: evidence.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("passing implementation evidence".utf8).write(to: evidence)
        let manifestFile = folder.appendingPathComponent("model.toml")
        let manifest = try String(contentsOf: manifestFile, encoding: .utf8)
            .replacingOccurrences(of: "kind = \"local.layers\"", with: "kind = \"decisions\"")
            .replacingOccurrences(of: "kind_file = \"kinds/local.layers.toml\"\n", with: "")
        try manifest.write(to: manifestFile, atomically: true, encoding: .utf8)
        let local = try LocalModelFolders.discoverSnapshot(folder)
        var passing = correctQuickCheck()
        passing.model = local.manifest.id
        passing.runner = "swift"
        passing.mode = "exact"
        passing.gate_passed = true
        passing.checks = .init(
            parity: .pass, task: .pass, implementationFingerprint: try ImplementationEvidence.fingerprint(local),
            fixtureIdentity: ImplementationEvidence.fixtures(local), evidence: ["results/retained.txt"])
        passing.checks?.evidenceHashes = ["results/retained.txt": try FileDigest.sha256(evidence)]
        var failed = passing
        failed.date = "2026-10-08T07:20:00Z"
        failed.checks?.parity = .fail
        switch change {
        case "fingerprint": failed.checks?.implementationFingerprint = "previous implementation"
        case "fixtures": failed.checks?.fixtureIdentity = "previous fixtures"
        case "retained": failed.checks?.evidenceHashes = [:]
        case "stale": failed.checks?.stale = true
        default: break
        }
        let files = ["results/pass.json": try JSONEncoder().encode(passing), "results/fail.json": try JSONEncoder().encode(failed)]
        let set = try #require(await fixture.service.fixtureAvailability(local: local, files: files)["bf16"])
        #expect(set.availability.state == (change == "current" ? .unavailable : .quickCheck))
        #expect(set.availability.reason == (change == "current" ? "The recorded reference parity check failed." : nil))
        #expect(await fixture.service.resultCache[local.manifest.id]?.count == 2)
        await fixture.service.shutdown()
    }

    private func realResult(_ model: String) throws -> ResultsFile {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try JSONDecoder().decode(ResultsFile.self, from: Data(contentsOf: root.appendingPathComponent("models/\(model)/results/m5-max.json")))
    }

    private func validateLegacyCases(_ result: ResultsFile) throws {
        let ids: [String]
        switch result.model {
        case "laya-multilingual", "laya-english": ids = ["yes-0", "no-0", "choice-0", "score-0", "batch"]
        case "kokoro-82m": ids = ["short-0", "long-0"]
        case "flux2-klein-4b": ids = ["photo-0", "text-0"]
        default: throw ManifestError.invalid("Unexpected legacy model")
        }
        guard result.cases.count == ids.count, Set(result.cases.map(\.id)) == Set(ids) else {
            throw ManifestError.invalid("Legacy case coverage changed")
        }
        if result.kind == "decisions" {
            try #require(result.cases.filter { $0.id != "batch" }.allSatisfy { $0.metrics["accuracy"] == 1 })
        }
    }

    @Test(arguments: ["flux2-klein-4b", "laya-multilingual", "laya-english", "kokoro-82m"], ["missing", "duplicate", "renamed"])
    func realLegacyCoverageRejectsMissingDuplicateOrRenamedCases(model: String, change: String) throws {
        var result = try realResult(model)
        switch change {
        case "missing": result.cases.removeLast()
        case "duplicate": result.cases.append(result.cases[0])
        default: result.cases[0].id = "unexpected"
        }
        #expect(throws: (any Error).self) { try validateLegacyCases(result) }
    }

    @Test(arguments: ["flux2-klein-4b", "laya-multilingual", "laya-english", "kokoro-82m"])
    func realLegacyFilesRemainAvailableWithQuickFigures(model: String) throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let path = root.appendingPathComponent("models/\(model)/results/m5-max.json")
        let result = try JSONDecoder().decode(ResultsFile.self, from: Data(contentsOf: path))
        try validateLegacyCases(result)
        #expect(!result.gate_passed && result.quickCheck && result.conditions.contended)
        #expect(result.recordedQualityFailure == nil)
        let mapped = try #require(PrecisionStatsReader.legacy(result, source: path.path))
        let sets = PrecisionStatsReader.sets(
            kind: result.kind, baseDtype: result.precision, runner: result.runner, supported: [result.precision], results: [mapped], chip: result.chip)
        let set = try #require(sets[result.precision])
        #expect(set.availability.state == .quickCheck && set.availability.reason == nil)
        let mapping: [(StatMetric, String, Double)]
        switch result.kind {
        case "decisions": mapping = [(.millisecondsPerDecision, "latency_s", 1000), (.batchedDecisionsPerSecond, "batch_questions_per_s", 1)]
        case "voices": mapping = [(.firstAudioSeconds, "first_audio_s", 1)]
        default: mapping = [(.secondsPerImage, "seconds_per_image", 1)]
        }
        for (metric, key, scale) in mapping + [(.peakMemoryGB, "peak_ram_mb", 0.001)] {
            let expected = try #require(result.metrics[key] ?? nil) * scale
            #expect(set.value(metric).state == .quickCheck)
            #expect(set.value(metric).value == expected)
        }
        if result.kind == "voices" {
            #expect(result.metrics["round_trip_wer"] == .some(nil))
            #expect(set.value(.realTimeFactor).state == .quickCheck)
            #expect(set.value(.realTimeFactor).value == 1 / (try #require(result.metrics["speed_x_realtime"] ?? nil)))
        }
        if result.kind == "images" { #expect(result.metrics["reference_deviation"] == .some(nil)) }
        if result.kind == "decisions" { #expect(result.metrics["accuracy"] == 1) }
        #expect(set.metrics.values.allSatisfy { $0.state != .measured })
        try set.validate()
    }

    @Test func realServiceRowCarriesAvailableQuickStats() async throws {
        let result = correctQuickCheck()
        let fixture = try await makeLocalCatalogFixture(kinds: ["decisions"], extraFiles: ["results/quick.json": JSONEncoder().encode(result)])
        defer {
            fixture.stop()
            try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent())
        }
        let row = try #require(await catalogRow(fixture.service))
        #expect(row.precisionStats["bf16"]?.availability.state == .quickCheck)
        #expect(row.verification == "quick check" && row.quickMetrics["latency"] == true)
        #expect(row.dtypeUnavailableReason("bf16") == nil)
        await fixture.service.shutdown()
    }
}

private extension RealModelsService {
    func fixtureAvailability(local: LocalModelFolder, files: [String: Data]) -> [String: PrecisionStatSet] {
        resultCache[local.manifest.id] = files.values.compactMap { try? JSONDecoder().decode(ResultsFile.self, from: $0) }
        // Discovery assesses the newly scanned implementation before installing it in localModels.
        cachePrecisionEvidence(local.manifest, files: files, local: local)
        localModels[local.manifest.id] = local
        return statsForRunner(local.manifest, runner: "swift")
    }
}
