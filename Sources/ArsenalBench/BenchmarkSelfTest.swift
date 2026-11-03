// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// CPU-only executable checks also work without XCTest runtime search paths on CLT-only hosts.
public enum BenchmarkSelfTest {
    public static func run(repository: URL) async throws {
        let root = URL(fileURLWithPath: "/tmp/arsenal-bench/self-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let service = StubModelsService()
        var results: [ResultsFile] = []
        for kind in ["voices", "decisions", "images", "chat"] {
            let id = "stub-\(kind)"; try await service.get(id: id, precision: "bf16")
            let result = try await BenchmarkHarness(service: service, configuration: .init(repository: repository, outputDirectory: root, stub: true)).run(.init(id: id))
            try ResultValidation.validate(data: JSONEncoder().encode(result))
            guard result.stub, !result.eligibleMeasurement, result.cases.count >= 2,
                MeasurementBadge.derive(results: [result], chip: result.chip) == .unverified
            else { throw BenchmarkError.invalid("Stub result incorrectly qualifies as measured.") }
            results.append(result)
        }
        guard BenchmarkRanking.build(results: results, chip: results[0].chip).isEmpty else { throw BenchmarkError.invalid("Stub results entered runner ranking.") }
        let loaded = ConditionProbe { phase in ConditionSample(at: "test", phase: phase, cpu: 251, gpu: 0, compositing: 0) }
        do {
            _ = try await BenchmarkHarness(service: service, configuration: .init(repository: repository, outputDirectory: root, stub: true), probe: loaded).run(
                .init(id: "stub-voices"))
            throw BenchmarkError.invalid("Contaminated window was accepted.")
        } catch let error as BenchmarkError { guard error.localizedDescription.contains("contaminated") else { throw error } }
        let allowedContended = try await BenchmarkHarness(service: service, configuration: .init(repository: repository, outputDirectory: root, stub: true), probe: loaded).run(
            .init(id: "stub-voices", quickCheck: true))
        guard allowedContended.conditions.contended, allowedContended.conditions.idle_samples.count == 3,
            allowedContended.cases.count == 2, !allowedContended.conditions.measurement_lock
        else {
            throw BenchmarkError.invalid("Contended quick check was refused or stopped early.")
        }
        let duringLoad = ConditionProbe { phase in ConditionSample(at: "test", phase: phase, cpu: 0, gpu: phase == "run" ? 11 : 0, compositing: 0) }
        let dirty = try await BenchmarkHarness(service: service, configuration: .init(repository: repository, outputDirectory: root, stub: true), probe: duringLoad).run(
            .init(id: "stub-voices"))
        guard dirty.conditions.contaminated else { throw BenchmarkError.invalid("During-run GPU contamination was not flagged.") }
        // A private fake repo proves the gate refuses a hold without changing the team's gate.
        let fakeRepo = root.appendingPathComponent("repo")
        try FileManager.default.createDirectory(at: fakeRepo, withIntermediateDirectories: true)
        try Data("test gate".utf8).write(to: fakeRepo.appendingPathComponent(".gpu-hold"))
        do { try BenchmarkGate.check(repository: fakeRepo, requireLock: false); throw BenchmarkError.invalid("Closed gate was accepted.") } catch let error as BenchmarkError {
            guard error.localizedDescription.contains("GPU gate closed") else { throw error }
        }
        do { try BenchmarkGate.check(repository: repository, environment: [:]); throw BenchmarkError.invalid("Missing measurement lock was accepted.") } catch let error
            as BenchmarkError
        { guard error.localizedDescription.contains("measure") || error.localizedDescription.contains("GPU gate closed") else { throw error } }
        var valid = results[0]
        valid.stub = false; valid.gate_passed = true; valid.conditions.measurement_lock = true
        valid.runner = "slow"; valid.metrics["speed_x_realtime"] = 2
        var fast = valid; fast.runner = "fast"; fast.metrics["speed_x_realtime"] = 4
        guard BenchmarkRanking.build(results: [valid, fast], chip: valid.chip)[valid.model] == ["fast", "slow"],
            MeasurementBadge.derive(results: [valid], chip: valid.chip) == .measured,
            MeasurementBadge.derive(results: [valid], chip: "other-chip") == .anotherChip
        else { throw BenchmarkError.invalid("Ranking or badge derivation failed.") }
        var quick = valid; quick.conditions.measurement_lock = false; quick.conditions.note = "Quick check under gpulock; not controlled measurement."
        guard quick.verificationLabel == "quick check", !quick.eligibleMeasurement,
            MeasurementBadge.derive(results: [quick], chip: quick.chip) == .unverified,
            BenchmarkRanking.build(results: [quick], chip: quick.chip).isEmpty
        else { throw BenchmarkError.invalid("Quick check was mislabelled Measured or ranked.") }
        var incompatible = fast; incompatible.suite.sha256 = String(repeating: "f", count: 64); incompatible.date = "2000-01-01T00:00:00Z"
        guard BenchmarkRanking.build(results: [valid, incompatible], chip: valid.chip)[valid.model] == ["slow"] else {
            throw BenchmarkError.invalid("Different suites were compared.")
        }
        var corrupt = valid; corrupt.conditions.idle_samples[0].other_cpu_pct = 300
        do { try ResultValidation.validate(data: JSONEncoder().encode(corrupt)); throw BenchmarkError.invalid("False telemetry validity was accepted.") } catch let error
            as BenchmarkError
        { guard error.localizedDescription.contains("validity") else { throw error } }
        let log =
            "+++\nhypothesis = \"Test\"\nchange = \"Test\"\nmeasurement = \"CPU only\"\nverdict = \"inconclusive\"\nnumbers = { runs = 0 }\nlinks = []\n+++\nNo real measurements.\n"
        try ResearchLogValidation.validate(markdown: log)
        let yaml = "---\nhypothesis: Test\nchange: Test\nmeasurement: CPU only\nverdict: inconclusive\nnumbers: No real runs\nlinks: [\"evidence.txt\"]\n---\nEvidence.\n"
        try ResearchLogValidation.validate(markdown: yaml)
        do { try ResearchLogValidation.validate(markdown: "No header"); throw BenchmarkError.invalid("Malformed log was accepted.") } catch let error as BenchmarkError {
            guard error.localizedDescription.contains("front matter") else { throw error }
        }
        let completedResult = valid
        let controller = BenchmarkController(execute: { _ in completedResult })
        let job = try await controller.start(.init(id: "stub-voices"))
        for _ in 0..<50 {
            if await controller.status(job.job)?.state == "succeeded" { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        guard await controller.status(job.job)?.state == "succeeded" else { throw BenchmarkError.invalid("Benchmark job failed to complete.") }
        // Only stub data is written; retain CPU evidence but never commit it into model folders.
        try Data("all four stub suites; injected load refusal; closed gate refusal; validators; badges; ranking; jobs passed\n".utf8).write(
            to: root.appendingPathComponent("checks.txt"))
    }
}
