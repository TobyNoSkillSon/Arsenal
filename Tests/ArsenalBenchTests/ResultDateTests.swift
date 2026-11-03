// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import Testing
@testable import ArsenalBench

@Test(arguments: ["2026-10-08T10:00:00.500Z", "2026-10-08T09:00:01-01:00"])
func rankingsChooseNewestParsedSuiteAndRunner(date: String) {
    let sample = ConditionSample(at: "2026-10-08T10:00:00Z", phase: "test", cpu: 0, gpu: 0, compositing: 0)
    var old = ResultsFile(
        model: "fixture", kind: "chat", chip: "m5-max", osBuild: "test", appBuild: "test", runner: "native", workerDigest: "fixture",
        precision: "bf16", mode: "exact", suite: .init(id: "old", sha256: "old"), profile: "full", stub: false, gatePassed: true,
        conditions: .init(locked: true, idle: [sample, sample, sample], run: [sample]), metrics: ["decode_tokens_per_s": 100], cases: [])
    old.date = "2026-10-08T10:00:00Z"
    var new = old; new.date = date; new.suite = .init(id: "new", sha256: "new"); new.runner = "python"
    #expect(BenchmarkRanking.build(results: [old, new], chip: "m5-max")["fixture"] == ["python"])
    old.suite = new.suite
    var native = old; native.date = date; native.metrics = ["decode_tokens_per_s": 1]
    new.metrics = ["decode_tokens_per_s": 2]
    #expect(BenchmarkRanking.build(results: [old, native, new], chip: "m5-max")["fixture"] == ["python", "native"])
    var invalid = old; invalid.date = "invalid-date"; invalid.runner = "invalid"
    #expect(BenchmarkRanking.build(results: [invalid], chip: "m5-max").isEmpty)
}
