// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import Testing

struct VoiceIntakeBlockedTests {
    @Test func allVoicePreflightsRetainEveryTierWithoutPublishingUnmeasuredFigures() throws {
        for model in ["breeze-tts-2", "qwen3-tts-base", "qwen3-tts-voicedesign", "pocket-tts"] {
            let folder = intakeFixtureRepository.appendingPathComponent("models/" + model)
            let runs = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("gate/evidence/intake"), includingPropertiesForKeys: nil)
            #expect(!runs.isEmpty)
            let run = try #require(runs.first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("blocked-result.json").path) })
            let result = try WireJSON.decode(Data(contentsOf: run.appendingPathComponent("blocked-result.json")))
            let rows = try #require(result["rows"]?.array)
            #expect(rows.count == 3 && Set(rows.compactMap { $0["precision"]?.string }) == ["fp16", "int8", "int4"])
            #expect(result["model_executions"]?.number == 0 && result["quality_bindings_written"]?.number == 0)
            for row in rows {
                #expect(row["state"]?.string == "blocked")
                #expect(row["qualified"] == .null && row["peak_gb"] == .null && row["warm_seconds"] == .null && row["quality_binding"] == .null)
                #expect(row["blocker"]?["code"]?.string == "half-reference-not-qualified")
            }
            let receipt = try WireJSON.decode(Data(contentsOf: run.appendingPathComponent("source-copy-receipt.json")))
            #expect(receipt["before_sha256"] == receipt["after_sha256"] && receipt["unchanged"]?.bool == true)
            let coverage = try WireJSON.decode(Data(contentsOf: run.appendingPathComponent("coverage.json")))
            #expect(coverage["packed_matrices"]?.number == 0)
            #expect(coverage["tensors"]?.array?.isEmpty == false)
        }
    }
}
