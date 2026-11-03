// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct SoundRoundTwoEvidenceTests {
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    @Test func smallFP16IntakeCannotTreatAAAsSourceQualification() throws {
        let model = try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/stable-audio-3-small-sfx"))
        let dummy = root.appendingPathComponent("tmp/absent-qualification")
        #expect(throws: (any Error).self) {
            try ModelIntakeCommand.qualifySoundFormat(model: model, path: nil, replay: dummy, evidence: dummy, reproducibility: ["component_gate_passed": true])
        }
        #expect(throws: (any Error).self) {
            try ModelIntakeCommand.qualifySoundFormat(model: model, path: dummy.path, replay: dummy, evidence: dummy, reproducibility: ["component_gate_passed": true])
        }
    }
    @Test(arguments: ["stable-audio-3-medium", "stable-audio-3-small-sfx"])
    func currentSoundResultsBindAllThreeOutcomes(id: String) throws {
        let folder = intakeFixtureRepository.appendingPathComponent("models/" + id)
        let model = try ModelManifest.load(folder)
        let binding = try latestIntakeBinding(folder)
        let run = try #require(binding["run"]?.string)
        let resultPath = folder.appendingPathComponent("results/intake/" + run + ".json")
        let result = try WireJSON.decode(Data(contentsOf: resultPath))
        #expect(binding["implementation_identity"] == result["implementation_identity"])
        #expect(binding["worker_sha256"] == result["worker_sha256"])
        let outcomes = try #require(binding["outcomes"]?.array)
        #expect(outcomes.count == 3)
        #expect(Set(outcomes.compactMap { $0["dtype"]?.string }) == Set(["fp16", "int8", "int4"]))
        for outcome in outcomes {
            let dtype = try #require(outcome["dtype"]?.string)
            let data = try Data(contentsOf: folder.appendingPathComponent(try #require(outcome["result"]?.string)))
            let stats = try PrecisionStatsResult.decode(data).stats
            #expect(stats.qualityCheckPassed == (dtype != "int4"))
            #expect(outcome["passed"]?.bool == stats.qualityCheckPassed)
            #expect(outcome["result_sha256"]?.string == FileDigest.sha256(data))
            #expect(outcome["capture_result_sha256"]?.string == (try FileDigest.sha256(resultPath)))
            #expect(outcome["recipe"]?.string == model.derive?[dtype])
            #expect(stats.qualityIdentity == outcome["quality_identity"]?.string)
            #expect(stats.value(.realTimeFactor).provenance?.run == run + "-" + dtype)
            if id == "stable-audio-3-small-sfx" && dtype == "fp16" {
                #expect(outcome["format_qualification_sha256"]?.string?.count == 64)
            }
        }
    }
}
