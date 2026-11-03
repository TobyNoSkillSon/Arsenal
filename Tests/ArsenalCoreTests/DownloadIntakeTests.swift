// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct DownloadIntakeTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @Test func downloadedAdaptersHaveExactKeysAndLanguageScopedStates() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("{\"num_hidden_layers\":28}".utf8).write(to: root.appendingPathComponent("config.json"))
        for id in ["von-1.1", "von-1.2", "re-use"] {
            let model = try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/" + id))
            let keys = try ModelIntakeCommand.downloadedComponentKeys(model, source: root)
            #expect(keys.count == (id == "re-use" ? 63 : 60))
            // RE-USE ships BF16 only since 22 Oct, Von since 26 Oct (under 1 GB at 16-bit) (removedQualityTiers).
            #expect(model.precisions == ["bf16"])
            if id != "re-use" { #expect(model.languages == ["en"] && ModelIntakeCommand.layaStates(model)[1].hasPrefix("The package")) }
        }
    }
    @Test func sourceGetRejectsUnapprovedModelAndInstalledSupportBeforeAnyNetwork() async throws {
        for id in ["laya-english", "von-1.1"] {
            await #expect(throws: (any Error).self) {
                try await ModelIntakeCommand.downloadIntakeSource([
                    "--go", "--model", intakeFixtureRepository.appendingPathComponent("models/" + id).path,
                    "--helpers", "/does-not-exist", "--support", "/Users/example/Library/Application Support/Arsenal"
                ])
            }
        }
    }
    @Test func completedDownloadedIntakesBindAllTiersAndFreshServingFigures() throws {
        for id in ["von-1.1", "von-1.2", "re-use"] {
            let folder = intakeFixtureRepository.appendingPathComponent("models/" + id), model = try ModelManifest.load(folder)
            let binding = try latestIntakeBinding(folder)
            let run = try #require(binding["run"]?.string), path = folder.appendingPathComponent("results/intake/" + run + ".json")
            let data = try Data(contentsOf: path), result = try WireJSON.decode(data)
            let rows = try #require(result["rows"]?.array), outcomes = try #require(binding["outcomes"]?.array)
            #expect(rows.count == 3 && outcomes.count == 3)
            #expect(Set(rows.compactMap { $0["precision"]?.string }) == Set(["bf16", "int8", "int4"]))
            #expect(result["resource_receipt"]?["initial_swap_bytes"] == result["resource_receipt"]?["peak_swap_bytes"])
            #expect(result["resource_receipt"]?["stopped"] == false)
            let loadID = try ModelLoadConfiguration.identity(model: model, runner: model.runners[0], folder: folder)
            #expect(binding["load_identity"]?.string == loadID)
            for row in rows {
                let dtype = try #require(row["precision"]?.string), outcome = try #require(outcomes.first { $0["dtype"]?.string == dtype })
                let tablePath = folder.appendingPathComponent(try #require(outcome["result"]?.string))
                if !model.precisions.contains(dtype) {
                    // A withdrawn tier keeps its captured row and bound outcome, but ships no table result.
                    #expect(removedQualityTiers.contains { $0.model == id && $0.tier == dtype } && !FileManager.default.fileExists(atPath: tablePath.path))
                    continue
                }
                let table = try Data(contentsOf: tablePath)
                let stats = try PrecisionStatsResult.decode(table).stats
                #expect(outcome["capture_result_sha256"]?.string == FileDigest.sha256(data))
                #expect(outcome["result_sha256"]?.string == FileDigest.sha256(table))
                #expect(stats.qualityCheckPassed == row["gate"]?["component_gate_passed"]?.bool)
                #expect(
                    stats.qualityIdentity
                        == PrecisionEvidence.qualityIdentity(
                            model: model, dtype: dtype, runner: "native", implementation: binding["implementation_identity"]?.string, loadIdentity: loadID))
                #expect(row["serving_worker"]?["measurement_components_requests"] == 0)
                #expect((row["peak_gb"]?.number ?? 0) > 0)
                #expect(row["load"]?["requested_load"]?["dtype"]?.string == dtype)
                #expect(stats.value(.peakMemoryGB).provenance?.defaultWorkload == true)
                if id == "re-use" {
                    #expect(stats.kind == .enhancement && result["policy"]?["component_keys"]?.array?.count == 63)
                    let scope = try #require(outcome["scope"]?.string)
                    #expect(dtype == "bf16" ? scope.hasPrefix("Qualified 16-bit") : scope.contains("BF16 baseline"))
                    #expect(result["policy"]?["reference_scope"]?.string?.contains("RE-USE") == true)
                }
            }
        }
    }

}
