// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct Int4PolicyTests {
    @Test func warningTracksCurrentRunnerQualityAndRoundTrips() throws {
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "images", precisions: ["bf16", "int8", "int4"])
        #expect(row.precisionQualityWarnings == ["int4": PrecisionStatSet.int4UncheckedWarning])
        var stats = PrecisionStatSet(kind: .images, dtype: "int4", runner: "native")
        stats.qualityCheckPassed = false; row.precisionStats["int4"] = stats
        #expect(row.precisionQualityWarnings == ["int4": PrecisionStatSet.int4QualityWarning])
        #expect(try JSONDecoder().decode(ModelRow.self, from: JSONEncoder().encode(row)).precisionQualityWarnings == row.precisionQualityWarnings)
        let json = try WireJSON.decode(JSONEncoder().encode(row))
        #expect(json["precisionQualityWarnings"]?["int4"]?.string == PrecisionStatSet.int4QualityWarning)
        stats.qualityCheckPassed = true; row.precisionStats["int4"] = stats
        #expect(row.precisionQualityWarnings.isEmpty)
        row.runner = "another"
        #expect(row.precisionQualityWarnings["int4"] == PrecisionStatSet.int4UncheckedWarning)
        row.precisions = ["bf16"]
        #expect(row.precisionQualityWarnings.isEmpty)
    }
    @Test func int8FailureAndUnsupportedInt4RemainBlocked() {
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "sound", precisions: ["fp16", "int8", "int4"])
        for dtype in ["int8", "int4"] {
            row.precisionStats[dtype] = PrecisionStatSet(kind: .sound, dtype: dtype, runner: "native", unavailableReason: "Did not pass the quality check")
        }
        #expect(row.dtypeUnavailableReason("int8") == "Did not pass the quality check")
        #expect(row.dtypeUnavailableReason("int4") == nil)
        row.precisionStats["int4"] = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native", unavailableReason: "This runner does not support this precision.")
        #expect(row.dtypeUnavailableReason("int4") == "This runner does not support this precision.")
    }
    @Test(arguments: [false, true]) func runtimeFailureSurvivesReaderAndRow(qualityAlsoFailed: Bool) {
        var stats = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native", unavailableReason: "This runner cannot execute these weights")
        stats.qualityCheckPassed = qualityAlsoFailed ? false : nil
        let sets = PrecisionStatsReader.sets(
            kind: "sound", baseDtype: "fp16", runner: "native", supported: ["fp16", "int4"], results: [.init(model: "fixture", mode: "exact", stats: stats)])
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "sound", precisions: ["fp16", "int4"])
        row.precisionStats = sets
        #expect(row.dtypeUnavailableReason("int4") == "This runner cannot execute these weights")
        #expect(row.precisionStats["int4"]?.qualityCheckPassed == stats.qualityCheckPassed)
    }
    @Test func legacyQualityOnlyFailureRemainsSelectableThroughReaderAndRow() {
        let stats = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native", unavailableReason: "Did not pass the quality check")
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "sound", precisions: ["fp16", "int4"])
        row.precisionStats = PrecisionStatsReader.sets(
            kind: "sound", baseDtype: "fp16", runner: "native", supported: ["fp16", "int4"], results: [.init(model: "fixture", mode: "exact", stats: stats)])
        #expect(row.dtypeUnavailableReason("int4") == nil)
        #expect(row.precisionStats["int4"]?.qualityCheckPassed == false)
    }
    @Test(arguments: [false, true]) func separateRuntimeAndQualityRecordsKeepRuntimeFailure(scoped: Bool) {
        var quality = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native", unavailableReason: "Did not pass the quality check")
        quality.qualityOnlyUnavailable = true
        quality.qualityIdentity = "current"
        quality.set(.peakMemoryGB, .init(state: .quickCheck, value: 2.7))
        let runtime = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native", unavailableReason: "This runner cannot execute these weights")
        let records = [quality, runtime].map { PrecisionStatsResult(model: "fixture", mode: "exact", stats: $0) }
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "sound", precisions: ["fp16", "int4"])
        row.precisionStats = PrecisionStatsReader.sets(
            kind: "sound", baseDtype: "fp16", runner: "native", supported: row.precisions,
            results: scoped ? records.map { PrecisionEvidence.scopeQuality($0, identity: "current") } : records)
        #expect(row.dtypeUnavailableReason("int4") == "This runner cannot execute these weights")
        #expect(row.precisionStats["int4"]?.availability.state == .unavailable)
        #expect(row.precisionStats["int4"]?.qualityOnlyUnavailable != true)
        #expect(row.precisionStats["int4"]?.value(.peakMemoryGB).value == 2.7)
    }
    @Test func intakeFlagsAndStatsSurviveSchemaRoundTrip() throws {
        #expect(ModelIntakeCommand.qualityWarnings(dtype: "int4", passed: false)["int4"]?.string == PrecisionStatSet.int4QualityWarning)
        #expect(ModelIntakeCommand.qualityWarnings(dtype: "int4", passed: nil)["int4"]?.string == PrecisionStatSet.int4UncheckedWarning)
        #expect(ModelIntakeCommand.qualityWarnings(dtype: "int4", passed: true).isEmpty)
        #expect(ModelIntakeCommand.qualityWarnings(dtype: "int8", passed: false).isEmpty)
        var stats = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native")
        stats.qualityCheckPassed = false
        let result = PrecisionStatsResult(model: "fixture", mode: "exact", stats: stats)
        #expect(try PrecisionStatsResult.decode(result.markdown()).stats.qualityCheckPassed == false)
    }
    @Test func newerCurrentSchemaQualityPassClearsTheWarning() throws {
        func result(_ date: String, _ passed: Bool) -> PrecisionStatsResult {
            var stats = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native")
            stats.qualityCheckPassed = passed
            stats.availability = StatValue(state: .quickCheck, provenance: .init(chip: "m5-max", date: date, method: "saved component gate", revision: "fixture-revision"))
            return .init(model: "fixture", mode: "exact", stats: stats)
        }
        let old = result("2026-10-15T10:00:00Z", false), new = result("2026-10-15T11:00:00Z", true)
        let sets = PrecisionStatsReader.sets(kind: "sound", baseDtype: "fp16", runner: "native", supported: ["fp16", "int4"], results: [old, new])
        #expect(sets["int4"]?.qualityCheckPassed == true)
        #expect(sets["int4"]?.selectionUnavailableReason == nil)
    }
    @Test func qualityIdentityChangesWithRecipeOrOwnRunnerButNotOtherWorkers() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var model = try ModelManifest.load(shippedOrRetiredFolder(repo, "stable-audio-3-medium"))
        let identity = try #require(PrecisionEvidence.qualityIdentity(model: model, dtype: "int4", runner: "native", implementation: "sound-source-hash"))
        for _ in 0..<8 {
            #expect(PrecisionEvidence.qualityIdentity(model: model, dtype: "int4", runner: "native", implementation: "sound-source-hash") == identity)
        }
        var stats = PrecisionStatSet(kind: .sound, dtype: "int4", runner: "native")
        stats.qualityCheckPassed = true; stats.qualityIdentity = identity
        let result = PrecisionStatsResult(model: model.id, mode: "fast", stats: stats)
        #expect(PrecisionEvidence.scopeQuality(result, identity: identity).stats.qualityCheckPassed == true)
        model.derive?["int4"] = "intake-v2:stable-audio-3-medium"
        let changed = PrecisionEvidence.qualityIdentity(model: model, dtype: "int4", runner: "native", implementation: "sound-source-hash")
        #expect(changed != identity)
        #expect(PrecisionEvidence.scopeQuality(result, identity: changed).stats.qualityCheckPassed == nil)
        #expect(PrecisionEvidence.scopeQuality(result, identity: nil).stats.qualityCheckPassed == nil)
        let support = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        let receipt = support.appendingPathComponent("quality-identities.json")
        try JSONEncoder().encode(["sound": "sound-source-hash", "voices": "old"]).write(to: receipt)
        let artifact = BundledWorkerArtifact(helpers: support, qualityIdentities: receipt)
        let runner = try #require(model.runners.first)
        let original = artifact.qualityImplementation(model: model, runner: runner, directory: support)
        try JSONEncoder().encode(["sound": "sound-source-hash", "voices": "new"]).write(to: receipt)
        #expect(artifact.qualityImplementation(model: model, runner: runner, directory: support) == original)
        try JSONEncoder().encode(["sound": "new-sound-hash"]).write(to: receipt)
        #expect(artifact.qualityImplementation(model: model, runner: runner, directory: support) != original)
    }
    /// The rule (AGENTS.md, Quality gate): int4 is declared only where its quality check passed or was never run. A model
    /// whose int4 check failed omits it, and a declared int4 never sits on the removed list.
    @Test func int4IsDeclaredOnlyWhereItsCheckPassedOrNeverRan() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folders = try FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("models"), includingPropertiesForKeys: nil)
        for folder in folders where FileManager.default.fileExists(atPath: folder.appendingPathComponent("model.toml").path) {
            let model = try ModelManifest.load(folder)
            guard CatalogShipping.includes(model) else { continue }
            // A model with a recorded int4 failure must not declare int4; any other model may (passed or never run).
            if removedQualityTiers.contains(where: { $0.model == model.id && $0.tier == "int4" }) {
                #expect(!model.precisions.contains("int4"), "\(model.id) declares int4 after its check failed")
            }
        }
    }

    @Test func everyShippingModelDeclaresAnInt4RecipeOrPinnedVendor() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folders = try FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("models"), includingPropertiesForKeys: nil)
        var count = 0
        for folder in folders where FileManager.default.fileExists(atPath: folder.appendingPathComponent("model.toml").path) {
            let model = try ModelManifest.load(folder)
            guard CatalogShipping.includes(model) else { continue }
            count += 1
            // A tier removed after a failed quality check is not declared (removedQualityTiers).
            if !declaredTiers(model).contains("int4") || offersSixteenBitOnly(model) { continue }
            #expect(model.precisions.contains("int4"), "Missing int4: \(model.id)")
            #expect(model.derive?["int4"] != nil || model.vendor?["int4"] != nil)
        }
        #expect(count >= 16) // Every shipping folder, the original 16 included.
    }
}
