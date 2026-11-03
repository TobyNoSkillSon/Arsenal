// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct IntakeEvidenceTests {
    private var repo: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    @Test func pilotCaseCountsAndCandidateResultsAreComplete() throws {
        for (model, run, tiers) in [
            ("flux2-klein-4b", "intake-2026-10-15", ["bf16", "fp16", "int8", "int4"]),
            ("stable-audio-3-medium", "intake-original-audio-2026-10-15", ["fp16", "int8", "int4"])
        ] {
            let folder = intakeFixtureRepository.appendingPathComponent("models/" + model)
            let data = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("results/" + run + ".json")))
            let rows = try #require(data["rows"]?.array)
            #expect(rows.count == tiers.count)
            #expect(rows.compactMap { $0["precision"]?.string } == tiers)
            for row in rows {
                #expect(row["error"] == nil)
                #expect(row["qualified"]?.bool == false)
                #expect((row["peak_gb"]?.number ?? 0) > 0)
            }
            for tier in tiers {
                let candidate = try PrecisionStatsResult.decode(Data(contentsOf: folder.appendingPathComponent("gate/evidence/intake-candidate-stats/" + tier + ".json")))
                #expect(candidate.model == model && candidate.stats.dtype == tier)
                #expect(candidate.stats.availability.state == .unavailable)
                #expect(candidate.stats.value(.peakMemoryGB).state == .quickCheck)
                let receipt = try JSONDecoder().decode(
                    SnapshotManifest.self, from: Data(contentsOf: folder.appendingPathComponent("gate/evidence/intake-2026-10-15-" + tier + "-snapshot.json")))
                #expect(receipt.model == model && receipt.precision == tier && !receipt.files.isEmpty)
                if tier != "bf16" {
                    #expect(receipt.derived?.sourceRepository != nil)
                    #expect(receipt.derived?.sourceFiles?.isEmpty == false)
                }
            }
        }
    }
    @Test func correctedPilotsLinkFrozenBoundsAndBalancedWarmWindows() throws {
        for model in ["flux2-klein-4b", "stable-audio-3-medium"] {
            let root = intakeFixtureRepository.appendingPathComponent("models/" + model)
            let historicalRun =
                model == "flux2-klein-4b"
                ? "intake-2026-10-15T11-14-00Z-acb8d191-2607-48f4-8a93-5c251b3ac926"
                : "intake-2026-10-15T11-22-27Z-534d1300-91ab-4f17-bdc4-6a9a49e08b33"
            let historicalBounds = root.appendingPathComponent("gate/evidence/intake/" + historicalRun + "/bounds.json")
            let pointer: JSONObject = [
                "run": .string(historicalRun), "results_path": .string("results/intake/" + historicalRun + ".json"),
                "bounds_path": .string("gate/evidence/intake/" + historicalRun + "/bounds.json"), "bounds_sha256": .string(try FileDigest.sha256(historicalBounds))
            ]
            let run = try #require(pointer["run"]?.string)
            let file = root.appendingPathComponent(try #require(pointer["results_path"]?.string))
            let result = try WireJSON.decode(Data(contentsOf: file))
            let bounds = root.appendingPathComponent(try #require(pointer["bounds_path"]?.string))
            #expect(try FileDigest.sha256(bounds) == pointer["bounds_sha256"]?.string)
            let rows = try #require(result["rows"]?.array)
            #expect(rows.count == (model == "flux2-klein-4b" ? 4 : 3))
            for (i, row) in rows.enumerated() {
                #expect(row["gate"]?["components"]?.object?.count == 5)
                #expect(row["gate"]?["bounds_sha256"] == pointer["bounds_sha256"])
                #expect(row["qualified"]?.bool == false)
                #expect((row["warm"]?["samples_s"]?.array?.count ?? 0) >= 4)
                if i > 0 { #expect(row["warm"]?["samples_s"]?.array?.count == 4 && row["paired_baseline_warm"]?["samples_s"]?.array?.count == 4) }
                #expect((row["load"]?["load_s"]?.number ?? 0) > 0 && (row["cold_seconds"]?.number ?? 0) > 0)
                let dtype = try #require(row["precision"]?.string)
                let stats = try PrecisionStatsResult.decode(Data(contentsOf: root.appendingPathComponent("gate/evidence/intake-candidate-stats/" + dtype + ".json")))
                #expect(stats.stats.value(.peakMemoryGB).provenance?.run == run + "-" + dtype)
                #expect(stats.stats.value(.peakMemoryGB).provenance?.source?.contains(run) == true)
            }
        }
    }
    @Test func qualifiedPilotTableReadsCurrentRecipesAndPerPrecisionFigures() throws {
        for id in ["flux2-klein-4b", "stable-audio-3-medium"] {
            let root = intakeFixtureRepository.appendingPathComponent("models/" + id), model = try ModelManifest.load(root)
            let pointer = try WireJSON.decode(Data(contentsOf: root.appendingPathComponent("gate/evidence/intake-latest.json")))
            let run = try #require(pointer["run"]?.string)
            let result = try WireJSON.decode(Data(contentsOf: root.appendingPathComponent(try #require(pointer["results_path"]?.string))))
            let rows = try #require(result["rows"]?.array)
            let table = try model.precisions.map { dtype in
                try PrecisionStatsResult.decode(
                    Data(contentsOf: root.appendingPathComponent("results/intake-m5-max-" + dtype + "-native-" + (id == "flux2-klein-4b" ? "exact" : "fast") + ".json")))
            }
            let sets = PrecisionStatsReader.sets(kind: model.kind, baseDtype: model.basePrecision, runner: "native", supported: model.precisions, results: table)
            for row in rows {
                let dtype = try #require(row["precision"]?.string)
                if dtype == "fp16" && id == "flux2-klein-4b" { #expect(row["qualified"]?.bool == false); continue }
                let stat = try #require(sets[dtype])
                #expect(stat.availability.state == .quickCheck)
                #expect(stat.availability.reason == nil)
                #expect(stat.qualityCheckPassed == (dtype != "int4"))
                #expect(stat.value(.peakMemoryGB).value == row["peak_gb"]?.number)
                let seconds = try #require(row["warm"]?["median_s"]?.number)
                #expect(stat.value(id == "flux2-klein-4b" ? .secondsPerImage : .realTimeFactor).value == seconds / (id == "flux2-klein-4b" ? 1 : 5))
                #expect(stat.availability.provenance?.run == run + "-" + dtype)
                #expect(row["qualified"]?.bool == (dtype != "int4"))
                let key = ["int8", "int4"].contains(dtype) ? "quantized_bounds_path" : "bounds_path"
                let bounds = root.appendingPathComponent(try #require(pointer[key]?.string))
                #expect(row["gate"]?["bounds_sha256"]?.string == (try FileDigest.sha256(bounds)))
            }
            #expect(model.derive?["int8"] == (id == "flux2-klein-4b" ? "mlx-affine-g64:" : "intake-v1:") + id)
            #expect(model.basePrecision == (id == "flux2-klein-4b" ? "bf16" : "fp16"))
            if id == "stable-audio-3-medium" { #expect(model.derive?["fp16"] == "cast:" + id) }
        }
    }
    @Test func publishedFluxTimingKeepsDefaultWorkloadAndOriginalPrecision() throws {
        let result = try PrecisionStatsResult.decode(
            Data(contentsOf: intakeFixtureRepository.appendingPathComponent("models/flux2-klein-4b/results/intake-m5-max-bf16-native-exact.json")))
        #expect(result.stats.dtype == "bf16" && result.mode == "exact")
        #expect(result.stats.availability.state == .quickCheck)
        #expect(result.stats.value(.secondsPerImage).provenance?.method.contains("1024x1024") == true)
        #expect(result.stats.value(.peakMemoryGB).provenance?.baseline?.contains("BF16") == true)
    }
}

/// Tiers removed because their quality check failed, or because they were not worth keeping: a tier whose check fails, or
/// that is no faster and no smaller than 16-bit, is removed rather than offered (AGENTS.md, Quality gate).
/// Int4PolicyTests and everyRepositoryModelDeclaresItsHalfInt8AndInt4 both read this one list.
let removedQualityTiers: [(model: String, tier: String, reason: String)] = [
    ("qwen3-asr-1.7b", "int8", "Vella's int8 loses a clip the 16-bit tier transcribes."),
    ("qwen3-asr-1.7b", "int4", "Vella's int4 loses a clip the 16-bit tier transcribes."),
    ("index-translate-2b", "int4", "The int4 check failed: role text and reasoning leaked into the translation, and a verb was mistranslated."),
    ("qwen-image-2.1", "int4", "The int4 check failed on fixed inputs: prompt encoder rel 0.30, cos 0.958; transformer rel 0.10 to 0.13, cos 0.992 (lab/qwen-image/STATE.md)."),
    (
        "qwen-image-2.1-turbo", "int4",
        "The int4 check failed on fixed inputs: prompt encoder rel 0.30, cos 0.958; transformer rel 0.10 to 0.13, cos 0.992 (lab/qwen-image/STATE.md)."
    ),
    ("birefnet", "int8", "Not worth keeping: no faster and at most 9% smaller than FP16 (0.47 vs 0.48 s per image; peak 5.3 vs 5.7 GB) (lab/night-2026-10-22/tiers.md)."),
    ("birefnet", "int4", "Not worth keeping: no faster and at most 9% smaller than FP16 (0.47 vs 0.48 s per image; peak 5.2 vs 5.7 GB) (lab/night-2026-10-22/tiers.md)."),
    ("chandra-ocr-2", "int8", "Not worth keeping: no faster than FP16 (8.1 vs 7.8 s per page) and only 16% smaller (lab/night-2026-10-22/tiers.md)."),
    ("chandra-ocr-2", "int4", "The check failed: read_document text changed (similarity 0.25 to 16-bit) (lab/night-2026-10-22/tiers.md)."),
    ("nuextract3", "int8", "The check failed: read_document text changed (similarity 0.26 to 16-bit) (lab/night-2026-10-22/tiers.md)."),
    ("nuextract3", "int4", "The check failed: read_document text changed (similarity 0.20 to 16-bit) (lab/night-2026-10-22/tiers.md)."),
    ("re-use", "int8", "Not worth keeping: slower than BF16 (1.14 vs 0.87 s) for a 0.2 GB smaller peak (1.22 vs 1.45 GB) (lab/night-2026-10-22/tiers.md)."),
    ("re-use", "int4", "Not worth keeping: slower than BF16 (1.13 vs 0.87 s) for a 0.2 GB smaller peak (1.27 vs 1.45 GB) (lab/night-2026-10-22/tiers.md)."),
    ("kokoro-82m", "int8", "Not worth keeping: no faster and no smaller than FP16 (peak 1.43 vs 1.40 GB) (lab/night-2026-10-22/tiers.md)."),
    ("kokoro-82m", "int4", sixteenBitOnly),
    ("laya-english", "int8", sixteenBitOnly),
    ("laya-english", "int4", sixteenBitOnly),
    ("laya-multilingual", "int8", sixteenBitOnly),
    ("laya-multilingual", "int4", sixteenBitOnly),
    ("laya-typed-decisions", "int8", sixteenBitOnly),
    ("laya-typed-decisions", "int4", sixteenBitOnly),
    ("pocket-tts", "int8", sixteenBitOnly),
    ("pocket-tts", "int4", sixteenBitOnly),
    ("von-1.1", "int8", sixteenBitOnly),
    ("von-1.1", "int4", sixteenBitOnly),
    ("von-1.2", "int8", sixteenBitOnly),
    ("von-1.2", "int4", sixteenBitOnly)
]
private let sixteenBitOnly = "16-bit only: its 16-bit weights are under 1 GB."

/// The tiers a model declares: its 16-bit base, int8 and int4, less the tiers `removedQualityTiers` lists for it.
func declaredTiers(_ model: ModelManifest) -> Set<String> {
    let removed = Set(removedQualityTiers.filter { $0.model == model.id }.map(\.tier))
    return Set([model.basePrecision, "int8", "int4"]).subtracting(removed)
}

/// Any model may offer its 16-bit base alone: quantized tiers are optional, never required. It then offers
/// neither int8 nor int4.
func offersSixteenBitOnly(_ model: ModelManifest) -> Bool {
    model.precisions == [model.basePrecision] && model.derive?["int8"] == nil && model.derive?["int4"] == nil
}

/// A model may offer, beside its 16-bit base, only checkpoints its makers publish, pinned in `[vendor.<tier>]` and never
/// derived (docs/MODEL-FOLDER.md): Drama 3 ships the mlx-community 4-bit release as stored. The int8/int4 pair rule covers
/// derived tiers only, so such a model derives neither.
func offersBaseAndVendorTiersOnly(_ model: ModelManifest) -> Bool {
    guard let vendor = model.vendor, !vendor.isEmpty else { return false }
    return Set(model.precisions) == Set([model.basePrecision]).union(vendor.keys) && model.derive?["int8"] == nil && model.derive?["int4"] == nil
}

@Test func everyRepositoryModelDeclaresItsHalfInt8AndInt4() throws {
    let folders = try FileManager.default.contentsOfDirectory(at: intakeFixtureRepository.appendingPathComponent("models"), includingPropertiesForKeys: nil)
        .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path) }
    // Every directory under models/ is a model folder; the 48 kept models remain.
    let directories = try FileManager.default.contentsOfDirectory(at: intakeFixtureRepository.appendingPathComponent("models"), includingPropertiesForKeys: [.isDirectoryKey])
        .filter { !$0.lastPathComponent.hasPrefix(".") && (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
    #expect(folders.count == directories.count && folders.count >= 48)
    for folder in folders {
        let model = try ModelManifest.load(folder)
        #expect(offersSixteenBitOnly(model) || offersBaseAndVendorTiersOnly(model) || Set(model.precisions) == declaredTiers(model), "\(model.id)")
        for removed in removedQualityTiers where removed.model == model.id {
            #expect(!model.precisions.contains(removed.tier), "\(model.id) declares its removed \(removed.tier)")
        }
        #expect(["bf16", "fp16"].contains(model.basePrecision))
    }
}

@Test func uncheckedInt8IsOfferedWithoutMarkLikeInt4() {
    let stats = PrecisionStatsReader.sets(kind: "decisions", baseDtype: "bf16", runner: "native", supported: ["bf16", "int8", "int4"], results: [])
    #expect(stats.count == 3)
    #expect(stats["int8"]?.selectionUnavailableReason == nil)
    #expect(stats["int8"]?.qualityCheckPassed == nil)
    #expect(stats["int4"]?.selectionUnavailableReason == nil)
    #expect(stats["int4"]?.qualityCheckPassed == nil)
    var row = ModelRow(id: "unchecked", name: "Unchecked", kind: "decisions", precisions: ["bf16", "int8", "int4"])
    row.precisionStats = stats
    #expect(row.dtypeUnavailableReason("int8") == nil)
    #expect(row.precisionQualityWarnings == ["int4": PrecisionStatSet.int4UncheckedWarning])
}

@Test func onlyAFailedInt8CheckGreysInt8() {
    func result(_ passed: Bool?, legacyReason: String? = nil, date: String = "2026-10-15T00:00:00Z") -> PrecisionStatsResult {
        var set = PrecisionStatSet(kind: .decisions, dtype: "int8", runner: "native")
        set.qualityCheckPassed = passed
        let provenance = StatProvenance(chip: "m5-max", date: date, method: "fixture", revision: String(repeating: "a", count: 40), run: "r-" + date)
        set.availability = legacyReason.map { StatValue(state: .unavailable, reason: $0, provenance: provenance) } ?? StatValue(state: .quickCheck, provenance: provenance)
        set.set(.peakMemoryGB, StatValue(state: .quickCheck, value: 1.5, provenance: provenance))
        return PrecisionStatsResult(model: "fixture", mode: "fast", stats: set)
    }
    func int8(_ results: [PrecisionStatsResult]) -> PrecisionStatSet {
        PrecisionStatsReader.sets(kind: "decisions", baseDtype: "fp16", runner: "native", supported: ["fp16", "int8", "int4"], results: results)["int8"]!
    }
    // Passed and unchecked are offered; their figures are kept.
    #expect(int8([result(true)]).selectionUnavailableReason == nil)
    #expect(int8([result(nil)]).selectionUnavailableReason == nil)
    #expect(int8([result(nil)]).value(.peakMemoryGB).value == 1.5)
    // A schema-2 failure and a legacy failure are greyed with their reason; figures stay.
    let failed = int8([result(false)])
    #expect(failed.selectionUnavailableReason == "Did not pass the quality check")
    #expect(failed.qualityCheckPassed == false)
    #expect(failed.value(.peakMemoryGB).value == 1.5)
    #expect(int8([result(false, legacyReason: "Quality gate failed")]).selectionUnavailableReason == "Did not pass the quality check")
    // A newer pass on the same revision clears an older failure.
    #expect(int8([result(false), result(true, date: "2026-10-16T00:00:00Z")]).selectionUnavailableReason == nil)
    // A pass whose recipe/implementation identity no longer matches is unchecked, not failed: offered without a mark.
    var stale = result(true); stale.stats.qualityIdentity = "old-identity"
    let scoped = PrecisionEvidence.scopeQuality(stale, identity: "current-identity")
    #expect(scoped.stats.qualityCheckPassed == nil)
    #expect(int8([scoped]).selectionUnavailableReason == nil)
    #expect(int8([scoped]).qualityCheckPassed == nil)
    // A failure under another recipe/implementation is stale too: unchecked, so int8 stays offered (schema-2 and legacy).
    for failed in [result(false), result(false, legacyReason: "Quality gate failed")] {
        var old = failed; old.stats.qualityIdentity = "old-identity"
        let scopedFailure = PrecisionEvidence.scopeQuality(old, identity: "current-identity")
        #expect(int8([scopedFailure]).selectionUnavailableReason == nil)
        #expect(int8([scopedFailure]).qualityCheckPassed == nil)
        #expect(int8([scopedFailure]).value(.peakMemoryGB).value == 1.5)
        // The same failure under the current identity still greys int8.
        #expect(
            int8([
                PrecisionEvidence.scopeQuality(
                    {
                        var c = failed; c.stats.qualityIdentity = "current-identity"; return c
                    }(), identity: "current-identity")
            ])
            .selectionUnavailableReason == "Did not pass the quality check")
    }
    // A stale int4 failure shows the not-yet-checked warning, not the failed one.
    var int4Failure = PrecisionStatSet(kind: .decisions, dtype: "int4", runner: "native", unavailableReason: "Did not pass the quality check")
    int4Failure.qualityOnlyUnavailable = true; int4Failure.qualityIdentity = "old-identity"
    var row = ModelRow(id: "fixture", name: "Fixture", kind: "decisions", precisions: ["fp16", "int8", "int4"])
    row.precisionStats = PrecisionStatsReader.sets(
        kind: "decisions", baseDtype: "fp16", runner: "native", supported: row.precisions,
        results: [PrecisionEvidence.scopeQuality(PrecisionStatsResult(model: "fixture", mode: "fast", stats: int4Failure), identity: "current-identity")])
    #expect(row.dtypeUnavailableReason("int4") == nil)
    #expect(row.precisionQualityWarnings == ["int4": PrecisionStatSet.int4UncheckedWarning])
    // A runtime failure (not a quality outcome) still greys any tier with its own reason.
    var broken = result(nil); broken.stats.availability = .unavailable("Worker cannot load int8 on this chip")
    #expect(int8([broken]).selectionUnavailableReason == "Worker cannot load int8 on this chip")
}

/// Current table files point at the latest immutable completion, never a fixed date.
func latestIntakeBinding(_ folder: URL) throws -> JSONObject {
    let files = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("gate/evidence"), includingPropertiesForKeys: nil)
        .filter { $0.lastPathComponent.hasPrefix("quality-binding-") && $0.pathExtension == "json" }.sorted { $0.path < $1.path }
    guard let file = files.last else { throw ManifestError.invalid("Missing intake binding") }
    return try WireJSON.decode(Data(contentsOf: file))
}
