// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct VoiceIntakeTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func latestBinding(_ folder: URL) throws -> URL {
        let files = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("gate/evidence"), includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("quality-binding-") }.sorted { $0.path < $1.path }
        return try #require(files.last)
    }
    @Test func everyVoiceHalfAndQuantUsesWholeComponentGetWithoutSourceRecursion() throws {
        for id in ModelIntakeCommand.voiceModels {
            let source = try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/" + id))
            for half in ["bf16", "fp16"] {
                let model = ModelIntakeCommand.voiceModel(source, half: half)
                #expect(model.basePrecision == half)
                #expect(model.precisions == [half, "int8", "int4"])
                #expect(model.problems(folder: intakeFixtureRepository.appendingPathComponent("models/" + id)).isEmpty)
                #expect(model.source(for: half) == nil && model.source(for: "int4") == nil)
                #expect(model.derive?[half] == "cast:" + id)
                #expect(model.vendor == nil)
            }
        }
    }
    @Test func selectionCoversEveryPredictorPrefixAndKokoroStyleClass() throws {
        let cases = ModelIntakeCommand.voiceSelectionCases
        // The worker uses 1 + offset % 3 for predictor cache prefix and offset % 3 for style.
        // Distinct text alone once missed the middle cache/style class.
        let classes = try Set(cases.map { Int(try #require($0["offset"]?.number)) % 3 })
        #expect(classes == [0, 1, 2])
        #expect(Set(cases.compactMap { $0["text"]?.string }).count == cases.count)
        let kokoro = ModelIntakeCommand.voiceSelectionCases(for: "kokoro-82m")
        let offsets = try kokoro.map { Int(try #require($0["offset"]?.number)) }
        #expect(Set(offsets.map { $0 % 3 }) == [0, 1, 2])
        #expect(Set(offsets.map { 5 + $0 % 13 }).count == 3)
        #expect(Set(offsets.map { 2 + $0 % 7 }).count == 3)
    }
    @Test func selectionRechecksAllCasesOfChangedOwnersAndFallsBackWhenUnmapped() {
        let all: Set<String> = ["talker-0-0", "talker-1-1", "predictor-2-14-2", "codec-decoder-0"]
        let owners: JSONObject = ["a": "talker", "b": "predictor", "c": "codec-decoder"]
        let current: Set<String> = ["a", "b", "c"]
        #expect(ModelIntakeCommand.voiceSelectionTargets(keeping: ["b", "c"], knownPassing: current, owners: owners, all: all) == ["talker-0-0", "talker-1-1"])
        #expect(ModelIntakeCommand.voiceSelectionTargets(keeping: ["a", "b"], knownPassing: current, owners: owners, all: all) == ["codec-decoder-0"])
        #expect(ModelIntakeCommand.voiceSelectionTargets(keeping: ["a"], knownPassing: current, owners: owners, all: all) == ["predictor-2-14-2", "codec-decoder-0"])
        #expect(ModelIntakeCommand.voiceSelectionTargets(keeping: [], knownPassing: nil, owners: owners, all: all) == all)
        #expect(ModelIntakeCommand.voiceSelectionTargets(keeping: ["unknown"], knownPassing: current, owners: owners, all: all) == all)
    }
    @Test func fixedSelectionNeverRetainsPassingPartsOrAcceptsUnknownMatrices() throws {
        let owners: JSONObject = ["a": "text-embedding", "b": "codec-decoder", "c": "talker"]
        #expect(try ModelIntakeCommand.voiceFixedSelection(seed: ["a", "b", "c"], owners: owners, failures: ["codec-decoder"]) == ["b"])
        #expect(try ModelIntakeCommand.voiceFixedSelection(seed: ["a", "b"], owners: owners, failures: []) == [])
        #expect(throws: (any Error).self) { try ModelIntakeCommand.voiceFixedSelection(seed: ["unknown"], owners: owners, failures: ["codec-decoder"]) }
    }
    @Test func rejectedHeldoutCandidateNeverPublishesACompleteResult() {
        var rows: [JSONValue] = [["precision": "bf16", "qualified": true], ["precision": "int8", "qualified": false], ["precision": "int4", "qualified": false]]
        #expect(!ModelIntakeCommand.voiceCanPublishResult(rows: rows, failure: nil))
        rows[1] = ["precision": "int8", "qualified": true]
        #expect(ModelIntakeCommand.voiceCanPublishResult(rows: rows, failure: nil))
        #expect(!ModelIntakeCommand.voiceCanPublishResult(rows: rows, failure: "incomplete"))
        #expect(!ModelIntakeCommand.voiceCanPublishResult(rows: Array(rows.prefix(2)), failure: nil))
    }
    @Test func recipeSeedPreservesColonsInsideEveryCanonicalMatrixSelector() {
        let matrices: Set<String> = ["matrix@model.safetensors:talker.a.weight", "matrix@tokenizer/model.safetensors:codec.b.weight"]
        let recipe = "intake-v3:qwen3-tts-base:" + matrices.sorted().joined(separator: ",")
        #expect(ModelIntakeCommand.voiceRecipeMatrices(recipe, model: "qwen3-tts-base") == matrices)
        #expect(ModelIntakeCommand.voiceRecipeMatrices("intake-v3:qwen3-tts-base:all", model: "qwen3-tts-base").isEmpty)
    }
    @Test func int8FallbackCollectsWholeOwnersAndKeepsPassingComponentsPacked() throws {
        let gate: JSONObject = [
            "components": [
                "talker-0": ["passed": false, "relative_l2": 0.25], "talker-2": ["passed": false],
                "predictor-1-13": ["passed": false], "speaker-encoder": ["passed": true],
                "text-embedding": ["passed": true]
            ]
        ]
        #expect(try ModelIntakeCommand.voiceFallbacks(gate) == ["talker", "predictor"])
        #expect(ModelIntakeCommand.voiceComponent("mimi-decoder-1") == "mimi-decoder")
        #expect(ModelIntakeCommand.voiceComponent("predictor-embedding-2-14") == "predictor-embedding")
        #expect(ModelIntakeCommand.gateEvidence(gate: gate, keeping: ["talker"]).count == 2)
        #expect(throws: (any Error).self) { try ModelIntakeCommand.voiceFallbacks([:]) }
        #expect(throws: (any Error).self) { try ModelIntakeCommand.voiceFallbacks(["components": ["talker-0": [:]]]) }
    }
    @Test func voiceRequestRetriesOnlyRejectedBusyCalls() async throws {
        var rejected = 0, executed = 0
        let reply = try await ModelIntakeCommand.voiceWhenReady {
            rejected += 1
            if rejected <= 2 { throw WorkerFailure(code: "busy", message: "admission release pending") }
            executed += 1; return ["ok": true]
        }
        #expect(reply["ok"] == true && rejected == 3 && executed == 1)
        var failures = 0
        await #expect(throws: (any Error).self) {
            try await ModelIntakeCommand.voiceWhenReady {
                failures += 1; throw WorkerFailure(code: "engine_error", message: "A real failure is not retried")
            }
        }
        #expect(failures == 1)
        var busy = 0
        await #expect(throws: (any Error).self) {
            try await ModelIntakeCommand.voiceWhenReady {
                busy += 1; throw WorkerFailure(code: "busy", message: "Persistent busy is bounded")
            }
        }
        #expect(busy == 25)
    }
    @Test func heldoutFormatRejectionNeverSelectsAnotherHalfAndReusesTheSameSet() throws {
        #expect(try ModelIntakeCommand.selectVoiceFP16(selectionPassed: false, heldoutPassed: false) == false)
        #expect(try ModelIntakeCommand.selectVoiceFP16(selectionPassed: false, heldoutPassed: true) == false)
        #expect(try ModelIntakeCommand.selectVoiceFP16(selectionPassed: true, heldoutPassed: true) == true)
        #expect(throws: (any Error).self) { try ModelIntakeCommand.selectVoiceFP16(selectionPassed: true, heldoutPassed: false) }
    }
    @Test func voiceWorkloadUsesTheSharedServedDefaults() throws {
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temp) }
        let value = try MeasurementWorkload.request(kind: .voices, model: "pocket-tts", batched: false, count: 1, directory: temp)
        #expect(value.op == "speak" && value.params["text"]?.string == MeasurementWorkload.speech)
        #expect(value.params["seed"]?.number == 0 && value.params["stream"]?.bool == true)
    }
    @Test func exactArchitectureCoverageIncludesConditioningCodecAndAllPredictorHeads() throws {
        #expect(ModelIntakeCommand.voiceKeys("kokoro-82m").count == 11)
        #expect(ModelIntakeCommand.voiceKeys("pocket-tts").count == 22)
        #expect(ModelIntakeCommand.voiceKeys("breeze-tts-2").count == 23)
        #expect(ModelIntakeCommand.voiceKeys("qwen3-tts-base").count == 101)
        #expect(ModelIntakeCommand.voiceKeys("qwen3-tts-voicedesign").count == 100)
        #expect(ModelIntakeCommand.voiceKeys("pocket-tts").contains("speaker-projection"))
        let policy = ModelIntakeCommand.voicePolicy(
            run: "fixture", workload: [:], original: try ModelManifest.load(intakeFixtureRepository.appendingPathComponent("models/pocket-tts")))
        #expect(policy["format_gate"]?.string == IntakeComponentGate.fp32SourceBasis)
        #expect(policy["pocket_decoder_boundary"]?.string?.hasPrefix("mimi-transformer-upsampling") == true)
        #expect(policy["pocket_waveform_input_policy_id"]?.string == "saved-unit-rms-floor-0p01-seanet-v1")
        #expect(ModelIntakeCommand.voiceKeys("qwen3-tts-base").contains("predictor-2-14"))
    }
    @Test func completedMixedVoiceInt8BindsItsSelectionAndRetainsMaximumInt4Coverage() throws {
        for id in ModelIntakeCommand.voiceModels {
            let folder = intakeFixtureRepository.appendingPathComponent("models/" + id)
            let manifest = try ModelManifest.load(folder)
            let files = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("gate/evidence"), includingPropertiesForKeys: nil).filter {
                $0.lastPathComponent.hasPrefix("quality-binding-")
            }.sorted { $0.path < $1.path }
            let binding = try WireJSON.decode(Data(contentsOf: #require(files.last)))
            // Kokoro's int8 was withdrawn on 22 Oct (removedQualityTiers): its bound recipe stays the evidence's record.
            let withdrawn = removedQualityTiers.contains { $0.model == id && $0.tier == "int8" }
            #expect(withdrawn == (manifest.derive?["int8"] == nil), "\(id)")
            let recorded = binding["outcomes"]?.array?.first { $0["dtype"]?.string == "int8" }?["recipe"]?.string
            let recipe = try #require(manifest.derive?["int8"] ?? (withdrawn ? recorded : nil))
            #expect(recipe.hasPrefix("intake-v3:"))
            let evidence = folder.appendingPathComponent("gate/evidence/intake/" + (try #require(binding["run"]?.string)))
            let all = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("int8-all-coverage.json")))
            let mixed = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("int8-coverage.json")))
            let int4 = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("int4-coverage.json")))
            let originalGate = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("int8-all-gate.json")))
            let keeping = try ModelIntakeCommand.voiceFallbacks(originalGate)
            #expect(mixed["rule"]?["id"]?.string == recipe)
            #expect((mixed["fallback_components"]?.array ?? []).isEmpty)
            #expect((mixed["fallback_matrices"]?.array ?? []).allSatisfy { $0.string?.hasPrefix("matrix@") == true })
            #expect(all["quantized_parameters"]?.number == int4["quantized_parameters"]?.number)
            #expect(mixed["quantized_parameters"]!.number! + mixed["fallback_parameters"]!.number! == all["quantized_parameters"]?.number)
            #expect(Set(mixed["fallback_evidence"]?.object?.keys.map { $0 } ?? []) == Set(ModelIntakeCommand.gateEvidence(gate: originalGate, keeping: keeping).keys))
            let outcome = try #require(binding["outcomes"]?.array?.first { $0["dtype"]?.string == "int8" })
            #expect(outcome["recipe"]?.string == recipe)
            #expect(outcome["qualification"]?["policy"]?.string == IntakeQualification.policyID)
            #expect(outcome["passed"] == outcome["qualification"]?["passed"])
            #expect(outcome["qualification"]?["fixtures"]?["selection"]?["inputs_sha256"] != outcome["qualification"]?["fixtures"]?["heldout"]?["inputs_sha256"])
        }
    }
    @Test func completedVoicesUseTheSharedFrozenSourceFloorAndCaptureFreeServingFigures() throws {
        for id in ModelIntakeCommand.voiceModels {
            let folder = intakeFixtureRepository.appendingPathComponent("models/" + id)
            let binding = try WireJSON.decode(Data(contentsOf: latestBinding(folder)))
            let run = try #require(binding["run"]?.string)
            let evidence = folder.appendingPathComponent("gate/evidence/intake/" + run)
            let bounds = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("bounds.json")))
            #expect(bounds["policy_kind"]?.string == IntakeComponentGate.fp32SourcePolicyID)
            let manifest = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("fixture-manifest.json")))
            let captures = Dictionary(
                uniqueKeysWithValues: try #require(manifest["files"]?.array).compactMap { item -> (String, String)? in
                    guard let path = item["path"]?.string, let hash = item["sha256"]?.string else { return nil }; return (path, hash)
                })
            #expect(bounds["calibration_outputs_sha256"]?.string == captures["bf16-truncation/outputs.safetensors"])
            // Archives may order safetensors headers differently while their decoded arrays are identical.
            #expect(bounds["repeat_outputs_sha256"]?.string == captures["reference-repeat/outputs.safetensors"])
            #expect(bounds["reference_outputs_sha256"]?.string == captures["reference/outputs.safetensors"])
            #expect(bounds["components"]?.object?.values.contains { ($0["floor_relative_l2"]?.number ?? 0) > 0 } == true)
            if id == "pocket-tts" {
                // This source has FP32 presets: replay must read the candidate's stored weights, not the frozen reference preset.
                #expect((bounds["components"]?["preset-conditioning-0"]?["floor_relative_l2"]?.number ?? 0) > 0)
            }
            for bound in try #require(bounds["components"]?.object).values {
                let floor = try #require(bound["floor_relative_l2"]?.number), ulp = try #require(bound["one_ulp_relative_l2"]?.number)
                #expect(bound["relative_l2_max"]?.number == 2 * floor + ulp)
                #expect(bound["repeat_relative_l2"]?.number == 0)
            }
            let format = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent("format-trial.json")))
            #expect(format["gate"]?["policy_kind"]?.string == IntakeComponentGate.fp32SourcePolicyID)
            let result = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("results/intake/" + run + ".json")))
            for row in try #require(result["rows"]?.array) {
                #expect(row["selection_gate"] == row["qualification"]?["selection_gate"])
                #expect(row["gate"] == row["qualification"]?["heldout_gate"])
                #expect(row["serving_worker"]?["measurement_components_requests"]?.number == 0)
                #expect((row["serving_worker"]?["pid"]?.number ?? 0) > 0)
            }

        }
    }
    @Test func completedVoiceBindingsIdentifyDefaultLoadAndRetainFailedQuantFigures() throws {
        // Completed runs are added by the pipeline; historical blocked records remain independent evidence.
        for id in ModelIntakeCommand.voiceModels {
            let folder = intakeFixtureRepository.appendingPathComponent("models/" + id)
            let bindings = [try latestBinding(folder)]
            for file in bindings {
                let binding = try WireJSON.decode(Data(contentsOf: file))
                guard binding["schema"]?.number == 3, binding["family"]?.string == "voices" else { continue }
                let outcomes = try #require(binding["outcomes"]?.array), model = try ModelManifest.load(folder)
                #expect(outcomes.count == 3)
                for outcome in outcomes {
                    let path = try #require(outcome["result"]?.string), dtype = try #require(outcome["dtype"]?.string)
                    if !model.precisions.contains(dtype) {
                        // A tier withdrawn since (Kokoro's int8, 22 Oct) keeps its bound outcome but no longer ships a table result.
                        #expect(removedQualityTiers.contains { $0.model == id && $0.tier == dtype }, "\(id) \(dtype)")
                        #expect(!FileManager.default.fileExists(atPath: folder.appendingPathComponent(path).path))
                        continue
                    }
                    let bytes = try Data(contentsOf: folder.appendingPathComponent(path))
                    let stats = try PrecisionStatsResult.decode(bytes).stats
                    #expect(outcome["result_sha256"]?.string == FileDigest.sha256(bytes))
                    #expect(stats.value(.peakMemoryGB).provenance?.defaultWorkload == true)
                    #expect((stats.value(.realTimeFactor).value ?? 0) > 0)
                    #expect(stats.qualityCheckPassed == outcome["passed"]?.bool)
                    if stats.dtype == "int4", stats.qualityCheckPassed == false {
                        #expect(ModelIntakeCommand.qualityWarnings(dtype: stats.dtype, passed: stats.qualityCheckPassed)["int4"] != nil)
                    }
                }
            }
        }
    }
}
