// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct SmallSFXIntakeTests {
    private var folder: URL {
        intakeFixtureRepository.appendingPathComponent("models/stable-audio-3-small-sfx")
    }
    private func documents() throws -> (JSONObject, JSONObject, URL) {
        let binding = try latestIntakeBinding(folder)
        let run = try #require(binding["run"]?.string)
        let result = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("results/intake/" + run + ".json")))
        return (binding, result, folder.appendingPathComponent("gate/evidence/intake/" + run))
    }
    @Test func threeTiersBindTheirExactRunBoundsAndDefaultWorkloadFigures() throws {
        let (binding, result, evidence) = try documents()
        let model = try ModelManifest.load(folder), runner = model.runners[0]
        #expect(model.derive?["fp16"] == "cast:stable-audio-3-small-sfx")
        let rows = try #require(result["rows"]?.array), outcomes = try #require(binding["outcomes"]?.array)
        #expect(rows.count == 3 && outcomes.count == 3)
        #expect(Set(rows.compactMap { $0["precision"]?.string }) == Set(["fp16", "int8", "int4"]))
        #expect(binding["implementation_identity"] == result["implementation_identity"] && binding["worker_sha256"] == result["worker_sha256"])
        #expect(binding["load_identity"]?.string == (try ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder)))
        for row in rows {
            let dtype = try #require(row["precision"]?.string)
            let outcome = try #require(outcomes.first { $0["dtype"]?.string == dtype })
            let path = try #require(outcome["result"]?.string), data = try Data(contentsOf: folder.appendingPathComponent(path))
            let stats = try PrecisionStatsResult.decode(data).stats
            #expect(outcome["result_sha256"]?.string == FileDigest.sha256(data))
            let capture = try #require(outcome["capture_result"]?.string)
            #expect(outcome["capture_result_sha256"]?.string == (try FileDigest.sha256(folder.appendingPathComponent(capture))))
            let bounds = evidence.appendingPathComponent(dtype == "fp16" ? "bounds.json" : "quantized-bounds.json")
            #expect(outcome["bounds_sha256"]?.string == (try FileDigest.sha256(bounds)))
            let median = try #require(row["warm"]?["median_s"]?.number)
            #expect(stats.value(.realTimeFactor).value == median / 5)
            #expect(stats.value(.peakMemoryGB).value == row["peak_gb"]?.number)
            #expect(row["serving_worker"]?["measurement_components_requests"] == 0)
            #expect(row["serving_worker"]?["pid"]?.number ?? 0 > 0)
            #expect(stats.value(.realTimeFactor).provenance?.defaultWorkload == true)
            #expect(row["warm"]?["samples_s"]?.array?.count == (dtype == "fp16" ? 8 : 4))
            #expect(stats.qualityCheckPassed == (dtype != "int4"))
            #expect(stats.qualityIdentity == outcome["quality_identity"]?.string)
            let components = try #require(row["gate"]?["components"]?.object)
            #expect(Set(components.keys) == Set(["prompt", "seconds", "decoder", "denoiser-0", "denoiser-3", "denoiser-7"]))
            #expect(components.values.allSatisfy { $0["finite"]?.bool == true && $0["clamp_count"]?.number == 0 })
            if dtype == "int8" {
                #expect(components.values.allSatisfy { ($0["relative_l2"]?.number ?? 1) <= 0.05 && ($0["cosine"]?.number ?? 0) >= 0.999 })
            }
        }
    }
    @Test func allLargePartsArePackedAndNoFP32WeightsRemain() throws {
        let (_, _, evidence) = try documents()
        for dtype in ["fp16", "int8", "int4"] {
            let coverage = try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent(dtype + "-component-coverage.json")))
            #expect(coverage.count == 3)
            for (name, expected) in [("dit_sm-sfx_f16", 171), ("t5gemma_f16", 85), ("same_s_decoder_f32", 25)] {
                let component = try #require(coverage["MLX/" + name + ".safetensors.coverage.json"]?.object)
                #expect(component["quantized_tensors"]?.number == Double(dtype == "fp16" ? 0 : expected))
                let types = try #require(component["output_dtypes"]?.object)
                #expect(types["float32"] == nil && types["float64"] == nil)
                #expect(types["float16"]?.number ?? 0 > 0)
                #expect(Set(component["unpacked_keys"]?.array?.compactMap(\.string) ?? []) == Set(component["skip_reasons"]?.object?.keys.map { $0 } ?? []))
            }
        }
    }
    @Test func resourcesAndFP32SourceQualificationAreExplicit() throws {
        let (_, result, evidence) = try documents()
        let resources = try #require(result["resource_receipt"]?.object)
        #expect(resources["initial_swap_bytes"] == resources["peak_swap_bytes"] && resources["stopped"] == false)
        #expect(resources["elapsed_s"]?.number ?? 600 < 590)
        let format = evidence.appendingPathComponent("format-qualification")
        let gate = try WireJSON.decode(Data(contentsOf: format.appendingPathComponent("format-gate.json")))
        let bounds = try WireJSON.decode(Data(contentsOf: format.appendingPathComponent("bounds.json")))
        #expect(gate["component_gate_passed"] == true && bounds["reference_half"] == "bf16")
        #expect(Set(gate["components"]?.object?.keys.map { $0 } ?? []) == Set(["decoder", "seconds", "padding"]))
        #expect(gate["bounds_sha256"]?.string == (try FileDigest.sha256(format.appendingPathComponent("bounds.json"))))
        for row in try #require(gate["components"]?.object).values {
            #expect(row["finite"] == true && row["clamp_count"]?.number == 0 && row["passed"] == true)
        }
        let baseline = try #require(result["rows"]?.array?.first)
        #expect(baseline["gate"]?["format_qualification"] == .object(gate))
        #expect(baseline["gate"]?["format_qualification_sha256"]?.string == (try FileDigest.sha256(format.appendingPathComponent("format-gate.json"))))
        #expect(result["policy"]?["format_gate"]?.string?.hasPrefix("Six") == true)
    }
}
