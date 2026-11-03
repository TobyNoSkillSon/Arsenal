// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ImageIO
@testable import ArsenalCore

struct BiRefNetTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var folder: URL { intakeFixtureRepository.appendingPathComponent("models/birefnet") }

    @Test func pinnedHalfFolderExposesImageRemovalAtFP16Only() throws {
        let model = try ModelManifest.load(folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        #expect(model.schema == 3 && model.kind == "images")
        #expect(model.weights.repo == "ZhengPeng7/BiRefNet" && model.weights.revision == "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4")
        #expect(model.weights.dtype == "fp16" && model.basePrecision == "fp16")
        #expect(model.weights.bytes?["model.safetensors"] == 444_473_596)
        #expect(model.weights.sha256?["model.safetensors"] == "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154")
        // int8 and int4 are withdrawn (22 Oct quick check: no faster and no smaller than FP16), so Get derives nothing.
        #expect(model.precisions == ["fp16"])
        #expect(model.derive == nil)
        #expect(model.operations.map(\.name) == ["remove_background"])
        #expect(model.runners.count == 1 && model.runners[0].language == "swift")
        #expect(model.runners[0].python == nil && model.runners[0].envDownloadBytes == nil)
        #expect(model.licence.spdx == "MIT" && model.licence.commercial)
        #expect(CatalogShipping.includes(model))
        // There are no model-specific settings; an empty settings.toml is invalid and must be absent.
        #expect(!FileManager.default.fileExists(atPath: folder.appendingPathComponent("settings.toml").path))
    }

    @Test func removalSchemaRejectsPromptsAndPreservesAnnotatedImageContract() throws {
        let input = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/remove_background.input.json")))
        #expect(input["required"] == ["image"] && input["additionalProperties"] == false)
        #expect(input["properties"]?["image"]?["x-arsenal-file"]?["accept"] == ["image/png", "image/jpeg"])
        let output = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/remove_background.output.json")))
        #expect(output["properties"]?["files"]?["minItems"] == 2 && output["properties"]?["files"]?["maxItems"] == 2)
        #expect(output["properties"]?["files"]?["items"]?["properties"]?["id"]?["enum"] == ["foreground", "mask"])
        #expect(input["properties"]?["prompt"] == nil)
    }

    @Test func nativeIntakeRecordsCorrectnessWithoutInventingPerformance() throws {
        let evidence = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/evidence/native/components.json")))
        let cases = try #require(evidence["cases"]?.array)
        #expect(evidence["status"] == "passed" && cases.count == 20)
        #expect(Set(cases.compactMap { $0["key"]?.string }).count == 20)
        for item in cases {
            #expect(item["finite"] == true)
            #expect(try #require(item["relative_l2"]?.number) < 0.01)
            #expect(try #require(item["cosine"]?.number) > 0.9999)
        }
        let stats = try PrecisionStatsResult.decode(Data(contentsOf: folder.appendingPathComponent("results/static-fp16-native-exact.json")))
        #expect(stats.stats.dtype == "fp16" && stats.stats.runner == "native")
        #expect(stats.stats.metrics.values.allSatisfy { $0.state == .notMeasured && $0.value == nil })
        let api = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/evidence/native/api.json")))
        #expect(api["http"]?["metadata"]?["engine"] == "mlx-swift")
        #expect(api["http"]?["files"]?.array?.count == 2 && api["invalid_input_status"] == 422)
    }
    @Test func changedFP16PathHasAnOriginalImplementationBridgeSeparateFromAA() throws {
        let parity = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/evidence/native/fp16-preservation.json")))
        #expect(parity["status"] == "passed" && parity["component_count"] == 20)
        #expect(parity["fixture"]?["base_commit"] == "b77f8aa59ab106b0b29c8282ddd9fc1954fe01c6")
        #expect(parity["fixture"]?["input_count"] == 16)
        let cases = try #require(parity["cases"]?.array)
        #expect(cases.count == 20 && Set(cases.compactMap { $0["key"]?.string }).count == 20)
        for item in cases {
            #expect(item["finite"] == true && item["byte_identical"] == true)
            #expect(item["relative_l2"] == 0 && item["cosine"] == 1 && item["max_abs"] == 0)
            #expect(item["reference_tensor_sha256"] == item["candidate_tensor_sha256"])
            #expect((item["count"]?.number ?? 0) > 0 && item["shape"]?.array?.isEmpty == false)
        }
        #expect(parity["implementations"]?["b77f8aa5"]?["source_closure_sha256"] != parity["implementations"]?["current"]?["source_closure_sha256"])
        #expect(parity["saved_intake_bridge"]?["component_count"] == 20 && parity["saved_intake_bridge"]?["all_byte_identical"] == true)
        #expect(parity["aa_reproducibility"]?["basis"]?.string?.contains("separate") == true)
        #expect(parity["resource_receipt"]?["swap_growth_bytes"] == 0)
    }

    @Test func tierIntakeFreezesCompleteKeysAndMaskDiagnostic() throws {
        #expect(ModelIntakeCommand.biRefNetComponentKeys.count == 20)
        #expect(ModelIntakeCommand.biRefNetMaskMAEBound == 0.02)
        #expect(try ModelIntakeCommand.biRefNetMaskMAE(reference: [0, 255], candidate: [0, 255]) == 0)
        #expect(try ModelIntakeCommand.biRefNetMaskMAE(reference: [0, 255], candidate: [255, 0]) == 1)
        #expect(throws: ManifestError.self) { try ModelIntakeCommand.biRefNetMaskMAE(reference: [0], candidate: []) }
        #expect(ModelIntakeCommand.offered(dtype: "int8", passed: false) == false)
        #expect(ModelIntakeCommand.offered(dtype: "int4", passed: false))
        #expect(ModelIntakeCommand.qualityWarnings(dtype: "int4", passed: false)["int4"] != nil)
    }

    @Test func maskFixturesIncludeActualPNGTransparency() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let images = try ModelIntakeCommand.biRefNetImages(in: directory)
        #expect(images.count == 3)
        let source = try #require(CGImageSourceCreateWithURL(images[2] as CFURL, nil))
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any])
        #expect(properties[kCGImagePropertyHasAlpha as String] as? Bool == true)
    }

}
