// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct IntegrationContractR2Tests {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    @Test func folderSelectorAndZeroMatrixPolicyAreValidatedAndPresented() throws {
        let folder = root.appendingPathComponent("models/birefnet")
        var model = try ModelManifest.load(folder)
        #expect(model.architecture == nil && model.quantization == nil && model.precisionNotes.isEmpty)
        // BiRefNet ships FP16 only since 22 Oct; the policy under test needs derived tiers, so the fixture declares them.
        model.derive = ["int8": "intake-v1:birefnet", "int4": "intake-v1:birefnet"]
        model.architecture = "rife"
        model.quantization = QuantizationDeclaration(eligibleMatrices: 0, reason: "Spatial convolutions only.")
        #expect(model.problems(folder: folder).isEmpty)
        #expect(Set(model.precisions) == ["fp16", "int8", "int4"])
        #expect(model.precisionNotes.count == 2)
        #expect(model.precisionNotes["int4"]?.contains("Runs as FP16") == true)
        let settings = try ModelLoadConfiguration.defaults(model)
        let config = try ModelLoadConfiguration.config(model: model, folder: folder, assets: [:], chip: "m5-max", fast: false, settings: settings)
        #expect(config["architecture"] == "rife")
        let encoded = try JSONEncoder().encode(model), restored = try JSONDecoder().decode(ModelManifest.self, from: encoded)
        #expect(restored.architecture == "rife" && restored.quantization == model.quantization)
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "restoration", precisions: model.precisions)
        row.precisionDtypes = model.precisionDtypes
        row.precisionNotes = model.precisionNotes
        let decoded = try JSONDecoder().decode(ModelRow.self, from: JSONEncoder().encode(row))
        #expect(decoded.precisionNotes == model.precisionNotes)
        #expect(decoded.precisionDtypes == ["int8": "fp16", "int4": "fp16"])
        #expect(decoded.precisionQualityWarnings.isEmpty)
        model.architecture = "../unsafe"
        #expect(model.problems(folder: folder).contains { $0.contains("architecture selector") })
        model.architecture = nil; model.quantization?.eligibleMatrices = 1
        #expect(model.problems(folder: folder).contains { $0.contains("zero eligible") })
        model.quantization?.eligibleMatrices = 0; model.quantization?.reason = ""
        #expect(model.problems(folder: folder).contains { $0.contains("with a reason") })
    }
    @Test func nativeHalfDefaultsAndDocumentedBF16Exception() throws {
        let folder = root.appendingPathComponent("models/birefnet")
        var model = try ModelManifest.load(folder)
        model.derive = ["int8": "intake-v1:birefnet", "int4": "intake-v1:birefnet"] // Withdrawn from the folder on 22 Oct; declared here.
        model.weights.dtype = "bf16"; model.runner[0].worker = "embed"
        #expect(model.basePrecision == "fp16" && model.source(for: "fp16") == nil)
        #expect(model.precisions == ["fp16", "int8", "int4"])
        #expect(model.conversionRecipe(for: "fp16") == NativeHalfPolicy.recipe)
        model.precision = NativeHalfPolicy(dtype: "bf16")
        #expect(model.problems(folder: folder).contains { $0.contains("BF16 exception needs") })
        model.precision = NativeHalfPolicy(dtype: "bf16", reason: "FP16 activation overflow", evidence: "README.md")
        #expect(model.problems(folder: folder).isEmpty)
        #expect(model.precisions == ["bf16", "int8", "int4"])
    }
    @Test func imagePreparationDeclarationSurvivesManifestAndOperationEncoding() throws {
        let declaration = try TOMLParser.decode(
            OpDeclaration.self,
            from: #"""
                name = "image_to_3d"
                title = "Make 3D"
                input = "input.json"
                output = "file"
                stream = false
                [prepare_image]
                model = "birefnet"
                operation = "remove_background"
                file_id = "foreground"
                """#)
        #expect(declaration.prepareImage == OperationImagePreparation(model: "birefnet", operation: "remove_background", fileID: "foreground"))
        var descriptor = ModelOperation(name: declaration.name, title: declaration.title, inputSchema: [:], output: "file", stream: false)
        descriptor.prepareImage = declaration.prepareImage
        #expect(try JSONDecoder().decode(ModelOperation.self, from: JSONEncoder().encode(descriptor)).prepareImage == declaration.prepareImage)
        let folder = root.appendingPathComponent("models/birefnet")
        var model = try ModelManifest.load(folder)
        model.op?[0].prepareImage = declaration.prepareImage
        #expect(model.problems(folder: folder).contains { $0.contains("another model") })
    }
    @Test func architectureChangesLoadIdentityAndGuideMirrorMatches() throws {
        let folder = root.appendingPathComponent("models/birefnet")
        var model = try ModelManifest.load(folder)
        let before = try ModelLoadConfiguration.identity(model: model, runner: model.runners[0], folder: folder)
        model.architecture = "rife"
        let after = try ModelLoadConfiguration.identity(model: model, runner: model.runners[0], folder: folder)
        #expect(before != after && !after.isEmpty)
        let doc = try String(contentsOf: root.appendingPathComponent("docs/INTEGRATING.md"), encoding: .utf8)
        #expect(IntegrationReferences.documents["INTEGRATING.md"]?.trimmingCharacters(in: .whitespacesAndNewlines) == doc.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
