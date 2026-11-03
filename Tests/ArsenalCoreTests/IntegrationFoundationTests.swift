// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct IntegrationFoundationTests {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    @Test func everyCapabilityHasAdmissibleSchemas() throws {
        let kinds = try KindSpec.load(directory: root.appendingPathComponent("Resources/kinds"))
        let all = kinds.flatMap(\.operations)
        for capability in CapabilityContract.all {
            #expect(all.contains { $0.name == capability.operation }, "Missing contract for \(capability.operation)")
        }
        #expect(Set(CapabilityContract.all.map(\.route)).count == CapabilityContract.all.count)
        #expect(Set(CapabilityContract.all.map(\.tool)).count == CapabilityContract.all.count)
        for kind in kinds {
            for op in kind.operations {
                let base = root.appendingPathComponent("Resources/kinds/" + kind.id)
                try ModelSchemaValidation.checkSchema(
                    schemaData: OperationSchema.read(path: op.input, folder: base, kindsDirectory: root.appendingPathComponent("Resources/kinds")))
                if let output = op.outputSchema {
                    try ModelSchemaValidation.checkSchema(
                        schemaData: OperationSchema.read(path: output, folder: base, kindsDirectory: root.appendingPathComponent("Resources/kinds")))
                }
            }
        }
    }
    @Test func specializedLanguageModelsShipWithoutPublishingGeneralChat() throws {
        var model = try ModelManifest.load(root.appendingPathComponent("models/qwen3.8-27b"))
        #expect(!CatalogShipping.includes(model))
        model.facets = ModelFacets(modalitiesIn: ["text", "image"], modalitiesOut: ["text"])
        #expect(CatalogShipping.includes(model))
        model.facets = ModelFacets(modalitiesIn: ["text"], modalitiesOut: ["text"])
        model.op = [OpDeclaration(name: "translate", title: "Translate", input: "input.json", output: "json", stream: false)]
        #expect(CatalogShipping.includes(model))
    }
    @Test func modelDocsAreCurrent() throws { try script("scripts/generate-model-docs.py", arguments: ["--check"]) }
    /// Core review 7: download and on-disk sizes per precision, derived tiers not repeating the 16-bit download.
    @Test func modelDocsGeneratorSizesAndTools() throws { try script("scripts/test-generate-model-docs.py", arguments: []) }
    @Test func headerCheckerNeverFetchesPayloads() throws { try script("scripts/test-model-check.py", arguments: []) }
    private func script(_ name: String, arguments: [String]) throws {
        let process = Process(), output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env"); process.arguments = ["python3", root.appendingPathComponent(name).path] + arguments
        process.standardOutput = output; process.standardError = output
        try process.run(); let data = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        #expect(process.terminationStatus == 0, "\(String(decoding: data, as: UTF8.self))")
    }
}
