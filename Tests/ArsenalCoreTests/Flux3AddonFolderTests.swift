// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// FLUX 3 Image Klein's folder (models/flux3-image-klein, a gated catalog model since 2 Nov). Copied into local-models it
/// still validates as a drop-in served by the bundled flux worker, and a local copy cannot shadow the signed id.
struct Flux3AddonFolderTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var folder: URL { repo.appendingPathComponent("models/flux3-image-klein") }

    @Test func folderValidatesAsASchema3LocalDropIn() throws {
        let model = try ModelManifest.load(folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        #expect(model.schema == 3 && model.id == "flux3-image-klein" && model.kind == "images" && model.architecture == "flux3-image-klein")
        #expect(model.runners.count == 1 && model.runners[0].worker == "flux" && model.runners[0].entry == nil && model.runners[0].language == "swift")
        #expect(model.weights.repo == "black-forest-labs/flux-3-image-klein" && model.weights.revision == "337337621b8526e120f2275d250242f4a6bedd75")
        #expect(model.weights.dtype == "bf16" && model.precisions == ["bf16"] && model.derive == nil && model.precision == nil)
        // Only the BF16 diffusers files: no FP8/NVFP4 variants and no BFL-format duplicates.
        #expect(model.weights.files.count == 8 && !model.weights.files.contains { $0.hasPrefix("variants/") || !$0.contains("/") })
        #expect(model.weights.bytes?.values.reduce(0, +) == 20_316_385_092)
        #expect(model.licence.spdx == "LicenseRef-FLUX-Kommunity-License-1.0" && !model.licence.commercial)
        #expect(model.licence.name == "FLUX Kommunity License v1.0" && model.licence.url.contains(model.weights.revision))
        #expect(model.licence.conditions?.contains("US$5M") == true)
        #expect(model.operations.map(\.name).sorted() == ["edit", "generate"])
        #expect(model.settings.first { $0.id == "steps" }?.defaultValue == .number(16))
        // In the signed catalog since 2 Nov (revenue-capped licence accepted on zero income), no longer an add-on.
        #expect(!FileManager.default.fileExists(atPath: repo.appendingPathComponent("addons/flux3-image-klein").path))
    }

    /// No size setting has a default: a default would be sent with every edit and replace the reference's size.
    @Test func editRequestsKeepTheReferenceSize() throws {
        let model = try ModelManifest.load(folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        #expect(model.settings.filter { ["width", "height"].contains($0.id) }.allSatisfy { $0.defaultValue == nil })
        var settings = ModelSettings()
        settings.declare(model.id, metadata: try WireJSON.decode(JSONEncoder().encode(model)))
        let edit = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/edit.input.json")))
        let request = settings.request(model.id, params: ["prompt": "x", "images": [["path": "/tmp/a.png"]]], schema: edit)
        #expect(request["width"] == nil && request["height"] == nil && request["steps"] == 16)
        // A saved size (from Make an Image's generate) never overrides "Same as reference" in an edit; generate keeps it.
        try settings.set(model.id, key: "width", value: 1024)
        try settings.set(model.id, key: "height", value: 768)
        let saved = settings.request(model.id, params: ["prompt": "x", "images": [["path": "/tmp/a.png"]]], schema: edit)
        #expect(saved["width"] == nil && saved["height"] == nil)
        let explicit = settings.request(model.id, params: ["prompt": "x", "images": [["path": "/tmp/a.png"]], "width": 512], schema: edit)
        #expect(explicit["width"] == 512 && explicit["height"] == nil)
        let generate = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/generate.input.json")))
        let made = settings.request(model.id, params: ["prompt": "x"], schema: generate)
        #expect(made["width"] == 1024 && made["height"] == 768)
    }

    @Test func operationSchemasDeclareGenerateAndEdit() throws {
        let generate = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/generate.input.json")))
        let edit = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/edit.input.json")))
        #expect(generate["required"] == ["prompt"] && edit["required"] == ["prompt", "images"])
        for schema in [generate, edit] {
            #expect(schema["additionalProperties"] == false)
            #expect(schema["properties"]?["width"]?["multipleOf"] == 16 && schema["properties"]?["height"]?["multipleOf"] == 16)
            #expect(schema["properties"]?["guidance"] == nil && schema["properties"]?["negative_prompt"] == nil)
        }
        #expect(generate["properties"]?["images"] == nil)
        #expect(edit["properties"]?["images"]?["minItems"] == 1 && edit["properties"]?["images"]?["maxItems"] == 10)
        #expect(edit["properties"]?["images"]?["items"]?["x-arsenal-file"]?["accept"] == ["image/png", "image/jpeg", "image/heic", "image/webp"])
    }

    @Test func dropInNeedsNoCodeConsentAndCannotShadowACatalogID() throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-flux3-addon-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let local = support.appendingPathComponent("local-models/flux3-image-klein")
        try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: folder, to: local)
        let kinds = repo.appendingPathComponent("Resources/kinds")
        let found = try LocalModelFolders.discover(support: support, signedIDs: ["flux2-klein-4b"], kindsDirectory: kinds)
        #expect(found.map(\.manifest.id) == ["flux3-image-klein"])
        #expect(found.first?.requiresCodeConsent == false)
        #expect(throws: ManifestError.self) { try LocalModelFolders.discover(support: support, signedIDs: ["flux3-image-klein"], kindsDirectory: kinds) }
    }

    @Test func getMoreModelsCarriesTheLicenceLine() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["images"], addToTable: false); defer { f.stop() }
        let local = f.paths.support.appendingPathComponent("local-models/flux3-image-klein")
        try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: folder, to: local)
        let entry = try #require(await f.service.catalog().first { $0.id == "flux3-image-klein" })
        #expect(entry.local && !entry.row.needsCodeConsent && !entry.commercial)
        #expect(entry.licence == "LicenseRef-FLUX-Kommunity-License-1.0" && entry.licenceName == "FLUX Kommunity License v1.0")
        #expect(
            entry.licenceConditions?.hasPrefix("FLUX Kommunity License: you accept it on Hugging Face") == true
                && entry.licenceConditions?.contains("Non-commercial use") == true && entry.licenceURL.hasSuffix("/LICENSE.md"))
        #expect(entry.limits.first?.hasPrefix("FLUX Kommunity License") == true)
        await f.service.shutdown()
    }
}
