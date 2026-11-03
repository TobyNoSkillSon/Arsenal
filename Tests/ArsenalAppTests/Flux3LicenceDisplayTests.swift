// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
@testable import ArsenalCore

/// The FLUX 3 Image Klein add-on's licence as the Models table hover and Get more models show it: named, never an
/// id, and marked limited in orange (not commercial).
@MainActor @Suite struct Flux3LicenceDisplayTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @Test func addOnLicenceReadsByName() throws {
        let model = try ModelManifest.load(repo.appendingPathComponent("models/flux3-image-klein"), kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        // Table hover (row carries the SPDX id only) and the browser's short name.
        #expect(LicenceName.short(model.licence.spdx, name: nil) == "FLUX Kommunity License 1.0")
        #expect(LicenceName.short(model.licence.spdx, name: model.licence.name) == "FLUX Kommunity License 1.0")
        #expect(LicenceName.full(model.licence.spdx, name: model.licence.name) == "FLUX Kommunity License v1.0")
        #expect(
            ModelHelp.licenceLine(LicenceName.short(model.licence.spdx, name: nil), commercial: model.licence.commercial)
                == "FLUX Kommunity License 1.0, commercial use limited, see licence")
        let entry = BrowserTests.entry("flux3-image-klein", model.name, kind: "images", params: model.params, bytes: 20_316_385_092, licence: model.licence.spdx, commercial: false)
        let item = BrowserItem(entry, kind: nil)
        #expect(item.licenceText == "FLUX Kommunity License 1.0" && !item.commercial)
    }
}

/// The Image window's size for an edit: "Same as reference" first, so an edit keeps its reference's size unless the
/// window picks one; a new image still starts at 1024 × 1024.
@MainActor @Suite struct EditSizeFollowsReferenceTests {
    private var folder: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("models/flux3-image-klein")
    }
    private func form(_ op: String) throws -> ToolForm {
        ToolForm(schema: try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/\(op).input.json"))), operation: op, output: "image")
    }

    @Test func editStartsAtTheReferenceSizeAndSendsNone() throws {
        let edit = try form("edit")
        let size = try #require(edit.fields.first { $0.key == "size" })
        guard case .size(let sizes) = size.control else { Issue.record("no size field"); return }
        #expect(sizes.reference && sizes.preferred == nil)
        #expect(edit.initialValue(size, setting: nil) == .text(""))
        #expect(edit.initialValue(size, settings: ["width": 768, "height": 768]) == nil)
        let png = FileManager.default.temporaryDirectory.appendingPathComponent("flux3-edit-\(UUID().uuidString).png")
        try Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLttAAAAABJRU5ErkJggg==")!.write(to: png)
        defer { try? FileManager.default.removeItem(at: png) }
        let input = try edit.input(["prompt": .text("the same scene at night"), "images": .files([png]), "size": .text("")], seed: 3, stage: { $0.path })
        #expect(input["width"] == nil && input["height"] == nil && input["images"] != nil)
        let picked = try edit.input(["prompt": .text("x"), "images": .files([png]), "size": .text("768x768")], seed: 3, stage: { $0.path })
        #expect(picked["width"] == 768 && picked["height"] == 768)
    }

    @Test func generateStillStartsAt1024() throws {
        let generate = try form("generate")
        let size = try #require(generate.fields.first { $0.key == "size" })
        guard case .size(let sizes) = size.control else { Issue.record("no size field"); return }
        #expect(!sizes.reference && generate.initialValue(size, setting: nil) == .text("1024x1024"))
    }
}
