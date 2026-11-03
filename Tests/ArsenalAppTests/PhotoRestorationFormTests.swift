// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import Testing
@testable import Arsenal

@MainActor struct PhotoRestorationFormTests {
    static func schema(_ op: String) throws -> JSONObject {
        let repo = (0..<3).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() }
        let data = try Data(contentsOf: repo.appendingPathComponent("Resources/kinds/restoration/ops/\(op).input.json"))
        guard case .object(let object) = try JSONDecoder().decode(JSONValue.self, from: data) else { throw CocoaError(.fileReadCorruptFile) }
        return object
    }
    /// Colorize and Inpaint have no capability window of their own: the generic Run… tool builds their forms from the
    /// kind schemas. Colorize offers the photo and the makers' input size; Inpaint offers the photo and its mask, both
    /// required, and nothing else.
    @Test func photoRestorationFormsOfferOnlyTheirInputs() throws {
        let colorize = ToolForm(schema: try Self.schema("colorize"))
        #expect(colorize.fields.map(\.key).sorted() == ["image", "input_size"])
        #expect(colorize.fields.first { $0.key == "image" }?.required == true)
        #expect(colorize.fields.first { $0.key == "input_size" }?.required == false)
        let photo: FieldValue = .files([URL(fileURLWithPath: "/tmp/a.png")])
        #expect(try colorize.input(["image": photo], seed: nil, stage: { $0.path })["image"] == ["path": "/tmp/a.png"])
        let inpaint = ToolForm(schema: try Self.schema("inpaint"))
        #expect(inpaint.fields.map(\.key).sorted() == ["image", "mask"])
        #expect(inpaint.fields.filter(\.required).count == 2)
        for field in inpaint.fields {
            guard case .generic(.file) = field.control else { Issue.record("\(field.key) is not a file picker"); continue }
        }
        #expect(ToolCatalog.windows(declared: ["colorize", "inpaint"]).isEmpty)
    }
}
