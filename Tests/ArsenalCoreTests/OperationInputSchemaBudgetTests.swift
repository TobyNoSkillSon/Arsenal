// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The file-input schema walk has a budget that bounds the schema, not the request: a valid request with many file
/// parts (two 768-frame videos) is admitted, while a schema whose expansion explodes is still stopped.
struct OperationInputSchemaBudgetTests {
    /// A 1x1 PNG.
    static let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")!

    static func support() throws -> (URL, String) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("schema-budget-" + UUID().uuidString)
        let inbox = root.appendingPathComponent("inbox")
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        let frame = inbox.appendingPathComponent("frame.png")
        try png.write(to: frame)
        return (root, frame.path)
    }

    /// The shape of a decisions model's media list: images, or videos whose frames are a path or an annotated file object.
    static let schema: JSONObject = {
        let file: JSONValue = [
            "type": "object", "required": ["path"], "properties": ["path": ["type": "string"]],
            "x-arsenal-file": ["accept": ["image/png", "image/jpeg"]]
        ]
        let image: JSONValue = [
            "type": "object", "required": ["type", "path"], "properties": ["type": ["const": "image"], "path": ["type": "string"]],
            "x-arsenal-file": ["accept": ["image/png", "image/jpeg"]]
        ]
        let video: JSONValue = [
            "type": "object", "required": ["type", "frames"],
            // Each frame is tested against two branches, so a long video tests the same fragments many times.
            "properties": [
                "type": ["const": "video"], "frames": ["type": "array", "maxItems": 768, "items": ["anyOf": [["type": "string"], file]]], "fps": ["type": "number"]
            ]
        ]
        return ["type": "object", "properties": ["media": ["type": "array", "items": ["anyOf": [image, video]]]]]
    }()

    @Test func twoLongVideosAreAdmitted() throws {
        let (support, frame) = try Self.support()
        defer { try? FileManager.default.removeItem(at: support) }
        let frames = JSONValue.array(Array(repeating: ["path": .string(frame)], count: 768))
        let video: JSONValue = ["type": "video", "frames": frames, "fps": 2]
        let request: JSONValue = ["media": [video, video, ["type": "image", "path": .string(frame)]]]
        try OperationArtifacts.input(request, schema: Self.schema, support: support)
    }

    @Test func framesOutsideTheInboxAreStillRefused() throws {
        let (support, _) = try Self.support()
        defer { try? FileManager.default.removeItem(at: support) }
        let request: JSONValue = ["media": [["type": "video", "frames": [["path": "/etc/hosts"], ["path": "/etc/hosts"]]]]]
        #expect(throws: WorkerFailure.self) { try OperationArtifacts.input(request, schema: Self.schema, support: support) }
    }

    @Test func anExplodingSchemaIsStopped() throws {
        let (support, _) = try Self.support()
        defer { try? FileManager.default.removeItem(at: support) }
        // Every level's anyOf holds two references to the next: expanding level 0 means 2^16 copies.
        var defs: JSONObject = ["d16": ["type": "number"]]
        for level in 0..<16 {
            let next = JSONValue.string("#/$defs/d\(level + 1)")
            defs["d\(level)"] = ["anyOf": [["$ref": next], ["$ref": next]]]
        }
        let schema: JSONObject = ["type": "object", "properties": ["x": ["$ref": "#/$defs/d0"]], "$defs": .object(defs)]
        do {
            try OperationArtifacts.input(["x": 1], schema: schema, support: support)
            Issue.record("an exploding schema was walked to the end")
        } catch let failure as WorkerFailure {
            #expect(failure.code == "op_schema")
        }
    }
}
