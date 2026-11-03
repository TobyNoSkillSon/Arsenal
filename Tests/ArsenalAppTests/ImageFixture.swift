// SPDX-License-Identifier: MIT
import Foundation
@testable import Arsenal
import ArsenalCore

/// FLUX.2 Klein's generate operation as its model folder declares it today, and the same with reference images (no
/// folder declares them yet: a fixture schema), for Make an Image's tests and renders.
enum ImageFixture {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static var schema: JSONObject {
        (try? WireJSON.decode(Data(contentsOf: repo.appendingPathComponent("models/flux2-klein-4b/ops/generate.input.json")))) ?? [:]
    }
    static var generate: OperationDescriptor { OperationDescriptor(name: "generate", title: "Generate", inputSchema: schema, output: "image", stream: false) }
    /// As the folder declares it (schema 2: several images come back as multipart/mixed).
    static var operation: ModelOperation { ModelOperation(name: "generate", title: "Generate", inputSchema: schema, output: "image", stream: false, schemaVersion: 2) }
    /// FLUX's schema with reference images (base64 strings, at most `most`).
    static func withReferences(_ most: Int = 4) -> OperationDescriptor {
        var schema = schema
        var properties = schema["properties"]?.object ?? [:]
        properties["reference_images"] = ["type": "array", "maxItems": .number(Double(most)), "items": ["type": "string"]]
        schema["properties"] = .object(properties)
        return OperationDescriptor(name: "generate", title: "Generate", inputSchema: schema, output: "image", stream: false)
    }
}
