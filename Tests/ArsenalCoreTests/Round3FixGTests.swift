// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct Round3FixGTests {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    @Test(arguments: [false, true]) func nearBodyLimitChatValidates(contentParts: Bool) throws {
        let count = contentParts ? 3_000 : 7_000
        let text = String(repeating: "w", count: contentParts ? 292 : 118)
        let messages: [JSONValue] = (0..<count).map { i in
            let content: JSONValue = contentParts ? .array([["type": "text", "text": .string(text)]]) : .string(text)
            return ["role": .string(i % 2 == 0 ? "user" : "assistant"), "content": content]
        }
        let input: JSONObject = ["messages": .array(messages)]
        let bytes = try WireJSON.encode(input).count
        #expect(bytes > 1_040_000 && bytes < 1_048_576)
        for path in ["models/qwen3.8-27b/ops/chat.input.json", "Resources/kinds/chat/ops/chat.input.json"] {
            let schema = try WireJSON.decode(Data(contentsOf: repo.appendingPathComponent(path)))
            try OperationSchemaValidator.validate(.object(input), schema: .object(schema))
        }
    }

    @Test(arguments: [(10, 4), (100, 4), (100, 5), (100, 6)]) func reviewerAmplificationProbesStillFail(width: Int, levels: Int) throws {
        var defs: JSONObject = ["leaf": ["type": "object"]]
        var previous = "leaf"
        for level in 0..<levels {
            let name = "n\(level)"
            defs[name] = ["allOf": .array(Array(repeating: ["$ref": .string("#/$defs/" + previous)], count: width))]
            previous = name
        }
        let schema: JSONObject = ["$defs": .object(defs), "$ref": .string("#/$defs/" + previous)]
        if levels == 6 {
            #expect(throws: ManifestError.self) { try ModelSchemaValidation.checkSchema(schemaData: WireJSON.encode(schema)) }
        } else {
            try ModelSchemaValidation.checkSchema(schemaData: WireJSON.encode(schema))
        }
        #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate([:], schema: .object(schema)) }
    }
}
