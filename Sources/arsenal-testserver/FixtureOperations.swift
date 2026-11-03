// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension FixtureModelsService {
    func operations(id: String) async -> [ModelOperation] {
        guard let row = await stub.catalog(query: CatalogQuery()).first(where: { $0.id == id })?.row else { return [] }
        func op(_ name: String, _ output: String, required: [String] = [], properties: JSONObject = [:], stream: Bool = false) -> ModelOperation {
            ModelOperation(
                name: name, title: name, inputSchema: ["type": "object", "required": .array(required.map(JSONValue.string)), "properties": .object(properties)], output: output,
                stream: stream)
        }
        if ["audio", "embeddings", "perception", "restoration", "geometry", "language"].contains(row.kind) {
            return (try? FixtureCapabilitySchemas.operations(kind: row.kind)) ?? []
        }
        switch row.kind {
        case "chat":
            return [op("chat", "text-stream", required: ["messages"], properties: ["messages": ["type": "array", "minItems": 1]], stream: true), op("count_tokens", "json")]
        case "images": return [op("generate", "image", required: ["prompt"], properties: ["prompt": ["type": "string", "minLength": 1]])]
        case "voices":
            return [
                op("speak", "audio", required: ["text"], properties: ["text": ["type": "string", "minLength": 1]], stream: true), op("voices", "json"),
                op("design_voice", "json", required: ["description"], properties: ["description": ["type": "string"]]),
                op(
                    "clone_voice", "json", required: ["audio", "transcript", "name"],
                    properties: ["audio": ["type": "object"], "transcript": ["type": "string", "minLength": 1], "name": ["type": "string"]])
            ]
        case "sound":
            return [
                op(
                    "generate_sound", "audio", required: ["prompt"],
                    properties: ["prompt": ["type": "string", "minLength": 1], "duration": ["type": "number", "exclusiveMinimum": 0, "maximum": 380]]),
                op("generate_music", "audio", required: ["prompt"], properties: ["prompt": ["type": "string", "minLength": 1]])
            ]
        case "enhancement": return [op("enhance", "audio", required: ["audio"], properties: ["audio": ["type": "object", "required": ["path"]]])]
        case "decisions": return [op("decide", "json", required: ["state", "questions"]), op("judge", "json", required: ["items", "questions"])]
        default: return []
        }
    }
}
