// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

enum FixtureCapabilitySchemas {
    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static func operations(kind: String) throws -> [ModelOperation] {
        let kind = kind == "language" ? "chat" : kind
        let folder = root.appendingPathComponent("Resources/kinds/" + kind)
        let spec = try TOMLParser.decode(KindSpec.self, from: String(contentsOf: root.appendingPathComponent("Resources/kinds/" + kind + ".toml"), encoding: .utf8))
        return try spec.operations.filter { !["chat", "count_tokens"].contains($0.name) }.map { op in
            var schema = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent(op.input)))
            if op.name == "translate", var properties = schema["properties"]?.object {
                properties["target_lang"] = ["type": "string", "enum": ["EN", "PL", "DE"]]
                properties["source_lang"] = ["type": "string", "enum": ["EN", "PL", "DE"]]
                schema["properties"] = .object(properties)
            }
            let output = try op.outputSchema.map { try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent($0))) }
            return ModelOperation(name: op.name, title: op.title, inputSchema: schema, output: op.output, stream: op.stream, outputSchema: output, schemaVersion: 3)
        }
    }
}

extension FixtureModelsService {
    func fixtureCapability(_ operation: String, params: JSONObject, sink: RunSink?) async throws -> JSONObject? {
        switch operation {
        case "transcribe":
            try await sink?(.json(["event": "transcript", "delta": "Hello "]))
            try await sink?(.json(["event": "transcript", "delta": "world."]))
            return [
                "text": "Hello world.", "language": "en", "duration": 1,
                "segments": [["start": 0, "end": 1, "text": "Hello world.", "speaker": "A"]],
                "words": [["word": "Hello", "start": 0, "end": 0.5], ["word": "world.", "start": 0.5, "end": 1]]
            ]
        case "diarize": return ["segments": [["start": 0, "end": 1, "speaker": "A"]], "speakers": ["A"]]
        case "end_of_turn": return ["prediction": true, "probability": 0.9]
        case "embed":
            let dimensions = Int(params["dimensions"]?.number ?? 3)
            let vectors = (params["input"]?.array ?? []).map { _ in JSONValue.array((0..<dimensions).map { .number(Double($0 + 1) / Double(dimensions)) }) }
            return ["embeddings": .array(vectors), "input_tokens": .number(Double((params["input"]?.array ?? []).reduce(0) { $0 + ($1["content"]?.array?.count ?? 1) }))]
        case "rerank":
            let documents = params["documents"]?.array ?? [], count = min(documents.count, Int(params["top_n"]?.number ?? Double(documents.count)))
            return [
                "results": .array(
                    (0..<count).map { ["index": .number(Double($0)), "relevance_score": .number((params["instruction"] == "Find the answer" ? 0.8 : 1) / Double($0 + 1))] })
            ]
        case "translate":
            return ["translations": .array((params["text"]?.array ?? []).map { ["text": .string("Translated: " + ($0.string ?? "")), "detected_source_language": "EN"] })]
        case "read_document":
            return ["pages": [["index": 0, "markdown": "# Fixture\n\n| A |\n|---|\n| 1 |\n\n$x^2$", "tables": [["id": "table-0", "content": "| A |\n|---|\n| 1 |"]]]]]
        case "extract": return ["data": ["name": "fixture"]]
        case "detect", "ground_screen":
            return ["detections": [["label": "button", "score": 0.9, "box": ["xmin": 0.1, "ymin": 0.2, "xmax": 0.5, "ymax": 0.6]]], "coordinate_system": "normalized"]
        case "segment_session", "video_session": return ["session_id": "fixture-session", "closed": .bool(params["action"]?.string == "close")]
        case "segment", "track_video", "estimate_depth", "upscale_image", "upscale_video", "interpolate_video", "image_to_3d", "separate_audio":
            if params["prompt"]?.string == "slow" { try await Task.sleep(for: .seconds(30)) }
            guard let output = params["out_dir"]?.string else { throw WorkerFailure(code: "invalid_request", message: "Fixture requires an output directory.") }
            let folder = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let file = folder.appendingPathComponent("fixture.json")
            try Data("{\"fixture\":true}".utf8).write(to: file)
            var result: JSONObject = ["files": [["id": "fixture-output", "path": .string(file.path), "mime": "application/json"]]]
            if operation == "estimate_depth" { result["mode"] = "relative"; result["depth_unit"] = "relative" }
            return result
        default: return nil
        }
    }
}
