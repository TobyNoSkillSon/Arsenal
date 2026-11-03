// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct IntegrationCapabilityContractTests {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    func validate(_ file: String, _ value: JSONObject) throws {
        let schema = try WireJSON.decode(Data(contentsOf: root.appendingPathComponent("Resources/kinds/" + file)))
        try OperationSchemaValidator.validate(.object(value), schema: .object(schema))
    }
    var exemplar: JSONValue {
        ["image": ["path": "/in/example.png"], "box": ["xmin": 0.1, "ymin": 0.2, "xmax": 0.6, "ymax": 0.8], "label": "foreground"]
    }
    @Test func visualConceptPromptsAreAdditive() throws {
        try validate("perception/ops/segment.input.json", ["image": ["path": "/in/target.png"], "visual_exemplars": .array([exemplar])])
        try validate("perception/ops/segment.input.json", ["session_id": "image-1", "text": "cups", "visual_exemplars": .array([exemplar])])
        try validate("perception/ops/track_video.input.json", ["video": ["path": "/in/target.mp4"], "visual_exemplars": .array([exemplar])])
        try validate(
            "perception/ops/track_video.input.json",
            ["session_id": "video-1", "prompt_updates": [["frame_index": 18, "object_id": 2, "action": "add", "visual_exemplars": .array([exemplar])]]])
        for bad: JSONValue in [
            [], [["box": ["xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1]]],
            [["image": ["path": "/in/example.png"], "box": ["xmin": -1, "ymin": 0, "xmax": 1, "ymax": 1]]],
            [["image": ["path": "/in/example.png"], "box": ["xmin": 0, "ymin": 0, "xmax": 1]]],
            .array(Array(repeating: exemplar, count: 33))
        ] {
            #expect(throws: (any Error).self) {
                try validate("perception/ops/segment.input.json", ["session_id": "image-1", "visual_exemplars": bad])
            }
        }
    }
    @Test func visualExemplarFilesUseProductionConfinement() throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let inbox = support.appendingPathComponent("inbox")
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: support) }
        let image = inbox.appendingPathComponent("example.png")
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1sAAAAASUVORK5CYII="
        try #require(Data(base64Encoded: png)).write(to: image)
        var example = try #require(exemplar.object)
        example["image"] = ["path": .string(image.path)]
        let schema = try WireJSON.decode(Data(contentsOf: root.appendingPathComponent("Resources/kinds/perception/ops/track_video.input.json")))
        func request(_ example: JSONObject) -> JSONValue {
            ["session_id": "video-1", "prompt_updates": [["frame_index": 1, "object_id": 1, "action": "add", "visual_exemplars": [.object(example)]]]]
        }
        try OperationArtifacts.input(request(example), schema: schema, support: support)
        example["image"] = ["path": "/outside/example.png"]
        #expect(throws: WorkerFailure.self) { try OperationArtifacts.input(request(example), schema: schema, support: support) }
        try Data("not an image".utf8).write(to: image)
        example["image"] = ["path": .string(image.path)]
        #expect(throws: WorkerFailure.self) { try OperationArtifacts.input(request(example), schema: schema, support: support) }
    }
    @Test func plannedMultimodalAndInteractiveRequests() throws {
        try validate("perception/ops/estimate_depth.input.json", ["images": [["path": "/in/a.png"], ["path": "/in/b.png"]], "intrinsics": true, "points": true])
        try validate("perception/ops/video_session.input.json", ["action": "create", "video": ["path": "/in/video.mp4"]])
        try validate("perception/ops/video_session.input.json", ["action": "close", "session_id": "video-1"])
        try validate(
            "perception/ops/track_video.input.json",
            [
                "session_id": "video-1",
                "prompt_updates": [
                    ["frame_index": 18, "object_id": 2, "action": "add", "points": [["x": 0.2, "y": 0.3, "label": "foreground"]]],
                    ["frame_index": 24, "object_id": 2, "action": "remove"]
                ]
            ])
        try validate("embeddings/ops/rerank.input.json", ["query": "q", "documents": ["d"], "instruction": "Find the answer"])
        try validate(
            "embeddings/ops/embed.input.json",
            [
                "input": [["type": "content", "content": ["caption", ["type": "image", "image": ["path": "/in/a.png"]]]]], "task": "query", "instruction": "Find this",
                "dimensions": 256
            ])
        try validate("embeddings/ops/embed.input.json", ["input": ["document"], "task": "document", "instruction": "Represent this", "dimensions": 128])
        try validate("audio/ops/transcribe.input.json", ["audio": ["path": "/in/a.wav"], "diarize": true, "word_timestamps": true])
        try validate("sound/ops/generate_music.input.json", ["prompt": "Acoustic folk", "lyrics": "A verse", "duration": 30, "language": "pl", "instrumental": false])
        try validate(
            "perception/ops/ground_screen.output.json",
            ["coordinate_system": "pixels", "image_size": ["width": 1920, "height": 1080], "detections": [["label": "Send", "score": 0.9, "point": ["x": 1730, "y": 980]]]])
        #expect(throws: (any Error).self) {
            try validate(
                "perception/ops/ground_screen.output.json", ["coordinate_system": "normalized", "detections": [["label": "Send", "score": 0.9, "point": ["x": 1730, "y": 980]]]])
        }
        try validate(
            "geometry/ops/image_to_3d.output.json",
            [
                "files": [
                    ["id": "model", "path": "/out/model.glb", "mime": "model/gltf-binary"],
                    ["id": "preview", "path": "/out/preview.png", "mime": "image/png", "width": 1, "height": 1]
                ]
            ])
    }
    @Test func fileSchemasAndProductionValidatorAgree() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("result.json"); try Data("{}".utf8).write(to: file)
        for name in ["upscale_image", "upscale_video", "interpolate_video"] {
            let schema = try WireJSON.decode(Data(contentsOf: root.appendingPathComponent("Resources/kinds/restoration/ops/" + name + ".output.json")))
            let operation = ModelOperation(name: name, title: name, inputSchema: [:], output: "file", stream: false, outputSchema: schema, schemaVersion: 3)
            let result: JSONObject = ["files": [["id": "result", "path": .string(file.path), "mime": "application/json"]]]
            let validated = try OperationArtifacts.output(result, operation: operation, directory: directory)
            #expect(validated["files"]?.array?.first?["sha256"]?.string?.count == 64)
            #expect(throws: WorkerFailure.self) {
                _ = try OperationArtifacts.output(["files": [["path": .string(file.path), "mime": "application/json"]]], operation: operation, directory: directory)
            }
        }
    }
}
