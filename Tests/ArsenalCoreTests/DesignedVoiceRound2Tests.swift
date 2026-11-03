// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct DesignedVoiceRound2Tests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @Test func everyDesignModelAcceptsDurableName() throws {
        let folders = try ["models", "retired-models"].flatMap { try FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent($0), includingPropertiesForKeys: nil) }
        var checked = 0
        for folder in folders where FileManager.default.fileExists(atPath: folder.appendingPathComponent("model.toml").path) {
            let model = try ModelManifest.load(folder)
            let ops = model.resolvedOperations(kind: nil)
            guard let op = ops.first(where: { $0.name == "design_voice" }) else { continue }
            let schema = try WireJSON.decode(OperationSchema.read(path: op.input, folder: folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds")))
            try OperationSchemaValidator.validate(.object(["description": "Warm narrator", "name": "Narrator"]), schema: .object(schema))
            #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate(.object(["description": "Warm narrator", "name": ""]), schema: .object(schema)) }
            checked += 1
        }
        #expect(checked == 2)
    }
    @Test func savingChosenPreviewKeepsSeedRevisionAndAudio() throws {
        let root = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-f/store-" + UUID().uuidString)
        let directory = root.appendingPathComponent("jobs/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let audio = directory.appendingPathComponent("preview.wav"), bytes = Data("chosen preview number two".utf8)
        try bytes.write(to: audio)
        let id = "preview-" + UUID().uuidString.lowercased()
        let design: JSONObject = [
            "seed": 24301, "model_revision": .string(String(repeating: "a", count: 40)), "dtype": "bf16", "fast": true, "preview_text": "Hello world.", "language": "english",
            "generation": ["guidance": 3, "temperature": 0.7]
        ]
        try DesignedVoicePreviews.record(
            model: "qwen3-tts-voicedesign", request: ["description": "Warm narrator"],
            result: ["voice_id": .string(id), "preview": ["path": .string(audio.path)], "design": .object(design)], support: root)
        let result = try DesignedVoicePreviews.save(model: "qwen3-tts-voicedesign", previewID: id, name: "Narrator", support: root)
        let saved = try #require(try SavedVoiceStore.list(model: "qwen3-tts-voicedesign", support: root).first?.object)
        #expect(saved["seed"] == 24301)
        #expect(saved["generation"] == design["generation"])
        #expect(saved["modelRevision"] == design["model_revision"])
        #expect(saved["previewText"] == "Hello world.")
        #expect(saved["language"] == "english")
        #expect(saved["id"] == result["voice_id"])
        let copied = try #require(saved["preview"]?.string)
        #expect(try Data(contentsOf: URL(fileURLWithPath: copied)) == bytes)
    }
}
