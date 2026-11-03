// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The line under a model's name: every catalog folder declares it, a folder without one gets it from its kind and
/// operations, and the manifest refuses one that would not fit one short line.
struct ModelPurposeTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @Test func everyModelFolderDeclaresAShortPurpose() throws {
        let models = repo.appendingPathComponent("models")
        let ids = try FileManager.default.contentsOfDirectory(atPath: models.path).filter {
            FileManager.default.fileExists(atPath: models.appendingPathComponent("\($0)/model.toml").path)
        }.sorted()
        // Derived, not hardcoded: every directory under models/ is a model folder: the 61 shipped models and one chat folder a user adds.
        let directories = try FileManager.default.contentsOfDirectory(at: models, includingPropertiesForKeys: [.isDirectoryKey])
            .filter { !$0.lastPathComponent.hasPrefix(".") && (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
        #expect(ids.count == directories.count && ids.count >= 45)
        var purposes: [String: String] = [:]
        for id in ids {
            let m = try ModelManifest.load(models.appendingPathComponent(id), requirePins: false, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
            let purpose = try #require(m.purpose, "\(id) declares no purpose")
            #expect(purpose.count <= ModelPurpose.maxLength)
            purposes[id] = purpose
        }
        // Each says what the model does that its group header and neighbours don't.
        let headers = ["Decisions", "Chat", "Images", "Voices", "Sound", "Enhancement", "Video"]
        #expect(purposes.values.allSatisfy { !headers.contains($0) })
        #expect(Set(purposes.filter { ["laya-english", "laya-multilingual", "laya-typed-decisions", "d1-3b", "von-1.1", "von-1.2"].contains($0.key) }.values).count == 6)
        #expect(purposes["flux2-klein-4b"] == "Image generation" && purposes["birefnet"] == "Background removal")
        // The two Stable Audio models are back in the catalog (revenue-capped licence, accepted on zero income).
        #expect(purposes["stable-audio-3-medium"] == "Sound and music generation" && purposes["stable-audio-3-small-sfx"] == "Sound effects")
        #expect(purposes["laya-english"] == "English decisions" && purposes["laya-multilingual"] == "Decisions in 100+ languages")
        #expect(purposes["laya-typed-decisions"] == "Typed workflow decisions" && purposes["d1-3b"] == "Text and image decisions")
        #expect(purposes["kokoro-82m"] == "Fast English speech")
    }

    @Test func aFolderWithoutOneGetsItFromItsKindAndOperations() {
        #expect(ModelPurpose.derive(kind: "images", operations: ["generate"]) == "Image generation")
        #expect(ModelPurpose.derive(kind: "images", operations: ["remove_background"]) == "Background removal")
        #expect(ModelPurpose.derive(kind: "images", operations: ["generate", "remove_background"]) == "Image generation, background removal")
        #expect(ModelPurpose.derive(kind: "sound", operations: ["generate_sound", "generate_music"]) == "Sound and music generation")
        #expect(ModelPurpose.derive(kind: "sound", operations: ["generate_sound"]) == "Sound generation")
        #expect(ModelPurpose.derive(kind: "voices", operations: ["speak", "voices"]) == "Speech")
        #expect(ModelPurpose.derive(kind: "voices", operations: ["speak", "clone_voice"]) == "Speech, voice cloning")
        #expect(ModelPurpose.derive(kind: "voices", operations: ["speak", "design_voice", "clone_voice"]) == "Speech, voice design and cloning")
        // Only the group header again: nothing.
        #expect(ModelPurpose.derive(kind: "decisions", operations: []) == "")
        #expect(ModelPurpose.derive(kind: "video", operations: []) == "")
        #expect(ModelPurpose.derive(kind: "enhancement", operations: ["enhance"]) == "Audio cleanup")
        #expect(ModelPurpose.derive(kind: "chat", operations: ["chat", "count_tokens"]) == "")
        // A user's own kind: its first operation, unless that is its title.
        #expect(ModelPurpose.derive(kind: "local.layers", operations: ["separate"], kindTitle: "Layers") == "Separate")
        #expect(ModelPurpose.derive(kind: "local.layers", operations: ["layers"], kindTitle: "Layers") == "")
        #expect(ModelPurpose.derive(kind: "local.layers", operations: ["separate_layers"]) == "Separate layers")
    }

    @Test func theManifestRefusesAPurposeThatIsNotOneShortLine() throws {
        let folder = repo.appendingPathComponent("models/birefnet")
        var m = try ModelManifest.load(folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        #expect(ModelPurpose.text(m, kind: nil) == "Background removal")
        m.purpose = nil
        #expect(ModelPurpose.text(m, kind: nil) == "Background removal", "derived from remove_background")
        for bad in ["", "   ", String(repeating: "x", count: 41), "Two\nlines"] {
            m.purpose = bad
            #expect(m.problems(folder: folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds")).contains { $0.contains("purpose") }, "\(bad.debugDescription)")
        }
        m.purpose = "Background removal"
        #expect(!m.problems(folder: folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds")).contains { $0.contains("purpose") })
    }
}
