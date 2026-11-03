// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// MiniMax Music 3's licence asks that the model's name be shown where it is used. Make Music lists the model by its
/// manifest name in its picker; this pins that name and the placement that puts the model in that window.
struct MiniMaxMusicNameTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @Test func makeMusicPickerShowsTheModelsName() throws {
        let kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
        let manifest = try ModelManifest.load(repo.appendingPathComponent("models/minimax-music3"), kinds: kinds)
        #expect(manifest.name == "MiniMax Music 3")
        #expect(manifest.licence.commercial == false && !(manifest.licence.conditions ?? "").isEmpty)
        let row = SoundFixture.row(manifest.id, manifest.name, kind: manifest.kind, tools: ["sound"])
        #expect(ModelPicker.title(SoundFixture.model(row)) == "MiniMax Music 3")
        #expect(ModelPicker.precisionLine(ToolModel(row)).hasPrefix("MiniMax Music 3 \u{00b7} "))
        #expect(ToolCatalog.resolve(tools: [.sound], declared: Set(manifest.operations.map(\.name))) == [.sound])
        #expect(SoundTask.music.operations == ["generate_music"])
    }
}
