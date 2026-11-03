// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// Core review, 2 Nov: Make Music showed "Unavailable" and never opened for ACE-Step, because the menu's admission asked
/// the sound kind's default operation (generate_sound), which a music-only model does not declare.
@Suite struct MakeMusicAdmissionTests {
    private static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    @Test func soundAdmissionAsksTheAudioOperationTheModelDeclares() throws {
        let kinds = try KindSpec.load(directory: Self.repo.appendingPathComponent("Resources/kinds"))
        let sound = try #require(kinds.first { $0.id == "sound" })
        let ace = try ModelManifest.load(Self.repo.appendingPathComponent("models/ace-step-1.5"), kindsDirectory: Self.repo.appendingPathComponent("Resources/kinds"))
        let declared = Set(ace.operations.map(\.name))
        #expect(!declared.contains("generate_sound") && declared.contains("generate_music"))
        #expect(ToolCenter.soundOperation(sound, declared: declared) == "generate_music")
        // A model with both, or with nothing declared yet, keeps the kind's default.
        #expect(ToolCenter.soundOperation(sound, declared: ["generate_sound", "generate_music"]) == "generate_sound")
        #expect(ToolCenter.soundOperation(sound) == "generate_sound")
    }
}
