// SPDX-License-Identifier: MIT
import Foundation
import Testing

@testable import ArsenalCore

/// A voice model run without an operation speaks: operations resolve in name order, so without the kind's primary
/// operation clone_voice and design_voice came first and a plain run cloned or designed instead.
@Suite struct VoicesDefaultOperationTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @Test func everyCatalogVoiceModelDefaultsToSpeak() throws {
        let kinds = repo.appendingPathComponent("Resources/kinds")
        let voices = try #require(try KindSpec.load(directory: kinds).first { $0.id == "voices" })
        #expect(voices.primaryOperation == "speak")
        for id in ["kokoro-82m", "pocket-tts", "qwen3-tts-base", "qwen3-tts-voicedesign"] {
            let model = try ModelManifest.load(repo.appendingPathComponent("models/\(id)"), kindsDirectory: kinds)
            let operations = model.resolvedOperations(kind: voices).map { declaration in
                ModelOperation(
                    name: declaration.name, title: declaration.title, inputSchema: [:], output: declaration.output, stream: declaration.stream,
                    annotations: voices.primaryOperation == declaration.name ? ["primary": true] : nil)
            }
            #expect(RunAdmission.primaryOperation(operations, preferred: nil) == "speak", "\(id)")
        }
    }
}
