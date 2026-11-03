// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore

/// The four auxiliary repositories that ship FP32 weights are `[[asset]] into` components: Get converts each once
/// into the model's FP16 base snapshot and keeps no separate FP32 snapshot on disk.
struct IncorporatedAssetFoldersTests {
    private var repo: URL { (0..<3).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() } }

    @Test func fp32AuxiliaryAssetsFoldIntoTheHalfBase() throws {
        let expected: [String: [String: String]] = [
            "sam-audio-small": ["sam-t5-base": "text_encoder"], "sam-audio-base": ["sam-t5-base": "text_encoder"],
            "sam-audio-large": ["sam-t5-base": "text_encoder"], "yue2-3b": ["yue2-vae": "vae"],
            "trellis2-4b": ["trellis-structure-decoder": "components/trellis-structure-decoder", "trellis-dinov3": "components/trellis-dinov3"],
            "idol": ["idol-dinov3": "components/idol-dinov3", "idol-flux2-vae": "components/idol-flux2-vae"]
        ]
        for (id, assets) in expected {
            let folder = FileManager.default.fileExists(atPath: repo.appendingPathComponent("models/" + id).path) ? "models/" : "retired-models/"
            let model = try ModelManifest.load(repo.appendingPathComponent(folder + id))
            #expect(Dictionary(uniqueKeysWithValues: model.incorporatedAssets.map { ($0.id, $0.into ?? "") }) == assets, "\(id)")
            #expect(model.separateAssets.isEmpty, "\(id) keeps no separate asset snapshot")
            #expect(model.conversionRecipe(for: model.basePrecision)?.hasPrefix("cast:") == true)
        }
    }
}
