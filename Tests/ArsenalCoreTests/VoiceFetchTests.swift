// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private struct VoiceFixtureConverter: PrecisionConverter {
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile] {
        let data = try Data(contentsOf: sourceDirectory.appendingPathComponent("weights.bin")) + Data(precision.utf8)
        try data.write(to: destinationDirectory.appendingPathComponent("weights.bin"))
        return [.init(path: "weights.bin", bytes: Int64(data.count), sha256: FileDigest.sha256(data))]
    }
}

extension FetchTests {
    @Test func halfVoiceCastsRecordPinnedSourceDtypeAndQuantsDeriveFromChosenHalf() async throws {
        for half in ["bf16", "fp16"] {
            let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
            let (folder, initial) = try fixture(root, dtype: "bf16")
            var original = initial; original.kind = "voices"
            let model = ModelIntakeCommand.voiceModel(original, half: half)
            let donor = root.appendingPathComponent("local/test/good/weights.bin")
            try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self.payload.write(to: donor)
            let fetcher = ModelFetcher(
                support: root.appendingPathComponent("support"), converter: VoiceFixtureConverter(), localSources: [root.appendingPathComponent("local").path])
            let base = try await fetcher.get(model: model, modelDirectory: folder, precision: half, approvedDownloads: [:])
            let baseSnapshot = try #require(await fetcher.store.verifyInstalledSnapshot(model: model, precision: half, forceHash: true))
            #expect(baseSnapshot.derived?.sourceDtype == "bf16")
            #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("support/weights/blobs/" + FileDigest.sha256(Self.payload)).path))
            #expect(try Data(contentsOf: base.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data(half.utf8))
            let quant = try await fetcher.get(model: model, modelDirectory: folder, precision: "int8", approvedDownloads: [:])
            let quantSnapshot = try #require(await fetcher.store.verifyInstalledSnapshot(model: model, precision: "int8", forceHash: true))
            #expect(quantSnapshot.derived?.sourceDtype == half)
            #expect(try Data(contentsOf: quant.weightsDirectory.appendingPathComponent("weights.bin")) == Self.payload + Data((half + "int8").utf8))
            #expect(try Data(contentsOf: donor) == Self.payload)
        }
    }
}
