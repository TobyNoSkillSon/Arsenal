// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import VoicesWorker

@Test func shippedSpeechMetadataMatchesTheRetainedSource() throws {
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    struct Catalog: Decodable { let models: [SpeechSpec] }
    let original = try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: package.appendingPathComponent("metadata/models.json")))
    #expect(SpeechCatalog.all.count == original.models.count)
    for (runtime, source) in zip(SpeechCatalog.all, original.models) {
        #expect(runtime.id == source.id)
        #expect(runtime.tier16_dtype == source.tier16_dtype)
        #expect(runtime.tiers_offered == source.tiers_offered)
        #expect(runtime.voices == source.voices)
        #expect(runtime.files == source.files)
    }
}
