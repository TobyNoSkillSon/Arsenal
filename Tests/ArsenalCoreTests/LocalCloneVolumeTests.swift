// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

extension FetchTests {
    /// Get clones a donor on the store's volume and copies one from another volume, so the plan says which, per file.
    @Test func availabilityPlanSeparatesClonesFromCopiesByDonorVolume() async throws {
        let root = try temp(); defer { try? FileManager.default.removeItem(at: root) }
        let (_, model) = try fixture(root)
        let donor = root.appendingPathComponent("sources/test/good/weights.bin")
        try FileManager.default.createDirectory(at: donor.deletingLastPathComponent(), withIntermediateDirectories: true); try Self.payload.write(to: donor)
        let support = root.appendingPathComponent("support")
        for (donorVolume, clone) in [("store", true), ("external", false)] {
            let fetcher = ModelFetcher(support: support, localSources: [root.appendingPathComponent("sources").path])
            await fetcher.setVolumeResolver { url in url.path.contains("/sources/") ? donorVolume : "store" }
            let plan = try await fetcher.availabilityPlan(model: model, precision: "fp16")
            #expect(plan.localCopyBytes == file().bytes)
            #expect(plan.localCloneBytes == (clone ? file().bytes : 0))
        }
        // An unknown volume counts as a copy.
        let unknown = ModelFetcher(support: support, localSources: [root.appendingPathComponent("sources").path])
        await unknown.setVolumeResolver { _ in nil }
        #expect(try await unknown.availabilityPlan(model: model, precision: "fp16").localCloneBytes == 0)
        // A symlinked source is judged by its target's volume, and a missing path by its nearest existing folder.
        let link = root.appendingPathComponent("link.bin"); try FileManager.default.createSymbolicLink(at: link, withDestinationURL: donor)
        #expect(ModelFetcher.volume(link) != nil && ModelFetcher.volume(link) == ModelFetcher.volume(donor))
        #expect(ModelFetcher.volume(root.appendingPathComponent("missing/deeper")) == ModelFetcher.volume(root))
    }
}
