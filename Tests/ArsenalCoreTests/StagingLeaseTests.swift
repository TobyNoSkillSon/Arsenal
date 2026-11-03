// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Transient Get staging: a crashed Get's leftovers go at the next Get by their lease, a live lease protects its
/// directory, and the startup sweep takes `.asset-` staging with its lease file.
struct StagingLeaseTests {
    private func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }
    private func directory(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: url.appendingPathComponent("source.bin"))
    }
    private func age(_ url: URL, days: Double) throws {
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-days * 86_400)], ofItemAtPath: url.path)
    }

    @Test func leftoverLeaseOfACrashedGetIsReclaimedWithoutWaitingForItsAge() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let weights = root.appendingPathComponent("weights")
        defer { try? FileManager.default.removeItem(at: root) }
        let dead = weights.appendingPathComponent(".cast-" + UUID().uuidString)
        try directory(dead)
        // A crashed process leaves its lease file behind, holding no lock.
        let lease = ConversionStaging.SiblingLease.leaseURL(for: dead)
        try Data().write(to: lease)
        try ConversionStaging.recover(in: weights)
        #expect(!exists(dead))
        #expect(!exists(lease))
    }

    @Test func liveLeasesProtectTransientAssetAndCastDirectories() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let weights = root.appendingPathComponent("weights")
        defer { try? FileManager.default.removeItem(at: root) }
        let asset = weights.appendingPathComponent(".asset-" + UUID().uuidString)
        let cast = weights.appendingPathComponent(".cast-" + UUID().uuidString)
        try directory(asset); try directory(cast)
        let assetLease = try ConversionStaging.SiblingLease(asset), castLease = try ConversionStaging.SiblingLease(cast)
        try ConversionStaging.recover(in: weights)
        #expect(exists(asset) && exists(cast))
        withExtendedLifetime((assetLease, castLease)) {}
    }

    @Test func leaselessLeftoversWaitForTheirAge() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let weights = root.appendingPathComponent("weights")
        defer { try? FileManager.default.removeItem(at: root) }
        let fresh = weights.appendingPathComponent(".asset-" + UUID().uuidString)
        let old = weights.appendingPathComponent(".cast-" + UUID().uuidString)
        try directory(fresh); try directory(old); try age(old, days: 2)
        try ConversionStaging.recover(in: weights)
        #expect(exists(fresh))
        #expect(!exists(old))
    }

    @Test func startupSweepTakesAssetStagingWithItsLeaseFile() throws {
        let paths = ArsenalPaths(support: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
        defer { try? FileManager.default.removeItem(at: paths.support) }
        try paths.create()
        let asset = paths.directory("weights").appendingPathComponent(".asset-" + UUID().uuidString)
        try directory(asset)
        let lease = ConversionStaging.SiblingLease.leaseURL(for: asset)
        try Data().write(to: lease)
        try age(lease, days: 2); try age(asset, days: 2)
        #expect(try StagingCleanup.run(paths: paths) == 2)
        #expect(!exists(asset) && !exists(lease))
    }
}
