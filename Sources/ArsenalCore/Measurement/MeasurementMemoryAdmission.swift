// SPDX-License-Identifier: MIT
import Foundation

struct MeasurementMemorySkip: Error, LocalizedError, Sendable {
    let needMB: Double
    var errorDescription: String? {
        String(format: "skipped: not enough free memory (needs ~%.1f GB)", locale: Locale(identifier: "en_US_POSIX"), needMB / 1000)
    }
}

enum MeasurementMemoryEstimate {
    // Bounded logits payload plus the ordinary runtime estimate's overhead. These are admission estimates, not measured peaks.
    static let payloadMB = Double(256 * 1024 * 1024) / 1e6
    static func megabytes(source: SnapshotManifest, installed: SnapshotManifest?, assets: [SnapshotManifest]) throws -> Double {
        let weights: Double
        if let installed {
            weights = try RuntimeMemoryEstimate.megabytes(snapshot: installed)
        } else {
            // A cold Get can retain source + candidate tensors. No optimistic all-quantizable assumption.
            weights = try 2 * RuntimeMemoryEstimate.megabytes(snapshot: source, overheadMB: 0) + 768
        }
        let assetMB = try assets.reduce(0.0) { try $0 + RuntimeMemoryEstimate.megabytes(snapshot: $1, overheadMB: 0) }
        return weights + assetMB + payloadMB
    }
}

extension RealModelsService {
    /// Reserve through ordinary admission before any conversion. The drain never waits or permits swap.
    func measurementGet(id: String, precision: String, approvedPlan: GetPlan) async throws {
        let model = try manifest(id)
        let installed = try await fetcher.store.verifyInstalledSnapshot(model: model, precision: precision)
        var source = installed
        if source == nil { source = try await fetcher.store.verifyInstalledSnapshot(model: model, precision: model.basePrecision) }
        guard let source else {
            throw ManifestError.invalid("Pinned local source snapshot missing; measurement never downloads")
        }
        var assets: [SnapshotManifest] = []
        for asset in model.separateAssets {
            guard let snapshot = try await fetcher.store.verifyPinnedSnapshot(model: "asset-" + asset.id, source: asset.weights, precision: "fp16") else {
                throw ManifestError.invalid("Pinned local asset snapshot missing: " + asset.id)
            }
            assets.append(snapshot)
        }
        let estimate = try MeasurementMemoryEstimate.megabytes(source: source, installed: installed, assets: assets)
        try await measurementReserve(id: id, precision: precision, estimateMB: estimate)
        defer { loadReservations[id] = nil; signalMemoryChange() }
        try await get(id: id, precision: precision, approvedPlan: approvedPlan)
    }

    /// Recheck current free memory immediately before ordinary Load, including the bounded measurement payload.
    func measurementLoad(id: String, precision: String) async throws {
        let model = try manifest(id)
        guard let installed = try await fetcher.store.verifyInstalledSnapshot(model: model, precision: precision) else {
            throw ManifestError.invalid("Served precision snapshot missing after Get")
        }
        let estimate = try RuntimeMemoryEstimate.megabytes(snapshot: installed) + MeasurementMemoryEstimate.payloadMB
        try await measurementReserve(id: id, precision: precision, estimateMB: estimate)
        // Load makes its own ordinary reservation; do not count this preflight twice.
        loadReservations[id] = nil; signalMemoryChange()
        do { try await ensureLoaded(id: id, precision: precision, fast: nil, residency: .manual, waitForMemory: false) } catch {
            if (error as? WorkerFailure)?.code == "insufficient_memory" { throw MeasurementMemorySkip(needMB: estimate + MemoryProbe.headroomMB) }
            throw error
        }
    }

    private func measurementReserve(id: String, precision: String, estimateMB: Double) async throws {
        guard !config.residency.allowSwap else { throw ManifestError.invalid("Measurement requires ordinary admission with swap disabled") }
        do { try await reserveMemory(id: id, precision: precision, estimateMB: estimateMB, waitForMemory: false) } catch {
            if (error as? WorkerFailure)?.code == "insufficient_memory" { throw MeasurementMemorySkip(needMB: estimateMB + MemoryProbe.headroomMB) }
            throw error
        }
    }
}
