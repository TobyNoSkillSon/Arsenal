// SPDX-License-Identifier: MIT
import Foundation
import os

/// Quiet collection of weights nothing will use: the weights of models the signed catalog no longer lists, orphan blobs and asset snapshots, and, at startup,
/// interrupted downloads and snapshot files that are copies of their blob. One log line per run; errors only reach the UI.
extension RealModelsService {
    private static let collectionLog = Logger(subsystem: "dev.arsenal.app", category: "WeightsCollection")

    /// The model IDs whose weights a collection deletes: on disk, not in the catalog, not a local folder model, not loaded.
    /// Models withdrawn from the catalog for their licence terms (28 Oct 2026). Weights already on a Mac stay until the user
    /// deletes them (Models table, `arsenal_delete`); the collection never removes them on its own.
    static let keptWithdrawnWeights: Set<String> = ["sam-audio-large", "sam-audio-small", "sam3.1", "trellis2-4b", "nemotron-3-diarization"]

    func droppedWeightIDs(onDisk: [String], kept: Set<String> = RealModelsService.keptWithdrawnWeights) -> [String] {
        let signed = Set(catalog.models.map(\.id)).union(kept)
        let localFolders = Set((try? FileManager.default.contentsOfDirectory(atPath: paths.directory("local-models").path)) ?? [])
        return onDisk.filter { !signed.contains($0) && !localFolders.contains($0) && localModels[$0] == nil && residents[$0] == nil }
    }

    /// `startup` adds what only a fresh launch may touch: interrupted downloads (a pinned partial from the last day stays to resume)
    /// and relinking snapshot copies. Skipped while another process owns the support folder (its Gets and loads are invisible here).
    @discardableResult
    func collectWeights(startup: Bool) async -> Int64 {
        guard gets.isEmpty, loadTasks.isEmpty, !StagingCleanup.anotherProcessOwns(paths) else { return 0 }
        let store = fetcher.store
        guard await store.unreadableRecords() == 0 else {
            Self.collectionLog.notice("weights collection skipped: unreadable store records")
            return 0
        }
        let pinned = allManifests.reduce(into: Set<String>()) { $0.formUnion(partialHashes($1)) }
        let keptManifests = Self.keptWithdrawnWeights.compactMap { droppedCatalogModels[$0]?.manifest }
        let keepAssets = Set((allManifests + keptManifests).flatMap { $0.separateAssets.map { $0.id + "/" + $0.revision } })
        var freed: Int64 = 0, dropped: [String] = [], failures = 0
        // A catalog override (a branch catalog in development) must never delete weights the shipped catalog still lists.
        if ProcessInfo.processInfo.environment["ARSENAL_CATALOG"] == nil {
            for id in droppedWeightIDs(onDisk: (try? await store.onDiskModelIDs()) ?? []) {
                let hashes = (droppedCatalogModels[id]?.manifest.map { partialHashes($0) } ?? []).subtracting(pinned)
                do { freed += try await store.delete(model: id, partialSHA256s: hashes); dropped.append(id) } catch {
                    failures += 1; emit(.error(id: id, message: "Removing \(id)'s weights failed: \(error.localizedDescription)"))
                }
            }
        }
        var interrupted: Int64 = 0, relinked: Int64 = 0, orphans: Int64 = 0
        if startup {
            do { interrupted = try await store.removeInterruptedDownloads(pinned: pinned, resumableSince: Date().addingTimeInterval(-86_400)) } catch {
                failures += 1
            }
            let relink = await store.relinkSnapshotCopies(); relinked = relink.freed; failures += relink.failures
        }
        do { orphans = try await store.deleteOrphans(keepAssets: keepAssets) } catch {
            failures += 1; emit(.error(id: nil, message: "Weights cleanup stopped: \(error.localizedDescription)"))
        }
        freed += interrupted + relinked + orphans
        let summary: [String: Any] = [
            "startup": startup, "dropped": dropped, "freed_bytes": freed, "interrupted_bytes": interrupted, "relinked_bytes": relinked,
            "orphan_bytes": orphans, "failures": failures
        ]
        try? await store.logCollection(summary)
        let line =
            "weights collection: freed \(freed) bytes; dropped [\(dropped.joined(separator: ","))]; "
            + "interrupted \(interrupted), relinked \(relinked), orphans \(orphans), failures \(failures)"
        Self.collectionLog.notice("\(line, privacy: .public)")
        if !dropped.isEmpty {
            // A dropped model left on the table now has nothing on disk, so reconciling forgets it like any emptied dropped model.
            try? await reconcileDroppedCatalogModels()
            await refreshInstalled(forceDownloadSizes: true); changed()
        }
        return freed
    }
}
