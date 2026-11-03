// SPDX-License-Identifier: MIT
import Foundation
import os

/// Quiet collection of what no model uses: orphan blobs and asset snapshots, and, at startup, interrupted downloads and
/// snapshot files that are copies of their blob. One log line per run; errors only reach the UI.
///
/// A model's weights are never collected, also when the signed catalog no longer lists the model: Arsenal never deletes
/// downloaded weights without the user's word (1.0.1; 1.0 deleted a dropped model's weights at first launch). A dropped
/// model stays on the table as "No longer in the catalog" with Remove, which deletes its weights when the user chooses.
extension RealModelsService {
    private static let collectionLog = Logger(subsystem: "dev.arsenal.app", category: "WeightsCollection")

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
        // A dropped model's separate assets (a vocoder, a lexicon) are part of its weights and stay with them, also when the
        // model has left the table and only its weights remain (its installed descriptor names them).
        var droppedManifests = droppedCatalogModels.values.compactMap(\.manifest)
        let listed = Set(allManifests.map(\.id)).union(droppedCatalogModels.keys)
        for id in ((try? await store.onDiskModelIDs()) ?? []) where !listed.contains(id) {
            if let manifest = try? await installer.installedManifest(id: id) { droppedManifests.append(manifest) }
        }
        let keepAssets = Set((allManifests + droppedManifests).flatMap { $0.separateAssets.map { $0.id + "/" + $0.revision } })
        var freed: Int64 = 0, failures = 0
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
            "startup": startup, "freed_bytes": freed, "interrupted_bytes": interrupted, "relinked_bytes": relinked,
            "orphan_bytes": orphans, "failures": failures
        ]
        try? await store.logCollection(summary)
        let line =
            "weights collection: freed \(freed) bytes; "
            + "interrupted \(interrupted), relinked \(relinked), orphans \(orphans), failures \(failures)"
        Self.collectionLog.notice("\(line, privacy: .public)")
        return freed
    }
}
