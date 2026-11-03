// SPDX-License-Identifier: MIT
import Foundation

extension RealModelsService {
    /// Status getters retain discovery errors but do not publish events that would re-enter the refresh.
    func refreshLocalModels(notifyChanges: Bool = true) async {
        do { try await discoverLocalModels(notifyChanges: notifyChanges); localDiscoveryError = nil } catch {
            let message = "Local models: \(error.localizedDescription)"
            localDiscoveryError = message
            if notifyChanges { emit(.error(id: nil, message: message)) }
        }
    }

    /// A model folder dropped into (or removed from) `local-models/` shows in an open table by itself: the same local
    /// rescan Refresh does, after FSEvents reports a change. Started by the app only.
    public func watchLocalFolders() throws {
        guard localFolderWatcher == nil else { return }
        let watcher = try LocalFolderWatcher(root: paths.support.appendingPathComponent("local-models", isDirectory: true)) { [weak self] in
            Task { await self?.localFoldersChanged() }
        }
        watcher.start()
        localFolderWatcher = watcher
    }
    /// One watcher rescan at a time; a change during it runs one more afterwards. Nothing once shutting down.
    func localFoldersChanged() async {
        guard !shuttingDown else { return }
        guard !localRescanRunning else { localRescanWanted = true; return }
        localRescanRunning = true
        defer { localRescanRunning = false }
        repeat {
            localRescanWanted = false
            await refreshLocalModels(notifyChanges: true)
        } while localRescanWanted && !shuttingDown
    }

    public func unload(id: String) async throws { try await unload(id: id, notifyChanges: true) }

    func unload(id: String, notifyChanges: Bool) async throws {
        beginUnload(id); defer { endUnload(id) }
        if let swap = swapTasks[id] { swap.cancel(); await swap.value }
        try await waitForRevisionSwap(id)
        if let task = loadTasks[id] { task.cancel(); _ = try? await task.value }
        try await unloadResident(id: id, notifyChanges: notifyChanges)
    }

    func beginUnload(_ id: String) { unloading[id, default: 0] += 1 }
    func endUnload(_ id: String) {
        unloading[id, default: 0] -= 1
        if unloading[id] == 0 { unloading[id] = nil }
    }
    private func lifecycleOwns(_ id: String) -> Bool {
        shuttingDown || swapTasks[id] != nil || reloading.contains(id) || loadTasks[id] != nil || gets[id] != nil || unloading[id, default: 0] > 0
    }
    /// A status read observes crashes; it never cancels lifecycle work through public Unload.
    func refreshExitedResidents(notifyChanges: Bool = true, _ hasExited: @Sendable (String) async -> Bool) async {
        for id in Array(residents.keys) {
            guard !lifecycleOwns(id), let resident = residents[id], resident.busy == 0 else { continue }
            let exited = await hasExited(id)
            // The exit query suspends: an intentional unload/swap may now own this resident,
            // or have replaced it completely. Activity also makes this observation stale.
            guard exited, !lifecycleOwns(id), residents[id]?.busy == 0, residents[id]?.lastUsed == resident.lastUsed else { continue }
            try? await unloadResident(id: id, notifyChanges: notifyChanges)
        }
    }
}
