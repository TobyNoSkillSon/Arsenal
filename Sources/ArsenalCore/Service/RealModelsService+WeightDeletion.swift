// SPDX-License-Identifier: MIT
import Foundation

/// Deletion and inventory of weights on disk: the table's per-precision and whole-model deletes, orphan cleanup, and the
/// on-disk listing. The store (`BlobStore`) does the file work; this layer checks what the service has in flight.
extension RealModelsService {
    /// The precision a loaded worker holds for this model, nil when none is loaded.
    func loadedPrecision(_ id: String) -> String? { residents[id] != nil ? loadedPreferences[id]?.precision : nil }
    /// Every pinned hash a model names (weights, vendor files, separate assets): the partial downloads of those are its caches.
    func partialHashes(_ m: ModelManifest) -> Set<String> {
        let specs = [m.weights] + Array((m.vendor ?? [:]).values) + (m.asset ?? []).map(\.weights)
        return Set(specs.flatMap { ($0.sha256 ?? [:]).values }.map { $0.lowercased() })
    }
    func partialHashes(id: String) -> Set<String> { (try? manifest(id)).map { partialHashes($0) } ?? [] }
    /// Whether a precision's snapshot verifies against its pins. Memoized on the stamps of its files, so a corrupt or replaced
    /// file re-verifies and an unchanged snapshot does not rehash on every table refresh.
    func snapshotIntact(_ m: ModelManifest, _ p: String) async -> Bool {
        let directory = snapshot(m, p)
        let stamps = ((try? SafePath.files(in: directory)) ?? []).sorted().map { InstalledRowCacheKey.File(directory.appendingPathComponent($0)) }
        let key = m.id + "/" + p
        if let memo = intactMemo[key], memo.stamps == stamps { return memo.intact }
        let intact = (try? await fetcher.store.verifyInstalledSnapshot(model: m, precision: p)) != nil
        intactMemo[key] = (stamps, intact)
        return intact
    }
    /// Every model with weights or caches on disk, catalog or not. A precision a loaded worker holds is `inUse`.
    public func onDiskModels() async throws -> [OnDiskModel] {
        let store = fetcher.store
        var manifests: [String: ModelManifest] = [:]
        for m in allManifests + droppedCatalogModels.values.compactMap(\.manifest) { manifests[m.id] = m }
        var ids = Set(try await store.onDiskModelIDs())
        for (id, m) in manifests {
            let partials = await store.partialBytes(sha256s: partialHashes(m))
            if partials > 0 { ids.insert(id) }
        }
        var result: [OnDiskModel] = []
        for id in ids.sorted() {
            do {
                let hashes = manifests[id].map { partialHashes($0) } ?? []
                let assets = try await store.assetCacheBytes(model: id)
                let cache = assets + (await store.partialBytes(sha256s: hashes))
                let loaded = loadedPrecision(id)
                let onDisk = try await store.precisionBytes(model: id).sorted { $0.key < $1.key }
                let dtypes = onDisk.filter { $0.value > 0 }.map(\.key)
                let precisions = onDisk.map { copy in
                    OnDiskPrecision(
                        dtype: copy.key, bytes: copy.value, inUse: copy.key == loaded,
                        dependents: manifests[id].map { Self.dependents(of: copy.key, in: $0, onDisk: dtypes) } ?? [])
                }
                guard !precisions.isEmpty || cache > 0 else { continue }
                let offered = catalog.models.contains { $0.id == id } || localModels[id] != nil
                result.append(OnDiskModel(id: id, precisions: precisions, cacheBytes: cache, offered: offered))
            } catch {
                // One unreadable record must not blank the table: this model is skipped and reported; the others still list.
                emit(.error(id: id, message: "\(id): its stored weights could not be read and are not listed (\(error.localizedDescription))."))
            }
        }
        return result
    }
    /// Orphan figures, with a report when a record is unreadable and so nothing is treated as an orphan.
    public func orphanBytes() async throws -> Int64 {
        let store = fetcher.store
        if await store.unreadableRecords() > 0 { emit(.error(id: nil, message: "Some stored weight records could not be read; orphaned weights are not counted.")) }
        return await store.orphanBytes()
    }
    /// Deletes one precision of a model, every revision. Refused while that precision is loaded, while a Get or Load of the model
    /// runs, or while a download or conversion holds its files. Other precisions of a loaded model stay deletable.
    public func deletePrecision(id: String, precision: String) async throws -> Int64 {
        try admitWork()
        let name = modelRows[id]?.name ?? id
        try await waitForRevisionSwap(id)
        // Checked after the wait: a Get or Load that started during the swap refuses this delete, whatever precision it is for.
        guard gets[id] == nil, loadTasks[id] == nil else { throw WorkerFailure(code: "busy", message: "Finish Get or Load before removing files.") }
        if loadedPrecision(id) == precision {
            throw WorkerFailure(code: "busy", message: "Unload \(name) before removing its \(precision.uppercased()) files.")
        }
        let onDisk = try await fetcher.store.precisionBytes(model: id).filter { $0.value > 0 }.map(\.key)
        let kept = (try? manifest(id)).map { Self.dependents(of: precision, in: $0, onDisk: onDisk) } ?? []
        guard kept.isEmpty else {
            throw WorkerFailure(code: "busy", message: "Delete \(kept.joined(separator: " and ")) first: it is built from this \(precision) copy.")
        }
        let freed = try await fetcher.store.delete(model: id, precision: precision, residentPrecision: loadedPrecision(id))
        // The files are gone whatever happens next: a failed switch is reported, never turned into "not deleted".
        do { try await reselectAfterDelete(id: id, deleted: [precision]) } catch { emit(.error(id: id, message: error.localizedDescription)) }
        await refreshInstalled(forceDownloadSizes: true); changed()
        return freed
    }
    /// When the precision a model is set to use was just deleted, the model switches to a copy still on disk: its 16-bit format
    /// first, then the next offered precision (after a cleanup nine models pointed at deleted copies and
    /// broke). With no copy left the choice stays, so the next Get fetches what was chosen. Returns the new precision.
    @discardableResult
    func reselectAfterDelete(id: String, deleted: [String]) async throws -> String? {
        guard let model = try? manifest(id) else { return nil }
        let chosen = config.selections[id]?.precision ?? modelRows[id]?.selectedPrecision ?? model.defaultPrecision
        guard deleted.contains(chosen) else { return nil }
        let onDisk = Set(try await fetcher.store.precisionBytes(model: id).filter { $0.value > 0 }.map(\.key)).subtracting(deleted)
        // A choice made while the store was read stands.
        guard (config.selections[id]?.precision ?? modelRows[id]?.selectedPrecision ?? model.defaultPrecision) == chosen else { return nil }
        let runner = selectedRunner(model)?.name
        // A copy of the current revision, complete: an older revision's leftover would need a download again.
        let candidates = model.precisions.filter { dtype in
            onDisk.contains(dtype) && snapshotPresent(model, dtype) && !RetiredDtype.isRetired(dtype) && runner.map { model.supports(dtype: dtype, runner: $0) } == true
                && modelRows[id]?.dtypeUnavailableReason(dtype) == nil
        }
        guard let next = candidates.first else { return nil }
        var preference = config.selections[id] ?? .init(precision: model.defaultPrecision, explicitSelection: config.table.contains(id))
        let previous = config.selections[id]
        preference.precision = next; config.selections[id] = preference
        do { try persist() } catch { config.selections[id] = previous; throw error }
        modelRows[id]?.selectedPrecision = next
        updateMetrics(model); updatePending(id)
        return next
    }
    /// Deletes every precision, revision, separate-asset snapshot, retention receipt and partial download of a model, whether or
    /// not it is in the catalog. A loaded model is refused: unload it first. Returns the bytes freed.
    public func deleteAllWeights(id: String) async throws -> Int64 {
        try admitWork()
        if let swap = swapTasks[id] { swap.cancel(); await swap.value }
        try await waitForRevisionSwap(id)
        guard gets[id] == nil, loadTasks[id] == nil else { throw WorkerFailure(code: "busy", message: "Finish Get or Load before deleting its weights.") }
        try requireIdleResident(id, forDelete: true)
        let freed = try await fetcher.store.delete(model: id, partialSHA256s: partialHashes(id: id))
        await refreshInstalled(forceDownloadSizes: true); changed()
        return freed
    }
    public func deleteOrphans() async throws -> Int64 {
        try admitWork()
        guard gets.isEmpty, loadTasks.isEmpty else { throw WorkerFailure(code: "busy", message: "Finish Get or Load before removing orphaned files.") }
        if await fetcher.store.unreadableRecords() > 0 {
            emit(.error(id: nil, message: "Some stored weight records could not be read; no orphaned weights were removed."))
            return 0
        }
        let freed = try await fetcher.store.deleteOrphans()
        changed()
        return freed
    }

    /// The copies on disk built from `dtype`'s copy, which deleting it would leave unverifiable. Only a 16-bit copy that is itself
    /// a conversion has any: its quantized copies verify through it (SnapshotValidation), while a precision with its own
    /// pinned source does not depend on it. The table's trash and the service's delete both use this rule.
    static func dependents(of dtype: String, in model: ModelManifest, onDisk: [String]) -> [String] {
        guard dtype == model.basePrecision, model.conversionRecipe(for: dtype) != nil else { return [] }
        return onDisk.filter { $0 != dtype && model.source(for: $0) == nil }.sorted()
    }

    /// Refuses while a request is in flight; with `forDelete`, also while the model is loaded. A loaded model is never unloaded
    /// behind the user's back to make room for a delete: the table disables Delete for it the same way.
    func requireIdleResident(_ id: String, forDelete: Bool = false) throws {
        guard residents[id]?.busy ?? 0 == 0 else { throw WorkerFailure(code: "busy", message: "Finish or cancel the active request before unloading.") }
        if forDelete, residents[id] != nil { throw WorkerFailure(code: "busy", message: "Unload \(modelRows[id]?.name ?? id) before deleting its weights.") }
    }
}
