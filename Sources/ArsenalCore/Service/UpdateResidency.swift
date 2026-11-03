// SPDX-License-Identifier: MIT
import Foundation

private struct UpdateResidentState: Codable, Sendable {
    let dtype: String
    let fast: Bool
    let residency: ResidencyClass
    let keepHot: Bool
    enum CodingKeys: String, CodingKey { case dtype, fast, residency; case keepHot = "keep_hot" }
}
extension RealModelsService {
    private var updateResidencyFile: URL { paths.support.appendingPathComponent("state/update-residency.json") }
    func saveUpdateResidency() throws {
        let state = try Dictionary(
            uniqueKeysWithValues: residents.map { id, resident in
                guard let preference = loadedPreferences[id] else { throw WorkerFailure(code: "update_restore", message: "Loaded settings are missing for \(id).") }
                return (id, UpdateResidentState(dtype: preference.precision, fast: preference.fast, residency: resident.residency, keepHot: resident.keepHot))
            })
        try writeUpdateResidency(state)
    }
    private func writeUpdateResidency(_ state: [String: UpdateResidentState]) throws {
        let data = try JSONEncoder().encode(state)
        try FileManager.default.createDirectory(at: updateResidencyFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: updateResidencyFile, options: .atomic)
    }
    func clearUpdateResidency() throws {
        if FileManager.default.fileExists(atPath: updateResidencyFile.path) { try FileManager.default.removeItem(at: updateResidencyFile) }
    }
    func emitUpdateRestoreError(id: String?, error: any Error) { emit(.error(id: id, message: "Update restore: \(error.localizedDescription)")) }
    public func restoreAfterUpdate(ids: [String]) async {
        var saved: [String: UpdateResidentState]
        do {
            saved = try JSONDecoder().decode([String: UpdateResidentState].self, from: Data(contentsOf: updateResidencyFile))
        } catch {
            for id in ids { markUpdateRestoreFailure(id: id, dtype: modelRows[id]?.selectedPrecision, error: error) }
            if ids.isEmpty { try? clearUpdateResidency() }
            return
        }
        var retryableFailure = false, pruned = false, seen = Set<String>()
        for id in ids where seen.insert(id).inserted {
            // IDs pruned from an earlier handoff attempt must not fail again on the next launch.
            guard let state = saved[id] else { continue }
            let permanentFailure = !config.table.contains(id)
            // A model off the table comes back through the off-table path when it is still downloaded (`loadable`); one that
            // is gone has nothing to restore, so it is dropped from the handoff without an error.
            if permanentFailure, !loadable(id) { saved[id] = nil; pruned = true; continue }
            let previousHot = modelRows[id]?.keepHot
            do {
                // Protect a restored Keep Hot model as soon as its new resident is registered.
                modelRows[id]?.keepHot = state.keepHot
                try await ensureLoaded(id: id, precision: state.dtype, fast: state.fast, residency: state.residency, waitForMemory: false)
                residents[id]?.residency = state.residency; residents[id]?.keepHot = state.keepHot
                modelRows[id]?.keepHot = state.keepHot
                var preference = config.selections[id] ?? .init(precision: state.dtype, fast: state.fast, keepHot: state.keepHot)
                preference.keepHot = state.keepHot; config.selections[id] = preference
                persistSelectionBestEffort(id); armTimer(id); changed()
            } catch {
                if permanentFailure { saved[id] = nil; pruned = true } else { retryableFailure = true }
                if let previousHot { modelRows[id]?.keepHot = previousHot }
                markUpdateRestoreFailure(id: id, dtype: state.dtype, error: error)
            }
        }
        do {
            if !retryableFailure || saved.isEmpty { try clearUpdateResidency() } else if pruned { try writeUpdateResidency(saved) }
        } catch { emitUpdateRestoreError(id: nil, error: error) }
    }
    private func markUpdateRestoreFailure(id: String, dtype: String?, error: any Error) {
        if let row = modelRows[id] {
            var dtypes = Set([row.selectedPrecision])
            if let dtype, row.precisions.contains(dtype) { dtypes.insert(dtype) }
            for dtype in dtypes { modelRows[id]?.states[dtype] = .init(state: .failed, error: error.localizedDescription, phase: "update_restore_failed") }
        }
        emitUpdateRestoreError(id: id, error: error); changed()
    }
}
