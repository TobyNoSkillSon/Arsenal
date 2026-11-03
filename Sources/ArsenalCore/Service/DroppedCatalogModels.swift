// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// The record itself marks withdrawal even when every installed descriptor is gone or unreadable.
struct DroppedCatalogModel: Sendable {
    var manifest: ModelManifest?
}

extension RealModelsService {
    func rememberCatalogIdentities(in configuration: inout ArsenalConfiguration) {
        for model in catalog.models where configuration.table.contains(model.id) {
            configuration.catalogModelIdentities[model.id] = .init(name: model.manifest.name, kind: model.manifest.kind)
        }
    }
    func rememberDroppedCatalogModels(next: CatalogDocument) {
        let signed = Set(next.models.map(\.id))
        droppedCatalogModels = droppedCatalogModels.filter { !signed.contains($0.key) }
        for model in catalog.models where !signed.contains(model.id) { droppedCatalogModels[model.id] = .init(manifest: model.manifest) }
    }

    func pruneUnlistedRows() {
        let signed = Set(catalog.models.map(\.id))
        for id in Array(modelRows.keys) where !signed.contains(id) && localModels[id] == nil && residents[id] == nil && droppedCatalogModels[id] == nil {
            modelRows[id] = nil; metadata[id] = nil
        }
    }

    static func hasInstalledSnapshot(id: String, support: URL) -> Bool {
        do { return try !installedDtypes(id: id, support: support).isEmpty } catch { return true }
    }
    var droppedCatalogEntries: [CatalogEntry] {
        droppedCatalogModels.keys.sorted().filter { config.table.contains($0) && onTable($0) }.compactMap { id in
            modelRows[id].map { CatalogEntry(row: $0, metadata: metadata[id] ?? [:], added: true) }
        }
    }
    /// Only a missing model root or a successful empty inspection proves that no snapshots survive.
    static func installedDtypes(id: String, support: URL) throws -> Set<String> {
        let root = try SafePath.resolve(id, inside: support.appendingPathComponent("weights/snapshots"))
        let revisions: [URL]
        do { revisions = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) } catch CocoaError.fileReadNoSuchFile { return [] }
        var dtypes = Set<String>()
        for revision in revisions {
            for dtype in try FileManager.default.contentsOfDirectory(at: revision, includingPropertiesForKeys: nil) {
                let directory = try SafePath.resolve(revision.lastPathComponent + "/" + dtype.lastPathComponent, inside: root)
                if StoredNames.hasSnapshot(directory) { dtypes.insert(dtype.lastPathComponent) }
            }
        }
        return dtypes
    }

    func requireCurrentCatalogModel(_ id: String) throws {
        if droppedCatalogModels[id] != nil {
            throw WorkerFailure(code: "catalog_removed", message: "No longer in the catalog. Remove this model to keep or delete its weights.")
        }
    }
    /// Forget per-model preferences as well as membership; shared kind ordering belongs to the user.
    func forgetDroppedCatalogModel(_ id: String) {
        config.table.removeAll { $0 == id }
        config.selections[id] = nil; config.settings[id] = nil; config.runnerOverrides[id] = nil
        config.localCodeApprovals[id] = nil; config.localCodeConsent[id] = nil; config.rejectedRevisions[id] = nil
        config.catalogModelIdentities[id] = nil
        modelSettings.forget(id); operationCache[id] = nil; operationFailures[id] = nil
        droppedCatalogModels[id] = nil; modelRows[id] = nil; metadata[id] = nil
    }
    /// Reconcile membership before local discovery can mistake a withdrawn catalog ID for a broken local folder.
    /// Installed descriptors are display/removal metadata; they grant no authority to run retired code.
    func reconcileDroppedCatalogModels() async throws {
        let signed = Set(catalog.models.map(\.id))
        droppedCatalogModels = droppedCatalogModels.filter { !signed.contains($0.key) && config.table.contains($0.key) }
        let previousIdentities = config.catalogModelIdentities
        let previousSettings = modelSettings
        rememberCatalogIdentities(in: &config)
        var removed = false
        for id in config.table where !signed.contains(id) {
            let localFolder = try SafePath.resolve(id, inside: paths.directory("local-models"))
            if localModels[id] != nil || FileManager.default.fileExists(atPath: localFolder.path) {
                droppedCatalogModels[id] = nil
                continue
            }
            let installed: Set<String>
            let inspected: Bool
            do { installed = try Self.installedDtypes(id: id, support: paths.support); inspected = true } catch { installed = []; inspected = false }
            if inspected && installed.isEmpty && residents[id] == nil {
                forgetDroppedCatalogModel(id); removed = true
                continue
            }
            // A malformed legacy descriptor must not bypass remembered/resident metadata or abort startup.
            let descriptor = try? await installer.installedManifest(id: id)
            let manifest = descriptor ?? droppedCatalogModels[id]?.manifest ?? residentManifests[id]
            droppedCatalogModels[id] = .init(manifest: manifest)
            if let manifest {
                metadata[id] = try WireJSON.decode(JSONEncoder().encode(manifest))
                config.catalogModelIdentities[id] = .init(name: manifest.name, kind: manifest.kind)
            }
            let identity = config.catalogModelIdentities[id]
            let preference = config.selections[id] ?? .init(precision: manifest?.defaultPrecision ?? installed.sorted().first ?? "fp16")
            var row = modelRows[id] ?? ModelRow(id: id, name: identity?.name ?? id, kind: identity?.kind ?? "unknown")
            row.name = identity?.name ?? row.name; row.kind = identity?.kind ?? row.kind
            row.params = manifest?.params ?? row.params; row.licence = manifest?.licence.spdx ?? row.licence
            row.precisions = Array(Set(manifest?.precisions ?? row.precisions).union(installed).union([preference.precision])).sorted()
            row.selectedPrecision = preference.precision
            row.local = false; row.needsCodeConsent = false; row.tools = []; row.needsReload = false
            row.verification = "No longer in the catalog"
            row.states = Dictionary(
                uniqueKeysWithValues: row.precisions.map { dtype in
                    (
                        dtype,
                        PrecisionState(state: loadedPreferences[id]?.precision == dtype ? .loaded : installed.contains(dtype) ? .ready : .notInstalled, phase: "catalogRemoved")
                    )
                })
            modelRows[id] = row
        }
        if removed || config.catalogModelIdentities != previousIdentities {
            do { try persist() } catch { modelSettings = previousSettings; throw error }
        }
    }
}
