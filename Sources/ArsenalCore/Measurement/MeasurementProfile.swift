// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import Darwin

/// Disposable APFS-cloned source snapshots. Source stores are opened only for reads; no verification receipts are written there.
public enum MeasurementProfile {
    public static func importSource(model: ModelManifest, sourceSupport: URL, paths: ArsenalPaths) async throws {
        let store = BlobStore(support: paths.support)
        let sources =
            [(model.id, model.weights.revision, model.basePrecision)]
            + ["int8", "int4"].compactMap { dtype in
                let revision = model.vendor?[dtype]?.revision ?? model.weights.revision
                let directory = sourceSupport.appendingPathComponent("weights/snapshots/\(model.id)/\(revision)/\(dtype)")
                return StoredNames.hasSnapshot(directory) ? (model.id, revision, dtype) : nil
            }
            + model.separateAssets.map { ("asset-" + $0.id, $0.revision, "fp16") }
        for (id, revision, dtype) in sources {
            let source = sourceSupport.appendingPathComponent("weights/snapshots/\(id)/\(revision)/\(dtype)")
            let metadata = StoredNames.snapshotURL(in: source)
            guard let bytes = try? FileDigest.bytes(metadata), bytes <= 1_048_576,
                let snapshot = try? JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: metadata)),
                snapshot.model == id, snapshot.revision == revision, snapshot.precision == dtype
            else {
                throw ManifestError.invalid("Pinned local source snapshot missing: " + id + "/" + dtype + "; measurement never downloads")
            }
            for file in snapshot.files {
                if try await store.hasBlob(sha256: file.sha256, bytes: file.bytes) { continue }
                let origin = try SafePath.resolve(file.path, inside: source).resolvingSymlinksInPath()
                let stage = paths.directory("weights").appendingPathComponent("clone-" + UUID().uuidString)
                guard clonefile(origin.path, stage.path, 0) == 0 else { throw ManifestError.invalid("Measurement requires an APFS clone of " + id + "/" + file.path) }
                do { try await store.promote(stage, sha256: file.sha256, bytes: file.bytes) } catch { try? FileManager.default.removeItem(at: stage); throw error }
            }
            try await store.commitSnapshot(snapshot)
        }
        // Admission checks against accepted pins, not only the donor's self-reported inventory.
        let fetcher = ModelFetcher(support: paths.support)
        for (id, _, dtype) in sources where id == model.id && [model.basePrecision, "int8", "int4"].contains(dtype) {
            _ = try await fetcher.verifyInstalled(model: model, precision: dtype)
        }
    }
}

extension RealModelsService {
    /// Inspection for the isolated drain; no alternate loader or inference operation.
    public func measurementManifest(id: String) throws -> ModelManifest { try manifest(id) }
}
