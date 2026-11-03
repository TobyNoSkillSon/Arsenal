// SPDX-License-Identifier: MIT
import Foundation

extension BlobStore {
    /// Non-fetching comparison to the accepted catalog pins, not just a writable snapshot's self-reported hashes.
    public func verifyPinnedSnapshot(model: String, source: WeightSpec, precision: String, forceHash: Bool = false) throws -> SnapshotManifest? {
        guard let snapshot = try verifySnapshot(model: model, revision: source.revision, precision: precision, forceHash: forceHash) else { return nil }
        guard snapshot.derived == nil, Set(snapshot.files) == Set(try Self.pins(source)), snapshot.files.count == source.files.count else {
            throw ManifestError.invalid("installed weights differ from accepted catalog pins")
        }
        return snapshot
    }
    public func verifyInstalledSnapshot(model: ModelManifest, precision: String, forceHash: Bool = false) throws -> SnapshotManifest? {
        guard model.precisions.contains(precision) else { throw ManifestError.invalid("precision is not offered by accepted catalog") }
        if let source = model.source(for: precision) { return try verifyPinnedSnapshot(model: model.id, source: source, precision: precision, forceHash: forceHash) }
        guard let snapshot = try verifySnapshot(model: model.id, revision: model.weights.revision, precision: precision, forceHash: forceHash) else { return nil }
        guard let derived = snapshot.derived, derived.schema == 1, derived.model == model.id, derived.precision == precision,
            derived.recipe == model.conversionRecipe(for: precision), derived.sourceRevision == model.weights.revision,
            derived.sourceRepository == model.weights.repo, derived.files == snapshot.files
        else { throw ManifestError.invalid("derived weights differ from accepted recipe/source") }
        if let recorded = derived.assetSources {
            let current = Dictionary(uniqueKeysWithValues: (model.asset ?? []).filter { model.assets == nil || model.assets!.contains($0.id) }.map { ($0.id, $0.weights) })
            guard recorded == current else { throw ManifestError.invalid("derived component sources differ from accepted asset pins") }
        }
        if precision == model.basePrecision, derived.normalization != model.nativeHalfNormalization {
            throw ManifestError.invalid("native half normalization policy changed; Get again")
        }
        let expected: [DerivedFile]
        if model.conversionRecipe(for: model.basePrecision) == nil || precision == model.basePrecision {
            expected = try Self.pins(model.weights)
            guard derived.sourceDtype == (precision == model.basePrecision ? model.weights.dtype : model.basePrecision) else {
                throw ManifestError.invalid("derive source precision differs from recipe")
            }
        } else {
            // Quantization of a cast checkpoint anchors through its retained provenance if the 16-bit view was pruned.
            let source =
                try verifySnapshot(model: model.id, revision: model.weights.revision, precision: model.basePrecision, forceHash: forceHash)
                ?? sourceRetentionReceipt(model: model.id, revision: model.weights.revision, sourcePrecision: model.basePrecision)?.sourceManifest
            guard let source, let cast = source.derived, cast.sourceDtype == model.weights.dtype, cast.sourceRevision == model.weights.revision,
                cast.sourceRepository == model.weights.repo, cast.recipe == model.conversionRecipe(for: model.basePrecision), cast.normalization == model.nativeHalfNormalization,
                let pins = cast.sourceFiles, pins.count == model.weights.files.count, Set(pins) == Set(try Self.pins(model.weights)),
                Set(source.files.map(\.path)).count == source.files.count,
                cast.sourceHashes == Dictionary(uniqueKeysWithValues: pins.map { ($0.path, $0.sha256) }),
                derived.sourceDtype == model.basePrecision
            else { throw ManifestError.invalid("cast source provenance differs from accepted pins") }
            expected = source.files
        }
        guard let pins = derived.sourceFiles, Set(pins) == Set(expected), pins.count == expected.count,
            derived.sourceHashes == Dictionary(uniqueKeysWithValues: expected.map { ($0.path, $0.sha256) })
        else { throw ManifestError.invalid("derived recovery pins differ from accepted source") }
        return snapshot
    }
    private static func pins(_ source: WeightSpec) throws -> [DerivedFile] {
        guard Set(source.files).count == source.files.count else { throw ManifestError.invalid("duplicate accepted weight paths") }
        return try source.files.map { path in
            guard let hash = source.sha256?[path], ModelManifest.isHex(hash, count: 64), let bytes = source.bytes?[path], bytes >= 0 else {
                throw ManifestError.invalid("accepted weights are not fully pinned")
            }
            return DerivedFile(path: path, bytes: bytes, sha256: hash.lowercased())
        }
    }
}
