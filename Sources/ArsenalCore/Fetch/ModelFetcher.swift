// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

public struct FetchResult: Sendable {
    public var weightsDirectory: URL
    public var assets: [String: URL]
}
public actor ModelFetcher {
    public let support: URL
    public let store: BlobStore
    public let client: HuggingFaceClient
    public let converter: (any PrecisionConverter)?
    public private(set) var localSources: LocalModelSources
    private let gate = GetSerialGate()
    private var inFlight: Set<String> = []
    private var approvedDownloads: [String: Int64]?
    private var rejectedDonors: Set<String> = []
    public init(support: URL, client: HuggingFaceClient = HuggingFaceClient(), converter: (any PrecisionConverter)? = nil, localSources: [String]? = nil) {
        self.support = support; self.store = BlobStore(support: support); self.client = client; self.converter = converter
        self.localSources = LocalModelSources(paths: localSources)
    }
    public func setLocalSources(_ paths: [String]?) { localSources = LocalModelSources(paths: paths) }
    /// The volume a file is on after resolving symlinks (nil when unknown, which counts as a copy). Tests inject one.
    public private(set) var volumeOf: @Sendable (URL) -> String? = ModelFetcher.volume
    public func setVolumeResolver(_ resolver: @escaping @Sendable (URL) -> String?) { volumeOf = resolver }
    public static func volume(_ url: URL) -> String? {
        var probe = url.resolvingSymlinksInPath()
        while !FileManager.default.fileExists(atPath: probe.path), probe.path != "/" { probe.deleteLastPathComponent() }
        return (try? probe.resolvingSymlinksInPath().resourceValues(forKeys: [.volumeIdentifierKey]).volumeIdentifier).map { String(describing: $0) }
    }
    public func get(
        model: ModelManifest, modelDirectory: URL, precision: String, kindsDirectory: URL? = nil, converter: (any PrecisionConverter)? = nil,
        approvedDownloads: [String: Int64]? = nil, progress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }, deriving: @escaping @Sendable () -> Void = {}
    ) async throws -> FetchResult {
        guard model.precisions.contains(precision) else { throw ManifestError.invalid("dtype is not offered by \(model.id)") }
        guard inFlight.insert(model.id).inserted else { throw ManifestError.invalid("Get already running for \(model.id)") }
        defer { inFlight.remove(model.id) }
        let errors = model.problems(folder: modelDirectory, kindsDirectory: kindsDirectory)
        guard errors.isEmpty else { throw ManifestError.invalid(errors.joined(separator: "; ")) }
        let transientCast = model.conversionRecipe(for: model.basePrecision)?.hasPrefix("cast:") == true
        let reserved = try ((transientCast ? [] : [model.weights]) + Array((model.vendor ?? [:]).values) + (model.asset ?? []).map(\.weights)).flatMap { try pinnedFiles($0) }
        let ticket = UUID(); try await gate.acquire(ticket)
        self.approvedDownloads = approvedDownloads
        let assetIDs = (model.asset ?? []).filter { model.assets == nil || model.assets!.contains($0.id) }.map { "asset-" + $0.id }
        await store.retain(model: model.id)
        for id in assetIDs { await store.retain(model: id) }
        await store.reserve(files: reserved)
        var released = false
        do {
            try Task.checkCancellation()
            try ConversionStaging.recover(in: support.appendingPathComponent("weights"))
            let result = try await performGet(
                model: model, modelDirectory: modelDirectory, precision: precision, converter: converter ?? self.converter, progress: progress, deriving: deriving)
            await store.unreserve(files: reserved)
            await store.release(model: model.id)
            for id in assetIDs { await store.release(model: id) }
            self.approvedDownloads = nil; await gate.release(ticket); released = true
            if model.normalizesNativeHalf { try await store.retireUnnormalizedSource(model) }
            if model.weights.dtype == "fp32" || transientCast { try await store.collectUnreferenced(files: pinnedFiles(model.weights)) }
            // Incorporated assets now live inside the base snapshot; their staged originals are not kept.
            for asset in model.incorporatedAssets { try await store.collectUnreferenced(files: pinnedFiles(asset.weights)) }
            return result
        } catch {
            if !released {
                await store.unreserve(files: reserved); await store.release(model: model.id)
                for id in assetIDs { await store.release(model: id) }
                self.approvedDownloads = nil; await gate.release(ticket)
            }
            throw error
        }
    }
    private func performGet(
        model: ModelManifest, modelDirectory: URL, precision: String, converter: (any PrecisionConverter)?,
        progress: @escaping @Sendable (DownloadProgress) -> Void,
        deriving: @escaping @Sendable () -> Void
    ) async throws -> FetchResult {
        var assets: [String: URL] = [:]
        for id in model.assets ?? (model.asset ?? []).map(\.id) {
            guard let asset = model.asset?.first(where: { $0.id == id }) else { throw ManifestError.invalid("missing asset \(id)") }
            guard asset.into == nil else { continue }
            let identity = "asset-" + asset.id
            assets[id] = try await downloadSnapshot(modelID: identity, source: asset.weights, precision: "fp16", progress: progress)
        }
        // Incorporated assets are inputs to the base conversion only: staged from verified blobs while it runs, never
        // committed as their own snapshot. A tier derived from an installed base already contains them.
        var staged: [URL] = [], stagedLeases: [ConversionStaging.SiblingLease] = []
        defer { for directory in staged { try? FileManager.default.removeItem(at: directory) }; withExtendedLifetime(stagedLeases) {} }
        if !model.incorporatedAssets.isEmpty, (try? await store.verifyInstalledSnapshot(model: model, precision: model.basePrecision)) == nil {
            for asset in model.incorporatedAssets {
                let directory = support.appendingPathComponent("weights/.asset-" + UUID().uuidString)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                stagedLeases.append(try ConversionStaging.SiblingLease(directory))
                staged.append(directory)
                for file in try pinnedFiles(asset.weights) {
                    try Task.checkCancellation()
                    try await ensureBlob(file, source: asset.weights, progress: progress)
                    try await store.linkVerifiedBlob(file, to: SafePath.resolve(file.path, inside: directory))
                }
                assets[asset.id] = directory
            }
        }
        let directory = try await getPrecision(
            model: model, modelDirectory: modelDirectory, precision: precision, converter: converter, assets: assets, progress: progress, deriving: deriving)
        let revisions = Dictionary(uniqueKeysWithValues: model.separateAssets.filter { assets[$0.id] != nil }.map { ($0.id, $0.revision) })
        try await store.recordAssets(model: model.id, revisions: revisions)
        return FetchResult(weightsDirectory: directory, assets: assets.filter { id, _ in model.separateAssets.contains { $0.id == id } })
    }
    private func getPrecision(
        model: ModelManifest, modelDirectory: URL, precision: String, converter: (any PrecisionConverter)?, assets: [String: URL],
        progress: @escaping @Sendable (DownloadProgress) -> Void,
        deriving: @escaping @Sendable () -> Void
    ) async throws -> URL {
        let revision = model.vendor?[precision]?.revision ?? model.weights.revision
        if (try? await store.verifyInstalledSnapshot(model: model, precision: precision)) != nil {
            return try await store.snapshotURL(model: model.id, revision: revision, precision: precision)
        }
        if let source = model.source(for: precision) { return try await downloadSnapshot(modelID: model.id, source: source, precision: precision, progress: progress) }
        guard let recipe = model.conversionRecipe(for: precision), let converter else { throw ManifestError.invalid("no converter for \(model.id) \(precision)") }
        // These model recipes read the original source file by file themselves. SAM 3.1 and SAM Audio declare a raw
        // source ([precision] source = "raw") and take the whole-snapshot raw path below, which skips half normalization.
        if recipe == NativeHalfPolicy.recipe
            || precision == "fp16" && ["cast:flux2-klein-4b", "cast:stable-audio-3-medium", "cast:stable-audio-3-small-sfx"].contains(recipe)
        {
            return try await castFiles(
                model: model, modelDirectory: modelDirectory, converter: converter, recipe: recipe, precision: precision, progress: progress, deriving: deriving)
        }
        let sourceDirectory: URL
        let sourceFiles: [DerivedFile]
        var transientSource: URL?, transientLease: ConversionStaging.SiblingLease?
        defer {
            if let transientSource { try? FileManager.default.removeItem(at: transientSource) }
            withExtendedLifetime(transientLease) {}
        }
        if ["fp16", "bf16"].contains(precision), precision == model.basePrecision, model.conversionRecipe(for: precision) != nil {
            let temp = support.appendingPathComponent("weights/.cast-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
            transientLease = try ConversionStaging.SiblingLease(temp)
            transientSource = temp
            sourceFiles = try pinnedFiles(model.weights)
            if model.normalizesNativeHalf && !model.castsRawSource {
                try await normalizedSource(model: model, modelDirectory: modelDirectory, converter: converter, destination: temp, progress: progress, deriving: deriving)
            } else {
                for file in sourceFiles {
                    try await ensureBlob(file, source: model.weights, progress: progress)
                    try await store.linkVerifiedBlob(file, to: SafePath.resolve(file.path, inside: temp))
                }
            }
            sourceDirectory = temp
        } else {
            sourceDirectory = try await getPrecision(
                model: model, modelDirectory: modelDirectory, precision: model.basePrecision,
                converter: converter, assets: assets, progress: progress, deriving: deriving)
            guard let source = try await store.verifyInstalledSnapshot(model: model, precision: model.basePrecision) else {
                throw ManifestError.invalid("derive source snapshot is missing")
            }
            sourceFiles = source.files
        }
        let sourceHashes = Dictionary(uniqueKeysWithValues: sourceFiles.map { ($0.path, $0.sha256) })
        try ConversionStaging.preflight(in: support.appendingPathComponent("weights"), sourceBytes: sourceFiles.reduce(0) { $0 + $1.bytes })
        deriving(); let staging = try ConversionStaging(support.appendingPathComponent("weights/.derive-\(UUID().uuidString)"))
        let output = staging.root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { withExtendedLifetime(staging) {} }
        let files = try await converter.convert(
            model: model, modelDirectory: modelDirectory, sourceDirectory: sourceDirectory, assets: assets, destinationDirectory: output, precision: precision, recipe: recipe)
        guard !files.isEmpty, Set(files.map(\.path)).count == files.count, Set(try SafePath.files(in: output)) == Set(files.map(\.path)) else {
            throw ManifestError.invalid("converter returned missing, duplicate or unlisted files")
        }
        for asset in model.incorporatedAssets {
            guard let into = asset.into, files.contains(where: { $0.path.hasPrefix(into + "/") }) else {
                throw ManifestError.invalid("conversion did not store asset \(asset.id) in \(asset.into ?? "")")
            }
        }
        await store.reserve(files: files)
        do {
            for file in files { try await store.promote(SafePath.resolve(file.path, inside: output), sha256: file.sha256, bytes: file.bytes) }
            let provenance = DerivedManifest(
                model: model.id, precision: precision, sourceRevision: model.weights.revision,
                sourceDtype: transientSource != nil ? model.weights.dtype : model.basePrecision, sourceHashes: sourceHashes, sourceFiles: sourceFiles,
                sourceRepository: model.weights.repo, normalization: precision == model.basePrecision ? model.nativeHalfNormalization : nil,
                assetSources: Dictionary(uniqueKeysWithValues: model.selectedAssets.filter { assets[$0.id] != nil || $0.into != nil }.map { ($0.id, $0.weights) }),
                recipe: recipe, files: files)
            let result = try await store.commitAlreadyVerifiedSnapshot(
                SnapshotManifest(model: model.id, revision: model.weights.revision, precision: precision, files: files, derived: provenance))
            await store.unreserve(files: files); return result
        } catch { await store.unreserve(files: files); throw error }
    }
    /// Cast one verified source file at a time. No source snapshot is committed; originals are collected after
    /// each converted file, unless another installed snapshot or in-flight Get actually references them.
    private func castFiles(
        model: ModelManifest, modelDirectory: URL, converter: any PrecisionConverter, recipe: String, precision: String,
        progress: @escaping @Sendable (DownloadProgress) -> Void, deriving: @escaping @Sendable () -> Void
    ) async throws -> URL {
        let sources = try pinnedFiles(model.weights)
        var converted: [DerivedFile] = []
        do {
            for file in sources {
                try Task.checkCancellation()
                let root = support.appendingPathComponent("weights/.cast-" + UUID().uuidString)
                let input = root.appendingPathComponent("input"), output = root.appendingPathComponent("output")
                try ConversionStaging.preflight(in: support.appendingPathComponent("weights"), sourceBytes: file.bytes)
                let staging = try ConversionStaging(root)
                await store.reserve(files: [file])
                defer { withExtendedLifetime(staging) {} }
                do {
                    try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
                    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                    try await ensureBlob(file, source: model.weights, progress: progress)
                    try await store.linkVerifiedBlob(file, to: SafePath.resolve(file.path, inside: input))
                    deriving()
                    let files = try await converter.convert(
                        model: model, modelDirectory: modelDirectory, sourceDirectory: input,
                        destinationDirectory: output, precision: precision, recipe: recipe)
                    guard !files.isEmpty, Set(files.map(\.path)).count == files.count,
                        Set(try SafePath.files(in: output)) == Set(files.map(\.path)),
                        Set(converted.map(\.path)).isDisjoint(with: files.map(\.path))
                    else { throw ManifestError.invalid("Cast returned missing, duplicate or unlisted files") }
                    await store.reserve(files: files)
                    converted += files
                    for result in files { try await store.promote(SafePath.resolve(result.path, inside: output), sha256: result.sha256, bytes: result.bytes) }
                    try FileManager.default.removeItem(at: root)
                    await store.unreserve(files: [file]); try await store.collectUnreferenced(files: [file])
                } catch {
                    try? FileManager.default.removeItem(at: root)
                    await store.unreserve(files: [file]); try? await store.collectUnreferenced(files: [file])
                    throw error
                }
            }
            let provenance = DerivedManifest(
                model: model.id, precision: precision, sourceRevision: model.weights.revision,
                sourceDtype: model.weights.dtype, sourceHashes: Dictionary(uniqueKeysWithValues: sources.map { ($0.path, $0.sha256) }),
                sourceFiles: sources, sourceRepository: model.weights.repo, normalization: model.nativeHalfNormalization, recipe: recipe, files: converted)
            let url = try await store.commitAlreadyVerifiedSnapshot(
                SnapshotManifest(
                    model: model.id, revision: model.weights.revision,
                    precision: precision, files: converted, derived: provenance))
            await store.unreserve(files: converted)
            return url
        } catch {
            await store.unreserve(files: converted); try? await store.collectUnreferenced(files: converted)
            throw error
        }
    }
    private func downloadSnapshot(modelID: String, source: WeightSpec, precision: String, progress: @escaping @Sendable (DownloadProgress) -> Void) async throws -> URL {
        let files = try pinnedFiles(source)
        if (try? await store.verifyPinnedSnapshot(model: modelID, source: source, precision: precision)) != nil {
            return try await store.snapshotURL(model: modelID, revision: source.revision, precision: precision)
        }
        for file in files {
            try Task.checkCancellation()
            try await ensureBlob(file, source: source, progress: progress)
        }
        return try await store.commitAlreadyVerifiedSnapshot(SnapshotManifest(model: modelID, revision: source.revision, precision: precision, files: files, derived: nil))
    }
    func ensureBlob(_ file: DerivedFile, source: WeightSpec, progress: @escaping @Sendable (DownloadProgress) -> Void) async throws {
        if try await store.hasBlob(sha256: file.sha256, bytes: file.bytes) {
            progress(DownloadProgress(file: file.path, completedBytes: file.bytes, totalBytes: file.bytes, bytesPerSecond: 0, contentID: file.sha256, phase: "verifyingLocalCopy"))
            return
        }
        for candidate in try localSources.candidates(repo: source.repo, revision: source.revision, path: file.path, repoType: source.repoType ?? "model") {
            try Task.checkCancellation()
            let reporter = LocalVerificationProgress(file: file, sink: progress)
            if try await store.importLocal(file: candidate, sha256: file.sha256, bytes: file.bytes, progress: { reporter.report($0) }) { return }
            rejectedDonors.insert(donorKey(candidate, hash: file.sha256))
            // Inspected bytes are not admitted bytes. Explicitly discard this failed donor's accounting.
            progress(
                DownloadProgress(file: file.path, completedBytes: 0, totalBytes: file.bytes, bytesPerSecond: 0, contentID: file.sha256, phase: "downloading", resetsProgress: true))
        }
        if let approvedDownloads, approvedDownloads[file.sha256] != file.bytes {
            throw WorkerFailure(code: "get_plan_changed", message: "Get needs an additional download; review the revised plan before retrying.")
        }
        let part = try await client.downloadVerified(
            repo: source.repo, revision: source.revision, file: file, directory: support.appendingPathComponent("weights/downloads"), repoType: source.repoType ?? "model",
            progress: report(file, to: progress))
        try await store.promote(part)
    }
    public func verifyInstalled(model: ModelManifest, precision: String, forceHash: Bool = false) async throws -> FetchResult {
        guard try await store.verifyInstalledSnapshot(model: model, precision: precision, forceHash: forceHash) != nil else {
            throw ManifestError.invalid("weights are not installed; use \(MCPNames.prefix)download(model: \(model.id))")
        }
        let directory = try await store.snapshotURL(model: model.id, revision: model.vendor?[precision]?.revision ?? model.weights.revision, precision: precision)
        var assets: [String: URL] = [:]
        for asset in model.separateAssets {
            guard try await store.verifyPinnedSnapshot(model: "asset-" + asset.id, source: asset.weights, precision: "fp16", forceHash: forceHash) != nil else {
                throw ManifestError.invalid("required asset is missing; use \(MCPNames.prefix)download(model: \(model.id))")
            }
            assets[asset.id] = try await store.snapshotURL(model: "asset-" + asset.id, revision: asset.revision, precision: "fp16")
        }
        return FetchResult(weightsDirectory: directory, assets: assets)
    }
    /// Table planning reads only tiny snapshot metadata, never weight bytes or blob hashes.
    public func updateRequired(model: ModelManifest, precision: String) async throws -> Bool {
        let revision = model.vendor?[precision]?.revision ?? model.weights.revision
        let directory = try await store.snapshotURL(model: model.id, revision: revision, precision: precision)
        if let data = try? Data(contentsOf: StoredNames.snapshotURL(in: directory)),
            let snapshot = try? JSONDecoder().decode(SnapshotManifest.self, from: data)
        {
            if let source = model.source(for: precision) {
                if Set(snapshot.files) != Set(try pinnedFiles(source)) || snapshot.derived != nil { return true }
            } else if snapshot.derived?.recipe != model.conversionRecipe(for: precision) || snapshot.derived?.sourceRevision != model.weights.revision
                || snapshot.derived?.sourceRepository != model.weights.repo
                || precision == model.basePrecision && snapshot.derived?.normalization != model.nativeHalfNormalization
            {
                return true
            } else if model.conversionRecipe(for: model.basePrecision) == nil || precision == model.basePrecision,
                Set(snapshot.derived?.sourceFiles ?? []) != Set(try pinnedFiles(model.weights))
            {
                return true
            }
            if let recordedAssets = snapshot.derived?.assetSources {
                let current = Dictionary(uniqueKeysWithValues: (model.asset ?? []).filter { model.assets == nil || model.assets!.contains($0.id) }.map { ($0.id, $0.weights) })
                if recordedAssets != current { return true }
            }
            if precision != model.basePrecision, model.conversionRecipe(for: model.basePrecision) != nil {
                // Compare the cast's tiny provenance metadata too; quantized pins describe its converted files,
                // so they cannot directly be compared with original BF16/FP32 catalog pins.
                let base = try await store.snapshotURL(model: model.id, revision: model.weights.revision, precision: model.basePrecision)
                var castSource: SnapshotManifest?
                if let data = try? Data(contentsOf: StoredNames.snapshotURL(in: base)) {
                    castSource = try? JSONDecoder().decode(SnapshotManifest.self, from: data)
                }
                if castSource == nil {
                    castSource = try await store.sourceRetentionReceipt(model: model.id, revision: model.weights.revision, sourcePrecision: model.basePrecision)?.sourceManifest
                }
                guard let cast = castSource?.derived, cast.recipe == model.conversionRecipe(for: model.basePrecision), cast.normalization == model.nativeHalfNormalization,
                    cast.sourceRevision == model.weights.revision, cast.sourceRepository == model.weights.repo,
                    Set(cast.sourceFiles ?? []) == Set(try pinnedFiles(model.weights)),
                    Set(snapshot.derived?.sourceFiles ?? []) == Set(castSource?.files ?? [])
                else { return true }
            }
            for asset in model.separateAssets {
                let directory = try await store.snapshotURL(model: "asset-" + asset.id, revision: asset.revision, precision: "fp16")
                if let data = try? Data(contentsOf: StoredNames.snapshotURL(in: directory)),
                    let installed = try? JSONDecoder().decode(SnapshotManifest.self, from: data)
                {
                    if Set(installed.files) != Set(try pinnedFiles(asset.weights)) { return true }
                } else {
                    return true
                }
            }
            return false
        }
        let root = support.appendingPathComponent("weights/snapshots/\(model.id)")
        return ((try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []).contains {
            StoredNames.hasSnapshot($0.appendingPathComponent(precision))
        }
    }
    public func availabilityPlan(model: ModelManifest, precision: String) async throws -> GetAvailabilityPlan {
        guard model.precisions.contains(precision) else { throw ManifestError.invalid("precision is not offered") }
        // Incorporated assets are fetched only while the base snapshot is still to be converted.
        let baseInstalled = (try? await store.verifyInstalledSnapshot(model: model, precision: model.basePrecision)) != nil
        let assetSources = (baseInstalled ? model.separateAssets : model.selectedAssets).map(\.weights)
        let sources = [model.vendor?[precision] ?? model.weights] + assetSources
        var plan = GetAvailabilityPlan(); var seen: Set<String> = []
        for source in sources {
            for file in try pinnedFiles(source) where seen.insert(file.sha256).inserted {
                if try await store.isVerifiedPresent(sha256: file.sha256, bytes: file.bytes) {
                    plan.presentBytes += file.bytes
                } else if let donor = try localSources.candidates(repo: source.repo, revision: source.revision, path: file.path, repoType: source.repoType ?? "model").first(
                    where: {
                        !rejectedDonors.contains(donorKey($0, hash: file.sha256)) && (try? FileDigest.bytes($0.resolvingSymlinksInPath())) == file.bytes
                    })
                {
                    plan.localCopyBytes += file.bytes
                    // BlobStore clones within a volume and copies across volumes; decide per donor, after resolving symlinks.
                    if let volume = volumeOf(donor), volume == volumeOf(store.root) { plan.localCloneBytes += file.bytes }
                } else {
                    plan.networkBytes += file.bytes; plan.networkContent[file.sha256] = file.bytes
                }
            }
        }
        return plan
    }
    private func donorKey(_ url: URL, hash: String) -> String {
        let file = url.resolvingSymlinksInPath()
        return hash + file.path + String(describing: try? FileStamp.read(file))
    }
    private func report(_ file: DerivedFile, to sink: @escaping @Sendable (DownloadProgress) -> Void) -> @Sendable (DownloadProgress) -> Void {
        { event in
            var event = event; event.contentID = file.sha256.lowercased(); sink(event)
        }
    }
    func pinnedFiles(_ source: WeightSpec) throws -> [DerivedFile] {
        try source.files.map { path in
            guard let sha = source.sha256?[path], let bytes = source.bytes?[path] else { throw ManifestError.invalid("Get requires pinned hash and size for \(path)") }
            return DerivedFile(path: path, bytes: bytes, sha256: sha.lowercased())
        }
    }
}
