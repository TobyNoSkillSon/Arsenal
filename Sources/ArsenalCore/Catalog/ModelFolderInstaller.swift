// SPDX-License-Identifier: MIT
import Foundation
import Darwin

/// Only accepts a signature-verified document. Consumers must use `accept` before installing folders.
public actor ModelFolderInstaller {
    public let support: URL
    private var accepted: CatalogDocument?
    private var kindsVerification = VerifiedInventoryCache()
    private let suppliedKinds: URL?
    public init(support: URL, kindsDirectory: URL? = nil) { self.support = support; suppliedKinds = kindsDirectory }
    @discardableResult public func accept(data: Data, signature: Data, publicKey: Data) throws -> CatalogDocument {
        let dir = support.appendingPathComponent("catalog"); let state = try SafePath.resolve("sequence.json", inside: dir)
        let minimum: Int64
        if FileManager.default.fileExists(atPath: state.path) { minimum = try JSONDecoder().decode(Int64.self, from: Data(contentsOf: state)) } else { minimum = 0 }
        let catalog = try CatalogTrust.verify(data: data, signature: signature, publicKey: publicKey, minimumSequence: minimum)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try data.write(to: dir.appendingPathComponent("catalog.json"), options: .atomic)
        try signature.write(to: dir.appendingPathComponent("catalog.json.sig"), options: .atomic)
        try JSONEncoder().encode(catalog.sequence).write(to: state, options: .atomic)
        accepted = catalog; return catalog
    }
    /// Install an already-verified staged kind inventory before advancing the accepted catalog sequence.
    public func acceptPrepared(data: Data, signature: Data, publicKey: Data, kindsDirectory: URL?) throws {
        let document = try CatalogTrust.verify(data: data, signature: signature, publicKey: publicKey, minimumSequence: accepted?.sequence ?? 0)
        if !(document.kindFiles ?? []).isEmpty, let kindsDirectory {
            try document.verifyKinds(directory: kindsDirectory)
            let parent = support.appendingPathComponent("catalog/kinds")
            let final = try SafePath.resolve(document.sourceCommit, inside: parent)
            if FileManager.default.fileExists(atPath: final.path) {
                try document.verifyKinds(directory: final)
            } else {
                let stage = parent.appendingPathComponent(".stage-" + UUID().uuidString)
                try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: kindsDirectory, to: stage)
                try document.verifyKinds(directory: stage)
                try FileManager.default.moveItem(at: stage, to: final)
            }
        }
        try accept(data: data, signature: signature, publicKey: publicKey)
    }
    public func verifiedKindsDirectory() throws -> URL? {
        guard let catalog = accepted, !(catalog.kindFiles ?? []).isEmpty else { return suppliedKinds }
        let cached = support.appendingPathComponent("catalog/kinds/" + catalog.sourceCommit)
        for directory in [suppliedKinds, cached].compactMap({ $0 }) where (try? kindsVerification.verify(directory: directory, files: catalog.kindFiles ?? [])) != nil {
            return directory
        }
        throw ManifestError.invalid("verified shared kinds directory is unavailable")
    }
    @discardableResult public func prepareKinds(source: any CatalogSource) async throws -> URL? {
        guard let catalog = accepted, let files = catalog.kindFiles, !files.isEmpty else { return suppliedKinds }
        // Opportunistic retention, never a timer; cleanup failure must not block using verified kinds.
        defer { try? Self.pruneKinds(support: support, current: catalog.sourceCommit) }
        if let root = try? verifiedKindsDirectory() { return root }
        let parent = support.appendingPathComponent("catalog/kinds")
        let final = try SafePath.resolve(catalog.sourceCommit, inside: parent)
        guard !FileManager.default.fileExists(atPath: final.path) else { throw ManifestError.invalid("cached shared kinds differ from signed inventory") }
        let stage = parent.appendingPathComponent(".stage-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
        for file in files {
            try Task.checkCancellation()
            let data = try await source.read(path: "kinds/" + file.path, commit: catalog.sourceCommit)
            guard data.count == file.bytes, FileDigest.sha256(data) == file.sha256.lowercased() else { throw ManifestError.invalid("kind checksum rejected: \(file.path)") }
            let target = try SafePath.resolve(file.path, inside: stage)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target, options: .atomic)
        }
        try catalog.verifyKinds(directory: stage)
        try FileManager.default.moveItem(at: stage, to: final)
        return final
    }
    static func pruneKinds(support: URL, current: String) throws {
        let parent = support.appendingPathComponent("catalog/kinds")
        let folders = try ((try? FileManager.default.contentsOfDirectory(at: parent, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []).filter {
            $0.lastPathComponent.count == 40 && $0.lastPathComponent.allSatisfy { "0123456789abcdef".contains($0) }
        }.map { folder in
            (
                try SafePath.resolve(folder.lastPathComponent, inside: parent),
                try folder.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? .distantPast
            )
        }
        .sorted { ($0.1, $0.0.path) > ($1.1, $1.0.path) }
        let previous = folders.first { $0.0.lastPathComponent != current }?.0
        let expired = folders.map(\.0).filter { $0.lastPathComponent != current && $0 != previous }
        if !expired.isEmpty { try OwnedCacheCleanup.remove(expired, support: support, operation: "shared_kinds_retention") }
    }
    private func recordURL(_ id: String) throws -> URL { try SafePath.resolve(id + ".json", inside: support.appendingPathComponent("state/installed-catalog")) }
    private func record(_ entry: CatalogModel) throws {
        let file = try recordURL(entry.id)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(entry).write(to: file, options: .atomic)
    }
    public func installedManifest(id: String) throws -> ModelManifest? {
        if let data = try? Data(contentsOf: recordURL(id)), let entry = try? JSONDecoder().decode(CatalogModel.self, from: data) { return entry.manifest }
        let folder = try SafePath.resolve(id, inside: support.appendingPathComponent("models"))
        guard FileManager.default.fileExists(atPath: folder.path) else { return nil }
        // Legacy installs have no inventory. This is only an update hint, never authority to execute code.
        return try ModelManifest.load(folder, kindsDirectory: verifiedKindsDirectory())
    }
    public func verifiedOrRefresh(id: String, source: any CatalogSource) async throws -> URL {
        _ = try await prepareKinds(source: source)
        do {
            let folder = try verifyInstalled(id: id)
            if let entry = accepted?.models.first(where: { $0.id == id }) { try record(entry) }
            return folder
        } catch {
            guard let entry = accepted?.models.first(where: { $0.id == id }) else { throw error }
            if let data = try? Data(contentsOf: recordURL(id)), let old = try? JSONDecoder().decode(CatalogModel.self, from: data), old.manifest == entry.manifest,
                old.files.sorted(by: { $0.path < $1.path }) == entry.files.sorted(by: { $0.path < $1.path })
            {
                throw error
            }
            return try await install(id: id, source: source, replacingExisting: true)
        }
    }
    public func install(id: String, source: any CatalogSource, replacingExisting: Bool = false) async throws -> URL {
        guard let catalog = accepted, let entry = catalog.models.first(where: { $0.id == id }) else { throw ManifestError.invalid("model is not in verified catalog") }
        _ = try await prepareKinds(source: source)
        let models = support.appendingPathComponent("models"); let final = try SafePath.resolve(id, inside: models)
        if FileManager.default.fileExists(atPath: final.path) {
            if (try? verify(folder: final, entry: entry)) != nil { try record(entry); return final }
            guard replacingExisting else { throw ManifestError.invalid("installed folder differs; unload and delete \(id) before updating") }
        }
        let staging = models.appendingPathComponent(".stage-\(UUID().uuidString)"); let folder = staging.appendingPathComponent(id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        // Batch the authenticated inventory once; each requested file still gets its own size/hash check.
        // Bundle sources otherwise rescan the entire folder for every individual read (quadratic).
        try Task.checkCancellation()
        let paths = entry.files.map { "models/\(id)/\($0.path)" }
        let contents = await source.readAvailable(paths: paths, commit: catalog.sourceCommit)
        // Keep failed staging and superseded code for recovery; no in-place edits or destructive repair.
        for file in entry.files {
            try Task.checkCancellation()
            let path = "models/\(id)/\(file.path)"
            let data: Data
            if let prefetched = contents[path] { data = prefetched } else { data = try await source.read(path: path, commit: catalog.sourceCommit) }
            guard data.count == file.bytes, FileDigest.sha256(data) == file.sha256.lowercased() else { throw ManifestError.invalid("catalog file checksum rejected: \(file.path)") }
            let target = try SafePath.resolve(file.path, inside: folder)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target, options: .atomic)
        }
        try verify(folder: folder, entry: entry)
        _ = try ModelManifest.load(folder, kindsDirectory: verifiedKindsDirectory())
        try Task.checkCancellation()
        if FileManager.default.fileExists(atPath: final.path) {
            guard renamex_np(folder.path, final.path, UInt32(RENAME_SWAP)) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        } else {
            try FileManager.default.moveItem(at: folder, to: final)
        }
        try record(entry)
        return final
    }
    @discardableResult public func verifyInstalled(id: String) throws -> URL {
        let path = try SafePath.resolve(id, inside: support.appendingPathComponent("models"))
        _ = try verify(installedFolder: path); return path
    }
    /// No source access, fetching, or activation. Core calls this immediately before resolving/launching installed code.
    @discardableResult public func verify(installedFolder: URL) throws -> ModelManifest {
        guard let catalog = accepted, let entry = catalog.models.first(where: { $0.id == installedFolder.lastPathComponent }) else {
            throw ManifestError.invalid("installed folder is not in verified catalog")
        }
        let expected = try SafePath.resolve(entry.id, inside: support.appendingPathComponent("models"))
        guard installedFolder.standardizedFileURL.path == expected.standardizedFileURL.path else {
            throw ManifestError.invalid("installed folder is outside the verified model store")
        }
        try verify(folder: installedFolder, entry: entry)
        return entry.manifest
    }
    private func verify(folder: URL, entry: CatalogModel) throws {
        // Finder metadata is inert. SafePath still rejects links/non-regular entries before this filter.
        let installed = try SafePath.files(in: folder).filter { URL(fileURLWithPath: $0).lastPathComponent != ".DS_Store" }
        let expected = entry.files.map(\.path).filter { URL(fileURLWithPath: $0).lastPathComponent != ".DS_Store" }
        let local = installed.filter { !expected.contains($0) && $0.hasPrefix(LocalPrecisionResults.prefix) && $0.hasSuffix(".md") }
        for path in local {
            let file = try SafePath.resolve(path, inside: folder)
            guard try FileDigest.bytes(file) <= 1_048_576 else { throw ManifestError.invalid("Local stat result exceeds 1 MiB") }
            _ = try LocalPrecisionResults.validate(Data(contentsOf: file), model: entry.manifest)
        }
        guard Set(installed).subtracting(local) == Set(expected) else { throw ManifestError.invalid("unlisted or missing installed files") }
        for file in entry.files { try FileDigest.verify(SafePath.resolve(file.path, inside: folder), sha256: file.sha256, bytes: file.bytes) }
        let manifest = try ModelManifest.load(folder, kindsDirectory: verifiedKindsDirectory())
        guard manifest == entry.manifest else { throw ManifestError.invalid("catalog manifest differs from folder") }
    }
    /// Offline reopening re-verifies both signature and previously installed immutable folders.
    public func restore(publicKey: Data) throws -> CatalogDocument {
        let dir = support.appendingPathComponent("catalog")
        return try accept(
            data: Data(contentsOf: dir.appendingPathComponent("catalog.json")), signature: Data(contentsOf: dir.appendingPathComponent("catalog.json.sig")), publicKey: publicKey)
    }
}
