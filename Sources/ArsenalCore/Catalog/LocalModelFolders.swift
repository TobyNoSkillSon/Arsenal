// SPDX-License-Identifier: MIT
import Foundation
import Darwin

public struct LocalModelFolder: Sendable, Equatable {
    public let manifest: ModelManifest
    public let directory: URL
    /// Consent is bound to every byte in the folder, not just the runner name.
    public let fileDigests: [String: String]
    public let contentDigest: String
    public let embeddedKind: EmbeddedKind?
    /// Date of this folder's declared figures (see DeclaredDate); nil when neither git nor the file system gives one.
    public let declaredDate: String?
    public init(manifest: ModelManifest, directory: URL, fileDigests: [String: String], contentDigest: String) {
        self.manifest = manifest; self.directory = directory; self.fileDigests = fileDigests; self.contentDigest = contentDigest
        embeddedKind = try? EmbeddedKind.resolve(model: manifest, folder: directory)
        declaredDate = DeclaredDate.localFolder(directory)
    }
    public var requiresCodeConsent: Bool {
        manifest.runners.contains { $0.entry != nil || $0.language != "swift" }
            || FileManager.default.fileExists(atPath: directory.appendingPathComponent("kernels/manifest.toml").path)
    }
}
struct LocalFolderProblem: Sendable, Equatable { let id: String; let message: String }
/// `superseded`: local folders whose id the signed catalog now lists (an add-on promoted to the catalog). They are ignored,
/// never deleted, and make no error row: the signed model takes their place (2 Nov: 37 old add-on folders would otherwise
/// each show "may not shadow signed model" after an upgrade).
struct LocalFolderDiscovery: Sendable, Equatable { var folders: [LocalModelFolder] = []; var problems: [LocalFolderProblem] = []; var superseded: [String] = [] }

/// Actor-isolated file reads. Only identities are read on an unchanged scan; digests/manifests are reused.
actor LocalFolderScanner {
    private struct Record { let identity: FileStamp; let hash: String }
    private var hashes: [String: Record] = [:]
    private var folders: [String: LocalModelFolder] = [:]
    /// Placeholder results by folder content digest, so an unfinished folder is not reread on every scan.
    private var placeholders: [String: String?] = [:]
    private var cachedDiscovery: LocalFolderDiscovery?
    private var directoryStamps: [URL: FileStamp] = [:]
    private var cachedSignedIDs = Set<String>()
    private var cachedKinds: Set<String>?
    private var cachedKindsDirectory: URL?
    /// The signed kinds the cached folders were checked against. A catalog refresh that changes them (an embedded kind's
    /// id just became signed) rechecks every folder's manifest; file hashes stay cached.
    private var folderKinds: (kinds: Set<String>?, directory: URL?)?
    private(set) var hashReads = 0
    func scan(support: URL, signedIDs: Set<String>, kinds: Set<String>? = nil, kindsDirectory: URL? = nil) throws -> LocalFolderDiscovery {
        let root = try SafePath.resolve("local-models", inside: support)
        guard FileManager.default.fileExists(atPath: root.path) else { hashes = [:]; folders = [:]; cachedDiscovery = nil; return .init() }
        if let cachedDiscovery, cachedSignedIDs == signedIDs, cachedKinds == kinds, cachedKindsDirectory == kindsDirectory,
            directoryStamps.allSatisfy({ (try? FileStamp.read($0.key, directory: true)) == $0.value }),
            hashes.allSatisfy({ (try? FileStamp.read(URL(fileURLWithPath: $0.key))) == $0.value.identity })
        {
            return cachedDiscovery
        }
        cachedDiscovery = nil
        if let checked = folderKinds, checked.kinds != kinds || checked.directory != kindsDirectory { folders = [:] }
        folderKinds = (kinds, kindsDirectory)
        // Capture directories before scanning, including empty subtrees. A concurrent addition cannot be cached unseen.
        var directories: [URL: FileStamp] = [root: try FileStamp.read(root, directory: true)]
        if let entries = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey]) {
            for case let url as URL in entries where (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                if let stamp = try? FileStamp.read(url, directory: true) { directories[url] = stamp }
            }
        }
        let candidates = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).filter { !$0.lastPathComponent.hasPrefix(".") }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        var result = LocalFolderDiscovery(), seenFiles = Set<String>(), present = Set<String>()
        for folder in candidates {
            let id = folder.lastPathComponent
            do {
                _ = try SafePath.resolve(id, inside: root)
                if signedIDs.contains(id) { result.superseded.append(id); continue }
                let paths = try SafePath.files(in: folder)
                let inventory = try paths.map { path -> CatalogFile in
                    let file = try SafePath.resolve(path, inside: folder), before = try FileStamp.read(file)
                    seenFiles.insert(file.path)
                    let hash: String
                    if let cached = hashes[file.path], cached.identity == before {
                        hash = cached.hash
                    } else {
                        hash = try FileDigest.sha256(file); hashReads += 1
                        guard try FileStamp.read(file) == before else { throw ManifestError.invalid("local file changed during discovery") }
                        hashes[file.path] = Record(identity: before, hash: hash)
                    }
                    return CatalogFile(path: path, sha256: hash, bytes: before.bytes)
                }
                let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
                let digest = FileDigest.sha256(try encoder.encode(inventory))
                let local: LocalModelFolder
                if let cached = folders[id], cached.contentDigest == digest {
                    local = cached
                } else {
                    // An unfinished scaffold (a kit placeholder left in its text) never reaches approval or Get; checked before the
                    // manifest decodes, so the row names the placeholder rather than a value it made invalid.
                    let placeholder = placeholders[digest] ?? LocalFolderPlaceholders.first(in: folder, files: inventory.map { ($0.path, $0.bytes) })
                    placeholders[digest] = .some(placeholder)
                    if let placeholder { throw ManifestError.invalid(placeholder) }
                    local = LocalModelFolder(
                        manifest: try ModelManifest.load(folder, kinds: kinds, kindsDirectory: kindsDirectory), directory: folder,
                        fileDigests: Dictionary(uniqueKeysWithValues: inventory.map { ($0.path, $0.sha256) }), contentDigest: digest)
                }
                if let embedded = local.embeddedKind {
                    let peers = folders.values.filter { $0.manifest.id != id && candidates.contains($0.directory) }
                    guard !peers.contains(where: { $0.embeddedKind.map { $0.spec.id == embedded.spec.id && $0.descriptorHash != embedded.descriptorHash } == true }) else {
                        throw ManifestError.invalid("conflicting local kind tree \(embedded.spec.id)")
                    }
                }
                folders[id] = local; present.insert(id); result.folders.append(local)
            } catch { result.problems.append(.init(id: id, message: error.localizedDescription)) }
        }
        hashes = hashes.filter { seenFiles.contains($0.key) }; folders = folders.filter { present.contains($0.key) }
        if placeholders.count > 256 { placeholders = [:] }
        if result.problems.isEmpty,
            directories.allSatisfy({ (try? FileStamp.read($0.key, directory: true)) == $0.value }),
            hashes.allSatisfy({ (try? FileStamp.read(URL(fileURLWithPath: $0.key))) == $0.value.identity })
        {
            directoryStamps = directories; cachedDiscovery = result
            cachedSignedIDs = signedIDs; cachedKinds = kinds; cachedKindsDirectory = kindsDirectory
        }
        return result
    }
}
public enum LocalModelFolders {
    public static func discoverSnapshot(_ folder: URL) throws -> LocalModelFolder {
        let manifest = try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: SafePath.resolve("model.toml", inside: folder), encoding: .utf8))
        let files = try SafePath.files(in: folder).map { path in
            CatalogFile(path: path, sha256: try FileDigest.sha256(SafePath.resolve(path, inside: folder)), bytes: try FileDigest.bytes(SafePath.resolve(path, inside: folder)))
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return LocalModelFolder(
            manifest: manifest, directory: folder, fileDigests: Dictionary(uniqueKeysWithValues: files.map { ($0.path, $0.sha256) }),
            contentDigest: FileDigest.sha256(try encoder.encode(files)))
    }
    /// Strict utility for contributor tools; the service uses per-folder diagnostics from its cached scanner.
    public static func discover(support: URL, signedIDs: Set<String>, kinds: Set<String>? = nil, kindsDirectory: URL? = nil) throws -> [LocalModelFolder] {
        let root = try SafePath.resolve("local-models", inside: support)
        guard FileManager.default.fileExists(atPath: root.path) else { return [] }
        let folders = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).filter { !$0.lastPathComponent.hasPrefix(".") }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }.map { folder in
            _ = try SafePath.resolve(folder.lastPathComponent, inside: root)
            guard !signedIDs.contains(folder.lastPathComponent) else { throw ManifestError.invalid("local model may not shadow signed model \(folder.lastPathComponent)") }
            let inventory = try SafePath.files(in: folder).map { path in
                CatalogFile(path: path, sha256: try FileDigest.sha256(folder.appendingPathComponent(path)), bytes: try FileDigest.bytes(folder.appendingPathComponent(path)))
            }
            if let placeholder = LocalFolderPlaceholders.first(in: folder, files: inventory.map { ($0.path, $0.bytes) }) { throw ManifestError.invalid(placeholder) }
            let manifest = try ModelManifest.load(folder, kinds: kinds, kindsDirectory: kindsDirectory)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            return LocalModelFolder(
                manifest: manifest, directory: folder, fileDigests: Dictionary(uniqueKeysWithValues: inventory.map { ($0.path, $0.sha256) }),
                contentDigest: FileDigest.sha256(try encoder.encode(inventory)))
        }
        var descriptors: [String: String] = [:]
        for embedded in folders.compactMap(\.embeddedKind) {
            if let hash = descriptors[embedded.spec.id], hash != embedded.descriptorHash { throw ManifestError.invalid("conflicting local kind tree \(embedded.spec.id)") }
            descriptors[embedded.spec.id] = embedded.descriptorHash
        }
        return folders
    }
}
