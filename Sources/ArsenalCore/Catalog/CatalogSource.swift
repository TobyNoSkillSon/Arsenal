// SPDX-License-Identifier: MIT
import Foundation

public protocol CatalogSource: Sendable {
    func read(path: String, commit: String) async throws -> Data
    /// Optional launch evidence: unavailable or invalid files are omitted, never trusted.
    func readAvailable(paths: [String], commit: String) async -> [String: Data]
}
extension CatalogSource {
    public func readAvailable(paths: [String], commit: String) async -> [String: Data] {
        var files: [String: Data] = [:]
        for path in paths { files[path] = try? await read(path: path, commit: commit) }
        return files
    }
}
public struct LocalGitCatalogSource: CatalogSource {
    /// Development checkout default; bundle Resources is data, never an inferred Git repository.
    public static let defaultRepository = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    public let repo: URL
    public init(repo: URL = Self.defaultRepository) { self.repo = repo }
    public func validateAvailability() throws {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: repo.path, isDirectory: &directory), directory.boolValue,
            FileManager.default.fileExists(atPath: repo.appendingPathComponent(".git").path)
        else { throw ManifestError.invalid("Arsenal catalog repository is missing at \(repo.path); set catalog.source.path in config.json") }
    }
    public func read(path: String, commit: String) async throws -> Data {
        guard SafePath.isRelative(path), ModelManifest.isHex(commit, count: 40) else { throw ManifestError.invalid("invalid catalog path or commit") }
        // Local development Git waits are blocking too; they must not hold Swift's cooperative pool.
        return try await processIO {
            try validateAvailability()
            // Signed catalogs can name an older source commit from before Resources consolidation.
            let candidates = path.hasPrefix("kinds/") ? ["Resources/" + path, path] : [path]
            for candidate in candidates {
                let mode = try GitRead.run(repo: repo, args: ["ls-tree", commit, "--", candidate])
                if let listing = String(data: mode, encoding: .utf8), listing.hasPrefix("100644 blob ") || listing.hasPrefix("100755 blob ") {
                    return try GitRead.run(repo: repo, args: ["show", "\(commit):\(candidate)"])
                }
            }
            throw ManifestError.invalid("catalog path is missing, unlisted or a symlink: \(path)")
        }
    }
}
public struct HTTPSCatalogSource: CatalogSource {
    public let rawBase: URL
    public init(rawBase: URL) { self.rawBase = rawBase }
    public func read(path: String, commit: String) async throws -> Data {
        guard SafePath.isRelative(path), ModelManifest.isHex(commit, count: 40),
            rawBase.scheme == "https" || (["localhost", "127.0.0.1", "::1"].contains(rawBase.host ?? "") && rawBase.scheme == "http")
        else { throw ManifestError.invalid("invalid HTTPS catalog source") }
        let candidates = path.hasPrefix("kinds/") ? ["Resources/" + path, path] : [path]
        for (index, candidate) in candidates.enumerated() {
            var url = rawBase.appendingPathComponent(commit)
            for segment in candidate.split(separator: "/") { url.appendPathComponent(String(segment)) }
            var request = URLRequest(url: url); request.timeoutInterval = 30
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { throw ManifestError.invalid("catalog file request failed") }
            if http.statusCode == 404, index + 1 < candidates.count { continue }
            guard http.statusCode == 200, data.count <= 64 * 1024 * 1024 else { throw ManifestError.invalid("catalog file request failed") }
            return data
        }
        throw ManifestError.invalid("catalog file request failed")
    }
}
public enum GitRead {
    /// Git has no network or hooks here; output goes to a file so pipe capacity cannot deadlock a large folder file.
    public static func run(repo: URL, args: [String]) throws -> Data {
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-fetch-git-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: temp.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: temp) }
        let handle = try FileHandle(forWritingTo: temp); defer { try? handle.close() }
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/git"); p.arguments = ["-C", repo.path] + args
        p.standardOutput = handle; p.standardError = FileHandle.nullDevice
        try p.run()
        let deadline = Date().addingTimeInterval(30)
        while p.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
        if p.isRunning { p.terminate(); throw ManifestError.invalid("git read timed out") }
        guard p.terminationStatus == 0 else { throw ManifestError.invalid("git read failed: \(args.first ?? "read")") }
        return try Data(contentsOf: temp)
    }
}

/// Reads only the inventory authenticated by the bundled catalog. A copied app is independent of its build repo.
public struct BundleCatalogSource: CatalogSource {
    public let resources: URL
    public let catalog: CatalogDocument
    public var kindsDirectory: URL { resources.appendingPathComponent("kinds") }
    public init(resources: URL, catalog: CatalogDocument) throws {
        try catalog.validate(); self.resources = resources; self.catalog = catalog; try catalog.verifyKinds(directory: kindsDirectory)
    }
    public func read(path: String, commit: String) async throws -> Data {
        guard commit == catalog.sourceCommit, SafePath.isRelative(path) else { throw ManifestError.invalid("bundle catalog commit/path mismatch") }
        if path.hasPrefix("kinds/") {
            let relative = String(path.dropFirst(6))
            guard let file = catalog.kindFiles?.first(where: { $0.path == relative }) else { throw ManifestError.invalid("unlisted bundled kind file") }
            try catalog.verifyKinds(directory: kindsDirectory)
            let data = try Data(contentsOf: SafePath.resolve(relative, inside: kindsDirectory))
            guard data.count == file.bytes, FileDigest.sha256(data) == file.sha256.lowercased() else { throw ManifestError.invalid("bundle kind checksum rejected") }
            return data
        }
        let parts = path.split(separator: "/", maxSplits: 2).map(String.init)
        guard parts.count == 3, parts[0] == "models", let model = catalog.models.first(where: { $0.id == parts[1] }), let file = model.files.first(where: { $0.path == parts[2] })
        else { throw ManifestError.invalid("unlisted bundle model file") }
        let folder = try verifiedFolder(model)
        return try readVerifiedFile(file, folder: folder)
    }
    public func readAvailable(paths: [String], commit: String) async -> [String: Data] {
        guard commit == catalog.sourceCommit else { return [:] }
        // Verify each complete folder inventory once, not once per log/results/kernel file.
        // This is a request-scoped reuse: a later read still checks the inventory and each requested hash.
        var files: [String: Data] = [:]
        for model in catalog.models {
            let prefix = "models/" + model.id + "/"
            let requested = paths.filter { $0.hasPrefix(prefix) && SafePath.isRelative($0) }
            guard !requested.isEmpty, let folder = try? verifiedFolder(model) else { continue }
            for path in requested {
                guard let file = model.files.first(where: { $0.path == String(path.dropFirst(prefix.count)) }) else { continue }
                files[path] = try? readVerifiedFile(file, folder: folder)
            }
        }
        for path in paths where path.hasPrefix("kinds/") { files[path] = try? await read(path: path, commit: commit) }
        return files
    }
    private func verifiedFolder(_ model: CatalogModel) throws -> URL {
        let folder = try SafePath.resolve("models/" + model.id, inside: resources)
        let present = try SafePath.files(in: folder).filter { URL(fileURLWithPath: $0).lastPathComponent != ".DS_Store" }
        guard Set(present) == Set(model.files.map(\.path).filter { URL(fileURLWithPath: $0).lastPathComponent != ".DS_Store" }) else {
            throw ManifestError.invalid("unlisted or missing bundled model files")
        }
        return folder
    }
    private func readVerifiedFile(_ file: CatalogFile, folder: URL) throws -> Data {
        let url = try SafePath.resolve(file.path, inside: folder)
        let data = try Data(contentsOf: url)
        guard data.count == file.bytes, FileDigest.sha256(data) == file.sha256.lowercased() else { throw ManifestError.invalid("bundle file checksum rejected: \(file.path)") }
        return data
    }
}
