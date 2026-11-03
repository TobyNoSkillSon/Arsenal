// SPDX-License-Identifier: MIT
import Foundation
import Darwin

public struct LocalModelSources: Sendable {
    public let directories: [URL]
    public static var defaultPaths: [String] { ["~/.cache/huggingface/hub", "~/.omlx/models"] }
    public init(paths: [String]? = nil) {
        directories = (paths ?? Self.defaultPaths).map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath).standardizedFileURL }
    }
    public func candidates(repo: String, revision: String, path: String, repoType: String = "model") throws -> [URL] {
        guard SafePath.isRelative(repo), repo.split(separator: "/").count == 2, ModelManifest.isHex(revision, count: 40), SafePath.isRelative(path) else {
            throw ManifestError.invalid("invalid local source identity")
        }
        let slug = (repoType == "dataset" ? "datasets--" : "models--") + repo.replacingOccurrences(of: "/", with: "--")
        let repoParts = repo.split(separator: "/").map(String.init)
        return directories.flatMap { directory in
            var result = [directory.appendingPathComponent("\(slug)/snapshots/\(revision)/\(path)"), directory.appendingPathComponent("\(repo)/\(path)")]
            if Array(directory.pathComponents.suffix(2)) == repoParts { result.append(directory.appendingPathComponent(path)) }
            return result.filter { FileManager.default.fileExists(atPath: $0.path) && Self.resolves($0, inside: directory) }
        }
    }
    /// A cache path that resolves outside its cache belongs to another app (another app once linked its own Drama 3
    /// conversion into the Fish Audio snapshot, and a later download wrote over it). Arsenal never
    /// imports from, or writes through, such a link: the files are not the pinned ones, and they are not Arsenal's.
    public static func resolves(_ url: URL, inside directory: URL) -> Bool {
        guard let file = realPath(url), let root = realPath(directory) else { return false }
        return file.hasPrefix(root.hasSuffix("/") ? root : root + "/")
    }
    static func realPath(_ url: URL) -> String? {
        guard let resolved = Darwin.realpath(url.path, nil) else { return nil }
        defer { free(resolved) }
        return String(cString: resolved)
    }
}
/// SHA callbacks are frequent, row callbacks are not: keep hashing off the UI without creating a task per chunk.
final class LocalVerificationProgress: @unchecked Sendable {
    private let file: DerivedFile
    private let sink: @Sendable (DownloadProgress) -> Void
    private let lock = NSLock()
    private var last = Date.distantPast
    private let started = Date()
    init(file: DerivedFile, sink: @escaping @Sendable (DownloadProgress) -> Void) { self.file = file; self.sink = sink }
    func report(_ bytes: Int64) {
        let now = Date()
        let send = lock.withLock {
            if now.timeIntervalSince(last) < 0.2 && bytes != file.bytes { return false }; last = now; return true
        }
        if send {
            sink(
                DownloadProgress(
                    file: file.path, completedBytes: bytes, totalBytes: file.bytes, bytesPerSecond: Double(bytes) / max(0.001, now.timeIntervalSince(started)),
                    contentID: file.sha256, phase: "verifyingLocalCopy"))
        }
    }
}

/// Local-copy bytes are size-matched candidates, not a hash promise; Get validates them before deciding to download.
public struct GetAvailabilityPlan: Codable, Sendable, Equatable {
    public var networkContent: [String: Int64] = [:]
    public var networkBytes: Int64 = 0
    public var localCopyBytes: Int64 = 0
    /// The part of `localCopyBytes` whose donor sits on the store's volume, so Get clones it (APFS) without using disk.
    public var localCloneBytes: Int64 = 0
    public var presentBytes: Int64 = 0
    public init() {}
    enum CodingKeys: String, CodingKey { case networkContent, networkBytes, localCopyBytes, localCloneBytes, presentBytes }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        networkContent = try c.decodeIfPresent([String: Int64].self, forKey: .networkContent) ?? [:]
        networkBytes = try c.decodeIfPresent(Int64.self, forKey: .networkBytes) ?? 0
        localCopyBytes = try c.decodeIfPresent(Int64.self, forKey: .localCopyBytes) ?? 0
        localCloneBytes = try c.decodeIfPresent(Int64.self, forKey: .localCloneBytes) ?? 0
        presentBytes = try c.decodeIfPresent(Int64.self, forKey: .presentBytes) ?? 0
    }
}
