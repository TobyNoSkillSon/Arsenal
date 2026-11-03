// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit

public enum SafePath {
    public static func isRelative(_ text: String) -> Bool {
        let pieces = text.split(separator: "/", omittingEmptySubsequences: false)
        return !text.isEmpty && !text.contains("\\") && !text.contains("\0") && !text.contains(":")
            && pieces.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\n") && !$0.contains("\r") }
    }
    public static func resolve(_ path: String, inside root: URL) throws -> URL {
        guard isRelative(path) else { throw ManifestError.invalid("unsafe path \(path)") }
        let result = root.appendingPathComponent(path)
        var cursor = result
        // Reject links all the way to the caller's root, including the root itself.
        while cursor.path == root.path || cursor.path.hasPrefix(root.path + "/") {
            if let attrs = try? FileManager.default.attributesOfItem(atPath: cursor.path), attrs[.type] as? FileAttributeType == .typeSymbolicLink {
                throw ManifestError.invalid("symlink rejected: \(path)")
            }
            if cursor == root { break }; cursor.deleteLastPathComponent()
        }
        return result
    }
    public static func files(in root: URL) throws -> [String] {
        _ = try resolve("model.toml", inside: root)
        guard let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isDirectoryKey], options: []) else {
            throw ManifestError.invalid("cannot list folder")
        }
        var files: [String] = []
        for case let url as URL in e {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isDirectoryKey])
            let rootPath = root.resolvingSymlinksInPath().standardizedFileURL.path
            let actualPath = url.standardizedFileURL.path
            let basePath = actualPath.hasPrefix(root.path + "/") ? root.path : rootPath
            guard actualPath.hasPrefix(basePath + "/") else { throw ManifestError.invalid("folder enumeration escaped root") }
            let path = String(actualPath.dropFirst(basePath.count + 1))
            guard isRelative(path), values.isSymbolicLink != true else { throw ManifestError.invalid("unsafe folder entry \(path)") }
            if values.isRegularFile == true { files.append(path) } else if values.isDirectory != true { throw ManifestError.invalid("non-regular folder entry \(path)") }
        }
        return files.sorted()
    }
}
public enum FileDigest {
    public static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    public static func sha256(_ file: URL, progress: (@Sendable (Int64) -> Void)? = nil) throws -> String {
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }; var hash = SHA256(); var received: Int64 = 0
        while try autoreleasepool(invoking: { () throws -> Bool in
            try Task<Never, Never>.checkCancellation()
            guard let bytes = try handle.read(upToCount: 4 * 1024 * 1024), !bytes.isEmpty else { return false }; hash.update(data: bytes); received += Int64(bytes.count)
            progress?(received); return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    public static func bytes(_ file: URL) throws -> Int64 { (try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? NSNumber)?.int64Value ?? 0 }
    public static func verify(_ file: URL, sha256: String, bytes: Int64) throws {
        guard try self.bytes(file) == bytes, try self.sha256(file) == sha256.lowercased() else {
            throw ManifestError.invalid("checksum or size mismatch: \(file.lastPathComponent)")
        }
    }
}
