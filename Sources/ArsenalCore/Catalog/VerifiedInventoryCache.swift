// SPDX-License-Identifier: MIT
import Foundation

/// Reuses successful hash checks only while every file and directory identity is unchanged.
/// Directory stamps include empty directories, so added/unlisted files cannot hide in an old subtree.
struct VerifiedInventoryCache {
    private struct Snapshot {
        let expected: [CatalogFile]
        let files: [String: FileStamp]
        let directories: [String: FileStamp]
        func unchanged(root: URL) -> Bool {
            directories.allSatisfy { (try? FileStamp.read($0.key.isEmpty ? root : root.appendingPathComponent($0.key), directory: true)) == $0.value }
                && files.allSatisfy { (try? FileStamp.read(root.appendingPathComponent($0.key))) == $0.value }
        }
    }
    private var snapshots: [String: Snapshot] = [:]
    mutating func verify(directory: URL, files: [CatalogFile]) throws {
        let expected = files.sorted { $0.path < $1.path }
        if let cached = snapshots[directory.path], cached.expected == expected, cached.unchanged(root: directory) { return }
        snapshots[directory.path] = nil
        let before = try inventory(directory, expected: expected)
        guard Set(before.files.keys.filter { URL(fileURLWithPath: $0).lastPathComponent != ".DS_Store" }) == Set(expected.map(\.path)) else {
            throw ManifestError.invalid("unlisted or missing kind files")
        }
        for file in expected { try FileDigest.verify(SafePath.resolve(file.path, inside: directory), sha256: file.sha256, bytes: file.bytes) }
        // Close the inventory→stamp window on first verification: a new file must not become cached
        // merely because it was added before its parent directory stamp was captured.
        let after = try SafePath.files(in: directory).filter { URL(fileURLWithPath: $0).lastPathComponent != ".DS_Store" }
        guard Set(after) == Set(expected.map(\.path)), before.unchanged(root: directory) else { throw ManifestError.invalid("kind files changed during verification") }
        snapshots[directory.path] = before
    }
    private func inventory(_ root: URL, expected: [CatalogFile]) throws -> Snapshot {
        let files = try SafePath.files(in: root)
        let stamps = try Dictionary(uniqueKeysWithValues: files.map { ($0, try FileStamp.read(root.appendingPathComponent($0))) })
        var directories: [String: FileStamp] = ["": try FileStamp.read(root, directory: true)]
        guard let entries = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey]) else { throw ManifestError.invalid("cannot list kind folders") }
        for case let url as URL in entries where try url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true {
            let actual = url.standardizedFileURL.path
            let base = actual.hasPrefix(root.path + "/") ? root.path : root.resolvingSymlinksInPath().standardizedFileURL.path
            let path = String(actual.dropFirst(base.count + 1))
            directories[path] = try FileStamp.read(url, directory: true)
        }
        // Finder metadata remains inert, but its regular-file identity is still checked.
        return Snapshot(expected: expected, files: stamps, directories: directories)
    }
}
