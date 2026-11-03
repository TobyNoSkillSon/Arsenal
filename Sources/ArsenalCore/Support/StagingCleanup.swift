// SPDX-License-Identifier: MIT
import Foundation
import Darwin

/// Startup-only recovery of old, owned private staging; actual resumable downloads and external donors stay untouched.
enum StagingCleanup {
    /// Whether another live Arsenal process (the app, or a tool opened on the same support folder) owns it now.
    static func anotherProcessOwns(_ paths: ArsenalPaths) -> Bool {
        guard let data = try? Data(contentsOf: paths.support.appendingPathComponent("status.json")), let status = try? WireJSON.decode(data),
            let n = (status["pid"] ?? status["app_pid"])?.number, let pid = Int32(exactly: n)
        else { return false }
        return pid != getpid() && kill(pid, 0) == 0
    }
    static func run(paths: ArsenalPaths, olderThan: Date = Date().addingTimeInterval(-86_400)) throws -> Int {
        if anotherProcessOwns(paths) { return 0 }
        var removed = 0
        for relative in ["weights", "models", "weights/imports", "weights/downloads", "state/local-folders"] {
            guard let directory = try? SafePath.resolve(relative, inside: paths.support) else { continue }
            for candidate in (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [] {
                guard let isDirectory = ownedPattern(candidate.lastPathComponent, in: relative),
                    (try? SafePath.resolve(candidate.lastPathComponent, inside: directory)) != nil,
                    let original = ownedIdentity(candidate, directory: isDirectory), original.modified < olderThan
                else { continue }
                let files: [URL]
                if isDirectory {
                    guard let listing = try? SafePath.files(in: candidate) else { continue }
                    files = try listing.map { try SafePath.resolve($0, inside: candidate) }
                } else {
                    files = [candidate]
                }
                // Refuse foreign or special entries even inside a recognizable staging directory.
                guard files.allSatisfy({ ownedIdentity($0, directory: false) != nil }) else { continue }
                var listing: [JSONValue] = [], bytes: Int64 = 0
                for file in files { let size = try FileDigest.bytes(file); listing.append(["path": .string(file.path), "bytes": .number(Double(size))]); bytes += size }
                let receiptDir = paths.directory("logs").appendingPathComponent("cleanup-receipts")
                try FileManager.default.createDirectory(at: receiptDir, withIntermediateDirectories: true)
                let receipt = receiptDir.appendingPathComponent(UUID().uuidString + ".json")
                var object: JSONObject = [
                    "operation": "staging_cleanup", "path": .string(candidate.path), "files": .array(listing), "bytes": .number(Double(bytes)), "verified": false
                ]
                try WireJSON.encode(object).write(to: receipt, options: .atomic)
                let handle = try FileHandle(forWritingTo: receipt); try handle.synchronize(); try handle.close()
                // A staging item touched after enumeration belongs to an active operation, not this sweep.
                guard ownedIdentity(candidate, directory: isDirectory) == original else { continue }
                try FileManager.default.removeItem(at: candidate)
                object["verified"] = .bool(!FileManager.default.fileExists(atPath: candidate.path)); object["unlinkedBytes"] = .number(Double(bytes))
                try WireJSON.encode(object).write(to: receipt, options: .atomic); removed += 1
            }
        }
        return removed
    }
    /// Returns whether the exact private name denotes a directory (true) or a file (false).
    private static func ownedPattern(_ name: String, in directory: String) -> Bool? {
        switch directory {
        case "weights", "models", "state/local-folders":
            let prefixes = directory == "weights" ? [".derive-", ".cast-", ".snapshot-", ".asset-"] : [".stage-"]
            for prefix in prefixes where name.hasPrefix(prefix) {
                let stem = name.dropFirst(prefix.count)
                if UUID(uuidString: String(stem)) != nil { return true }
                // The lease beside a converter-read staging directory is an owned file, swept with its directory.
                if directory == "weights", stem.hasSuffix(".lease"), UUID(uuidString: String(stem.dropLast(6))) != nil { return false }
            }
        case "weights/imports":
            if name.hasSuffix(".part"), UUID(uuidString: String(name.dropLast(5))) != nil { return false }
        case "weights/downloads":
            for marker in [".part.old-", ".part.bad-", ".json.bad-"] {
                if name.count == 64 + marker.count + 36, ModelManifest.isHex(String(name.prefix(64)), count: 64), String(name.dropFirst(64).dropLast(36)) == marker,
                    UUID(uuidString: String(name.suffix(36))) != nil
                {
                    return false
                }
            }
        default: break
        }
        return nil
    }
    private struct Identity: Equatable { let device: Int64; let inode: UInt64; let bytes: Int64; let modified: Date; let nanos: Int64 }
    private static func ownedIdentity(_ url: URL, directory: Bool) -> Identity? {
        var info = stat()
        guard lstat(url.path, &info) == 0, info.st_uid == getuid(), (info.st_mode & S_IFMT) == (directory ? S_IFDIR : S_IFREG) else { return nil }
        return Identity(
            device: Int64(info.st_dev), inode: UInt64(info.st_ino), bytes: info.st_size, modified: Date(timeIntervalSince1970: Double(info.st_mtimespec.tv_sec)),
            nanos: Int64(info.st_mtimespec.tv_nsec))
    }
}
