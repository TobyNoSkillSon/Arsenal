// SPDX-License-Identifier: MIT
import Darwin
import Foundation

/// Nanosecond ctime detects same-size rewrites even if an editor restores mtime.
struct FileStamp: Sendable, Equatable {
    let device: Int32
    let inode: UInt64
    let bytes: Int64
    let modifiedSeconds: Int
    let modifiedNanos: Int
    let changedSeconds: Int
    let changedNanos: Int
    static func read(_ file: URL, directory: Bool = false) throws -> Self {
        var value = stat()
        guard lstat(file.path, &value) == 0, (value.st_mode & S_IFMT) == (directory ? S_IFDIR : S_IFREG) else {
            throw ManifestError.invalid("verified path is missing or not \(directory ? "a directory" : "regular"): \(file.lastPathComponent)")
        }
        return Self(
            device: value.st_dev, inode: value.st_ino, bytes: value.st_size, modifiedSeconds: value.st_mtimespec.tv_sec, modifiedNanos: value.st_mtimespec.tv_nsec,
            changedSeconds: value.st_ctimespec.tv_sec, changedNanos: value.st_ctimespec.tv_nsec)
    }
}
