// SPDX-License-Identifier: MIT
import Foundation
import Darwin

public struct FileIdentity: Codable, Sendable, Equatable {
    public let device: Int64
    public let inode: UInt64
    public let bytes: Int64
    public let mtimeSeconds: Int64
    public let mtimeNanoseconds: Int64
    public static func read(_ url: URL) throws -> Self {
        var info = stat()
        guard lstat(url.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
            throw ManifestError.invalid("verified file is missing or not regular: \(url.lastPathComponent)")
        }
        return Self(
            device: Int64(info.st_dev), inode: UInt64(info.st_ino), bytes: info.st_size, mtimeSeconds: Int64(info.st_mtimespec.tv_sec),
            mtimeNanoseconds: Int64(info.st_mtimespec.tv_nsec))
    }
}
public struct VerifiedDownload: Sendable {
    public let url: URL
    public let sha256: String
    public let bytes: Int64
    public let identity: FileIdentity
    // Only the native verifier creates these; the store checks that the file did not change before promotion.
    init(url: URL, sha256: String, bytes: Int64) throws { self.url = url; self.sha256 = sha256.lowercased(); self.bytes = bytes; identity = try FileIdentity.read(url) }
}
struct BlobVerificationRecord: Codable, Sendable { let sha256: String; let identity: FileIdentity; var localSource: String?; var sourceKind: String? }
