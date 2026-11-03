// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import ArsenalWire

/// Worker-side metadata production. The service remains the authority for schema, MIME and artifact validation.
/// Keep the worker independent of the app's catalog, service and schema-runtime dependency closure.
public enum NativeFiles {
    /// One job output file: the service's per-job artifact limit (OperationArtifacts.maxArtifactBytes), not the inline cap.
    public static let maxBytes = 4 * 1024 * 1024 * 1024
    static func confined(_ path: String, root: URL) throws -> URL {
        let url = URL(fileURLWithPath: path).standardizedFileURL, resolved = url.resolvingSymlinksInPath()
        let original = root.standardizedFileURL.path + "/"
        guard path.hasPrefix("/"), url.path.hasPrefix(original) else { throw invalid() }
        let canonical = root.standardizedFileURL.resolvingSymlinksInPath().appendingPathComponent(String(url.path.dropFirst(original.count))).standardizedFileURL
        guard canonical.path == resolved.path,
            (try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])).map({ $0.isRegularFile == true && $0.isSymbolicLink != true }) == true
        else { throw invalid() }
        return url
    }
    static func bytes(_ url: URL) throws -> Int64 {
        guard let bytes = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { throw invalid() }
        return Int64(bytes)
    }
    static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        var hash = SHA256()
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty else { return false }
            hash.update(data: chunk); return true
        }) {}
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private static func invalid() -> WorkerFailure { .init(code: "invalid_output", message: "Output must be a regular file in its job directory without symlinks.") }
}
