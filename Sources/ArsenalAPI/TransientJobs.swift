// SPDX-License-Identifier: MIT
import Foundation
import OSLog

private let cleanupLog = Logger(subsystem: "dev.arsenal.app", category: "api-transient-cleanup")

extension APIRouter {
    // Only call for an API-owned, non-durable job after its request stops using
    // the files. Durable prediction records and successful URL outputs stay.
    func removeTransientJob(_ directory: URL) {
        let root = jobsRoot.resolvingSymlinksInPath().standardizedFileURL
        let resolved = directory.resolvingSymlinksInPath().standardizedFileURL
        guard resolved.deletingLastPathComponent() == root,
            UUID(uuidString: resolved.lastPathComponent) != nil,
            FileManager.default.fileExists(atPath: resolved.path)
        else { return }
        do {
            cleanupLog.notice("Deletion receipt directory: \(resolved.path, privacy: .public)")
            if let entries = FileManager.default.enumerator(at: resolved, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey]) {
                for case let file as URL in entries {
                    let values = try file.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey])
                    if values.isDirectory != true {
                        // One line per file avoids truncating a long receipt.
                        cleanupLog.notice("Deletion receipt file: \(file.path, privacy: .public), bytes=\(values.fileSize ?? 0)")
                    }
                }
            }
            // Unified logging is OS-bounded, unlike another append-only file.
            // Emit the complete per-file receipt before touching any output.
            try FileManager.default.removeItem(at: resolved)
            let gone = !FileManager.default.fileExists(atPath: resolved.path)
            cleanupLog.notice("Deletion verified: \(resolved.path, privacy: .public), gone=\(gone)")
        } catch {
            cleanupLog.error("Transient job cleanup failed: \(resolved.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }
}
