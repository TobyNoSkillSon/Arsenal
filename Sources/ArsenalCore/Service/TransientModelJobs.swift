// SPDX-License-Identifier: MIT
import Foundation

/// Opportunistic expiry, never an idle timer. Provider routes may unlink their delivered job immediately.
enum TransientModelJobs {
    static func prepare(_ directory: URL, support: URL) throws {
        try? prune(support: support)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    static func prune(support: URL, before: Date = Date().addingTimeInterval(-3600)) throws {
        let root = support.appendingPathComponent("jobs")
        for directory in (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [] {
            guard UUID(uuidString: directory.lastPathComponent) != nil,
                (try? SafePath.resolve(directory.lastPathComponent, inside: root)) != nil,
                let modified = try? directory.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate, modified < before
            else { continue }
            // A malformed/vanished expired job must not block fresh work or other entries' cleanup.
            try? OwnedCacheCleanup.remove([directory], support: support, operation: "transient_job_expiry")
        }
    }
}
