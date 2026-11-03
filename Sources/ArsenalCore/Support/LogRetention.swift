// SPDX-License-Identifier: MIT
import Foundation

/// Logs stay bounded without a timer: every new worker log first trims `logs/` to the newest `keep` files and at
/// most `maxBytes` in total (oldest first). One file per process, so the newest ones cover the resident workers.
public enum LogRetention {
    public static let keep = 30
    public static let maxBytes: Int64 = 50 * 1_048_576

    public static func prune(_ directory: URL, keep: Int = LogRetention.keep, maxBytes: Int64 = LogRetention.maxBytes) {
        let keys: [URLResourceKey] = [.contentModificationDateKey, .fileSizeKey, .isRegularFileKey]
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: keys) else { return }
        let logs = files.compactMap { url -> (URL, Date, Int64)? in
            guard url.pathExtension == "log", let v = try? url.resourceValues(forKeys: Set(keys)), v.isRegularFile == true else { return nil }
            return (url, v.contentModificationDate ?? .distantPast, Int64(v.fileSize ?? 0))
        }.sorted { $0.1 > $1.1 }
        var total: Int64 = 0
        // Leave room for the log about to be created.
        for (index, log) in logs.enumerated() {
            total += log.2
            if index >= keep - 1 || total > maxBytes { try? FileManager.default.removeItem(at: log.0) }
        }
    }
}
