// SPDX-License-Identifier: MIT
import Foundation

/// Provenance date of a declared (not measured) figure, never the wall clock. A local folder uses its last git commit;
/// without git history it uses model.toml's modification time. Signed catalog models use the catalog's source_date.
public enum DeclaredDate {
    public static func localFolder(_ folder: URL) -> String? {
        if let data = try? GitRead.run(repo: folder, args: ["log", "-1", "--format=%cI", "--", "."]),
            let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty
        {
            return text
        }
        guard let modified = (try? FileManager.default.attributesOfItem(atPath: folder.appendingPathComponent("model.toml").path))?[.modificationDate] as? Date
        else { return nil }
        return ISO8601DateFormatter().string(from: modified)
    }
}
