// SPDX-License-Identifier: MIT
import Foundation

/// Display-only cache key. Load/Get retain their independent integrity checks.
struct InstalledRowCacheKey: Equatable {
    struct File: Equatable {
        let exists: Bool
        let stamp: FileStamp?
        init(_ url: URL) {
            exists = FileManager.default.fileExists(atPath: url.path)
            stamp = exists ? try? FileStamp.read(url) : nil
        }
    }
    let snapshots: [File]
    let nativeConfig: File?
    let localResults: [File]
    static func localResults(folder: URL) -> [File] {
        guard let directory = try? SafePath.resolve("results/local", inside: folder), let files = try? SafePath.files(in: directory) else { return [] }
        return files.sorted().compactMap { path in (try? SafePath.resolve(path, inside: directory)).map(File.init) }
    }
    let retention: File
    let dtype: String
    let fast: Bool
    let runner: String
    let localDigest: String?
    let problem: String?
}

struct CatalogEntryCacheKey: Equatable {
    let row: ModelRow
    let added: Bool
    let settings: JSONObject?
}
