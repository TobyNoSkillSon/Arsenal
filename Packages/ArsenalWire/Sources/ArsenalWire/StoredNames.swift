// SPDX-License-Identifier: MIT
import Foundation

/// Names Arsenal writes into a model's weights directories. A store written before the app's rename holds
/// the same files under legacy names; they are read where they are and never renamed, so an older app reading
/// the same store still finds them. New files always get the current names.
public enum StoredNames {
    /// The snapshot manifest and derived provenance of `weights/snapshots/<model>/<revision>/<precision>/`.
    public static let snapshot = "arsenal-snapshot.json", derived = "arsenal-derived.json"
    /// Legacy names, read only: snapshots written before the rename.
    public static let legacySnapshot = "mobius-snapshot.json", legacyDerived = "mobius-derived.json" // legacy-compat: reads snapshots written before the rename
    /// Store bookkeeping files: never a model file, never copied into a converted tier.
    public static let reserved: Set<String> = [snapshot, derived, legacySnapshot, legacyDerived]
    public static let snapshotNames: Set<String> = [snapshot, legacySnapshot]

    /// The manifest of a snapshot directory: the current name, or the legacy one when only that exists.
    public static func snapshotURL(in directory: URL) -> URL {
        current(snapshot, legacy: legacySnapshot, in: directory)
    }
    /// The derived provenance written beside the manifest, under the same generation of names as the manifest.
    public static func derivedURL(in directory: URL) -> URL {
        snapshotURL(in: directory).lastPathComponent == legacySnapshot ? directory.appendingPathComponent(legacyDerived) : directory.appendingPathComponent(derived)
    }
    /// Whether the directory holds a snapshot manifest under either name.
    public static func hasSnapshot(_ directory: URL) -> Bool {
        snapshotNames.contains { FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
    /// `name` in `directory`, else `legacy` when only the legacy file exists (a file or folder written before the rename).
    public static func current(_ name: String, legacy: String, in directory: URL) -> URL {
        let url = directory.appendingPathComponent(name), old = directory.appendingPathComponent(legacy)
        return !FileManager.default.fileExists(atPath: url.path) && FileManager.default.fileExists(atPath: old.path) ? old : url
    }
    /// A JSON key written before the rename, read under its legacy name when the current one is absent.
    public static func value(_ key: String, legacy: String, in object: [String: Any]?) -> Any? { object?[key] ?? object?[legacy] }
}
