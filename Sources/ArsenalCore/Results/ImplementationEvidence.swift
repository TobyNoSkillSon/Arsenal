// SPDX-License-Identifier: MIT
import Foundation

public enum ImplementationEvidence {
    public static func fingerprint(_ local: LocalModelFolder) throws -> String {
        var manifest = try WireJSON.decode(JSONEncoder().encode(local.manifest))
        for key in [
            "name", "description", "links", "summary", "publisher", "released", "suggested", "popularity", "licence", "languages", "tasks", "params", "tools", "activity", "menu",
            "gate", "bench"
        ] {
            manifest[key] = nil
        }
        var data = try WireJSON.encode(manifest)
        for path in local.fileDigests.keys.sorted()
        // The settings panel (settings.toml) arranges the menu only: a layout change never touches what runs, so it does
        // not stale the runner's evidence. It stays in the folder's content digest (consent, revisions).
        where path != "model.toml" && path != "README.md" && path != SettingsPanel.fileName && !path.hasPrefix("log/")
            && !path.hasPrefix("results/") && !path.hasPrefix("gate/") && !path.hasPrefix("bench/")
        {
            data.append(Data(path.utf8)); data.append(Data(local.fileDigests[path]!.utf8))
        }
        return FileDigest.sha256(data)
    }
    public static func fixtures(_ local: LocalModelFolder) -> String {
        var data = Data()
        for path in local.fileDigests.keys.sorted() where path.hasPrefix("gate/") {
            data.append(Data(path.utf8)); data.append(Data(local.fileDigests[path]!.utf8))
        }
        return FileDigest.sha256(data)
    }
    public static func compatible(_ checks: ModelChecksStatus, local: LocalModelFolder) -> Bool {
        checks.implementationFingerprint == (try? fingerprint(local)) && checks.fixtureIdentity == fixtures(local)
            && !checks.evidence.isEmpty
            && checks.evidence.allSatisfy { path in
                path.hasPrefix("results/") && local.fileDigests[path] != nil && checks.evidenceHashes[path] == local.fileDigests[path]
                    && (try? SafePath.resolve(path, inside: local.directory)) != nil
            }
    }
}
