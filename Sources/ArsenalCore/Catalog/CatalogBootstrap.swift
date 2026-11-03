// SPDX-License-Identifier: MIT
import Foundation

public struct VerifiedCatalogSelection: Sendable {
    public let data: Data
    public let signature: Data
    public let document: CatalogDocument
    public let directory: URL
}
public enum CatalogBootstrap {
    /// Rollback keeps the newest authenticated cached catalog; invalid candidates do not eclipse a valid one.
    public static func select(bundledDirectory: URL, cachedDirectory: URL, publicKey: Data) throws -> VerifiedCatalogSelection {
        var valid: [VerifiedCatalogSelection] = []
        for directory in [bundledDirectory, cachedDirectory] {
            if let data = try? Data(contentsOf: directory.appendingPathComponent("catalog.json")),
                let signature = try? Data(contentsOf: directory.appendingPathComponent("catalog.json.sig")),
                let doc = try? CatalogTrust.verify(data: data, signature: signature, publicKey: publicKey)
            {
                valid.append(VerifiedCatalogSelection(data: data, signature: signature, document: doc, directory: directory))
            }
        }
        guard let result = valid.max(by: { $0.document.sequence < $1.document.sequence }) else {
            throw ManifestError.invalid("No verified bundled or cached model catalog; reinstall Arsenal")
        }
        return result
    }
    public static func verifiedBundleSource(resources: URL) throws -> BundleCatalogSource {
        let data = try Data(contentsOf: SafePath.resolve("catalog/catalog.json", inside: resources))
        let signature = try Data(contentsOf: SafePath.resolve("catalog/catalog.json.sig", inside: resources))
        let key = try Data(contentsOf: SafePath.resolve("catalog-public-key", inside: resources))
        let document = try CatalogTrust.verify(data: data, signature: signature, publicKey: key)
        return try BundleCatalogSource(resources: resources, catalog: document)
    }
    /// Legacy v1 development descriptor. New releases use verifiedBundleSource.
    public static func bundledSource(resources: URL) throws -> LocalGitCatalogSource {
        struct Descriptor: Decodable { let local_git: String }
        let file = try SafePath.resolve("catalog/source.json", inside: resources)
        guard FileManager.default.fileExists(atPath: file.path) else { throw ManifestError.invalid("Bundled catalog source is missing; set catalog.source.path in config.json") }
        let descriptor = try JSONDecoder().decode(Descriptor.self, from: Data(contentsOf: file))
        guard descriptor.local_git.hasPrefix("/") else { throw ManifestError.invalid("Bundled catalog source must be an absolute local Git path") }
        let source = LocalGitCatalogSource(repo: URL(fileURLWithPath: descriptor.local_git)); try source.validateAvailability(); return source
    }
}
