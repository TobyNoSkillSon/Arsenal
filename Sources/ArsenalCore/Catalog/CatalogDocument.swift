// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit

public struct CatalogFile: Codable, Sendable, Equatable {
    public var path: String; public var sha256: String; public var bytes: Int64
    public init(path: String, sha256: String, bytes: Int64) { self.path = path; self.sha256 = sha256; self.bytes = bytes }
}
public struct CatalogModel: Codable, Sendable, Equatable {
    public var manifest: ModelManifest
    public var files: [CatalogFile]
    public init(manifest: ModelManifest, files: [CatalogFile]) { self.manifest = manifest; self.files = files }
    public var id: String { manifest.id }
    public var runners: [RunnerSpec] { manifest.runners }
    public var weights: WeightSpec { manifest.weights }
}
public struct CatalogDocument: Codable, Sendable, Equatable {
    public var schema: Int
    public var sequence: Int64
    public var sourceCommit: String
    public var models: [CatalogModel]
    public var kindFiles: [CatalogFile]?
    public var ranking: [String: [String: [String]]]
    /// Git committer time of `sourceCommit` (ISO 8601), recorded by the catalog build. Declared figures use it as their date.
    public var sourceDate: String?
    enum CodingKeys: String, CodingKey {
        case schema, sequence, models, ranking; case kindFiles = "kind_files"; case sourceCommit = "source_commit"; case sourceDate = "source_date"
    }
    public init(
        sequence: Int64, sourceCommit: String, models: [CatalogModel], ranking: [String: [String: [String]]] = [:], kindFiles: [CatalogFile] = [],
        sourceDate: String? = nil
    ) {
        schema = 2; self.sequence = sequence; self.sourceCommit = sourceCommit; self.models = models; self.ranking = ranking; self.kindFiles = kindFiles
        self.sourceDate = sourceDate
    }
    public func encoded() throws -> Data { let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]; return try e.encode(self) }
    public func validate() throws {
        guard [1, 2].contains(schema), sequence > 0, ModelManifest.isHex(sourceCommit, count: 40), Set(models.map(\.id)).count == models.count else {
            throw ManifestError.invalid("invalid catalog schema, sequence, commit or duplicate model")
        }
        if let sourceDate { guard ISO8601DateFormatter().date(from: sourceDate) != nil else { throw ManifestError.invalid("invalid catalog source_date") } }
        let kinds = kindFiles ?? []
        guard Set(kinds.map(\.path)).count == kinds.count else { throw ManifestError.invalid("duplicate catalog kind file") }
        for file in kinds {
            guard SafePath.isRelative(file.path), ModelManifest.isHex(file.sha256, count: 64), file.bytes >= 0, file.bytes <= 64 * 1024 * 1024 else {
                throw ManifestError.invalid("unsafe catalog kind file \(file.path)")
            }
        }
        for model in models {
            for path in model.manifest.operations.flatMap({ [$0.input, $0.outputSchema].compactMap { $0 } }) where path.hasPrefix("kinds/") {
                guard kinds.contains(where: { $0.path == String(path.dropFirst(6)) }) else { throw ManifestError.invalid("unlisted shared schema \(path) for \(model.id)") }
            }
        }
        let assetNames = Set(models.flatMap { ($0.manifest.asset ?? []).map { "asset-" + $0.id } })
        guard Set(models.map(\.id)).isDisjoint(with: assetNames) else { throw ManifestError.invalid("model id collides with shared asset namespace") }
        for model in models {
            guard ModelManifest.validID(model.id), Set(model.files.map(\.path)).count == model.files.count, model.files.contains(where: { $0.path == "model.toml" }),
                !model.files.isEmpty
            else { throw ManifestError.invalid("invalid catalog model \(model.id)") }
            for file in model.files {
                guard SafePath.isRelative(file.path), ModelManifest.isHex(file.sha256, count: 64), file.bytes >= 0, file.bytes <= 64 * 1024 * 1024 else {
                    throw ManifestError.invalid("unsafe catalog file \(file.path)")
                }
            }
        }
    }
}
public enum CatalogTrust {
    /// Signature is detached raw Ed25519 bytes; public key is raw 32 bytes.
    public static func verify(data: Data, signature: Data, publicKey: Data, minimumSequence: Int64 = 0) throws -> CatalogDocument {
        let key = try Curve25519.Signing.PublicKey(rawRepresentation: publicKey)
        guard key.isValidSignature(signature, for: data) else { throw ManifestError.invalid("catalog signature rejected") }
        let catalog = try JSONDecoder().decode(CatalogDocument.self, from: data); try catalog.validate()
        guard catalog.sequence >= minimumSequence else { throw ManifestError.invalid("catalog downgrade rejected (\(catalog.sequence) < \(minimumSequence))") }; return catalog
    }
    /// Signs with the raw 32-byte Ed25519 key in `keyFile`. A missing file makes a new development key there, unless
    /// `createIfMissing` is false (an explicitly named key, such as the release key): then a missing file is an error.
    public static func sign(data: Data, keyFile: URL, createIfMissing: Bool = true) throws -> (signature: Data, publicKey: Data) {
        let key: Curve25519.Signing.PrivateKey
        if FileManager.default.fileExists(atPath: keyFile.path) {
            let raw = try Data(contentsOf: keyFile)
            guard raw.count == 32 else { throw ManifestError.invalid("signing key \(keyFile.lastPathComponent) is not a raw 32-byte Ed25519 key") }
            key = try Curve25519.Signing.PrivateKey(rawRepresentation: raw)
        } else if !createIfMissing {
            throw ManifestError.invalid("signing key not found: \(keyFile.path)")
        } else {
            try FileManager.default.createDirectory(at: keyFile.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            key = Curve25519.Signing.PrivateKey(); try key.rawRepresentation.write(to: keyFile, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: keyFile.path)
        }
        return (try key.signature(for: data), key.publicKey.rawRepresentation)
    }
}
