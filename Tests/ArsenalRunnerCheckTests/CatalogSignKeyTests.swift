// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalCommands

/// `catalog sign --key <path> [--rotate]`: an explicit key is never created, and replacing the committed public key
/// needs `--rotate`.
struct CatalogSignKeyTests {
    private func fixture() throws -> (root: URL, repo: URL, committedKey: Curve25519.Signing.PrivateKey) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("catalog-sign-\(UUID().uuidString)")
        let repo = root.appendingPathComponent("repo")
        let catalog = repo.appendingPathComponent("Resources/catalog")
        try FileManager.default.createDirectory(at: catalog, withIntermediateDirectories: true)
        let doc = CatalogDocument(sequence: 3, sourceCommit: String(repeating: "a", count: 40), models: [], kindFiles: [], sourceDate: "2026-10-28T00:00:00Z")
        try doc.encoded().write(to: catalog.appendingPathComponent("catalog.json"))
        let committed = Curve25519.Signing.PrivateKey()
        try committed.publicKey.rawRepresentation.write(to: repo.appendingPathComponent("Resources/catalog-public-key"))
        return (root, repo, committed)
    }

    @Test func anExplicitMissingKeyIsAnErrorAndNoKeyIsCreated() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let missing = f.root.appendingPathComponent("keys/release.key")
        #expect(throws: (any Error).self) { try CatalogCommand.signCatalog(repo: f.repo, arguments: ["--key", missing.path], environment: [:]) }
        #expect(throws: (any Error).self) {
            try CatalogCommand.signCatalog(repo: f.repo, arguments: [], environment: ["ARSENAL_CATALOG_SIGNING_KEY": missing.path])
        }
        #expect(!FileManager.default.fileExists(atPath: missing.path) && !FileManager.default.fileExists(atPath: missing.deletingLastPathComponent().path))
        #expect(!FileManager.default.fileExists(atPath: f.repo.appendingPathComponent("Resources/catalog/catalog.json.sig").path))
    }

    @Test func aDifferentKeyNeedsRotateAndThenReplacesThePublicKey() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let release = Curve25519.Signing.PrivateKey(), keyFile = f.root.appendingPathComponent("release.key")
        try release.rawRepresentation.write(to: keyFile)
        let publicKey = f.repo.appendingPathComponent("Resources/catalog-public-key")
        #expect(throws: (any Error).self) { try CatalogCommand.signCatalog(repo: f.repo, arguments: ["--key", keyFile.path], environment: [:]) }
        #expect(try Data(contentsOf: publicKey) == f.committedKey.publicKey.rawRepresentation)
        let line = try CatalogCommand.signCatalog(repo: f.repo, arguments: ["--key", keyFile.path, "--rotate"], environment: [:])
        #expect(line == "Signed catalog sequence 3; Resources/catalog-public-key replaced")
        #expect(try Data(contentsOf: publicKey) == release.publicKey.rawRepresentation)
        let catalog = f.repo.appendingPathComponent("Resources/catalog")
        let verified = try CatalogTrust.verify(
            data: Data(contentsOf: catalog.appendingPathComponent("catalog.json")), signature: Data(contentsOf: catalog.appendingPathComponent("catalog.json.sig")),
            publicKey: Data(contentsOf: publicKey))
        #expect(verified.sequence == 3)
        // The same key again signs without --rotate; the output does not mention a replacement.
        #expect(try CatalogCommand.signCatalog(repo: f.repo, arguments: [], environment: ["ARSENAL_CATALOG_SIGNING_KEY": keyFile.path]) == "Signed catalog sequence 3")
    }

    @Test func unknownOptionsAndAMalformedKeyAreRefused() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        #expect(throws: (any Error).self) { try CatalogCommand.signCatalog(repo: f.repo, arguments: ["--force"], environment: [:]) }
        #expect(throws: (any Error).self) { try CatalogCommand.signCatalog(repo: f.repo, arguments: ["--key"], environment: [:]) }
        let short = f.root.appendingPathComponent("short.key"); try Data(repeating: 1, count: 16).write(to: short)
        #expect(throws: (any Error).self) { try CatalogCommand.signCatalog(repo: f.repo, arguments: ["--key", short.path, "--rotate"], environment: [:]) }
    }
}
