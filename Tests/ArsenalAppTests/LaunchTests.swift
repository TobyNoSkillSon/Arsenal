// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// The app's service under a `ARSENAL_CATALOG` override: the bundle's signed catalog stays the fallback (O-1).
@MainActor @Suite(.menuState, .serialized) final class LaunchTests {
    private let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-app-launch-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: root) }
    private static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    /// An app bundle's Resources as the packager lays them out: this checkout's folders and kinds under a dev-signed
    /// catalog (the committed signature lags shared-schema edits until the release re-sign).
    private func bundle() throws -> URL {
        let resources = root.appendingPathComponent("bundle-fixture/Contents/Resources")
        let fm = FileManager.default
        try fm.createDirectory(at: resources, withIntermediateDirectories: true)
        let kinds = resources.appendingPathComponent("kinds"), models = resources.appendingPathComponent("models")
        try fm.copyItem(at: Self.repo.appendingPathComponent("Resources/kinds"), to: kinds)
        try fm.createDirectory(at: models, withIntermediateDirectories: true)
        func inventory(_ dir: URL) throws -> [CatalogFile] {
            try SafePath.files(in: dir).map {
                CatalogFile(path: $0, sha256: try FileDigest.sha256(dir.appendingPathComponent($0)), bytes: try FileDigest.bytes(dir.appendingPathComponent($0)))
            }
        }
        var entries: [CatalogModel] = []
        for id in ["kokoro-82m"] {
            let folder = models.appendingPathComponent(id)
            try fm.copyItem(at: Self.repo.appendingPathComponent("models/\(id)"), to: folder)
            entries.append(CatalogModel(manifest: try ModelManifest.load(folder, kindsDirectory: kinds), files: try inventory(folder)))
        }
        let data = try CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: entries, kindFiles: try inventory(kinds)).encoded()
        let key = Curve25519.Signing.PrivateKey()
        try fm.createDirectory(at: resources.appendingPathComponent("catalog"), withIntermediateDirectories: true)
        try data.write(to: resources.appendingPathComponent("catalog/catalog.json"))
        try key.signature(for: data).write(to: resources.appendingPathComponent("catalog/catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
        return resources
    }

    @Test func aMissingOverrideFallsBackToTheBundleWithAFooterLine() async throws {
        let resources = try bundle()
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        let (service, error) = await ServiceFactory.make(paths: paths, environment: ["ARSENAL_CATALOG": root.appendingPathComponent("no-such-repo").path], bundle: resources)
        #expect(error == nil)
        #expect(service is RealModelsService)
        let status = await service.catalogStatus()
        #expect(status.source == "bundle")
        #expect(status.error?.hasPrefix("Using bundled catalog") == true)
        #expect(!(await service.catalog(query: CatalogQuery())).isEmpty)
        await service.shutdown()
    }

    @Test func resourcesPreferTheBundleAndABareDevBinaryUsesTheOverride() throws {
        let resources = try bundle()
        let dev = root.appendingPathComponent("repo"), bare = root.appendingPathComponent("debug")
        #expect(ServiceFactory.resources(bundle: resources, override: dev) == resources)
        #expect(ServiceFactory.resources(bundle: bare, override: dev) == dev)
        #expect(ServiceFactory.resources(bundle: bare, override: nil) == bare)
    }
}
