// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// Development-only catalog preparation uses the same signed catalog and immutable-folder installer as Get.
public enum BenchmarkLocalProof {
    public static let allowedModels = ["kokoro-82m"]
    public static func prepareCatalog(repository: URL, resources: URL, paths: ArsenalPaths) async throws {
        let persistent = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Arsenal-dev/bench-support").path
        guard paths.support.path.hasPrefix("/tmp/arsenal-") || paths.support.path == persistent, resources.path.hasPrefix("/tmp/arsenal-") else {
            throw BenchmarkError.invalid("Local proof preparation requires isolated Arsenal-dev bench-support or /tmp/arsenal- paths.")
        }
        try paths.create()
        let commit = String(decoding: try GitRead.run(repo: repository, args: ["rev-parse", "HEAD"]), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        let source = LocalGitCatalogSource(repo: repository)
        var models: [CatalogModel] = []
        for id in allowedModels {
            let folder = repository.appendingPathComponent("models/\(id)")
            guard FileManager.default.fileExists(atPath: folder.path) else { continue }
            let manifest = try ModelManifest.load(folder)
            let files = try await catalogFiles(folder: folder, id: id, source: source, commit: commit)
            models.append(CatalogModel(manifest: manifest, files: files))
        }
        let previous = try? JSONDecoder().decode(CatalogDocument.self, from: Data(contentsOf: resources.appendingPathComponent("catalog/catalog.json")))
        let data = try CatalogDocument(sequence: (previous?.sequence ?? 0) + 1, sourceCommit: commit, models: models).encoded()
        let signed = try CatalogTrust.sign(data: data, keyFile: paths.directory("dev-signing").appendingPathComponent("proof.key"))
        let catalogDir = resources.appendingPathComponent("catalog")
        try FileManager.default.createDirectory(at: catalogDir, withIntermediateDirectories: true)
        try data.write(to: catalogDir.appendingPathComponent("catalog.json"), options: .atomic)
        try signed.signature.write(to: catalogDir.appendingPathComponent("catalog.json.sig"), options: .atomic)
        try signed.publicKey.write(to: resources.appendingPathComponent("catalog-public-key"), options: .atomic)
        let installer = ModelFolderInstaller(support: paths.support)
        try await installer.accept(data: data, signature: signed.signature, publicKey: signed.publicKey)
    }
    private static func catalogFiles(folder: URL, id: String, source: LocalGitCatalogSource, commit: String) async throws -> [CatalogFile] {
        var files: [CatalogFile] = []
        for path in try SafePath.files(in: folder) {
            let data = try await source.read(path: "models/\(id)/\(path)", commit: commit)
            guard data == (try Data(contentsOf: folder.appendingPathComponent(path))) else {
                throw BenchmarkError.invalid("Commit model folder changes before preparing the proof catalog.")
            }
            if path.hasPrefix("log/"), path.hasSuffix(".md") { try ResearchLogValidation.validate(markdown: String(decoding: data, as: UTF8.self)) }
            files.append(CatalogFile(path: path, sha256: FileDigest.sha256(data), bytes: Int64(data.count)))
        }
        return files
    }
}
