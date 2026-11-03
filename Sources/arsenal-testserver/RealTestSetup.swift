// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// Private QA catalog generated from committed model folders; it never edits the repository catalog or app data.
enum RealTestSetup {
    static let allowed = [
        "kokoro-82m", "pocket-tts", "qwen3-tts-voicedesign", "qwen3-tts-base", "breeze-tts-2", "d1-3b", "stable-audio-3-medium", "stable-audio-3-small-sfx", "re-use",
        "laya-english",
        "laya-multilingual", "laya-typed-decisions", "von-1.2", "von-1.1", "qwen3.8-27b", "flux2-klein-4b",
        "yue2-3b", "qwen3-embedding-8b", "qwen3-embedding-0.6b", "sam3.1", "sam-audio-large", "nemotron-3-diarization", "trellis2-4b", "birefnet",
        "rf-detr-large", "mossformer2-se-48k", "sortformer-4spk-v2", "minimax-music3", "melband-roformer-kim",
        "lco-embedding-omni-7b", "elephant-1", "moss-transcribe-diarize", "deepfilternet3", "embeddinggemma-2",
        "voxcpm2", "diarizen-large-v2", "gimm-vfi-r", "unipase", "t0-gamma", "timestral-209m", "esm3-sm-open-v1",
        "fish-drama-3", "chandra-ocr-2", "idol"
    ]
    static func prepare(repository: URL, paths: ArsenalPaths) async throws {
        try paths.create()
        let requested = ProcessInfo.processInfo.environment["ARSENAL_TESTSERVER_MODELS"].map { $0.split(separator: ",").map(String.init) } ?? allowed
        guard !requested.isEmpty, Set(requested).isSubset(of: Set(allowed)) else {
            throw WorkerFailure(code: "invalid_request", message: "Choose supported Arsenal v2 models for isolated Layer 3.")
        }
        let commit = String(decoding: try GitRead.run(repo: repository, args: ["rev-parse", "HEAD"]), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        let source = LocalGitCatalogSource(repo: repository)
        let kinds = Set(try KindSpec.load(directory: repository.appendingPathComponent("Resources/kinds")).map(\.id))
        var kindFiles: [CatalogFile] = []
        for relative in try SafePath.files(in: repository.appendingPathComponent("Resources/kinds")) {
            let bytes = try await source.read(path: "kinds/\(relative)", commit: commit)
            kindFiles.append(CatalogFile(path: relative, sha256: FileDigest.sha256(bytes), bytes: Int64(bytes.count)))
        }
        var models: [CatalogModel] = []
        for id in requested {
            let folder = repository.appendingPathComponent("models/" + id)
            let manifest = try ModelManifest.load(folder, kinds: kinds)
            var files: [CatalogFile] = []
            for relative in try SafePath.files(in: folder) {
                let bytes = try await source.read(path: "models/\(id)/\(relative)", commit: commit)
                files.append(CatalogFile(path: relative, sha256: FileDigest.sha256(bytes), bytes: Int64(bytes.count)))
            }
            models.append(CatalogModel(manifest: manifest, files: files))
        }
        let resources = paths.support.appendingPathComponent("real-resources")
        let directory = resources.appendingPathComponent("catalog")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let old = try? JSONDecoder().decode(CatalogDocument.self, from: Data(contentsOf: directory.appendingPathComponent("catalog.json")))
        let data = try CatalogDocument(sequence: (old?.sequence ?? 0) + 1, sourceCommit: commit, models: models, kindFiles: kindFiles).encoded()
        let signed = try CatalogTrust.sign(data: data, keyFile: paths.support.appendingPathComponent("dev-signing/ed25519.key"))
        try data.write(to: directory.appendingPathComponent("catalog.json"), options: .atomic)
        try signed.signature.write(to: directory.appendingPathComponent("catalog.json.sig"), options: .atomic)
        try signed.publicKey.write(to: resources.appendingPathComponent("catalog-public-key"), options: .atomic)
        let installer = ModelFolderInstaller(support: paths.support)
        _ = try await installer.accept(data: data, signature: signed.signature, publicKey: signed.publicKey)
        for id in requested { _ = try await installer.install(id: id, source: source, replacingExisting: true) }
        var config = try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json"))
        config.catalog["source"] = .object(["path": .string(repository.path)])
        try config.write(paths.support.appendingPathComponent("config.json"))
        print("Prepared signed local QA catalog and verified folders: \(requested.count) models.")
    }
    static func open(repository: URL, paths: ArsenalPaths, helpers: URL) async throws -> RealModelsService {
        let resources = paths.support.appendingPathComponent("real-resources")
        // Layer 3 uses local files: an absent pin fails immediately at loopback instead of turning a QA call into a
        // network download. ARSENAL_TESTSERVER_DOWNLOAD=1 is the one exception, for a download someone approved: Get
        // then fetches missing pins from Hugging Face into this private profile's store, as the app does.
        var client = HuggingFaceClient()
        if ProcessInfo.processInfo.environment["ARSENAL_TESTSERVER_DOWNLOAD"] != "1" {
            client = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:1")!)
            client.maximumAttempts = 1; client.stallTimeout = 1; client.token = { nil }
        }
        let data = try Data(contentsOf: resources.appendingPathComponent("catalog/catalog.json"))
        let signature = try Data(contentsOf: resources.appendingPathComponent("catalog/catalog.json.sig"))
        let key = try Data(contentsOf: resources.appendingPathComponent("catalog-public-key"))
        let source = LocalGitCatalogSource(repo: repository)
        let installer = ModelFolderInstaller(support: paths.support)
        _ = try await installer.accept(data: data, signature: signature, publicKey: key)
        let verifiedKinds = try await installer.prepareKinds(source: source) ?? repository.appendingPathComponent("Resources/kinds")
        return try await RealModelsService.open(
            catalogData: data, signature: signature, publicKey: key, source: source, sourceName: repository.path, kinds: KindSpec.load(directory: verifiedKinds), paths: paths,
            artifacts: BundledWorkerArtifact(helpers: helpers), client: client, kindsDirectory: verifiedKinds)
    }

}
