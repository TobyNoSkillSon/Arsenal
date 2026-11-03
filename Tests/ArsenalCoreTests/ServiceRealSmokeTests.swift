// SPDX-License-Identifier: MIT
import Foundation
import Testing
import CryptoKit
@testable import ArsenalCore

/// Opt-in, bounded gpulock smoke. No network: imports the existing, provenance-verified Kokoro FP16 local copy.
@Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_REAL_SERVICE_KOKORO"] == "1"))
func v2KokoroLocalGetSettingsApplyAndUnload() async throws {
    #expect(ProcessInfo.processInfo.environment["VELLA_LOCK_HELD"] == "1")
    guard ProcessInfo.processInfo.environment["VELLA_LOCK_HELD"] == "1" else { return }
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let paths = ArsenalPaths()
    #expect(paths.support.path.hasPrefix("/tmp/arsenal-v2-service/"))
    guard paths.support.path.hasPrefix("/tmp/arsenal-v2-service/") else { return }
    try paths.create()
    let origin = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
        "Library/Application Support/Arsenal-dev/weights/voices/kokoro-catalog-proof/weights/snapshots")
    let kindRoot = repository.appendingPathComponent("Resources/kinds")
    let original = try ModelManifest.load(repository.appendingPathComponent("models/kokoro-82m"), kindsDirectory: kindRoot)
    let base = origin.appendingPathComponent("kokoro-82m/\(original.weights.revision)/fp16")
    let snapshot = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: base.appendingPathComponent("arsenal-snapshot.json")))
    #expect(snapshot.derived?.sourceRevision == original.weights.revision)
    // This fixture imports the already-cast FP16 result, not a pretend vendor download. The shipped source pins
    // stay untouched; this proves the ordinary local importer/service path using real Kokoro tensor bytes.
    let source = WeightSpec(
        repo: "proof/kokoro-local-fp16", revision: original.weights.revision, dtype: "fp16", files: snapshot.files.map(\.path),
        sha256: Dictionary(uniqueKeysWithValues: snapshot.files.map { ($0.path, $0.sha256) }), bytes: Dictionary(uniqueKeysWithValues: snapshot.files.map { ($0.path, $0.bytes) }))
    let donor = paths.support.deletingLastPathComponent().appendingPathComponent("local-donor")
    func clone(_ from: URL, _ to: URL) async throws {
        try FileManager.default.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: to.path) { return }
        let cp = try ChildProcess(
            executable: URL(fileURLWithPath: "/bin/cp"), arguments: ["-c", from.path, to.path], environment: RunnerEnvironment.scrub(),
            log: paths.directory("logs").appendingPathComponent("clone-\(UUID().uuidString).log"))
        let status = await cp.awaitExitAsync(); #expect(status == 0); cp.closeInput(); cp.finishOutput(); await cp.finishReaders()
    }
    for file in snapshot.files { try await clone(base.appendingPathComponent(file.path), donor.appendingPathComponent(source.repo + "/" + file.path)) }
    for asset in original.asset ?? [] {
        let directory = origin.appendingPathComponent("asset-\(asset.id)/\(asset.revision)/fp16")
        for file in asset.files { try await clone(directory.appendingPathComponent(file), donor.appendingPathComponent(asset.repo + "/" + file)) }
    }
    var model = original; model.weights = source; model.derive = nil; model.vendor = nil
    let folder = repository.appendingPathComponent("models/kokoro-82m")
    var parser = TOMLParser(try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8)); var toml = try parser.parse()
    toml["derive"] = nil; toml["vendor"] = nil
    toml["weights"] = .table([
        "repo": .string(source.repo), "revision": .string(source.revision), "dtype": .string("fp16"), "files": .array(source.files.map(TOMLValue.string)),
        "sha256": .table(source.sha256!.mapValues(TOMLValue.string)), "bytes": .table(source.bytes!.mapValues(TOMLValue.integer))
    ])
    var content: [String: Data] = [:]
    for file in try SafePath.files(in: folder) {
        content["models/kokoro-82m/" + file] = file == "model.toml" ? Data(TOMLWriter.encode(toml).utf8) : try Data(contentsOf: folder.appendingPathComponent(file))
    }
    let listed = content.map { path, data in CatalogFile(path: String(path.dropFirst("models/kokoro-82m/".count)), sha256: FileDigest.sha256(data), bytes: Int64(data.count)) }
    let kindInventory = try SafePath.files(in: kindRoot).map { path -> CatalogFile in
        let data = try Data(contentsOf: kindRoot.appendingPathComponent(path)); content["kinds/" + path] = data
        return CatalogFile(path: path, sha256: FileDigest.sha256(data), bytes: Int64(data.count))
    }
    let document = CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: [CatalogModel(manifest: model, files: listed)], kindFiles: kindInventory)
    let data = try document.encoded(), key = Curve25519.Signing.PrivateKey()
    var config = ArsenalConfiguration(); config.get = ["localSources": [.string(donor.path)]]; try config.write(paths.support.appendingPathComponent("config.json"))
    var client = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:9")!); client.maximumAttempts = 1; client.stallTimeout = 1
    let helpers = repository.appendingPathComponent("dist/Arsenal.app/Contents/Helpers")
    let service = try await RealModelsService.open(
        catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: FixtureCatalogSource(files: content),
        kinds: KindSpec.load(directory: repository.appendingPathComponent("Resources/kinds")), paths: paths,
        artifacts: BundledWorkerArtifact(helpers: helpers), client: client, kindsDirectory: repository.appendingPathComponent("Resources/kinds"))
    do {
        #expect(await service.rows().isEmpty)
        try await service.add(id: model.id)
        #expect(await service.rows().first?.downloadBytes["fp16"] == 0)
        try await service.get(id: model.id, precision: "fp16")
        try await service.load(id: model.id, precision: "fp16", fast: false)
        try await service.setSetting(id: model.id, key: "voice", value: "am_michael")
        #expect(await service.rows().first?.needsReload == false) // Kokoro's settings are request-time only.
        let voices = try await service.run(id: model.id, op: "voices")
        #expect(voices["voices"]?.array?.isEmpty == false)
        try await service.applySettings(id: model.id)
        let result = try await service.run(id: model.id, op: "speak", params: ["text": "Hello world.", "stream": false])
        let wav = try #require(result["path"]?.string)
        #expect(result["duration_s"]?.number ?? 0 > 0 && FileManager.default.fileExists(atPath: wav))
        try await service.unload(id: model.id); #expect(await service.memory().workersMB == 0)
        let evidence: JSONObject = [
            "model": .string(model.id), "dtype": "fp16", "weights": "existing verified FP32-to-FP16 local cast; test catalog only", "networkBytes": 0,
            "networkFallback": "disabled", "requestVoice": "am_michael", "needsReload": false, "applySettings": "unload + real Kokoro reload", "speak": .object(result),
            "workersAfterUnload": 0, "loadSettingRedProof": "v2RealSettingsDeliveryReloadRequestDefaultsAndKernels CPU wire fixture; Kokoro declares request-time settings only",
            "benchmark": false
        ]
        try WireJSON.encode(evidence).write(to: paths.support.appendingPathComponent("kokoro-service-proof.json"))
        await service.shutdown()
    } catch { await service.shutdown(); throw error }
}
