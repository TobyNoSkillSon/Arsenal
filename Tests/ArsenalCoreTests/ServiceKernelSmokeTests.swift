// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import Testing
@testable import ArsenalCore

@Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_REAL_SERVICE_LAYA_KERNELS"] == "1"))
func v2LayaFolderKernelsActiveThroughServiceFastAndExact() async throws {
    guard ProcessInfo.processInfo.environment["VELLA_LOCK_HELD"] == "1" else { Issue.record("Use bounded gpulock"); return }
    let paths = ArsenalPaths(); guard paths.support.path.hasPrefix("/tmp/arsenal-v2-service/") else { Issue.record("Use isolated support"); return }
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let kindRoot = repo.appendingPathComponent("Resources/kinds"), folder = repo.appendingPathComponent("models/laya-english")
    let model = try ModelManifest.load(folder, kindsDirectory: kindRoot)
    var content: [String: Data] = [:]
    let modelFiles = try SafePath.files(in: folder).map { path -> CatalogFile in
        let data = try Data(contentsOf: folder.appendingPathComponent(path)); content["models/\(model.id)/" + path] = data
        return CatalogFile(path: path, sha256: FileDigest.sha256(data), bytes: Int64(data.count))
    }
    let kindFiles = try SafePath.files(in: kindRoot).map { path -> CatalogFile in
        let data = try Data(contentsOf: kindRoot.appendingPathComponent(path)); content["kinds/" + path] = data
        return CatalogFile(path: path, sha256: FileDigest.sha256(data), bytes: Int64(data.count))
    }
    let catalog = CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: [CatalogModel(manifest: model, files: modelFiles)], kindFiles: kindFiles)
    let data = try catalog.encoded(), key = Curve25519.Signing.PrivateKey()
    var client = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:9")!); client.maximumAttempts = 1; client.stallTimeout = 1
    let helpers = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_LAYA_HELPERS"] ?? "/tmp/arsenal-v2-service/laya-helpers")
    let s = try await RealModelsService.open(
        catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: FixtureCatalogSource(files: content),
        kinds: KindSpec.load(directory: kindRoot), paths: paths, artifacts: BundledWorkerArtifact(helpers: helpers), client: client,
        kindsDirectory: kindRoot)
    do {
        try await s.add(id: model.id)
        #expect(await s.rows().first?.downloadBytes["fp16"] == 0)
        try await s.get(id: model.id, precision: "fp16")
        let sample = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/runner-check.json")))["decide"]?.object ?? [:]
        var modes: [JSONValue] = []
        for fast in [true, false] {
            try await s.load(id: model.id, precision: "fp16", fast: fast)
            let expected = fast ? "smallm-tile" : "splitk-exact"
            let row = try #require(await s.rows().first)
            #expect(row.kernels.contains { $0["id"]?.string == expected && $0["state"] == "active" })
            let status = try await s.supervisor.request(id: model.id, op: "status")
            #expect(status["kernels"]?.array?.contains { $0["id"]?.string == expected && $0["state"] == "active" } == true)
            let result = try await s.run(id: model.id, op: "decide", params: sample)
            #expect(result["answers"]?.object?.isEmpty == false)
            modes.append(["mode": .string(fast ? "fast" : "exact"), "loadKernels": .array(row.kernels.map(JSONValue.object)), "status": .object(status), "decide": .object(result)])
            try await s.unload(id: model.id)
            #expect(await s.memory().workersMB == 0)
        }
        let evidence: JSONObject = [
            "model": .string(model.id), "path": "RealModelsService Get → Load → status → decide → Unload", "networkBytes": 0, "chip": .string(HostChip.current),
            "modes": .array(modes), "benchmark": false
        ]
        try WireJSON.encode(evidence).write(to: paths.support.appendingPathComponent("laya-service-kernels-proof.json"))
        await s.shutdown()
    } catch { await s.shutdown(); throw error }
}
