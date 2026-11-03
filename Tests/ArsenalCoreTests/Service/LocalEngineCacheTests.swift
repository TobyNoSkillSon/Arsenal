// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import ArsenalCore

private struct EmptyLocalSource: CatalogSource {
    func read(path: String, commit: String) async throws -> Data { throw CancellationError() }
}
private final class MutableEngineAvailability: WorkerArtifact, @unchecked Sendable {
    private let lock = NSLock()
    private var missing = Set(["native"])
    func setMissing(_ names: Set<String>) { lock.withLock { missing = names } }
    func validateEngine(model: ModelManifest, runner: RunnerSpec) throws {
        if lock.withLock({ missing.contains(runner.name) }) { throw WorkerFailure(code: "worker_missing", message: "Probe engine unavailable") }
    }
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch { throw CancellationError() }
}

@Test func localEngineDiagnosticsRefreshWithoutFolderEdits() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-pol-fix-core/local-" + UUID().uuidString)
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
    let folder = paths.support.appendingPathComponent("local-models/kokoro-82m")
    try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: repo.appendingPathComponent("models/kokoro-82m"), to: folder)
    let manifest = folder.appendingPathComponent("model.toml")
    var text = "context = 8192\n" + (try String(contentsOf: manifest, encoding: .utf8))
    text += "\n[[runner]]\nname = \"backup\"\nlanguage = \"swift\"\nworker = \"voices\"\nprotocol = 1\nchips = [\"*\"]\nmin_macos = \"26.0\"\nfast_exact = true\n"
    try Data(text.utf8).write(to: manifest)
    let data = try CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: []).encoded()
    let key = Curve25519.Signing.PrivateKey(), artifact = MutableEngineAvailability()
    let service = try await RealModelsService.open(
        catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: EmptyLocalSource(),
        kinds: KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")), paths: paths, artifacts: artifact,
        kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    let first = try #require(await service.catalog().first?.row)
    #expect(first.verification == "Probe engine unavailable")
    artifact.setMissing([])
    let repaired = try #require(await service.catalog().first?.row)
    #expect(repaired.verification == "Local · unsigned folder")
    #expect(repaired.states[repaired.selectedPrecision]?.state == .notInstalled)
    #expect(repaired.contentDigest == first.contentDigest)
    artifact.setMissing(["native"])
    #expect(await service.catalog().first?.row.verification == "Probe engine unavailable")
    try await service.setRunner(id: first.id, runner: "backup")
    let switched = try #require(await service.catalog().first?.row)
    #expect(switched.verification == "Local · unsigned folder")
    #expect(switched.states[switched.selectedPrecision]?.state == .notInstalled)
    #expect(switched.runner == "backup")
    try await service.add(id: first.id)
    for turn in 0..<3 {
        // Recorded by Add without weights: read from Get more models, not on the table.
        #expect(await catalogRow(service, first.id)?.metrics["context"] == 8192)
        #expect(await service.rows().isEmpty)
        let held = root.appendingPathComponent("held-folder-\(turn)")
        try FileManager.default.moveItem(at: folder, to: held)
        #expect(await catalogRow(service, first.id)?.states[switched.selectedPrecision]?.phase == "localFolderError")
        try await service.remove(id: first.id)
        #expect(await service.rows().isEmpty)
        try FileManager.default.moveItem(at: held, to: folder)
        try await service.add(id: first.id)
        let restored = try #require(await catalogRow(service, first.id))
        #expect(restored.metrics["context"] == 8192)
        #expect(restored.verification == "Local · unsigned folder")
        #expect(restored.states[restored.selectedPrecision]?.state == .notInstalled)
    }
    await service.shutdown()
}
