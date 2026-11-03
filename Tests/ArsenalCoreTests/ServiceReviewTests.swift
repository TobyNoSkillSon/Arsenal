// SPDX-License-Identifier: MIT
import Foundation
import Testing
import CryptoKit
import Darwin
@testable import ArsenalCore

struct ProtocolFixtureArtifact: WorkerArtifact {
    let scenario: String
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws { if scenario == "hang-prepare" { try await Task.sleep(for: .seconds(60)) } }
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let marker = paths.support.appendingPathComponent("fixture-process.pid")
        let code =
            "import sys; sys.path.insert(0, \(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers").path))); sys.argv=[\(String(reflecting: repo.appendingPathComponent("Tests/Fixtures/Workers/service_worker.py").path)),\(String(reflecting: model.kind)),\(String(reflecting: scenario)),\(String(reflecting: marker.path))]; exec(open(sys.argv[0]).read())"
        return WorkerLaunch(
            executable: URL(fileURLWithPath: "/opt/homebrew/bin/python3"), arguments: ["-I", "-c", code], kind: model.kind, runner: "native",
            environment: ["ARSENAL_SUPPORT_DIR": paths.support.path], log: paths.directory("logs").appendingPathComponent("protocol-fixture.log"))
    }
}

func fixturePID(_ paths: ArsenalPaths) -> Int32? {
    (try? String(contentsOf: paths.support.appendingPathComponent("fixture-process.pid"), encoding: .utf8)).flatMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
}
func waitForFixturePID(_ paths: ArsenalPaths) async throws -> Int32 {
    // Fixture startup is not the shutdown latency being tested. Allow scheduling contention
    // before the fixture writes its readiness marker; callers still time shutdown separately.
    let limit = ContinuousClock.now + .seconds(15)
    while ContinuousClock.now < limit { if let pid = fixturePID(paths) { return pid }; try await Task.sleep(for: .milliseconds(10)) }
    throw WorkerFailure(code: "fixture", message: "Fixture process did not start.")
}

@Test func loadVerifiesSignedCodeMembershipAndPinsBeforeSpawn() async throws {
    for change in ["manifest", "added-file", "weights", "asset"] {
        let f = try await makeLocalCatalogFixture(kinds: ["chat"], assetPayloads: ["asset.bin": Data([9])], artifacts: ProtocolFixtureArtifact(scenario: "normal"))
        defer { f.stop() }
        try await f.service.get(id: "fixture-chat", precision: "bf16")
        let folder = f.paths.directory("models").appendingPathComponent("fixture-chat")
        switch change {
        case "manifest":
            let url = folder.appendingPathComponent("model.toml"); var data = try Data(contentsOf: url); data.append(Data("\n# tampered\n".utf8)); try data.write(to: url)
        case "added-file": try Data("shadow module".utf8).write(to: folder.appendingPathComponent("shadow.py"))
        case "weights":
            let store = BlobStore(support: f.paths.support); let dir = try await store.snapshotURL(model: "fixture-chat", revision: f.revision, precision: "bf16")
            let file = dir.appendingPathComponent("base.bin"); try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: file.path)
            try Data([99, 98, 97]).write(to: file)
            var manifest = try JSONDecoder().decode(SnapshotManifest.self, from: Data(contentsOf: dir.appendingPathComponent("arsenal-snapshot.json")))
            manifest.files[0].sha256 = FileDigest.sha256(Data([99, 98, 97])); try JSONEncoder().encode(manifest).write(to: dir.appendingPathComponent("arsenal-snapshot.json"))
        default:
            let store = BlobStore(support: f.paths.support); let dir = try await store.snapshotURL(model: "asset-fixture-asset", revision: f.revision, precision: "fp16")
            let file = dir.appendingPathComponent("asset.bin"); try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: file.path)
            try Data([8]).write(to: file)
        }
        do { try await f.service.load(id: "fixture-chat"); Issue.record("Tampered installation was loaded: \(change)") } catch {}
        #expect(fixturePID(f.paths) == nil)
        #expect(await f.service.supervisor.residentIDs().isEmpty)
        await f.service.shutdown()
    }
}

@Test func maximumContextReachesRowsAndLoad() async throws {
    let f = try await makeLocalCatalogFixture(
        kinds: ["chat"], filePayloads: ["config.json": Data("{\"max_position_embeddings\":262144}".utf8)], artifacts: ProtocolFixtureArtifact(scenario: "normal"), context: 32768,
        nativeContext: 262144)
    defer { f.stop() }
    #expect(await catalogRow(f.service, "fixture-chat")?.metrics["native_context"] == 262144)
    try await f.service.get(id: "fixture-chat", precision: "bf16"); try await f.service.load(id: "fixture-chat")
    let result = try await f.service.run(id: "fixture-chat", op: "inspect_load")
    #expect(result["config"]?["context_limit"] == 262144)
    let row = try #require(await f.service.rows().first)
    #expect(row.metrics["context"] == 262144 && row.metrics["native_context"] == 262144)
    await f.service.shutdown()
}

@Test func precisionMemoryUsesInstalledBytesNotVendorSource() throws {
    let snapshot = SnapshotManifest(
        model: "qwen3.8-27b", revision: String(repeating: "a", count: 40), precision: "int4",
        files: [.init(path: "weights.safetensors", bytes: 15_200_000_000, sha256: String(repeating: "b", count: 64))], derived: nil)
    #expect(try RuntimeMemoryEstimate.megabytes(snapshot: snapshot) == 15_968)
    #expect(try RuntimeMemoryEstimate.megabytes(snapshot: snapshot, measuredMB: 16_100) == 16_100)
}

@Test func shutdownDrainsGetLoadRunAndStopsAdmission() async throws {
    for scenario in ["hang-prepare", "hang-load", "hang-run", "hang-convert"] {
        let f = try await makeLocalCatalogFixture(kinds: ["chat"], artifacts: ProtocolFixtureArtifact(scenario: scenario)); defer { f.stop() }
        if scenario == "hang-load" || scenario == "hang-run" { try await f.service.get(id: "fixture-chat", precision: "bf16") }
        if scenario == "hang-run" { try await f.service.load(id: "fixture-chat") }
        let work = Task {
            if scenario == "hang-prepare" {
                try await f.service.get(id: "fixture-chat", precision: "bf16")
            } else if scenario == "hang-convert" {
                try await f.service.get(id: "fixture-chat", precision: "int4")
            } else if scenario == "hang-load" {
                try await f.service.load(id: "fixture-chat")
            } else {
                _ = try await f.service.run(id: "fixture-chat", op: "chat")
            }
        }
        let pid: Int32?
        if scenario == "hang-prepare" {
            try await Task.sleep(for: .milliseconds(30)); pid = nil
        } else {
            pid = try await waitForFixturePID(f.paths); try await Task.sleep(for: .milliseconds(30))
        }
        let begin = Date(); await f.service.shutdown(); _ = try? await work.value
        #expect(Date().timeIntervalSince(begin) < 8)
        if let pid { #expect(kill(pid, 0) == -1) }
        #expect(await f.service.supervisor.residentIDs().isEmpty)
        #expect(await f.service.memory().workersMB == 0)
        do { try await f.service.get(id: "fixture-chat", precision: "bf16"); Issue.record("Get admitted after shutdown") } catch let e as WorkerFailure {
            #expect(e.code == "shutting_down")
        }
        do { try await f.service.load(id: "fixture-chat"); Issue.record("Load admitted after shutdown") } catch let e as WorkerFailure { #expect(e.code == "shutting_down") }
        do { _ = try await f.service.run(id: "fixture-chat", op: "chat"); Issue.record("Run admitted after shutdown") } catch let e as WorkerFailure {
            #expect(e.code == "shutting_down")
        }
    }
}

@Test func supervisorShutdownOwnsSlowStartsWithoutLateRegistration() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let model = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData).models[0].manifest
    let launch = try await ProtocolFixtureArtifact(scenario: "slow-hello").launch(model: model, runner: model.runner[0], directory: f.paths.directory("models"), paths: f.paths)
    let supervisor = Supervisor()
    let start = Task { try await supervisor.worker(id: model.id, launch: launch) }
    let pid = try await waitForFixturePID(f.paths)
    await supervisor.shutdown(); _ = try? await start.value
    #expect(await supervisor.residentIDs().isEmpty)
    #expect(kill(pid, 0) == -1)
    do { _ = try await supervisor.worker(id: model.id, launch: launch); Issue.record("Supervisor admitted work after shutdown") } catch let e as WorkerFailure {
        #expect(e.code == "shutting_down")
    }
    await f.service.shutdown()
}

private struct NoHelloConversionArtifact: WorkerArtifact {
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        let marker = paths.support.appendingPathComponent("fixture-process.pid")
        // A live worker that never sends hello, and cannot exit until its owner escalates to SIGKILL.
        return WorkerLaunch(
            executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", "trap '' TERM; echo $$ > '\(marker.path)'; exec /bin/sleep 60"],
            kind: model.kind, runner: runner.name, environment: ["ARSENAL_SUPPORT_DIR": paths.support.path, "ARSENAL_CACHE_DIR": paths.cache.path],
            log: paths.directory("logs").appendingPathComponent("no-hello.log"))
    }
}

@Test(.timeLimit(.minutes(1))) func converterCancellationBeforeHelloAwaitsWorkerExit() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"]); defer { f.stop() }
    let model = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData).models[0].manifest
    let converter = WorkerPrecisionConverter(artifacts: NoHelloConversionArtifact(), paths: f.paths)
    let conversion = Task {
        try await converter.convert(
            model: model, modelDirectory: f.paths.support, sourceDirectory: f.paths.support,
            destinationDirectory: f.paths.support.appendingPathComponent("derived"), precision: "int4", recipe: "fixture")
    }
    let pid: Int32
    do { pid = try await waitForFixturePID(f.paths) } catch { conversion.cancel(); _ = try? await conversion.value; await f.service.shutdown(); throw error }
    defer { _ = kill(-pid, SIGKILL) }
    let before = ContinuousClock.now
    conversion.cancel()
    do { _ = try await conversion.value; Issue.record("Cancelled conversion succeeded") } catch is CancellationError {} catch { Issue.record("Expected cancellation: \(error)") }
    #expect(ContinuousClock.now - before < .seconds(15))
    #expect(kill(pid, 0) == -1)
    #expect(kill(-pid, 0) == -1)
    await f.service.shutdown()
}

@Test func localCopyAvailabilityAndVerificationPhaseReachRows() async throws {
    let donor = URL(fileURLWithPath: "/private/tmp/arsenal-core/local-donor-\(UUID().uuidString)")
    let directory = donor.appendingPathComponent("test/fixture"); try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let payload = Data(repeating: 0x5A, count: 16 * 1024 * 1024); try payload.write(to: directory.appendingPathComponent("base.bin"))
    var configuration = ArsenalConfiguration(); configuration.get = ["localSources": [.string(donor.path)]]
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], filePayloads: ["base.bin": payload], configuration: configuration); defer { f.stop() }
    let initial = try #require(await catalogRow(f.service, "fixture-voices"))
    #expect(initial.downloadBytes["bf16"] == 0 && initial.localCopyBytes["bf16"] == Int64(payload.count))
    let states = DownloadStateCollector(), events = await f.service.events()
    let observer = Task {
        for await event in events {
            if case .progress = event, let state = await f.service.rows().first?.states["bf16"], state.state == .downloading { await states.append(state) }
        }
    }
    f.stop(); try await f.service.get(id: "fixture-voices", precision: "bf16")
    try await Task.sleep(for: .milliseconds(20)); observer.cancel(); await observer.value
    // A queued progress event may observe final transaction verification after the local-copy hash completed.
    #expect(await states.states.allSatisfy { ["verifyingLocalCopy", "verifying"].contains($0.phase ?? "") })
    #expect(await f.service.rows().first?.states["bf16"]?.state == .ready)
    await f.service.shutdown()
}

@Test func bundledBootstrapPersistsSourceAndKeepsNewerCache() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    await f.service.shutdown()
    let base = f.paths.support.deletingLastPathComponent()
    let repository = base.appendingPathComponent("fixture-git"), resources = base.appendingPathComponent("fixture-resources")
    try FileManager.default.createDirectory(at: repository, withIntermediateDirectories: true)
    for (path, data) in f.source.files {
        let url = repository.appendingPathComponent(path); try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
    func git(_ arguments: [String]) throws -> String {
        String(decoding: try GitRead.run(repo: repository, args: arguments), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
    _ = try git(["init", "--quiet"]); _ = try git(["add", "models"])
    _ = try git([
        "-c", "user.name=ArsenalFixture", "-c", "user.email=fixture@example.com", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"
    ])
    var catalog = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData); catalog.sourceCommit = try git(["rev-parse", "HEAD"]); catalog.sequence = 2
    let key = Curve25519.Signing.PrivateKey(), catalogDir = resources.appendingPathComponent("catalog")
    try FileManager.default.createDirectory(at: catalogDir, withIntermediateDirectories: true)
    try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
    try WireJSON.encode(["local_git": .string(repository.path)]).write(to: catalogDir.appendingPathComponent("source.json"))
    let actualRepo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    try FileManager.default.copyItem(at: actualRepo.appendingPathComponent("Resources/kinds"), to: resources.appendingPathComponent("kinds"))
    func writeCatalog() throws {
        let data = try catalog.encoded(); try data.write(to: catalogDir.appendingPathComponent("catalog.json"))
        try key.signature(for: data).write(to: catalogDir.appendingPathComponent("catalog.json.sig"))
    }
    try writeCatalog()
    try FileManager.default.copyItem(at: repository.appendingPathComponent("models"), to: resources.appendingPathComponent("models"))
    let paths = ArsenalPaths(support: base.appendingPathComponent("bootstrap-support")); try paths.create()
    let store = BlobStore(support: paths.support), seed = paths.support.appendingPathComponent("seed"); try Data([12, 34, 56]).write(to: seed)
    try await store.promote(seed, sha256: f.weightHash, bytes: 3)
    _ = try await store.commitSnapshot(
        SnapshotManifest(model: "fixture-voices", revision: f.revision, precision: "bf16", files: [.init(path: "base.bin", bytes: 3, sha256: f.weightHash)], derived: nil))
    let service = try await RealModelsService.open(repository: resources, resources: resources, paths: paths, artifacts: StubWorkerArtifact())
    #expect(await service.catalogStatus().error == nil)
    #expect(try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json")).catalog["source"] == "bundle")
    // The bf16 snapshot is already in the store: weights on this Mac put the model on the table before any Add.
    #expect(await service.rows().map(\.id) == ["fixture-voices"])
    #expect(await service.catalog().count == 1)
    try await service.add(id: "fixture-voices")
    try await service.get(id: "fixture-voices", precision: "bf16") // Bundled folder Get, no network
    #expect(await service.rows().first?.states["bf16"]?.state == .ready)
    await service.shutdown()
    catalog.sequence = 1; try writeCatalog()
    let rollback = try await RealModelsService.open(repository: resources, resources: resources, paths: paths, artifacts: StubWorkerArtifact())
    #expect(await rollback.rows().count == 1)
    #expect(try JSONDecoder().decode(CatalogDocument.self, from: Data(contentsOf: paths.directory("catalog").appendingPathComponent("catalog.json"))).sequence == 2)
    await rollback.shutdown()
    var configured = try ArsenalConfiguration.read(paths.support.appendingPathComponent("config.json"))
    configured.catalog["source"] = ["path": "/missing/explicit-user-repository"]; try configured.write(paths.support.appendingPathComponent("config.json"))
    // Writer intentionally preserves an existing user override; edit the same user-owned key as a settings control would.
    var raw = try WireJSON.decode(Data(contentsOf: paths.support.appendingPathComponent("config.json")))
    raw["catalog"] = ["source": ["path": "/missing/explicit-user-repository"]]; try WireJSON.encode(raw).write(to: paths.support.appendingPathComponent("config.json"))
    let missing = try await RealModelsService.open(repository: resources, resources: resources, paths: paths, artifacts: StubWorkerArtifact())
    #expect(await missing.catalogStatus().source == "bundle")
    #expect(await missing.catalogStatus().error?.contains("repository is missing") == true)
    await missing.shutdown()
}

@Test func staleStagingCleanupReceiptsPrecedeRemovalAndPreserveDownloads() throws {
    let paths = ArsenalPaths(support: URL(fileURLWithPath: "/private/tmp/arsenal-core/staging-\(UUID().uuidString)")); try paths.create()
    let stale = paths.directory("weights").appendingPathComponent(".derive-" + UUID().uuidString)
    let active = paths.directory("weights").appendingPathComponent(".derive-" + UUID().uuidString)
    let downloads = paths.directory("weights").appendingPathComponent("downloads")
    for folder in [stale, active, downloads] {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true); try Data([1, 2, 3]).write(to: folder.appendingPathComponent("keep-or-remove.bin"))
    }
    try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-172_800)], ofItemAtPath: stale.path)
    #expect(try StagingCleanup.run(paths: paths) == 1)
    #expect(!FileManager.default.fileExists(atPath: stale.path))
    #expect(FileManager.default.fileExists(atPath: active.path) && FileManager.default.fileExists(atPath: downloads.path))
    let receipts = try FileManager.default.contentsOfDirectory(at: paths.directory("logs").appendingPathComponent("cleanup-receipts"), includingPropertiesForKeys: nil)
    let receipt = try WireJSON.decode(Data(contentsOf: receipts[0]))
    #expect(receipt["verified"] == true && receipt["files"]?.array?.count == 1 && receipt["unlinkedBytes"] == 3)
}
