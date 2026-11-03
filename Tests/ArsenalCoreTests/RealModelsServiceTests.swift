// SPDX-License-Identifier: MIT
import Foundation
import Testing
import CryptoKit
import Darwin
@testable import ArsenalCore

struct FixtureCatalogSource: CatalogSource {
    var files: [String: Data]
    func read(path: String, commit: String) async throws -> Data { guard let data = files[path] else { throw ManifestError.invalid("fixture file missing") }; return data }
}
struct StubWorkerArtifact: WorkerArtifact {
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        var value = stubLaunch(model.kind); value.environment["ARSENAL_SUPPORT_DIR"] = paths.support.path; return value
    }
}

final class FixtureHTTPServer: @unchecked Sendable {
    private let child: ChildProcess
    private let lock = NSLock()
    private var stopped = false
    init(_ child: ChildProcess) { self.child = child }
    func readOutput(count: Int) async throws -> Data? { try await child.readOutputAsync(count: count) }
    func stop() {
        guard
            lock.withLock({
                if stopped { return false }; stopped = true; return true
            })
        else { return }
        child.signal(SIGKILL); _ = child.awaitExit(); child.closeInput(); child.finishOutput()
    }
    deinit { stop() }
}
struct LocalCatalogFixture {
    let service: RealModelsService
    let paths: ArsenalPaths
    let server: FixtureHTTPServer
    let catalogData: Data
    let signature: Data
    let publicKey: Data
    let source: FixtureCatalogSource
    let client: HuggingFaceClient
    let weightHash: String
    let revision: String
    func stop() { server.stop() }
    func reopen(threshold: Int64) async throws -> RealModelsService {
        try await RealModelsService.open(
            catalogData: catalogData, signature: signature, publicKey: publicKey, source: source, kinds: KindSpec.defaults, paths: paths, artifacts: StubWorkerArtifact(),
            client: client, minuteSeconds: 0.01, sourceRetentionThresholdBytes: threshold)
    }
}
/// Removes the fixture's 16-bit copy the way an older build did: a verified derive wrote a retention receipt and unlinked the source.
/// The current policy never does that, so tests that need a missing 16-bit copy reach it through the store's own API.
@discardableResult
func simulateOlderSourceRetention(_ f: LocalCatalogFixture, derivedPrecision: String) async throws -> SourceRetentionReceipt {
    let store = BlobStore(support: f.paths.support)
    let receipt = try await store.retainVerifiedDerived(
        model: "fixture-chat", revision: f.revision, sourcePrecision: "bf16", derivedPrecision: derivedPrecision, thresholdBytes: 1)
    // The same refresh the service runs after its own retention, so the row's download sizes and note follow the receipt.
    await f.service.refreshInstalled(forceDownloadSizes: true)
    return try #require(receipt)
}
/// A model's row as Get more models lists it (`catalog()`). The Models table (`rows()`) holds only the models with weights
/// on this Mac or a Get under way, so a test about a catalog model's own data reads it here. `id` nil: the first entry.
func catalogRow(_ service: any ModelsService, _ id: String? = nil) async -> ModelRow? {
    await service.catalog(query: CatalogQuery()).first { id == nil || $0.id == id }?.row
}
func makeLocalCatalogFixture(
    kinds: [String] = ["voices", "decisions", "images", "chat"], threshold: Int64 = 8_000_000_000, filePayloads: [String: Data] = ["base.bin": Data([12, 34, 56])],
    assetPayloads: [String: Data] = [:], sourceDtype: String = "bf16", configuration: ArsenalConfiguration? = nil, artifacts: any WorkerArtifact = StubWorkerArtifact(),
    context: Int? = nil, nativeContext: Int? = nil, addToTable: Bool = true, extraTable: [String: TOMLValue] = [:],
    extraFiles: [String: Data] = [:], kindSpecs: [KindSpec]? = nil, kindsDirectory: URL? = nil, kindFiles: [CatalogFile] = [], kindContents: [String: Data] = [:],
    minuteSeconds: Double = 0.01, sourceDate: String? = "2026-10-01T12:00:00+00:00",
    gatedRequests: Int? = nil, token: String? = nil
) async throws -> LocalCatalogFixture {
    let root = URL(fileURLWithPath: "/private/tmp/arsenal-core/real-service-\(UUID().uuidString)").resolvingSymlinksInPath()
    let paths = ArsenalPaths(support: root.appendingPathComponent("support"))
    if let configuration { try configuration.write(paths.support.appendingPathComponent("config.json")) }
    let serverRoot = root.appendingPathComponent("hf"); let revision = String(repeating: "a", count: 40)
    let hash = FileDigest.sha256(filePayloads["base.bin"] ?? filePayloads.values.first!)
    for (repo, payloads) in [("fixture", filePayloads), ("asset", assetPayloads)] {
        let directory = serverRoot.appendingPathComponent("test/\(repo)/resolve/\(revision)")
        for (name, bytes) in payloads {
            let file = directory.appendingPathComponent(name); try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: file)
        }
    }
    var code =
        "import http.server,functools,os,sys,threading; threading.Thread(target=lambda:(sys.stdin.buffer.read(),os._exit(0)),daemon=True).start(); s=http.server.HTTPServer(('127.0.0.1',0),functools.partial(http.server.SimpleHTTPRequestHandler,directory=\(String(reflecting: serverRoot.path)))); print(s.server_port,flush=True); s.serve_forever()"
    // A gated repository: the first `gatedRequests` file requests (every one when negative) are refused as Hugging Face
    // refuses them (401 without a token, 403 with one, `X-Error-Code: GatedRepo`); each refusal appends a line to
    // `refusals` in the server root, and each request records whether a token came, never its value.
    if let gatedRequests { code = gatedServerCode(root: serverRoot, refusals: gatedRequests) }
    let server = FixtureHTTPServer(
        try ChildProcess(
            executable: URL(fileURLWithPath: "/opt/homebrew/bin/python3"), arguments: ["-I", "-c", code], environment: RunnerEnvironment.scrub(),
            log: root.appendingPathComponent("http-server.log")))
    var transferred = false
    defer { if !transferred { server.stop() } }
    var portText = Data(); while let next = try await server.readOutput(count: 1), !next.isEmpty, next != Data([10]) { portText.append(next) }
    let port = try #require(Int(String(decoding: portText, as: UTF8.self)))
    var catalogModels: [CatalogModel] = []; var sourceFiles: [String: Data] = [:]
    for kind in kinds {
        let id = "fixture-\(kind)"; let folder = root.appendingPathComponent("folders/\(id)")
        for dir in ["gate", "bench", "log"] { try FileManager.default.createDirectory(at: folder.appendingPathComponent(dir), withIntermediateDirectories: true) }
        let toml = """
            schema = 1
            id = "\(id)"
            name = "\(kind == "chat" ? "Qwen3.8-27B protocol fixture" : "Fixture \(kind)")"
            kind = "\(kind)"
            summary = "Stub-only integration fixture."
            publisher = "Arsenal"
            released = 2026
            tasks = ["fixture"]
            languages = ["en"]
            params = "\(kind == "chat" ? "27B" : "0")"
            [licence]
            spdx = "Apache-2.0"
            commercial = true
            url = "https://example.com/licence"
            [weights]
            repo = "test/fixture"
            revision = "\(revision)"
            dtype = "bf16"
            files = ["base.bin"]
            [weights.sha256]
            "base.bin" = "\(hash)"
            [weights.bytes]
            "base.bin" = 3
            [derive]
            int4 = "opaque-fixture-recipe"
            int8 = "opaque-fixture-recipe"
            [[runner]]
            name = "native"
            language = "swift"
            worker = "stub"
            protocol = 1
            chips = ["*"]
            min_macos = "26.0"
            fast_exact = true
            [[menu]]
            id = "fixture-action"
            title = "Fixture Action"
            op = "ping"
            input = "none"
            [gate]
            recipe = "gate/gate.toml"
            [bench]
            recipe = "bench/bench.toml"
            """
        var parser = TOMLParser(toml); var table = try parser.parse()
        func weightTable(repo: String, payloads: [String: Data], dtype: String) -> [String: TOMLValue] {
            [
                "repo": .string("test/" + repo), "revision": .string(revision), "dtype": .string(dtype), "files": .array(payloads.keys.sorted().map { .string($0) }),
                "sha256": .table(payloads.mapValues { .string(FileDigest.sha256($0)) }), "bytes": .table(payloads.mapValues { .integer(Int64($0.count)) })
            ]
        }
        if let context { table["context"] = .integer(Int64(context)) }
        if let nativeContext { table["native_context"] = .integer(Int64(nativeContext)) }
        table["weights"] = .table(weightTable(repo: "fixture", payloads: filePayloads, dtype: sourceDtype))
        if sourceDtype == "fp32" { table["derive"] = .table(["fp16": .string("opaque-cast"), "int8": .string("opaque-fixture-recipe"), "int4": .string("opaque-fixture-recipe")]) }
        if !assetPayloads.isEmpty {
            table["assets"] = .array([.string("fixture-asset")])
            var asset = weightTable(repo: "asset", payloads: assetPayloads, dtype: "asset"); asset["dtype"] = nil; asset["id"] = .string("fixture-asset")
            asset["repo_type"] = .string("model"); asset["licence"] = .string("Apache-2.0")
            table["asset"] = .array([.table(asset)])
        }
        table.merge(extraTable) { _, new in new }
        var content: [String: Data] = [
            "model.toml": Data(TOMLWriter.encode(table).utf8), "gate/gate.toml": Data("schema=1".utf8), "bench/bench.toml": Data("schema=1".utf8),
            "README.md": Data("fixture".utf8), "log/001-fixture.md": Data("fixture".utf8)
        ]
        content.merge(extraFiles) { _, new in new }
        for (path, data) in content {
            let target = folder.appendingPathComponent(path); try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target); sourceFiles["models/\(id)/\(path)"] = data
        }
        let listed = try SafePath.files(in: folder); #expect(listed == content.keys.sorted())
        let model = try ModelManifest.load(folder, kindsDirectory: kindsDirectory)
        catalogModels.append(CatalogModel(manifest: model, files: content.map { CatalogFile(path: $0.key, sha256: FileDigest.sha256($0.value), bytes: Int64($0.value.count)) }))
    }
    for (path, data) in kindContents { sourceFiles["kinds/" + path] = data }
    let catalog = CatalogDocument(sequence: 1, sourceCommit: revision, models: catalogModels, kindFiles: kindFiles, sourceDate: sourceDate); let data = try catalog.encoded()
    let key = Curve25519.Signing.PrivateKey()
    var client = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!)
    if gatedRequests != nil || token != nil { client.token = { token }; client.sendsTokenToLoopback = true }
    let s = try await RealModelsService.open(
        catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: FixtureCatalogSource(files: sourceFiles),
        kinds: kindSpecs ?? KindSpec.defaults, paths: paths, artifacts: artifacts, client: client,
        minuteSeconds: minuteSeconds, sourceRetentionThresholdBytes: threshold, kindsDirectory: kindsDirectory)
    if addToTable { for model in catalogModels { try await s.add(id: model.id) } }
    transferred = true
    return LocalCatalogFixture(
        service: s, paths: paths, server: server, catalogData: data, signature: try key.signature(for: data), publicKey: key.publicKey.rawRepresentation,
        source: FixtureCatalogSource(files: sourceFiles), client: client, weightHash: hash, revision: revision)
}
/// The fixture's file server, refusing the first `refusals` file requests (all when negative) as a gated Hub repository does.
private func gatedServerCode(root: URL, refusals: Int) -> String {
    """
    import http.server,functools,os,sys,threading
    threading.Thread(target=lambda:(sys.stdin.buffer.read(),os._exit(0)),daemon=True).start()
    ROOT=\(String(reflecting: root.path)); LIMIT=\(refusals); COUNT=[0]
    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*a): pass
        def do_GET(self):
            if '/resolve/' in self.path:
                token=bool(self.headers.get('Authorization'))
                open(os.path.join(ROOT,'tokens'),'a').write(('token' if token else 'none')+'\\n')
                COUNT[0]+=1
                if LIMIT<0 or COUNT[0]<=LIMIT:
                    open(os.path.join(ROOT,'refusals'),'a').write('refused\\n')
                    self.send_response(403 if token else 401); self.send_header('X-Error-Code','GatedRepo'); self.send_header('Content-Length','0'); self.end_headers(); return
            return super().do_GET()
    s=http.server.HTTPServer(('127.0.0.1',0),functools.partial(H,directory=ROOT)); print(s.server_port,flush=True); s.serve_forever()
    """
}

@Test func signedCatalogLocalDownloadDeriveAllKindsLifecycle() async throws {
    let fixture = try await makeLocalCatalogFixture(); defer { fixture.stop() }
    let s = fixture.service, paths = fixture.paths
    // Added but not downloaded: in Get more models, not yet on the table.
    #expect(await s.rows().isEmpty)
    let before = await s.catalog().map(\.row); #expect(before.count == 4)
    #expect(before.allSatisfy { $0.downloadBytes["int4"] == 3 && $0.verification == "Not tried on this Mac yet" })
    do { _ = try await s.run(id: "fixture-chat", op: "chat"); Issue.record("Run downloaded automatically") } catch let e as WorkerFailure { #expect(e.code == "not_installed") }
    let ops = ["voices": "speak", "decisions": "decide", "images": "generate", "chat": "chat"]
    for row in before {
        try await s.get(id: row.id, precision: "int4")
        let installed = try #require(await s.rows().first { $0.id == row.id }); #expect(installed.downloadBytes["int4"] == 0); #expect(installed.state == .ready)
        try await s.load(id: row.id, precision: "int4", fast: false)
        #expect(await s.memory().workersMB == 16)
        #expect(await s.menuActions().contains { $0.modelID == row.id })
        let result = try await s.run(id: row.id, op: ops[row.kind]!, params: ["text": "hello", "prompt": "fixture", "stream": false])
        #expect(result["ok"] == true)
        try await s.unload(id: row.id); #expect(await s.memory().workersMB == 0)
    }
    // On-demand loads expire; an explicit Keep Hot keeps the same runner resident.
    _ = try await s.run(id: "fixture-chat", op: "chat")
    try await Task.sleep(for: .milliseconds(300)); #expect(await s.memory().workersMB == 0)
    try await s.setKeepHot(id: "fixture-chat", on: true)
    _ = try await s.run(id: "fixture-chat", op: "chat")
    try await Task.sleep(for: .milliseconds(300)); #expect(await s.memory().workersMB == 16)
    #expect(!FileManager.default.fileExists(atPath: paths.directory("state").appendingPathComponent("runner-choice.json").path))
    await s.shutdown(); #expect(await s.memory().workersMB == 0)
    #expect(!FileManager.default.fileExists(atPath: paths.directory("envs").appendingPathComponent("fixture-chat").path))
}

@Test func sourceRetentionRequiresLoadAndRestoresViaGet() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], threshold: 1); defer { f.stop() }
    let s = f.service, id = "fixture-chat"
    try await s.get(id: id, precision: "int4")
    #expect(await s.rows().first?.states["bf16"]?.state == .ready)
    try await s.load(id: id, precision: "bf16", fast: false)
    #expect(await s.rows().first?.states["bf16"]?.state == .loaded)
    try await s.load(id: id, precision: "int4", fast: false)
    // Loading the derived tier keeps the 16-bit copy: the snapshot stays on disk and its row stays ready.
    let loaded = try #require(await s.rows().first)
    #expect(loaded.states["int4"]?.state == .loaded)
    #expect(loaded.states["bf16"]?.state == .ready && loaded.states["bf16"]?.phase == nil)
    #expect(loaded.retentionNote == nil)
    let store = BlobStore(support: f.paths.support)
    #expect(try await store.verifySnapshot(model: id, revision: f.revision, precision: "bf16") != nil)
    #expect(try await store.sourceRetentionReceipt(model: id, revision: f.revision, sourcePrecision: "bf16") == nil)
    // A 16-bit copy missing for another reason (an older build's receipt) must still show as removed and restore through Get.
    try await simulateOlderSourceRetention(f, derivedPrecision: "int4")
    let row = try #require(await s.rows().first)
    #expect(row.states["int4"]?.state == .loaded)
    #expect(row.states["bf16"]?.state == .notInstalled)
    #expect(row.states["int8"]?.state == .notInstalled)
    #expect(row.downloadBytes["bf16"] == 3 && row.downloadBytes["int8"] == 3)
    #expect(row.retentionNote == "16-bit copy removed while int4 was in use; Get of bf16 or another tier downloads it again (3 bytes)")
    #expect(row.states["bf16"]?.phase == PrecisionState.sourceRemovedPhase)
    #expect(try await store.verifySnapshot(model: id, revision: f.revision, precision: "bf16") == nil)
    let derived = try #require(await store.verifySnapshot(model: id, revision: f.revision, precision: "int4"))
    #expect(derived.derived?.sourceFiles?.first?.sha256 == f.weightHash)
    let receipt = try #require(await store.sourceRetentionReceipt(model: id, revision: f.revision, sourcePrecision: "bf16"))
    #expect(receipt.reclaimedBytes == 3)
    #expect(!FileManager.default.fileExists(atPath: receipt.path))
    await s.shutdown()
    let reopened = try await f.reopen(threshold: 1)
    #expect(await reopened.rows().first?.retentionNote != nil)
    try await reopened.get(id: id, precision: "bf16")
    #expect(await reopened.rows().first?.retentionNote == nil)
    #expect(await reopened.rows().first?.states["bf16"]?.state == .ready)
    // int8 is refused now, so the derived tier is re-derived as int4 from the restored 16-bit copy, never from the old int4.
    try await reopened.deletePrecision(id: id, precision: "int4")
    try await reopened.get(id: id, precision: "int4")
    let next = try #require(await store.verifySnapshot(model: id, revision: f.revision, precision: "int4"))
    #expect(next.derived?.sourceDtype == "bf16")
    #expect(next.derived?.sourceHashes["base.bin"] == f.weightHash) // derived from the 16-bit source, not from int4
    await reopened.shutdown()
}

@Test func sourceRetentionKeepsSharedBlobsAndSmallSources() async throws {
    // int8 is refused (RetiredInt8Tests); the derived tier this retention covers is int4.
    for tier in ["int4"] {
        // Small source (3 bytes, under the threshold) and large source (over it) both keep the 16-bit copy ready after a derived load.
        let small = try await makeLocalCatalogFixture(kinds: ["chat"], threshold: 3); defer { small.stop() }
        try await small.service.get(id: "fixture-chat", precision: tier)
        try await small.service.load(id: "fixture-chat", precision: tier, fast: false)
        #expect(await small.service.rows().first?.states["bf16"]?.state == .ready)
        #expect(await small.service.rows().first?.retentionNote == nil)
        await small.service.shutdown()
        let shared = try await makeLocalCatalogFixture(kinds: ["chat"], threshold: 1); defer { shared.stop() }
        try await shared.service.get(id: "fixture-chat", precision: tier)
        let store = BlobStore(support: shared.paths.support)
        let asset = try await store.commitSnapshot(
            SnapshotManifest(
                model: "asset-shared-fixture", revision: shared.revision, precision: "fp16", files: [DerivedFile(path: "shared.bin", bytes: 3, sha256: shared.weightHash)],
                derived: nil
            ))
        try await shared.service.load(id: "fixture-chat", precision: tier, fast: false)
        #expect(await shared.service.rows().first?.states["bf16"]?.state == .ready)
        #expect(await shared.service.rows().first?.retentionNote == nil)
        #expect(try await store.verifySnapshot(model: "fixture-chat", revision: shared.revision, precision: "bf16") != nil)
        // An older build's receipt for this source: the blob it shares with the asset is never unlinked.
        let receipt = try await simulateOlderSourceRetention(shared, derivedPrecision: tier)
        #expect(receipt.reclaimedBytes == 0)
        #expect(receipt.retainedSharedBytes == 3)
        #expect(try await store.hasBlob(sha256: shared.weightHash, bytes: 3))
        #expect(FileManager.default.fileExists(atPath: asset.appendingPathComponent("shared.bin").path))
        await shared.service.shutdown()
    }
}

actor DownloadStateCollector { var states: [PrecisionState] = []; func append(_ value: PrecisionState) { states.append(value) } }
@Test func aggregateMultiFileGetCountsWholeModelAndAssets() async throws {
    let primary: [String: Data] = [
        "config.json": Data(repeating: 0xAA, count: 4096), "weights.bin": Data(repeating: 0xBB, count: 131_072), "present.txt": Data(repeating: 0xDD, count: 8)
    ]
    let assets: [String: Data] = ["config.json": Data(repeating: 0xCC, count: 262_144), "shared.bin": primary["weights.bin"]!]
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], filePayloads: primary, assetPayloads: assets); defer { f.stop() }
    let store = BlobStore(support: f.paths.support)
    let present = f.paths.support.appendingPathComponent("preseed"); try primary["present.txt"]!.write(to: present)
    try await store.promote(present, sha256: FileDigest.sha256(primary["present.txt"]!), bytes: 8)
    let states = DownloadStateCollector(); let events = await f.service.events()
    let observer = Task {
        for await event in events {
            if case .progress = event, let state = await f.service.rows().first?.states["bf16"], state.state == .downloading { await states.append(state) }
        }
    }
    try await f.service.get(id: "fixture-voices", precision: "bf16")
    try await Task.sleep(for: .milliseconds(20)); observer.cancel(); await observer.value
    let expected: Int64 = 4096 + 131_072 + 262_144 // shared hash once; present blob excluded
    let finished = try #require(await f.service.rows().first?.states["bf16"])
    #expect(finished.totalBytes == expected && finished.bytes == expected)
    #expect(finished.progress == 1 && finished.bytesPerSecond?.isFinite == true)
    let observed = await states.states
    #expect(!observed.isEmpty)
    #expect(observed.allSatisfy { $0.totalBytes == expected && ($0.bytes ?? 0) <= expected })
    for (a, b) in zip(observed, observed.dropFirst()) { #expect((a.bytes ?? 0) <= (b.bytes ?? 0)) }
    await f.service.shutdown()
}

@Test func derivedGetUsesInstalledCastSourceWithoutNetwork() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], sourceDtype: "fp32"); defer { f.stop() }
    try await f.service.get(id: "fixture-voices", precision: "fp16")
    #expect(await f.service.rows().first?.downloadBytes["int4"] == 0)
    f.stop() // A wrong attempt to download the fp32 vendor source now fails decisively.
    try await f.service.get(id: "fixture-voices", precision: "int4")
    let state = try #require(await f.service.rows().first?.states["int4"])
    #expect(state.totalBytes == 0 && state.bytes == 0 && state.state == .ready)
    await f.service.shutdown()
}

/// "Unload Idle Models and Load" used to act on the idle list taken when the pop-up opened, so a model set
/// to Keep loaded (or busy) since then was unloaded anyway. The service checks each one again right before unloading.
@Test func unloadIdleSparesAModelThatIsNoLongerIdle() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    try await f.service.get(id: "fixture-voices", precision: "bf16")
    try await f.service.load(id: "fixture-voices")
    try await f.service.setKeepHot(id: "fixture-voices", on: true)
    await f.service.unloadIdle(ids: ["fixture-voices"])
    #expect(await f.service.rows().first { $0.id == "fixture-voices" }?.residentDType != nil, "Keep loaded since the pop-up: kept")
    try await f.service.setKeepHot(id: "fixture-voices", on: false)
    await f.service.unloadIdle(ids: ["fixture-voices"])
    #expect(await f.service.rows().first { $0.id == "fixture-voices" }?.residentDType == nil, "idle: unloaded")
    await f.service.shutdown()
}
