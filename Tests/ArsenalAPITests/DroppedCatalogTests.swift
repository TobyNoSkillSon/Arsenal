// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import Testing
@testable import ArsenalCore
@testable import ArsenalAPI

private struct DroppedCatalogSource: CatalogSource {
    let files: [String: Data]
    func read(path: String, commit: String) async throws -> Data {
        guard let data = files[path] else { throw ManifestError.invalid("Missing fixture file") }
        return data
    }
}

private struct DroppedCatalogWorker: WorkerArtifact {
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        throw WorkerFailure(code: "unexpected_launch", message: "Catalog removal must not launch a worker.")
    }
}

private struct DroppedCatalogFixture {
    let paths: ArsenalPaths
    let key: Curve25519.Signing.PrivateKey
    let entry: CatalogModel
    let source: DroppedCatalogSource
    let data: Data
    let service: RealModelsService
    static let id = "former-chat"
    static let payload = Data([1, 2, 3])

    static func make(root: URL, installed: Bool) async throws -> Self {
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        try paths.create()
        let descriptor = """
            schema = 1
            id = "former-chat"
            name = "Former Chat Model"
            kind = "chat"
            summary = "Catalog removal regression fixture"
            params = "1B"
            publisher = "Arsenal"
            released = 2026
            tasks = ["chat"]
            languages = ["en"]
            [licence]
            spdx = "MIT"
            commercial = true
            url = "https://example.com/license"
            [weights]
            repo = "fixture/former-chat"
            revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            dtype = "fp16"
            files = ["base.bin"]
            [weights.sha256]
            "base.bin" = "\(FileDigest.sha256(payload))"
            [weights.bytes]
            "base.bin" = 3
            [derive]
            int4 = "opaque-fixture-recipe"
            [[runner]]
            name = "native"
            language = "swift"
            worker = "stub"
            protocol = 1
            chips = ["*"]
            min_macos = "26.0"
            fast_exact = false
            [gate]
            recipe = "gate.toml"
            [bench]
            recipe = "bench.toml"
            """
        let files = [
            "model.toml": Data(descriptor.utf8), "gate.toml": Data("schema=1".utf8), "bench.toml": Data("schema=1".utf8),
            "README.md": Data("Fixture".utf8), "log/001-fixture.md": Data("Fixture".utf8)
        ]
        let folder = root.appendingPathComponent("source/" + id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for (path, bytes) in files {
            let destination = folder.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: destination)
        }
        let manifest = try ModelManifest.load(folder)
        let entry = CatalogModel(manifest: manifest, files: files.map { CatalogFile(path: $0.key, sha256: FileDigest.sha256($0.value), bytes: Int64($0.value.count)) })
        let source = DroppedCatalogSource(files: Dictionary(uniqueKeysWithValues: files.map { ("models/\(id)/\($0.key)", $0.value) }))
        let key = Curve25519.Signing.PrivateKey()
        let data = try CatalogDocument(sequence: 1, sourceCommit: manifest.weights.revision, models: [entry]).encoded()
        let service = try await RealModelsService.open(
            catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: source,
            kinds: KindSpec.defaults, paths: paths, artifacts: DroppedCatalogWorker())
        try await service.add(id: id)
        if installed {
            _ = try await service.installer.install(id: id, source: source)
            let store = BlobStore(support: paths.support), staged = root.appendingPathComponent("weight-fixture.bin")
            try payload.write(to: staged)
            let file = DerivedFile(path: "base.bin", bytes: 3, sha256: FileDigest.sha256(payload))
            try await store.promote(staged, sha256: file.sha256, bytes: file.bytes)
            // Only a non-selected dtype in an older revision survives; none may be silently discarded.
            try await store.commitSnapshot(.init(model: id, revision: String(repeating: "b", count: 40), precision: "int4", files: [file], derived: nil))
        }
        return Self(paths: paths, key: key, entry: entry, source: source, data: data, service: service)
    }
    func publish(sequence: Int64, included: Bool) throws -> Data {
        let data = try CatalogDocument(sequence: sequence, sourceCommit: entry.weights.revision, models: included ? [entry] : []).encoded()
        try data.write(to: paths.directory("catalog").appendingPathComponent("catalog.json"), options: .atomic)
        try key.signature(for: data).write(to: paths.directory("catalog").appendingPathComponent("catalog.json.sig"), options: .atomic)
        return data
    }
    func reopen(_ data: Data) async throws -> RealModelsService {
        try await RealModelsService.open(
            catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation, source: source,
            kinds: KindSpec.defaults, paths: paths, artifacts: DroppedCatalogWorker())
    }
    func weightEvidence() throws -> [String: Data] {
        let root = paths.directory("weights")
        return try Dictionary(
            uniqueKeysWithValues: SafePath.files(in: root).filter { $0.hasPrefix("snapshots/") || $0.hasPrefix("blobs/") }.map {
                ($0, try Data(contentsOf: root.appendingPathComponent($0)))
            })
    }
}

struct DroppedCatalogTests {
    @Test(arguments: ["service", "http", "mcp"], [false, true])
    func refreshDeletesDroppedWeightsAndForgetsTheRow(path: String, installed: Bool) async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tmp/catalog-dropped-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let f = try await DroppedCatalogFixture.make(root: root, installed: installed)
        let server = ArsenalAPIServer(service: f.service, supportDirectory: f.paths.support, port: 0)
        let port = try await server.start()
        defer { server.stop() }
        let router = APIRouter(service: f.service, support: f.paths.support)
        func http(_ route: String, post: JSONObject? = nil) async throws -> Data {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/" + route)!)
            request.timeoutInterval = 10
            if let post {
                request.httpMethod = "POST"; request.httpBody = try WireJSON.encode(post)
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            }
            let (data, response) = try await URLSession.shared.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == 200, "\(String(decoding: data, as: UTF8.self))")
            return data
        }
        func refresh() async throws {
            switch path {
            case "service": try await f.service.refresh()
            case "http": _ = try await http("refresh", post: [:])
            default:
                let result = try await router.callMCPTool(MCPNames.tool("settings"), arguments: ["action": "refresh"], requester: "Catalog regression", server: nil, progress: nil)
                #expect(result.isError != true)
            }
        }
        func checkRows() async throws {
            let rows = await f.service.rows()
            let httpRows = try JSONDecoder().decode([ModelRow].self, from: await http("models"))
            let result = try await router.callMCPTool(
                MCPNames.tool("find"), arguments: ["on_mac": "table", "format": "json"], requester: "Catalog regression", server: nil, progress: nil)
            let object = try WireJSON.decode(JSONEncoder().encode(try #require(result.structuredContent)))
            // "delete everything that is dropped". The withdrawn model's weights go and its row with them.
            #expect(object["models"]?.array?.compactMap { $0["id"]?.string } == [])
            for rows in [rows, httpRows] { #expect(rows.isEmpty) }
        }
        do {
            let before = try f.weightEvidence()
            _ = try f.publish(sequence: 2, included: true)
            try await refresh()
            // The table holds the model only while its weights are on this Mac; the catalog lists it either way.
            #expect(await f.service.rows().count == (installed ? 1 : 0))
            #expect(await f.service.catalog().first { $0.id == DroppedCatalogFixture.id }?.row.catalogRemoved == false)
            let removedCatalog = try f.publish(sequence: 3, included: false)
            try await refresh(); try await checkRows()
            #expect(installed ? try f.weightEvidence().isEmpty : try f.weightEvidence() == before)
            #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).table.isEmpty)
            // Repeated reads/refresh cannot resurrect the withdrawn row.
            try await refresh(); try await checkRows()
            #expect(await f.service.supervisor.residentIDs().isEmpty)
            await f.service.shutdown()
            let restarted = try await f.reopen(removedCatalog)
            #expect(await restarted.rows().isEmpty)
            #expect(installed ? try f.weightEvidence().isEmpty : try f.weightEvidence() == before)
            await restarted.shutdown()
        } catch { await f.service.shutdown(); throw error }
    }

    @Test(arguments: [false, true]) func upgradeWithAlreadyRemovedCatalogReconcilesAtOpen(installed: Bool) async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tmp/catalog-upgrade-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let f = try await DroppedCatalogFixture.make(root: root, installed: installed)
        await f.service.shutdown()
        let before = try f.weightEvidence(), data = try f.publish(sequence: 2, included: false)
        let service = try await f.reopen(data)
        #expect(await service.rows().isEmpty)
        #expect(installed ? try f.weightEvidence().isEmpty : try f.weightEvidence() == before)
        await service.shutdown()
    }

    @Test(arguments: [false, true]) func restartDeletesWithdrawnWeightsWithoutInstalledMetadata(knownIdentity: Bool) async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tmp/catalog-missing-metadata-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let f = try await DroppedCatalogFixture.make(root: root, installed: true)
        await f.service.shutdown()
        try FileManager.default.removeItem(at: f.paths.directory("models").appendingPathComponent(DroppedCatalogFixture.id))
        try FileManager.default.removeItem(at: f.paths.support.appendingPathComponent("state/installed-catalog/" + DroppedCatalogFixture.id + ".json"))
        if !knownIdentity {
            var config = try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json"))
            config.catalogModelIdentities = [:]
            try config.write(f.paths.support.appendingPathComponent("config.json"))
        }
        let data = try f.publish(sequence: 2, included: false)
        let service = try await f.reopen(data)
        // Missing manifests must not keep the withdrawn weights on disk.
        #expect(await service.rows().isEmpty)
        #expect(try f.weightEvidence().isEmpty)
        try await service.refresh()
        #expect(await service.rows().isEmpty)
        await service.shutdown()
    }

    @Test(arguments: [false, true]) func malformedLegacyDescriptorStillDeletesWithdrawnWeights(restart: Bool) async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tmp/catalog-malformed-descriptor-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let f = try await DroppedCatalogFixture.make(root: root, installed: true)
        try FileManager.default.removeItem(at: f.paths.support.appendingPathComponent("state/installed-catalog/" + DroppedCatalogFixture.id + ".json"))
        try Data("malformed = [".utf8).write(to: f.paths.directory("models").appendingPathComponent(DroppedCatalogFixture.id + "/model.toml"))
        let data = try f.publish(sequence: 2, included: false)
        let service: RealModelsService
        if restart { await f.service.shutdown(); service = try await f.reopen(data) } else { service = f.service; try await service.refresh() }
        #expect(await service.rows().isEmpty)
        #expect(try f.weightEvidence().isEmpty)
        await service.shutdown()
    }

    @Test(arguments: ["root", "revision"]) func unreadableSnapshotDirectoryKeepsWithdrawnRowUntilReadable(level: String) async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tmp/catalog-unreadable-snapshot-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let f = try await DroppedCatalogFixture.make(root: root, installed: true)
        let snapshots = f.paths.support.appendingPathComponent("weights/snapshots/" + DroppedCatalogFixture.id)
        let unreadable = level == "root" ? snapshots : snapshots.appendingPathComponent(String(repeating: "b", count: 40))
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: unreadable.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: unreadable.path) }
        #expect(throws: (any Error).self) { try RealModelsService.installedDtypes(id: DroppedCatalogFixture.id, support: f.paths.support) }
        #expect(RealModelsService.hasInstalledSnapshot(id: DroppedCatalogFixture.id, support: f.paths.support))
        let data = try f.publish(sequence: 2, included: false)
        try await f.service.refresh()
        let row = try #require(await f.service.rows().first)
        #expect(row.catalogRemoved && !row.local)
        #expect(row.name == f.entry.manifest.name && row.kind == "chat")
        #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).table == [DroppedCatalogFixture.id])
        await f.service.shutdown()
        let restarted = try await f.reopen(data)
        #expect(await restarted.rows().first?.catalogRemoved == true)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: unreadable.path)
        // Once the directory reads again, the next refresh deletes the withdrawn weights and forgets the row.
        try await restarted.refresh()
        #expect(await restarted.rows().isEmpty)
        #expect(try f.weightEvidence().isEmpty)
        await restarted.shutdown()
    }

    /// Weights of the five models withdrawn for their licences stay on disk until the user deletes them; any other model the
    /// catalog no longer lists is still collected.
    @Test func licenceWithdrawnModelsKeepTheirWeights() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tmp/catalog-kept-withdrawn-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let f = try await DroppedCatalogFixture.make(root: root, installed: false)
        let kept = RealModelsService.keptWithdrawnWeights.sorted()
        #expect(kept == ["nemotron-3-diarization", "sam-audio-large", "sam-audio-small", "sam3.1", "trellis2-4b"])
        #expect(await f.service.droppedWeightIDs(onDisk: kept + ["gone-model"]) == ["gone-model"])
        await f.service.shutdown()
    }

    @Test func quietRemovalForgetsPerModelStateWithoutChangingKindOrder() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tmp/catalog-forget-state-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let f = try await DroppedCatalogFixture.make(root: root, installed: false)
        try await f.service.seedWithdrawnState()
        let before = try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json"))
        #expect(before.selections[DroppedCatalogFixture.id]?.keepHot == true)
        #expect(before.settings[DroppedCatalogFixture.id]?["limit"] == 2)
        #expect(before.runnerOverrides[DroppedCatalogFixture.id] != nil)
        _ = try f.publish(sequence: 2, included: false)
        try await f.service.refresh()
        let saved = try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json"))
        #expect(saved.table.isEmpty && saved.selections.isEmpty && saved.settings.isEmpty && saved.runnerOverrides.isEmpty)
        #expect(saved.localCodeApprovals.isEmpty && saved.localCodeConsent.isEmpty && saved.rejectedRevisions.isEmpty && saved.catalogModelIdentities.isEmpty)
        #expect(saved.kindOrder == before.kindOrder)
        #expect(await f.service.withdrawnSettingsForgotten())
        _ = try f.publish(sequence: 3, included: true)
        try await f.service.refresh()
        #expect(await f.service.rows().isEmpty)
        try await f.service.add(id: DroppedCatalogFixture.id)
        // Added without weights, it waits in the catalog for its Get; its forgotten preferences stay forgotten.
        #expect(await f.service.rows().isEmpty)
        let row = try #require(await f.service.catalog().first { $0.id == DroppedCatalogFixture.id }?.row)
        #expect(!row.keepHot && row.runnerOverride == nil)
        await f.service.shutdown()
    }

    @Test(arguments: [false, true]) func realLocalFolderIsNotDroppedByCatalogReconciliation(broken: Bool) async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("tmp/catalog-local-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let f = try await DroppedCatalogFixture.make(root: root, installed: false)
        let local = f.paths.directory("local-models").appendingPathComponent(DroppedCatalogFixture.id)
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        for (path, bytes) in f.source.files {
            let destination = local.appendingPathComponent(String(path.dropFirst("models/\(DroppedCatalogFixture.id)/".count)))
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: destination)
        }
        if broken { try Data("broken".utf8).write(to: local.appendingPathComponent("model.toml")) }
        _ = try f.publish(sequence: 2, included: false)
        try await f.service.refresh()
        // A local folder without weights is only in the catalog; a broken one is on the table as its failure.
        let tableRow = await f.service.rows().first
        let catalogRow = await f.service.catalog().first { $0.id == DroppedCatalogFixture.id }?.row
        #expect((tableRow != nil) == broken)
        let row = try #require(broken ? tableRow : catalogRow)
        #expect(row.local && !row.catalogRemoved)
        #expect(row.state == (broken ? .failed : .notInstalled))
        #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).table == [DroppedCatalogFixture.id])
        await f.service.shutdown()
    }
}

private extension RealModelsService {
    func seedWithdrawnState() throws {
        let id = DroppedCatalogFixture.id
        config.selections[id] = .init(precision: "int4", keepHot: true)
        config.runnerOverrides[id] = "obsolete-runner"
        config.localCodeApprovals[id] = .init(by: .api, digest: "obsolete")
        config.localCodeConsent[id] = "obsolete"; config.rejectedRevisions[id] = "obsolete"
        config.kindOrder = ["voices", "chat"]
        modelSettings.declare(
            id,
            metadata: [
                "setting": .array([
                    .object([
                        "id": "limit", "type": "number", "default": 1, "applies": "load", "maps_to": "config.limit"
                    ])
                ])
            ])
        modelSettings.load([id: ["limit": 2]])
        modelSettings.applied[id] = ["limit": 2]
        modelSettings.aliases[id] = ["legacy": ["small": ["limit": 2]]]
        try persist()
    }
    func withdrawnSettingsForgotten() -> Bool {
        let id = DroppedCatalogFixture.id
        return modelSettings.values[id] == nil && modelSettings.applied[id] == nil && modelSettings.declarations[id] == nil && modelSettings.aliases[id] == nil
    }
}
