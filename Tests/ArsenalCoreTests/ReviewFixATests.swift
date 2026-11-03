// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct ReviewFixATests {
    @Test func explicitVoiceAndReferenceOwnTheWholeGroup() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for id in ["pocket-tts", "qwen3-tts-base", "breeze-tts-2"] {
            let model = try ModelManifest.load(shippedOrRetiredFolder(repo, id), kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
            var settings = ModelSettings()
            let data = try WireJSON.decode(JSONEncoder().encode(model))
            settings.declarations[id] = data["setting"]!.array!.compactMap(\.object)
            try settings.set(id, key: "reference_path", value: "/default.wav")
            if settings.declarations[id]!.contains(where: { $0["id"] == "reference_text" }) { try settings.set(id, key: "reference_text", value: "default transcript") }
            #expect(settings.request(id, params: ["voice": "alba"])["reference"] == nil)
            #expect(settings.request(id, params: ["reference": ["path": "/mine.wav"]])["reference"] == ["path": "/mine.wav"])
            #expect(settings.request(id, params: [:])["reference"]?["path"] == "/default.wav")
        }
    }
    @Test func staleSavedChoiceFallsBack() {
        var settings = ModelSettings()
        settings.declarations["m"] = [
            ["id": "context", "title": "Context", "type": "choice", "default": 8192, "choices": [8192, 32768], "applies": "load", "maps_to": "config.context_limit"]
        ]
        settings.load(["m": ["context": 131072]])
        #expect(settings.loadConfig("m", base: [:])["context_limit"] == 8192)
    }
    @Test func badLocalEntryDoesNotHideGoodFolder() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let root = f.paths.support.appendingPathComponent("local-models")
        let folder = root.appendingPathComponent("local-voices")
        for (path, data) in f.source.files {
            let relative = String(path.dropFirst("models/fixture-voices/".count)), target = folder.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try (relative == "model.toml" ? Data(String(decoding: data, as: UTF8.self).replacingOccurrences(of: "fixture-voices", with: "local-voices").utf8) : data).write(
                to: target)
        }
        try FileManager.default.createDirectory(at: root.appendingPathComponent("stray"), withIntermediateDirectories: true)
        #expect(await f.service.catalog().contains { $0.id == "local-voices" })
        await f.service.shutdown()
    }
    @Test func liveAndAdmissionHaveSameWideDAGBudget() throws {
        var defs: JSONObject = ["leaf": ["type": "object"]]
        for level in 0..<4 { defs["d\(level)"] = ["allOf": .array(Array(repeating: ["$ref": .string("#/$defs/" + (level == 0 ? "leaf" : "d\(level-1)"))], count: 10))] }
        let schema: JSONObject = ["$defs": .object(defs), "$ref": "#/$defs/d3"]
        let errors = try ModelSchemaValidation.validate(value: [String: Any](), schemaData: WireJSON.encode(schema))
        #expect(errors.contains { $0.contains("budget") })
        #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate([:], schema: .object(schema)) }
    }
}

private func reviewBundle(_ fixture: LocalCatalogFixture) throws -> URL {
    let resources = fixture.paths.support.appendingPathComponent("resources")
    let cat = resources.appendingPathComponent("catalog")
    try FileManager.default.createDirectory(at: cat, withIntermediateDirectories: true)
    try fixture.catalogData.write(to: cat.appendingPathComponent("catalog.json"))
    try fixture.signature.write(to: cat.appendingPathComponent("catalog.json.sig"))
    try fixture.publicKey.write(to: resources.appendingPathComponent("catalog-public-key"))
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    try FileManager.default.copyItem(at: repo.appendingPathComponent("Resources/kinds"), to: resources.appendingPathComponent("kinds"))
    for (path, data) in fixture.source.files {
        let file = resources.appendingPathComponent(path); try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: file)
    }
    return resources
}
@Test func reviewFixLegacyCatalogMigratesAndMissingOverrideFallsBack() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }; await f.service.shutdown()
    let resources = try reviewBundle(f)
    let config = f.paths.support.appendingPathComponent("config.json")
    try WireJSON.encode(["catalog": ["source": ["path": .string(LocalGitCatalogSource.defaultRepository.path)]]]).write(to: config)
    let upgraded = try await RealModelsService.open(repository: resources, resources: resources, paths: f.paths, artifacts: StubWorkerArtifact())
    #expect(await upgraded.catalogStatus().source == "bundle")
    #expect(try ArsenalConfiguration.read(config).catalog["source"] == "bundle")
    await upgraded.shutdown()
    try WireJSON.encode(["catalog": ["source": ["path": "/missing/user-override"]]]).write(to: config)
    let fallback = try await RealModelsService.open(repository: resources, resources: resources, paths: f.paths, artifacts: StubWorkerArtifact())
    #expect(await fallback.catalogStatus().source == "bundle")
    #expect(await fallback.catalogStatus().error?.contains("Using bundled catalog") == true)
    #expect(try ArsenalConfiguration.read(config).catalog["source"]?["path"] == "/missing/user-override")
    await fallback.shutdown()
}
@Test func reviewFixV1TableMigratesInstalledWeightsOnce() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], addToTable: false); defer { f.stop() }
    try await f.service.get(id: "fixture-voices", precision: "bf16"); await f.service.shutdown()
    let config = f.paths.support.appendingPathComponent("config.json")
    try WireJSON.encode(["selections": ["fixture-voices": ["precision": "bf16", "fast": true, "keepHot": false]]]).write(to: config)
    let upgraded = try await f.reopen(threshold: 8_000_000_000)
    #expect(await upgraded.rows().map(\.id) == ["fixture-voices"])
    try await upgraded.remove(id: "fixture-voices"); await upgraded.shutdown()
    let next = try await f.reopen(threshold: 8_000_000_000)
    // The migration ran once: the removed record is not written back. Its weights still keep it on the table.
    #expect(try ArsenalConfiguration.read(config).table.isEmpty)
    #expect(await next.rows().map(\.id) == ["fixture-voices"]); await next.shutdown()
}

@Test func reviewFixSynthesizedDeleteDoesNotStartWorker() async throws {
    let op: TOMLValue = .table(["name": .string("clone_voice"), "title": .string("Clone"), "input": .string("ops/clone.json"), "output": .string("json"), "stream": .bool(false)])
    let schema = Data(#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}"#.utf8)
    let f = try await makeLocalCatalogFixture(
        kinds: ["voices"], artifacts: ProtocolFixtureArtifact(scenario: "normal"), extraTable: ["op": .array([op])], extraFiles: ["ops/clone.json": schema])
    defer { f.stop() }
    #expect(await f.service.operations(id: "fixture-voices").contains { $0.name == "delete_voice" })
    let dir = f.paths.support.appendingPathComponent("voices/fixture-voices"), voice = "voice-" + UUID().uuidString.lowercased()
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try WireJSON.encode(["id": .string(voice)]).write(to: dir.appendingPathComponent(voice + ".json"))
    #expect(try await f.service.run(id: "fixture-voices", op: "delete_voice", params: ["voice_id": .string(voice)])["deleted"] == .string(voice))
    #expect(fixturePID(f.paths) == nil)
    #expect(await f.service.supervisor.residentIDs().isEmpty)
    await f.service.shutdown()
}
@Test func reviewFixLocalScannerOnlyRehashesChangedIdentity() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-a/cache-\(UUID().uuidString)")
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let folder = root.appendingPathComponent("local-models/qwen3.8-27b")
    try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: repo.appendingPathComponent("models/qwen3.8-27b"), to: folder)
    let scanner = LocalFolderScanner()
    let first = try await scanner.scan(support: root, signedIDs: [], kindsDirectory: repo.appendingPathComponent("Resources/kinds")), reads = await scanner.hashReads
    _ = try await scanner.scan(support: root, signedIDs: [], kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    #expect(await scanner.hashReads == reads)
    try Data("changed".utf8).write(to: folder.appendingPathComponent("README.md"))
    let changed = try await scanner.scan(support: root, signedIDs: [], kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    #expect(await scanner.hashReads == reads + 1)
    #expect(first.folders.first?.contentDigest != changed.folders.first?.contentDigest)
}
@Test func reviewFixSnapshotRetentionAndTransientExpiryHaveReceipts() throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-a/retention-\(UUID().uuidString)")
    for n in 0..<12 {
        let dir = root.appendingPathComponent("state/local-folders/" + String(format: "%064x", n))
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true); try Data([1]).write(to: dir.appendingPathComponent("code.py"))
    }
    let protected = String(format: "%064x", 0)
    try LocalCodeSnapshot.prune(support: root, protecting: [protected], limit: 2)
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("state/local-folders").path).count == 3)
    #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("state/local-folders/" + protected).path))
    let job = root.appendingPathComponent("jobs/" + UUID().uuidString)
    try FileManager.default.createDirectory(at: job, withIntermediateDirectories: true); try Data([2]).write(to: job.appendingPathComponent("preview.wav"))
    try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 0)], ofItemAtPath: job.path)
    try TransientModelJobs.prune(support: root)
    #expect(!FileManager.default.fileExists(atPath: job.path))
    #expect(try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("logs/cleanup-receipts").path).count == 10)
}
@Test func reviewFixUnbundledEngineRefusesBeforeConsentAndThinkingDefaultCounts() throws {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let model = try ModelManifest.load(repo.appendingPathComponent("models/qwen3.8-27b"))
    let paths = ArsenalPaths(support: URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-a/no-engine-\(UUID().uuidString)"))
    let artifact = BundledWorkerArtifact(helpers: paths.support)
    do { try artifact.validateEngine(model: model, runner: model.runners[0]); Issue.record("Missing engine accepted") } catch let error as WorkerFailure {
        #expect(error.code == "worker_missing" && !error.message.contains("reinstall"))
    }
    var settings = ModelSettings()
    settings.declarations[model.id] = try WireJSON.decode(JSONEncoder().encode(model))["setting"]!.array!.compactMap(\.object)
    try settings.set(model.id, key: "thinking", value: true)
    let schema = try WireJSON.decode(Data(contentsOf: repo.appendingPathComponent("models/qwen3.8-27b/ops/count_tokens.input.json")))
    #expect(settings.request(model.id, params: ["text": "hi"], schema: schema)["thinking"] == true)
    #expect(settings.request(model.id, params: ["text": "hi", "thinking": false], schema: schema)["thinking"] == false)
}

@Test func reviewFixInvalidSavedSettingWarnsOnRowAndUsesDefault() async throws {
    var configuration = ArsenalConfiguration(); configuration.settings = ["fixture-voices": ["context": 131072]]
    let setting: TOMLValue = .table([
        "id": .string("context"), "title": .string("Context"), "type": .string("choice"), "default": .integer(8192), "choices": .array([.integer(8192), .integer(32768)]),
        "applies": .string("load"), "maps_to": .string("config.context_limit")
    ])
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], configuration: configuration, extraTable: ["setting": .array([setting])]); defer { f.stop() }
    #expect(try await f.service.settings(id: "fixture-voices").first?.value == 8192)
    #expect(await catalogRow(f.service, "fixture-voices")?.settingsWarning?.contains("using the model default") == true)
    await f.service.shutdown()
}
@Test func reviewFixVeryWideDAGAndUniqueItemsBoundTotalWork() throws {
    var defs: JSONObject = ["leaf": ["type": "object"]]
    for level in 0..<4 { defs["d\(level)"] = ["allOf": .array(Array(repeating: ["$ref": .string("#/$defs/" + (level == 0 ? "leaf" : "d\(level-1)"))], count: 100))] }
    let schema: JSONObject = ["$defs": .object(defs), "$ref": "#/$defs/d3"]
    try ModelSchemaValidation.checkSchema(schemaData: WireJSON.encode(schema))
    #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate([:], schema: .object(schema)) }
    let values = JSONValue.array((0..<20_000).map { .number(Double($0)) })
    // Linear canonicalisation is allowed for a large input; repeated schema work is still bounded.
    try OperationSchemaValidator.validate(values, schema: ["type": "array", "uniqueItems": true])
    let amplified: JSONValue = ["type": "array", "allOf": .array(Array(repeating: ["uniqueItems": true], count: 32))]
    #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate(values, schema: amplified) }
}

@Test func reviewFixV1MigrationIncludesOldRevisionWeights() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], addToTable: false); defer { f.stop() }; await f.service.shutdown()
    let old = f.paths.support.appendingPathComponent("weights/snapshots/fixture-voices/" + String(repeating: "b", count: 40) + "/int4")
    try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: old.appendingPathComponent("arsenal-snapshot.json"))
    try WireJSON.encode(["catalog": [:]]).write(to: f.paths.support.appendingPathComponent("config.json"))
    let upgraded = try await f.reopen(threshold: 8_000_000_000)
    #expect(await upgraded.rows().map(\.id) == ["fixture-voices"])
    await upgraded.shutdown()
}

@Test func reviewFixQuitSweepsExpiredPreviewsAfterDrain() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
    let expired = f.paths.directory("jobs").appendingPathComponent(UUID().uuidString), recent = f.paths.directory("jobs").appendingPathComponent(UUID().uuidString)
    for dir in [expired, recent] {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true); try Data([1]).write(to: dir.appendingPathComponent("preview.wav"))
    }
    try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 0)], ofItemAtPath: expired.path)
    await f.service.shutdown()
    #expect(!FileManager.default.fileExists(atPath: expired.path))
    #expect(FileManager.default.fileExists(atPath: recent.path))
    #expect(await f.service.supervisor.residentIDs().isEmpty)
}

@Test func reviewFixUnbundledLocalEngineIsRejectedBeforeApprovalOrGet() async throws {
    let absent = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-a/absent-engine-\(UUID().uuidString)")
    let artifacts = BundledWorkerArtifact(helpers: absent)
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], artifacts: artifacts); defer { f.stop() }
    let local = f.paths.support.appendingPathComponent("local-models/local-voices")
    for (path, data) in f.source.files {
        let relative = String(path.dropFirst("models/fixture-voices/".count)), file = local.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let bytes =
            relative == "model.toml"
            ? Data(
                String(decoding: data, as: UTF8.self).replacingOccurrences(of: "fixture-voices", with: "local-voices").replacingOccurrences(
                    of: "language = \"swift\"", with: "language = \"rust\""
                ).utf8) : data
        try bytes.write(to: file)
    }
    let entry = try #require(await f.service.catalog().first { $0.id == "local-voices" })
    #expect(!entry.row.needsCodeConsent)
    #expect(entry.row.verification == "This model needs an engine that is not in this Arsenal build")
    for operation in ["approve", "get"] {
        do {
            if operation == "approve" {
                try await f.service.approveLocalCode(id: entry.id, contentDigest: entry.row.contentDigest!, approvedBy: .app)
            } else {
                try await f.service.get(id: entry.id, precision: "bf16")
            }
            Issue.record("Unbundled engine was admitted")
        } catch let error as WorkerFailure { #expect(error.code == "worker_missing") }
    }
    #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).localCodeApprovals[entry.id] == nil)
    await f.service.shutdown()
}

@Test func reviewFixEnumComparisonChargesRepeatedPayloadWork() throws {
    let value = JSONValue.string(String(repeating: "x", count: 64_000))
    let schema: JSONValue = ["enum": .array((0..<100).map { .string("choice-\($0)") })]
    do { try OperationSchemaValidator.validate(value, schema: schema); Issue.record("Large repeated comparison accepted") } catch let error as WorkerFailure {
        #expect(error.message.contains("budget"))
    }
}
