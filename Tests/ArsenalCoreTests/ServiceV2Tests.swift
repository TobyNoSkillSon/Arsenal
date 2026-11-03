// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private let v2Settings: [JSONObject] = [
    ["id": "context", "title": "Context", "type": "choice", "default": 16, "choices": [16, 32], "applies": "load", "maps_to": "config.context_limit"],
    ["id": "speed", "title": "Speed", "type": "slider", "default": 1, "min": 0.5, "max": 2, "step": 0.25, "applies": "request", "maps_to": "request.speed"],
    ["id": "enabled", "title": "Enabled", "type": "toggle", "default": true, "applies": "load"]
]
@Test func v2SettingsCanonicalMappingPendingAndOverrides() throws {
    var settings = ModelSettings(); settings.declarations["m"] = v2Settings
    #expect(settings.states("m", loaded: false).allSatisfy { !$0.pending })
    let load = settings.loadConfig("m", base: ["assets": [:]])
    #expect(load["settings"]?["context"] == 16 && load["context_limit"] == 16 && load["settings"]?["enabled"] == true)
    settings.applied["m"] = settings.loadValues("m")
    try settings.set("m", key: "context", value: 32)
    #expect(settings.states("m", loaded: true).filter(\.pending).map(\.id) == ["context"])
    try settings.set("m", key: "context", value: 16)
    #expect(settings.states("m", loaded: true).allSatisfy { !$0.pending })
    try settings.set("m", key: "speed", value: 1.5)
    #expect(settings.request("m", params: [:])["speed"] == 1.5)
    #expect(settings.request("m", params: ["speed": 2])["speed"] == 2)
    #expect(throws: WorkerFailure.self) { try settings.set("m", key: "speed", value: 1.6) }
    #expect(throws: WorkerFailure.self) { try settings.set("m", key: "context", value: 99) }
    #expect(throws: WorkerFailure.self) { try settings.set("m", key: "enabled", value: 1) }
    #expect(throws: WorkerFailure.self) { try settings.set("m", key: "missing", value: 1) }
}
@Test func v2StubMembershipSettingsAndOperations() async throws {
    let row = ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])
    let op = ModelOperation(
        name: "voices", title: "Voices", inputSchema: ["type": "object", "properties": ["speed": ["type": "number", "maximum": 2]]], output: "json", stream: false)
    let s = StubModelsService(rows: [row], metadata: [row.id: ["setting": .array(v2Settings.map(JSONValue.object))]], operations: [row.id: [op]])
    try await s.add(id: row.id); #expect(await s.rows().count == 1)
    try await s.load(id: row.id)
    try await s.setSetting(id: row.id, key: "context", value: 32)
    #expect(await s.rows().first?.needsReload == true)
    try await s.setSetting(id: row.id, key: "speed", value: 1.5)
    _ = try await s.run(id: row.id, op: "voices")
    #expect(await s.rows().first?.needsReload == true) // A request must not silently apply pending load settings.
    try await s.applySettings(id: row.id)
    #expect(await s.rows().first?.needsReload == false)
    #expect(try await s.settings(id: row.id).allSatisfy { !$0.pending })
    #expect(await s.operations(id: row.id) == [op])
    await #expect(throws: WorkerFailure.self) { _ = try await s.run(id: row.id, op: "voices", params: ["speed": 3]) }
    // Its weights are on this Mac, so removing it without deleting them leaves it on the table; deleting them takes it off.
    try await s.remove(id: row.id, deleteWeights: false)
    #expect(await s.rows().map(\.id) == [row.id]); #expect(await s.catalog().first?.installed == true)
    try await s.add(id: row.id); try await s.remove(id: row.id, deleteWeights: true)
    #expect(await s.catalog().first?.installed == false)
    #expect(await s.rows().isEmpty)
}
@Test func v2RealEmptyTablePersistenceAndWeightsKeepTheModelOnTheTable() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices", "images"], addToTable: false); defer { f.stop() }
    let s = f.service
    #expect(await s.rows().isEmpty); #expect(await s.catalog().count == 2)
    // Add records each model once, in order; without weights neither is on the table yet.
    try await s.add(id: "fixture-images"); try await s.add(id: "fixture-voices"); try await s.add(id: "fixture-images")
    #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).table == ["fixture-images", "fixture-voices"])
    #expect(await s.rows().isEmpty)
    try await s.get(id: "fixture-voices", precision: "bf16")
    #expect(await s.rows().map(\.id) == ["fixture-voices"])
    // Removing it without deleting the weights leaves it on the table: no weights hide in Get more models.
    try await s.remove(id: "fixture-voices", deleteWeights: false)
    #expect(await s.catalog().first { $0.id == "fixture-voices" }?.installed == true)
    #expect(await s.rows().map(\.id) == ["fixture-voices"])
    await s.shutdown()
    let reopened = try await f.reopen(threshold: 8_000_000_000)
    #expect(await reopened.rows().map(\.id) == ["fixture-voices"])
    try await reopened.add(id: "fixture-voices")
    try await reopened.remove(id: "fixture-voices", deleteWeights: true)
    #expect(await reopened.catalog().first { $0.id == "fixture-voices" }?.installed == false)
    #expect(await reopened.rows().isEmpty)
    await reopened.shutdown()
}
@Test func v2CatalogFacetsNaturalSortAndRAMFit() throws {
    let metadata: JSONObject = [
        "suggested": true, "languages": ["en"], "tasks": ["tts"], "facets": ["modalities_in": ["text"], "modalities_out": ["audio"]], "popularity": ["hf_downloads": 50]
    ]
    let large = CatalogEntry(
        row: ModelRow(id: "large", name: "Model 10", kind: "voices", params: "27B"), metadata: metadata, sizesByDtype: ["bf16": 10_000_000_000], memoryMB: 8000)
    let small = CatalogEntry(row: ModelRow(id: "small", name: "Model 2", kind: "voices", params: "82M"), metadata: metadata, sizesByDtype: ["bf16": 164_000_000], memoryMB: 8000)
    #expect(!large.fitsRAM && small.fitsRAM)
    #expect(CatalogEntry.query([large, small], .init()).map(\.id) == ["small", "large"])
    var query = CatalogQuery(text: "model", kinds: ["voices"], sort: .params); query.languages = ["en"]; query.modalitiesOut = ["audio"]; query.fitsRAM = true
    #expect(CatalogEntry.query([large, small], query).map(\.id) == ["small"])
    query.fitsRAM = nil; query.descending = true
    #expect(CatalogEntry.query([large, small], query).map(\.id) == ["large", "small"])
    #expect(try JSONDecoder().decode(CatalogEntry.self, from: JSONEncoder().encode(small)) == small)
}
@Test func v2OperationSchemaValidationErrorsAndReferences() throws {
    let schema: JSONValue = [
        "type": "object", "required": ["text"], "additionalProperties": false,
        "properties": ["text": ["$ref": "#/$defs/text"], "speed": ["type": "number", "minimum": 0.5, "maximum": 2]], "$defs": ["text": ["type": "string", "minLength": 1]]
    ]
    try OperationSchemaValidator.validate(["text": "Hi", "speed": 1], schema: schema)
    let invalid: [JSONValue] = [[:], ["text": ""], ["text": 1], ["text": "Hi", "speed": true], ["text": "Hi", "unknown": 1]]
    for value in invalid {
        #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate(value, schema: schema) }
    }
    #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate("x", schema: ["$ref": "https://example.com/schema"]) }
    #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate("x", schema: ["unevaluatedProperties": false]) }
    try OperationSchemaValidator.validate("x", schema: ["oneOf": [["type": "string"], ["type": "number"]]])
    #expect(throws: WorkerFailure.self) { try OperationSchemaValidator.validate(true, schema: ["oneOf": [["type": "string"], ["type": "number"]]]) }
}

private func fixtureV2Table() -> [String: TOMLValue] {
    let declarations = v2Settings.map {
        $0.mapValues { value -> TOMLValue in
            switch value {
            case .string(let v): return .string(v)
            case .number(let v): return .decimal(v)
            case .bool(let v): return .bool(v)
            case .array(let vs): return .array(vs.map { .decimal($0.number!) })
            default: return .string("")
            }
        }
    }
    let ops = ["test_config", "test_echo"].map { name -> TOMLValue in
        .table(["name": .string(name), "title": .string(name), "input": .string("ops/\(name).input.json"), "output": .string("json"), "stream": .bool(false)])
    }
    return ["schema": .integer(2), "selection": .table(["default": .string("native")]), "setting": .array(declarations.map(TOMLValue.table)), "op": .array(ops)]
}
private let fixtureV2Files: [String: Data] = [
    "ops/test_config.input.json": Data(#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}"#.utf8),
    "ops/test_echo.input.json": Data(
        #"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["text"],"properties":{"text":{"type":"string","minLength":1},"speed":{"type":"number","minimum":0.5,"maximum":2}}}"#
            .utf8),
    "kernels/manifest.toml": Data(
        """
        [[kernel]]
        id = "fixture"
        chip = "*"
        mode = "exact"
        api = "metal"
        header = "test.metal"
        files = ["test.metal"]
        entry = "test"
        replaces = "fixture"
        """.utf8),
    "kernels/test.metal": Data("// CPU fixture only\n".utf8)
]
@Test func v2RealSettingsDeliveryReloadRequestDefaultsAndKernels() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: fixtureV2Table(), extraFiles: fixtureV2Files); defer { f.stop() }
    let s = f.service, id = "fixture-voices"
    #expect(await s.operations(id: id).map(\.name) == ["test_config", "test_echo"])
    // Validation precedes on-demand loading: a malformed request cannot cause any model work.
    do { _ = try await s.run(id: id, op: "test_echo", params: [:]); Issue.record("Invalid request accepted") } catch let error as WorkerFailure {
        #expect(error.code == "invalid_request")
    }
    #expect(await s.memory().models.isEmpty)
    try await s.get(id: id, precision: "bf16"); try await s.load(id: id, precision: nil, fast: false)
    let initial = try await s.run(id: id, op: "test_config")
    #expect(initial["config"]?["settings"]?["context"] == 16 && initial["config"]?["context_limit"] == 16)
    let kernel = try #require(initial["config"]?["kernels"]?.array?.first)
    #expect(kernel["files"]?.array?.first?.string?.hasPrefix("/") == true)
    #expect(kernel["header"]?.string?.hasPrefix("/") == true && kernel["api"] == "metal")
    #expect(await s.rows().first?.kernels.first?["state"] == "stock")
    try await s.setSetting(id: id, key: "context", value: 32)
    #expect(await s.rows().first?.needsReload == true)
    #expect(try await s.settings(id: id).filter(\.pending).map(\.id) == ["context"])
    try await s.setSetting(id: id, key: "speed", value: 1.5)
    #expect(try await s.run(id: id, op: "test_echo", params: ["text": "hello"])["params"]?["speed"] == 1.5)
    #expect(try await s.run(id: id, op: "test_echo", params: ["text": "hello", "speed": 2])["params"]?["speed"] == 2)
    #expect(try await s.run(id: id, op: "test_config")["config"]?["context_limit"] == 16)
    try await s.applySettings(id: id)
    #expect(await s.rows().first?.needsReload == false)
    #expect(try await s.run(id: id, op: "test_config")["config"]?["settings"]?["context"] == 32)
    try await s.unload(id: id); #expect(await s.rows().first?.kernels.isEmpty == true)
    await s.shutdown()
    let reopened = try await f.reopen(threshold: 8_000_000_000)
    #expect(try await reopened.settings(id: id).first { $0.id == "context" }?.value == 32)
    #expect(try await reopened.settings(id: id).allSatisfy { !$0.pending })
    await reopened.shutdown()
}

@Test func v2NativeArtifactCreatesNoPythonEnvironment() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }; defer { Task { await f.service.shutdown() } }
    var model = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData).models[0].manifest
    model.runner[0].language = "rust"
    let helpers = f.paths.support.appendingPathComponent("helpers"); try FileManager.default.createDirectory(at: helpers, withIntermediateDirectories: true)
    let helper = helpers.appendingPathComponent("arsenal-worker-stub")
    try Data("#!/bin/sh\nexit 0\n".utf8).write(to: helper); try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helper.path)
    let artifacts = BundledWorkerArtifact(helpers: helpers)
    try await artifacts.prepare(model: model, runner: model.runner[0], directory: f.paths.support)
    let launch = try await artifacts.launch(model: model, runner: model.runner[0], directory: f.paths.support, paths: f.paths)
    #expect(launch.executable == helper && launch.arguments.isEmpty)
    #expect(!FileManager.default.fileExists(atPath: f.paths.support.appendingPathComponent("tools/uv").path))
    #expect(!FileManager.default.fileExists(atPath: f.paths.directory("envs").path))
    await f.service.shutdown()
}

@Test func v2RequestSettingsScopedToSchemaAndEmptyReferenceOmitted() throws {
    var settings = ModelSettings()
    settings.declarations["voice"] = v2Settings + [["id": "reference_path", "type": "text", "default": "", "applies": "request", "maps_to": "request.reference.path"]]
    let voices: JSONObject = ["type": "object", "additionalProperties": false, "properties": [:]]
    #expect(settings.request("voice", params: [:], schema: voices).isEmpty)
    let speak: JSONObject = [
        "type": "object", "properties": ["speed": ["type": "number"], "reference": ["type": "object", "properties": ["path": ["type": "string", "minLength": 1]]]]
    ]
    let params = settings.request("voice", params: [:], schema: speak)
    #expect(params["speed"] == 1 && params["reference"] == nil)
    try settings.set("voice", key: "reference_path", value: "/tmp/test.wav")
    #expect(settings.request("voice", params: [:], schema: speak)["reference"]?["path"] == "/tmp/test.wav")
    #expect(settings.request("voice", params: ["reference": ["path": "/tmp/caller.wav"]], schema: speak)["reference"]?["path"] == "/tmp/caller.wav")
    #expect(settings.request("voice", params: ["unknown": true], schema: voices)["unknown"] == true)
}
@Test func v2FolderOpsAreAuthoritativeNotKindUnion() async throws {
    let standards = [OpDeclaration(name: "design_voice", title: "Design", input: "kinds/voices/ops/design.input.json", output: "json", stream: false)]
    let kinds = [KindSpec(id: "voices", title: "Voices", order: 0, columns: [], operations: standards)]
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: fixtureV2Table(), extraFiles: fixtureV2Files, kindSpecs: kinds); defer { f.stop() }
    #expect(await f.service.operations(id: "fixture-voices").map(\.name) == ["test_config", "test_echo"])
    await #expect(throws: WorkerFailure.self) { _ = try await f.service.run(id: "fixture-voices", op: "design_voice", params: [:]) }
    await f.service.shutdown()
}

@Test func v2LocalFolderConsentDigestReapprovalAndImmutableCode() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], addToTable: false); defer { f.stop() }
    let id = "local-voices", local = f.paths.support.appendingPathComponent("local-models/" + id)
    for (path, data) in f.source.files {
        let relative = String(path.dropFirst("models/fixture-voices/".count)); let target = local.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        let bytes =
            relative == "model.toml"
            ? Data(
                String(decoding: data, as: UTF8.self).replacingOccurrences(of: "fixture-voices", with: id).replacingOccurrences(
                    of: "language = \"swift\"", with: "language = \"rust\""
                ).utf8) : data
        try bytes.write(to: target)
    }
    let entry = try #require(await f.service.catalog().first { $0.id == id })
    #expect(entry.local && entry.row.needsCodeConsent)
    let digest = try #require(entry.row.contentDigest)
    try await f.service.add(id: id)
    do { try await f.service.get(id: id, precision: "bf16"); Issue.record("Unapproved code got weights") } catch let e as WorkerFailure { #expect(e.code == "code_consent") }
    try await f.service.approveLocalCode(id: id, contentDigest: digest)
    try await f.service.get(id: id, precision: "bf16"); try await f.service.load(id: id)
    #expect(await f.service.rows().first?.needsCodeConsent == false)
    try await f.service.unload(id: id)
    let discovered = try LocalModelFolders.discover(support: f.paths.support, signedIDs: ["fixture-voices"]).first!
    let snapshot = try LocalCodeSnapshot.install(discovered, support: f.paths.support)
    let original = try Data(contentsOf: snapshot.appendingPathComponent("README.md"))
    try Data("Changed local code.".utf8).write(to: local.appendingPathComponent("README.md"))
    #expect(try Data(contentsOf: snapshot.appendingPathComponent("README.md")) == original)
    #expect(await f.service.catalog().first { $0.id == id }?.row.needsCodeConsent == true)
    await #expect(throws: WorkerFailure.self) { try await f.service.approveLocalCode(id: id, contentDigest: digest) }
    await #expect(throws: WorkerFailure.self) { try await f.service.load(id: id) }
    await f.service.shutdown()
    let reopened = try await f.reopen(threshold: 8_000_000_000)
    #expect(await reopened.catalog().first { $0.id == id }?.row.needsCodeConsent == true)
    try await reopened.approveLocalCode(id: id)
    #expect(await reopened.rows().first?.needsCodeConsent == false)
    await reopened.shutdown()
    let approved = try await f.reopen(threshold: 8_000_000_000)
    #expect(await approved.catalog().first { $0.id == id }?.row.needsCodeConsent == false)
    await approved.shutdown()
}
