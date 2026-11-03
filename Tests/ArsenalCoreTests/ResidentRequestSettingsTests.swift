// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import ArsenalCore

/// A loaded model keeps its load-time settings until Reload, but request-time settings never need a reload: after a
/// Refresh changes the folder's declarations, requests to the still-loaded model follow the live ones (values, defaults
/// and the unset meaning). Driven through a real service, a signed catalog update, Refresh and the stub worker's echo.
private actor MutableSource: CatalogSource {
    var files: [String: Data]
    init(_ files: [String: Data]) { self.files = files }
    func read(path: String, commit: String) throws -> Data {
        guard let data = files[path] else { throw ManifestError.invalid("missing fixture file") }; return data
    }
    func replace(path: String, data: Data) { files[path] = data }
}

struct ResidentRequestSettingsTests {
    static func toml(_ value: JSONValue) -> TOMLValue {
        switch value {
        case .string(let v): return .string(v)
        case .number(let n): return n == n.rounded() && abs(n) < 1e15 ? .integer(Int64(n)) : .decimal(n)
        case .bool(let b): return .bool(b)
        case .array(let a): return .array(a.map(toml))
        case .object(let o): return .table(o.mapValues(toml))
        case .null: return .string("")
        }
    }
    static let echo = toml(["name": "test_echo", "title": "Echo", "input": "ops/test_echo.input.json", "output": "json", "stream": false])
    static let schema = Data(
        #"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"seed":{"type":"integer","minimum":0},"max_tokens":{"type":"integer","minimum":1},"top_p":{"type":"number"}}}"#
            .utf8)
    static func setting(_ fields: JSONObject) -> TOMLValue { toml(.object(fields.merging(["applies": "request"]) { a, _ in a })) }
    static let seedOld = setting(["id": "seed", "title": "Seed", "type": "number", "default": 0, "min": 0, "max": 4_294_967_295, "step": 1, "maps_to": "request.seed"])
    static let seedNew = setting(["id": "seed", "title": "Seed", "type": "number", "min": 0, "max": 4_294_967_295, "step": 1, "maps_to": "request.seed"])
    static let maxTokens = setting([
        "id": "max_tokens", "title": "Answer limit", "type": "number", "default": 1024, "min": 1, "max": 8192, "step": 1, "maps_to": "request.max_tokens"
    ])
    static let topP = setting(["id": "top_p", "title": "Top P", "type": "slider", "default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05, "maps_to": "request.top_p"])

    /// Loads the model with `old` settings, publishes a catalog whose model declares `new`, refreshes, and returns the
    /// service with the model still resident on its old snapshot.
    func refreshed(old: [TOMLValue], new: [TOMLValue]) async throws -> (RealModelsService, LocalCatalogFixture) {
        let base: [String: TOMLValue] = ["schema": .integer(2), "selection": .table(["default": .string("native")]), "op": .array([Self.echo])]
        let f = try await makeLocalCatalogFixture(
            kinds: ["images"], extraTable: base.merging(["setting": .array(old)]) { _, b in b }, extraFiles: ["ops/test_echo.input.json": Self.schema])
        await f.service.shutdown()
        let key = Curve25519.Signing.PrivateKey(), source = MutableSource(f.source.files)
        let s = try await RealModelsService.open(
            catalogData: f.catalogData, signature: key.signature(for: f.catalogData), publicKey: key.publicKey.rawRepresentation, source: source, kinds: KindSpec.defaults,
            paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
        let id = "fixture-images"
        try await s.get(id: id, precision: "bf16"); try await s.load(id: id, precision: "bf16", fast: false)
        // The new folder: same model, new request declarations.
        let path = "models/\(id)/model.toml"
        var parser = TOMLParser(String(decoding: try #require(f.source.files[path]), as: UTF8.self))
        var table = try parser.parse(); table["setting"] = .array(new)
        let updated = Data(TOMLWriter.encode(table).utf8)
        await source.replace(path: path, data: updated)
        var doc = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData)
        doc.sequence += 1; doc.sourceCommit = String(repeating: "b", count: 40)
        doc.models[0].manifest = try TOMLParser.decode(ModelManifest.self, from: String(decoding: updated, as: UTF8.self))
        let index = try #require(doc.models[0].files.firstIndex { $0.path == "model.toml" })
        doc.models[0].files[index] = CatalogFile(path: "model.toml", sha256: FileDigest.sha256(updated), bytes: Int64(updated.count))
        let data = try doc.encoded(), directory = f.paths.directory("catalog")
        try data.write(to: directory.appendingPathComponent("catalog.json")); try key.signature(for: data).write(to: directory.appendingPathComponent("catalog.json.sig"))
        try await s.refresh()
        #expect(await s.rows().first?.states["bf16"]?.state == .loaded)
        return (s, f)
    }
    func echoed(_ s: RealModelsService) async throws -> JSONObject {
        try #require(try await s.run(id: "fixture-images", op: "test_echo")["params"]?.object)
    }

    /// FLUX's seed lost its default 0 (unset = random): the still-loaded model gets no seed, and a pinned seed that is
    /// cleared goes back to none.
    @Test func anUnsetSeedReachesTheResidentAsNone() async throws {
        let (s, f) = try await refreshed(old: [Self.seedOld], new: [Self.seedNew]); defer { f.stop() }
        #expect(try await echoed(s)["seed"] == nil)
        try await s.setSetting(id: "fixture-images", key: "seed", value: 421_337)
        #expect(try await echoed(s)["seed"] == 421_337)
        try await s.setSetting(id: "fixture-images", key: "seed", value: .null)
        #expect(try await echoed(s)["seed"] == nil)
        await s.shutdown()
    }

    /// Qwen gained Answer limit and Top P: saving them reaches the still-loaded model, with no Reload in between.
    @Test func newRequestSettingsReachTheResident() async throws {
        let (s, f) = try await refreshed(old: [], new: [Self.maxTokens, Self.topP]); defer { f.stop() }
        #expect(try await echoed(s)["max_tokens"] == 1024)
        try await s.setSetting(id: "fixture-images", key: "max_tokens", value: 128)
        try await s.setSetting(id: "fixture-images", key: "top_p", value: 0.9)
        let params = try await echoed(s)
        #expect(params["max_tokens"] == 128 && params["top_p"] == 0.9)
        await s.shutdown()
    }
}
