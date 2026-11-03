// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// Signed, weight-free kind inventory: restart and refresh use the production service and config file.
@MainActor final class OrderRestartFixture {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".build/round7/fixtures/" + UUID().uuidString)
    private let key = Curve25519.Signing.PrivateKey()
    private var sequence: Int64 = 0
    var paths: ArsenalPaths { ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache")) }
    var resources: URL { root.appendingPathComponent("Resources") }
    var configFile: URL { paths.support.appendingPathComponent("config.json") }
    deinit { try? FileManager.default.removeItem(at: root) }

    func inventory(_ ids: [String]) throws {
        let kinds = resources.appendingPathComponent("kinds")
        if FileManager.default.fileExists(atPath: kinds.path) { try FileManager.default.removeItem(at: kinds) }
        try FileManager.default.createDirectory(at: kinds, withIntermediateDirectories: true)
        let files = try ids.map { id in
            let rank = id == "c" ? -10 : (id == "a" ? 10 : 20)
            let bytes = Data("id=\"\(id)\"\ntitle=\"\(id.uppercased())\"\norder=\(rank)\ncolumns=[]\napi={}\nsuite=\"fixture\"\n".utf8)
            try bytes.write(to: kinds.appendingPathComponent(id + ".toml"))
            return CatalogFile(path: id + ".toml", sha256: FileDigest.sha256(bytes), bytes: Int64(bytes.count))
        }
        sequence += 1
        let commit = String(format: "%040llx", sequence)
        let document = CatalogDocument(sequence: sequence, sourceCommit: commit, models: [], kindFiles: files)
        let data = try document.encoded(), catalog = resources.appendingPathComponent("catalog")
        try FileManager.default.createDirectory(at: catalog, withIntermediateDirectories: true)
        try data.write(to: catalog.appendingPathComponent("catalog.json"))
        try key.signature(for: data).write(to: catalog.appendingPathComponent("catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
    }
    func open() async throws -> RealModelsService {
        if !FileManager.default.fileExists(atPath: configFile.path) {
            var config = ArsenalConfiguration(); config.catalog = ["source": "bundle"]
            try config.write(configFile)
        }
        let catalog = resources.appendingPathComponent("catalog")
        let data = try Data(contentsOf: catalog.appendingPathComponent("catalog.json"))
        let document = try JSONDecoder().decode(CatalogDocument.self, from: data)
        return try await RealModelsService.open(
            catalogData: data, signature: Data(contentsOf: catalog.appendingPathComponent("catalog.json.sig")), publicKey: key.publicKey.rawRepresentation,
            source: BundleCatalogSource(resources: resources, catalog: document), kinds: KindSpec.load(directory: resources.appendingPathComponent("kinds")), paths: paths)
    }
    func saved() throws -> [String] { try ArsenalConfiguration.read(configFile).kindOrder }
}

@MainActor @Suite(.menuState, .serialized) struct TableOrderRestartTests {
    @Test func absentRememberedKindReturnsToItsPositionAfterRestart() async throws {
        let fixture = OrderRestartFixture()
        try fixture.inventory(["a", "b", "c"])
        let first = try await fixture.open()
        try await first.setKindOrder(["c", "a", "b"])
        await first.shutdown()
        try fixture.inventory(["a", "b"])
        let restarted = try await fixture.open(), table = TableController(service: restarted)
        await table.refreshRows() // The initial progress path must also read the raw persistence seed.
        #expect(table.groupOrder == ["a", "b"])
        #expect(await restarted.persistedKindOrder() == ["c", "a", "b"])
        #expect(try fixture.saved() == ["c", "a", "b"])
        try fixture.inventory(["a", "b", "c"])
        try await restarted.refresh(); await table.refresh(); await table.refreshRows()
        #expect(table.groupOrder == ["c", "a", "b"] && table.kinds.map(\.id) == ["c", "a", "b"])
        #expect(await restarted.kinds().map(\.id) == ["c", "a", "b"])
        #expect(try fixture.saved() == ["c", "a", "b"])
        await restarted.shutdown()
    }

    @Test func newLowerRankKindAppendsToSavedPositionsAfterRestart() async throws {
        let fixture = OrderRestartFixture()
        try fixture.inventory(["a", "b"])
        let first = try await fixture.open()
        try await first.setKindOrder(["b", "a"])
        await first.shutdown()
        try fixture.inventory(["a", "b", "c"])
        let restarted = try await fixture.open(), table = TableController(service: restarted)
        await table.refresh(); await table.refreshRows(); await table.refresh()
        #expect(table.groupOrder == ["b", "a", "c"] && table.kinds.map(\.id) == ["b", "a", "c"])
        #expect(await restarted.kinds().map(\.id) == ["b", "a", "c"])
        #expect(await restarted.persistedKindOrder() == ["b", "a"])
        #expect(try fixture.saved() == ["b", "a"])
        await restarted.shutdown()
    }

    @Test func persistenceAcceptsOrOmitsAbsentKindsAndRejectsDuplicates() async throws {
        let fixture = OrderRestartFixture()
        try fixture.inventory(["a", "b", "c"])
        let first = try await fixture.open()
        try await first.setKindOrder(["c", "a", "b"])
        await first.shutdown()
        try fixture.inventory(["a", "b"])
        let restarted = try await fixture.open()
        try await restarted.setKindOrder(["c", "b", "a"])
        #expect(try fixture.saved() == ["c", "b", "a"])
        try await restarted.setKindOrder(["b", "a"])
        #expect(try fixture.saved() == ["b", "a"])
        await #expect(throws: WorkerFailure.self) { try await restarted.setKindOrder(["b", "b", "a"]) }
        await #expect(throws: WorkerFailure.self) { try await restarted.setKindOrder(["b", "local-unknown"]) }
        await #expect(throws: WorkerFailure.self) { try await restarted.setKindOrder(["b", "Local models"]) }
        #expect(try fixture.saved() == ["b", "a"])
        try fixture.inventory(["a", "b", "c"])
        try await restarted.refresh()
        let table = TableController(service: restarted)
        await table.refresh()
        #expect(table.groupOrder == ["b", "a", "c"])
        await restarted.shutdown()
    }

    @Test func explicitSaveReplacesOutsideOrderEvenWhenItMatchesTheBaseline() async throws {
        let fixture = OrderRestartFixture()
        try fixture.inventory(["a", "b", "c"])
        let service = try await fixture.open()
        try await service.setKindOrder(["b", "a"])
        var disk = try WireJSON.decode(Data(contentsOf: fixture.configFile))
        disk["kindOrder"] = ["c", "a", "b"]; disk["future"] = "kept"
        try WireJSON.encode(disk).write(to: fixture.configFile)
        try await service.setKindOrder(["b", "a"])
        #expect(try fixture.saved() == ["b", "a"])
        #expect(await service.persistedKindOrder() == ["b", "a"])
        #expect(try WireJSON.decode(Data(contentsOf: fixture.configFile))["future"] == "kept")
        try await service.setKindOrder([])
        #expect(try fixture.saved().isEmpty)
        disk = try WireJSON.decode(Data(contentsOf: fixture.configFile)); disk["kindOrder"] = ["c", "b", "a"]
        try WireJSON.encode(disk).write(to: fixture.configFile)
        // The explicit replacement intent was consumed; an unrelated save preserves the outside edit.
        try await service.setResidencySettings(.init(onDemandIdleMinutes: 5))
        #expect(try fixture.saved() == ["c", "b", "a"])
        try await service.setKindOrder([])
        #expect(try fixture.saved().isEmpty)
        disk["kindOrder"] = ["b", "a"]; try WireJSON.encode(disk).write(to: fixture.configFile)
        try await service.setKindOrder([]) // Empty list equal to the stale baseline must still replace disk.
        #expect(try fixture.saved().isEmpty)
        await service.shutdown()
    }

    @Test func rememberedSyntheticLocalGroupCannotBeSaved() async throws {
        let fixture = OrderRestartFixture()
        try fixture.inventory(["a", "b"])
        var config = ArsenalConfiguration(); config.catalog = ["source": "bundle"]; config.kindOrder = ["local", "b", "a"]
        try config.write(fixture.configFile)
        let service = try await fixture.open()
        #expect(await service.persistedKindOrder() == ["local", "b", "a"])
        await #expect(throws: WorkerFailure.self) { try await service.setKindOrder(["local", "a", "b"]) }
        #expect(try fixture.saved() == ["local", "b", "a"])
        try await service.setKindOrder(["a", "b"])
        #expect(try fixture.saved() == ["a", "b"])
        await service.shutdown()
    }
}
