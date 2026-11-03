// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import ArsenalCore

struct CatalogQualityIdentityTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    private func entry(_ folder: URL) throws -> CatalogModel {
        try CatalogModel(
            manifest: ModelManifest.load(folder),
            files: CatalogShipping.files(in: folder, model: ModelManifest.load(folder)).map {
                CatalogFile(path: $0, sha256: try FileDigest.sha256(folder.appendingPathComponent($0)), bytes: try FileDigest.bytes(folder.appendingPathComponent($0)))
            })
    }

    @Test(arguments: ["m1", "m3-max", "m5-max"])
    func catalogAndFolderIdentitiesMatchForChipModeAndLoadSettings(chip: String) throws {
        let folder = repo.appendingPathComponent("models/laya-typed-decisions"), item = try entry(folder), model = item.manifest, runner = model.runners[0]
        let manifest = try Data(contentsOf: folder.appendingPathComponent("kernels/manifest.toml"))
        for fast in [false, true] {
            for stock in [false, true] {
                var settings = try ModelLoadConfiguration.defaults(model); try settings.set(model.id, key: "stock_mlx", value: .bool(stock))
                let disk = try ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder, chip: chip, fast: fast, settings: settings)
                let catalog = try ModelLoadConfiguration.identity(catalog: item, runner: runner, kernelManifest: manifest, chip: chip, fast: fast, settings: settings)
                #expect(catalog == disk)
            }
        }
    }

    @Test(arguments: ["m5/smallm_tile.metal", "m5/smallm_tile.h.metal", "any/splitk_exact.metal"])
    func changedCatalogKernelHashInvalidatesPassAndFailure(path: String) throws {
        let folder = repo.appendingPathComponent("models/laya-typed-decisions"), item = try entry(folder), model = item.manifest, runner = model.runners[0]
        let manifest = try Data(contentsOf: folder.appendingPathComponent("kernels/manifest.toml"))
        let original = try ModelLoadConfiguration.identity(catalog: item, runner: runner, kernelManifest: manifest, chip: "m5-max")
        var changed = item
        let index = try #require(changed.files.firstIndex { $0.path == "kernels/" + path })
        changed.files[index].sha256 = FileDigest.sha256(Data("tampered".utf8))
        let identity = try ModelLoadConfiguration.identity(catalog: changed, runner: runner, kernelManifest: manifest, chip: "m5-max")
        #expect(identity != original)
        // Laya ships 16-bit only since 26 Oct: its FP16 outcome is the one bound to the kernels.
        for dtype in [model.basePrecision] {
            let result = try PrecisionStatsResult.decode(Data(contentsOf: folder.appendingPathComponent("results/intake-m5-max-\(dtype)-native-fast.json")))
            let current = PrecisionEvidence.qualityIdentity(model: model, dtype: dtype, runner: runner.name, implementation: "worker", loadIdentity: original)
            var bound = result; bound.stats.qualityIdentity = current
            #expect(PrecisionEvidence.scopeQuality(bound, identity: current).stats.qualityCheckPassed == result.stats.qualityCheckPassed)
            let tampered = PrecisionEvidence.qualityIdentity(model: model, dtype: dtype, runner: runner.name, implementation: "worker", loadIdentity: identity)
            #expect(PrecisionEvidence.scopeQuality(bound, identity: tampered).stats.qualityCheckPassed == nil)
        }
    }

    @Test func missingOrUnauthenticatedCatalogDeclarationsCannotProduceIdentity() throws {
        let folder = repo.appendingPathComponent("models/laya-typed-decisions"), item = try entry(folder), runner = item.manifest.runners[0]
        let manifest = try Data(contentsOf: folder.appendingPathComponent("kernels/manifest.toml"))
        for data in [nil, Data("[[kernel]]\nid = 'changed'".utf8), manifest + Data("\n# changed".utf8)] as [Data?] {
            #expect(throws: (any Error).self) { try ModelLoadConfiguration.identity(catalog: item, runner: runner, kernelManifest: data) }
        }
        var missing = item; missing.files.removeAll { $0.path == "kernels/m5/smallm_tile.h.metal" }
        #expect(throws: (any Error).self) { try ModelLoadConfiguration.identity(catalog: missing, runner: runner, kernelManifest: manifest) }
        var unsafe = item
        let data = Data(String(decoding: manifest, as: UTF8.self).replacingOccurrences(of: "m5/smallm_tile.metal", with: "../outside.metal").utf8)
        let index = try #require(unsafe.files.firstIndex { $0.path == "kernels/manifest.toml" })
        unsafe.files[index] = CatalogFile(path: "kernels/manifest.toml", sha256: FileDigest.sha256(data), bytes: Int64(data.count))
        unsafe.files.append(CatalogFile(path: "kernels/../outside.metal", sha256: String(repeating: "a", count: 64), bytes: 1))
        #expect(throws: (any Error).self) { try ModelLoadConfiguration.identity(catalog: unsafe, runner: runner, kernelManifest: data) }
    }

    @Test func catalogOnlyLayaRetainsIntakeOutcomesBeforeAndAfterFolderInstallation() async throws {
        let root = repo.appendingPathComponent("tmp/catalog-quality-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = repo.appendingPathComponent("models/laya-typed-decisions"), model = try ModelManifest.load(folder)
        var contents: [String: Data] = [:]
        let files = try CatalogShipping.files(in: folder, model: model).map { path in
            let data = try Data(contentsOf: folder.appendingPathComponent(path)); contents["models/\(model.id)/\(path)"] = data
            return CatalogFile(path: path, sha256: FileDigest.sha256(data), bytes: Int64(data.count))
        }
        let kinds = repo.appendingPathComponent("Resources/kinds")
        let kindFiles = try SafePath.files(in: kinds).map { path in
            let data = try Data(contentsOf: kinds.appendingPathComponent(path)); contents["kinds/" + path] = data
            return CatalogFile(path: path, sha256: FileDigest.sha256(data), bytes: Int64(data.count))
        }
        let catalog = CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: [.init(manifest: model, files: files)], kindFiles: kindFiles)
        let data = try catalog.encoded(), key = Curve25519.Signing.PrivateKey()
        let receipt = try latestIntakeBinding(folder)
        let resources = root.appendingPathComponent("Resources"), helpers = root.appendingPathComponent("Helpers")
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        try WireJSON.encode(["decisions": try #require(receipt["implementation_identity"])]).write(to: resources.appendingPathComponent("quality-identities.json"))
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        let service = try await RealModelsService.open(
            catalogData: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation,
            source: FixtureCatalogSource(files: contents), kinds: KindSpec.defaults, paths: paths,
            artifacts: BundledWorkerArtifact(helpers: helpers), kindsDirectory: kinds)
        let installedFolder = paths.directory("models").appendingPathComponent(model.id)
        #expect(!FileManager.default.fileExists(atPath: installedFolder.path))
        let unadded = try #require(await service.catalog().first { $0.id == model.id })
        #expect(!unadded.added)
        // Laya ships its 16-bit format only since 26 Oct (under 1 GB at 16-bit): the FP16 outcome is the one bound to the catalog.
        #expect(unadded.row.precisionStats["fp16"]?.qualityCheckPassed == true)
        #expect(unadded.row.precisionStats["int8"]?.qualityCheckPassed == nil && unadded.row.precisionStats["int4"]?.qualityCheckPassed == nil)
        let runner = try #require(unadded.runners.first { $0.name == "native" })
        #expect(runner.stats["fp16"]?.qualityCheckPassed == true)
        try await service.add(id: model.id)
        // Added without weights: recorded, still only in Get more models.
        #expect(await service.rows().contains { $0.id == model.id } == false)
        let row = try #require(await catalogRow(service, model.id))
        #expect(row.state == .notInstalled)
        #expect(row.precisionStats["fp16"]?.qualityCheckPassed == true)
        // Install only the authenticated model folder, never Get/download weights or launch a worker.
        _ = try await service.executableFolder(id: model.id, installing: true)
        let installed = await service.catalogIdentityTestStats(model)
        #expect(installed == row.precisionStats)
        let sourceIdentity = try ModelLoadConfiguration.identity(model: model, runner: model.runners[0], folder: folder)
        #expect(
            try ModelLoadConfiguration.identity(
                catalog: catalog.models[0], runner: model.runners[0], kernelManifest: contents["models/\(model.id)/kernels/manifest.toml"])
                == sourceIdentity)
        #expect(try ModelLoadConfiguration.identity(model: model, runner: model.runners[0], folder: installedFolder) == sourceIdentity)
        #expect(sourceIdentity == receipt["load_identity"]?.string)
        // An installed folder must be checked from disk, even though its catalog remains untampered.
        let kernel = installedFolder.appendingPathComponent("kernels/m5/smallm_tile.h.metal")
        var changed = try Data(contentsOf: kernel); changed.append(Data("\n// changed\n".utf8)); try changed.write(to: kernel)
        let tampered = await service.catalogIdentityTestStats(model)
        #expect(tampered["fp16"]?.qualityCheckPassed == nil)
        await service.shutdown()
    }
}

private extension RealModelsService {
    func catalogIdentityTestStats(_ model: ModelManifest) -> [String: PrecisionStatSet] { statsForRunner(model, runner: "native") }
}
