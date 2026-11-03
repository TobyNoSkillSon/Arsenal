// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import ArsenalCore

struct CatalogShippingTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    private func python(_ arguments: [String]) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3"); process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        try process.run()
        let deadline = Date().addingTimeInterval(30)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.02) }
        if process.isRunning {
            process.terminate()
            throw ManifestError.invalid("catalog shipping test Python timed out")
        }
        return process.terminationStatus
    }

    @Test func realCatalogShipsOnlyRuntimeFilesWithIdenticalCatalogAndInstalledOutcomes() async throws {
        let fm = FileManager.default, root = repo.appendingPathComponent("tmp/catalog-shipping-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let catalogData = try Data(contentsOf: repo.appendingPathComponent("Resources/catalog/catalog.json"))
        let catalogSignature = try Data(contentsOf: repo.appendingPathComponent("Resources/catalog/catalog.json.sig"))
        let catalogKey = try Data(contentsOf: repo.appendingPathComponent("Resources/catalog-public-key"))
        var original = try CatalogTrust.verify(data: catalogData, signature: catalogSignature, publicKey: catalogKey)
        // The signed inventory is exactly the repository's shipping folders (derived, never a hardcoded count).
        let shippingFolders = try fm.contentsOfDirectory(at: repo.appendingPathComponent("models"), includingPropertiesForKeys: nil)
            .filter { fm.fileExists(atPath: $0.appendingPathComponent("model.toml").path) }
            .filter { CatalogShipping.includes(try ModelManifest.load($0)) }
        #expect(original.models.count == shippingFolders.count && original.models.count >= 16)
        // Always compare with the complete repository folders, including after the coordinator signs
        // the slim inventory. Only these temporary fixture documents use an ephemeral test key.
        for index in original.models.indices {
            let folder = repo.appendingPathComponent("models/" + original.models[index].id)
            // Local implementation branches deliberately leave the signed release catalog untouched.
            original.models[index].manifest = try ModelManifest.load(folder)
            original.models[index].files = try SafePath.files(in: folder).map { path in
                CatalogFile(path: path, sha256: try FileDigest.sha256(folder.appendingPathComponent(path)), bytes: try FileDigest.bytes(folder.appendingPathComponent(path)))
            }
        }
        let currentKinds = repo.appendingPathComponent("Resources/kinds")
        original.kindFiles = try SafePath.files(in: currentKinds).map { path in
            CatalogFile(
                path: path, sha256: try FileDigest.sha256(currentKinds.appendingPathComponent(path)), bytes: try FileDigest.bytes(currentKinds.appendingPathComponent(path)))
        }
        let originalData = try original.encoded(), originalKey = Curve25519.Signing.PrivateKey()
        let originalSignature = try originalKey.signature(for: originalData)
        var slim = original
        let staging = root.appendingPathComponent("source"), resources = root.appendingPathComponent("Resources")
        try fm.createDirectory(at: staging.appendingPathComponent("catalog"), withIntermediateDirectories: true)
        try fm.createDirectory(at: resources.appendingPathComponent("catalog"), withIntermediateDirectories: true)
        var originalBytes: Int64 = 0, shippedBytes: Int64 = 0
        for index in slim.models.indices {
            let item = original.models[index], folder = repo.appendingPathComponent("models/" + item.id)
            let paths = Set(try CatalogShipping.files(in: folder, model: item.manifest))
            #expect(paths.isSubset(of: Set(item.files.map(\.path))))
            #expect(!paths.contains { $0.hasPrefix("gate/evidence/") || $0.hasPrefix("results/intake/") })
            #expect(paths.contains(item.manifest.gate.recipe) && paths.contains(item.manifest.bench.recipe))
            slim.models[index].files = item.files.filter { paths.contains($0.path) }
            originalBytes += item.files.reduce(0) { $0 + $1.bytes }
            shippedBytes += slim.models[index].files.reduce(0) { $0 + $1.bytes }
            for file in slim.models[index].files {
                let target = staging.appendingPathComponent("models/\(item.id)/" + file.path)
                try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fm.copyItem(at: folder.appendingPathComponent(file.path), to: target)
            }
        }
        #expect(originalBytes >= shippedBytes)
        // 101 models since 2 Nov (3.3 MB of shipped model files; 2.4 MB for the earlier 77); the bound catches evidence or
        // captures leaking into Get.
        #expect(shippedBytes < 4 * 1_048_576)
        print("CatalogShipping: \(originalBytes) repository model bytes; \(shippedBytes) shipped model bytes")
        for file in original.kindFiles ?? [] {
            let target = staging.appendingPathComponent("kinds/" + file.path)
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(at: repo.appendingPathComponent("Resources/kinds/" + file.path), to: target)
        }
        // A source checkout may hold extra tracked evidence; the bundle must contain exactly the signed list.
        let extra = "models/\(slim.models[0].id)/gate/evidence/intake/repo-only.json"
        let extraSource = staging.appendingPathComponent(extra)
        try fm.createDirectory(at: extraSource.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: extraSource)
        let data = try slim.encoded(), key = Curve25519.Signing.PrivateKey(), signature = try key.signature(for: data)
        try data.write(to: staging.appendingPathComponent("catalog/catalog.json"))
        try data.write(to: resources.appendingPathComponent("catalog/catalog.json"))
        try signature.write(to: resources.appendingPathComponent("catalog/catalog.json.sig"))
        let inventory = repo.appendingPathComponent("scripts/bundle-inventory.py").path
        #expect(try python([inventory, "copy", staging.path, resources.path]) == 0)
        #expect(!fm.fileExists(atPath: resources.appendingPathComponent(extra).path))
        #expect(try python([inventory, "verify", resources.path]) == 0)
        let extraBundle = resources.appendingPathComponent(extra)
        try fm.createDirectory(at: extraBundle.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: extraBundle)
        #expect(try python([inventory, "verify", resources.path]) != 0)
        try fm.removeItem(at: extraBundle)
        #expect(try python([repo.appendingPathComponent("scripts/quality-identities.py").path, resources.appendingPathComponent("quality-identities.json").path]) == 0)
        let artifacts = BundledWorkerArtifact(helpers: root.appendingPathComponent("Helpers"), qualityIdentities: resources.appendingPathComponent("quality-identities.json"))
        let baselineResources = root.appendingPathComponent("baseline-resources")
        try fm.createDirectory(at: baselineResources, withIntermediateDirectories: true)
        try fm.copyItem(at: repo.appendingPathComponent("models"), to: baselineResources.appendingPathComponent("models"))
        try fm.copyItem(at: repo.appendingPathComponent("Resources/kinds"), to: baselineResources.appendingPathComponent("kinds"))
        let baseline = try await RealModelsService.open(
            catalogData: originalData, signature: originalSignature, publicKey: originalKey.publicKey.rawRepresentation,
            source: BundleCatalogSource(resources: baselineResources, catalog: original), kinds: KindSpec.defaults,
            paths: ArsenalPaths(support: root.appendingPathComponent("baseline-support"), cache: root.appendingPathComponent("baseline-cache")),
            artifacts: artifacts, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        let service = try await RealModelsService.open(
            catalogData: data, signature: signature, publicKey: key.publicKey.rawRepresentation,
            source: BundleCatalogSource(resources: resources, catalog: slim), kinds: KindSpec.defaults, paths: paths,
            artifacts: artifacts, kindsDirectory: resources.appendingPathComponent("kinds"))
        let before = await baseline.catalog(), after = await service.catalog()
        #expect(before.count == after.count && before.count >= 16)
        // Fixtures must carry real recorded pass/fail outcomes. Current implementation branches may
        // correctly invalidate every binding; packing must preserve that unchecked state as well.
        var recordedOutcomes = 0
        for item in original.models {
            for file in item.files where file.path.hasPrefix("results/") && file.path.hasSuffix(".json") {
                let data = try Data(contentsOf: repo.appendingPathComponent("models/" + item.id + "/" + file.path))
                if let value = try? WireJSON.decode(data), value["stats"]?["qualityCheckPassed"]?.bool != nil { recordedOutcomes += 1 }
            }
        }
        #expect(recordedOutcomes > 0)
        var checkedOutcomes = 0
        for item in slim.models {
            let expected = try #require(before.first { $0.id == item.id }), actual = try #require(after.first { $0.id == item.id })
            #expect(!actual.added)
            #expect(actual.row.precisionStats == expected.row.precisionStats, "\(item.id): catalog-only stats")
            #expect(actual.row.precisionQualityWarnings == expected.row.precisionQualityWarnings)
            #expect(actual.runners == expected.runners)
            checkedOutcomes += actual.row.precisionStats.values.filter { $0.qualityCheckPassed != nil }.count
            let operations = await service.operations(id: item.id), baselineOperations = await baseline.operations(id: item.id)
            #expect(!operations.isEmpty && operations == baselineOperations)
            let readme = try await service.modelREADME(id: item.id), baselineReadme = try await baseline.modelREADME(id: item.id)
            #expect(readme == baselineReadme)
            try await service.add(id: item.id)
            let installed = try await service.executableFolder(id: item.id, installing: true)
            #expect(Set(try SafePath.files(in: installed)) == Set(item.files.map(\.path)))
            #expect(try ModelManifest.load(installed, kindsDirectory: resources.appendingPathComponent("kinds")) == item.manifest)
            for runner in item.runners {
                let disk = try ModelLoadConfiguration.identity(model: item.manifest, runner: runner, folder: installed)
                #expect(disk == (try ModelLoadConfiguration.identity(model: item.manifest, runner: runner, folder: repo.appendingPathComponent("models/" + item.id))))
                #expect(await service.shippingStats(item.manifest, runner: runner.name) == expected.runners.first { $0.name == runner.name }?.stats)
            }
        }
        #expect(checkedOutcomes == before.flatMap { $0.row.precisionStats.values }.filter { $0.qualityCheckPassed != nil }.count)
        await service.shutdown(); await baseline.shutdown()
    }
}

private extension RealModelsService {
    func shippingStats(_ model: ModelManifest, runner: String) -> [String: PrecisionStatSet] { statsForRunner(model, runner: runner) }
}

/// A model whose 16-bit weights are under 1 GB ships at 16-bit only; no int8 or int4 tier is offered (every Mac
/// has the RAM for it). FP32 sources count at half their bytes, since they are converted to 16-bit at Get.
struct SmallModelsShipSixteenBitOnlyTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @Test func shippedModelsUnderOneGigabyteDeclareNoQuantizedTier() throws {
        let fm = FileManager.default
        let folders = try fm.contentsOfDirectory(at: repo.appendingPathComponent("models"), includingPropertiesForKeys: nil)
            .filter { fm.fileExists(atPath: $0.appendingPathComponent("model.toml").path) }
        var checked = 0, offenders: [String] = []
        for folder in folders {
            let m = try ModelManifest.load(folder)
            guard CatalogShipping.includes(m), let bytes = m.weights.bytes, bytes.count == m.weights.files.count else { continue }
            let total = bytes.values.reduce(0, +), sixteenBit = m.weights.dtype == "fp32" ? total / 2 : total
            guard sixteenBit < 1_000_000_000 else { continue }
            checked += 1
            let quantized = m.precisions.filter { ["int8", "int4"].contains($0) }
            if !quantized.isEmpty { offenders.append("\(m.id): \(quantized.joined(separator: ", "))") }
        }
        #expect(checked >= 10, "The rule covers the small shipped models")
        #expect(offenders.isEmpty, "Under 1 GB at 16-bit, yet quantized tiers: \(offenders)")
    }
}
