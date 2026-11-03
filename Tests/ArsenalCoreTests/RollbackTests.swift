// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import ArsenalCore

private func rollbackCommand(_ executable: String, _ arguments: [String], environment: [String: String] = [:]) throws {
    let process = Process(), output = Pipe()
    process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
    process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, new in new }
    process.standardOutput = output; process.standardError = output
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw ManifestError.invalid(String(decoding: data, as: UTF8.self)) }
}

@Test func rollbackRestoresChangedCommitAndInstructionInventoryWithoutWeights() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["voices"])
    defer {
        unregisterFixtureBundles(in: fixture.paths.support.appendingPathComponent("rollback-test"))
        fixture.stop(); try? FileManager.default.removeItem(at: fixture.paths.support.deletingLastPathComponent())
    }
    await fixture.service.shutdown()
    let base = fixture.paths.support.appendingPathComponent("rollback-test")
    let support = base.appendingPathComponent("profile"), old = base.appendingPathComponent("Arsenal.app"), candidate = base.appendingPathComponent("candidate.app")
    let key = Curve25519.Signing.PrivateKey()
    var document = try JSONDecoder().decode(CatalogDocument.self, from: fixture.catalogData)
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    func bundle(_ app: URL, version: String, sequence: Int64, commit: String, addition: String) throws -> BundleCatalogSource {
        let contents = app.appendingPathComponent("Contents"), resources = contents.appendingPathComponent("Resources")
        try FileManager.default.createDirectory(at: contents.appendingPathComponent("MacOS"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: URL(fileURLWithPath: "/usr/bin/true"), to: contents.appendingPathComponent("MacOS/Arsenal"))
        let info = ["CFBundleIdentifier": "dev.arsenal.rollback-test", "CFBundleExecutable": "Arsenal", "CFBundlePackageType": "APPL", "CFBundleShortVersionString": version]
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0).write(to: contents.appendingPathComponent("Info.plist"))
        for (path, data) in fixture.source.files {
            let target = resources.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target)
        }
        try FileManager.default.copyItem(at: repo.appendingPathComponent("Resources/kinds"), to: resources.appendingPathComponent("kinds"))
        document.sequence = sequence; document.sourceCommit = commit
        let extra = Data(addition.utf8), relative = "rollback-evidence.txt"
        let folder = resources.appendingPathComponent("models/fixture-voices")
        try extra.write(to: folder.appendingPathComponent(relative))
        document.models[0].files.removeAll { $0.path == relative }
        document.models[0].files.append(.init(path: relative, sha256: FileDigest.sha256(extra), bytes: Int64(extra.count)))
        let catalog = resources.appendingPathComponent("catalog")
        try FileManager.default.createDirectory(at: catalog, withIntermediateDirectories: true)
        let data = try document.encoded()
        try data.write(to: catalog.appendingPathComponent("catalog.json"))
        try key.signature(for: data).write(to: catalog.appendingPathComponent("catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
        try rollbackCommand("/usr/bin/codesign", ["--force", "--sign", "-", app.path])
        return try BundleCatalogSource(resources: resources, catalog: document)
    }
    let previous = try bundle(old, version: "2.2", sequence: 7, commit: String(repeating: "a", count: 40), addition: "old instructions")
    let next = try bundle(candidate, version: "3.0", sequence: 8, commit: String(repeating: "b", count: 40), addition: "changed instructions")
    let installer = ModelFolderInstaller(support: support, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    func accept(_ source: BundleCatalogSource) async throws {
        let data = try source.catalog.encoded()
        try await installer.accept(data: data, signature: key.signature(for: data), publicKey: key.publicKey.rawRepresentation)
        _ = try await installer.install(id: "fixture-voices", source: source, replacingExisting: true)
    }
    try await accept(previous)
    let weights = support.appendingPathComponent("weights/keep.bin")
    try FileManager.default.createDirectory(at: weights.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data([1, 2, 3]).write(to: weights)
    var env = [
        "ARSENAL_REVIEWED": "1", "ARSENAL_DESTINATION_APP": old.path, "ARSENAL_SUPPORT_DIR": support.path
    ]
    // The fixtures are signed ad hoc, so without the test override the installer refuses before touching anything.
    let refusal = #expect(throws: ManifestError.self) {
        try rollbackCommand("/bin/bash", [repo.appendingPathComponent("scripts/install-for-review.sh").path, candidate.path], environment: env)
    }
    #expect("\(String(describing: refusal))".contains("not signed with Arsenal Release Signing"))
    #expect(FileManager.default.fileExists(atPath: old.path) && !FileManager.default.fileExists(atPath: old.path + ".previous"))
    env["ARSENAL_REVIEW_ALLOW_UNPINNED"] = "1"
    // One rollback: the older rollback is deleted (rm -rf, never the Trash), and the replaced app becomes the rollback.
    let retained = URL(fileURLWithPath: old.path + ".previous")
    try FileManager.default.copyItem(at: old, to: retained)
    let olderMarker = retained.appendingPathComponent("older-rollback-marker")
    try Data("older".utf8).write(to: olderMarker)
    try Data("historical snapshot pointer".utf8).write(to: URL(fileURLWithPath: old.path + ".previous.rollback-state"))
    try rollbackCommand("/bin/bash", [repo.appendingPathComponent("scripts/install-for-review.sh").path, candidate.path], environment: env)
    let archives = try FileManager.default.contentsOfDirectory(at: base, includingPropertiesForKeys: nil).filter {
        $0.lastPathComponent.hasPrefix("Arsenal.app.rollback-")
    }
    #expect(archives.isEmpty)
    #expect(!FileManager.default.fileExists(atPath: olderMarker.path))
    #expect(FileManager.default.fileExists(atPath: retained.path))
    #expect(try String(contentsOf: URL(fileURLWithPath: old.path + ".previous.rollback-state"), encoding: .utf8) != "historical snapshot pointer")
    try await accept(next)
    // Before rollback, the older bundle cannot fulfill the newly accepted source commit.
    await #expect(throws: (any Error).self) { try await previous.read(path: "models/fixture-voices/rollback-evidence.txt", commit: next.catalog.sourceCommit) }
    try rollbackCommand("/bin/bash", [repo.appendingPathComponent("scripts/rollback.sh").path], environment: env)
    let selected = try CatalogBootstrap.select(
        bundledDirectory: old.appendingPathComponent("Contents/Resources/catalog"), cachedDirectory: support.appendingPathComponent("catalog"),
        publicKey: key.publicKey.rawRepresentation)
    #expect(selected.document.sequence == 7 && selected.document.sourceCommit == previous.catalog.sourceCommit)
    let restored = ModelFolderInstaller(support: support, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
    try await restored.accept(data: selected.data, signature: selected.signature, publicKey: key.publicKey.rawRepresentation)
    let folder = try await restored.verifyInstalled(id: "fixture-voices")
    #expect(try String(contentsOf: folder.appendingPathComponent("rollback-evidence.txt"), encoding: .utf8) == "old instructions")
    #expect(try await previous.read(path: "models/fixture-voices/rollback-evidence.txt", commit: selected.document.sourceCommit) == Data("old instructions".utf8))
    #expect(try Data(contentsOf: weights) == Data([1, 2, 3]))
    let record = try await restored.installedManifest(id: "fixture-voices")
    #expect(record == previous.catalog.models[0].manifest)
    let snapshot = support.appendingPathComponent("rollback")
    #expect(try SafePath.files(in: snapshot).allSatisfy { !$0.contains("weights/") && !$0.contains("envs/") })
}
