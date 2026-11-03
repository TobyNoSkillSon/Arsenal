// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalCommands

private actor InventorySource: CatalogSource {
    let values: [String: Data]
    var reads: [String] = []
    init(_ values: [String: Data]) { self.values = values }
    func read(path: String, commit: String) throws -> Data {
        reads.append(path)
        guard let data = values[path] else { throw ManifestError.invalid("untracked path: \(path)") }
        return data
    }
}

struct CatalogBuildInventoryTests {
    @Test func catalogReadsOnlyCommittedShippingFilesAndPreservesIgnoredCaptures() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let original = repo.appendingPathComponent("models/birefnet")
        let model = try ModelManifest.load(original)
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temp) }
        let folder = temp.appendingPathComponent("birefnet")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let listed = try GitRead.run(repo: repo, args: ["ls-tree", "-r", "--name-only", "-z", "HEAD", "--", "models/birefnet/"])
        let tracked = Set(String(decoding: listed, as: UTF8.self).split(separator: "\0").map { String($0.dropFirst("models/birefnet/".count)) })
        let shipped = try CatalogShipping.files(in: original, model: model).filter { tracked.contains($0) }
        var committed: [String: Data] = [:]
        for path in shipped {
            let bytes = try Data(contentsOf: original.appendingPathComponent(path)), target = folder.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: target); committed["models/birefnet/" + path] = bytes
        }
        let fixture = folder.appendingPathComponent("gate/evidence/native/local.safetensors")
        try FileManager.default.createDirectory(at: fixture.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fixtureBytes = Data([0, 1, 2, 3]); try fixtureBytes.write(to: fixture)
        let generated = folder.appendingPathComponent("runners/swift/.build/.buildSystem_debug")
        try FileManager.default.createDirectory(at: generated.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("ignored SwiftPM marker".utf8).write(to: generated)
        let source = InventorySource(committed)
        let inventory = try await CatalogCommand.committedFiles(folder: folder, manifest: model, source: source, commit: String(repeating: "a", count: 40), tracked: tracked)
        #expect(Set(inventory.map(\.path)) == Set(shipped))
        #expect(Set(await source.reads) == Set(committed.keys))
        #expect(try Data(contentsOf: fixture) == fixtureBytes)
        #expect(!inventory.contains { $0.path.hasSuffix(".safetensors") })
        #expect(inventory.contains { $0.path == "README.md" } && inventory.contains { $0.path == "ops/remove_background.input.json" })
        // Tracked runtime changes remain rejected; untracked files are never part of a Git catalog.
        try Data("uncommitted".utf8).write(to: folder.appendingPathComponent("README.md"))
        await #expect(throws: (any Error).self) {
            try await CatalogCommand.committedFiles(folder: folder, manifest: model, source: source, commit: String(repeating: "a", count: 40), tracked: tracked)
        }
        try committed["models/birefnet/README.md"]!.write(to: folder.appendingPathComponent("README.md"))
        try Data("untracked runtime".utf8).write(to: folder.appendingPathComponent("runners/new-runtime.swift"))
        let unchanged = try await CatalogCommand.committedFiles(
            folder: folder, manifest: model, source: source,
            commit: String(repeating: "a", count: 40), tracked: tracked)
        #expect(unchanged == inventory)
        #expect(try Data(contentsOf: generated) == Data("ignored SwiftPM marker".utf8))
    }
}
