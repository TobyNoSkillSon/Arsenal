// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCommands
@testable import ArsenalCore

/// `catalog add-local` without the network: every registry row turns into a folder that the app's own loader accepts,
/// and a folder that fails validation leaves no trace.
struct LocalModelAddTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func build(_ entry: ArchitectureEntry, id: String, templateFolder: URL) throws -> LocalModelAddCommand.Build {
        var parser = TOMLParser(try String(contentsOf: templateFolder.appendingPathComponent("model.toml"), encoding: .utf8))
        let template = try parser.parse()
        guard case .table(let weights)? = template["weights"], case .array(let listed)? = weights["files"] else { throw ManifestError.invalid("template has no weights.files") }
        let files = listed.compactMap { value -> String? in
            guard case .string(let path) = value else { return nil }
            return path
        }
        let metadata = HFModelMetadata(
            repo: "example/\(id)", revision: String(repeating: "a", count: 40), pipelineTag: nil, licence: "apache-2.0", languages: ["en"], parameterCount: nil,
            downloads: 0, likes: 0, files: files.map { HFMetadataFile(path: $0, bytes: 10, sha256: String(repeating: "b", count: 64)) })
        return LocalModelAddCommand.Build(
            id: id, template: template, templateFolder: templateFolder, entry: entry, metadata: metadata, dtype: "bf16", files: files,
            pins: files.map { DerivedFile(path: $0, bytes: 10, sha256: String(repeating: "c", count: 64)) }, licence: "apache-2.0", open: true, accepted: false)
    }

    /// One scratch directory per test: `local-models/` and any copied template live inside it.
    func scratch() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("add-local-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test func everyRegistryRowStagesAFolderTheLoaderAccepts() throws {
        let registry = try ArchitectureRegistry.load(from: repo.appendingPathComponent("Resources/runners/architectures.json"))
        let kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
        let root = try scratch()
        defer { try? FileManager.default.removeItem(at: root) }
        let localRoot = root.appendingPathComponent("local-models", isDirectory: true)
        for entry in registry.architectures {
            let id = "copy-\(entry.template)"
            let templateFolder = repo.appendingPathComponent("models/\(entry.template)")
            let final = try LocalModelAddCommand.stage(
                try build(entry, id: id, templateFolder: templateFolder), localRoot: localRoot, kinds: kinds, year: 2026)
            #expect(final.path == localRoot.appendingPathComponent(id).path)
            #expect(!FileManager.default.fileExists(atPath: localRoot.appendingPathComponent(".staging-\(id)").path))
            _ = try ModelManifest.load(final, requirePins: true, kinds: kinds, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        }
    }

    @Test func aFolderThatFailsValidationLeavesNothingBehind() throws {
        let registry = try ArchitectureRegistry.load(from: repo.appendingPathComponent("Resources/runners/architectures.json"))
        let kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
        let entry = try #require(registry.architectures.first)
        let root = try scratch()
        defer { try? FileManager.default.removeItem(at: root) }
        let broken = root.appendingPathComponent("broken-template", isDirectory: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("models/\(entry.template)"), to: broken)
        // Without ops/, the copied [[op]] entries point at files that do not exist, so validation must refuse.
        try FileManager.default.removeItem(at: broken.appendingPathComponent("ops"))
        let localRoot = root.appendingPathComponent("local-models", isDirectory: true)
        #expect(throws: LocalModelAddCommand.Refusal.self) {
            try LocalModelAddCommand.stage(try build(entry, id: "broken-copy", templateFolder: broken), localRoot: localRoot, kinds: kinds, year: 2026)
        }
        #expect(!FileManager.default.fileExists(atPath: localRoot.appendingPathComponent("broken-copy").path))
        #expect(!FileManager.default.fileExists(atPath: localRoot.appendingPathComponent(".staging-broken-copy").path))
    }

    @Test func optionsRejectAnythingButOneRepositoryAndKnownFlags() throws {
        let options = try LocalModelAddCommand.Options.parse(["Qwen/Qwen3-Embedding-0.6B", "--plan", "--kind", "embeddings"])
        #expect(options.repo == "Qwen/Qwen3-Embedding-0.6B" && options.plan && options.kind == "embeddings")
        #expect(throws: LocalModelAddCommand.Refusal.self) { try LocalModelAddCommand.Options.parse(["not-a-repo"]) }
        #expect(throws: LocalModelAddCommand.Refusal.self) { try LocalModelAddCommand.Options.parse(["a/b", "c/d"]) }
        #expect(throws: LocalModelAddCommand.Refusal.self) { try LocalModelAddCommand.Options.parse(["a/b", "--unknown"]) }
    }
}
