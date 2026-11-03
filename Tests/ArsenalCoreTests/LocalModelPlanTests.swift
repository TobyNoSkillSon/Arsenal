// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct LocalModelPlanTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    static let sha = String(repeating: "a", count: 64)

    @Test func onlyNamedOpenLicencesPass() {
        for open in ["Apache-2.0", "mit", " BSD-3-Clause ", "cc-by-4.0", "CC0-1.0"] {
            #expect(LocalModelPlan.isOpen(licence: open))
        }
        for closed in ["cc-by-nc-4.0", "other", "gemma", "LicenseRef-Qwen-Research", "openrail", "", nil] as [String?] {
            #expect(!LocalModelPlan.isOpen(licence: closed))
        }
    }

    @Test func weightFilesKeepTemplateFilesAndEverySafetensorsShard() {
        let repository = [
            "config.json", "model.safetensors", "tokenizer.json", "merges.txt", "README.md", "pytorch_model.bin",
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"
        ].map { HFMetadataFile(path: $0, bytes: 1, sha256: nil) }
        let selection = LocalModelPlan.weightFiles(repository: repository, template: ["config.json", "model.safetensors", "tokenizer_config.json"])
        #expect(selection.missing == ["tokenizer_config.json"])
        #expect(
            selection.files == [
                "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors", "model.safetensors.index.json"
            ])
    }

    @Test func dtypeFollowsTheCheckpointDeclaration() {
        #expect(LocalModelPlan.dtype(torchDType: "bfloat16") == "bf16")
        #expect(LocalModelPlan.dtype(torchDType: "float16") == "fp16")
        #expect(LocalModelPlan.dtype(torchDType: "float32") == "fp32")
        #expect(LocalModelPlan.dtype(torchDType: nil) == nil)
    }

    @Test func aShardedCheckpointNeedsNoSingleSafetensorsFile() {
        let repository = ["config.json", "tokenizer.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"]
            .map { HFMetadataFile(path: $0, bytes: 1, sha256: nil) }
        let selection = LocalModelPlan.weightFiles(repository: repository, template: ["config.json", "model.safetensors", "tokenizer.json"])
        #expect(selection.missing.isEmpty)
        #expect(selection.files.contains("model-00001-of-00002.safetensors") && selection.files.contains("model.safetensors.index.json"))
    }

    @Test func aRepositoryWithoutAnySafetensorsIsStillMissingItsWeights() {
        let repository = ["config.json", "pytorch_model.bin"].map { HFMetadataFile(path: $0, bytes: 1, sha256: nil) }
        let selection = LocalModelPlan.weightFiles(repository: repository, template: ["config.json", "model.safetensors"])
        #expect(selection.missing == ["model.safetensors"])
        #expect(!selection.files.contains { $0.hasSuffix(".safetensors") })
    }

    @Test func adapterFilesAreNeverPinnedAsBaseWeights() {
        let repository = ["config.json", "model.safetensors", "adapter_model.safetensors", "lora-merged.safetensors"]
            .map { HFMetadataFile(path: $0, bytes: 1, sha256: nil) }
        let selection = LocalModelPlan.weightFiles(repository: repository, template: ["config.json", "model.safetensors"])
        #expect(selection.files == ["config.json", "model.safetensors"])
    }

    @Test func theEmbeddingRowServesOnlyEmbeddingPipelines() throws {
        let registry = try ArchitectureRegistry.load(from: repo.appendingPathComponent("Resources/runners/architectures.json"))
        let row = try #require(registry.entries(modelType: "qwen3", kind: "embeddings").first)
        #expect(row.pipelineTags?.contains("feature-extraction") == true)
        #expect(row.pipelineTags?.contains("text-generation") != true)
    }

    @Test func parameterLabelsUseFolderStyle() {
        #expect(LocalModelPlan.parameterLabel(596_000_000) == "596M")
        #expect(LocalModelPlan.parameterLabel(1_700_000_000) == "1.7B")
        #expect(LocalModelPlan.parameterLabel(8_000_000_000) == "8B")
        #expect(LocalModelPlan.parameterLabel(nil) == "unknown")
    }

    @Test func manifestTakesIdentityAndWeightsFromTheRepositoryAndKeepsTheRunner() throws {
        var parser = TOMLParser(try String(contentsOf: repo.appendingPathComponent("models/harrier-oss-v1-0.6b/model.toml"), encoding: .utf8))
        let template = try parser.parse()
        let metadata = HFModelMetadata(
            repo: "Qwen/Qwen3-Embedding-0.6B", revision: String(repeating: "f", count: 40), pipelineTag: "feature-extraction", licence: "apache-2.0",
            languages: ["en"], parameterCount: 596_000_000, downloads: 1, likes: 1, files: [])
        let doc = LocalModelPlan.manifest(
            template: template, id: "qwen3-embedding-0.6b", metadata: metadata, architecture: nil, dtype: "bf16", files: ["config.json", "model.safetensors"],
            sha256: ["config.json": Self.sha, "model.safetensors": Self.sha], bytes: ["config.json": 727, "model.safetensors": 1_191_586_416],
            licenceAccepted: false, year: 2026)
        #expect(doc["id"] == .string("qwen3-embedding-0.6b"))
        #expect(doc["suggested"] == .bool(false))
        #expect(doc["popularity"] == nil)
        #expect(doc["good_for"] == nil)
        #expect(doc["derive"] == template["derive"])
        #expect(doc["runner"] == template["runner"])
        let revision = String(repeating: "f", count: 40)
        #expect(
            doc["licence"]
                == .table([
                    "spdx": .string("apache-2.0"), "name": .string("apache-2.0"), "commercial": .bool(true),
                    "url": .string("https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/" + revision)
                ]))
        guard case .table(let weights)? = doc["weights"] else { Issue.record("weights missing"); return }
        #expect(weights["repo"] == .string("Qwen/Qwen3-Embedding-0.6B"))
        #expect(weights["files"] == .array([.string("config.json"), .string("model.safetensors")]))
    }

    @Test func unacceptedNonOpenLicenceIsNamedAsNotOpen() throws {
        var parser = TOMLParser(try String(contentsOf: repo.appendingPathComponent("models/harrier-oss-v1-0.6b/model.toml"), encoding: .utf8))
        let template = try parser.parse()
        let metadata = HFModelMetadata(
            repo: "example/model", revision: String(repeating: "f", count: 40), pipelineTag: nil, licence: "other", languages: [], parameterCount: nil,
            downloads: 0, likes: 0, files: [])
        let doc = LocalModelPlan.manifest(
            template: template, id: "model", metadata: metadata, architecture: "custom", dtype: "bf16", files: [], sha256: [:], bytes: [:],
            licenceAccepted: true, year: 2026)
        guard case .table(let licence)? = doc["licence"] else { Issue.record("licence missing"); return }
        #expect(licence["commercial"] == .bool(false))
        #expect(licence["name"] == .string("other, accepted by the user"))
        #expect(doc["architecture"] == .string("custom"))
    }

    @Test func registryRowsAgreeWithTheirTemplatesAndWorkerSources() throws {
        let registry = try ArchitectureRegistry.load(from: repo.appendingPathComponent("Resources/runners/architectures.json"))
        #expect(registry.schema == 1)
        #expect(!registry.architectures.isEmpty)
        let sources = try swiftSources(under: repo.appendingPathComponent("Workers"))
        for entry in registry.architectures {
            let folder = repo.appendingPathComponent("models/\(entry.template)")
            var parser = TOMLParser(try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
            let template = try parser.parse()
            #expect(template["asset"] == nil && template["assets"] == nil, "\(entry.template) declares assets a local add does not copy")
            #expect(template["kind"] == .string(entry.kind), "\(entry.template) is not kind \(entry.kind)")
            guard case .array(let runners)? = template["runner"], case .table(let runner)? = runners.first else {
                Issue.record("\(entry.template) has no runner")
                continue
            }
            #expect(runner["worker"] == .string(entry.worker), "\(entry.template) runs worker \(String(describing: runner["worker"]))")
            #expect(runner["entry"] == nil, "\(entry.template) carries folder code, so it is not a bundled-worker template")
            #expect(sources.contains("\"\(entry.modelType)\""), "no worker source registers \(entry.modelType)")
        }
    }

    /// Every production Swift file under Workers (not the test trees), read once. Used only to confirm that a registry
    /// model_type is registered by a worker's source.
    func swiftSources(under root: URL) throws -> String {
        guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else { return "" }
        var text = ""
        for case let url as URL in walker where url.pathExtension == "swift" && url.path.contains("/Sources/") {
            text += (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        }
        return text
    }
}
