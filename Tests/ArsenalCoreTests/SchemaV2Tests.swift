// SPDX-License-Identifier: MIT
import Foundation
import Testing
import CryptoKit
@testable import ArsenalCore

struct SchemaV2Tests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func temp() throws -> URL {
        let url = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-schema/\(UUID().uuidString)"); try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    private func fixture(_ root: URL) throws -> URL {
        let folder = root.appendingPathComponent("qwen3.8-27b")
        try FileManager.default.copyItem(at: repo.appendingPathComponent("models/qwen3.8-27b"), to: folder)
        return folder
    }
    @Test func settingsValidateDefaultAndValueTypes() throws {
        let choice = SettingDeclaration(
            id: "context_limit", title: "Context", type: "choice", defaultValue: .number(32768), choices: [.number(8192), .number(32768)], applies: "load")
        #expect(choice.problems().isEmpty); #expect(choice.accepts(.number(8192))); #expect(!choice.accepts(.bool(true)))
        var invalid = choice; invalid.defaultValue = .number(1024); #expect(!invalid.problems().isEmpty)
        var slider = SettingDeclaration(
            id: "temperature", title: "Temperature", type: "slider", defaultValue: .number(0.6), min: 0, max: 2, step: 0.1, applies: "request", mapsTo: "request.temperature")
        #expect(slider.problems().isEmpty); #expect(!slider.accepts(.number(0.65))); slider.mapsTo = nil; #expect(!slider.problems().isEmpty)
    }
    @Test func v1RemainsValidAndV2RequiresKnownSelection() throws {
        let root = try temp(); let folder = try fixture(root)
        var model = try ModelManifest.load(folder)
        #expect(model.schema == 2); #expect(model.settings.count == 6); #expect(model.isSuggested)
        #expect(model.settings.first { $0.id == "context_limit" }?.choices == [.number(262144)])
        model.selection = RunnerSelection(defaultRunner: "missing")
        #expect(model.problems(folder: folder).contains { $0.contains("unknown runner") })
        model.selection = nil; #expect(model.problems(folder: folder).contains { $0.contains("requires selection") })
        model.schema = 1; #expect(model.problems(folder: folder).isEmpty)
    }
    @Test func genericSchemaEnforcesBoundsReferencesAndAlternatives() throws {
        let data = Data(
            ##"{"$defs":{"positive":{"type":"integer","minimum":1}},"type":"object","required":["n"],"properties":{"n":{"$ref":"#/$defs/positive","maximum":5},"value":{"anyOf":[{"type":"null"},{"type":"string","maxLength":3}]},"step":{"type":"number","multipleOf":0.5}},"additionalProperties":false}"##
                .utf8)
        #expect(try ModelSchemaValidation.validate(value: ["n": 3, "value": NSNull(), "step": 1.5], schemaData: data).isEmpty)
        for bad: [String: Any] in [["n": true], ["n": 6], ["n": 2, "unknown": 1], ["n": 2, "value": "long"], ["n": 2, "step": 1.3]] {
            #expect(try !ModelSchemaValidation.validate(value: bad, schemaData: data).isEmpty)
        }
        #expect(throws: (any Error).self) { try ModelSchemaValidation.checkSchema(schemaData: Data(##"{"$ref":"https://example.com/schema"}"##.utf8)) }
        #expect(throws: (any Error).self) { try ModelSchemaValidation.checkSchema(schemaData: Data(##"{"unevaluatedProperties":false}"##.utf8)) }
        let recursive = Data(##"{"$defs":{"loop":{"allOf":[{"$ref":"#/$defs/loop"},{"$ref":"#/$defs/loop"}]}},"$ref":"#/$defs/loop"}"##.utf8)
        #expect(throws: (any Error).self) { try ModelSchemaValidation.validate(value: [:], schemaData: recursive) }
        for malformed in [##"{"minimum":"0"}"##, ##"{"type":"imaginary"}"##, ##"{"required":"x"}"##, ##"{"properties":{"x":false}}"##, ##"{"multipleOf":0}"##] {
            #expect(throws: (any Error).self) { try ModelSchemaValidation.checkSchema(schemaData: Data(malformed.utf8)) }
        }
    }
    @Test func flatRunnerCheckValidatesMergedInputs() throws {
        let root = try temp(); let folder = try fixture(root); let model = try ModelManifest.load(folder)
        let file = folder.appendingPathComponent("gate/runner-check.json")
        try Data(##"{"chat":{"messages":[{"role":"user","content":"Hello"}]},"count_tokens":{"text":"Hi"}}"##.utf8).write(to: file)
        try RunnerCheckSamples.validate(folder: folder, model: model)
        try Data(##"{"chat":{"messages":[]}}"##.utf8).write(to: file)
        #expect(throws: (any Error).self) { try RunnerCheckSamples.validate(folder: folder, model: model) }
        try Data(##"{"unknown":{}}"##.utf8).write(to: file)
        #expect(throws: (any Error).self) { try RunnerCheckSamples.validate(folder: folder, model: model) }
    }
    @Test func structuredRunnerCheckValidatesOpsAndCancelNotLoad() throws {
        let root = try temp(); let folder = try fixture(root); let model = try ModelManifest.load(folder)
        let file = folder.appendingPathComponent("gate/runner-check.json")
        try Data(
            ##"{"load":{"config":{"arbitrary":true},"dtype":"bf16"},"ops":{"count_tokens":{"text":"Hello"}},"cancel":{"op":"chat","params":{"messages":[{"role":"user","content":"Hello"}]}}}"##
                .utf8
        ).write(to: file)
        try RunnerCheckSamples.validate(folder: folder, model: model)
        try Data(##"{"cancel":{"op":"chat","params":{"messages":[]}}}"##.utf8).write(to: file)
        #expect(throws: (any Error).self) { try RunnerCheckSamples.validate(folder: folder, model: model) }
        try Data(##"{"ops":[],"load":{}}"##.utf8).write(to: file)
        #expect(throws: (any Error).self) { try RunnerCheckSamples.validate(folder: folder, model: model) }
    }
    @Test func modelOpsAreAuthoritativeAndOnlyExplicitKindDefaultsFallback() throws {
        let root = try temp(); let folder = try fixture(root); var model = try ModelManifest.load(folder)
        let kind = try #require(KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).first { $0.id == "chat" })
        model.op = [OpDeclaration(name: "chat", title: "Custom", input: "ops/chat.input.json", output: "json", stream: false)]
        let ops = model.resolvedOperations(kind: kind)
        #expect(ops.count == 1); #expect(ops.first { $0.name == "chat" }?.title == "Custom")
        model.op = []; #expect(model.resolvedOperations(kind: kind).isEmpty)
        model.op = nil; #expect(model.resolvedOperations(kind: kind).count == 2)
        var noDefaults = kind; noDefaults.op = [.init(name: "design", title: "Design", input: "ops/chat.input.json", output: "audio", stream: false)]
        #expect(model.resolvedOperations(kind: noDefaults).isEmpty)
    }
    @Test func kernelResolutionUsesReplacementSitesAndFastExactFallback() throws {
        func k(_ id: String, _ chip: String, _ mode: String, _ site: String) -> KernelDeclaration {
            .init(id: id, chip: chip, mode: mode, files: ["a.metal"], entry: "entry", replaces: site, api: "metal")
        }
        let manifest = KernelManifest(kernels: [
            k("base", "*", "exact", "linear"), k("family", "m5", "exact", "linear"), k("fast", "*", "fast", "linear"), k("other", "*", "exact", "attention"),
            k("chip-fast", "m5", "fast", "linear")
        ])
        #expect(manifest.resolved(chip: "m5-max", fast: false).map(\.id) == ["family", "other"])
        #expect(manifest.resolved(chip: "m5", fast: true).map(\.id) == ["other", "chip-fast"])
        #expect(manifest.resolved(chip: "m4", fast: true).map(\.id) == ["fast", "other"])
        #expect(manifest.resolved(chip: "m4", fast: false).map(\.id) == ["base", "other"])
    }
    @Test func kernelValidatorRejectsMissingMlxIOAndDuplicateSites() throws {
        let root = try temp(); let folder = try fixture(root); let dir = folder.appendingPathComponent("kernels")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("kernel source".utf8).write(to: dir.appendingPathComponent("a.metal"))
        let declaration = "[[kernel]]\nid = \"test\"\nchip = \"*\"\nmode = \"exact\"\nfiles = [\"a.metal\"]\nentry = \"entry\"\nreplaces = \"linear\"\n"
        let path = dir.appendingPathComponent("manifest.toml")
        try declaration.write(to: path, atomically: true, encoding: .utf8)
        #expect(throws: (any Error).self) { try KernelManifest.loadIfPresent(folder: folder) }
        try (declaration + "inputs = [\"x\"]\noutputs = [\"out\"]\n").write(to: path, atomically: true, encoding: .utf8)
        #expect(try KernelManifest.loadIfPresent(folder: folder).kernels.count == 1)
        try (declaration + "api = \"metal\"\n" + declaration.replacingOccurrences(of: "id = \"test\"", with: "id = \"different\"") + "api = \"metal\"\n").write(
            to: path, atomically: true, encoding: .utf8)
        #expect(throws: (any Error).self) { try KernelManifest.loadIfPresent(folder: folder) }
    }
    @Test func bundleSourceWorksOfflineAndRejectsTamperUnlistedAndWrongCommit() async throws {
        let root = try temp(); let resources = root.appendingPathComponent("App/Contents/Resources")
        let models = resources.appendingPathComponent("models"); try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        let folder = try fixture(models); let model = try ModelManifest.load(folder)
        let files = try SafePath.files(in: folder).map { p in
            CatalogFile(path: p, sha256: try FileDigest.sha256(folder.appendingPathComponent(p)), bytes: try FileDigest.bytes(folder.appendingPathComponent(p)))
        }
        let doc = CatalogDocument(sequence: 4, sourceCommit: String(repeating: "a", count: 40), models: [CatalogModel(manifest: model, files: files)])
        let data = try doc.encoded(); let key = Curve25519.Signing.PrivateKey(); let signature = try key.signature(for: data)
        let dir = resources.appendingPathComponent("catalog"); try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try data.write(to: dir.appendingPathComponent("catalog.json")); try signature.write(to: dir.appendingPathComponent("catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
        let source = try CatalogBootstrap.verifiedBundleSource(resources: resources)
        let installer = ModelFolderInstaller(support: root.appendingPathComponent("support"))
        try await installer.accept(data: data, signature: signature, publicKey: key.publicKey.rawRepresentation)
        let installed = try await installer.install(id: model.id, source: source)
        #expect(try ModelManifest.load(installed) == model)
        do { _ = try await source.read(path: "models/\(model.id)/model.toml", commit: String(repeating: "b", count: 40)); Issue.record("wrong commit accepted") } catch {}
        try Data("unexpected".utf8).write(to: folder.appendingPathComponent("extra.py"))
        do { _ = try await source.read(path: "models/\(model.id)/model.toml", commit: doc.sourceCommit); Issue.record("unlisted file accepted") } catch {}
        // Preserve the unexpected file, move it outside this inventory to exercise checksum rejection separately.
        try FileManager.default.moveItem(at: folder.appendingPathComponent("extra.py"), to: root.appendingPathComponent("extra.py"))
        try Data("tampered".utf8).write(to: folder.appendingPathComponent("model.toml"))
        do { _ = try await source.read(path: "models/\(model.id)/model.toml", commit: doc.sourceCommit); Issue.record("tampered file accepted") } catch {}
    }
    @Test func localDiscoveryRejectsSignedShadowAndTracksContent() throws {
        let root = try temp(); let parent = root.appendingPathComponent("local-models"); try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let folder = try fixture(parent)
        #expect(throws: (any Error).self) { try LocalModelFolders.discover(support: root, signedIDs: ["qwen3.8-27b"]) }
        let first = try #require(LocalModelFolders.discover(support: root, signedIDs: []).first)
        #expect(!first.requiresCodeConsent)
        try Data("changed README".utf8).write(to: folder.appendingPathComponent("README.md"))
        let second = try #require(LocalModelFolders.discover(support: root, signedIDs: []).first)
        #expect(first.contentDigest != second.contentDigest)
    }
    @Test func scaffoldUsesOnlyFixtureMetadataAndReportsGitPins() throws {
        let root = try temp()
        let data = Data(
            ##"{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pipeline_tag":"text-to-speech","downloads":42,"likes":7,"cardData":{"license":"apache-2.0","language":["en"]},"siblings":[{"rfilename":"config.json","size":4},{"rfilename":"model.safetensors","size":9000000000,"lfs":{"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","size":9000000000}}]}"##
                .utf8)
        let metadata = try HFModelMetadata.decode(data, repo: "test/Fixture-Voice")
        let kind = KindSpec(id: "voices", title: "Voices", order: 0, columns: [], facets: ModelFacets(modalitiesIn: ["text"], modalitiesOut: ["audio"]))
        let result = try HFScaffolder.write(metadata: metadata, kind: kind, parent: root, date: "2026-10-03")
        let model = try TOMLParser.decode(
            ModelManifest.self, from: String(contentsOf: URL(fileURLWithPath: result.directory).appendingPathComponent("model.toml"), encoding: .utf8))
        #expect(model.scaffold == true)
        #expect(model.problems(folder: URL(fileURLWithPath: result.directory), requirePins: false).contains { $0.contains("unfinished") })
        #expect(model.popularity?.hfDownloads == 42); #expect(model.weights.bytes?["model.safetensors"] == 9_000_000_000)
        #expect(model.weights.sha256?["config.json"] == nil); #expect(result.pending.contains { $0.contains("config.json") })
        #expect(!FileManager.default.fileExists(atPath: result.directory + "/model.safetensors"))
        #expect(throws: (any Error).self) { try HFScaffolder.write(metadata: metadata, kind: kind, parent: root, date: "2026-10-03") }
        #expect(throws: (any Error).self) { try ModelManifest.load(URL(fileURLWithPath: result.directory)) }
    }
    @Test func allKindsResolveThroughSignedSharedRootDuringOfflineInstallAndTamperFails() async throws {
        let root = try temp(); let resources = root.appendingPathComponent("App/Contents/Resources")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        // Assemble the committed checkout through the production signed-inventory copier.
        let checkout = root.appendingPathComponent("checkout")
        try FileManager.default.createDirectory(at: checkout, withIntermediateDirectories: true)
        let archive = root.appendingPathComponent("published.tar")
        try GitRead.run(repo: repo, args: ["archive", "--format=tar", "HEAD", "models", "Resources/catalog", "Resources/kinds", "scripts/bundle-inventory.py"]).write(to: archive)
        let extract = Process(); extract.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        extract.arguments = ["-xf", archive.path, "-C", checkout.path]; extract.standardError = FileHandle.nullDevice
        try extract.run(); let deadline = Date().addingTimeInterval(30)
        while extract.isRunning && Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        if extract.isRunning { extract.terminate(); throw ManifestError.invalid("fixture archive extraction timed out") }
        guard extract.terminationStatus == 0 else { throw ManifestError.invalid("fixture archive extraction failed") }
        // Implementation branches leave the signed release catalog untouched. Refresh only
        // this private archive's inventory; its ephemeral signature and tamper checks follow below.
        let fixtureCatalog = checkout.appendingPathComponent("Resources/catalog/catalog.json")
        var inventory = try JSONDecoder().decode(CatalogDocument.self, from: Data(contentsOf: fixtureCatalog))
        for index in inventory.models.indices {
            let folder = checkout.appendingPathComponent("models/" + inventory.models[index].id)
            inventory.models[index].files = try inventory.models[index].files.map { file in
                CatalogFile(
                    path: file.path, sha256: try FileDigest.sha256(folder.appendingPathComponent(file.path)),
                    bytes: try FileDigest.bytes(folder.appendingPathComponent(file.path)))
            }
        }
        let currentKinds = checkout.appendingPathComponent("Resources/kinds")
        inventory.kindFiles = try SafePath.files(in: currentKinds).map { path in
            CatalogFile(
                path: path, sha256: try FileDigest.sha256(currentKinds.appendingPathComponent(path)), bytes: try FileDigest.bytes(currentKinds.appendingPathComponent(path)))
        }
        try inventory.encoded().write(to: fixtureCatalog)
        try FileManager.default.copyItem(at: checkout.appendingPathComponent("Resources/catalog"), to: resources.appendingPathComponent("catalog"))
        let copy = Process(); copy.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        copy.arguments = ["-I", "-B", checkout.appendingPathComponent("scripts/bundle-inventory.py").path, "copy", checkout.path, resources.path]
        copy.standardOutput = FileHandle.nullDevice; copy.standardError = FileHandle.nullDevice
        try copy.run(); let copyDeadline = Date().addingTimeInterval(30)
        while copy.isRunning && Date() < copyDeadline { try await Task.sleep(for: .milliseconds(10)) }
        if copy.isRunning { copy.terminate(); throw ManifestError.invalid("fixture inventory copy timed out") }
        guard copy.terminationStatus == 0 else { throw ManifestError.invalid("fixture inventory copy failed") }
        let kinds = resources.appendingPathComponent("kinds")
        let kindFiles = try SafePath.files(in: kinds).map { p in
            CatalogFile(path: p, sha256: try FileDigest.sha256(kinds.appendingPathComponent(p)), bytes: try FileDigest.bytes(kinds.appendingPathComponent(p)))
        }
        let modelRoot = resources.appendingPathComponent("models")
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        var entries: [CatalogModel] = []
        let sourceFolders = try FileManager.default.contentsOfDirectory(at: modelRoot, includingPropertiesForKeys: [.isDirectoryKey]).filter {
            !$0.lastPathComponent.hasPrefix(".") && (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
        for sourceFolder in sourceFolders {
            let folder = sourceFolder
            let model = try ModelManifest.load(folder, kindsDirectory: kinds)
            // The original folders keep their schema; every later integration is schema 3, except a model that joins the voices
            // family: that family's streamed speak/voices/clone_voice protocol is schema 2 (no output schemas, no schema-3 artifact
            // routing), so Drama 3 and VoxCPM2 declare schema 2 like the other voices folders; DeepFilterNet3, MossFormer2 and
            // UniPASE follow RE-USE's enhancement folder layout, also schema 2.
            let schemaTwo: Set = [
                "laya-english", "laya-multilingual", "laya-typed-decisions", "von-1.1", "von-1.2", "d1-3b", "kokoro-82m", "pocket-tts", "breeze-tts-2",
                "qwen3-tts-base", "qwen3-tts-voicedesign", "stable-audio-3-medium", "stable-audio-3-small-sfx", "re-use", "qwen3.8-27b", "flux2-klein-4b",
                "fish-drama-3", "voxcpm2", "deepfilternet3", "mossformer2-se-48k", "unipase"
            ]
            #expect(model.schema == (schemaTwo.contains(model.id) ? 2 : 3), "\(model.id)")
            guard CatalogShipping.includes(model) else { continue }
            let files = try SafePath.files(in: folder).map { p in
                CatalogFile(path: p, sha256: try FileDigest.sha256(folder.appendingPathComponent(p)), bytes: try FileDigest.bytes(folder.appendingPathComponent(p)))
            }
            entries.append(CatalogModel(manifest: model, files: files))
        }
        #expect(entries.count == inventory.models.count && entries.count >= 16)
        #expect(entries.allSatisfy { $0.manifest.kind != "chat" || CatalogShipping.includes($0.manifest) })
        let doc = CatalogDocument(sequence: 4, sourceCommit: String(repeating: "a", count: 40), models: entries, kindFiles: kindFiles)
        let data = try doc.encoded(); let key = Curve25519.Signing.PrivateKey(); let signature = try key.signature(for: data)
        let cat = resources.appendingPathComponent("catalog"); try FileManager.default.createDirectory(at: cat, withIntermediateDirectories: true)
        try data.write(to: cat.appendingPathComponent("catalog.json")); try signature.write(to: cat.appendingPathComponent("catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
        let source = try CatalogBootstrap.verifiedBundleSource(resources: resources)
        let reads = InventoryReadProbe(source)
        let installer = ModelFolderInstaller(support: root.appendingPathComponent("support"), kindsDirectory: source.kindsDirectory)
        try await installer.accept(data: data, signature: signature, publicKey: key.publicKey.rawRepresentation)
        for entry in entries {
            let installed = try await installer.install(id: entry.id, source: reads)
            #expect(try ModelManifest.load(installed, kindsDirectory: kinds) == entry.manifest)
        }
        // A fresh installer without a bundle-root hint prepares the SAME signed kinds once, not per model.
        let cached = ModelFolderInstaller(support: root.appendingPathComponent("cached-support"))
        try await cached.accept(data: data, signature: signature, publicKey: key.publicKey.rawRepresentation)
        let folder = try await cached.install(id: "laya-english", source: reads)
        let cachedKinds = try #require(await cached.verifiedKindsDirectory())
        #expect(try ModelManifest.load(folder, kindsDirectory: cachedKinds).id == "laya-english")
        let counts = await reads.counts()
        #expect(counts.individualModels == 0 && counts.batches == entries.count + 1)
        // Optional prefetch may be unavailable: mandatory authenticated reads still install.
        let fallback = ModelFolderInstaller(support: root.appendingPathComponent("fallback-support"), kindsDirectory: source.kindsDirectory)
        try await fallback.accept(data: data, signature: signature, publicKey: key.publicKey.rawRepresentation)
        let omitted = InventoryReadProbe(source, omitBatch: true)
        let fallbackFolder = try await fallback.install(id: "laya-english", source: omitted)
        #expect(try ModelManifest.load(fallbackFolder, kindsDirectory: kinds).id == "laya-english")
        let fallbackCounts = await omitted.counts()
        let expectedReads = try #require(entries.first { $0.id == "laya-english" }).files.count
        #expect(fallbackCounts.individualModels == expectedReads && fallbackCounts.batches == 1)
        let victim = kinds.appendingPathComponent("decisions/ops/decide.input.json")
        let original = try Data(contentsOf: victim)
        try Data("tampered schema".utf8).write(to: victim)
        #expect(throws: (any Error).self) { try CatalogBootstrap.verifiedBundleSource(resources: resources) }
        do { _ = try await installer.verifyInstalled(id: "laya-english"); Issue.record("tampered shared schema accepted") } catch {}
        try original.write(to: victim)
        try FileManager.default.createSymbolicLink(atPath: kinds.appendingPathComponent("unlisted.json").path, withDestinationPath: victim.path)
        #expect(throws: (any Error).self) { try doc.verifyKinds(directory: kinds) }
    }

}

extension SchemaV2Tests {
    @Test func utf8ByteLimitIsSeparateFromCharacterLimitAndSchemaChecked() throws {
        let schema: JSONObject = ["type": "string", "maxLength": 1024, "x-arsenal-maxUTF8Bytes": 1024]
        try ModelSchemaValidation.checkSchema(schemaData: WireJSON.encode(schema))
        try OperationSchemaValidator.validate(.string(String(repeating: "🛑", count: 256)), schema: .object(schema))
        #expect(throws: WorkerFailure.self) {
            try OperationSchemaValidator.validate(.string(String(repeating: "🛑", count: 257)), schema: .object(schema))
        }
        #expect(throws: (any Error).self) { try ModelSchemaValidation.checkSchema(schemaData: WireJSON.encode(["x-arsenal-maxUTF8Bytes": -1])) }
    }
}

private actor InventoryReadProbe: CatalogSource {
    let base: BundleCatalogSource
    let omitBatch: Bool
    var individualModels = 0, batches = 0
    init(_ base: BundleCatalogSource, omitBatch: Bool = false) { self.base = base; self.omitBatch = omitBatch }
    func read(path: String, commit: String) async throws -> Data {
        if path.hasPrefix("models/") { individualModels += 1 }
        return try await base.read(path: path, commit: commit)
    }
    func readAvailable(paths: [String], commit: String) async -> [String: Data] {
        if paths.contains(where: { $0.hasPrefix("models/") }) { batches += 1 }
        if omitBatch { return [:] }
        return await base.readAvailable(paths: paths, commit: commit)
    }
    func counts() -> (individualModels: Int, batches: Int) { (individualModels, batches) }
}
