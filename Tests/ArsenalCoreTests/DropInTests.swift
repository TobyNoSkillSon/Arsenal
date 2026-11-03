// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct DropInTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    var fixture: URL { repo.appendingPathComponent("Tests/Fixtures/DropIn/fixture-layers") }
    @Test func legacySixteenDescriptorsRemainEquivalent() throws {
        let baseline = try JSONDecoder().decode([JSONObject].self, from: Data(contentsOf: repo.appendingPathComponent("Tests/Fixtures/DropIn/legacy-descriptors.json")))
        #expect(baseline.count == 16)
        let kindsDir = repo.appendingPathComponent("Resources/kinds"), kinds = try KindSpec.load(directory: kindsDir)
        for record in baseline {
            let id = try #require(record["id"]?.string), folder = intakeFixtureRepository.appendingPathComponent("models/" + id)
            var parser = TOMLParser(try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
            let raw = try parser.parse().mapValues(\.json)
            let parsed = try WireJSON.decode(JSONSerialization.data(withJSONObject: raw, options: [.sortedKeys]))
            var expected = try #require(record["manifest"]?.object)
            if id == "qwen3.8-27b" {
                // Retain the historical descriptor; account only for the authorized maximum-context migration.
                expected["context"] = 262144
                var settings = expected["setting"]?.array ?? []
                for index in settings.indices where settings[index]["id"]?.string == "context_limit" {
                    var declaration = settings[index].object ?? [:]
                    declaration["default"] = 262144; declaration["choices"] = [262144]
                    declaration["help"] = "Model maximum, read from config. Memory grows with the conversation."
                    settings[index] = .object(declaration)
                }
                // Settings panel (stage 2): the runner's top_p and max_tokens, declared before the system prompt.
                let extra: [JSONValue] = [
                    [
                        "applies": "request", "default": 1.0, "id": "top_p", "help": "Samples only from the most likely words that add up to this share.",
                        "maps_to": "request.top_p", "max": 1, "min": 0.05, "step": 0.05, "title": "Top P", "type": "slider"
                    ],
                    [
                        "applies": "request", "default": 1024, "id": "max_tokens", "help": "Longest answer, in tokens.", "maps_to": "request.max_tokens",
                        "max": 8192, "min": 1, "step": 1, "title": "Answer limit", "type": "number"
                    ]
                ]
                let prompt = settings.firstIndex { $0["id"]?.string == "system_prompt" } ?? settings.endIndex
                settings.insert(contentsOf: extra, at: prompt)
                expected["setting"] = .array(settings)
            }
            if id == "flux2-klein-4b" {
                // Settings panel (stage 2): the runner's free width and height replace the resolution choice, and the
                // seed has no default, so each image gets a new random one until one is pinned.
                func size(_ id: String, _ title: String, _ what: String) -> JSONValue {
                    [
                        "applies": "request", "default": 1024, "excludes": ["resolution"], "id": .string(id),
                        "help": .string("Image \(what) in pixels, a multiple of 16."),
                        "maps_to": .string("request." + id), "max": 2048, "min": 64, "step": 16, "title": .string(title), "type": "number"
                    ]
                }
                expected["setting"] = .array(
                    (expected["setting"]?.array ?? []).flatMap { value -> [JSONValue] in
                        guard var declaration = value.object else { return [value] }
                        if declaration["id"]?.string == "resolution" { return [size("width", "Width", "width"), size("height", "Height", "height")] }
                        guard declaration["id"]?.string == "seed" else { return [value] }
                        declaration["default"] = nil
                        declaration["help"] = "Starting noise: empty is a new random seed each request; the same seed and prompt give the same image."
                        return [.object(declaration)]
                    })
                // The retired resolution setting stays accepted as an alias for width and height.
                func square(_ n: Double) -> JSONValue { ["width": .number(n), "height": .number(n)] }
                expected["setting_alias"] = [["id": "resolution", "values": ["512x512": square(512), "768x768": square(768), "1024x1024": square(1024)]]]
            }
            // Qualified intake evolves recipes without rewriting the historical descriptor fixture.
            if id == "flux2-klein-4b" { expected["derive"] = ["int8": "mlx-affine-g64:flux2-klein-4b", "int4": "mlx-affine-g64:flux2-klein-4b"] }
            if id == "stable-audio-3-medium" {
                expected["derive"] = ["fp16": "cast:stable-audio-3-medium", "int8": "intake-v1:stable-audio-3-medium", "int4": "intake-v1:stable-audio-3-medium"]
            }
            // Current model contracts intentionally evolve the retained historical descriptors.
            if id.hasPrefix("laya-") { expected["context"] = id == "laya-english" ? 512 : 1024 }
            if ["stable-audio-3-medium", "stable-audio-3-small-sfx"].contains(id) { expected["tools"] = ["sound"] }
            if ["kokoro-82m", "pocket-tts", "breeze-tts-2", "qwen3-tts-base", "qwen3-tts-voicedesign"].contains(id) { expected["tools"] = ["voice", "sound"] }
            // Full official names: Qwen3-TTS-12Hz-1.7B-Base and -VoiceDesign.
            if id == "qwen3-tts-base" { expected["name"] = "Qwen3 TTS 12Hz 1.7B Base" }
            if id == "qwen3-tts-voicedesign" { expected["name"] = "Qwen3 TTS 12Hz 1.7B VoiceDesign" }
            // Reviewed model contracts: these engines infer/support language without using request.language.
            if ["breeze-tts-2", "pocket-tts"].contains(id) {
                expected["setting"] = .array((expected["setting"]?.array ?? []).filter { $0["id"]?.string != "language" })
            }
            if ["breeze-tts-2", "qwen3-tts-voicedesign"].contains(id) {
                expected["setting"] = .array(
                    (expected["setting"]?.array ?? []).map { value in
                        guard var setting = value.object, setting["id"] == "instructions" else { return value }
                        setting["excludes"] = ["description", "reference", "voice"]
                        return .object(setting)
                    })
            }
            if ["d1-3b", "kokoro-82m", "pocket-tts", "stable-audio-3-small-sfx"].contains(id) {
                var derive = expected["derive"]?.object ?? [:]
                derive["int4"] = .string("mlx-affine-g64:" + id); expected["derive"] = .object(derive)
            }
            if id == "d1-3b" {
                expected["derive"] = ["int8": "mlx-affine-g64:d1-3b", "int4": "mlx-affine-g64:d1-3b"]
                // The LFM licence caps commercial use by annual revenue (28 Oct): commercial is false, its conditions explain.
                if var licence = expected["licence"]?.object { licence["commercial"] = false; expected["licence"] = .object(licence) }
            }
            if id == "stable-audio-3-small-sfx" {
                // All-component conversion replaces the former DiT-only recipe for both derived tiers.
                expected["derive"] = ["fp16": "cast:stable-audio-3-small-sfx", "int8": "intake-v2:stable-audio-3-small-sfx", "int4": "intake-v2:stable-audio-3-small-sfx"]
            }
            if ["breeze-tts-2", "qwen3-tts-base", "qwen3-tts-voicedesign", "pocket-tts", "kokoro-82m"].contains(id),
                let derived = parsed["derive"]?.object, derived["bf16"] != nil || derived["fp16"] != nil
            {
                let half = derived["fp16"] != nil ? "fp16" : "bf16"
                let bindings = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("gate/evidence"), includingPropertiesForKeys: nil)
                    .filter { $0.lastPathComponent.hasPrefix("quality-binding-") }.sorted { $0.path < $1.path }
                let binding = try WireJSON.decode(Data(contentsOf: #require(bindings.last)))
                let intake = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("results/intake/" + (try #require(binding["run"]?.string)) + ".json")))
                // Recipe identity is evidence-bound, including the matrix fallback list, rather than a copied component default.
                expected["derive"] = intake["recipes"]
                #expect(derived[half] == .string("cast:" + id))
                if id == "breeze-tts-2" { expected["vendor"] = nil }
            }
            if id == "pocket-tts", var weights = expected["weights"]?.object {
                // 2 Nov: the cosette (Expresso) and jean (EARS) presets are CC BY-NC 4.0 data, so they are no longer pinned.
                let dropped: Set<String> = ["embeddings/cosette.safetensors", "embeddings/jean.safetensors"]
                weights["files"] = weights["files"]?.array.map { .array($0.filter { !dropped.contains($0.string ?? "") }) }
                for key in ["bytes", "sha256"] { weights[key] = weights[key]?.object.map { .object($0.filter { !dropped.contains($0.key) }) } }
                expected["weights"] = .object(weights)
            }
            if ["kokoro-82m", "pocket-tts", "qwen3-tts-base", "qwen3-tts-voicedesign"].contains(id), var licence = expected["licence"]?.object {
                // 2 Nov: the licence links pointed at LICENSE files these repositories do not have; they point at the pinned cards.
                licence["url"] = parsed["licence"]?["url"]; expected["licence"] = .object(licence)
            }
            if ["von-1.1", "von-1.2"].contains(id) {
                // Uniform half source and streamed embedding/all-matrix quantization replace the legacy scorer-FP32 recipe.
                expected["derive"] = [
                    "bf16": "cast:von-bf16-2", "int8": "mlx-affine-g64:von-all-v2",
                    "int4": "mlx-affine-g64:von-all-v2"
                ]
            }
            if ["von-1.1", "von-1.2", "re-use"].contains(id) {
                expected["format"] = [["id": "bf16", "bits_per_weight": 16, "storage": "bf16", "source": "derived", "runners": ["native"]]]
            }
            if id == "re-use" {
                // int8 and int4 withdrawn on 22 Oct (removedQualityTiers): slower than BF16 for about 0.2 GB less peak.
                expected["derive"] = ["bf16": "cast:re-use"]
                // The declared denoise task (22 Oct) routes noise searches here rather than to AudioSR's shared `enhance` op.
                expected["tasks"] = ["audio-to-audio", "denoise"]
                // Re-pinned on 2 Nov to NVIDIA's own repository (OpenMDW-1.1, new weights); Get casts its PyTorch checkpoint.
                #expect(parsed["weights"]?["repo"]?.string == "nvidia/NVIDIA-NemotronLabs-RE-USE" && parsed["licence"]?["spdx"]?.string == "OpenMDW-1.1")
                for key in ["weights", "licence", "summary"] { expected[key] = parsed[key] }
            }
            if var derive = expected["derive"]?.object {
                // Tiers withdrawn since (removedQualityTiers): Kokoro's int8 on 22 Oct (no faster, no smaller than FP16), and every
                // quantized tier of a model under 1 GB at 16-bit on 26 Oct (16-bit only).
                for removed in removedQualityTiers where removed.model == id { derive[removed.tier] = nil }
                expected["derive"] = derive.isEmpty ? nil : .object(derive)
            }
            // Display-only memory payload evidence was added after the retained descriptor snapshot.
            expected["memory"] = parsed["memory"]
            // The purpose shown under each name (16 Oct) was added after the retained descriptor snapshot.
            #expect(parsed["purpose"]?.string?.isEmpty == false)
            expected["purpose"] = parsed["purpose"]
            #expect(parsed == expected)
            let manifest = try ModelManifest.load(folder, kinds: Set(kinds.map(\.id)), kindsDirectory: kindsDir)
            let ops = manifest.resolvedOperations(kind: kinds.first { $0.id == manifest.kind })
            #expect(ops.map(\.name) == record["operations"]?.array?.compactMap { $0["name"]?.string })
            for op in ops {
                let prior = try #require(record["operations"]?.array?.first { $0["name"]?.string == op.name })
                #expect(op.output == prior["output"]?.string && op.stream == prior["stream"]?.bool && op.input == prior["input"]?.string)
            }
            #expect(manifest.resolvedTools == (expected["tools"]?.array?.compactMap(\.string) ?? []))
        }
    }
    @Test func embeddedKindAndEvidenceFingerprints() throws {
        let model = try ModelManifest.load(fixture, kinds: [], kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        let resolved = try EmbeddedKind.resolve(model: model, folder: fixture)
        let embedded = try #require(resolved)
        #expect(model.schema == 3 && model.resolvedTools == ["operation"] && embedded.spec.suite.isEmpty)
        #expect(embedded.spec.primaryOperation == "separate" && model.resolvedOperations(kind: embedded.spec).count == 1)
        let local = try LocalModelFolders.discoverSnapshot(fixture)
        var changed = local
        changed = LocalModelFolder(
            manifest: model, directory: fixture, fileDigests: local.fileDigests.merging(["results/new.json": String(repeating: "1", count: 64)]) { _, b in b },
            contentDigest: "changed")
        #expect(try ImplementationEvidence.fingerprint(changed) == ImplementationEvidence.fingerprint(local))
        #expect(ImplementationEvidence.fixtures(changed) == ImplementationEvidence.fixtures(local))
        let different = LocalModelFolder(
            manifest: model, directory: fixture, fileDigests: local.fileDigests.merging(["runners/swift/Sources/FixtureLayers/main.swift": "different"]) { _, b in b },
            contentDigest: "changed")
        #expect(try ImplementationEvidence.fingerprint(different) != ImplementationEvidence.fingerprint(local))
        let changedLock = LocalModelFolder(
            manifest: model, directory: fixture, fileDigests: local.fileDigests.merging(["runners/swift/Package.swift": "different-lock"]) { _, b in b },
            contentDigest: "changed")
        #expect(try ImplementationEvidence.fingerprint(changedLock) != ImplementationEvidence.fingerprint(local))
    }
    /// A settings panel changes the folder's content digest (consent and revisions see it) but not the implementation
    /// fingerprint, so a layout-only edit never marks the runner's evidence stale.
    @Test func aSettingsPanelChangesTheDigestButNotTheImplementation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("panel-fingerprint-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let folder = root.appendingPathComponent(fixture.lastPathComponent)
        try FileManager.default.copyItem(at: fixture, to: folder)
        let before = try LocalModelFolders.discoverSnapshot(folder)
        try "schema = 1\n".write(to: folder.appendingPathComponent(SettingsPanel.fileName), atomically: true, encoding: .utf8)
        let after = try LocalModelFolders.discoverSnapshot(folder)
        #expect(after.contentDigest != before.contentDigest)
        #expect(try ImplementationEvidence.fingerprint(after) == ImplementationEvidence.fingerprint(before))
        try "schema = 1\nhidden = []\n".write(to: folder.appendingPathComponent(SettingsPanel.fileName), atomically: true, encoding: .utf8)
        #expect(try LocalModelFolders.discoverSnapshot(folder).contentDigest != after.contentDigest)
    }
    @Test func localSwiftEntryRequiresReviewedSourceAndExactExecutable() async throws {
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-dropin-swift-admission-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let folder = root.appendingPathComponent("fixture-layers")
        try FileManager.default.copyItem(at: fixture, to: folder)
        let binary = folder.appendingPathComponent("runners/swift/bin/admission-fixture")
        try FileManager.default.createDirectory(at: binary.deletingLastPathComponent(), withIntermediateDirectories: true)
        // Admission checks regular executable/source presence; OS/hello and parity gates establish executable validity.
        try Data("fixture binary bytes".utf8).write(to: binary)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
        var model = try ModelManifest.load(folder, kinds: [], kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        var runner = model.runner[0]; runner.language = "swift"; runner.python = nil; runner.lock = nil; runner.entry = "runners/swift/bin/admission-fixture"
        model.runner = [runner]
        #expect(!model.problems(folder: folder, kinds: []).isEmpty)
        let files = try SafePath.files(in: folder).filter { $0.hasPrefix("runners/swift/") && $0 != "runners/swift/build-receipt.json" && !$0.hasSuffix("build.sh") }
        let hashes = try Dictionary(uniqueKeysWithValues: files.map { ($0, JSONValue.string(try FileDigest.sha256(SafePath.resolve($0, inside: folder)))) })
        try WireJSON.encode(["command": "admission fixture", "toolchain": "test", "sha256": .object(hashes)]).write(
            to: folder.appendingPathComponent("runners/swift/build-receipt.json"))
        #expect(model.problems(folder: folder, kinds: []).isEmpty)
        try BundledWorkerArtifact(helpers: folder.appendingPathComponent("absent-helpers")).validateEngine(model: model, runner: runner)
        let paths = ArsenalPaths(support: folder.appendingPathComponent("support"), cache: folder.appendingPathComponent("cache")); try paths.create()
        let launch = try await BundledWorkerArtifact().launch(model: model, runner: runner, directory: folder, paths: paths)
        #expect(launch.executable == binary && launch.arguments.isEmpty)
        let local = try LocalModelFolders.discoverSnapshot(fixture)
        let changed = LocalModelFolder(
            manifest: model, directory: folder,
            fileDigests: local.fileDigests.merging([runner.entry!: "different-binary"]) { _, new in new }, contentDigest: "different")
        #expect(try ImplementationEvidence.fingerprint(changed) != ImplementationEvidence.fingerprint(local))
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: binary.path)
        #expect(!model.problems(folder: folder, kinds: []).isEmpty)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
        try FileManager.default.moveItem(at: folder.appendingPathComponent("runners/swift/Package.swift"), to: folder.appendingPathComponent("runners/swift/Package.saved"))
        #expect(!model.problems(folder: folder, kinds: []).isEmpty)
        runner.entry = "runners/swift/bin/../escape"; model.runner = [runner]
        #expect(!model.problems(folder: folder, kinds: []).isEmpty)
    }
    @Test func annotatedInputConfinementRecursesAndChecksActualMIME() throws {
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-dropin-input-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let inbox = root.appendingPathComponent("inbox/sample")
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        let file = inbox.appendingPathComponent("fake.png"); try Data("not an image".utf8).write(to: file)
        let annotation: JSONObject = ["type": "object", "x-arsenal-file": ["accept": ["image/png"]], "properties": ["path": ["type": "string"]]]
        let schema: JSONObject = ["type": "object", "properties": ["nested": ["type": "array", "items": .object(annotation)]]]
        do {
            try OperationArtifacts.input(["nested": [["path": .string(file.path)]]], schema: schema, support: root)
            Issue.record("Invalid image input was accepted")
        } catch let error as WorkerFailure { #expect(error.code == "invalid_request") }
        #expect(throws: WorkerFailure.self) { try OperationArtifacts.input(["nested": [["path": "/etc/hosts"]]], schema: schema, support: root) }
        // A NUL hides a path outside the inbox from standardization while C file calls stop at it: the same real
        // inbox PNG is accepted on its own and refused behind one.
        let pixel = inbox.appendingPathComponent("pixel.png")
        try Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLttAAAAABJRU5ErkJggg==")!.write(to: pixel)
        try OperationArtifacts.input(["nested": [["path": .string(pixel.path)]]], schema: schema, support: root)
        let hidden = "/etc/hosts\u{0}/../.." + pixel.path
        #expect(throws: WorkerFailure.self) { try OperationArtifacts.input(["nested": [["path": .string(hidden)]]], schema: schema, support: root) }
        #expect(throws: WorkerFailure.self) { try OperationArtifacts.input(["nested": [["path": "https://example.com/image.png"]]], schema: schema, support: root) }
        let union: JSONObject = ["$defs": ["file": .object(annotation)], "anyOf": [["$ref": "#/$defs/file"], ["type": "string"]]]
        #expect(throws: WorkerFailure.self) { try OperationArtifacts.input(["path": "/etc/hosts"], schema: union, support: root) }
        let symlink = inbox.appendingPathComponent("alias.png")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: file)
        #expect(throws: WorkerFailure.self) { try OperationArtifacts.input(["path": .string(symlink.path)], schema: annotation, support: root) }
    }
    @Test func embeddedKindsRejectConflictingTreesWithoutReplacingAcceptedFolder() async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-dropin-conflict-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let first = support.appendingPathComponent("local-models/fixture-layers"), second = support.appendingPathComponent("local-models/other-layers")
        try FileManager.default.createDirectory(at: first.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: fixture, to: first); try FileManager.default.copyItem(at: fixture, to: second)
        let manifest = second.appendingPathComponent("model.toml")
        try String(contentsOf: manifest, encoding: .utf8).replacingOccurrences(of: "id = \"fixture-layers\"", with: "id = \"other-layers\"").write(
            to: manifest, atomically: true, encoding: .utf8)
        let scanner = LocalFolderScanner()
        #expect(try await scanner.scan(support: support, signedIDs: [], kinds: []).folders.count == 2)
        let kind = second.appendingPathComponent("kinds/local.layers.toml")
        try String(contentsOf: kind, encoding: .utf8).replacingOccurrences(of: "title = \"Layers\"", with: "title = \"Conflicting\"").write(
            to: kind, atomically: true, encoding: .utf8)
        let result = try await scanner.scan(support: support, signedIDs: [], kinds: [])
        #expect(result.folders.map { $0.manifest.id } == ["fixture-layers"] && result.problems.count == 1)
        #expect(throws: ManifestError.self) { try LocalModelFolders.discover(support: support, signedIDs: [], kinds: []) }
    }

    @Test func inputAllowsSymlinkAboveSupportButRejectsOneInside() throws {
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-fix31-links-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let actual = root.appendingPathComponent("actual"), alias = root.appendingPathComponent("alias")
        try FileManager.default.createDirectory(at: actual.appendingPathComponent("inbox"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: actual)
        let image = actual.appendingPathComponent("inbox/pixel.png")
        try Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLttAAAAABJRU5ErkJggg==")!.write(to: image)
        let annotation: JSONObject = ["type": "object", "x-arsenal-file": ["accept": ["image/png"]], "properties": ["path": ["type": "string"]]]
        try OperationArtifacts.input(["path": .string(alias.appendingPathComponent("inbox/pixel.png").path)], schema: annotation, support: alias)
        let inside = actual.appendingPathComponent("inbox/inside.png")
        try FileManager.default.createSymbolicLink(at: inside, withDestinationURL: image)
        #expect(throws: WorkerFailure.self) {
            try OperationArtifacts.input(["path": .string(alias.appendingPathComponent("inbox/inside.png").path)], schema: annotation, support: alias)
        }
    }
    @Test func bundledIntegrationReferencesMatchDocs() throws {
        for (name, path) in [
            ("SKILL.md", "skill/SKILL.md"), ("DECISIONS.md", "skill/DECISIONS.md"), ("ADDING.md", "skill/ADDING.md"),
            ("MODEL-FOLDER.md", "docs/MODEL-FOLDER.md"), ("RUNNERS.md", "docs/RUNNERS.md"), ("ADDING-A-MODEL.md", "docs/ADDING-A-MODEL.md"),
            ("MODELS.md", "docs/MODELS.md"), ("ADD-A-MODEL.md", "docs/ADD-A-MODEL.md")
        ] {
            #expect(
                IntegrationReferences.documents[name]?.trimmingCharacters(in: .whitespacesAndNewlines)
                    == (try String(contentsOf: repo.appendingPathComponent(path), encoding: .utf8)).trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }
}
