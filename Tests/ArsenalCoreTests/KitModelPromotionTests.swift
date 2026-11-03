// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Testing
@testable import ArsenalCore

/// Kit models (Workers/Kit) ship in the signed catalog and run in the bundled worker: each model folder names
/// worker = "kit-<id>", the dispatcher has its link, scripts/pkg-lib.sh ships the link, and the folder validates.
struct KitModelPromotionTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func text(_ path: String) throws -> String { try String(contentsOf: repo.appendingPathComponent(path), encoding: .utf8) }
    private func matches(_ pattern: String, in text: String) throws -> [String] {
        let regex = try NSRegularExpression(pattern: pattern, options: [.anchorsMatchLines])
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).map { String(text[Range($0.range(at: 1), in: text)!]) }
    }
    /// Model id to module, from each module's Entry.swift.
    private func modules() throws -> [String: String] {
        let sources = repo.appendingPathComponent("Workers/Kit/Sources")
        var result: [String: String] = [:]
        for module in try FileManager.default.contentsOfDirectory(atPath: sources.path) where !module.hasPrefix(".") {
            let entry = try text("Workers/Kit/Sources/\(module)/Entry.swift")
            let id = try #require(try matches(#"^    public static let modelID = "(.+)"$"#, in: entry).first)
            #expect(entry.contains("public enum \(module) {"), "\(module)/Entry.swift")
            result[id] = module
        }
        return result
    }
    private func kitFolders() throws -> [String: ModelManifest] {
        var result: [String: ModelManifest] = [:]
        for id in try FileManager.default.contentsOfDirectory(atPath: repo.appendingPathComponent("models").path) where !id.hasPrefix(".") {
            let folder = repo.appendingPathComponent("models/\(id)")
            guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("model.toml").path) else { continue }
            let manifest = try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
            if manifest.runners.contains(where: { $0.worker?.hasPrefix("kit-") == true }) { result[id] = manifest }
        }
        return result
    }

    @Test func everyKitFolderHasItsModuleDispatcherLinkAndShippedFamily() throws {
        let modules = try modules(), folders = try kitFolders()
        #expect(modules.count == 38)
        #expect(Set(folders.keys) == Set(modules.keys))
        for (id, manifest) in folders {
            #expect(manifest.runners.map(\.worker) == ["kit-\(id)"], "\(id): one runner, worker kit-<id>")
            #expect(manifest.runners.allSatisfy { $0.entry == nil && $0.name == "kit" }, "\(id): the bundled kit runner, named as the kit's hello")
            #expect(manifest.kindFile == nil, "\(id): kit kinds are signed (Resources/kinds)")
            #expect(!FileManager.default.fileExists(atPath: repo.appendingPathComponent("models/\(id)/runners").path), "\(id): runner sources live in Workers/Kit")
        }
        let dispatcher = try text("Workers/Native/Sources/arsenal-worker-native/KitWorkers.swift")
        let links = try matches(#"^    ("arsenal-worker-kit-.+": \w+)\.run,?$"#, in: dispatcher).map { pair in
            pair.dropFirst(#""arsenal-worker-kit-"#.count).components(separatedBy: "\": ")
        }
        #expect(links.count == modules.count)
        #expect(Dictionary(uniqueKeysWithValues: links.map { ($0[0], $0[1]) }) == modules, "KitWorkers.swift maps each link to its module")
        #expect(Set(try matches(#"^import (Kit\w+)$"#, in: dispatcher)) == Set(modules.values))
        let package = try text("Workers/Kit/Package.swift"), root = try text("Workers/Package.swift")
        #expect(Set(try matches(#"^    "(Kit\w+)",?$"#, in: package)) == Set(modules.values), "Workers/Kit/Package.swift models")
        #expect(Set(try matches(#"^    "(Kit\w+)",?$"#, in: root)) == Set(modules.values), "Workers/Package.swift kitModels")
        let families = try UpdateVerification.tool(
            "/bin/bash", ["-c", "source \"$1\"; printf '%s\\n' \"${NATIVE_FAMILIES[@]}\"", "fixture", repo.appendingPathComponent("scripts/pkg-lib.sh").path])
        let shipped = Set(families.split(whereSeparator: \.isNewline).map(String.init))
        #expect(Set(shipped.filter { $0.hasPrefix("kit-") }) == Set(modules.keys.map { "kit-" + $0 }), "scripts/pkg-lib.sh NATIVE_FAMILIES")
        #expect(shipped.isSuperset(of: ["voices", "embed", "perception"]))
        // A kit worker must not be a normalized native family: the stored 16-bit snapshot records no normalization.
        #expect(Set(NativeHalfPolicy.normalizedWorkers).isDisjoint(with: shipped.filter { $0.hasPrefix("kit-") }))
    }

    @Test func kitModulesKeepTheKitTemplateAndNoEntryPoint() throws {
        let template = repo.appendingPathComponent("kit/runner/Sources/Runner")
        for module in try modules().values {
            let folder = repo.appendingPathComponent("Workers/Kit/Sources/\(module)")
            for name in ["Kit.swift", "Wire.swift"] {
                #expect(try Data(contentsOf: folder.appendingPathComponent(name)) == Data(contentsOf: template.appendingPathComponent(name)), "\(module)/\(name)")
            }
            #expect(!FileManager.default.fileExists(atPath: folder.appendingPathComponent("main.swift").path), "\(module): a library has no main.swift")
        }
    }

    /// The worker's hello must name the manifest's kind and every operation it resolves, or Load fails its handshake.
    @Test func everyKitModuleServesItsManifestKindAndOperations() throws {
        let kindsDirectory = repo.appendingPathComponent("Resources/kinds")
        let kinds = try KindSpec.load(directory: kindsDirectory)
        let modules = try modules()
        for (id, manifest) in try kitFolders() {
            let module = try #require(modules[id])
            let model = try text("Workers/Kit/Sources/\(module)/Model.swift")
            #expect(try matches(#"^let modelKind = "(.+)"$"#, in: model) == [manifest.kind], "\(id): modelKind")
            let list = try #require(try matches(#"^let modelOps = \[(.*)\]$"#, in: model).first, "\(id): modelOps")
            let served = Set(try matches(#""([a-z0-9_]+)""#, in: list))
            let declared = Set(manifest.resolvedOperations(kind: kinds.first { $0.id == manifest.kind }).map(\.name))
            #expect(!declared.isEmpty && declared.isSubset(of: served), "\(id): declares \(declared.sorted()), serves \(served.sorted())")
        }
        #expect(try text("Workers/Native/Sources/arsenal-worker-native/Main.swift").contains("if let kit = kitWorkers[name] { kit() }"))
    }

    @Test func everyKitFolderValidatesAgainstTheSignedKinds() throws {
        let kindsDirectory = repo.appendingPathComponent("Resources/kinds")
        let kinds = Set(try KindSpec.load(directory: kindsDirectory).map(\.id))
        let folders = try kitFolders()
        #expect(folders.count == 38)
        for id in folders.keys.sorted() {
            let folder = repo.appendingPathComponent("models/\(id)")
            let manifest = try ModelManifest.load(folder, kinds: kinds, kindsDirectory: kindsDirectory)
            #expect(manifest.problems(folder: folder, kinds: kinds, kindsDirectory: kindsDirectory).isEmpty, "\(id)")
            #expect(kinds.contains(manifest.kind), "\(id): kind \(manifest.kind)")
        }
    }

    /// A local folder may keep an embedded kind whose id is now signed, when its tree is the signed one (BLIP's
    /// local.vision beside Florence-2's); a different tree under a signed id is still refused.
    @Test func embeddedKindMayRepeatAnIdenticalSignedKindOnly() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("kit-kind-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let kindsDirectory = repo.appendingPathComponent("Resources/kinds")
        let kinds = Set(try KindSpec.load(directory: kindsDirectory).map(\.id))
        let folder = root.appendingPathComponent("florence-2-base")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("models/florence-2-base"), to: folder)
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("kinds"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: kindsDirectory.appendingPathComponent("local.vision"), to: folder.appendingPathComponent("kinds/local.vision"))
        let descriptor = try String(contentsOf: kindsDirectory.appendingPathComponent("local.vision.toml"), encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false).filter { !$0.hasPrefix("suite = ") }.joined(separator: "\n")
        try descriptor.write(to: folder.appendingPathComponent("kinds/local.vision.toml"), atomically: true, encoding: .utf8)
        let manifestURL = folder.appendingPathComponent("model.toml")
        let manifestText = try String(contentsOf: manifestURL, encoding: .utf8).replacingOccurrences(
            of: "kind = \"local.vision\"\n", with: "kind = \"local.vision\"\nkind_file = \"kinds/local.vision.toml\"\n")
        try manifestText.write(to: manifestURL, atomically: true, encoding: .utf8)
        var manifest = try TOMLParser.decode(ModelManifest.self, from: manifestText)
        #expect(manifest.kindFile == "kinds/local.vision.toml")
        #expect(!manifest.problems(folder: folder, kinds: kinds, kindsDirectory: kindsDirectory).contains { $0.contains("signed kind") })
        let schema = folder.appendingPathComponent("kinds/local.vision/ops/caption.input.json")
        try Data(try String(contentsOf: schema, encoding: .utf8).replacingOccurrences(of: "\"type\"", with: " \"type\"").utf8).write(to: schema)
        manifest = try TOMLParser.decode(ModelManifest.self, from: manifestText)
        #expect(manifest.problems(folder: folder, kinds: kinds, kindsDirectory: kindsDirectory).contains { $0.contains("embedded kind cannot override a signed kind") })
    }

    /// Fold review: a live catalog refresh that signs a local folder's embedded kind rechecks that folder, though its
    /// bytes did not change. Its own different tree under the now-signed id is refused, not kept from the earlier scan.
    @Test func aCatalogRefreshRechecksFoldersWhoseEmbeddedKindBecameSigned() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("kit-kind-refresh-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let signed = repo.appendingPathComponent("Resources/kinds")
        // Before the refresh: the signed kinds without local.vision.
        let before = root.appendingPathComponent("kinds-before")
        try FileManager.default.createDirectory(at: before, withIntermediateDirectories: true)
        for name in try FileManager.default.contentsOfDirectory(atPath: signed.path) where !name.hasPrefix("local.vision") {
            try FileManager.default.copyItem(at: signed.appendingPathComponent(name), to: before.appendingPathComponent(name))
        }
        let folder = root.appendingPathComponent("support/local-models/florence-2-base")
        try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repo.appendingPathComponent("models/florence-2-base"), to: folder)
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("kinds"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: signed.appendingPathComponent("local.vision"), to: folder.appendingPathComponent("kinds/local.vision"))
        try FileManager.default.copyItem(at: signed.appendingPathComponent("local.vision.toml"), to: folder.appendingPathComponent("kinds/local.vision.toml"))
        let manifestURL = folder.appendingPathComponent("model.toml")
        try String(contentsOf: manifestURL, encoding: .utf8).replacingOccurrences(
            of: "kind = \"local.vision\"\n", with: "kind = \"local.vision\"\nkind_file = \"kinds/local.vision.toml\"\n"
        ).write(to: manifestURL, atomically: true, encoding: .utf8)
        // Its own tree differs from the one the catalog will sign.
        let schema = folder.appendingPathComponent("kinds/local.vision/ops/caption.input.json")
        try Data(try String(contentsOf: schema, encoding: .utf8).replacingOccurrences(of: "\"type\"", with: " \"type\"").utf8).write(to: schema)
        let scanner = LocalFolderScanner()
        let support = root.appendingPathComponent("support")
        let first = try await scanner.scan(support: support, signedIDs: [], kinds: Set(try KindSpec.load(directory: before).map(\.id)), kindsDirectory: before)
        #expect(first.problems.isEmpty && first.folders.map(\.manifest.id) == ["florence-2-base"], "\(first.problems)")
        let reads = await scanner.hashReads
        let after = try await scanner.scan(support: support, signedIDs: [], kinds: Set(try KindSpec.load(directory: signed).map(\.id)), kindsDirectory: signed)
        #expect(after.folders.isEmpty && after.problems.first?.message.contains("embedded kind cannot override a signed kind") == true, "\(after)")
        #expect(await scanner.hashReads == reads, "the files were not read again")
    }
}

private actor PromotionSource: CatalogSource {
    var files: [String: Data]
    init(_ files: [String: Data]) { self.files = files }
    func read(path: String, commit: String) throws -> Data {
        guard let data = files[path] else { throw ManifestError.invalid("missing fixture file") }; return data
    }
    func add(_ more: [String: Data]) { files.merge(more) { _, new in new } }
}

/// A local folder whose id the next signed catalog lists is superseded (ignored, kept, no error row); the signed row takes
/// its place on a live refresh, with the same 16-bit weights already on disk (nothing downloaded again).
@Test func signedCatalogTakesOverPromotedLocalFolderOnLiveRefresh() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }; await f.service.shutdown()
    let key = Curve25519.Signing.PrivateKey(), source = PromotionSource(f.source.files)
    let s = try await RealModelsService.open(
        catalogData: f.catalogData, signature: key.signature(for: f.catalogData), publicKey: key.publicKey.rawRepresentation, source: source, kinds: KindSpec.defaults,
        paths: f.paths, artifacts: StubWorkerArtifact(), client: f.client)
    let id = "promoted-voice"
    let local = f.paths.support.appendingPathComponent("local-models/\(id)")
    try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: f.paths.support.deletingLastPathComponent().appendingPathComponent("folders/fixture-voices"), to: local)
    let manifestURL = local.appendingPathComponent("model.toml")
    try String(contentsOf: manifestURL, encoding: .utf8).replacingOccurrences(of: "fixture-voices", with: id).write(to: manifestURL, atomically: true, encoding: .utf8)
    try await s.refresh()
    #expect(await s.catalog().contains { $0.id == id && $0.local })
    try await s.get(id: id, precision: "bf16")
    #expect(await s.rows().first { $0.id == id }?.states["bf16"]?.state == .ready)
    let snapshot = f.paths.support.appendingPathComponent("weights/snapshots/\(id)")
    #expect(FileManager.default.fileExists(atPath: snapshot.path))

    // The next signed catalog lists the same folder, byte for byte, as a signed model.
    var doc = try JSONDecoder().decode(CatalogDocument.self, from: f.catalogData)
    var entry = doc.models[0]
    var files: [String: Data] = [:]
    for path in try SafePath.files(in: local) {
        files["models/\(id)/\(path)"] = try Data(contentsOf: local.appendingPathComponent(path))
    }
    entry.manifest = try ModelManifest.load(local, kinds: Set(KindSpec.defaults.map(\.id)))
    entry.files = try entry.files.map { file in
        let data = try #require(files["models/\(id)/\(file.path)"])
        return CatalogFile(path: file.path, sha256: FileDigest.sha256(data), bytes: Int64(data.count))
    }
    await source.add(files)
    doc.models.append(entry); doc.sequence += 1; doc.sourceCommit = String(repeating: "c", count: 40)
    let data = try doc.encoded(), directory = f.paths.directory("catalog")
    try data.write(to: directory.appendingPathComponent("catalog.json")); try key.signature(for: data).write(to: directory.appendingPathComponent("catalog.json.sig"))
    try await s.refresh()

    #expect(await s.catalog().filter { $0.id == id }.map(\.local) == [false])
    let row = try #require(await s.rows().first { $0.id == id })
    #expect(row.states["bf16"]?.state == .ready, "the signed row finds the weights the local folder stored")
    #expect(!row.local && row.approval == nil && !row.needsCodeConsent)
    // The superseded folder makes no error row (2 Nov: an upgrade would show 37 of them) and stays on disk untouched;
    // the signed row keeps its metadata.
    #expect(await s.catalog().filter { $0.id.hasPrefix("local-error-") }.isEmpty)
    #expect(FileManager.default.fileExists(atPath: manifestURL.path))
    #expect(await s.metadata[id] != nil)
    let config = try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json"))
    #expect(config.localCodeApprovals[id] == nil && config.localCodeConsent[id] == nil)
    #expect(FileManager.default.fileExists(atPath: snapshot.path))
    await s.shutdown()
}
