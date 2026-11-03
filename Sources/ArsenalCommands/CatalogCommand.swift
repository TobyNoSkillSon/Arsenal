// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalBench

public enum CatalogCommand {
    public static func run(arguments: [String]) async -> Int32 {
        do { try await execute(arguments); return 0 } catch {
            FileHandle.standardError.write(Data(("arsenal-catalog: " + error.localizedDescription.replacingOccurrences(of: "\n", with: " ") + "\n").utf8))
            return (error as? LocalModelAddCommand.Refusal)?.exitCode ?? 1
        }
    }
    /// The profile holding the development signing key.
    static func signingSupport(home: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        home.appendingPathComponent("Library/Application Support/Arsenal")
    }

    static func resource(_ path: String) -> URL {
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
        let bundled = executable.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Resources/" + path)
        // The installed app's own Resources are the authority. A working directory's Resources/ serves development runs only.
        if executable.path.contains(".app/Contents/MacOS/") { return bundled }
        let local = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Resources/" + path)
        if FileManager.default.fileExists(atPath: local.path) { return local }
        return bundled
    }
    static func execute(_ args: [String]) async throws {
        guard let command = args.first else { help(); return }
        if command == "--help" || command == "help" { help(); return }
        let repo = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let models = repo.appendingPathComponent("models"); let catalogDir = repo.appendingPathComponent("Resources/catalog")
        let folders = ((try? FileManager.default.contentsOfDirectory(at: models, includingPropertiesForKeys: [.isDirectoryKey])) ?? []).filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true && !$0.lastPathComponent.hasPrefix(".")
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        let kinds = Set(try KindSpec.load(directory: resource("kinds")).map(\.id))
        switch command {
        case "fingerprint":
            guard args.count == 2 else { throw ManifestError.invalid("fingerprint requires a model folder") }
            let local = try LocalModelFolders.discoverSnapshot(URL(fileURLWithPath: args[1], relativeTo: repo).absoluteURL)
            print(
                String(
                    decoding: try WireJSON.encode([
                        "implementation_fingerprint": .string(try ImplementationEvidence.fingerprint(local)), "fixture_identity": .string(ImplementationEvidence.fixtures(local)),
                        "folder_digest": .string(local.contentDigest)
                    ]), as: UTF8.self))
        case "validate":
            let targets: [URL]
            targets = args.dropFirst().first == "--all" || args.count == 1 ? folders : [URL(fileURLWithPath: args[1], relativeTo: repo).absoluteURL]
            var errors: [String] = []
            for folder in targets { errors += validation(folder: folder, kinds: kinds) }
            if !errors.isEmpty { for e in errors { FileHandle.standardError.write(Data((e.replacingOccurrences(of: "\n", with: " ") + "\n").utf8)) }; exit(1) }
            print("Valid: \(targets.count) model folders")
        case "pin":
            if let id = args.dropFirst().first, id != "--all", !ModelManifest.validID(id) { throw ManifestError.invalid("pin requires a model id, not a path") }
            let targets: [URL]
            targets = args.count == 1 || args[1] == "--all" ? folders : [models.appendingPathComponent(args[1])]
            for folder in targets { try await pin(folder: folder, kinds: kinds, metadataOnly: args.contains("--metadata-only")) }
            print("Pinned: \(targets.count) model folders")
        case "build":
            let commitData = try GitRead.run(repo: repo, args: ["rev-parse", "HEAD"])
            let commit = String(data: commitData, encoding: .utf8)!.trimmingCharacters(in: .whitespacesAndNewlines)
            var entries: [CatalogModel] = []
            let source = LocalGitCatalogSource(repo: repo)
            for folder in folders {
                let errors = validation(folder: folder, kinds: kinds); guard errors.isEmpty else { throw ManifestError.invalid(errors.joined(separator: "; ")) }
                let manifest = try ModelManifest.load(folder, kinds: kinds, kindsDirectory: resource("kinds"))
                guard CatalogShipping.includes(manifest) else { continue }
                let prefix = "models/" + manifest.id + "/"
                let listed = try GitRead.run(repo: repo, args: ["ls-tree", "-r", "--name-only", "-z", commit, "--", prefix])
                let tracked = Set(String(decoding: listed, as: UTF8.self).split(separator: "\0").map { String($0.dropFirst(prefix.count)) })
                let files = try await committedFiles(folder: folder, manifest: manifest, source: source, commit: commit, tracked: tracked)
                entries.append(CatalogModel(manifest: manifest, files: files))
            }
            var kindFiles: [CatalogFile] = []
            let kindRoot = resource("kinds")
            for path in try SafePath.files(in: kindRoot) {
                let data = try Data(contentsOf: kindRoot.appendingPathComponent(path))
                let committed = try await source.read(path: "kinds/" + path, commit: commit)
                guard data == committed else { throw ManifestError.invalid("commit kind changes before catalog build: \(path)") }
                kindFiles.append(CatalogFile(path: path, sha256: FileDigest.sha256(committed), bytes: Int64(committed.count)))
            }
            let old = try? JSONDecoder().decode(CatalogDocument.self, from: Data(contentsOf: catalogDir.appendingPathComponent("catalog.json")))
            let sourceDate = String(decoding: try GitRead.run(repo: repo, args: ["show", "-s", "--format=%cI", commit]), as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let doc = CatalogDocument(
                sequence: (old?.sequence ?? 0) + 1, sourceCommit: commit, models: entries, kindFiles: kindFiles, sourceDate: sourceDate)
            try doc.validate()
            try FileManager.default.createDirectory(at: catalogDir, withIntermediateDirectories: true)
            let output: URL
            if args.count == 3, args[1] == "--output" {
                output = URL(fileURLWithPath: args[2], relativeTo: repo).absoluteURL
            } else {
                guard args.count == 1 else { throw ManifestError.invalid("build accepts only --output <unsigned-json-path>") }
                output = catalogDir.appendingPathComponent("catalog.json")
            }
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            try doc.encoded().write(to: output, options: .atomic)
            print("Built catalog sequence \(doc.sequence): \(entries.count) models at \(commit.prefix(12))")
        case "new":
            func option(_ flag: String) -> String? { guard let index = args.firstIndex(of: flag), args.indices.contains(index + 1) else { return nil }; return args[index + 1] }
            guard let hfRepo = option("--from-hf") else { throw ManifestError.invalid("new requires --from-hf owner/repo") }
            let metadata = try await HuggingFaceClient().metadata(repo: hfRepo)
            let kindID = option("--kind") ?? metadata.inferredKind
            guard let kind = try KindSpec.load(directory: resource("kinds")).first(where: { $0.id == kindID }) else {
                throw ManifestError.invalid("choose a known --kind; HF task is not mapped")
            }
            let output = option("--output").map { URL(fileURLWithPath: $0, relativeTo: repo).absoluteURL } ?? models
            let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.timeZone = TimeZone(secondsFromGMT: 0)
            formatter.dateFormat = "yyyy-MM-dd"
            let result = try HFScaffolder.write(
                metadata: metadata, kind: kind, parent: output, date: formatter.string(from: Date()), kindsDirectory: resource("kinds"))
            if args.contains("--json") {
                print(String(data: try JSONEncoder().encode(result), encoding: .utf8)!)
            } else {
                print("Scaffolded \(result.directory) at \(result.revision.prefix(12)); finish: " + result.pending.joined(separator: "; "))
            }
        case "add-local":
            let text = try await LocalModelAddCommand.run(arguments: Array(args.dropFirst()), defaultSupport: ArsenalPaths().support)
            print(text)
        case "sign":
            print(try signCatalog(repo: repo, arguments: Array(args.dropFirst()), environment: ProcessInfo.processInfo.environment))
        case "verify":
            let doc = try CatalogTrust.verify(
                data: Data(contentsOf: catalogDir.appendingPathComponent("catalog.json")), signature: Data(contentsOf: catalogDir.appendingPathComponent("catalog.json.sig")),
                publicKey: Data(contentsOf: repo.appendingPathComponent("Resources/catalog-public-key")))
            print("Verified catalog sequence \(doc.sequence): \(doc.models.count) models")
        default: throw ManifestError.invalid("unknown command; use arsenal-catalog --help")
        }
    }
    /// `sign [--key <path>] [--rotate]`. `--key` (or ARSENAL_CATALOG_SIGNING_KEY) names the signing key, such as the
    /// release key; it must already exist and is never created. Without one, the profile's development key is used (and
    /// made on first use). A key whose public half differs from Resources/catalog-public-key is refused unless `--rotate`
    /// explicitly replaces the committed public key.
    static func signCatalog(repo: URL, arguments: [String], environment: [String: String]) throws -> String {
        var rest = arguments, rotate = false
        var explicitKey = environment["ARSENAL_CATALOG_SIGNING_KEY"].flatMap { $0.isEmpty ? nil : $0 }
        while let flag = rest.first {
            rest.removeFirst()
            switch flag {
            case "--rotate": rotate = true
            case "--key":
                guard let path = rest.first, !path.hasPrefix("--") else { throw ManifestError.invalid("sign --key requires a key file path") }
                explicitKey = path; rest.removeFirst()
            default: throw ManifestError.invalid("sign accepts only --key <path> and --rotate")
            }
        }
        let catalogDir = repo.appendingPathComponent("Resources/catalog")
        let data = try Data(contentsOf: catalogDir.appendingPathComponent("catalog.json")); let doc = try JSONDecoder().decode(CatalogDocument.self, from: data)
        try doc.validate()
        let signed: (signature: Data, publicKey: Data)
        if let explicitKey {
            signed = try CatalogTrust.sign(data: data, keyFile: URL(fileURLWithPath: explicitKey, relativeTo: repo).absoluteURL, createIfMissing: false)
        } else {
            let support = environment["ARSENAL_SUPPORT_DIR"].map { URL(fileURLWithPath: $0) } ?? Self.signingSupport()
            signed = try CatalogTrust.sign(data: data, keyFile: support.appendingPathComponent("dev-signing/ed25519.key"))
        }
        let keyURL = repo.appendingPathComponent("Resources/catalog-public-key")
        let rotated = try FileManager.default.fileExists(atPath: keyURL.path) && Data(contentsOf: keyURL) != signed.publicKey
        if rotated && !rotate {
            throw ManifestError.invalid("signing key differs from committed public key; pass --rotate to replace Resources/catalog-public-key")
        }
        try signed.signature.write(to: catalogDir.appendingPathComponent("catalog.json.sig"), options: .atomic)
        try signed.publicKey.write(to: keyURL, options: .atomic)
        return "Signed catalog sequence \(doc.sequence)" + (rotated ? "; Resources/catalog-public-key replaced" : "")
    }
    /// Read only the shipping closure from HEAD. Ignored local captures are neither read nor required
    /// to exist in Git; every shipped byte must still match its committed source.
    static func committedFiles(folder: URL, manifest: ModelManifest, source: any CatalogSource, commit: String, tracked: Set<String>) async throws -> [CatalogFile] {
        var files: [CatalogFile] = []
        for path in try CatalogShipping.files(in: folder, model: manifest).filter({ tracked.contains($0) }).sorted() {
            let local = try Data(contentsOf: folder.appendingPathComponent(path))
            let pinned = try await source.read(path: "models/\(manifest.id)/\(path)", commit: commit)
            guard local == pinned else { throw ManifestError.invalid("commit model folder changes before catalog build: \(manifest.id)/\(path)") }
            files.append(CatalogFile(path: path, sha256: FileDigest.sha256(pinned), bytes: Int64(pinned.count)))
        }
        return files
    }

    static func validation(folder: URL, kinds: Set<String>) -> [String] {
        do {
            _ = try SafePath.resolve("model.toml", inside: folder)
            let text = try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8)
            let manifest = try TOMLParser.decode(ModelManifest.self, from: text)
            var parser = TOMLParser(text); let document = try parser.parse()
            let schemaURL = resource("schemas/model.schema.json")
            var errors = try ModelSchemaValidation.validate(document: document, schemaData: Data(contentsOf: schemaURL)).map { "\(manifest.id): " + $0 }
            errors += manifest.problems(
                folder: folder, kinds: kinds, kindsDirectory: resource("kinds"))
            for path in try SafePath.files(in: folder) {
                do {
                    if path.hasPrefix("log/") && path.hasSuffix(".md") {
                        try ResearchLogValidation.validate(markdown: String(contentsOf: folder.appendingPathComponent(path), encoding: .utf8))
                    }
                    if path.hasPrefix("results/") && path.hasSuffix(".json") {
                        try CatalogResultValidation.validate(data: Data(contentsOf: folder.appendingPathComponent(path)), path: path, folder: folder, manifest: manifest)
                    }
                } catch { errors.append("\(manifest.id)/\(path): \(error.localizedDescription)") }
            }; return errors
        } catch { return ["\(folder.lastPathComponent): \(error.localizedDescription)"] }
    }
    static func pin(folder: URL, kinds: Set<String>, metadataOnly: Bool = false) async throws {
        _ = try ModelManifest.load(
            folder, requirePins: false, kinds: kinds, kindsDirectory: resource("kinds"))
        let manifestURL = folder.appendingPathComponent("model.toml")
        var parser = TOMLParser(try String(contentsOf: manifestURL, encoding: .utf8)); var doc = try parser.parse()
        let client = HuggingFaceClient()
        if case .table(let source) = doc["weights"], case .string(let repo) = source["repo"], case .string(let revision) = source["revision"] {
            let metadata = try await client.metadata(repo: repo, revision: revision)
            let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.timeZone = TimeZone(secondsFromGMT: 0)
            formatter.dateFormat = "yyyy-MM-dd"
            doc["popularity"] = .table(["hf_downloads": .integer(metadata.downloads), "hf_likes": .integer(metadata.likes), "as_of": .string(formatter.string(from: Date()))])
        }
        func pinned(_ value: TOMLValue, dtype: String? = nil) async throws -> TOMLValue {
            guard case .table(var table) = value else { throw ManifestError.invalid("source is not a table") }
            if table["dtype"] == nil { table["dtype"] = .string(dtype ?? "asset") }
            let source = try TOMLParser.decode(WeightSpec.self, from: TOMLWriter.encode(table))
            let pins = try await client.pins(source)
            table["sha256"] = .table(Dictionary(uniqueKeysWithValues: pins.map { ($0.path, .string($0.sha256)) }))
            table["bytes"] = .table(Dictionary(uniqueKeysWithValues: pins.map { ($0.path, .integer($0.bytes)) }))
            if dtype == "asset" { table.removeValue(forKey: "dtype") }
            return .table(table)
        }
        if !metadataOnly, let source = doc["weights"] { doc["weights"] = try await pinned(source) }
        if !metadataOnly, case .table(var vendors) = doc["vendor"] {
            for precision in vendors.keys { vendors[precision] = try await pinned(vendors[precision]!, dtype: precision) }; doc["vendor"] = .table(vendors)
        }
        if !metadataOnly, case .array(let assets) = doc["asset"] {
            var changed: [TOMLValue] = []; for asset in assets { changed.append(try await pinned(asset, dtype: "asset")) }; doc["asset"] = .array(changed)
        }
        try TOMLWriter.encode(doc).write(to: manifestURL, atomically: true, encoding: .utf8)
        let errors = validation(folder: folder, kinds: kinds); guard errors.isEmpty else { throw ManifestError.invalid(errors.joined(separator: "; ")) }
    }
    static func help() {
        print(
            "arsenal-catalog add-local owner/repo [--kind K] [--id ID] [--accept-licence LICENCE] [--support DIR] | fingerprint <dir> | validate <dir|--all> | pin <id|--all> [--metadata-only] | new --from-hf owner/repo [--kind K] [--output DIR] [--json] | build [--output unsigned.json] | sign [--key <path>] [--rotate] | verify (run from repo root)"
        )
    }
}
