// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// `Arsenal catalog add-local owner/repo`: turns a Hugging Face repository into a local model folder, or refuses with one
/// line that names the gate and what to tell the user. Each gate runs before anything is written. The folder is built in
/// `.staging-<id>`, validated, and only then renamed into `local-models/<id>`, so a refused add leaves nothing behind.
enum LocalModelAddCommand {
    struct Refusal: LocalizedError {
        let category: String; let message: String; let exitCode: Int32
        var errorDescription: String? { "[\(category)] \(message)" }
    }

    static let usage = "add-local owner/repo [--plan] [--kind K] [--id ID] [--accept-licence LICENCE] [--support DIR]"

    /// Everything a local folder is built from, after the gates have passed.
    struct Build {
        var id: String; var template: [String: TOMLValue]; var templateFolder: URL; var entry: ArchitectureEntry
        var metadata: HFModelMetadata; var dtype: String; var files: [String]; var pins: [DerivedFile]
        var licence: String; var open: Bool; var accepted: Bool
    }

    static func run(arguments: [String], defaultSupport: URL, client: HuggingFaceClient = HuggingFaceClient()) async throws -> String {
        let options = try Options.parse(arguments)
        let support = options.support ?? defaultSupport
        let kinds = try KindSpec.load(directory: CatalogCommand.resource("kinds"))
        let metadata = try await fetchMetadata(repo: options.repo, client: client)

        // Licence first: it needs only the card, so a refused licence never fetches anything else.
        let reported = metadata.licence.map(LocalModelPlan.normalised).flatMap { $0.isEmpty ? nil : $0 } ?? "none"
        let open = LocalModelPlan.isOpen(licence: metadata.licence)
        let accepted = options.acceptLicence.map(LocalModelPlan.normalised) == reported
        if !open && !accepted {
            throw Refusal(
                category: "licence",
                message: "\(options.repo) is under licence \(reported), which is not an open licence; Arsenal ships only open models. Nothing was written. "
                    + "Ask the user whether to add it locally anyway; only if they say yes, rerun with --accept-licence \(reported).",
                exitCode: 2)
        }

        let config = try await fetchConfig(metadata: metadata, client: client)
        guard let modelType = config["model_type"] as? String, !modelType.isEmpty else {
            throw Refusal(category: "architecture", message: "\(options.repo) has no model_type in config.json, so no runner can be chosen. Nothing was written.", exitCode: 3)
        }
        let registry = try ArchitectureRegistry.load(from: CatalogCommand.resource("runners/architectures.json"))
        let entries = registry.entries(modelType: modelType, kind: options.kind)
        guard let entry = entries.first, entries.count == 1 else {
            if entries.isEmpty {
                // A kind the model_type does not run for is a wrong --kind, not a missing runner: name the kinds it does run for.
                let kindsFound = options.kind == nil ? [] : Set(registry.entries(modelType: modelType, kind: nil).map(\.kind)).sorted()
                if !kindsFound.isEmpty {
                    throw Refusal(
                        category: "kind",
                        message:
                            "model_type \(modelType) runs for \(kindsFound.joined(separator: ", ")), not \(options.kind ?? ""). Nothing was written. Rerun with one of those kinds.",
                        exitCode: 4)
                }
                throw Refusal(
                    category: "unsupported",
                    message: "model_type \(modelType) has no bundled runner in this Arsenal. Nothing was written. Say so and stop: a new runner is needed "
                        + "(arsenal://integration/ADD-A-MODEL.md, step 2). Do not try to add it another way.",
                    exitCode: 3)
            }
            let choices = entries.map(\.kind).sorted().joined(separator: ", ")
            throw Refusal(category: "kind", message: "model_type \(modelType) runs for several kinds (\(choices)). Nothing was written. Rerun with --kind.", exitCode: 4)
        }
        guard kinds.contains(where: { $0.id == entry.kind }) else {
            throw Refusal(category: "kind", message: "the registry names kind \(entry.kind), which this Arsenal does not declare. Nothing was written.", exitCode: 4)
        }
        // One model_type can serve a chat model and an embedding model alike; the Hugging Face pipeline tells them apart.
        if let allowed = entry.pipelineTags, let tag = metadata.pipelineTag, !allowed.contains(tag) {
            throw Refusal(
                category: "kind",
                message:
                    "\(options.repo) is tagged \(tag) on Hugging Face, which is not a \(entry.kind) model this runner serves. Nothing was written. Stop and tell the user; do not force it.",
                exitCode: 4)
        }

        let templateFolder = try CatalogCommand.resource("models/\(entry.template)")
        guard FileManager.default.fileExists(atPath: templateFolder.appendingPathComponent("model.toml").path) else {
            throw Refusal(category: "template", message: "the template \(entry.template) is not in this Arsenal. Nothing was written.", exitCode: 4)
        }
        let templateText = try String(contentsOf: templateFolder.appendingPathComponent("model.toml"), encoding: .utf8)
        var parser = TOMLParser(templateText)
        let template = try parser.parse()
        guard template["asset"] == nil, template["assets"] == nil else {
            throw Refusal(
                category: "template",
                message: "the template \(entry.template) declares assets, which a local add does not copy; the registry entry is wrong. Nothing was written.", exitCode: 4)
        }
        // A folder that runs its own code needs the user's consent per folder; a local add only ever uses bundled workers.
        if case .array(let runners)? = template["runner"],
            runners.contains(where: { runner in
                if case .table(let table) = runner { return table["entry"] != nil }
                return false
            })
        {
            throw Refusal(category: "template", message: "the template \(entry.template) runs folder code, which a local add never copies. Nothing was written.", exitCode: 4)
        }
        guard case .table(let weightTable)? = template["weights"], case .array(let listed) = weightTable["files"] ?? .array([]) else {
            throw Refusal(category: "template", message: "the template \(entry.template) lists no weight files. Nothing was written.", exitCode: 4)
        }
        let templateFiles = listed.compactMap { value -> String? in
            guard case .string(let path) = value else { return nil }
            return path
        }

        let selection = LocalModelPlan.weightFiles(repository: metadata.files, template: templateFiles)
        guard selection.missing.isEmpty else {
            throw Refusal(
                category: "weights", message: "\(options.repo) has no \(selection.missing.joined(separator: ", ")), which the \(entry.template) runner needs. Nothing was written.",
                exitCode: 4)
        }
        guard selection.files.contains(where: { $0.hasSuffix(".safetensors") }) else {
            throw Refusal(category: "weights", message: "\(options.repo) has no safetensors weights. Nothing was written.", exitCode: 4)
        }

        let id = options.id ?? defaultID(repo: options.repo)
        guard ModelManifest.validID(id) else {
            throw Refusal(category: "invalid_input", message: "\(id) is not a valid model id; pass --id with lowercase letters, digits, dots and dashes.", exitCode: 4)
        }
        let shippedFolder = try CatalogCommand.resource("models/\(id)")
        guard !FileManager.default.fileExists(atPath: shippedFolder.path) else {
            throw Refusal(
                category: "exists", message: "\(id) is already a shipped Arsenal model. Nothing was written. Pass a different --id only if the user wants a second copy.",
                exitCode: 5)
        }
        let localRoot = support.appendingPathComponent("local-models")
        guard !FileManager.default.fileExists(atPath: localRoot.appendingPathComponent(id).path) else {
            throw Refusal(
                category: "exists",
                message: "\(id) is already in local-models. Nothing was written. Use arsenal_settings(action:\"refresh\") to see it, or ask the user before replacing it.",
                exitCode: 5)
        }

        let sizes = Dictionary(uniqueKeysWithValues: metadata.files.map { ($0.path, $0.bytes ?? 0) })
        let downloadBytes = selection.files.reduce(Int64(0)) { $0 + (sizes[$1] ?? 0) }
        guard let dtype = LocalModelPlan.dtype(torchDType: (config["torch_dtype"] as? String) ?? (config["dtype"] as? String)) else {
            throw Refusal(
                category: "weights",
                message: "\(options.repo) config.json declares no torch_dtype, so its weight precision is unknown. Nothing was written. Stop and tell the user.",
                exitCode: 4)
        }
        if options.plan {
            return planText(
                id: id, entry: entry, metadata: metadata, licence: reported, open: open, accepted: accepted,
                fileCount: selection.files.count, bytes: downloadBytes, dtype: dtype)
        }

        // Pins come from the Hub's LFS metadata; small git-tracked files are read once and hashed.
        let spec = try JSONDecoder().decode(
            WeightSpec.self,
            from: JSONSerialization.data(withJSONObject: ["repo": metadata.repo, "revision": metadata.revision, "dtype": dtype, "files": selection.files]))
        let build = Build(
            id: id, template: template, templateFolder: templateFolder, entry: entry, metadata: metadata, dtype: dtype, files: selection.files,
            pins: try await client.pins(spec), licence: reported, open: open, accepted: accepted)
        let final = try stage(build, localRoot: localRoot, kinds: Set(kinds.map(\.id)), year: Calendar(identifier: .gregorian).component(.year, from: Date()))
        return summary(
            id: id, final: final, entry: entry, metadata: metadata, licence: reported, open: open, accepted: accepted,
            fileCount: selection.files.count, bytes: downloadBytes)
    }

    /// Writes the folder into `.staging-<id>`, validates it with the same rules as `validate`, and renames it into place.
    /// A folder that fails validation is removed, so nothing half-written stays where the app scans.
    static func stage(_ build: Build, localRoot: URL, kinds: Set<String>, year: Int) throws -> URL {
        let id = build.id
        let final = localRoot.appendingPathComponent(id)
        // Validation requires the folder's name to equal its id, so the folder is built inside the hidden staging directory.
        let staging = localRoot.appendingPathComponent(".staging-\(id)", isDirectory: true)
        let folder = staging.appendingPathComponent(id, isDirectory: true)
        try? FileManager.default.removeItem(at: staging)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let doc = LocalModelPlan.manifest(
                template: build.template, id: id, metadata: build.metadata, architecture: build.entry.architecture, dtype: build.dtype, files: build.files,
                sha256: Dictionary(uniqueKeysWithValues: build.pins.map { ($0.path, $0.sha256) }),
                bytes: Dictionary(uniqueKeysWithValues: build.pins.map { ($0.path, $0.bytes) }), licenceAccepted: build.accepted, year: year)
            try TOMLWriter.encode(doc).write(to: folder.appendingPathComponent("model.toml"), atomically: true, encoding: .utf8)
            try copyTemplateFiles(from: build.templateFolder, to: folder, templateID: build.entry.template, id: id)
            try readme(id: id, metadata: build.metadata, licence: build.licence, open: build.open, template: build.entry.template).write(
                to: folder.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
            try FileManager.default.createDirectory(at: folder.appendingPathComponent("log"), withIntermediateDirectories: true)
            try logStub(metadata: build.metadata).write(to: folder.appendingPathComponent("log/001-local-add.md"), atomically: true, encoding: .utf8)
            let errors = CatalogCommand.validation(folder: folder, kinds: kinds)
            guard errors.isEmpty else {
                throw Refusal(category: "invalid", message: "the folder did not validate, so nothing was added: " + errors.joined(separator: "; "), exitCode: 4)
            }
            guard !FileManager.default.fileExists(atPath: final.path) else {
                throw Refusal(category: "exists", message: "\(id) appeared in local-models while adding it. Nothing was replaced.", exitCode: 5)
            }
            try FileManager.default.moveItem(at: folder, to: final)
            try? FileManager.default.removeItem(at: staging)
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw error
        }
        return final
    }

    struct Options {
        var repo: String; var kind: String?; var id: String?; var acceptLicence: String?; var support: URL?; var plan: Bool

        static func parse(_ arguments: [String]) throws -> Options {
            var repo: String?; var kind: String?; var id: String?; var accept: String?; var support: URL?; var plan = false
            var index = 0
            while index < arguments.count {
                let argument = arguments[index]
                func value() throws -> String {
                    guard index + 1 < arguments.count else { throw Refusal(category: "invalid_input", message: "\(argument) needs a value. Usage: \(usage)", exitCode: 4) }
                    index += 1; return arguments[index]
                }
                switch argument {
                case "--kind": kind = try value()
                case "--id": id = try value()
                case "--accept-licence": accept = try value()
                case "--plan": plan = true
                case "--support": support = URL(fileURLWithPath: try value(), isDirectory: true)
                default:
                    guard repo == nil, !argument.hasPrefix("--") else {
                        throw Refusal(category: "invalid_input", message: "unexpected \(argument). Usage: \(usage)", exitCode: 4)
                    }
                    repo = argument
                }
                index += 1
            }
            guard let repo, repo.split(separator: "/").count == 2 else {
                throw Refusal(category: "invalid_input", message: "give one Hugging Face repository as owner/name. Usage: \(usage)", exitCode: 4)
            }
            return Options(repo: repo, kind: kind, id: id, acceptLicence: accept, support: support, plan: plan)
        }
    }

    static func fetchMetadata(repo: String, client: HuggingFaceClient) async throws -> HFModelMetadata {
        do { return try await client.metadata(repo: repo) } catch {
            throw Refusal(category: "network", message: "could not read \(repo) from Hugging Face (\(error.localizedDescription)). Nothing was written.", exitCode: 6)
        }
    }

    /// config.json only: a few kilobytes, read to choose the runner. No weights are fetched here.
    static func fetchConfig(metadata: HFModelMetadata, client: HuggingFaceClient) async throws -> [String: Any] {
        let url: URL
        do { url = try client.resolve(repo: metadata.repo, revision: metadata.revision, path: "config.json") } catch {
            throw Refusal(category: "architecture", message: "\(metadata.repo) has no config.json, so its architecture is unknown. Nothing was written.", exitCode: 3)
        }
        do {
            let (data, reply) = try await URLSession.shared.data(from: url)
            guard (reply as? HTTPURLResponse)?.statusCode == 200, data.count <= 1 << 20,
                let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                throw Refusal(category: "architecture", message: "\(metadata.repo) has no readable config.json, so its architecture is unknown. Nothing was written.", exitCode: 3)
            }
            return object
        } catch let refusal as Refusal {
            throw refusal
        } catch {
            throw Refusal(category: "network", message: "could not read config.json of \(metadata.repo) (\(error.localizedDescription)). Nothing was written.", exitCode: 6)
        }
    }

    static func defaultID(repo: String) -> String {
        let name = String(repo.split(separator: "/").last ?? "model").lowercased()
        return name.replacingOccurrences(of: "[^a-z0-9.-]", with: "-", options: .regularExpression)
    }

    /// The operations, settings, gate and bench recipes come from the template, with its id replaced. Nothing else of the
    /// template's evidence or assets is copied.
    static func copyTemplateFiles(from template: URL, to folder: URL, templateID: String, id: String) throws {
        let fileManager = FileManager.default
        for relative in ["ops", "settings.toml", "gate/gate.toml", "bench/bench.toml"] {
            let source = template.appendingPathComponent(relative)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: source.path, isDirectory: &isDirectory) else { continue }
            let targets = isDirectory.boolValue ? try fileManager.contentsOfDirectory(atPath: source.path).map { "\(relative)/\($0)" } : [relative]
            for target in targets {
                let from = template.appendingPathComponent(target), to = folder.appendingPathComponent(target)
                try fileManager.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
                if let text = try? String(contentsOf: from, encoding: .utf8) {
                    try text.replacingOccurrences(of: templateID, with: id).write(to: to, atomically: true, encoding: .utf8)
                } else {
                    try fileManager.copyItem(at: from, to: to)
                }
            }
        }
    }

    static func readme(id: String, metadata: HFModelMetadata, licence: String, open: Bool, template: String) -> String {
        let licenceLine = open ? "Licence \(licence) (open)." : "Licence \(licence): not an open licence; the user accepted it when this folder was added."
        return """
            # \(id)

            Added from \(metadata.repo) at revision \(metadata.revision.prefix(12)). \(licenceLine)

            Runs on the bundled worker through the \(template) folder's operations and settings. Not reviewed: quality, parity and speed are not checked.
            Weights are downloaded only with the user's yes, through Get.

            """
    }

    static func logStub(metadata: HFModelMetadata) -> String {
        """
        +++
        hypothesis = "A Hugging Face repository with a registered model_type runs through the bundled worker."
        change = "Added from the repository's metadata and config.json; weights were not downloaded."
        measurement = "Validation of the generated folder only."
        verdict = "inconclusive"
        numbers = { real_runs = 0, weight_download_bytes = 0 }
        links = ["https://huggingface.co/\(metadata.repo)/tree/\(metadata.revision)"]
        +++

        Generated by `Arsenal catalog add-local`. Parity and quality are not checked.

        """
    }

    /// What `--plan` prints: every gate has passed, nothing is written and no weight is fetched.
    static func planText(
        id: String, entry: ArchitectureEntry, metadata: HFModelMetadata, licence: String, open: Bool, accepted: Bool, fileCount: Int, bytes: Int64, dtype: String
    ) -> String {
        let licenceState = open ? "open" : (accepted ? "not open; the user accepted it" : "unknown")
        return """
            Plan for \(metadata.repo) as \(id): \(entry.kind), runs on the bundled \(entry.worker) worker, weights \(dtype).
            Licence \(licence): \(licenceState). Weights: \(fileCount) files, \(gigabytes(bytes)) to download after the user's yes.
            Nothing was written. Show the user this plan; after their yes, run the same command without --plan.
            """
    }

    static func summary(
        id: String, final: URL, entry: ArchitectureEntry, metadata: HFModelMetadata, licence: String, open: Bool, accepted: Bool, fileCount: Int, bytes: Int64
    ) -> String {
        let licenceState = open ? "open" : (accepted ? "not open, accepted by the user" : "unknown")
        return """
            Added \(id) to \(final.path): \(entry.kind), runs on the bundled \(entry.worker) worker. Licence \(licence) (\(licenceState)).
            Weights: \(fileCount) files, \(gigabytes(bytes)) to download; not downloaded yet. No code consent is needed: no folder code runs.
            Next: arsenal_settings(action:"refresh"), then arsenal_model(model:"\(id)"). Download only with the user's yes.
            """
    }

    static func gigabytes(_ bytes: Int64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
}
