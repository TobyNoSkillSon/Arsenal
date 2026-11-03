// SPDX-License-Identifier: MIT
import Foundation

extension RealModelsService {
    /// A failure is reported when a read fails; a failure answered from `operationFailures` was reported by the read
    /// that found it (as this feedback, or as the admission refusal that read it first), so it is not repeated per call.
    public func operations(id: String) async -> [ModelOperation] {
        var known = false
        do {
            if localModels[id] != nil { try await discoverLocalModels(notifyChanges: false) }
            known = operationFailures[id]?.key == operationFailureKey(id)
            return try await resolvedOperations(id: id)
        } catch {
            if !known { emit(.readFeedback(.error(id: id, message: error.localizedDescription))) }
            return []
        }
    }
    /// What would change a failed schema resolution: the catalog commit, the local folder's digest and the kinds directory.
    func operationFailureKey(_ id: String) -> String { "\(catalog.sourceCommit)|\(localModels[id]?.contentDigest ?? "")|\(kindsDirectory?.path ?? "")" }
    /// The folder's declared operation names, from its manifest (kind defaults when it lists none): no schema is read,
    /// nothing is admitted and nothing is reported. A resident or recovery revision answers with its own declarations.
    public func declaredOperationNames(id: String) async -> [String] {
        if residents[id] != nil || reloading.contains(id), let operations = residentOperations[id] { return operations.map(\.name) }
        if let previous = recoveryRevision(id) { return previous.operations.map(\.name) }
        if let cached = operationCache[id] { return cached.map(\.name) }
        guard let model = try? manifest(id) else { return [] }
        var kind: KindSpec?
        if model.op == nil {
            let local = localModels[id]
            let immutable = local.map { paths.support.appendingPathComponent("state/local-folders/\($0.contentDigest)/\(id)") }
            let root = immutable.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil } ?? local?.directory
            kind = (root.flatMap { try? EmbeddedKind.resolve(model: model, folder: $0) })?.spec ?? specs.first { $0.id == model.kind }
        }
        var names = model.resolvedOperations(kind: kind).map(\.name)
        if model.kind == "voices", names.contains(where: { ["design_voice", "clone_voice"].contains($0) }), !names.contains("delete_voice") { names.append("delete_voice") }
        return names
    }
    func resolvedOperations(id: String, candidate: Bool = false, modelOverride: ModelManifest? = nil, folderOverride: URL? = nil) async throws -> [ModelOperation] {
        if !candidate {
            if residents[id] != nil || reloading.contains(id), let operations = residentOperations[id] { return operations }
            if let previous = recoveryRevision(id) { return previous.operations }
        }
        let model = try modelOverride ?? manifest(id)
        // Inside a batch of admissions, the directory that batch verified; any other caller verifies it now.
        if let batch = KindsBatch.verified { kindsDirectory = batch } else { kindsDirectory = try await installer.verifiedKindsDirectory() }
        if modelOverride == nil, let cached = operationCache[id] { return cached }
        let failureKey = operationFailureKey(id)
        if modelOverride == nil, folderOverride == nil, let failure = operationFailures[id], failure.key == failureKey { throw failure.error }
        do {
            let ops = try await readOperations(id: id, model: model, folderOverride: folderOverride)
            if modelOverride == nil { operationCache[id] = ops; operationFailures[id] = nil }
            return ops
        } catch {
            if modelOverride == nil, folderOverride == nil { operationFailures[id] = (failureKey, error) }
            throw error
        }
    }
    private func readOperations(id: String, model: ModelManifest, folderOverride: URL?) async throws -> [ModelOperation] {
        let local = localModels[id]
        let immutable = local.map { paths.support.appendingPathComponent("state/local-folders/\($0.contentDigest)/\(id)") }
        let localRoot = folderOverride ?? immutable.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil } ?? local?.directory
        let embedded = try localRoot.flatMap { try EmbeddedKind.resolve(model: model, folder: $0) }
        let kind = embedded?.spec ?? specs.first { $0.id == model.kind }
        var ops: [ModelOperation] = []
        for declaration in model.resolvedOperations(kind: kind) {
            func read(_ path: String) async throws -> JSONObject {
                let data: Data
                if let localRoot {
                    let base =
                        model.operations.contains(where: { $0.name == declaration.name })
                        ? localRoot
                        : embedded != nil ? localRoot.appendingPathComponent("kinds/\(model.kind)") : kindsDirectory?.appendingPathComponent(model.kind) ?? localRoot
                    data = try OperationSchema.read(path: path, folder: base, kindsDirectory: embedded != nil ? localRoot.appendingPathComponent("kinds") : kindsDirectory)
                } else if path.hasPrefix("kinds/"), let kindsDirectory {
                    data = try OperationSchema.read(path: path, folder: kindsDirectory, kindsDirectory: kindsDirectory)
                } else if model.operations.contains(where: { $0.name == declaration.name }), let entry = catalog.models.first(where: { $0.id == id }),
                    let file = entry.files.first(where: { $0.path == path }), file.bytes <= 1_048_576
                {
                    data = try await source.read(path: "models/\(id)/\(path)", commit: catalog.sourceCommit)
                    guard Int64(data.count) == file.bytes, FileDigest.sha256(data) == file.sha256 else {
                        throw WorkerFailure(code: "integrity", message: "Operation schema checksum rejected.")
                    }
                } else if let kindsDirectory {
                    data = try OperationSchema.read(path: path, folder: kindsDirectory.appendingPathComponent(model.kind), kindsDirectory: kindsDirectory)
                } else {
                    throw WorkerFailure(code: "op_schema", message: "Operation schema is missing from the verified folder.")
                }
                guard data.count <= 1_048_576 else { throw WorkerFailure(code: "op_schema", message: "Operation schema exceeds 1 MiB.") }
                return try WireJSON.decode(data)
            }
            let input = try await read(declaration.input)
            let output: JSONObject?
            if let path = declaration.outputSchema { output = try await read(path) } else { output = nil }
            let timeout = declaration.timeoutS ?? (model.schema < 3 && ["generate", "generate_sound", "generate_music"].contains(declaration.name) ? 600 : 180)
            let first = declaration.firstResponseTimeoutS ?? (model.schema < 3 && declaration.name == "chat" ? 600 : timeout)
            var annotations = declaration.annotations ?? [:]
            if kind?.primaryOperation == declaration.name { annotations["primary"] = true }
            var resolved = ModelOperation(
                name: declaration.name, title: declaration.title, inputSchema: input, output: declaration.output,
                stream: declaration.stream, outputSchema: output, policy: OperationPolicy(timeoutS: timeout, firstResponseTimeoutS: first), annotations: annotations,
                schemaVersion: model.schema)
            resolved.prepareImage = declaration.prepareImage
            ops.append(resolved)
        }
        if model.kind == "voices", ops.contains(where: { ["design_voice", "clone_voice"].contains($0.name) }) {
            ops.removeAll { $0.name == "delete_voice" }
            ops.append(
                ModelOperation(
                    name: "delete_voice", title: "Delete saved voice",
                    inputSchema: ["type": "object", "required": ["voice_id"], "additionalProperties": false, "properties": ["voice_id": ["type": "string", "minLength": 1]]],
                    output: "json", stream: false))
        }
        return ops
    }
    func validatedRequest(id: String, op: String, params: JSONObject) async throws -> JSONObject {
        let (model, operation) = try await requestOperation(id: id, op: op, rawDecision: params.count == 1 && params["body"]?.string != nil)
        // TypeSafe deliberately preserves raw JSON labels/order; the decision worker owns its Verdict validator
        // and exact 422/409 body. Decoding or adding defaults here would alter that transport contract.
        if model.kind == "decisions", ["decide", "judge"].contains(op), params.count == 1, params["body"]?.string != nil,
            model.schema < 2 || operation != nil
        {
            // The bytes still go through untouched, but a schema-3 body is checked as every other schema-3 input is below:
            // against its schema (so no file object can dodge the walk by matching no branch), then for jobs/inbox
            // confinement. Both HTTP decisions routes arrive here with the raw body.
            if model.schema >= 3, let operation, let body = params["body"]?.string {
                try Self.checkSchema3Input(try LastKeyJSON.parse(Data(body.utf8)), schema: operation.inputSchema, support: paths.support)
            }
            return params
        }
        // A decisions worker reads a `body` member as a whole raw request; beside other members it would skip every check.
        if model.kind == "decisions", params["body"] != nil, operation?.inputSchema["properties"]?["body"] == nil {
            throw WorkerFailure(code: "invalid_request", message: "body is a whole raw request; send it alone or send its members.")
        }
        if let operation {
            // Ordered members arrive as their exact JSON text; validate the object, forward the text alongside.
            let (params, ordered) = try OrderedJSONMembers.split(params, schema: operation.inputSchema)
            // Request-time settings from the live state; the resident's snapshot only adds retired keys it still declares.
            let snapshot = residentSettings[id] ?? recoveryRevision(id)?.settings ?? modelSettings
            let request = snapshot.residentRequest(id, params: params, live: modelSettings, schema: operation.inputSchema)
            var schemaInput = request
            if model.schema == 3, operation.inputSchema["properties"]?["out_dir"] == nil { schemaInput["out_dir"] = nil }
            if model.schema == 3 {
                try Self.checkSchema3Input(.object(request), schema: operation.inputSchema, support: paths.support, validating: .object(schemaInput))
            } else {
                try OperationSchemaValidator.validate(.object(schemaInput), schema: .object(operation.inputSchema))
            }
            return OrderedJSONMembers.attach(ordered, to: request)
        }
        return modelSettings.request(id, params: params)
    }
    /// A schema-3 input's checks before any worker sees it: against its schema (`validating`, the input itself unless
    /// given), so no file object can dodge the walk by matching no branch, then every file path for inbox/jobs confinement.
    static func checkSchema3Input(_ input: JSONValue, schema: JSONObject, support: URL, validating: JSONValue? = nil) throws {
        try OperationSchemaValidator.validate(validating ?? input, schema: .object(schema))
        try OperationArtifacts.input(input, schema: schema, support: support)
    }
    func usesVoicePersistence(_ id: String) -> Bool {
        (residentManifests[id] ?? localModels[id]?.manifest ?? catalog.models.first { $0.id == id }?.manifest)?.kind == "voices"
    }
    func needsOutputDirectory(id: String, op: String, output: String?, request: JSONObject) -> Bool {
        let legacy = (residentManifests[id]?.schema ?? 1) < 3
        return request["out_dir"] == nil && (!legacy || request["stream"]?.bool != true)
            && (["audio", "image", "file"].contains(output ?? "") || legacy && ["generate", "speak", "design_voice"].contains(op))
    }
}

extension RealModelsService {
    public func modelREADME(id: String) async throws -> String {
        if let local = localModels[id] {
            let url = try SafePath.resolve("README.md", inside: local.directory)
            let data = try Data(contentsOf: url)
            guard data.count <= 1_048_576, FileDigest.sha256(data) == local.fileDigests["README.md"] else {
                throw WorkerFailure(code: "integrity", message: "README changed; Refresh first.")
            }
            return String(decoding: data, as: UTF8.self)
        }
        guard let entry = catalog.models.first(where: { $0.id == id }), let file = entry.files.first(where: { $0.path == "README.md" }), file.bytes <= 1_048_576 else {
            throw WorkerFailure(code: "not_found", message: "Model README unavailable.")
        }
        let data = try await source.read(path: "models/\(id)/README.md", commit: catalog.sourceCommit)
        guard Int64(data.count) == file.bytes, FileDigest.sha256(data) == file.sha256 else { throw WorkerFailure(code: "integrity", message: "README checksum rejected.") }
        return String(decoding: data, as: UTF8.self)
    }
}
