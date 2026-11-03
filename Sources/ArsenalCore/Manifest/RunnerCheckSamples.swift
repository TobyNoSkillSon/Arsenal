// SPDX-License-Identifier: MIT
import Foundation

public enum RunnerCheckSamples {
    public static func validate(folder: URL, model: ModelManifest, kind: KindSpec? = nil, kindsDirectory: URL? = nil) throws {
        let url = try SafePath.resolve("gate/runner-check.json", inside: folder)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let data = try Data(contentsOf: url)
        guard data.count <= 1024 * 1024, let document = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ManifestError.invalid("runner-check.json must be an op-to-input object (max 1 MiB)")
        }
        let samples: [String: Any]
        var cancel: (String, Any)?
        if document["ops"] != nil || document["load"] != nil || document["cancel"] != nil {
            guard Set(document.keys).isSubset(of: ["load", "ops", "cancel", "expected"]), document["load"] == nil || document["load"] is [String: Any],
                document["ops"] == nil || document["ops"] is [String: Any]
            else { throw ManifestError.invalid("invalid structured runner-check document") }
            samples = document["ops"] as? [String: Any] ?? [:]
            if let raw = document["cancel"] {
                guard let c = raw as? [String: Any], let op = c["op"] as? String, let params = c["params"], Set(c.keys) == ["op", "params"] else {
                    throw ManifestError.invalid("runner-check cancel needs op and params")
                }
                cancel = (op, params)
            }
        } else {
            samples = document
        }
        let ops = model.resolvedOperations(kind: kind)
        var cases = samples.map { ($0.key, $0.value) }
        if let cancel { cases.append(cancel) }
        for (name, params) in cases {
            guard let op = ops.first(where: { $0.name == name }) else { throw ManifestError.invalid("runner-check sample names unknown op \(name)") }
            let schema: Data
            if model.operations.contains(where: { $0.name == name }) {
                schema = try OperationSchema.read(path: op.input, folder: folder, kindsDirectory: kindsDirectory)
            } else {
                guard let kind, let kindsDirectory else { throw ManifestError.invalid("kind schemas missing for runner-check sample \(name)") }
                schema = try OperationSchema.read(path: op.input, folder: kindsDirectory.appendingPathComponent(kind.id), kindsDirectory: kindsDirectory)
            }
            let errors = try ModelSchemaValidation.validate(value: params, schemaData: schema)
            guard errors.isEmpty else { throw ManifestError.invalid("runner-check \(name): " + errors.joined(separator: "; ")) }
        }
    }
}
