// SPDX-License-Identifier: MIT
import Foundation

public struct ScaffoldResult: Codable, Sendable {
    public var id: String; public var directory: String; public var revision: String; public var pending: [String]
}
public enum HFScaffolder {
    public static func write(metadata: HFModelMetadata, kind: KindSpec, parent: URL, date: String, kindsDirectory: URL? = nil) throws -> ScaffoldResult {
        let parts = metadata.repo.split(separator: "/")
        guard parts.count == 2, let publisher = parts.first, let name = parts.last else { throw ManifestError.invalid("metadata needs publisher/repo") }
        let raw = name.lowercased()
        let id = raw.replacingOccurrences(of: "[^a-z0-9.-]", with: "-", options: .regularExpression)
        guard ModelManifest.validID(id), !metadata.files.isEmpty else { throw ManifestError.invalid("metadata needs a safe model id and files") }
        let folder = try SafePath.resolve(id, inside: parent)
        guard !FileManager.default.fileExists(atPath: folder.path) else { throw ManifestError.invalid("\(id) already exists; scaffolding never overwrites") }
        var pending = [
            "Review weights.files and source dtype (bf16 placeholder)", "Fill runner worker/entry/lock and accepted selection", "Confirm licence and commercial terms",
            "Fill gate and benchmark recipes", "Declare model settings and check operation schemas", "Replace scaffold log with port evidence"
        ]
        let files = metadata.files.filter { $0.path != ".gitattributes" }
        for file in files where file.sha256 == nil || file.bytes == nil { pending.append("Pin SHA256/bytes for \(file.path) (metadata has no SHA256 for Git blobs)") }
        var weights: [String: TOMLValue] = [
            "repo": .string(metadata.repo), "revision": .string(metadata.revision), "dtype": .string("bf16"), "files": .array(files.map { .string($0.path) }),
            "sha256": .table(Dictionary(uniqueKeysWithValues: files.compactMap { f in f.sha256.map { (f.path, .string($0)) } })),
            "bytes": .table(Dictionary(uniqueKeysWithValues: files.compactMap { f in f.bytes.map { (f.path, .integer($0)) } }))
        ]
        // Keeping a complete inventory lets the contributor prune intentionally instead of hiding vendor alternatives.
        if files.isEmpty { weights["files"] = .array([]); pending.append("Select runtime files") }
        let params = metadata.parameterCount.map { String($0) } ?? "TODO"
        var doc: [String: TOMLValue] = [
            "scaffold": .bool(true), "schema": .integer(2), "id": .string(id), "name": .string(String(name)), "kind": .string(kind.id),
            "summary": .string("TODO: describe this model's useful role"), "publisher": .string(String(publisher)), "released": .integer(Int64(date.prefix(4)) ?? 2026),
            "tasks": .array([.string(metadata.pipelineTag ?? kind.id)]), "languages": .array((metadata.languages.isEmpty ? ["und"] : metadata.languages).map(TOMLValue.string)),
            "params": .string(params), "suggested": .bool(false), "weights": .table(weights),
            "licence": .table([
                "spdx": .string(metadata.licence ?? "TODO"), "commercial": .bool(false), "url": .string("https://huggingface.co/\(metadata.repo)/tree/\(metadata.revision)")
            ]), "popularity": .table(["hf_downloads": .integer(metadata.downloads), "hf_likes": .integer(metadata.likes), "as_of": .string(date)]),
            "selection": .table(["default": .string("todo")]),
            "runner": .array([
                .table([
                    "name": .string("todo"), "language": .string("swift"), "worker": .string("todo"), "protocol": .integer(1), "chips": .array([.string("*")]),
                    "min_macos": .string("26.0"), "fast_exact": .bool(false)
                ])
            ]), "gate": .table(["recipe": .string("gate/gate.toml")]), "bench": .table(["recipe": .string("bench/bench.toml")])
        ]
        if let facets = kind.facets {
            doc["facets"] = .table(["modalities_in": .array(facets.modalitiesIn.map(TOMLValue.string)), "modalities_out": .array(facets.modalitiesOut.map(TOMLValue.string))])
        }
        let kindRoot = kindsDirectory ?? parent.deletingLastPathComponent().appendingPathComponent("kinds")
        let errors = kind.operations.flatMap { $0.problems(folder: kindRoot.appendingPathComponent(kind.id), kindsDirectory: kindRoot) }
        guard errors.isEmpty else { throw ManifestError.invalid(errors.joined(separator: "; ")) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for path in ["gate", "bench", "log", "ops"] { try FileManager.default.createDirectory(at: folder.appendingPathComponent(path), withIntermediateDirectories: true) }
        var ops: [TOMLValue] = []
        for op in kind.operations {
            let input = op.input.hasPrefix("kinds/") ? op.input : "kinds/\(kind.id)/" + op.input
            var fields: [String: TOMLValue] = [
                "name": .string(op.name), "title": .string(op.title), "input": .string(input), "output": .string(op.output), "stream": .bool(op.stream)
            ]
            if let output = op.outputSchema { fields["output_schema"] = .string(output.hasPrefix("kinds/") ? output : "kinds/\(kind.id)/" + output) }
            ops.append(.table(fields))
        }
        if !ops.isEmpty { doc["op"] = .array(ops) }
        try TOMLWriter.encode(doc).write(to: folder.appendingPathComponent("model.toml"), atomically: true, encoding: .utf8)
        try "# \(id)\n\nScaffold from HF metadata only. Not reviewed or run.\n\n## Remaining work\n\n".appending(pending.map { "- " + $0 }.joined(separator: "\n")).appending("\n")
            .write(to: folder.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        for path in ["gate/gate.toml", "bench/bench.toml"] {
            try "# SPDX-License-Identifier: MIT\n# TODO: replace with the reviewed recipe\n".write(to: folder.appendingPathComponent(path), atomically: true, encoding: .utf8)
        }
        try """
        +++
        hypothesis = "A folder can be scaffolded safely from immutable HF metadata."
        change = "Recorded metadata only; runner and recipes need review."
        measurement = "No weights downloaded, no model run."
        verdict = "inconclusive"
        numbers = { real_runs = 0, weight_download_bytes = 0 }
        links = ["https://huggingface.co/\(metadata.repo)/tree/\(metadata.revision)"]
        +++

        TODO: replace this scaffold with the actual port research and its evidence.

        """.write(to: folder.appendingPathComponent("log/001-scaffold.md"), atomically: true, encoding: .utf8)
        return ScaffoldResult(id: id, directory: folder.path, revision: metadata.revision, pending: pending)
    }
}
