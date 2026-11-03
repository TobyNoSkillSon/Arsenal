// SPDX-License-Identifier: MIT
import Foundation

/// Embedded roots are explicit: schemas never fall through to the shared signed inventory.
public struct EmbeddedKind: Sendable, Equatable {
    public var spec: KindSpec
    public var sourceRoot: URL
    public var descriptorHash: String
    public static func resolve(model: ModelManifest, folder: URL) throws -> Self? {
        guard let path = model.kindFile else { return nil }
        guard model.schema == 3, model.kind.hasPrefix("local."), path == "kinds/\(model.kind).toml" else {
            throw ManifestError.invalid("kind_file must be kinds/<local.kind>.toml in schema 3")
        }
        let file = try SafePath.resolve(path, inside: folder)
        let spec = try TOMLParser.decode(KindSpec.self, from: String(contentsOf: file, encoding: .utf8))
        guard spec.id == model.kind, !spec.title.isEmpty, !(spec.icon?.isEmpty ?? true), !(spec.titleSingular?.isEmpty ?? true),
            Set(spec.columns.map(\.id)).count == spec.columns.count,
            Set(spec.operations.map(\.name)).count == spec.operations.count,
            spec.primaryOperation == nil || spec.operations.contains(where: { $0.name == spec.primaryOperation }),
            spec.comparisonDirection == nil || ["higher", "lower"].contains(spec.comparisonDirection!)
        else { throw ManifestError.invalid("invalid embedded kind descriptor") }
        let root = folder.appendingPathComponent("kinds")
        let problems = spec.operations.flatMap { $0.problems(folder: root.appendingPathComponent(spec.id), kindsDirectory: root) } + (spec.facets?.problems() ?? [])
        guard problems.isEmpty, !spec.operations.contains(where: { $0.outputSchema == nil }) else {
            throw ManifestError.invalid("embedded kind needs complete valid operations: " + problems.joined(separator: "; "))
        }
        let paths = try SafePath.files(in: folder).filter { $0 == path || $0.hasPrefix("kinds/\(spec.id)/") }
        var bytes = Data()
        for path in paths.sorted() { bytes.append(Data(path.utf8)); bytes.append(Data(try FileDigest.sha256(SafePath.resolve(path, inside: folder)).utf8)) }
        return Self(spec: spec, sourceRoot: folder, descriptorHash: FileDigest.sha256(bytes))
    }

    /// An embedded kind may repeat a signed kind's ID only when it is that kind: the same descriptor (the signed copy's
    /// bench suite aside) and byte-identical operation schemas, as a local folder made before its kind was signed has.
    public static func repeatsSigned(model: ModelManifest, folder: URL, kindsDirectory: URL) -> Bool {
        func tree(_ root: URL) -> [String: String]? {
            guard let paths = try? SafePath.files(in: root) else { return nil }
            var hashes: [String: String] = [:]
            for path in paths { guard let file = try? SafePath.resolve(path, inside: root), let hash = try? FileDigest.sha256(file) else { return nil }; hashes[path] = hash }
            return hashes
        }
        guard let path = model.kindFile, let file = try? SafePath.resolve(path, inside: folder),
            let embedded = try? TOMLParser.decode(KindSpec.self, from: String(contentsOf: file, encoding: .utf8)),
            var signed = try? TOMLParser.decode(KindSpec.self, from: String(contentsOf: kindsDirectory.appendingPathComponent(model.kind + ".toml"), encoding: .utf8))
        else { return false }
        signed.suite = embedded.suite
        guard signed == embedded, let local = tree(folder.appendingPathComponent("kinds/\(model.kind)")) else { return false }
        return local == tree(kindsDirectory.appendingPathComponent(model.kind))
    }
}
