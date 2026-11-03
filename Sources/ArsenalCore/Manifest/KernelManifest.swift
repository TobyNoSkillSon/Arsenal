// SPDX-License-Identifier: MIT
import Foundation

public struct KernelDeclaration: Codable, Sendable, Equatable {
    public var id: String; public var chip: String; public var mode: String; public var files: [String]; public var entry: String; public var replaces: String
    public var api: String?; public var header: String?; public var inputs: [String]?; public var outputs: [String]?
    public var resolvedAPI: String { api ?? "mlx" }
    public init(
        id: String, chip: String, mode: String, files: [String], entry: String, replaces: String, api: String? = nil, header: String? = nil, inputs: [String]? = nil,
        outputs: [String]? = nil
    ) {
        self.id = id; self.chip = chip; self.mode = mode; self.files = files; self.entry = entry; self.replaces = replaces; self.api = api; self.header = header
        self.inputs = inputs; self.outputs = outputs
    }
}
public struct KernelManifest: Codable, Sendable, Equatable {
    public var kernel: [KernelDeclaration]
    public var kernels: [KernelDeclaration] { kernel }
    public init(kernels: [KernelDeclaration] = []) { kernel = kernels }
    public static func loadIfPresent(folder: URL) throws -> Self {
        let path = try SafePath.resolve("kernels/manifest.toml", inside: folder)
        guard FileManager.default.fileExists(atPath: path.path) else { return Self() }
        return try decode(Data(contentsOf: path)) { file in
            let url = try SafePath.resolve(file, inside: folder.appendingPathComponent("kernels"))
            return FileManager.default.fileExists(atPath: url.path)
        }
    }
    /// Validate the same declarations against either folder files or an authenticated catalog inventory.
    static func decode(_ data: Data, hasSource: (String) throws -> Bool) throws -> Self {
        guard let text = String(data: data, encoding: .utf8) else { throw ManifestError.invalid("invalid kernel manifest UTF-8") }
        let manifest = try TOMLParser.decode(Self.self, from: text)
        var keys = Set<String>(); var sites = Set<String>()
        for k in manifest.kernel {
            guard ModelManifest.validID(k.id), k.chip == "*" || k.chip.range(of: "^m[0-9]+$", options: .regularExpression) != nil, ["exact", "fast"].contains(k.mode),
                k.entry.range(of: "^[A-Za-z_][A-Za-z0-9_]*$", options: .regularExpression) != nil, !k.replaces.isEmpty, !k.files.isEmpty, Set(k.files).count == k.files.count,
                keys.insert("\(k.id):\(k.chip):\(k.mode)").inserted, sites.insert("\(k.replaces):\(k.chip):\(k.mode)").inserted, ["mlx", "metal"].contains(k.resolvedAPI),
                k.header.map({ k.files.contains($0) }) ?? true
            else { throw ManifestError.invalid("invalid/duplicate kernel declaration \(k.id)") }
            if k.resolvedAPI == "mlx" {
                guard let inputs = k.inputs, let outputs = k.outputs, !inputs.isEmpty, !outputs.isEmpty,
                    [inputs, outputs].allSatisfy({ Set($0).count == $0.count && $0.allSatisfy { $0.range(of: "^[A-Za-z_][A-Za-z0-9_]*$", options: .regularExpression) != nil } })
                else { throw ManifestError.invalid("MLX kernel \(k.id) needs input/output names") }
            }
            for file in k.files {
                guard SafePath.isRelative(file), URL(fileURLWithPath: file).pathExtension == "metal", try hasSource(file) else {
                    throw ManifestError.invalid("missing kernel Metal source \(file)")
                }
            }
        }
        return manifest
    }
    /// One implementation per replacement site. Fast includes accepted Exact tricks where no Fast override exists.
    public func resolved(chip: String, fast: Bool) -> [KernelDeclaration] {
        let family = RunnerSelection.family(chip)
        let priorities = fast ? ["fast:\(family)", "fast:*", "exact:\(family)", "exact:*"] : ["exact:\(family)", "exact:*"]
        var choices: [String: (Int, Int, KernelDeclaration)] = [:]
        for (index, k) in kernel.enumerated() {
            guard let priority = priorities.firstIndex(of: "\(k.mode):\(k.chip)") else { continue }
            if choices[k.replaces].map({ priority < $0.0 }) ?? true { choices[k.replaces] = (priority, index, k) }
        }
        return choices.values.sorted { $0.1 < $1.1 }.map { $0.2 }
    }
}
