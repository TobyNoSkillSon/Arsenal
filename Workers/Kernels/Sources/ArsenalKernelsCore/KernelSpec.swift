// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit

/// One resolved kernel declaration as the supervisor delivers it in `load.config.kernels` (docs/KERNELS.md):
/// `{"id","chip","mode","replaces","entry","api","files":[abs paths],"header":abs path|null,"inputs":[…],"outputs":[…]}`.
public struct KernelSpec: Sendable, Equatable, Codable {
    public enum Mode: String, Sendable, Codable { case exact, fast }
    /// mlx: `files` (minus `header`) are the body of an `MLXFast.metalKernel`; metal: a complete MSL library.
    public enum API: String, Sendable, Codable { case mlx, metal }

    public var id: String
    public var chip: String
    public var mode: Mode
    public var replaces: String
    public var entry: String
    public var api: API
    public var files: [String]
    public var header: String?
    public var inputs: [String]
    public var outputs: [String]

    public init(
        id: String, chip: String = "*", mode: Mode, replaces: String, entry: String, api: API = .mlx,
        files: [String], header: String? = nil, inputs: [String] = [], outputs: [String] = []
    ) {
        self.id = id; self.chip = chip; self.mode = mode; self.replaces = replaces; self.entry = entry; self.api = api
        self.files = files; self.header = header; self.inputs = inputs; self.outputs = outputs
    }

    /// `load.config.kernels`: nil = key absent (the runner keeps its built-in path); [] = no kernels for this chip and
    /// mode. Throws on a malformed entry (the supervisor validated the folder; a bad entry is a supervisor bug).
    public static func parse(config value: Any?) throws -> [KernelSpec]? {
        guard let value, !(value is NSNull) else { return nil }
        guard let list = value as? [Any] else { throw KernelError.invalid("config.kernels must be an array.") }
        return try list.map { item in
            guard let o = item as? [String: Any] else { throw KernelError.invalid("config.kernels entries must be objects.") }
            func text(_ key: String) throws -> String {
                guard let s = o[key] as? String, !s.isEmpty else { throw KernelError.invalid("config.kernels entry lacks \(key).") }
                return s
            }
            func texts(_ key: String) -> [String] { (o[key] as? [Any])?.compactMap { $0 as? String } ?? [] }
            let id = try text("id")
            guard let mode = Mode(rawValue: try text("mode")) else { throw KernelError.invalid("Kernel \(id): mode must be exact or fast.") }
            let apiName = (o["api"] as? String) ?? "mlx"
            guard let api = API(rawValue: apiName) else { throw KernelError.invalid("Kernel \(id): api must be mlx or metal.") }
            let spec = KernelSpec(
                id: id, chip: (o["chip"] as? String) ?? "*", mode: mode, replaces: try text("replaces"),
                entry: try text("entry"), api: api, files: texts("files"), header: o["header"] as? String,
                inputs: texts("inputs"), outputs: texts("outputs"))
            var normalized = spec
            normalized.header = spec.headerFile
            try normalized.validate()
            return normalized
        }
    }

    /// Parses `config.kernels` from the JSON text of the array (runners with their own JSON reader).
    public static func parse(json: Data) throws -> [KernelSpec] {
        try parse(config: try JSONSerialization.jsonObject(with: json)) ?? []
    }

    /// `header` as one of `files`: the supervisor makes `files` absolute and may pass `header` as written in the
    /// manifest (relative to kernels/), so a relative header matches the file it ends with.
    var headerFile: String? {
        guard let header else { return nil }
        if files.contains(header) { return header }
        return files.first { $0.hasSuffix("/" + header) } ?? header
    }

    public func validate() throws {
        guard !files.isEmpty else { throw KernelError.invalid("Kernel \(id) lists no files.") }
        if let header, !files.contains(header) { throw KernelError.invalid("Kernel \(id): header must be one of its files.") }
        if api == .mlx {
            guard !inputs.isEmpty, !outputs.isEmpty else { throw KernelError.invalid("Kernel \(id): an mlx kernel needs inputs and outputs.") }
        }
        guard entry.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "_") }), entry.first?.isLetter == true else {
            throw KernelError.invalid("Kernel \(id): entry must be a Metal identifier.")
        }
    }

    /// The JSON object the supervisor sends (and tests/tools build).
    public var configObject: [String: Any] {
        var o: [String: Any] = [
            "id": id, "chip": chip, "mode": mode.rawValue, "replaces": replaces, "entry": entry,
            "api": api.rawValue, "files": files, "inputs": inputs, "outputs": outputs
        ]
        o["header"] = header ?? NSNull()
        return o
    }

    /// The resolution rule (docs/KERNELS.md): one kernel per `replaces` site. exact: exact@family > exact@*.
    /// fast: fast@family > fast@* > exact@family > exact@* (Fast includes the Exact tricks). Output in declaration order.
    public static func resolve(_ declared: [KernelSpec], chipFamily: String, mode: Mode) -> [KernelSpec] {
        func rank(_ k: KernelSpec) -> Int? {
            let chipRank: Int
            if k.chip == chipFamily { chipRank = 0 } else if k.chip == "*" { chipRank = 1 } else { return nil }
            switch (mode, k.mode) {
            case (.exact, .exact): return chipRank
            case (.exact, .fast): return nil
            case (.fast, .fast): return chipRank
            case (.fast, .exact): return 2 + chipRank
            }
        }
        var best: [String: (rank: Int, index: Int)] = [:]
        for (index, k) in declared.enumerated() {
            guard let r = rank(k) else { continue }
            if let current = best[k.replaces], current.rank <= r { continue }
            best[k.replaces] = (r, index)
        }
        return best.values.map(\.index).sorted().map { declared[$0] }
    }
}

/// A kernel's sources read from the folder, checked and digested.
public struct LoadedKernel: Sendable {
    public let spec: KernelSpec
    /// api mlx: text before MLX's generated signature; api metal: empty.
    public let header: String
    /// api mlx: the kernel body; api metal: the whole library source.
    public let body: String
    /// The kernel's identity in every cache and verdict key: sha256 over everything that decides what it computes on
    /// the runner's buffers, i.e. the site, mode, api and entry, the ordered input and output names (MLX binds
    /// buffers to names by position, so swapping `x`/`w` turns `x - w` into `w - x` with the same source text) and
    /// the header and body. `id` and `chip` are labels and stay out. Versioned: a new encoding never matches an old
    /// verdict.
    public let digest: String
    static let identityVersion = "mobius-kernel-identity-2" // legacy-compat: hashed into every recorded kernel identity

    public init(spec: KernelSpec, header: String, body: String) throws {
        // MLX hands sources to Metal as ASCII (NS::ASCIIStringEncoding): anything else fails inside MLX's compile.
        for (name, text) in [("header", header), ("body", body)] where !text.utf8.allSatisfy({ $0 < 0x80 }) {
            throw KernelError.invalid("Kernel \(spec.id): the \(name) has non-ASCII characters; Metal sources must be ASCII.")
        }
        self.spec = spec; self.header = header; self.body = body
        self.digest = Self.identity(spec: spec, header: header, body: body)
    }

    /// Length-prefixed fields in a fixed order (unambiguous: no separator can be forged by a name or a source).
    static func identity(spec: KernelSpec, header: String, body: String) -> String {
        func field(_ text: String) -> String { "\(text.utf8.count):\(text)" }
        func list(_ items: [String]) -> String { "\(items.count)[" + items.map(field).joined() + "]" }
        let encoded = [
            field(identityVersion), field(spec.replaces), field(spec.mode.rawValue), field(spec.api.rawValue),
            field(spec.entry), list(spec.inputs), list(spec.outputs), field(header), field(body)
        ].joined()
        return KernelDigest.sha256(encoded)
    }

    /// Reads the spec's files (absolute paths, or relative to `base`).
    public static func load(_ spec: KernelSpec, base: URL? = nil) throws -> LoadedKernel {
        func read(_ path: String) throws -> String {
            let url = path.hasPrefix("/") || base == nil ? URL(fileURLWithPath: path) : base!.appendingPathComponent(path)
            guard url.pathExtension == "metal" else { throw KernelError.invalid("Kernel \(spec.id): \(url.lastPathComponent) is not a .metal file.") }
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { throw KernelError.invalid("Kernel \(spec.id): cannot read \(url.path).") }
            return text
        }
        let header = try spec.header.map(read) ?? ""
        let body = try spec.files.filter { $0 != spec.header }.map(read).joined(separator: "\n")
        return try LoadedKernel(spec: spec, header: header, body: body)
    }
}

public enum KernelDigest {
    public static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

public struct KernelError: Error, CustomStringConvertible, Sendable {
    public let message: String
    public static func invalid(_ message: String) -> KernelError { KernelError(message: message) }
    public var description: String { message }
}
