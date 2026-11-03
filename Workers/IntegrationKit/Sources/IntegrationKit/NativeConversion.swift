// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import MLX
import ModelPrecision
import ArsenalWire

/// Verified source snapshots are read-only. Write a complete independently loadable selected snapshot into destination.
public struct NativeConversionContext {
    public let source: URL
    public let assets: [String: URL]
    public let destination: URL
    public let precision: String
    public let half: DType
    /// The verified model folder (`model_dir`), for converters whose packing rule reads the folder's runner spec.
    public let folder: URL?
    public init(source: URL, assets: [String: URL], destination: URL, precision: String, half: DType, folder: URL? = nil) {
        self.source = source; self.assets = assets; self.destination = destination; self.precision = precision; self.half = half; self.folder = folder
    }
    public var bits: Int? { precision == "int8" ? 8 : precision == "int4" ? 4 : nil }

    /// Sorted relative regular files, including nested tokenizer/config paths. Links and special files are refused.
    public static func files(in root: URL) throws -> [String] {
        let fm = FileManager.default
        guard try root.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else { throw PrecisionError("Snapshot root contains a symbolic link") }
        var result: [String] = []
        func walk(_ directory: URL, prefix: String) throws {
            for file in try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]) {
                let info = try file.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey])
                guard info.isSymbolicLink != true else { throw PrecisionError("Snapshot contains a symbolic link") }
                let relative = prefix + file.lastPathComponent
                if info.isDirectory == true {
                    try walk(file, prefix: relative + "/")
                } else if info.isRegularFile == true {
                    result.append(relative)
                } else {
                    throw PrecisionError("Snapshot contains a special file")
                }
            }
        }
        try walk(root, prefix: "")
        return result.sorted()
    }

    /// Apply the ordinary streamed precision conversion to a component, preserving its relative layout.
    /// Custom converters may instead use TensorArchive/TensorWriter and StreamedPrecision for bounded groups.
    public func convertTree(from root: URL, into relative: String = "", cancellation: NativeCancellation, preserveHalfFiles: Bool = false, eligible: (String, [Int]) -> Bool) throws
    {
        guard relative.isEmpty || Self.safe(relative) else { throw PrecisionError("Unsafe component destination") }
        let output = relative.isEmpty ? destination : destination.appendingPathComponent(relative)
        for path in try Self.files(in: root) {
            try cancellation.check()
            if StoredNames.reserved.contains(URL(fileURLWithPath: path).lastPathComponent) { continue }
            let source = root.appendingPathComponent(path), target = output.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard !FileManager.default.fileExists(atPath: target.path) else { throw PrecisionError("Duplicate conversion output: " + path) }
            if source.pathExtension == "safetensors" {
                let archive = try TensorArchive(source)
                let packed = archive.entries.keys.contains { $0.hasSuffix(".scales") || $0.hasSuffix(".biases") }
                guard !packed else { throw PrecisionError("Normalization/conversion requires an original half source, never a packed source") }
                if preserveHalfFiles, archive.entries.values.allSatisfy({ ![.float16, .bfloat16, .float32, .float64].contains($0.dtype) || $0.dtype == half }) {
                    if half == .float16 {
                        for key in archive.entries.keys.sorted() where archive.entries[key]?.dtype == .float16 {
                            try autoreleasepool {
                                try cancellation.check()
                                let tensor = try archive.array(key)
                                guard !MLX.any(.!MLX.isFinite(tensor, stream: .cpu), stream: .cpu).item(Bool.self) else { throw PrecisionError("FP16 non-finite tensor: " + key) }
                            }
                        }
                    }
                    if clonefile(source.path, target.path, 0) != 0 { try FileManager.default.copyItem(at: source, to: target) }
                    continue
                }
                _ = try StreamedPrecision.convert(
                    keys: archive.entries.keys.sorted(),
                    read: { key in
                        try cancellation.check(); return try archive.array(key)
                    }, to: target, half: half, bits: bits, eligible: eligible)
            } else {
                try FileManager.default.copyItem(at: source, to: target)
            }
        }
    }

    /// Ordinary Get receipt: weights have no operation-artifact size limit or file ID/MIME fields.
    public func receipt(cancellation: NativeCancellation) throws -> JSONObject {
        let files = try Self.files(in: destination).map { path -> JSONValue in
            try cancellation.check()
            let file = destination.appendingPathComponent(path)
            return ["path": .string(path), "bytes": .number(Double(try NativeFiles.bytes(file))), "sha256": .string(try NativeFiles.sha256(file))]
        }
        return ["files": .array(files)]
    }
    private static func safe(_ path: String) -> Bool {
        !path.isEmpty && !path.hasPrefix("/") && !path.contains("\\")
            && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
}

/// Versioned model-owned conversion. The existing initializer remains the recursive identity-layout policy.
public struct NativeConversion {
    public let recipe: String
    public let half: DType
    public let eligible: (String, [Int]) -> Bool
    private let implementation: ((NativeConversionContext, NativeCancellation) throws -> JSONObject)?
    public init(recipe: String, half: DType = .float16, eligible: @escaping (String, [Int]) -> Bool) {
        self.recipe = recipe; self.half = half; self.eligible = eligible; implementation = nil
    }
    public init(recipe: String, half: DType = .float16, convert: @escaping (NativeConversionContext, NativeCancellation) throws -> JSONObject) {
        self.recipe = recipe; self.half = half; eligible = { _, _ in false }; implementation = convert
    }
    /// The flat streamed policy plus auxiliary repositories stored once inside the snapshot. `components` maps each
    /// `[[asset]]` id to the subdirectory its folder declares as `into`. The base `cast:` call converts the source and
    /// every component into the selected half; a derived tier receives that base snapshot, whose components it packs
    /// like any other tensors (the raw asset is not supplied then). Load reads `weights/<into>` and converts nothing.
    public init(
        recipe: String, half: DType = .float16, eligible: @escaping (String, [Int]) -> Bool, components: [String: String],
        componentEligible: ((String, [Int]) -> Bool)? = nil
    ) {
        self.init(recipe: recipe, half: half) { context, cancellation in
            try context.convertTree(from: context.source, cancellation: cancellation, eligible: eligible)
            for (id, directory) in components.sorted(by: { $0.key < $1.key }) {
                guard !FileManager.default.fileExists(atPath: context.destination.appendingPathComponent(directory).path) else { continue }
                guard let asset = context.assets[id] else { throw PrecisionError("Missing component asset " + id) }
                try context.convertTree(from: asset, into: directory, cancellation: cancellation, eligible: componentEligible ?? eligible)
            }
            return try context.receipt(cancellation: cancellation)
        }
    }
    public static let normalizationRecipe = "cast:native-half-v1"
    public static func normalize(_ request: JSONObject, cancellation: NativeCancellation) throws -> JSONObject {
        guard let dtype = request["dtype"]?.string, ["fp16", "bf16"].contains(dtype) else { throw PrecisionError("Normalization requires a selected half format") }
        return try NativeConversion(recipe: normalizationRecipe, half: dtype == "bf16" ? .bfloat16 : .float16, eligible: { _, _ in false }).run(
            request, cancellation: cancellation, metadataOnly: true)
    }
    public func run(_ request: JSONObject, cancellation: NativeCancellation) throws -> JSONObject { try run(request, cancellation: cancellation, metadataOnly: false) }
    private func run(_ request: JSONObject, cancellation: NativeCancellation, metadataOnly: Bool) throws -> JSONObject {
        guard let source = request["src_dir"]?.string, let destination = request["dst_dir"]?.string,
            let dtype = request["dtype"]?.string, ["fp16", "bf16", "int8", "int4"].contains(dtype), request["group_size"]?.number == 64,
            request["recipe"]?.string == recipe, [.float16, .bfloat16].contains(half),
            dtype != "fp16" || half == .float16, dtype != "bf16" || half == .bfloat16
        else { throw WorkerFailure(code: "unsupported_recipe", message: "Conversion recipe or precision does not match its registered model.") }
        let src = URL(fileURLWithPath: source).standardizedFileURL, dst = URL(fileURLWithPath: destination).standardizedFileURL
        let rawAssets = request["assets"]?.object ?? [:]
        guard rawAssets.values.allSatisfy({ $0.string != nil }) else { throw PrecisionError("Conversion assets must be snapshot paths") }
        let assets = rawAssets.mapValues { URL(fileURLWithPath: $0.string!).standardizedFileURL }
        for input in [src] + Array(assets.values) {
            let input = input.resolvingSymlinksInPath(), dst = dst.resolvingSymlinksInPath()
            guard input != dst, !dst.path.hasPrefix(input.path + "/"), !input.path.hasPrefix(dst.path + "/") else {
                throw PrecisionError("Convert into separate staging, outside source/assets")
            }
            _ = try NativeConversionContext.files(in: input)
        }
        try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
        guard try NativeConversionContext.files(in: dst).isEmpty else { throw PrecisionError("Conversion destination must be empty") }
        let folder = request["model_dir"]?.string.map { URL(fileURLWithPath: $0).standardizedFileURL }
        let context = NativeConversionContext(source: src, assets: assets, destination: dst, precision: dtype, half: half, folder: folder)
        try cancellation.check()
        let result: JSONObject
        if let implementation {
            result = try implementation(context, cancellation)
        } else {
            try context.convertTree(from: src, cancellation: cancellation, preserveHalfFiles: metadataOnly, eligible: eligible)
            result = try context.receipt(cancellation: cancellation)
        }
        try cancellation.check()
        let paths = result["files"]?.array?.compactMap { $0.object?["path"]?.string } ?? []
        let actual = try NativeConversionContext.files(in: dst)
        // A per-file cast (Get's castFiles) also hands a recipe a source holding only config files; those yield no tensors.
        // Any weight file in the source still requires converted tensors.
        let weightFormats: Set<String> = ["safetensors", "bin", "pt", "pth", "npz", "gguf", "ckpt"]
        let sourceWeights = try NativeConversionContext.files(in: src).contains { weightFormats.contains(URL(fileURLWithPath: $0).pathExtension) }
        let tensors = actual.contains { $0.hasSuffix(".safetensors") }
        guard !paths.isEmpty, Set(paths).count == paths.count, Set(paths) == Set(actual), metadataOnly || tensors || !sourceWeights else {
            throw PrecisionError("Incomplete conversion receipt or no tensors")
        }
        return result
    }
}
