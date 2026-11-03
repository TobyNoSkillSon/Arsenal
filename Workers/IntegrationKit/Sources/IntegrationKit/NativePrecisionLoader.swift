// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import ModelPrecision

/// Get stores affine packs; Load installs those packs directly. Half sources may also derive a tier at Load.
public struct NativePrecisionTier {
    public let half: DType
    public let bits: Int?
    public let retainedHalf: Set<String>
    public init(_ precision: String, half: DType = .float16, retainedHalf: Set<String> = []) throws {
        guard half == .float16 || half == .bfloat16, ["fp16", "bf16", "int8", "int4"].contains(precision),
            precision != "fp16" || half == .float16, precision != "bf16" || half == .bfloat16
        else { throw PrecisionError("Invalid native precision tier") }
        guard precision != "int4" || retainedHalf.isEmpty else { throw PrecisionError("Int4 packs every eligible component; half fallback is an int8 quality decision") }
        self.half = half; self.bits = precision == "int8" ? 8 : precision == "int4" ? 4 : nil; self.retainedHalf = retainedHalf
    }
    public func load(
        _ module: Module, archives: [URL], map: (String, MLXArray) throws -> (String, MLXArray)?,
        eligible: (String, [Int]) -> Bool
    ) throws -> StreamedPrecision.Coverage {
        let expected = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.shape) })
        var sources: [String: TensorArchive] = [:]
        for url in archives {
            let archive = try TensorArchive(url)
            for key in archive.entries.keys {
                guard sources.updateValue(archive, forKey: key) == nil else { throw PrecisionError("Duplicate checkpoint tensor: \(key)") }
            }
        }
        var seen = Set<String>(), consumed = Set<String>(), coverage = StreamedPrecision.Coverage()
        func read(_ key: String) throws -> (String, MLXArray)? {
            guard let archive = sources[key] else { throw PrecisionError("Missing affine companion: \(key)") }
            let array = try archive.array(key)
            coverage.tensors += 1; coverage.sourceBytes += array.nbytes
            coverage.largestTensorBytes = max(coverage.largestTensorBytes, array.nbytes)
            consumed.insert(key)
            return try map(key, array)
        }
        for key in sources.keys.sorted() where !key.hasSuffix(".scales") && !key.hasSuffix(".biases") {
            try autoreleasepool {
                guard let (name, source) = try read(key) else { return }
                guard seen.insert(name).inserted, let shape = expected[name] else { throw PrecisionError("Duplicate or unknown mapped parameter: \(name)") }
                let owner = name.hasSuffix(".weight") ? String(name.dropLast(7)) : ""
                let packed = source.dtype == .uint32 && name.hasSuffix(".weight")
                if packed {
                    guard let bits, !retainedHalf.contains(owner), shape.count == 2, shape[1] % 64 == 0,
                        eligible(name, shape), source.shape == [shape[0], shape[1] * bits / 32]
                    else { throw PrecisionError("Stored pack dimensions/selected bits do not match \(name)") }
                    let prefix = String(key.dropLast(7))
                    guard let (sn, scales) = try read(prefix + ".scales"), let (bn, biases) = try read(prefix + ".biases"),
                        sn == owner + ".scales", bn == owner + ".biases", scales.shape == [shape[0], shape[1] / 64],
                        biases.shape == scales.shape, scales.dtype == half, biases.dtype == half
                    else { throw PrecisionError("Invalid mapped affine companions for \(name)") }
                    try install(module, owner: owner, weight: source, scales: scales, biases: biases, bits: bits)
                    seen.formUnion([sn, bn]); coverage.quantizedTensors += 1
                    coverage.outputBytes += source.nbytes + scales.nbytes + biases.nbytes
                } else {
                    guard source.shape == shape else { throw PrecisionError("Checkpoint shape mismatch: \(name): \(source.shape) != \(shape)") }
                    let floating = source.dtype.isFloatingPoint
                    guard !floating || source.dtype == .float16 || source.dtype == .bfloat16 else { throw PrecisionError("Prepare the half checkpoint before Load: \(key)") }
                    let value = floating ? source.asType(half) : source
                    if floating, MLX.any(.!MLX.isFinite(value)).item(Bool.self) { throw PrecisionError("Non-finite half tensor: \(key)") }
                    if let bits, floating, name.hasSuffix(".weight"), shape.count == 2, shape[1] % 64 == 0,
                        !retainedHalf.contains(owner), eligible(name, shape)
                    {
                        let (weight, scales, biases) = MLX.quantized(value, groupSize: 64, bits: bits, mode: .affine)
                        guard let biases else { throw PrecisionError("Missing affine biases") }
                        try install(module, owner: owner, weight: weight, scales: scales, biases: biases, bits: bits)
                        coverage.quantizedTensors += 1; coverage.outputBytes += weight.nbytes + scales.nbytes + biases.nbytes
                    } else {
                        eval(value)
                        try module.update(parameters: ModuleParameters.unflattened([(name, value)]), verify: [.noUnusedKeys, .shapeMismatch])
                        coverage.outputBytes += value.nbytes
                    }
                }
            }
        }
        guard Set(sources.keys).subtracting(consumed).isEmpty else { throw PrecisionError("Orphan affine companions in checkpoint") }
        let missing = Set(expected.keys).subtracting(seen)
        guard missing.isEmpty else { throw PrecisionError("Checkpoint omitted model parameters: " + missing.sorted().prefix(8).joined(separator: ", ")) }
        guard module.parameters().flattened().allSatisfy({ !$0.1.dtype.isFloatingPoint || $0.1.dtype == half }) else {
            throw PrecisionError("Native Load retained full-precision learned parameters")
        }
        module.train(false); return coverage
    }
    private func install(_ module: Module, owner: String, weight: MLXArray, scales: MLXArray, biases: MLXArray, bits: Int) throws {
        guard let layer = module.leafModules().flattened().first(where: { $0.0 == owner })?.1 else { throw PrecisionError("No module for \(owner)") }
        let replacement: Module
        if let linear = layer as? Linear {
            if let bias = linear.bias, bias.shape != [weight.dim(0)] { throw PrecisionError("Incompatible linear bias: \(owner)") }
            replacement = QuantizedLinear(weight: weight, bias: linear.bias, scales: scales, biases: biases, groupSize: 64, bits: bits)
        } else if layer is Embedding {
            replacement = QuantizedEmbedding(weight: weight, scales: scales, biases: biases, groupSize: 64, bits: bits)
        } else {
            throw PrecisionError("Eligible matrix must be a Linear or Embedding: \(owner)")
        }
        eval(weight, scales, biases)
        try Self.replace(module, path: owner, with: replacement)
    }
    /// Same immediate-owner rule as StoredVoiceWeights: sparse recursive arrays discard unchanged prefixes.
    private static func replace(_ module: Module, path: String, with replacement: Module) throws {
        guard
            let (parentPath, parent) = module.namedModules().sorted(by: { $0.0.count > $1.0.count }).first(where: {
                $0.0.isEmpty || path.hasPrefix($0.0 + ".") && $0.0 != path
            })
        else { throw PrecisionError("Missing pack owner") }
        let tail = parentPath.isEmpty ? path : String(path.dropFirst(parentPath.count + 1))
        let pieces = tail.split(separator: ".").map(String.init)
        if pieces.count == 1 {
            try parent.update(modules: ModuleChildren(values: [pieces[0]: .value(replacement)]), verify: .none)
        } else if pieces.count == 2, let index = Int(pieces[1]), case .array(let old) = parent.children()[pieces[0]], old.indices.contains(index) {
            var array = old; array[index] = .value(replacement)
            try parent.update(modules: ModuleChildren(values: [pieces[0]: .array(array)]), verify: .none)
        } else {
            throw PrecisionError("Unsupported pack owner: " + path)
        }
    }
}
