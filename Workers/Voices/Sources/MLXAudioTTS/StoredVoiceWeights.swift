// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN

/// Install stored affine packs without quantizing a randomly initialized dense placeholder.
/// Used by the ordinary voice loaders, including their separately loaded codecs and speaker encoder.
public enum StoredVoiceWeights {
    public static func dtype(_ layer: Linear) -> DType {
        (layer as? QuantizedLinear)?.scales.dtype ?? layer.weight.dtype
    }
    public static func halfType(_ module: Module) -> DType {
        module.parameters().flattened().first { [.float16, .bfloat16].contains($0.1.dtype) }?.1.dtype ?? .bfloat16
    }
    public static func isIntake(_ config: Data) -> Bool {
        let object = (try? JSONSerialization.jsonObject(with: config)) as? [String: Any]
        return object?["arsenal_voice_intake"] != nil || object?["mobius_voice_intake"] != nil // legacy-compat: read in snapshots written before the rename
    }
    @discardableResult
    public static func prepare(_ module: Module, weights: [String: MLXArray], group: Int = 64) throws -> Int {
        guard group == 64 else { throw invalid("Stored voice packs require group 64") }
        let scaled = Set(weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(7)) })
        var updates: [(String, Module)] = []
        var installed = Set<String>()
        for (path, layer) in module.leafModules().flattened() {
            guard scaled.contains(path) else { continue }
            let shape: [Int]
            let bias: MLXArray?
            if let linear = layer as? Linear, !(linear is QuantizedLinear) {
                shape = linear.weight.shape; bias = linear.bias
            } else if let embedding = layer as? Embedding, !(embedding is QuantizedEmbedding) {
                shape = embedding.weight.shape; bias = nil
            } else if let conv = layer as? Conv1d, !(conv is StoredVoicePointwise), conv.weight.ndim == 3, conv.weight.dim(1) == 1,
                      conv.stride == 1, conv.padding == 0, conv.groups == 1 {
                shape = [conv.weight.dim(0), conv.weight.dim(2)]; bias = conv.bias
            } else { throw invalid("\(path) has a stored pack but is not a supported Linear, Embedding or ungrouped 1x1 Conv1d") }
            guard shape.count == 2, shape[0] > 0, shape[1] > 0, let weight = weights[path + ".weight"], let scales = weights[path + ".scales"],
                let biases = weights[path + ".biases"], weight.dtype == .uint32,
                scales.shape == [shape[0], shape[1] / group], biases.shape == scales.shape,
                shape[1] % group == 0, weight.ndim == 2, weight.dim(0) == shape[0],
                weight.dim(1) * 32 % shape[1] == 0
            else { throw invalid("\(path): incomplete or incompatible affine pack") }
            let bits = weight.dim(1) * 32 / shape[1]
            guard bits == 8 || bits == 4, scales.dtype == .float16 || scales.dtype == .bfloat16, biases.dtype == scales.dtype else {
                throw invalid("\(path): expected affine int8/int4 with half scales and biases")
            }
            let packed: Module
            if layer is Embedding {
                packed = QuantizedEmbedding(weight: weight, scales: scales, biases: biases, groupSize: group, bits: bits)
            } else if layer is Conv1d {
                packed = StoredVoicePointwise(weight: weight, bias: bias, scales: scales, biases: biases, inputs: shape[1], group: group, bits: bits)
            } else {
                packed = QuantizedLinear(weight: weight, bias: bias, scales: scales, biases: biases, groupSize: group, bits: bits)
            }
            updates.append((path, packed)); installed.insert(path)
        }
        guard scaled == installed else { throw invalid("Stored pack has no matching model module") }
        // Update the immediate owning module. MLX's recursive array updater rejects sparse
        // arrays beginning with an unchanged activation (e.g. [SiLU, packed Linear]).
        // Preserve the complete immediate array, without replacing unrelated let modules.
        for (path, replacement) in updates { try replace(module, path: path, with: replacement) }
        return installed.count
    }
    public static func replace(_ module: Module, path: String, with replacement: Module) throws {
        let parents = module.namedModules().sorted { $0.0.count > $1.0.count }

        guard let (parentPath, parent) = parents.first(where: { $0.0.isEmpty || path.hasPrefix($0.0 + ".") && $0.0 != path }) else { throw invalid("Missing pack owner") }
        let tail = parentPath.isEmpty ? path : String(path.dropFirst(parentPath.count + 1))
        let pieces = tail.split(separator: ".").map(String.init)
        if pieces.count == 1 {
            try parent.update(modules: ModuleChildren(values: [pieces[0]: .value(replacement)]), verify: .none)
        } else if pieces.count == 2, let index = Int(pieces[1]), case .array(let old) = parent.children()[pieces[0]] {
            var array = try old.map { item -> NestedItem<String, Module> in
                guard case .value(let child) = item else { throw invalid("Unsupported packed module array") }
                return .value(child)
            }
            guard array.indices.contains(index) else { throw invalid("Packed module array index out of range") }
            array[index] = .value(replacement)
            try parent.update(modules: ModuleChildren(values: [pieces[0]: .array(array)]), verify: .none)
        } else { throw invalid("Unsupported stored pack owner: " + path) }
    }
    public static func affinePackingAudit(dense: MLXArray, packed: MLXArray, scales: MLXArray, biases: MLXArray) throws -> [String: Double] {
        let rows = min(4, dense.dim(0)), width = dense.dim(1)
        let w = dense[0..<rows].asArray(Float.self)
        let codes = packed[0..<rows].asArray(UInt32.self)
        let s = scales[0..<rows].asArray(Float.self), b = biases[0..<rows].asArray(Float.self)
        let unpacked = dequantized(packed[0..<rows], scales: scales[0..<rows], biases: biases[0..<rows], groupSize: 64, bits: 8, stream: .gpu)
        let decoded = unpacked.asArray(Float.self)
        var energy = 0.0, error = 0.0, unpackError = 0.0
        for i in w.indices {
            let code = (codes[i / 4] >> UInt32((i % 4) * 8)) & 255
            let expected = Float(code) * s[i / 64] + b[i / 64]
            // Half rounding at the public dequantization boundary is expected.
            let rounded = dense.dtype == .float16 ? Float(Float16(expected)) : Float(bitPattern: (expected.bitPattern &+ 0x7fff &+ ((expected.bitPattern >> 16) & 1)) & 0xffff0000)
            energy += Double(w[i]) * Double(w[i]); error += pow(Double(decoded[i] - w[i]), 2)
            unpackError = max(unpackError, abs(Double(decoded[i] - rounded)))
        }
        guard unpackError == 0 else { throw invalid("Affine scalar unpack disagrees with GPU dequantization: max \(unpackError), \(dense.dtype)") }
        return ["sample_rows": Double(rows), "input_width": Double(width), "groups_per_row": Double(width / 64),
                "weight_relative_l2": sqrt(error / max(energy, 1e-300)), "manual_unpack_max_abs": unpackError,
                "scales_bytes": Double(scales.dtype.size), "biases_bytes": Double(biases.dtype.size)]
    }
    private static func invalid(_ message: String) -> NSError {
        NSError(domain: "StoredVoiceWeights", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
