// SPDX-License-Identifier: MIT
// Original PyTorch checkpoint stays byte-identical on disk. Layout views are made on the GPU.
import Foundation
import MLX
import MLXNN
import ModelPrecision
extension BiRefNet {
    public func loadWeights(url: URL, dtype: DType = .float16, strict: Bool = true, bits: Int? = nil) throws {
        guard dtype == .float16 else { throw LoadError.precision }
        let archive = try TensorArchive(url)
        let scaled = archive.entries.keys.filter { $0.hasSuffix(".scales") }.count
        guard archive.entries.count == 754 + 2 * scaled, (bits == nil) == (scaled == 0) else { throw LoadError.inventory }
        var remapped: [(String, MLXArray)] = []
        var buffers = 0
        for key in archive.entries.keys.sorted() {
            let value = try autoreleasepool { try archive.array(key) }
            if key.hasSuffix(".relative_position_index") || key.hasSuffix(".num_batches_tracked") {
                guard value.dtype == .int64 else { throw LoadError.inventory }
                buffers += 1
                continue
            }
            guard value.dtype == .float16 || (bits != nil && key.hasSuffix(".weight") && value.dtype == .uint32) else { throw LoadError.precision }
            let name = key.replacingOccurrences(of: "conv_out1.0.", with: "conv_out1.")
            remapped.append((name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value))
        }
        guard buffers == 67, remapped.count == 687 + 2 * scaled else { throw LoadError.inventory }
        let weights = Dictionary(uniqueKeysWithValues: remapped)
        if let bits {
            try BiRefNetPrecision.prepare(self, weights: weights, bits: bits)
        }
        let expected = storedWeightNames()
        guard Set(remapped.map { $0.0 }) == expected else {
            print("missing", expected.subtracting(remapped.map { $0.0 }).sorted())
            print("extra", Set(remapped.map { $0.0 }).subtracting(expected).sorted())
            throw LoadError.inventory
        }
        try update(parameters: ModuleParameters.unflattened(remapped), verify: strict ? [.noUnusedKeys, .shapeMismatch] : [])
        guard parameters().flattened().allSatisfy({ !$0.1.dtype.isFloatingPoint || $0.1.dtype == .float16 }) else { throw LoadError.precision }
        train(false)
        eval(parameters())
    }
    // Computed integer relative-position indices are module buffers, not checkpoint parameters.
    // Packed uint32 matrices ARE stored parameters and must survive this inventory filter.
    func storedWeightNames() -> Set<String> {
        Set(parameters().flattened().filter { $0.1.dtype.isFloatingPoint || $0.1.dtype == .uint32 }.map { $0.0 })
    }
    public enum LoadError: Error, Equatable { case precision, inventory }
}
