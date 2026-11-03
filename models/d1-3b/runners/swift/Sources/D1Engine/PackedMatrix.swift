// SPDX-License-Identifier: MIT
import Foundation
import MLX
import ModelPrecision

/// Dense or stored affine matrix. Quantized embeddings unpack only the requested rows.
public struct D1Matrix {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let bits: Int
    public var dtype: DType { scales?.dtype ?? weight.dtype }
    public var shape: [Int] { [weight.dim(0), inputs] }
    var inputs: Int { scales.map { $0.dim(1) * 64 } ?? weight.dim(1) }
    var arrays: [MLXArray] { [weight] + [scales, biases].compactMap { $0 } }
    init(weight: MLXArray, scales: MLXArray? = nil, biases: MLXArray? = nil) throws {
        self.weight = weight; self.scales = scales; self.biases = biases
        if weight.dtype == .uint32 {
            guard let scales, let biases, scales.ndim == 2, biases.shape == scales.shape,
                scales.dtype == .bfloat16, biases.dtype == scales.dtype,
                weight.ndim == 2, weight.dim(0) == scales.dim(0), scales.dim(1) > 0,
                [4, 8].contains(weight.dim(1) * 32 / (scales.dim(1) * 64)),
                weight.dim(1) * 32 % (scales.dim(1) * 64) == 0
            else { throw JSONError("Invalid d1 affine int4/int8 matrix") }
            bits = weight.dim(1) * 32 / (scales.dim(1) * 64)
        } else {
            guard scales == nil, biases == nil, weight.ndim == 2 else { throw JSONError("Invalid d1 dense matrix") }
            bits = 16
        }
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let scales, let biases {
            return quantizedMM(x, weight, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: bits)
        }
        return matmul(x, weight.T)
    }
    public subscript(rows: MLXArray) -> MLXArray {
        if let scales, let biases {
            return dequantized(weight[rows], scales: scales[rows], biases: biases[rows], groupSize: 64, bits: bits)
        }
        return weight[rows]
    }
    private init(validatedWeight: MLXArray, scales: MLXArray?, biases: MLXArray?, bits: Int) {
        weight = validatedWeight; self.scales = scales; self.biases = biases; self.bits = bits
    }
    func rows(_ range: Range<Int>) -> Self {
        // The source matrix has already validated the matching row shapes and bit width.
        var result = self
        result = Self(validatedWeight: weight[range], scales: scales.map { $0[range] }, biases: biases.map { $0[range] }, bits: bits)
        return result
    }
    static func concatenate(_ values: [Self]) throws -> Self {
        guard let first = values.first, values.allSatisfy({ $0.bits == first.bits && $0.inputs == first.inputs }) else {
            throw JSONError("Incompatible d1 matrix fusion")
        }
        return try Self(
            weight: concatenated(values.map(\.weight), axis: 0),
            scales: first.scales == nil ? nil : concatenated(values.compactMap(\.scales), axis: 0),
            biases: first.biases == nil ? nil : concatenated(values.compactMap(\.biases), axis: 0))
    }
}

public enum D1Precision {
    /// Get-only tensor streaming; eligible language/vision/projector matrices and the token embedding.
    public static func convert(source: URL, destination: URL, bits: Int = 4) throws {
        guard [4, 8].contains(bits) else { throw JSONError("Expected int4 or int8") }
        guard try !FileManager.default.fileExists(atPath: destination.path) || FileManager.default.contentsOfDirectory(atPath: destination.path).isEmpty else {
            throw JSONError("Conversion destination must be empty")
        }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let archive = try TensorArchive(source.appendingPathComponent("model.safetensors"))
        let coverage = try StreamedPrecision.convert(
            keys: Array(archive.entries.keys), read: archive.array, to: destination.appendingPathComponent("model.safetensors"), half: .bfloat16, bits: bits,
            eligible: { key, shape in
                key.hasSuffix(".weight") && !key.contains("position_embedding") && shape.reduce(1, *) >= 65_536
            })
        var skipped: [[String: Any]] = []
        for (key, entry) in archive.entries.sorted(by: { $0.key < $1.key }) {
            let shape = entry.shape
            let reason: String?
            if shape.count != 2 {
                reason = "Vector/norm/bias or nonmatrix; stays BF16"
            } else if key.contains("position_embedding") {
                reason = "Position lookup grid; interpolation consumes the original BF16 values"
            } else if shape.reduce(1, *) < 65_536 {
                reason = "Below the 65,536-parameter matrix cutoff"
            } else if shape[1] % 64 != 0 {
                reason = "Input width is not divisible by affine group 64; stays BF16"
            } else {
                reason = nil
            }
            if let reason { skipped.append(["key": key, "shape": shape, "reason": reason]) }
        }
        let summary: [String: Any] = [
            "bits": bits, "group_size": 64, "half": "bf16", "skipped": skipped,
            "coverage": try JSONSerialization.jsonObject(with: JSONEncoder().encode(coverage))
        ]
        try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]).write(to: destination.appendingPathComponent("d1-coverage.json"))
        for file in try FileManager.default.contentsOfDirectory(atPath: source.path) where file != "model.safetensors" && file != "arsenal-snapshot.json" {
            try FileManager.default.copyItem(at: source.appendingPathComponent(file), to: destination.appendingPathComponent(file))
        }
        let config = destination.appendingPathComponent("config.json")
        var value = try JSONSerialization.jsonObject(with: Data(contentsOf: config)) as? [String: Any] ?? [:]
        value["quantization"] = ["bits": bits, "group_size": 64, "mode": "affine"]
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: config, options: .atomic)
    }
}
