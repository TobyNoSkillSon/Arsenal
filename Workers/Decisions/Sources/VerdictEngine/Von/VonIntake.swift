// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import ModelPrecision

/// Saved component inputs are canonical BF16 during source-format qualification, then FP16 for quants.
/// Floating inputs cross only the selected half cast; integer IDs, masks and marker layouts must match exactly.
final class VonIntakeSession {
    var inputs: [String: MLXArray]
    let capture: Bool
    var valid = true
    var used = Set<String>()
    var outputs: [String: MLXArray] = [:]
    let recorder = ComponentCapture()
    init(inputs: [String: MLXArray], capture: Bool) { self.inputs = inputs; self.capture = capture }
    func input(_ key: String, _ value: MLXArray) -> MLXArray {
        used.insert(key)
        if capture { eval(value); inputs[key] = value }
        guard let saved = inputs[key], saved.shape == value.shape else { valid = false; return value }
        if [.float16, .bfloat16].contains(value.dtype) { return saved.asType(value.dtype) }
        if saved.dtype != value.dtype || saved.asData(access: .copy).data != value.asData(access: .copy).data { valid = false }
        return saved
    }
    func output(_ key: String, _ value: MLXArray) { eval(value); outputs[key] = value }
}

extension VonNetwork {
    static func packEmbedding(file: URL, archive: TensorArchive, key: String, bits: Int) throws -> QuantizedEmbedding {
        guard let entry = archive.entries[key], entry.shape.count == 2, entry.shape[1] % 64 == 0 else { throw VonError.invalid("Invalid Von embedding") }
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
        let rowBytes = entry.bytes / entry.shape[0]
        var weights: [MLXArray] = [], scales: [MLXArray] = [], biases: [MLXArray] = []
        for start in stride(from: 0, to: entry.shape[0], by: 4096) {
            try autoreleasepool {
                let rows = Swift.min(4096, entry.shape[0] - start)
                try handle.seek(toOffset: entry.offset + UInt64(start * rowBytes))
                guard let data = try handle.read(upToCount: rows * rowBytes), data.count == rows * rowBytes else { throw VonError.invalid("Truncated embedding") }
                let dense = MLXArray(data, [rows, entry.shape[1]], dtype: entry.dtype)
                let (w, s, b) = MLX.quantized(dense, groupSize: 64, bits: bits)
                guard let b else { throw VonError.invalid("Missing affine bias") }
                eval(w, s, b); weights.append(w); scales.append(s); biases.append(b)
            }
        }
        let result = QuantizedEmbedding(weight: concatenated(weights), scales: concatenated(scales), biases: concatenated(biases), groupSize: 64, bits: bits)
        eval(result.parameters()); return result
    }
}
