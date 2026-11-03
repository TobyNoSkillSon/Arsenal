// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXLMCommon
import ChatContract

/// Fixed teacher-forced continuation. A model reference is captured once; candidates consume the same token IDs.
public enum ChatMeasurementProbe {
    /// Extract native half bytes without casting the GPU logits to fp32.
    /// An existing model's fp32 output leaf may be read as-is; it never changes model precision.
    public static func hostLogits(_ logits: MLXArray) throws -> [Double] {
        switch logits.dtype {
        case .float16: return logits.asArray(Float16.self).map(Double.init)
        case .bfloat16:
            let data = logits.asData().data
            return data.withUnsafeBytes { bytes in
                (0..<logits.size).map { Double(Float(bitPattern: UInt32(bytes.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self)) << 16)) }
            }
        case .float32: return logits.asArray(Float.self).map(Double.init)
        default: throw ChatFailure("measurement", "Unsupported logits dtype")
        }
    }
    public static func capture(model: any LanguageModel, prefix: [Int], continuation: [Int]) throws -> [[Double]] {
        guard !prefix.isEmpty, !continuation.isEmpty, continuation.count <= 32 else { throw ChatFailure("measurement", "Expected a short nonempty fixed continuation") }
        let cache = model.newCache(parameters: nil)
        var output = model(LMInput.Text(tokens: MLXArray(prefix).expandedDimensions(axis: 0)), cache: cache, state: nil)
        var result: [[Double]] = []
        for token in continuation {
            eval(output.logits, cache.map { $0.state })
            result.append(try hostLogits(output.logits[0, -1]))
            output = model(LMInput.Text(tokens: MLXArray([token]).expandedDimensions(axis: 0)), cache: cache, state: output.state)
        }
        eval(output.logits, cache.map { $0.state })
        return result
    }
}
