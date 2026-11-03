// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// The checkpoint's tensors through MLX's own safetensors loader, which is lazy: a tensor is read from the file
/// (large sequential reads) when first evaluated, so a load reads only what it uses (the language model at load, the
/// picture tower on the first picture), as the Python runner's `mx.load` did.
public final class SafetensorsFile: @unchecked Sendable {
    private var arrays: [String: MLXArray]
    private let lock = NSLock()

    public init(url: URL) throws {
        arrays = try loadArrays(url: url)
    }

    public func contains(_ name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return arrays[name] != nil
    }

    public func matrix(_ name: String) throws -> D1Matrix {
        let weight = try take(name)
        let prefix = String(name.dropLast(7))
        return try D1Matrix(
            weight: weight,
            scales: contains(prefix + ".scales") ? take(prefix + ".scales") : nil,
            biases: contains(prefix + ".biases") ? take(prefix + ".biases") : nil)
    }

    /// Hands a tensor over and forgets it, so an unfused source is freed once its fused copy is built.
    public func take(_ name: String) throws -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        guard let a = arrays.removeValue(forKey: name) else { throw JSONError("model.safetensors lacks \(name)") }
        return a
    }
}
