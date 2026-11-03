// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// Lab-only capture. Inputs are written once; every component consumes them independently.
public final class ComponentCapture {
    public var traces: [String: [String: Any]] = [:]
    public init() {}
    public func observe(_ key: String, _ value: MLXArray) {
        eval(value)
        let finite = all(isFinite(value)).item(Bool.self)
        let maximum = abs(value).max().item(Float.self)
        traces[key] = [
            "shape": value.shape, "count": value.size, "dtype": String(describing: value.dtype),
            "finite": finite, "max_abs": maximum.isFinite ? Double(maximum) : NSNull(),
            "clamp_count": value.dtype == .float16 ? sum((abs(value) .> 65504).asType(.int32)).item(Int.self) : 0
        ]
    }
    public static func inputs(at directory: URL, create: Bool, build: () throws -> [String: MLXArray]) throws -> [String: MLXArray] {
        let file = directory.appendingPathComponent("inputs.safetensors")
        if create {
            guard !FileManager.default.fileExists(atPath: directory.path) else { throw PrecisionError("Fixture directory already exists") }
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let inputs = try build(); eval(Array(inputs.values))
            try save(arrays: inputs, url: file)
            try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: file.path)
        }
        return try TensorArchive(file).arrays()
    }
    public static func validate(_ inputs: [String: MLXArray], shapes: [String: [Int]]) throws {
        guard Set(inputs.keys) == Set(shapes.keys), shapes.allSatisfy({ inputs[$0.key]?.shape == $0.value }) else {
            throw PrecisionError("Component fixture keys/counts/shapes differ")
        }
    }
    public func write(_ outputs: [String: MLXArray], to directory: URL) throws {
        guard !FileManager.default.fileExists(atPath: directory.path) else { throw PrecisionError("Capture output already exists") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (key, value) in outputs { observe(key + "/output", value) }
        try save(arrays: outputs, url: directory.appendingPathComponent("outputs.safetensors"))
        try JSONSerialization.data(withJSONObject: traces, options: [.sortedKeys]).write(to: directory.appendingPathComponent("activations.json"))
    }
}
