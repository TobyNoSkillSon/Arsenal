// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import MLXRandom
import ModelPrecision

/// The port's named tensors are installed once in native Linear modules or half spatial/norm slots.
/// Graph objects are constructed after installation, so affine modules are used directly without dense reconstruction.
public final class TrellisWeights: Module {
    @ModuleInfo var layers: [Linear]
    public let tensors: [MLXArray]
    public let shapes: [String: [Int]]
    private let matrixIndex: [String: Int]
    private let tensorIndex: [String: Int]
    public private(set) var used = Set<String>()

    public init(shapes: [String: [Int]]) {
        self.shapes = shapes
        let matrices = shapes.keys.filter { $0.hasSuffix(".weight") && shapes[$0]!.count == 2 }.sorted()
        let remaining = shapes.keys.filter { !matrices.contains($0) }.sorted()
        self.matrixIndex = Dictionary(uniqueKeysWithValues: matrices.enumerated().map { ($0.element, $0.offset) })
        self.tensorIndex = Dictionary(uniqueKeysWithValues: remaining.enumerated().map { ($0.element, $0.offset) })
        self._layers.wrappedValue = matrices.map {
            let shape = shapes[$0]!; return Linear(shape[1], shape[0], bias: false)
        }
        self.tensors = remaining.map { MLXArray.zeros(shapes[$0]!, dtype: .float16) }
        super.init()
    }
    public static func eligible(_ name: String, _ shape: [Int]) -> Bool {
        name.hasSuffix(".weight") && shape.count == 2 && shape[1] % 64 == 0
    }
    public subscript(_ key: String) -> MLXArray? {
        guard let index = tensorIndex[key] else { return nil }
        used.insert(key); return tensors[index]
    }
    public func matrix(_ key: String) -> Linear {
        precondition(matrixIndex[key] != nil, "Missing TRELLIS matrix: \(key)")
        used.insert(key); return layers[matrixIndex[key]!]
    }
    public func internalName(_ key: String) -> String? {
        if let index = matrixIndex[key] { return "layers.\(index).weight" }
        if let index = tensorIndex[key] { return "tensors.\(index)" }
        for suffix in [".scales", ".biases"] where key.hasSuffix(suffix) {
            let weight = String(key.dropLast(suffix.count)) + ".weight"
            if let index = matrixIndex[weight] { return "layers.\(index)" + suffix }
        }
        return nil
    }
    public var checkpointParameters: [String: MLXArray] {
        let current = Dictionary(uniqueKeysWithValues: parameters().flattened())
        var result: [String: MLXArray] = [:]
        for key in shapes.keys {
            if let name = internalName(key), let value = current[name] { result[key] = value }
            if matrixIndex[key] != nil {
                for suffix in [".scales", ".biases"] {
                    let name = String(key.dropLast(7)) + suffix
                    if let internalName = internalName(name), let value = current[internalName] { result[name] = value }
                }
            }
        }
        return result
    }
    public func load(archives: [URL], precision: String, cancellation: NativeCancellation) throws {
        try NativeParameters.initializeRandom(self)
        _ = try NativePrecisionTier(precision).load(
            self, archives: archives,
            map: { key, value in
                try cancellation.check()
                guard let name = self.internalName(key) else { throw PrecisionError("Unknown TRELLIS parameter: " + key) }
                // Get stores every component in half precision (the FP32 assets are converted into the base); Load casts nothing.
                return (name, value)
            }, eligible: Self.eligible)
    }
    public func initializeLab(precision: String, seed: UInt64) throws {
        MLXRandom.seed(seed)
        var updates: [String: MLXArray] = [:]
        for key in shapes.keys.sorted() {
            let shape = shapes[key]!
            let name = internalName(key)!
            let isScale = key.hasSuffix("gamma") || key.hasSuffix("lambda1") || (key.contains("norm") && key.hasSuffix(".weight"))
            let value: MLXArray
            if isScale {
                value = MLXArray.ones(shape, dtype: .float16)
            } else if key.hasSuffix(".bias") {
                value = MLXArray.zeros(shape, dtype: .float16)
            } else {
                value = MLXRandom.normal(shape) * 0.005
            }
            updates[name] = value.asType(.float16)
        }
        // Healthy random initial topology; these are learned bias initializers, not output substitutions.
        for key in shapes.keys where key.hasSuffix("to_subdiv.bias") || key == "out_layer.2.bias" {
            updates[internalName(key)!] = MLXArray.ones(shapes[key]!, dtype: .float16) * 0.2
        }
        if shapes["output_layer.bias"] == [7] {
            updates[internalName("output_layer.bias")!] = MLXArray([Float(0), 0, 0, 0.2, 0.2, 0.2, 0.2]).asType(.float16)
        }
        try update(parameters: ModuleParameters.unflattened(updates), verify: [.allModelKeysSet, .noUnusedKeys, .shapeMismatch])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("trellis-random-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("random.safetensors")
        try MLX.save(arrays: checkpointParameters, url: file)
        try load(archives: [file], precision: precision, cancellation: NativeCancellation())
    }
}
