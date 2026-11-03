// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import ModelPrecision

public enum BiRefNetPrecision {
    public static let recipe = "intake-v1:birefnet"
    public static func skipReason(key: String, shape: [Int], floating: Bool) -> String? {
        guard floating else { return "integer position/batch buffer" }
        guard key.hasSuffix(".weight") else { return "bias, running statistics or position bias" }
        let matrix = shape.count == 4 && Array(shape.suffix(2)) == [1, 1] ? Array(shape.prefix(2)) : shape
        guard matrix.count == 2 else {
            return shape.count == 4
                ? "OIHW spatial/deformable convolution: not a group-64 Linear layout; stays FP16"
                : "norm/vector stays FP16"
        }
        guard matrix.reduce(1, *) >= 65_536 else { return "matrix below 65,536 parameters" }
        guard matrix[1] % 64 == 0 else { return "input width incompatible with group 64" }
        return nil
    }
    // PyTorch OI11 -> [O,I]; NHWC stride-one 1x1 convolution is a per-position Linear.
    static func conversionLayout(_ value: MLXArray, key: String) -> MLXArray {
        if value.ndim == 4, skipReason(key: key, shape: value.shape, floating: value.dtype.isFloatingPoint) == nil {
            return value.reshaped([value.dim(0), value.dim(1)])
        }
        return value
    }
    /// Shared with the loader so numerical regressions exercise the exact stored-pack adapter.
    static func prepare(_ model: Module, weights: [String: MLXArray], bits: Int) throws {
        for (key, value) in weights where key.hasSuffix(".scales") {
            let prefix = String(key.dropLast(7))
            guard let packed = weights[prefix + ".weight"], packed.ndim == 2, value.ndim == 2,
                packed.dim(1) * 32 == value.dim(1) * 64 * bits else { throw BiRefNet.LoadError.precision }
        }
        var pointwise: [(String, Module)] = []
        for (path, layer) in model.leafModules().flattened() {
            guard let conv = layer as? Conv2d, weights[path + ".scales"] != nil else { continue }
            guard Array(conv.weight.shape[1...2]) == [1, 1], conv.groups == 1,
                conv.stride == (1, 1), conv.padding == (0, 0), conv.dilation == (1, 1)
            else { throw BiRefNet.LoadError.inventory }
            // A lazy shape-only placeholder is replaced by the stored pack before any evaluation.
            pointwise.append((path, Linear(weight: MLXArray.zeros([conv.weight.dim(0), conv.weight.dim(3)], dtype: .float16), bias: weights[path + ".bias"])))
        }
        if !pointwise.isEmpty { try model.update(modules: ModuleChildren.unflattened(pointwise), verify: .none) }
        try StreamedPrecision.prepare(model, weights: weights)
    }
    public static func convert(source: URL, destination: URL, bits: Int) throws {
        guard [4, 8].contains(bits) else { throw BiRefNet.LoadError.precision }
        let archive = try TensorArchive(source.appendingPathComponent("model.safetensors"))
        guard archive.entries.count == 754 else { throw BiRefNet.LoadError.inventory }
        let coverage = try StreamedPrecision.convert(
            keys: archive.entries.keys.sorted(), read: { key in
                try conversionLayout(archive.array(key), key: key)
            },
            to: destination.appendingPathComponent("model.safetensors"), half: .float16, bits: bits
        ) { key, shape in skipReason(key: key, shape: shape, floating: true) == nil }
        var kept: [String: String] = [:], packed: [String] = []
        var total = 0, quantized = 0
        for (key, entry) in archive.entries {
            let count = entry.shape.reduce(1, *)
            if entry.dtype.isFloatingPoint { total += count }
            if let reason = skipReason(key: key, shape: entry.shape, floating: entry.dtype.isFloatingPoint) { kept[key] = reason }
            else { packed.append(key); quantized += count }
        }
        try JSONEncoder().encode(coverage).write(to: destination.appendingPathComponent("coverage.json"))
        try JSONSerialization.data(withJSONObject: [
            "recipe": recipe, "bits": bits, "group_size": 64, "source_half": "fp16",
            "total_parameters": total, "packed_parameters": quantized,
            "half_parameters": total - quantized, "fallback_parameters": 0,
            "packed": packed.sorted(), "kept": kept, "searched": false
        ], options: [.sortedKeys]).write(to: destination.appendingPathComponent("component-coverage.json"))
        try FileManager.default.copyItem(at: source.appendingPathComponent("config.json"), to: destination.appendingPathComponent("config.json"))
    }
}

/// Only set during sequential intake requests. Normal serving never observes or retains tensors.
enum BiRefNetTrace {
    static var recorder: ComponentCapture?
    static var context = "", sequence = 0
    static func observe(_ name: String, _ value: MLXArray) {
        guard let recorder else { return }
        recorder.observe(context + "/\(sequence)-" + name, value)
        sequence += 1
    }
}
