// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import ModelPrecision
import Testing
@testable import MLXBiRefNet

struct PrecisionTests {
    init() {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
        Memory.memoryLimit = 128 * 1024 * 1024
        Memory.cacheLimit = 8 * 1024 * 1024
    }
    @Test func recipeCoversEveryLargeCompatibleMatrixAndExplainsSkips() {
        #expect(BiRefNetPrecision.skipReason(key: "bb.layers.0.qkv.weight", shape: [576, 192], floating: true) == nil)
        #expect(BiRefNetPrecision.skipReason(key: "projection.weight", shape: [256, 256], floating: true) == nil)
        #expect(BiRefNetPrecision.skipReason(key: "projection.weight", shape: [1000, 63], floating: true) != nil)
        #expect(BiRefNetPrecision.skipReason(key: "projection.weight", shape: [1100, 65], floating: true) == "input width incompatible with group 64")
        #expect(BiRefNetPrecision.skipReason(key: "conv.weight", shape: [64, 64, 3, 3], floating: true)!.contains("OIHW"))
        #expect(BiRefNetPrecision.skipReason(key: "norm.weight", shape: [192], floating: true) != nil)
        #expect(BiRefNetPrecision.skipReason(key: "lateral.conv.weight", shape: [1536, 1536, 1, 1], floating: true) == nil)
        #expect(BiRefNet.intakeKeys.count == 20)
    }
    @Test(.heavy) func packedLoaderReachesSelectedBitWidthGuard() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: file) }
        // Valid archive inventory and buffer counts; this is an int4 pack requested as int8.
        var values = Dictionary(uniqueKeysWithValues: (0..<686).map { ("unused\($0)", MLXArray([Float16(1)])) })
        for i in 0..<67 { values["buffer\(i).num_batches_tracked"] = MLXArray([Int64(0)]) }
        values["projection.weight"] = MLXArray.zeros([1, 8], dtype: .uint32)
        values["projection.scales"] = MLXArray.ones([1, 1], dtype: .float16)
        values["projection.biases"] = MLXArray.zeros([1, 1], dtype: .float16)
        #expect(values.count == 756)
        try MLX.save(arrays: values, url: file)
        #expect(throws: BiRefNet.LoadError.precision) { try BiRefNet().loadWeights(url: file, bits: 8) }
    }

    @Test(.heavy, arguments: [8, 4]) func storedOI11AdapterPreservesNHWCChannelsAndBias(bits: Int) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("source"), target = directory.appendingPathComponent("packed")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        // Neither channels nor spatial positions are symmetric; a transpose or lost bias changes values.
        let original = MLXArray((0..<65_536).map { Float16(0.11 * sin(Double($0) * 0.037) + Double($0 / 256) * 0.0003) }, [256, 256, 1, 1])
        let bias = MLXArray((0..<256).map { Float16(Double($0) * 0.001 - 0.12) })
        var values = Dictionary(uniqueKeysWithValues: (0..<752).map { ("unused\($0)", MLXArray([Float16(0)])) })
        values["conv.weight"] = original; values["conv.bias"] = bias
        try MLX.save(arrays: values, url: source.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
        try BiRefNetPrecision.convert(source: source, destination: target, bits: bits)
        let archive = try TensorArchive(target.appendingPathComponent("model.safetensors"))
        let stored = try Dictionary(uniqueKeysWithValues: ["conv.weight", "conv.scales", "conv.biases", "conv.bias"].map { ($0, try archive.array($0)) })
        #expect(stored["conv.weight"]!.shape == [256, 256 * bits / 32])
        // Independent OI11 interpretation catches conversion transposes as well as load-time ones.
        let (expectedPack, expectedScales, expectedBiases) = MLX.quantized(
            original.reshaped([256, 256]), groupSize: 64, bits: bits, mode: .affine, stream: .cpu)
        #expect(stored["conv.weight"]!.asArray(UInt32.self) == expectedPack.asArray(UInt32.self))
        #expect(stored["conv.scales"]!.asArray(Float16.self) == expectedScales.asArray(Float16.self))
        #expect(stored["conv.biases"]!.asArray(Float16.self) == expectedBiases!.asArray(Float16.self))
        let dense = MLX.dequantized(expectedPack, scales: expectedScales, biases: expectedBiases, groupSize: 64, bits: bits)
        let reference = Conv2d(inputChannels: 256, outputChannels: 256, kernelSize: 1, bias: true)
        try reference.update(
            parameters: ModuleParameters.unflattened([
                ("weight", dense.reshaped([256, 1, 1, 256])), ("bias", bias)
            ]), verify: [.noUnusedKeys, .shapeMismatch])
        let model = PointwiseFixture()
        try BiRefNetPrecision.prepare(model, weights: stored, bits: bits)
        #expect(model.conv is QuantizedLinear)
        let input = MLXArray((0..<(2 * 3 * 256)).map { Float16(0.4 * cos(Double($0) * 0.023) - 0.1) }, [1, 2, 3, 256])
        let actual = model.conv(input), expected = reference(input)
        #expect(actual.shape == [1, 2, 3, 256] && actual.dtype == .float16)
        let a = actual.asArray(Float16.self).map(Double.init), b = expected.asArray(Float16.self).map(Double.init)
        #expect(a.allSatisfy(\.isFinite) && b.allSatisfy(\.isFinite))
        let squaredError = zip(a, b).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) }
        let squaredReference = b.reduce(0.0) { $0 + $1 * $1 }
        #expect(sqrt(squaredError / squaredReference) < 0.002)
        #expect(zip(a, b).map { abs($0 - $1) }.max()! < 0.004)
    }

    @Test(.heavy) func storedInventoryIncludesPackedMatricesButExcludesComputedIndices() throws {
        let model = BiRefNet()
        #expect(model.storedWeightNames().count == 687)
        let path = "bb.layers.0.blocks.0.attn.qkv"
        let packed = QuantizedLinear(
            weight: MLXArray.zeros([576, 48], dtype: .uint32), bias: MLXArray.zeros([576], dtype: .float16),
            scales: MLXArray.zeros([576, 3], dtype: .float16), biases: MLXArray.zeros([576, 3], dtype: .float16), groupSize: 64, bits: 8)
        try model.update(modules: ModuleChildren.unflattened([(path, packed as Module)]), verify: .none)
        #expect(model.storedWeightNames().count == 689)
        #expect(model.storedWeightNames().contains(path + ".weight"))
        #expect(model.storedWeightNames().contains(path + ".scales"))
        #expect(!model.storedWeightNames().contains { $0.hasSuffix("relativePositionIndex") })
    }

}

private final class PointwiseFixture: Module {
    @ModuleInfo(key: "conv") var conv: any UnaryLayer
    override init() {
        self._conv.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256, kernelSize: 1, bias: true)
        super.init()
    }
}
