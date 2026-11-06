// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
import ModelPrecision
@testable import IntegrationKit

@Suite(.serialized) struct NativeStoredTierTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }
    final class Block: Module {
        @ModuleInfo var projection: Linear
        override init() { _projection.wrappedValue = Linear(64, 64); super.init() }
    }
    final class Network: Module {
        @ModuleInfo var layers: [Block]
        @ModuleInfo var sequence: Sequential
        @ModuleInfo var table: Embedding
        override init() {
            _layers.wrappedValue = [Block(), Block()]
            _sequence.wrappedValue = Sequential(layers: SiLU(), Linear(64, 64), Tanh(), Linear(64, 4))
            _table.wrappedValue = Embedding(embeddingCount: 8, dimensions: 64)
            super.init()
        }
    }
    @Test(.heavy, arguments: ["int8", "int4"]) func getConversionThenStoredLoadAndOperations(precision: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), installed = root.appendingPathComponent(precision)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let reference = Network(), writer = try TensorWriter(source.appendingPathComponent("model.safetensors"))
        for (name, value) in reference.parameters().flattened() { try writer.append(name, value.asType(.float16)) }
        try writer.finish(); try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
        _ = try NativeConversion(recipe: "fixture", eligible: { _, _ in true }).run(
            ["recipe": "fixture", "src_dir": .string(source.path), "dst_dir": .string(installed.path), "dtype": .string(precision), "group_size": 64],
            cancellation: NativeCancellation())
        let model = Network()
        let coverage = try NativePrecisionTier(precision).load(
            model, archives: [installed.appendingPathComponent("model.safetensors")], map: { ($0, $1) }, eligible: { _, _ in true })
        #expect(coverage.quantizedTensors == 5)
        #expect(model.layers.count == 2 && model.sequence.layers.count == 4)
        #expect(model.layers.allSatisfy { $0.projection is QuantizedLinear })
        #expect(model.sequence.layers[0] is SiLU && model.sequence.layers[2] is Tanh)
        #expect(model.sequence.layers[1] is QuantizedLinear && model.sequence.layers[3] is QuantizedLinear)
        #expect(model.table is QuantizedEmbedding)
        let input = MLXArray.ones([2, 64], dtype: .float16)
        for block in model.layers { let result = block.projection(input); eval(result); #expect(result.shape == [2, 64] && result.dtype == .float16) }
        let output = model.sequence(input), embedded = model.table(MLXArray([Int32(0), 1])); eval(output, embedded)
        #expect(output.shape == [2, 4] && embedded.shape == [2, 64])
        #expect(output.dtype == .float16 && embedded.dtype == .float16)
        #expect(throws: PrecisionError.self) {
            _ = try NativePrecisionTier(precision == "int8" ? "int4" : "int8").load(
                Network(), archives: [installed.appendingPathComponent("model.safetensors")], map: { ($0, $1) }, eligible: { _, _ in true })
        }
    }
    @Test(.heavy, arguments: ["layers.1.projection.weight", "table.weight", "layers.1.projection.bias"])
    func mappedShapesAreCheckedBeforeQuantization(parameter: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("bad.safetensors"), model = Network(), writer = try TensorWriter(file)
        for (key, value) in model.parameters().flattened() {
            var shape = value.shape; if key == parameter { shape[0] += 1 }
            try writer.append(key, MLXArray.zeros(shape, dtype: .float16))
        }
        try writer.finish()
        #expect(throws: PrecisionError.self) { _ = try NativePrecisionTier("int4").load(model, archives: [file], map: { ($0, $1) }, eligible: { _, _ in true }) }
    }
}
