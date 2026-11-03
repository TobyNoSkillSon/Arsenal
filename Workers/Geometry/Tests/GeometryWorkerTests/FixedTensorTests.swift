import Foundation
import GeometryWorker
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import MLXDA3
import MLXMoGe
import Testing

@Suite(.serialized)
struct FixedTensorTests {
    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func spatialOperatorsMatchFixedReference() throws {
        NativeRuntime.prepare()
        // Independent scalar reference: scatter a value 2 through this 2x2 kernel, then add bias 1.
        let x = MLXArray([Float(2)], [1, 1, 1, 1]).asType(.float16)
        let kernel = MLXArray([Float(1), 2, 3, 4], [1, 2, 2, 1]).asType(.float16)
        let transposed = MLXMoGe.convTranspose2d(x, weight: kernel, bias: MLXArray([Float(1)]).asType(.float16))
        eval(transposed)
        #expect(transposed.shape == [1, 2, 2, 1])
        #expect(transposed.dtype == .float16)
        #expect(transposed.asType(.float32).asArray(Float.self) == [3, 5, 7, 9])

        // A bicubic partition of unity preserves a constant at every border and retains activation precision.
        let image = MLXArray.ones([1, 3, 3, 2], dtype: .float16)
        for resized in [mogeBicubicResample(image, outH: 5, outW: 7), pytorchBicubicResample(image, outH: 5, outW: 7)] {
            eval(resized)
            #expect(resized.dtype == .float16)
            #expect(resized.shape == [1, 5, 7, 2])
            #expect(resized.asType(.float32).asArray(Float.self).allSatisfy { abs($0 - 1) < 0.002 })
        }
    }

    private final class Projection: Module {
        @ModuleInfo var projection: Linear
        override init() { _projection.wrappedValue = Linear(64, 2); super.init() }
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func storedPacksPreserveNontrivialBias() throws {
        NativeRuntime.prepare()
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("geometry-tier-test-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let source = temporary.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let rows = [Float](repeating: 0.125, count: 64) + [Float](repeating: 0.25, count: 64)
        try MLX.save(
            arrays: [
                "projection.weight": MLXArray(rows, [2, 64]).asType(.float16),
                "projection.bias": MLXArray([Float(3), 7]).asType(.float16)
            ], url: source.appendingPathComponent("model.safetensors"))
        for (precision, bits) in [("int8", 8), ("int4", 4)] {
            let destination = temporary.appendingPathComponent(precision)
            let conversion = NativeConversion(recipe: "geometry-fixed-test", eligible: GeometryWeights.eligible)
            _ = try conversion.run(
                [
                    "src_dir": .string(source.path), "dst_dir": .string(destination.path), "dtype": .string(precision),
                    "group_size": 64, "recipe": "geometry-fixed-test"
                ], cancellation: NativeCancellation())
            let model = Projection()
            try NativeParameters.initializeRandom(model)
            let coverage = try NativePrecisionTier(precision).load(
                model, archives: [destination.appendingPathComponent("model.safetensors")],
                map: { ($0, $1) }, eligible: GeometryWeights.eligible)
            let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            #expect(coverage.quantizedTensors == 1)
            #expect(parameters["projection.weight"]?.dtype == .uint32)
            #expect(parameters["projection.weight"]?.shape == [2, 64 * bits / 32])
            let output = model.projection(MLXArray.ones([1, 64], dtype: .float16))
            eval(output)
            #expect(output.dtype == .float16)
            // Independent scalar result: 64*0.125+3 = 11; 64*0.25+7 = 23.
            #expect(output.asType(.float32).asArray(Float.self) == [11, 23])
        }
    }
}
