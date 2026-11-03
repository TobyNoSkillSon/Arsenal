// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
import ModelPrecision
@testable import IntegrationKit

@Suite(.serialized) struct NativeTierLoadTests {
    init() {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
    }
    final class Tiny: Module {
        @ModuleInfo var projection: Linear
        override init() { _projection.wrappedValue = Linear(64, 4); super.init() }
    }
    @Test(.heavy, arguments: ["fp16", "int8", "int4"]) func streamedLoadInstallsEveryParameterAndPreservesBias(precision: String) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("model.safetensors")
        let weight = MLXArray((0..<256).map { Float(($0 * 13) % 31 - 15) / 32 }, [4, 64]).asType(.float16)
        let bias = MLXArray([Float(0.25), -0.125, 0.5, 0.75]).asType(.float16)
        let writer = try TensorWriter(file)
        try writer.append("projection.weight", weight); try writer.append("projection.bias", bias); try writer.finish()
        let model = Tiny()
        let tier = try NativePrecisionTier(precision)
        let coverage = try tier.load(model, archives: [file], map: { ($0, $1) }, eligible: { _, _ in true })
        #expect(coverage.tensors == 2)
        #expect(coverage.quantizedTensors == (precision == "fp16" ? 0 : 1))
        #expect(model.parameters().flattened().allSatisfy { !$0.1.dtype.isFloatingPoint || $0.1.dtype == .float16 })
        #expect(model.projection.bias?.asArray(Float.self) == bias.asArray(Float.self))
        let input = MLXArray((0..<128).map { Float($0 % 17 + 1) / 32 }, [2, 64]).asType(.float16)
        let output = model.projection(input); eval(output)
        #expect(output.shape == [2, 4] && output.dtype == .float16)
        let referenceWeight: MLXArray
        if let packed = model.projection as? QuantizedLinear {
            referenceWeight = dequantized(packed.weight, scales: packed.scales, biases: packed.biases, groupSize: 64, bits: packed.bits)
        } else {
            referenceWeight = weight
        }
        let expected = matmul(input, referenceWeight.transposed()) + bias
        #expect(abs(output - expected).max().item(Float.self) < 0.02)
    }
    @Test(.heavy) func ordinaryServingHasNoAllocatorBudget() {
        #expect(NativeRuntime.labBudget([:]) == nil)
        #expect(NativeRuntime.labBudget(["ARSENAL_LAB_MAX_BYTES": "1073741824"]) == nil)
        #expect(NativeRuntime.labBudget(["ARSENAL_NATIVE_LAB": "1", "ARSENAL_LAB_MAX_BYTES": "1073741824"]) == 1073741824)
        #expect(NativeRuntime.labBudget(["ARSENAL_NATIVE_LAB": "1", "ARSENAL_LAB_MAX_BYTES": "1"]) == nil)
    }
    /// Every family worker starts through `NativeRuntime.prepare` (NativeFamilyWorker.run); MLX's default cache limit
    /// is most of the Mac's memory, so start-up must leave it at the shared cap.
    @Test func startUpCapsTheMLXBufferCache() {
        NativeRuntime.prepare()
        #expect(Memory.cacheLimit > 0)
        #expect(Memory.cacheLimit <= NativeRuntime.cacheLimitBytes)
        #expect(NativeRuntime.cacheLimitBytes == 256 * 1024 * 1024)
    }
    @Test(.heavy) func int4CannotRetainAnUngatedHalfOverride() {
        #expect(throws: PrecisionError.self) { try NativePrecisionTier("int4", retainedHalf: ["projection"]) }
    }
}
