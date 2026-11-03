import Foundation
import Testing
@testable import ModelPrecision

struct ArchiveTests {
    @Test func truncatedHeadersAreRejected() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        for data in [Data(), Data(repeating: 255, count: 8), Data(repeating: 0, count: 12)] {
            try data.write(to: file)
            #expect(throws: (any Error).self) { try TensorArchive(file) }
        }
    }
    @Test func descriptorSizeMustMatch() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let header = Data(#"{"x":{"dtype":"F16","shape":[2],"data_offsets":[0,2]}}"#.utf8)
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(header); data.append(Data([0, 0]))
        try data.write(to: file)
        #expect(throws: (any Error).self) { try TensorArchive(file) }
    }
    @Test func headersCanBeInspectedWithoutArrays() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let header = Data(#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#.utf8)
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(header); data.append(Data(repeating: 0, count: 4))
        try data.write(to: file)
        let archive = try TensorArchive(file)
        #expect(archive.entries.count == 1)
        #expect(archive.entries["x"]?.bytes == 4)
        #expect(archive.entries["x"]?.shape == [2])
    }
}

import MLX
import MLXNN

@Suite(.serialized) struct TensorRoundTripTests {
    init() {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
    }
    @Test(arguments: [4, 8]) func streamedPackedWeightsLoadDirectly(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent("intake-tensor-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let source = MLXArray((0..<256).map { Float16(sin(Double($0) * 0.13)) }, [2, 128])
            let file = root.appendingPathComponent("packed.safetensors")
            let coverage = try StreamedPrecision.convert(
                keys: ["weight"], read: { _ in source }, to: file,
                half: .float16, bits: nil, eligible: { _, _ in false })
            #expect(coverage.tensors == 1 && coverage.sourceBytes == 512 && coverage.outputBytes == 512)
            let read = try TensorArchive(file).arrays()
            #expect(read["weight"]?.asArray(Float16.self) == source.asArray(Float16.self))
            let packedFile = root.appendingPathComponent("quant.safetensors")
            _ = try StreamedPrecision.convert(
                keys: ["layer.weight"], read: { _ in source }, to: packedFile,
                half: .float16, bits: bits, eligible: { _, _ in true })
            let packed = try TensorArchive(packedFile).arrays()
            #expect(packed.count == 3 && packed["layer.weight"]?.dtype == .uint32)
            #expect(packed["layer.scales"]?.dtype == .float16)
            let reconstructed = MLX.dequantized(packed["layer.weight"]!, scales: packed["layer.scales"]!, biases: packed["layer.biases"]!, groupSize: 64, bits: bits)
            let error = sqrt(mean(square((reconstructed - source).asType(.float32)))).item(Float.self)
            #expect(error < (bits == 4 ? 0.05 : 0.004))
            #expect(throws: (any Error).self) {
                try StreamedPrecision.convert(
                    keys: packed.keys.sorted(), read: { packed[$0]! }, to: root.appendingPathComponent("bad.safetensors"),
                    half: .float16, bits: bits, eligible: { _, _ in true })
            }
        }
    }
    @Test func halfStoredConditioningKeepsFloat32Calculation() {
        Device.withDefaultDevice(.cpu) {
            let x = MLXArray([Float(0.1), 0.25, -0.5, 0.75], [1, 4])
            let w = MLXArray([Float16(0.7), -0.25, 0.5, 0.125, -0.5, 0.1, 0.2, 0.3], [2, 4])
            let bias = MLXArray([Float16(0.125), -0.25])
            let a = MLX.matmul(x, w.transposed()) + bias
            let b = MLX.matmul(x, w.asType(.float32).transposed()) + bias.asType(.float32)
            #expect(a.dtype == .float32)
            #expect(a.asArray(Float.self) == b.asArray(Float.self))
        }
    }
    @Test func halfOverflowRefusesOutput() throws {
        try Device.withDefaultDevice(.cpu) {
            let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
            defer { try? FileManager.default.removeItem(at: file) }
            #expect(throws: (any Error).self) {
                try StreamedPrecision.convert(
                    keys: ["weight"], read: { _ in MLXArray([Float(70_000)]) }, to: file,
                    half: .float16, eligible: { _, _ in false })
            }
            #expect(!FileManager.default.fileExists(atPath: file.path))
        }
    }
}

private final class PackedFixture: Module {
    @ModuleInfo var layer: Linear
    @ModuleInfo var embed: Embedding
    override init() {
        _layer.wrappedValue = Linear(128, 2, bias: false)
        _embed.wrappedValue = Embedding(embeddingCount: 2, dimensions: 128)
    }
}

extension TensorRoundTripTests {
    @Test(arguments: [4, 8]) func prepareInstallsLinearAndEmbeddingAndExecutes(bits: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let source = MLXArray((0..<256).map { Float16(sin(Double($0) * 0.13)) }, [2, 128])
            let file = root.appendingPathComponent("packed.safetensors")
            _ = try StreamedPrecision.convert(keys: ["layer.weight", "embed.weight"], read: { _ in source }, to: file, half: .float16, bits: bits, eligible: { _, _ in true })
            let weights = try TensorArchive(file).arrays()
            let model = PackedFixture()
            try StreamedPrecision.prepare(model, weights: weights)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            #expect(model.layer is QuantizedLinear && model.embed is QuantizedEmbedding)
            let dense = MLX.dequantized(weights["layer.weight"]!, scales: weights["layer.scales"]!, biases: weights["layer.biases"]!, groupSize: 64, bits: bits)
            let x = MLXArray.ones([1, 128], dtype: .float16)
            #expect(abs(model.layer(x) - matmul(x, dense.transposed())).max().item(Float.self) < 0.02)
            #expect(abs(model.embed(MLXArray([Int32(0), 1])) - dense).max().item(Float.self) < 0.002)
            var malformed = weights; malformed.removeValue(forKey: "layer.biases")
            #expect(throws: (any Error).self) { try StreamedPrecision.prepare(PackedFixture(), weights: malformed) }
            malformed = weights; malformed.removeValue(forKey: "layer.scales")
            #expect(throws: (any Error).self) { try StreamedPrecision.prepare(PackedFixture(), weights: malformed) }
            malformed = weights; malformed["layer.scales"] = MLXArray.ones([2, 3], dtype: .float16)
            #expect(throws: (any Error).self) { try StreamedPrecision.prepare(PackedFixture(), weights: malformed) }
            malformed = weights; malformed["layer.scales"] = weights["layer.scales"]!.asType(.float32)
            #expect(throws: (any Error).self) { try StreamedPrecision.prepare(PackedFixture(), weights: malformed) }
        }
    }
    @Test func writerShiftsPayloadAcrossChunkBoundaryWithoutSecondPayload() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let file = dir.appendingPathComponent("large.safetensors")
            let source = MLXArray(Array(repeating: Float16(0.125), count: (4 << 20) / 2 + 17))
            _ = try StreamedPrecision.convert(keys: ["x"], read: { _ in source }, to: file, half: .float16, eligible: { _, _ in false })
            let read = try TensorArchive(file).array("x")
            #expect(read.asArray(Float16.self) == source.asArray(Float16.self))
            #expect(try FileManager.default.contentsOfDirectory(atPath: dir.path) == ["large.safetensors"])
        }
    }
}
