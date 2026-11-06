// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
import ModelPrecision
import ArsenalWire
@testable import IntegrationKit

@Suite(.serialized) struct NativeConversionContractTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }
    /// Get casts pinned files one at a time, so a recipe may see config.json alone; any weight file still requires tensors.
    @Test(arguments: [("config.json", true), ("weights.bin", false), ("model.pt", false)]) func tensorFreeOutputOnlyForWeightFreeSources(name: String, accepted: Bool) throws {
        let root = URL(fileURLWithPath: "/private/tmp/conversion-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: source.appendingPathComponent(name))
        let copy = NativeConversion(recipe: "copy") { context, cancellation in
            try FileManager.default.copyItem(at: context.source.appendingPathComponent(name), to: context.destination.appendingPathComponent(name))
            return try context.receipt(cancellation: cancellation)
        }
        let request: JSONObject = [
            "recipe": "copy", "src_dir": .string(source.path), "dst_dir": .string(root.appendingPathComponent("output").path), "dtype": "fp16", "group_size": 64
        ]
        if accepted {
            #expect(try copy.run(request, cancellation: NativeCancellation())["files"]?.array?.count == 1)
        } else {
            #expect(throws: PrecisionError.self) { try copy.run(request, cancellation: NativeCancellation()) }
        }
    }
    @Test(.heavy, arguments: ["fp16", "int8", "int4"]) func boundedWeightNormLayoutAndSplitComponents(precision: String) throws {
        let root = URL(fileURLWithPath: "/private/tmp/conversion-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), asset = root.appendingPathComponent("asset"), output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: source.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: asset, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer/config.json"))
        let writer = try TensorWriter(source.appendingPathComponent("main.safetensors"))
        try writer.append("fused", MLXArray.ones([64, 4], dtype: .float32)); try writer.finish()
        let codec = try TensorWriter(asset.appendingPathComponent("vae.safetensors"))
        try codec.append("weight_g", MLXArray([Float(2), 4], [2, 1]))
        try codec.append("weight_v", MLXArray.ones([2, 4], dtype: .float32)); try codec.finish()
        let converter = NativeConversion(recipe: "transformed") { context, cancellation in
            let archive = try TensorArchive(context.source.appendingPathComponent("main.safetensors"))
            // A bounded fused pair; no dictionary of the complete checkpoint is loaded.
            let fused = try archive.array("fused").transposed()
            let values = ["q.weight": fused[0..<2], "k.weight": fused[2..<4]]
            _ = try StreamedPrecision.convert(
                keys: values.keys.sorted(), read: { values[$0]! }, to: context.destination.appendingPathComponent("lm/model.safetensors"), half: context.half, bits: context.bits,
                eligible: { _, _ in true })
            try cancellation.check()
            let vae = try TensorArchive(try #require(context.assets["vae"]).appendingPathComponent("vae.safetensors"))
            let v = try vae.array("weight_v"), g = try vae.array("weight_g")
            let folded = v * g / MLX.sqrt(MLX.sum(v * v, axis: 1, keepDims: true))
            _ = try StreamedPrecision.convert(
                keys: ["conv.weight"], read: { _ in folded }, to: context.destination.appendingPathComponent("vae/model.safetensors"), half: context.half, bits: context.bits,
                eligible: { _, _ in false })
            try FileManager.default.createDirectory(at: context.destination.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
            try FileManager.default.copyItem(
                at: context.source.appendingPathComponent("tokenizer/config.json"), to: context.destination.appendingPathComponent("tokenizer/config.json"))
            return try context.receipt(cancellation: cancellation)
        }
        let request: JSONObject = [
            "recipe": "transformed", "src_dir": .string(source.path), "assets": ["vae": .string(asset.path)], "dst_dir": .string(output.path), "dtype": .string(precision),
            "group_size": 64
        ]
        let receipt = try converter.run(request, cancellation: NativeCancellation())
        #expect(receipt["files"]?.array?.count == 3)
        let converted = try TensorArchive(output.appendingPathComponent("lm/model.safetensors"))
        #expect(converted.entries["q.weight"]?.shape == [2, precision == "fp16" ? 64 : precision == "int8" ? 16 : 8])
        #expect(converted.entries["q.weight"]?.dtype == (precision == "fp16" ? .float16 : .uint32))
        let folded = try TensorArchive(output.appendingPathComponent("vae/model.safetensors")).array("conv.weight")
        #expect(folded.dtype == .float16 && folded.asArray(Float.self) == [1, 1, 1, 1, 2, 2, 2, 2])
        for file in receipt["files"]!.array! {
            let target = output.appendingPathComponent(file["path"]!.string!)
            #expect(file["bytes"]?.number == Double(try NativeFiles.bytes(target)))
            #expect(file["sha256"]?.string == (try NativeFiles.sha256(target)))
        }
        let canceled = NativeCancellation(); canceled.cancel()
        var canceledRequest = request; canceledRequest["dst_dir"] = .string(root.appendingPathComponent("canceled").path)
        #expect(throws: (any Error).self) { _ = try converter.run(canceledRequest, cancellation: canceled) }
        try FileManager.default.createSymbolicLink(at: source.appendingPathComponent("link"), withDestinationURL: asset)
        #expect(throws: (any Error).self) { _ = try converter.run(canceledRequest, cancellation: NativeCancellation()) }
    }
    @Test(.heavy) func normalizationPreservesHalfFilesAndStillRejectsNonFiniteWeights() throws {
        let root = URL(fileURLWithPath: "/private/tmp/half-normalize-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for (index, value) in [Float16(1), .infinity, .nan].enumerated() {
            let source = root.appendingPathComponent("source-" + String(index)), output = root.appendingPathComponent("output-" + String(index))
            let original = source.appendingPathComponent("weights.safetensors"), writer = try TensorWriter(original)
            try writer.append("weight", MLXArray([value])); try writer.finish()
            let request: JSONObject = [
                "src_dir": .string(source.path), "dst_dir": .string(output.path), "recipe": .string(NativeConversion.normalizationRecipe), "dtype": "fp16", "group_size": 64
            ]
            if index == 0 {
                _ = try NativeConversion.normalize(request, cancellation: NativeCancellation())
                #expect(try Data(contentsOf: original) == Data(contentsOf: output.appendingPathComponent("weights.safetensors")))
            } else {
                #expect(throws: PrecisionError.self) { _ = try NativeConversion.normalize(request, cancellation: NativeCancellation()) }
            }
        }
    }
    final class Spatial: Module {
        @ModuleInfo var conv: Conv2d
        override init() { _conv.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 2, kernelSize: 1); super.init() }
    }
    @Test(.heavy, arguments: ["fp16", "int8", "int4"]) func zeroMatrixGetLoadAndConvolutionRemainHalf(precision: String) throws {
        let root = URL(fileURLWithPath: "/private/tmp/spatial-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), output = root.appendingPathComponent("output")
        let writer = try TensorWriter(source.appendingPathComponent("model.safetensors"))
        try writer.append("conv.weight", MLXArray.ones([2, 1, 1, 3], dtype: .float32) * 0.25)
        try writer.append("conv.bias", MLXArray([Float(0.5), 1])); try writer.finish()
        let converter = NativeConversion(recipe: "spatial", eligible: { _, _ in false })
        _ = try converter.run(
            ["recipe": "spatial", "src_dir": .string(source.path), "dst_dir": .string(output.path), "dtype": .string(precision), "group_size": 64],
            cancellation: NativeCancellation())
        let archive = try TensorArchive(output.appendingPathComponent("model.safetensors"))
        #expect(archive.entries.count == 2 && archive.entries.values.allSatisfy { $0.dtype == .float16 })
        let graph = Spatial()
        let coverage = try NativePrecisionTier(precision).load(
            graph, archives: [output.appendingPathComponent("model.safetensors")], map: { ($0, $1) }, eligible: { _, _ in false })
        #expect(coverage.quantizedTensors == 0)
        let result = graph.conv(MLXArray.ones([1, 2, 2, 3], dtype: .float16)); eval(result)
        #expect(result.dtype == .float16 && result.shape == [1, 2, 2, 2])
        #expect(result.asArray(Float.self) == [1.25, 1.75, 1.25, 1.75, 1.25, 1.75, 1.25, 1.75])
    }
    @Test(.heavy) func architectureSelectorWinsAndCannotAmbiguouslyDispatch() throws {
        let registrations = [NativeRegistration(["integration_fixture"]) { Adapter() }]
        #expect(try NativeRegistration.resolve(checkpoint: [:], architecture: "integration_fixture", registrations: registrations).modelTypes == ["integration_fixture"])
        #expect(
            try NativeRegistration.resolve(checkpoint: ["model_type": "integration_fixture"], architecture: nil, registrations: registrations).modelTypes == ["integration_fixture"]
        )
        for checkpoint: JSONObject in [["model_name": "da3-original"], ["model_type": "vendor-name"]] {
            #expect(try NativeRegistration.resolve(checkpoint: checkpoint, architecture: "integration_fixture", registrations: registrations).modelTypes == ["integration_fixture"])
        }
        for (checkpoint, selector): (JSONObject, String?) in [([:], nil), ([:], "unknown")] {
            #expect(throws: (any Error).self) { _ = try NativeRegistration.resolve(checkpoint: checkpoint, architecture: selector, registrations: registrations) }
        }
        #expect(throws: (any Error).self) { _ = try NativeRegistration.resolve(checkpoint: [:], architecture: "integration_fixture", registrations: registrations + registrations) }
    }
    private final class Adapter: NativeModel {
        let architecture = "integration_fixture", operations: Set<String> = []
        let peakBytes = 0, parameters: [String: MLXArray] = [:]
        func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {}
        func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject { [:] }
        func unload() {}
    }
}
