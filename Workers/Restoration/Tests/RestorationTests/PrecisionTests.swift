// SPDX-License-Identifier: MIT
import Foundation
@testable import RestorationWorker
import Testing
import IntegrationKit
import MLX
import MLXNN
import SeedVR2Native

@Suite(.serialized)
struct PrecisionTests {
    private func graph() -> (SeedVR2Transformer, SeedVR2VAE) {
        var config = SeedVR2Config.r3B
        config.vidDim = 128; config.heads = 1; config.txtInDim = 64
        config.numLayers = 2; config.mmLayers = 1; config.expandRatio = 2
        return (SeedVR2Transformer(config), SeedVR2VAE(blockOut: [64, 64, 64, 64]))
    }
    @Test(.heavy) func storedTiersCoverTransformerAndVAE() throws {
        NativeRuntime.prepare()
        let repo = (0..<5).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() }
        let lab = repo.appendingPathComponent("lab/g4/tier-tests")
        try FileManager.default.createDirectory(at: lab, withIntermediateDirectories: true)
        let root = lab.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let (transformer, vae) = graph()
        for (name, module) in [("transformer", transformer as Module), ("vae", vae as Module)] {
            try NativeParameters.initializeRandom(module)
            // Distinct bias values expose streamed weight/bias ordering mistakes.
            var biasUpdates: [String: MLXArray] = [:]
            for (key, value) in module.parameters().flattened() where key.hasSuffix(".bias") {
                biasUpdates[key] = MLXArray((0..<value.size).map { Float(($0 % 13) - 6) / 100 }).reshaped(value.shape).asType(.float16)
            }
            try module.update(parameters: ModuleParameters.unflattened(biasUpdates), verify: [.noUnusedKeys, .shapeMismatch])
            try save(arrays: Dictionary(uniqueKeysWithValues: module.parameters().flattened()), url: source.appendingPathComponent(name + ".safetensors"))
        }
        let converter = NativeConversion(recipe: "native:seedvr2", eligible: SeedVR2Adapter.eligible)
        for precision in ["fp16", "int8", "int4"] {
            let destination = root.appendingPathComponent(precision)
            _ = try converter.run(
                ["src_dir": .string(source.path), "dst_dir": .string(destination.path), "dtype": .string(precision), "group_size": 64, "recipe": "native:seedvr2"],
                cancellation: NativeCancellation())
            let (targetTransformer, targetVAE) = graph()
            for (name, original, target) in [("transformer", transformer as Module, targetTransformer as Module), ("vae", vae as Module, targetVAE as Module)] {
                let expected = original.parameters().flattened().filter { SeedVR2Adapter.eligible($0.0, $0.1.shape) }.count
                let coverage = try NativePrecisionTier(precision).load(
                    target, archives: [destination.appendingPathComponent(name + ".safetensors")], map: { ($0, $1) }, eligible: SeedVR2Adapter.eligible)
                #expect(expected > 0)
                #expect(coverage.quantizedTensors == (precision == "fp16" ? 0 : expected))
                let loaded = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
                try NativeParameters.requireHalf(loaded)
                for (key, value) in original.parameters().flattened() where key.hasSuffix(".bias") {
                    #expect(MLX.all(value .== loaded[key]!).item(Bool.self))
                }
            }
        }
    }
}
