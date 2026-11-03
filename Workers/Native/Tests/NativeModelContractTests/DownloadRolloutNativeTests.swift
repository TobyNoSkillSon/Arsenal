// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
import ModelPrecision
import VireoCore
import DecisionsProtocol
@testable import DecisionsWorker
import VerdictCore
@testable import VerdictEngine
@testable import SoundEngine

struct DownloadRolloutNativeTests {
    @Test func vonSourceRoundingHandlesTiesWithoutExecutingFP32() throws {
        let values: [UInt32] = [0x3f808000, 0x3f818000, 0xbf818000, 0x00000000]
        let source = values.withUnsafeBytes { Data($0) }
        func words(_ bytes: Data) -> [UInt16] { bytes.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) } }
        #expect(words(Safetensors.bfloat16(fromFloat32: source, truncate: true)) == [0x3f80, 0x3f81, 0xbf81, 0])
        #expect(words(Safetensors.bfloat16(fromFloat32: source, truncate: false)) == [0x3f80, 0x3f82, 0xbf82, 0])
        for id in ["von-1.1", "von-1.2"] {
            let spec = try #require(DecisionsCatalog.verdict.spec(id))
            #expect(spec.helperBits(dtype: "fp32") == nil && spec.effectiveBits(0) == 16)
            #expect(VonStore.keepsF32("scorer.dense.weight") == false)
            #expect(VonStore.keepsF32("final_norm.weight") == false)
        }
    }

    @Test func decisionViewMetadataSurvivesStoreProvenanceReceipt() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("decision-view-test-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), target = root.appendingPathComponent("derived")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data([0, 1]).write(to: source.appendingPathComponent("weights.bin"))
        let request = OrderedJSON.object([
            ("src_dir", .string(source.path)), ("dst_dir", .string(target.path)), ("dtype", .string("int8")),
            ("recipe", .string("mlx-affine-g64:verdict-at-load"))
        ])
        _ = try Convert.run(request)
        try Data("{\"precision\":\"int8\",\"schema\":1}".utf8).write(to: target.appendingPathComponent("arsenal-derived.json"))
        #expect(Convert.manifest != "arsenal-derived.json" && Convert.derivedDtype(target) == "int8")
    }

    @Test(.heavy) func reUsePacksSSMProjectionAndPreservesOnlyDeclaredHalfFallback() throws {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("reuse-intake-test-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let weights = [
            "xProj.weight": MLXArray.ones([36, 256], dtype: .float16), "dtProj.weight": MLXArray.ones([256, 4], dtype: .float16),
            "inProj.weight": MLXArray.ones([512, 64], dtype: .float16)
        ]
        try save(arrays: weights, url: source.appendingPathComponent(ReUseEngine.weightsFile))
        for tier in ["int8", "int4"] {
            let output = root.appendingPathComponent(tier)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            let fallback = tier == "int8" ? "inProj" : ""
            let model = ModelSource(id: "re-use", engine: "re-use", directory: source, precision: "fp16", options: ["intake_half_modules": fallback])
            try SoundEngines.convert(model, dtype: tier, into: output)
            let stored = try TensorArchive(output.appendingPathComponent(ReUseEngine.weightsFile)).arrays()
            #expect(stored["xProj.weight"]?.dtype == .uint32 && stored["xProj.scales"]?.dtype == .float16)
            #expect(stored["dtProj.weight"]?.dtype == .float16)
            #expect(stored["inProj.weight"]?.dtype == (tier == "int8" ? .float16 : .uint32))
            #expect(stored.values.allSatisfy { [.float16, .uint32].contains($0.dtype) })
        }
    }

    @Test(.heavy) func reUseHalfMatmulsRemainHalfAcrossFusedScanAndConv() throws {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
        for half: DType in [.float16, .bfloat16] {
            let model = ReUseMamba(dModel: 64, kernels: .metal)
            let weights = Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                    (key, MLXArray.full(value.shape, values: MLXArray(Float(0.001)), dtype: half))
                })
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            let result = model(MLXArray.full([1, 5, 64], values: MLXArray(Float(0.1)), dtype: half))
            eval(result)
            #expect(result.dtype == half && all(isFinite(result)).item(Bool.self))
            #expect(model.parameters().flattened().allSatisfy { $0.1.dtype == half })
        }
    }
}
