import Foundation
import Testing
import MLX
import MLXNN
import IntegrationKit
import ModelPrecision
import PerceptionSupport
@testable import MLXTIPS
@testable import SAM31

@Suite(.serialized) struct PrecisionTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) } }
    @Test(.heavy, arguments: ["fp16", "int8", "int4"]) func sourceConversionThenStoredModelGraph(tier: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-tips-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), derived = root.appendingPathComponent("derived")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let reference = NativeTIPS.makeModel(large: false, shrunk: true)
        try NativeParameters.initializeRandom(reference)
        let writer = try TensorWriter(source.appendingPathComponent("model.safetensors"))
        // Original checkpoint names and OIHW layout, including fused in_proj_weight/bias and nontrivial linear biases.
        for (name, value) in reference.parameters().flattened() {
            var raw = name
            if raw.hasPrefix("text_encoder.blocks.") { raw = raw.replacingOccurrences(of: "text_encoder.blocks.", with: "text_encoder.transformer.resblocks.") }
            raw = raw.replacingOccurrences(of: ".in_proj.weight", with: ".in_proj_weight").replacingOccurrences(of: ".in_proj.bias", with: ".in_proj_bias")
                .replacingOccurrences(of: ".mlp_c_fc.", with: ".mlp.c_fc.").replacingOccurrences(of: ".mlp_c_proj.", with: ".mlp.c_proj.")
            let tensor = name == "vision_encoder.patch_embed.proj.weight" ? value.transposed(0, 3, 1, 2) : value
            try writer.append(raw, tensor.asType(.float32))
        }
        try writer.finish()
        try Data("{\"model_type\":\"tipsv2\"}".utf8).write(to: source.appendingPathComponent("config.json"))
        _ = try NativeConversion(recipe: "cast:tipsv2-affine-v1", eligible: PerceptionIO.eligible).run(
            ["recipe": "cast:tipsv2-affine-v1", "src_dir": .string(source.path), "dst_dir": .string(derived.path), "dtype": .string(tier), "group_size": 64],
            cancellation: NativeCancellation())
        let candidate = NativeTIPS.makeModel(large: false, shrunk: true)
        let coverage = try NativePrecisionTier(tier).load(
            candidate, archives: [derived.appendingPathComponent("model.safetensors")], map: NativeTIPS.map, eligible: PerceptionIO.eligible)
        #expect(coverage.quantizedTensors == (tier == "fp16" ? 0 : 9))
        let image = MLXArray.ones([1, 28, 28, 3], dtype: .float16)
        let ids = MLXArray([Int32(3), 5, 7], [1, 3]), padding = MLXArray.zeros([1, 3], dtype: .int32)
        let vision = candidate.encodeImage(image), text = candidate.encodeText(ids: ids, paddings: padding)
        eval(vision.clsToken, text)
        #expect(vision.clsToken.shape == [1, 1, 64] && text.shape == [1, 64])
        #expect(MLX.all(MLX.isFinite(text)).item(Bool.self))
        if tier == "fp16" {
            let originalImage = reference.encodeImage(image).clsToken, originalText = reference.encodeText(ids: ids, paddings: padding)
            #expect(MLX.max(abs(originalImage - vision.clsToken)).item(Float.self) == 0)
            #expect(MLX.max(abs(originalText - text)).item(Float.self) == 0)
        }
    }
    @Test(.heavy, arguments: [8, 4]) func samLookupTablesExecuteAfterPacking(bits: Int) throws {
        let table = Embedding(embeddingCount: 3, dimensions: 256)
        try NativeParameters.initializeRandom(table)
        let packed = try #require(table.toQuantized(groupSize: 64, bits: bits) as? Embedding)
        let decoded = samEmbeddingValues(packed)
        eval(decoded)
        #expect(decoded.shape == [3, 256])
        #expect(decoded.dtype == .float16)
        #expect(MLX.all(MLX.isFinite(decoded)).item(Bool.self))
    }
}
