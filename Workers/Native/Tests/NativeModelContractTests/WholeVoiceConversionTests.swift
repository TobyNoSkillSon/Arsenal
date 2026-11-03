// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import MLX
import MLXNN
import ModelPrecision
import Testing
@testable import MLXAudioTTS
@testable import VireoEngine

@Suite(.serialized) struct WholeVoiceConversionTests {
    final class Fixture: Module {
        @ModuleInfo var linear: Linear
        @ModuleInfo var embedding: Embedding
        override init() {
            _linear.wrappedValue = Linear(256, 256, bias: false)
            _embedding.wrappedValue = Embedding(embeddingCount: 256, dimensions: 256)
        }
    }
    func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-whole-voice-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    func source(_ root: URL, nonfinite: Bool = false) throws -> URL {
        let input = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: input.appendingPathComponent("config.json"))
        try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: input.appendingPathComponent("config.json").path)
        for relative in ["model.safetensors", "speech_tokenizer/model.safetensors"] {
            let writer = try TensorWriter(input.appendingPathComponent(relative))
            let weight = MLXArray((0..<65536).map { Float($0 % 31 - 15) / 16 }, [256, 256]).asType(.bfloat16)
            try writer.append("linear.weight", weight)
            try writer.append("embedding.weight", weight)
            try writer.append("small.weight", MLXArray.ones([4, 64], dtype: .float32))
            try writer.append("norm.weight", MLXArray.ones([256], dtype: .float32))
            try writer.append("incompatible.weight", MLXArray.ones([1024, 65], dtype: .float32))
            if nonfinite { try writer.append("bad.weight", MLXArray.full([256, 256], values: MLXArray(Float.infinity))) }
            try writer.finish()
        }
        let preset = try TensorWriter(input.appendingPathComponent("embeddings/alba.safetensors"))
        try preset.append("audio_prompt", MLXArray.ones([1, 8, 256], dtype: .float32)); try preset.finish()
        return input
    }
    let files = ["config.json", "model.safetensors", "speech_tokenizer/model.safetensors", "embeddings/alba.safetensors"]

    @Test func ruleCoversLargePartsAndExplainsEverySkip() {
        for key in ["mimi.decoder_transformer.linear.weight", "flow_lm.flow_net.proj.weight", "flow_lm.conditioner.embed.weight", "speaker_encoder.fc.weight"] {
            #expect(WholeVoiceConversion.skipReason(key: key, shape: [256, 256]) == nil)
        }
        #expect(WholeVoiceConversion.skipReason(key: "norm.weight", shape: [256]) != nil)
        #expect(WholeVoiceConversion.skipReason(key: "small.weight", shape: [4, 64]) != nil)
        #expect(WholeVoiceConversion.skipReason(key: "conv.weight", shape: [256, 3, 256]) != nil)
        #expect(WholeVoiceConversion.skipReason(key: "proj.weight", shape: [1024, 65]) != nil)
        #expect(WholeVoiceConversion.skipReason(key: "audio_prompt", shape: [1, 500, 256]) != nil)
    }

    @Test(.heavy, arguments: ["fp16", "int8", "int4"]) func allArchivesAreStreamedWithHalfRemaindersAndStoredPacksLoad(dtype: String) throws {
        _ = NativePrecisionTestRuntime.ready
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        try Device.withDefaultDevice(.cpu) {
            let input = try source(root), output = root.appendingPathComponent(dtype)
            let before = try Data(contentsOf: input.appendingPathComponent("model.safetensors"))
            let report = try WholeVoiceConversion.convert(source: input, files: files, model: "pocket-tts", dtype: dtype, half: "fp16", into: output)
            #expect(report.quantized.count == (dtype == "fp16" ? 0 : 4))
            #expect(report.linears == 4)
            #expect(try Data(contentsOf: input.appendingPathComponent("model.safetensors")) == before)
            #expect(try Data(contentsOf: input.appendingPathComponent("config.json")) == Data("{}".utf8))
            #expect(StoredVoiceWeights.isIntake(try Data(contentsOf: output.appendingPathComponent("config.json"))))
            for relative in files where relative.hasSuffix(".safetensors") {
                let archive = try TensorArchive(output.appendingPathComponent(relative))
                #expect(!archive.entries.values.contains { $0.dtype == .float32 || $0.dtype == .float64 })
                if relative.hasPrefix("embeddings/") { continue }
                #expect(archive.entries["small.weight"]?.dtype == .float16)
                #expect(archive.entries["incompatible.weight"]?.dtype == .float16)
                if dtype != "fp16" {
                    let model = Fixture()
                    let stored = try archive.arrays().filter { $0.key.hasPrefix("linear.") || $0.key.hasPrefix("embedding.") }
                    #expect(try StoredVoiceWeights.prepare(model, weights: stored) == 2)
                    try model.update(parameters: ModuleParameters.unflattened(stored), verify: .all)
                    #expect(model.linear is QuantizedLinear && model.embedding is QuantizedEmbedding)
                    let value = model.linear(MLXArray.ones([1, 256], dtype: .float16))
                    #expect(value.shape == [1, 256] && value.dtype == .float16)
                    let embed = model.embedding(MLXArray([Int32(1)]))
                    #expect(embed.shape == [1, 256] && embed.dtype == .float16)
                }
            }
        }
    }

    @Test(.heavy) func mixedInt8KeepsOnlyFailingOwnersAndInt4PacksBoth() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        try Device.withDefaultDevice(.cpu) {
            let input = root.appendingPathComponent("source")
            try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: input.appendingPathComponent("config.json"))
            let writer = try TensorWriter(input.appendingPathComponent("model.safetensors"))
            for key in ["flow_lm.transformer.layers.0.linear1.weight", "flow_lm.conditioner.embed.weight"] {
                try writer.append(key, MLXArray.ones([256, 256], dtype: .bfloat16))
            }
            try writer.finish()
            let files = ["config.json", "model.safetensors"]
            let mixed = root.appendingPathComponent("int8")
            let report = try WholeVoiceConversion.convert(source: input, files: files, model: "pocket-tts", dtype: "int8", half: "bf16", keeping: ["backbone"], into: mixed)
            let stored = try TensorArchive(mixed.appendingPathComponent("model.safetensors"))
            #expect(stored.entries["flow_lm.transformer.layers.0.linear1.weight"]?.dtype == .bfloat16)
            #expect(stored.entries["flow_lm.conditioner.embed.weight"]?.dtype == .uint32)
            #expect(report.fallbackParameters == 65536 && report.quantizedParameters == 65536)
            #expect(report.fallbackComponents == ["backbone"] && report.kept.values.contains { $0.contains("component gate failed") })
            let int4 = try WholeVoiceConversion.convert(source: input, files: files, model: "pocket-tts", dtype: "int4", half: "bf16", into: root.appendingPathComponent("int4"))
            #expect(int4.quantizedParameters == 131072 && int4.fallbackParameters == 0)
            #expect(throws: (any Error).self) {
                try WholeVoiceConversion.convert(
                    source: input, files: files, model: "pocket-tts", dtype: "int4", half: "bf16", keeping: ["backbone"], into: root.appendingPathComponent("invalid"))
            }
        }
    }
    @Test(.heavy) func matrixFallbackKeepsOneMatrixInAnOtherwiseQuantizedLayer() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let input = try source(root), output = root.appendingPathComponent("mixed")
        let selector = "matrix@model.safetensors:linear.weight"
        let recipe = WholeVoiceConversion.matrixRecipe(model: "pocket-tts", keeping: [selector])
        #expect(try WholeVoiceConversion.keeping(recipe: recipe, model: "pocket-tts", dtype: "int8") == [selector])
        let special = "matrix@model.safetensors:speaker_proj_weight"
        let specialRecipe = WholeVoiceConversion.matrixRecipe(model: "pocket-tts", keeping: [special])
        #expect(try WholeVoiceConversion.keeping(recipe: specialRecipe, model: "pocket-tts", dtype: "int8") == [special])
        let auxiliary: Set<String> = [
            "matrix@asset-kokoro-g2p/us_bart.safetensors:model.encoder.layers.0.fc1.weight", "matrix@asset-kokoro-g2p/us_bart.safetensors:model.decoder.layers.0.fc2.weight"
        ]
        #expect(
            try WholeVoiceConversion.keeping(recipe: WholeVoiceConversion.matrixRecipe(model: "kokoro-82m", keeping: auxiliary), model: "kokoro-82m", dtype: "int8") == auxiliary)
        #expect(throws: (any Error).self) { try WholeVoiceConversion.keeping(recipe: recipe, model: "pocket-tts", dtype: "int4") }
        let report = try WholeVoiceConversion.convert(
            source: input, files: files, model: "pocket-tts", dtype: "int8", half: "bf16",
            keeping: [selector], matrixPolicy: true, into: output)
        let arrays = try TensorArchive(output.appendingPathComponent("model.safetensors")).arrays()
        #expect(arrays["linear.weight"]?.dtype == .bfloat16)
        #expect(arrays["embedding.weight"]?.dtype == .uint32)
        #expect(report.fallbackMatrices == [selector] && report.fallbackParameters == 65536)
        #expect(report.matrixParameters.count == 4 && report.matrixScalesSHA256.count == 3)
        #expect(report.rule.id == recipe)
        let fixture = Fixture()
        let stored = arrays.filter { $0.key.hasPrefix("linear.") || $0.key.hasPrefix("embedding.") }
        _ = try StoredVoiceWeights.prepare(fixture, weights: stored)
        try fixture.update(parameters: ModuleParameters.unflattened(stored), verify: .all)
        let packed = fixture.embedding
        let dense = Embedding(weight: MLXArray.ones([256, 256], dtype: .bfloat16))
        try StoredVoiceWeights.replace(fixture, path: "embedding", with: dense)
        #expect(fixture.embedding === dense && !(fixture.linear is QuantizedLinear))
        try StoredVoiceWeights.replace(fixture, path: "embedding", with: packed)
        #expect(fixture.embedding === packed && fixture.embedding is QuantizedEmbedding)
    }
    @Test(.heavy) func ablationMapsOnlyFailingMatricesAndLeavesUnrelatedFusedPacksAlone() throws {
        _ = NativePrecisionTestRuntime.ready
        let candidate = Fixture(), donor = Fixture()
        let raw = MLXArray.ones([256, 256], dtype: .bfloat16)
        let (weight, scales, biases) = MLX.quantized(raw, groupSize: 64, bits: 8, mode: .affine)
        let stored = ["linear.weight": weight, "linear.scales": scales, "linear.biases": biases!]
        _ = try StoredVoiceWeights.prepare(candidate, weights: stored)
        try candidate.update(parameters: ModuleParameters.unflattened(stored), verify: .none)
        let hash = SHA256.hash(data: scales.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
        let selector = "matrix@model.safetensors:linear.weight"
        let pairs = try VoiceMatrixDiagnosis.pairs(roots: [("fixture", candidate)], denseRoots: ["fixture": donor], hashes: [selector: hash])
        #expect(pairs.count == 1 && pairs[0].selectors == [selector] && pairs[0].path == "linear")
        #expect(pairs[0].half === donor.linear && pairs[0].packed === candidate.linear)
        #expect(throws: (any Error).self) {
            try VoiceMatrixDiagnosis.pairs(roots: [("fixture", candidate)], denseRoots: ["fixture": donor], hashes: [selector: "missing"])
        }
        let unrelated = QuantizedEmbedding(weight: weight, scales: scales * 2, biases: biases!, groupSize: 64, bits: 8)
        try StoredVoiceWeights.replace(candidate, path: "embedding", with: unrelated)
        let again = try VoiceMatrixDiagnosis.pairs(roots: [("fixture", candidate)], denseRoots: ["fixture": donor], hashes: [selector: hash])
        #expect(again.count == 1 && candidate.embedding === unrelated)
    }
    final class FusedProjection: Module {
        @ModuleInfo(key: "in_proj") var projection: Linear
        override init() { _projection.wrappedValue = Linear(256, 768, bias: false) }
    }
    @Test(.heavy) func ablationMapsAllThreeSourceBlocksOfAFusedCodecProjection() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.gpu) {
            let candidate = FusedProjection(), donor = FusedProjection()
            var weights: [MLXArray] = [], scales: [MLXArray] = [], biases: [MLXArray] = [], hashes: [String: String] = [:]
            for index in 0..<3 {
                let raw = MLXArray((0..<65536).map { Float(($0 + index * 7) % (31 + index * 2) - 16) / 32 }, [256, 256]).asType(.bfloat16)
                let (w, s, b) = MLX.quantized(raw, groupSize: 64, bits: 8, mode: .affine)
                weights.append(w); scales.append(s); biases.append(try #require(b))
                hashes["matrix@tokenizer/model.safetensors:encoder.\(index).weight"] = SHA256.hash(data: s.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
            }
            let stored = ["in_proj.weight": concatenated(weights, axis: 0), "in_proj.scales": concatenated(scales, axis: 0), "in_proj.biases": concatenated(biases, axis: 0)]
            _ = try StoredVoiceWeights.prepare(candidate, weights: stored)
            try candidate.update(parameters: ModuleParameters.unflattened(stored), verify: .all)
            let pairs = try VoiceMatrixDiagnosis.pairs(roots: [("codec", candidate)], denseRoots: ["codec": donor], hashes: hashes)
            #expect(pairs.count == 1 && pairs[0].selectors == Set(hashes.keys))
            try StoredVoiceWeights.replace(pairs[0].root, path: pairs[0].path, with: pairs[0].half)
            #expect(candidate.projection === donor.projection)
            try StoredVoiceWeights.replace(pairs[0].root, path: pairs[0].path, with: pairs[0].packed)
            hashes.removeValue(forKey: "matrix@tokenizer/model.safetensors:encoder.2.weight")
            #expect(throws: (any Error).self) {
                try VoiceMatrixDiagnosis.pairs(roots: [("codec", candidate)], denseRoots: ["codec": donor], hashes: hashes)
            }
        }
    }
    @Test(.heavy) func sourceTruncationPerturbsOnlyOriginalFP32WeightsAndNeverExecutesThem() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: input.appendingPathComponent("config.json"))
        let writer = try TensorWriter(input.appendingPathComponent("model.safetensors"))
        let values = MLXArray([Float(1.006), -1.006, 2.013, -2.013], [2, 2])
        try writer.append("fp32.weight", values)
        try writer.append("native.weight", values.asType(.bfloat16)); try writer.finish()
        var outputs: [[String: MLXArray]] = []
        for truncate in [false, true] {
            let output = root.appendingPathComponent(truncate ? "trunc" : "nearest")
            _ = try WholeVoiceConversion.convert(
                source: input, files: ["config.json", "model.safetensors"], model: "pocket-tts",
                dtype: "bf16", half: "bf16", truncateSource: truncate, into: output)
            outputs.append(try TensorArchive(output.appendingPathComponent("model.safetensors")).arrays())
        }
        #expect(outputs[0]["native.weight"]!.asArray(Float.self) == outputs[1]["native.weight"]!.asArray(Float.self))
        #expect(outputs[0]["fp32.weight"]!.asArray(Float.self) != outputs[1]["fp32.weight"]!.asArray(Float.self))
        #expect(outputs[1]["fp32.weight"]!.asArray(Float.self) == [1, -1, 2, -2])
        #expect(outputs.flatMap { $0.values }.allSatisfy { $0.dtype == .bfloat16 })
    }

    @Test func invalidCastCleansItsPartialOutputAndOccupiedDestinationsArePreserved() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        try Device.withDefaultDevice(.cpu) {
            let input = try source(root, nonfinite: true), output = root.appendingPathComponent("failed")
            #expect(throws: (any Error).self) {
                try WholeVoiceConversion.convert(source: input, files: files, model: "pocket-tts", dtype: "int4", half: "fp16", into: output)
            }
            #expect(!FileManager.default.fileExists(atPath: output.path))
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            let sentinel = output.appendingPathComponent("owned-by-someone-else")
            try Data("keep".utf8).write(to: sentinel)
            #expect(throws: (any Error).self) {
                try WholeVoiceConversion.convert(source: input, files: files, model: "pocket-tts", dtype: "int4", half: "fp16", into: output)
            }
            #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
        }
    }

    @Test(.heavy) func missingPackCompanionAndUnknownModuleAreRejected() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu) {
            let model = Fixture()
            #expect(throws: (any Error).self) { try StoredVoiceWeights.prepare(model, weights: ["linear.scales": MLXArray.ones([256, 4], dtype: .float16)]) }
            #expect(throws: (any Error).self) { try StoredVoiceWeights.prepare(model, weights: ["unknown.scales": MLXArray.ones([256, 4], dtype: .float16)]) }
            #expect(throws: (any Error).self) { try StoredVoiceWeights.prepare(model, weights: [:], group: 0) }
        }
    }
    @Test(.heavy) func codecAndBreezeRemappingKeepAllAffineCompanionsTogether() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu) {
            let weight = MLXArray.zeros([384, 8], dtype: .uint32)
            let scales = MLXArray.ones([384, 1], dtype: .float16)
            let biases = MLXArray.zeros([384, 1], dtype: .float16)
            let raw = "encoder.encoder_transformer.layers.0.self_attn.qkv"
            let mapped = Qwen3TTSSpeechTokenizer.sanitize(weights: [raw + ".weight": weight, raw + ".scales": scales, raw + ".biases": biases])
            let key = "encoder_model.encoder_transformer.transformer.layers.0.self_attn.in_proj"
            #expect(Set(mapped.keys) == Set([key + ".weight", key + ".scales", key + ".biases"]))
            #expect(mapped[key + ".weight"]?.shape == [384, 8])
            #expect(mapped[key + ".scales"]?.shape == [384, 1])
            #expect(mapped[key + ".biases"]?.shape == [384, 1])
            let tied = BreezeTTSModel.sanitize(weights: [
                "depth_decoder.model.embed_tokens.weight": weight,
                "depth_decoder.model.embed_tokens.scales": scales, "depth_decoder.model.embed_tokens.biases": biases
            ])
            #expect(tied.count == 6)
            #expect(tied["backbone_model.embed_tokens.embed_audio_tokens.scales"]?.shape == [384, 1])
            #expect(tied["backbone_model.embed_tokens.embed_audio_tokens.biases"]?.shape == [384, 1])
        }
    }

}
