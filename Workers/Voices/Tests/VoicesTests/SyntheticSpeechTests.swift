// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXAudioTTS
import VireoCore
import VireoEngine

// Random checkpoints prove architecture/transport, never speech quality or vendor-weight compatibility.
@Suite(.serialized) struct SyntheticSpeechTests {
    let dev = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Arsenal-dev/weights/voices")
    let root = URL(fileURLWithPath: "/tmp/arsenal-voices/synthetic")
    func requireGPU() throws {
        guard ProcessInfo.processInfo.environment["ARSENAL_HEAVY_TESTS"] == "1" else {
            throw EngineError("Heavy stand-in tests run only with scripts/test.sh --heavy --filter VoicesTests")
        }
    }
    func config(_ name: String) throws -> Data { try Data(contentsOf: Bundle.module.url(forResource: name, withExtension: "json")!) }
    func save(_ module: Module, to file: URL) throws {
        var tensors = Dictionary(
            uniqueKeysWithValues: module.parameters().flattened().map { ($0.0.replacingOccurrences(of: "speakerEncoder.", with: "speaker_encoder."), $0.1.asType(.bfloat16)) })
        if module is Qwen3TTSSpeechTokenizer {
            var raw: [String: MLXArray] = [:]
            for (key, value) in tensors {
                if key.hasSuffix(".codebook.initialized") { continue }
                if key.hasPrefix("encoder_model.") {
                    var k = key
                    if k.hasPrefix("encoder_model.encoder.init_conv1d.conv.") {
                        k = k.replacingOccurrences(of: "encoder_model.encoder.init_conv1d.conv.", with: "encoder.encoder.layers.0.")
                    } else if k.hasPrefix("encoder_model.encoder.final_conv1d.conv.") {
                        k = k.replacingOccurrences(of: "encoder_model.encoder.final_conv1d.conv.", with: "encoder.encoder.layers.14.")
                    } else if k.hasPrefix("encoder_model.encoder.layers.") {
                        let parts = k.split(separator: ".").map(String.init), i = Int(parts[3])!
                        if parts[4] == "downsample" {
                            k = "encoder.encoder.layers.\(3 + i * 3)." + parts.dropFirst(6).joined(separator: ".")
                        } else {
                            let block = Int(parts[7])!
                            k = "encoder.encoder.layers.\(1 + i * 3).block.\(block == 0 ? 1 : 3)." + parts.dropFirst(9).joined(separator: ".")
                        }
                    } else if k.hasPrefix("encoder_model.encoder_transformer.transformer.layers.") {
                        k = k.replacingOccurrences(of: "encoder_model.encoder_transformer.transformer.layers.", with: "encoder.encoder_transformer.layers.")
                        for (a, b) in [
                            ("self_attn.in_proj.weight", "self_attn.qkv.weight"), ("gating.linear1", "mlp.fc1"), ("gating.linear2", "mlp.fc2"), ("norm1", "input_layernorm"),
                            ("norm2", "post_attention_layernorm"), ("layer_scale_1", "self_attn_layer_scale"), ("layer_scale_2", "mlp_layer_scale")
                        ] { k = k.replacingOccurrences(of: a, with: b) }
                    } else if k.hasPrefix("encoder_model.downsample.conv.conv.") {
                        k = k.replacingOccurrences(of: "encoder_model.downsample.conv.conv.", with: "encoder.downsample.")
                    } else if k.hasPrefix("encoder_model.quantizer.") {
                        k = k.replacingOccurrences(of: "encoder_model.quantizer.", with: "encoder.quantizer.").replacingOccurrences(of: ".vq.layers.", with: ".layers.")
                    } else {
                        throw EngineError("Unknown synthetic encoder tensor \(key)")
                    }
                    raw[k] = value.ndim == 3 && key.hasSuffix("weight") ? value.transposed(0, 2, 1) : value
                } else if key.hasSuffix(".codebook.cluster_usage") || key.hasSuffix(".codebook.embedding_sum") {
                    raw[key.replacingOccurrences(of: ".codebook.", with: "._codebook.")] = value
                } else {
                    raw[key] = value
                }
            }
            let decoded = Qwen3TTSSpeechTokenizer.sanitize(weights: raw)
            for (key, expected) in tensors {
                guard decoded[key]?.shape == expected.shape else {
                    throw EngineError("Synthetic codec layout round trip failed at \(key): \(decoded[key]?.shape.description ?? "missing") vs \(expected.shape)")
                }
            }
            tensors = raw
        }
        eval(Array(tensors.values)); try MLX.save(arrays: tensors, url: file)
    }
    func fixture(_ id: String) throws -> URL {
        MLXRandom.seed(24301)
        let folder = root.appendingPathComponent(id + "-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let data = try config(id); try data.write(to: folder.appendingPathComponent("config.json"))
        let isBreeze = id == "breeze-tts-2"
        let codecFolder = folder.appendingPathComponent(isBreeze ? "audio_tokenizer" : "speech_tokenizer")
        try FileManager.default.createDirectory(at: codecFolder, withIntermediateDirectories: true)
        let codecData = try config(id + "-codec"); try codecData.write(to: codecFolder.appendingPathComponent("config.json"))
        let codec = Qwen3TTSSpeechTokenizer(config: try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: codecData))
        try save(codec, to: codecFolder.appendingPathComponent("model.safetensors"))
        if isBreeze {
            let model = BreezeTTSModel(config: try JSONDecoder().decode(BreezeTTSConfig.self, from: data))
            try save(model, to: folder.appendingPathComponent("model.safetensors"))
        } else {
            let model = Qwen3TTSModel(config: try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: data))
            try save(model, to: folder.appendingPathComponent("model.safetensors"))
        }
        for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"] {
            let from = dev.appendingPathComponent(id + "/bf16/" + name)
            if FileManager.default.fileExists(atPath: from.path) { try FileManager.default.copyItem(at: from, to: folder.appendingPathComponent(name)) }
        }
        return folder
    }
    func audio(_ model: any SpeechGenerationModel, clone: Bool = false) async throws -> Int {
        let reference = clone ? MLXArray((0..<2048).map { 0.01 * sin(Float($0) * 0.1) }) : nil
        let output = try await model.generate(
            text: "Hello.", voice: clone ? nil : "Warm narrator", refAudio: reference,
            refText: clone ? "Hello." : nil, language: "english",
            generationParameters: .init(maxTokens: 2, temperature: 0, topP: 1, topK: 1))
        eval(output); let samples = output.asArray(Float.self)
        #expect(!samples.isEmpty); let finite = samples.allSatisfy { $0.isFinite }; #expect(finite)
        return samples.count
    }
    func convert(_ id: String, from base: URL, bits: Int = 8) async throws -> URL {
        let folder = root.appendingPathComponent(id + "-int\(bits)-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let source = ModelSource(id: id, engine: "mlx-audio", directory: base, precision: "bf16")
        let report = try await Engines.convertQuantized(source, bits: bits, into: folder)
        #expect(report.quantized.count == (id == "breeze-tts-2" ? 468 : 250))
        return folder
    }
    func record(_ id: String, _ result: [String: Any]) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent(id + "-proof.json"))
    }
    @Test(.heavy) func testQwenVoiceDesignSynthetic() async throws {
        try requireGPU(); let id = "qwen3-tts-voicedesign", base = try fixture(id)
        let model = try await Qwen3TTSModel.fromModelDirectory(base)
        let frames = try await audio(model)
        let derived = try await convert(id, from: base)
        let quant = try await Qwen3TTSModel.fromModelDirectory(derived)
        let quantFrames = try await audio(quant)
        let derived4 = try await convert(id, from: base, bits: 4)
        let quant4 = try await Qwen3TTSModel.fromModelDirectory(derived4)
        let quant4Frames = try await audio(quant4, clone: id == "qwen3-tts-base")
        try record(id, ["stand_in": true, "base_samples": frames, "int8_samples": quantFrames, "int4_samples": quant4Frames, "layers": 250])
    }
    @Test(.heavy) func testQwenBaseSynthetic() async throws {
        try requireGPU(); let id = "qwen3-tts-base", base = try fixture(id)
        let model = try await Qwen3TTSModel.fromModelDirectory(base)
        let frames = try await audio(model, clone: true)
        let derived = try await convert(id, from: base)
        let quant = try await Qwen3TTSModel.fromModelDirectory(derived)
        let quantFrames = try await audio(quant, clone: true)
        let derived4 = try await convert(id, from: base, bits: 4)
        let quant4 = try await Qwen3TTSModel.fromModelDirectory(derived4)
        let quant4Frames = try await audio(quant4, clone: id == "qwen3-tts-base")
        try record(id, ["stand_in": true, "base_samples": frames, "int8_samples": quantFrames, "int4_samples": quant4Frames, "layers": 250])
    }
    @Test(.heavy) func testBreezeSyntheticAndMXFP4() async throws {
        try requireGPU(); let id = "breeze-tts-2", base = try fixture(id)
        let model = try await BreezeTTSModel.fromModelDirectory(base)
        let frames = try await audio(model)
        let derived = try await convert(id, from: base)
        let quant = try await BreezeTTSModel.fromModelDirectory(derived)
        let quantFrames = try await audio(quant)
        let packed = root.appendingPathComponent(id + "-mxfp4-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: packed, withIntermediateDirectories: true)
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathComponent("config.json"))) as? [String: Any])
        config["quantization"] = ["group_size": 32, "bits": 4, "mode": "mxfp4"]
        try JSONSerialization.data(withJSONObject: config).write(to: packed.appendingPathComponent("config.json"))
        // Construct packed tensors from this synthetic checkpoint, not a vendor checkpoint or affine conversion.
        let arrays = try MLX.loadArrays(url: base.appendingPathComponent("model.safetensors"))
        var out = arrays; var count = 0
        for (key, value) in arrays where key.hasSuffix(".weight") {
            let path = String(key.dropLast(7))
            guard QuantRules.breeze.selects(path), value.ndim == 2, value.dim(1) % 32 == 0 else { continue }
            let (w, scales, _) = MLX.quantized(value, groupSize: 32, bits: 4, mode: .mxfp4)
            out[key] = w; out[path + ".scales"] = scales; count += 1
        }
        eval(Array(out.values)); try MLX.save(arrays: out, url: packed.appendingPathComponent("model.safetensors"))
        for name in ["audio_tokenizer", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"] {
            let src = base.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: src.path) { try FileManager.default.copyItem(at: src, to: packed.appendingPathComponent(name)) }
        }
        let mx = try await BreezeTTSModel.fromModelDirectory(packed)
        let mxFrames = try await audio(mx)
        #expect(count == 468)
        try record(id, ["stand_in": true, "base_samples": frames, "int8_samples": quantFrames, "mxfp4_samples": mxFrames, "layers": count])
    }
}
