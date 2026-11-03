// SPDX-License-Identifier: MIT
import Foundation
import Testing
import IntegrationKit
import ArsenalWire
import ModelPrecision
import MLX
import MLXNN
@testable import MusicWorker

@Suite(.serialized)
struct ConversionTests {
    private var repo: URL { (0..<5).reduce(URL(fileURLWithPath: #filePath)) { url, _ in url.deletingLastPathComponent() } }
    private func context(_ weights: URL, model: URL, precision: String, random: Bool) throws -> NativeLoadContext {
        try NativeLoadContext([
            "weights_dir": .string(weights.path), "model_dir": .string(model.path), "dtype": .string(precision), "config": ["random_weights": .bool(random), "small": true]
        ])
    }
    private func object(_ value: Any, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: value, options: .sortedKeys).write(to: url)
    }
    private func zip(_ payload: Data, to url: URL) throws {
        var data = Data(); let name = Data("x/data/0".utf8)
        func u16(_ x: Int) { var x = UInt16(x).littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u32(_ x: Int) { var x = UInt32(x).littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        u32(0x04034b50); u16(20); u16(0); u16(0); u16(0); u16(0); u32(0); u32(payload.count); u32(payload.count); u16(name.count); u16(0); data.append(name); data.append(payload)
        let offset = data.count
        u32(0x02014b50); u16(20); u16(20); u16(0); u16(0); u16(0); u16(0); u32(0); u32(payload.count); u32(payload.count); u16(name.count); u16(0); u16(0); u16(0); u16(0); u32(0)
        u32(0); data.append(name)
        let size = data.count - offset
        u32(0x06054b50); u16(0); u16(0); u16(1); u16(1); u32(size); u32(offset); u16(0)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
    private func source(ident: String, original: [String: MLXArray], at root: URL) throws -> [String: URL] {
        let model = repo.appendingPathComponent("models/" + ident)
        let rawMap = try JSONSerialization.jsonObject(with: Data(contentsOf: model.appendingPathComponent("gate/component-map.json")))
        let map = try #require(rawMap as? [String: Any])
        let rules = try #require(map["tensors"] as? [[String: Any]])
        let files: [String: String] =
            ident == "yue2-3b"
            ? ["transformer": "model.safetensors", "vae": "asset/model.safetensors"]
            : [
                "dit": "acestep-v15-turbo/model.safetensors", "vae": "vae/diffusion_pytorch_model.safetensors", "planner": "acestep-5Hz-lm-1.7B/model.safetensors",
                "text_encoder": "Qwen3-Embedding-0.6B/model.safetensors"
            ]
        for (component, path) in files {
            let writer = try TensorWriter(root.appendingPathComponent(path))
            for rule in rules where rule["component"] as? String == component {
                guard let target = rule["target"] as? String, let value = original[target], let source = rule["source"] as? String else { continue }
                try autoreleasepool {
                    var raw = value
                    if let axes = rule["permutation"] as? [Int] {
                        let inverse = (0..<axes.count).map { axes.firstIndex(of: $0)! }; raw = raw.transposed(axes: inverse)
                    }
                    if let gain = rule["weight_norm_gain"] as? String {
                        raw = raw.asType(.float32)
                        let norms = MLX.sqrt((raw.reshaped(raw.dim(0), -1) * raw.reshaped(raw.dim(0), -1)).sum(axis: 1))
                        try writer.append(gain, norms.reshaped([raw.dim(0), 1, 1]))
                    } else if component == "vae", ident == "ace-step-1.5", source.hasSuffix(".alpha") || source.hasSuffix(".beta") {
                        raw = raw.reshaped([1, raw.size, 1])
                    }
                    let dtype: DType = ident == "yue2-3b" && component == "vae" ? .float32 : .bfloat16
                    try writer.append(source, raw.asType(dtype))
                }
            }
            try writer.finish()
        }
        if ident == "yue2-3b" {
            try object(
                [
                    "model_type": "yue2", "hidden_size": 256, "num_hidden_layers": 2, "intermediate_size": 512, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128,
                    "vocab_size": 184704
                ], to: root.appendingPathComponent("config.json"))
            try object(
                ["decoder_config": ["channels": 2, "c_mults": [1, 2, 4, 4, 4, 4], "latent_dim": 64, "out_channels": 2]], to: root.appendingPathComponent("asset/config.json"))
            let rows = (0..<YuE2Protocol.eod).map { id -> String in
                let bytes = id < 256 ? Data([UInt8(id)]) : Data(" token\(id)".utf8)
                return bytes.base64EncodedString() + " \(id)"
            }
            try (rows.joined(separator: "\n") + "\n").write(to: root.appendingPathComponent("qwen.tiktoken"), atomically: true, encoding: .utf8)
            // Asset is outside primary, as required by the verified conversion contract.
            let asset = root.deletingLastPathComponent().appendingPathComponent("vae-source")
            try FileManager.default.moveItem(at: root.appendingPathComponent("asset"), to: asset)
            return ["yue2-vae": asset]
        }
        let config: [String: Any] = [
            "model_type": "acestep", "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "text_hidden_dim": 128, "num_lyric_encoder_hidden_layers": 1, "num_attention_pooler_hidden_layers": 1, "num_timbre_encoder_hidden_layers": 1, "timbre_fix_frame": 16,
            "is_turbo": true
        ]
        try object(config, to: root.appendingPathComponent("config.json")); try object(config, to: root.appendingPathComponent("acestep-v15-turbo/config.json"))
        try object(
            [
                "audio_channels": 2, "channel_multiples": [1, 2, 4, 4, 4], "decoder_channels": 2, "decoder_input_channels": 64, "downsampling_ratios": [2, 4, 4, 6, 10],
                "encoder_hidden_size": 128, "sampling_rate": 48000
            ], to: root.appendingPathComponent("vae/config.json"))
        let qwen: [String: Any] = [
            "hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "rms_norm_eps": 0.000001,
            "vocab_size": 1024, "tie_word_embeddings": true, "model_type": "qwen3"
        ]
        let (tokenizerConfig, tokenizer) = MusicSupport.fixtureTokenizerFiles()
        for component in ["Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"] {
            try object(qwen, to: root.appendingPathComponent(component + "/config.json"))
            try object(tokenizerConfig, to: root.appendingPathComponent(component + "/tokenizer_config.json"))
            try object(tokenizer, to: root.appendingPathComponent(component + "/tokenizer.json"))
            try object(Dictionary(uniqueKeysWithValues: (0..<64).map { ("<|audio_code_\($0)|>", 256 + $0) }), to: root.appendingPathComponent(component + "/added_tokens.json"))
        }
        let silence = original["silence.silence"]!.transposed(1, 0).asType(.float32).asData(access: .noCopyIfContiguous).data
        try zip(silence, to: root.appendingPathComponent("acestep-v15-turbo/silence_latent.pt"))
        return [:]
    }
    private func run(_ adapter: any NativeModel, ident: String, at root: URL) throws {
        let input: JSONObject
        if ident == "yue2-3b" {
            input = [
                "prompt": "piano", "lyrics": "[Verse] Hello", "duration": 0.04, "seed": 42, "steps": 1, "out_dir": .string(root.path),
                "yue": ["planning": "off", "semantic_min_tokens": 1, "semantic_max_tokens": 1, "decoder_tile_frames": 16]
            ]
        } else {
            input = [
                "prompt": "piano", "lyrics": "Hello", "language": "en", "duration": 10, "seed": 42, "steps": 1, "out_dir": .string(root.path), "thinking": true,
                "plan_metadata": false, "rewrite_caption": false, "detect_language": false, "dcw": false
            ]
        }
        let output = try adapter.run("generate_music", input: input, cancellation: NativeCancellation(), emit: { _ in })
        #expect(output["files"]?.array?.count == 1)
        #expect(output["files"]?.array?.first?.object?["mime"]?.string == "audio/wav")
        try NativeParameters.requireHalf(adapter.parameters)
    }

    @Test(.heavy) func nontrivialWeightNormAndConvolutionLayout() throws {
        NativeRuntime.prepare()
        let lab = repo.appendingPathComponent("lab/g4/conversion-math/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: lab, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: lab) }
        let raw = lab.appendingPathComponent("raw.safetensors"), prepared = lab.appendingPathComponent("prepared.safetensors")
        let vector = (0..<24).map { Float($0 - 9) / 20 }
        let gain: [Float] = [0.7, 1.9]
        let writer = try TensorWriter(raw)
        try writer.append("decoder.conv1.weight_v", MLXArray(vector).reshaped(2, 4, 3))
        try writer.append("decoder.conv1.weight_g", MLXArray(gain).reshaped(2, 1, 1)); try writer.finish()
        try MusicConversion.transform(
            raw, to: prepared, half: .float16, name: { $0 }, value: { _, x in x.swappedAxes(1, 2) }, foldWeightNorm: true, cancellation: NativeCancellation())
        let actual = try TensorArchive(prepared).array("decoder.conv1.weight")
        #expect(actual.shape == [2, 3, 4]); #expect(actual.dtype == .float16)
        var expected = [Float]()
        for output in 0..<2 {
            let norm = sqrt(vector[(output * 12)..<(output * 12 + 12)].reduce(Double(0)) { $0 + Double($1) * Double($1) })
            for kernel in 0..<3 { for input in 0..<4 { expected.append(Float(Double(gain[output]) * Double(vector[output * 12 + input * 3 + kernel]) / (norm + 1e-9))) } }
        }
        let gold = MLXArray(expected).reshaped(2, 3, 4).asType(.float16)
        #expect(MLX.max(MLX.abs(actual - gold)).item(Float.self) <= 0.001)
    }

    @Test func audioFileFormats() throws {
        let root = repo.appendingPathComponent("lab/g4/audio-formats/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let samples = (0..<4800).map { Float(sin(Double($0) * 2 * Double.pi * 220 / 48000)) * 0.1 }
        let audio = SongAudio(channels: [samples, samples], sampleRate: 48000)
        for format in AudioFileFormat.allCases {
            let file = root.appendingPathComponent(format.rawValue + "." + format.fileExtension)
            try AudioFiles.write(audio, to: file, format: format)
            let decoded = try NativeMedia.audio(file, sampleRate: 48000, channels: 2)
            #expect(decoded.channels == 2)
            #expect(decoded.samples.count >= 2 * 4000)
            let allFinite = decoded.samples.allSatisfy(\.isFinite)
            #expect(allFinite)
        }
    }

    @Test(.heavy) func completeSnapshotsConvertAndReloadAllTiers() throws {
        NativeRuntime.prepare(); setenv("ARSENAL_NATIVE_LAB", "1", 1)
        let lab = repo.appendingPathComponent("lab/g4/conversion-tests")
        try FileManager.default.createDirectory(at: lab, withIntermediateDirectories: true)
        for ident in ["yue2-3b", "ace-step-1.5"] {
            let directory = lab.appendingPathComponent(UUID().uuidString); let raw = directory.appendingPathComponent("source")
            try FileManager.default.createDirectory(at: raw, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let model = repo.appendingPathComponent("models/" + ident)
            let adapter: any NativeModel = ident == "yue2-3b" ? YuEAdapter() : ACEAdapter()
            try adapter.load(context(raw, model: model, precision: "fp16", random: true), cancellation: NativeCancellation())
            let originals = adapter.parameters
            let assets = try source(ident: ident, original: originals, at: raw)
            adapter.unload()
            let base = directory.appendingPathComponent("base")
            let converter = ident == "yue2-3b" ? MusicConversion.yueBase : MusicConversion.aceBase
            _ = try converter.run(
                [
                    "src_dir": .string(raw.path), "dst_dir": .string(base.path), "dtype": "fp16", "group_size": 64, "recipe": .string(converter.recipe),
                    "assets": .object(assets.mapValues { .string($0.path) })
                ], cancellation: NativeCancellation())
            try adapter.load(context(base, model: model, precision: "fp16", random: false), cancellation: NativeCancellation())
            let baseline = adapter.parameters
            #expect(Set(baseline.keys) == Set(originals.keys))
            try NativeParameters.requireHalf(baseline)
            try run(adapter, ident: ident, at: directory.appendingPathComponent("output-fp16"))
            adapter.unload()
            for precision in ["int8", "int4"] {
                let tier = directory.appendingPathComponent(precision), converter = ident == "yue2-3b" ? MusicConversion.yueTiers : MusicConversion.aceTiers
                _ = try converter.run(
                    ["src_dir": .string(base.path), "dst_dir": .string(tier.path), "dtype": .string(precision), "group_size": 64, "recipe": .string(converter.recipe)],
                    cancellation: NativeCancellation())
                try adapter.load(context(tier, model: model, precision: precision, random: false), cancellation: NativeCancellation())
                let packed = adapter.parameters
                let eligible = baseline.filter { MusicConversion.eligible($0.key, $0.value.shape) }
                #expect(!eligible.isEmpty)
                #expect(packed.filter { $0.value.dtype == .uint32 }.count == eligible.count)
                for (key, value) in baseline where key.hasSuffix(".bias") { #expect(MLX.all(value .== packed[key]!).item(Bool.self)) }
                try NativeParameters.requireHalf(packed)
                try run(adapter, ident: ident, at: directory.appendingPathComponent("output-" + precision))
                adapter.unload()
            }
            Memory.clearCache()
        }
    }
}
