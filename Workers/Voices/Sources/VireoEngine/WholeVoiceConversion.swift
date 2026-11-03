// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import MLX
import ModelPrecision
import MLXAudioTTS
import VireoCore

/// Separate recipe identity: legacy transformer-only results never qualify these packs.
public enum WholeVoiceConversion {
    public static func owner(file: String, key: String) -> String {
        if file.contains("tokenizer/") {
            return key.hasPrefix("encoder.encoder_transformer.") ? "codec-encoder-transformer" : key.hasPrefix("encoder.") ? "codec-encoder" : "codec-decoder"
        }
        if key.hasPrefix("bert.") { return "kokoro-bert" }
        if key.hasPrefix("bert_encoder.") { return "kokoro-projection" }
        if key.hasPrefix("predictor.") { return "kokoro-prosody" }
        if key.hasPrefix("decoder.") { return "kokoro-decoder" }
        if key.hasPrefix("text_encoder.") && file.contains("kokoro") { return "kokoro-text" }
        let mappings = [
            ("talker.code_predictor.model.codec_embedding.", "predictor-embedding"), ("talker.code_predictor.", "predictor"), ("talker.model.text_embedding.", "text-embedding"),
            ("talker.model.codec_embedding.", "codec-embedding"), ("talker.text_projection.", "text-projection"),
            ("talker.", "talker"), ("speaker_encoder.", "speaker-encoder"),
            ("flow_lm.conditioner.", "text-conditioner"), ("flow_lm.transformer.", "backbone"),
            ("flow_lm.flow_net.", "flow"), ("flow_lm.input_linear.", "latent-projection"),
            ("flow_lm.out_eos.", "eos"), ("mimi.decoder_transformer.", "mimi-transformer-upsampling"),
            ("mimi.encoder_transformer.", "mimi-encoder"), ("speaker_proj", "speaker-projection"),
            ("text_encoder_proj.", "text-projection"), ("text_encoder.", "text-encoder"),
            ("embed_text_tokens.", "text-embedding"), ("backbone_model.embed_tokens.", "audio-embedding"),
            ("backbone_model.", "backbone"), ("depth_decoder.", "depth"), ("lm_head.", "head")
        ]
        return mappings.first { key.hasPrefix($0.0) }?.1 ?? "unmapped"
    }
    public static func matrixRecipe(model: String, keeping: Set<String> = []) -> String {
        "intake-v3:" + model + ":" + (keeping.isEmpty ? "all" : keeping.sorted().joined(separator: ","))
    }
    public static func recipe(model: String, keeping: Set<String> = []) -> String {
        "intake-v2:" + model + ":" + (keeping.isEmpty ? "all" : keeping.sorted().joined(separator: ","))
    }
    public static func keeping(recipe: String, model: String, dtype: String) throws -> Set<String> {
        if recipe.hasPrefix("intake-v3:" + model + ":") {
            let suffix = String(recipe.dropFirst(("intake-v3:" + model + ":").count))
            let matrices = suffix == "all" ? Set<String>() : Set(suffix.split(separator: ",").map(String.init))
            guard matrices.allSatisfy({ $0.hasPrefix("matrix@") && $0.contains(".safetensors:") && ($0.hasSuffix(".weight") || $0.hasSuffix(":speaker_proj_weight")) && !$0.contains("..") }),
                dtype == "int8" || matrices.isEmpty, recipe == matrixRecipe(model: model, keeping: matrices)
            else { throw EngineError("Noncanonical or unsupported voice matrix fallback") }
            return matrices
        }
        let prefix = "intake-v2:" + model + ":"
        guard recipe.hasPrefix(prefix) else { throw EngineError("Invalid voice component recipe") }
        let suffix = String(recipe.dropFirst(prefix.count))
        let parts = suffix == "all" ? Set<String>() : Set(suffix.split(separator: ",").map(String.init))
        let allowed: Set<String> = ["predictor", "predictor-embedding", "text-embedding", "codec-embedding", "text-projection", "talker", "speaker-encoder", "text-conditioner", "backbone", "flow", "latent-projection", "eos", "mimi-decoder", "mimi-encoder", "speaker-projection", "text-encoder", "audio-embedding", "depth", "head", "codec-encoder-transformer", "codec-encoder", "codec-decoder"]
        guard parts.isSubset(of: allowed), dtype == "int8" || parts.isEmpty,
            recipe == Self.recipe(model: model, keeping: parts) else { throw EngineError("Noncanonical or unsupported voice component fallback") }
        return parts
    }
    public static let models: Set<String> = ["breeze-tts-2", "qwen3-tts-base", "qwen3-tts-voicedesign", "pocket-tts", "kokoro-82m"]
    public static func pointwiseSpeaker(key: String, shape: [Int]) -> Bool {
        key.hasPrefix("speaker_encoder.") && key.hasSuffix(".weight") && shape.count == 3 && shape[1] == 1
    }
    public static func skipReason(key: String, shape sourceShape: [Int]) -> String? {
        let shape = pointwiseSpeaker(key: key, shape: sourceShape) ? [sourceShape[0], sourceShape[2]] : sourceShape
        if key.contains(".Wx_") || key.contains(".Wh_") { return "recurrent gate matrix: BiLSTM has no stored quantized-matmul interface; selected half" }
        guard key.hasSuffix(".weight") || key == "speaker_proj_weight" else { return "not a Linear/Embedding weight (buffers, biases, norms stay half)" }
        guard shape.count == 2 else { return "non-matrix layout (convolution/norm stays half)" }
        guard shape.reduce(1, *) >= 65_536 else { return "fewer than 65,536 parameters" }
        guard shape[1] % 64 == 0 else { return "input width is incompatible with affine group 64" }
        return nil
    }

    /// Streams ALL declared archives, including nested codecs and preset conditioning. Does not execute a model.
    /// `half` is selected by a separately qualified format gate; conversion alone never selects it.
    public static func convert(source: URL, files: [String], model: String, dtype: String, half: String, keeping: Set<String> = [], truncateSource: Bool = false, matrixPolicy: Bool = false, into output: URL) throws -> QuantReport {
        guard models.contains(model), ["fp16", "bf16"].contains(half), [half, "int8", "int4"].contains(dtype) else {
            throw EngineError("Whole voice conversion needs a supported model and a selected half format")
        }
        let fm = FileManager.default
        guard try !fm.fileExists(atPath: output.path) || fm.contentsOfDirectory(atPath: output.path).isEmpty else {
            throw EngineError("Whole voice conversion destination must be empty")
        }
        let sourceConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as? [String: Any]
        let alreadyStored = sourceConfig?["arsenal_voice_intake"] != nil || sourceConfig?["mobius_voice_intake"] != nil // legacy-compat: read in snapshots written before the rename
        let bits = dtype == "int8" ? 8 : dtype == "int4" ? 4 : 16
        guard dtype == "int8" || keeping.isEmpty else { throw EngineError("Fallback is only valid in int8") }
        try fm.createDirectory(at: output, withIntermediateDirectories: true)
        var report = QuantReport(rule: QuantRule(id: bits == 16 ? "cast:" + model + (truncateSource ? ":bf16-truncation" : "") : matrixPolicy ? matrixRecipe(model: model, keeping: keeping) : recipe(model: model, keeping: keeping), version: 2, include: [".*"], exclude: [],
            note: "All declared components: affine g64 for every compatible Linear/Embedding weight matrix with at least 65,536 parameters; other tensors stay selected half. Int8 retains genuinely failing components at the selected half; int4 packs every eligible part. Unqualified until component and task gates pass."), bits: bits, groupSize: 64)
        report.mode = bits == 16 ? "cast" : "affine"
        report.fallbackComponents = keeping.filter { !$0.hasPrefix("matrix@") }.sorted()
        report.fallbackMatrices = keeping.filter { $0.hasPrefix("matrix@") }.sorted()
        do {
            for relative in files.sorted() {
                guard !relative.hasPrefix("/"), !relative.split(separator: "/").contains("..") else { throw EngineError("Unsafe voice source path") }
                let input = source.appendingPathComponent(relative), target = output.appendingPathComponent(relative)
                try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                // Preserve shard filenames; rebuild the key index after packing/remapping below.
                if input.pathExtension != "safetensors" {
                    guard relative != "config.json" else { continue }
                    try fm.copyItem(at: input.resolvingSymlinksInPath(), to: target)
                    report.files.append(.init(source: relative, output: relative)); continue
                }
                let archive = try TensorArchive(input)
                guard !archive.entries.keys.contains(where: { $0.hasSuffix(".scales") || $0.hasSuffix(".biases") }) else {
                    throw EngineError("Whole voice recipes require an unpacked source")
                }
                let writer = try TensorWriter(target)
                for key in archive.entries.keys.sorted() {
                    try autoreleasepool {
                        let tensor = try archive.array(key)
                        let parts: [(String, MLXArray)]
                        if model == "kokoro-82m", !alreadyStored, !relative.hasPrefix("voices/") {
                            parts = KokoroModel.sanitizeWeights([key: tensor]).sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
                        } else if key == "depth_decoder.codebooks_head.weight", tensor.ndim == 3 {
                            parts = (0..<tensor.dim(0)).map { ("depth_decoder.codebooks_head.heads.\($0).weight", tensor[$0].transposed().contiguous(allowColMajor: false, stream: .gpu)) }
                        } else if key == "text_encoder.embed_tokens.weight" {
                            parts = [("text_encoder.embed_tokens.table.weight", tensor)]
                        } else { parts = [(key, tensor)] }
                        for (mappedKey, raw) in parts {
                            let floating = [.float16, .bfloat16, .float32, .float64].contains(raw.dtype)
                            let rounded: MLXArray
                            if truncateSource, raw.dtype == .float32 || raw.dtype == .float64 {
                                guard half == "bf16", bits == 16 else { throw EngineError("Source truncation is BF16 calibration only") }
                                rounded = MLXArray(raw.asArray(Float.self).map { Float(bitPattern: $0.bitPattern & 0xffff0000) }, raw.shape)
                            } else { rounded = raw }
                            let value = floating ? rounded.asType(half == "fp16" ? .float16 : .bfloat16, stream: .gpu) : raw
                            if floating, !MLX.all(MLX.isFinite(value, stream: .gpu), stream: .gpu).item(Bool.self) {
                                throw EngineError("\(relative):\(key) is non-finite after \(half) cast; format gate required")
                            }
                            report.parameters += raw.size
                            let component = owner(file: relative, key: mappedKey)
                            let eligibleReason = skipReason(key: mappedKey, shape: raw.shape)
                            let matrix = "matrix@" + relative + ":" + mappedKey
                            let retained = keeping.contains(component) || keeping.contains(matrix)
                            let reason = eligibleReason ?? (retained ? "component gate failed: " + component + "; matrix retained " + half : nil)
                            if floating, eligibleReason == nil {
                                report.componentParameters[component, default: 0] += raw.size
                                report.matrixParameters[matrix] = raw.size
                                report.matrixComponents[matrix] = component
                                if retained { report.fallbackParameters += raw.size }
                            }
                            if floating, eligibleReason == nil { report.linears += 1 }
                            if bits < 16, floating, reason == nil {
                                let prefix = mappedKey == "speaker_proj_weight" ? "speaker_proj" : String(mappedKey.dropLast(7))
                                let dense = pointwiseSpeaker(key: mappedKey, shape: value.shape) ? value.squeezed(axis: 1) : value
                                let (packed, scales, biases) = MLX.quantized(dense, groupSize: 64, bits: bits, mode: .affine, stream: .gpu)
                                guard let biases else { throw EngineError("Missing affine biases") }
                                eval(packed, scales, biases)
                                report.matrixScalesSHA256[matrix] = SHA256.hash(data: scales.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
                                if bits == 8, report.packingAudit[relative + ":" + mappedKey] == nil {
                                    report.packingAudit[relative + ":" + mappedKey] = try packingAudit(dense: dense, packed: packed, scales: scales, biases: biases)
                                }
                                try writer.append(prefix + ".weight", packed); try writer.append(prefix + ".scales", scales); try writer.append(prefix + ".biases", biases)
                                report.quantized.append(relative + ":" + prefix)
                                report.quantizedParameters += raw.size
                            } else {
                                eval(value)
                                try writer.append(mappedKey, value)
                                report.kept[relative + ":" + mappedKey] = floating ? reason ?? "selected half tier" : "integer/bool buffer"
                            }
                        }
                    }
                    Memory.clearCache()
                }
                try writer.finish()
                report.files.append(.init(source: relative, output: relative))
            }
            var config = try JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as! [String: Any]
            guard config["quantization"] == nil, config["quantization_config"] == nil else { throw EngineError("Source config is already quantized") }
            config["arsenal_voice_intake"] = ["recipe": report.rule.id, "half": half, "bits": bits, "group_size": 64, "source_truncation": truncateSource]
            if bits < 16 { config["quantization"] = ["group_size": 64, "bits": bits, "mode": "affine"] }
            try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(to: output.appendingPathComponent("config.json"))
            report.files.append(.init(source: "config.json", output: "config.json"))
            // Companion arrays and split batched heads are real stored keys; never leave a vendor index stale.
            for relative in files where relative.hasSuffix(".safetensors.index.json") {
                let file = output.appendingPathComponent(relative)
                var index = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
                let old = index["weight_map"] as? [String: String] ?? [:]
                var mapped: [String: String] = [:]
                for shard in Set(old.values) {
                    let shardURL = file.deletingLastPathComponent().appendingPathComponent(shard)
                    for key in try TensorArchive(shardURL).entries.keys { mapped[key] = shard }
                }
                index["weight_map"] = mapped
                index["metadata"] = [:] as [String: String]
                try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys]).write(to: file, options: .atomic)
            }
            guard keeping.filter({ !$0.hasPrefix("matrix@asset-kokoro-g2p/us_bart.safetensors:") }).isSubset(of: Set(report.componentParameters.keys).union(report.matrixParameters.keys)) else { throw EngineError("Fallback names a component without eligible weights") }
            report.quantized.sort()
            return report
        } catch {
            // Only this invocation's fresh destination is owned here.
            try? fm.removeItem(at: output)
            throw error
        }
    }
}

public extension WholeVoiceConversion {
    /// First four rows per eligible matrix, all input groups. Independent scalar unpack reference;
    /// no FP32 learned model weights or model execution. Host doubles are statistics only.
    static func packingAudit(dense: MLXArray, packed: MLXArray, scales: MLXArray, biases: MLXArray) throws -> [String: Double] {
        try StoredVoiceWeights.affinePackingAudit(dense: dense, packed: packed, scales: scales, biases: biases)
    }

}

public extension Engines {
    static func convertWholeVoice(_ source: ModelSource, files: [String], dtype: String, half: String, keeping: Set<String> = [], truncateSource: Bool = false, matrixPolicy: Bool = false, into output: URL) throws -> QuantReport {
        try Device.withDefaultDevice(.gpu) {
            try WholeVoiceConversion.convert(source: source.directory, files: files, model: source.id, dtype: dtype, half: half, keeping: keeping, truncateSource: truncateSource, matrixPolicy: matrixPolicy, into: output)
        }
    }
}
