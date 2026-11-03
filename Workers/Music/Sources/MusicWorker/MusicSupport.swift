// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import ArsenalWire
import MLX
import MLXNN
import MLXLLM
import Hub
import Tokenizers

enum MusicSupport {
    static func failure(_ message: String) -> WorkerFailure { .init(code: "invalid_request", message: message) }
    /// A progress event the service shows in Activity, tool windows and job records: the stage and its percentage as
    /// the label (stages report fractions, not countable steps, so no step/total is invented), the fraction as
    /// `progress`, and the original stage/fraction/item fields.
    static func progress(_ progress: SongProgress, batch: Int) -> WireFrame {
        let names: [SongProgress.Stage: String] = [
            .loading: "Loading", .planning: "Planning the score", .codes: "Writing song tokens", .conditioning: "Conditioning",
            .diffusion: "Rendering audio", .decoding: "Decoding audio"
        ]
        let fraction = Swift.min(1, Swift.max(0, progress.fraction.isFinite ? progress.fraction : 0))
        var label = (names[progress.stage] ?? progress.stage.rawValue) + " \(Int((fraction * 100).rounded()))%"
        if batch > 1 { label = "Song \(progress.item + 1) of \(batch) \u{00b7} " + label }
        return .json([
            "event": "progress", "stage": .string(progress.stage.rawValue), "fraction": .number(progress.fraction), "item": .number(Double(progress.item)),
            "label": .string(label), "progress": .number(fraction)
        ])
    }
    static func wait<T>(_ body: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0); var result: Result<T, Error>!
        Task {
            do { result = .success(try await body()) } catch { result = .failure(error) }; semaphore.signal()
        }
        semaphore.wait(); return try result.get()
    }
    static func parameters(_ components: [(String, Module)]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (prefix, module) in components { for (name, value) in module.parameters().flattened() { result[prefix + name] = value } }
        return result
    }
    static func initialize(_ module: Module, precision: String) throws {
        try NativeParameters.initializeRandom(module)
        if let bits = try NativePrecisionTier(precision).bits {
            quantize(model: module, groupSize: 64, bits: bits) { path, layer in
                if let embedding = layer as? Embedding { return MusicConversion.eligible(path + ".weight", embedding.weight.shape) }
                if let linear = layer as? Linear { return MusicConversion.eligible(path + ".weight", linear.weight.shape) }
                return false
            }
        }
        module.train(false)
    }
    static func load(_ module: Module, file: URL, precision: String) throws {
        _ = try NativePrecisionTier(precision).load(module, archives: [file], map: { ($0, $1) }, eligible: MusicConversion.eligible)
    }
    static func qwenConfiguration(_ data: Data, small: Bool) throws -> MLXLLM.Qwen3Configuration {
        if small {
            return try JSONDecoder().decode(
                MLXLLM.Qwen3Configuration.self,
                from: JSONSerialization.data(withJSONObject: [
                    "hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                    "rms_norm_eps": 0.000001, "vocab_size": 1024, "tie_word_embeddings": true, "model_type": "qwen3"
                ]))
        }
        return try JSONDecoder().decode(MLXLLM.Qwen3Configuration.self, from: data)
    }
    /// Small valid byte-level BPE for random-weight graphs; serving always loads the pinned tokenizer.
    static func fixtureTokenizerFiles() -> (config: [String: Any], data: [String: Any]) {
        var alphabet = Array(33...126) + Array(161...172) + Array(174...255)
        var points = alphabet; var extra = 0
        for byte in 0..<256 where !alphabet.contains(byte) { alphabet.append(byte); points.append(256 + extra); extra += 1 }
        let vocab = Dictionary(uniqueKeysWithValues: zip(alphabet, points).map { (String(UnicodeScalar($0.1)!), $0.0) })
        let config: [String: Any] = ["tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>", "unk_token": "<|im_end|>"]
        let data: [String: Any] = [
            "version": "1.0", "model": ["type": "BPE", "vocab": vocab, "merges": []], "pre_tokenizer": ["type": "ByteLevel", "add_prefix_space": false, "use_regex": false],
            "decoder": ["type": "ByteLevel"],
            "added_tokens": [["id": 1023, "content": "<|im_end|>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false]]
        ]
        return (config, data)
    }
    static func fixtureTokenizer() throws -> any Tokenizers.Tokenizer {
        let (config, data) = fixtureTokenizerFiles()
        func dict(_ x: [String: Any]) -> [NSString: Any] { Dictionary(uniqueKeysWithValues: x.map { ($0.key as NSString, $0.value) }) }
        return try AutoTokenizer.from(tokenizerConfig: Config(dict(config)), tokenizerData: Config(dict(data)))
    }
}
