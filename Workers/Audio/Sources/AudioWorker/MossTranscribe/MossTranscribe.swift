// SPDX-License-Identifier: MIT
// MOSS-Transcribe-Diarize 0.9B (OpenMOSS, Apache-2.0), ported to MLX from the maker's remote code in the pinned
// repository (modeling_moss_transcribe_diarize.py, processing_moss_transcribe_diarize.py) and its GitHub helpers
// (inference_utils.py, transcript_parser.py at 79c675f0): transformers' WhisperEncoder (medium size), a 4× time merge,
// the VQ adaptor and Qwen3-0.6B. The decoder is the audio family's Qwen3 text model (Qwen3ASRTextModel); the log-mel
// frontend is the family's Whisper filterbank (smartTurnLogMelSpectrogram).
import Foundation
import MLX
import MLXFast
import MLXNN
import Tokenizers

struct MossConfig {
    var text: Qwen3TextConfig
    var melBins = 80
    var width = 1024
    var layers = 24
    var heads = 16
    var ffn = 4096
    var positions = 1500
    var audioTokenID = 151671
    var mergeSize = 4
    var adaptorInput = 4096
    // processor_config.json and preprocessor_config.json
    var tokensPerSecond = 12.5
    var markerSeconds = 5
    var chunkSamples = 480_000
    var hop = 160
    var nFFT = 400

    static func load(_ directory: URL) throws -> MossConfig {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any],
            root["model_type"] as? String == "moss_transcribe_diarize", let text = root["text_config"], let audio = root["audio_config"] as? [String: Any]
        else { throw audioFailure("Not a MOSS-Transcribe-Diarize configuration.") }
        var config = MossConfig(text: try JSONDecoder().decode(Qwen3TextConfig.self, from: JSONSerialization.data(withJSONObject: text)))
        func int(_ table: [String: Any], _ key: String) -> Int? { (table[key] as? NSNumber)?.intValue }
        config.melBins = int(audio, "num_mel_bins") ?? config.melBins
        config.width = int(audio, "d_model") ?? config.width
        config.layers = int(audio, "encoder_layers") ?? config.layers
        config.heads = int(audio, "encoder_attention_heads") ?? config.heads
        config.ffn = int(audio, "encoder_ffn_dim") ?? config.ffn
        config.positions = int(audio, "max_source_positions") ?? config.positions
        config.audioTokenID = int(root, "audio_token_id") ?? config.audioTokenID
        config.mergeSize = int(root, "audio_merge_size") ?? config.mergeSize
        config.adaptorInput = int(root, "adaptor_input_dim") ?? config.adaptorInput
        let processor = directory.appendingPathComponent("processor_config.json")
        if let table = try? JSONSerialization.jsonObject(with: Data(contentsOf: processor)) as? [String: Any] {
            config.tokensPerSecond = (table["audio_tokens_per_second"] as? NSNumber)?.doubleValue ?? config.tokensPerSecond
            config.markerSeconds = (table["enable_time_marker"] as? Bool) == false ? 0 : int(table, "time_marker_every_seconds") ?? config.markerSeconds
        }
        guard (audio["activation_function"] as? String ?? "gelu") == "gelu", (audio["scale_embedding"] as? Bool) != true,
            config.adaptorInput == config.width * config.mergeSize, config.text.tieWordEmbeddings
        else { throw audioFailure("Unsupported MOSS-Transcribe-Diarize configuration.") }
        return config
    }
}

// MARK: - Whisper encoder (transformers WhisperEncoder)

final class MossWhisperAttention: Module {
    let heads: Int, headDim: Int
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var out: Linear
    init(width: Int, heads: Int) {
        self.heads = heads; headDim = width / heads
        _q.wrappedValue = Linear(width, width); _k.wrappedValue = Linear(width, width, bias: false)
        _v.wrappedValue = Linear(width, width); _out.wrappedValue = Linear(width, width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(b, n, heads, headDim).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(q(x)), keys: split(k(x)), values: split(v(x)), scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return out(attended.transposed(0, 2, 1, 3).reshaped(b, n, heads * headDim))
    }
}

final class MossWhisperLayer: Module {
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: MossWhisperAttention
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(_ config: MossConfig) {
        _attentionNorm.wrappedValue = LayerNorm(dimensions: config.width)
        _attention.wrappedValue = MossWhisperAttention(width: config.width, heads: config.heads)
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.width)
        _fc1.wrappedValue = Linear(config.width, config.ffn)
        _fc2.wrappedValue = Linear(config.ffn, config.width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(attentionNorm(x))
        return h + fc2(gelu(fc1(finalNorm(h))))
    }
}

final class MossWhisperEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var positions: Embedding
    @ModuleInfo(key: "layers") var layers: [MossWhisperLayer]
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    init(_ config: MossConfig) {
        _conv1.wrappedValue = Conv1d(inputChannels: config.melBins, outputChannels: config.width, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: config.width, outputChannels: config.width, kernelSize: 3, stride: 2, padding: 1)
        _positions.wrappedValue = Embedding(embeddingCount: config.positions, dimensions: config.width)
        _layers.wrappedValue = (0..<config.layers).map { _ in MossWhisperLayer(config) }
        _norm.wrappedValue = LayerNorm(dimensions: config.width)
    }
    /// Log-mel `[chunks, frames, mels]` (channels last) → `[chunks, frames / 2, width]`.
    func callAsFunction(_ features: MLXArray, check: () throws -> Void = {}) rethrows -> MLXArray {
        var x = gelu(conv2(gelu(conv1(features.asType(conv1.weight.dtype)))))
        x = positions.weight[..<x.dim(1)] + x
        for layer in layers {
            x = layer(x)
            try check()
        }
        return norm(x)
    }
}

/// `Linear(in → hidden) → SiLU → Linear(hidden → hidden) → LayerNorm` (layers.0, .2 and .3 in the checkpoint).
final class MossVQAdaptor: Module {
    @ModuleInfo(key: "layers") var layers: [Module]
    init(_ config: MossConfig) {
        _layers.wrappedValue = [
            Linear(config.adaptorInput, config.text.hiddenSize), SiLU(), Linear(config.text.hiddenSize, config.text.hiddenSize),
            LayerNorm(dimensions: config.text.hiddenSize, eps: config.text.rmsNormEps)
        ]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for case let layer as UnaryLayer in layers { h = layer(h) }
        return h
    }
}

final class MossModel: Module {
    let config: MossConfig
    @ModuleInfo(key: "language_model") var language: Qwen3ASRTextModel
    @ModuleInfo(key: "whisper_encoder") var whisper: MossWhisperEncoder
    @ModuleInfo(key: "vq_adaptor") var adaptor: MossVQAdaptor
    init(_ config: MossConfig) {
        self.config = config
        _language.wrappedValue = Qwen3ASRTextModel(config.text)
        _whisper.wrappedValue = MossWhisperEncoder(config)
        _adaptor.wrappedValue = MossVQAdaptor(config)
    }
}

final class MossTranscriber: Module {
    @ModuleInfo(key: "model") var model: MossModel
    init(_ config: MossConfig) { _model.wrappedValue = MossModel(config) }
}

// MARK: - Processor (processing_moss_transcribe_diarize.py)

enum MossProcessor {
    static let defaultPrompt = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
    static let audioStart = 151669, audioEnd = 151670, imEnd = 151645, endOfText = 151643

    /// Tokens of one chunk: `(samples - 1) // (hop · 2 · merge) + 1`.
    static func tokenCount(samples: Int, config: MossConfig) -> Int { (samples - 1) / (config.hop * 2 * config.mergeSize) + 1 }

    /// The audio span: placeholders with the seconds written as digit tokens every `markerSeconds` (`_audio_span_ids`).
    static func span(tokens: Int, config: MossConfig, digits: [Character: Int]) -> [Int] {
        let placeholder = config.audioTokenID
        guard config.markerSeconds > 0, tokens > 0 else { return Array(repeating: placeholder, count: max(tokens, 0)) }
        let perMarker = Int(config.tokensPerSecond * Double(config.markerSeconds))
        guard perMarker > 0 else { return Array(repeating: placeholder, count: tokens) }
        let duration = Double(tokens) / config.tokensPerSecond
        var output: [Int] = [], consumed = 0
        for second in stride(from: config.markerSeconds, through: Int(duration), by: config.markerSeconds) {
            let position = (second / config.markerSeconds) * perMarker
            if position - consumed > 0 {
                output += Array(repeating: placeholder, count: position - consumed)
                consumed = position
            }
            output += String(second).map { digits[$0]! }
        }
        if tokens - consumed > 0 { output += Array(repeating: placeholder, count: tokens - consumed) }
        return output
    }

    /// The chat template with one audio item and the instruction, then the generation prompt.
    static func prompt(instruction: String, tokens: Int, tokenizer: any Tokenizers.Tokenizer, config: MossConfig, digits: [Character: Int]) -> [Int] {
        let before = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|>"
        let after = "<|audio_end|>\n" + instruction + "<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer.encode(text: before, addSpecialTokens: false) + span(tokens: tokens, config: config, digits: digits)
            + tokenizer.encode(text: after, addSpecialTokens: false)
    }

    /// The instruction: the maker's default (or the caller's own), with the maker's hotword hint appended.
    /// The caller's own instruction, or nil when the prompt is missing, empty or whitespace (the default instruction runs).
    static func ownPrompt(_ prompt: String?) -> String? {
        (prompt?.trimmingCharacters(in: .whitespacesAndNewlines)).flatMap { $0.isEmpty ? nil : $0 }
    }

    static func instruction(prompt: String?, hotwords: [String]) -> String {
        let own = ownPrompt(prompt)
        let base = own ?? defaultPrompt
        guard !hotwords.isEmpty else { return base }
        // The maker's Chinese hint follows its default instruction; an own instruction gets the English one.
        return own == nil ? base + "热词提示：" + hotwords.joined(separator: ", ") : base + " Hotwords: " + hotwords.joined(separator: ", ")
    }
}
