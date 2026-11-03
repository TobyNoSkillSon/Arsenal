// SPDX-License-Identifier: MIT
import Foundation
import Hub
import IntegrationKit
import MLX
import ArsenalWire
import Tokenizers

/// Qwen's release stores vocab/merges instead of tokenizer.json. Build the same tokenizer in memory.
func qwenTokenizer(_ directory: URL) throws -> any Tokenizer {
    if FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path) { return try localTokenizer(directory) }
    let configData = try Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))
    let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] ?? [:]
    let vocab = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
    let merges = try String(contentsOf: directory.appendingPathComponent("merges.txt"), encoding: .utf8)
        .components(separatedBy: "\n").filter { !$0.hasPrefix("#") && !$0.isEmpty }
    let added = (config["added_tokens_decoder"] as? [String: [String: Any]] ?? [:]).compactMap { key, value -> [String: Any]? in
        guard let id = Int(key) else { return nil }
        var item = value; item["id"] = id
        return item
    }.sorted { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
    // Qwen2Tokenizer's PRETOKENIZE_REGEX (transformers tokenization_qwen2.py; Qwen3-ASR names Qwen2Tokenizer): one digit per piece.
    let pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    let object: [String: Any] = [
        "version": "1.0", "added_tokens": added, "normalizer": ["type": "NFC"],
        "pre_tokenizer": [
            "type": "Sequence",
            "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": pattern], "behavior": "Isolated", "invert": false],
                ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]
            ]
        ],
        "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
        "model": ["type": "BPE", "vocab": vocab, "merges": merges, "fuse_unk": false, "byte_fallback": false]
    ]
    return try AutoTokenizer.from(
        tokenizerConfig: JSONDecoder().decode(Config.self, from: configData),
        tokenizerData: JSONDecoder().decode(Config.self, from: JSONSerialization.data(withJSONObject: object)), strict: true)
}

/// Whether a model.toml declares `[[op]] name = "<name>"`.
func manifestDeclaresOperation(_ manifest: String, _ name: String) -> Bool {
    var inOperation = false
    for line in manifest.split(separator: "\n", omittingEmptySubsequences: false) {
        let text = line.trimmingCharacters(in: .whitespaces)
        if text.hasPrefix("[") { inOperation = text == "[[op]]"; continue }
        if inOperation, text.replacingOccurrences(of: " ", with: "") == "name=\"\(name)\"" { return true }
    }
    return false
}

final class QwenASRAdapter: AudioAdapter {
    override var architecture: String { "qwen3_asr" }
    override var operations: Set<String> { streamingEnabled ? ["transcribe", "transcribe_session"] : ["transcribe"] }
    var tokenLimit = 8192
    private var streamingEnabled = false
    private var sessions: [String: ConfuciusSession] = [:]
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        // The verified folder, not its directory name, declares whether this checkpoint streams.
        streamingEnabled = try manifestDeclaresOperation(
            String(contentsOf: context.folder.appendingPathComponent("model.toml"), encoding: .utf8), "transcribe_session")
        let configuration = try config(context, as: Qwen3ASRConfig.self)
        let model = Qwen3ASRModel(configuration)
        tokenLimit = context.randomWeights ? 3 : 8192
        model.tokenizer = try qwenTokenizer(context.randomWeights ? context.folder.appendingPathComponent("gate/tokenizer") : context.weights)
        try prepare(
            model, context, cancellation: cancellation,
            map: { key, value in
                let original = key.hasPrefix("thinker.")
                let name = original ? String(key.dropFirst(8)) : key
                if name == "lm_head.weight", configuration.textConfig.tieWordEmbeddings { return nil }
                let tensor = original && name.contains("conv2d") && name.hasSuffix("weight") && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
                return (name, tensor)
            })
    }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        if operation == "transcribe_session" { return try session(input, cancellation: cancellation) }
        try audioKeys(input, allowed: ["audio", "language", "task", "prompt", "stream"])
        guard operation == "transcribe", let model = graph as? Qwen3ASRModel, model.tokenizer != nil else { throw audioFailure("ASR is not loaded.") }
        guard input["task"] == nil || input["task"]?.string == "transcribe", input["word_timestamps"]?.bool != true, input["diarize"]?.bool != true else {
            throw audioFailure("This checkpoint transcribes speech; alignment and diarization require separate models.")
        }
        try cancellation.check()
        let pcm = try audioInput(input, rate: 16000)
        let chunks = splitAudioIntoChunks(MLXArray(pcm.samples), sampleRate: 16000, chunkDuration: 30, minChunkDuration: 1)
        var texts: [String] = [], languages: [String?] = [], segments: [JSONValue] = []
        for (chunk, offset) in chunks {
            try cancellation.check()
            let decoded = try qwenDecode(
                model, audio: chunk, context: input["prompt"]?.string ?? "", language: input["language"]?.string, tokenLimit: tokenLimit, cancellation: cancellation)
            let parsed = model.parseGeneratedChunk(decoded, forcedLanguage: input["language"]?.string)
            texts.append(parsed.text); languages.append(parsed.language)
            segments.append([
                "start": .number(Double(offset)), "end": .number(min(Double(pcm.samples.count) / 16000, Double(offset) + Double(chunk.size) / 16000)), "text": .string(parsed.text)
            ])
            if input["stream"]?.bool == true { try emit(.json(["event": "token", "text": .string(parsed.text)])) }
        }
        var output: JSONObject = ["text": .string(texts.joined(separator: " ")), "duration": .number(Double(pcm.samples.count) / 16000), "segments": .array(segments)]
        if let language = dominantAudioLanguage(languages) { output["language"] = .string(language) }
        try cancellation.check()
        return output
    }
    override func unload() { sessions.removeAll(); streamingEnabled = false; super.unload() }
    private func session(_ input: JSONObject, cancellation: NativeCancellation) throws -> JSONObject {
        try audioKeys(input, allowed: ["action", "session_id", "audio", "prompt", "language", "chunk_size_ms", "rollback_tokens"])
        guard streamingEnabled, let model = graph as? Qwen3ASRModel, let action = input["action"]?.string else {
            throw audioFailure("This operation needs the loaded Confucius streaming model.")
        }
        try cancellation.check()
        if action == "create" {
            let id = input["session_id"]?.string ?? UUID().uuidString.lowercased()
            let chunk = try audioInteger(input, key: "chunk_size_ms", fallback: 160, range: 80...2000),
                rollback = try audioInteger(input, key: "rollback_tokens", fallback: 1, range: 0...128)
            guard !id.isEmpty, id.count <= 128, sessions[id] == nil, sessions.count < 2, (80...2000).contains(chunk), (0...128).contains(rollback), input["audio"] == nil else {
                throw audioFailure("Use a unique session ID, 80...2000 ms chunks and 0...128 rollback tokens; at most two sessions can remain open.")
            }
            sessions[id] = ConfuciusSession(
                chunkMS: chunk, rollback: rollback, prompt: input["prompt"]?.string ?? "", language: model.normalizeLanguageName(input["language"]?.string))
            return ["session_id": .string(id), "text": "", "partial_text": "", "delta": "", "duration": 0, "done": false]
        }
        guard let id = input["session_id"]?.string, let state = sessions[id] else { throw audioFailure("The transcription session is closed or belongs to an unloaded model.") }
        guard input["prompt"] == nil, input["language"] == nil, input["chunk_size_ms"] == nil, input["rollback_tokens"] == nil else {
            throw audioFailure("Stream settings are set at create.")
        }
        let previous = state.committed
        do {
            if action == "push" {
                let pcm = try audioInput(input, rate: 16000, seconds: 30)
                try state.push(pcm.samples, model: model, tokenLimit: min(tokenLimit, 512), cancellation: cancellation)
            } else if action == "finish", input["audio"] == nil {
                try state.finish(model: model, tokenLimit: min(tokenLimit, 512), cancellation: cancellation)
            } else if action == "close", input["audio"] == nil {
                sessions.removeValue(forKey: id)
            } else {
                throw audioFailure("Use create, push, finish or close for a transcription stream.")
            }
        } catch { sessions.removeValue(forKey: id); throw error }
        if action == "finish" { sessions.removeValue(forKey: id) }
        var output: JSONObject = [
            "session_id": .string(id), "text": .string(state.committed), "partial_text": .string(state.hypothesis),
            "delta": .string(String(state.committed.dropFirst(previous.count))), "duration": .number(Double(state.totalSamples) / 16000),
            "done": .bool(action == "finish" || action == "close")
        ]
        if let language = audioLanguageCode(state.language) { output["language"] = .string(language) }
        try cancellation.check()
        return output
    }

}
