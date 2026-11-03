// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire

final class CohereAdapter: AudioAdapter {
    override var architecture: String { "cohere_asr" }
    override var operations: Set<String> { ["transcribe"] }
    var tokenizer: CohereTranscribeTokenizer?
    var tokenLimit = 8192
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let configuration = try config(context, as: CohereTranscribeConfig.self)
        tokenLimit = context.randomWeights ? 3 : configuration.decoder.maxSequenceLength
        tokenizer = try CohereTranscribeTokenizer(
            modelDir: context.randomWeights ? context.folder.appendingPathComponent("gate/tokenizer") : context.weights, config: configuration)
        try prepare(
            CohereTranscribeModel(configuration), context, cancellation: cancellation,
            map: { name, tensor in
                if name.hasSuffix(".num_batches_tracked") || name.hasPrefix("preprocessor.") || name == "decoder.embedding.position_embedding.pos_enc" { return nil }
                var mapped = name
                for index in [0, 2, 3, 5, 6] { mapped = mapped.replacingOccurrences(of: "encoder.subsampling.conv.\(index).", with: "encoder.subsampling.conv\(index).") }
                if tensor.ndim == 4 && name.hasSuffix("weight") { return (mapped, tensor.transposed(0, 2, 3, 1)) }
                if tensor.ndim == 3 && name.hasSuffix("weight") { return (mapped, tensor.transposed(0, 2, 1)) }
                return (mapped, tensor)
            })
    }
    override func unload() { super.unload(); tokenizer = nil }
    /// The Hugging Face feature extractor's split (transformers `CohereAsrFeatureExtractor._split_audio_chunks_energy`):
    /// audio up to max_audio_clip_s minus the 5 s boundary window is one chunk; longer audio is cut, chunk by chunk, at the
    /// quietest 1,600-sample window (lowest RMS) among the last 5 s before each max_audio_clip_s limit.
    static func chunks(_ samples: [Float], rate: Int, maximumSeconds: Double, boundarySeconds: Double = 5, window: Int = 1600) -> [Range<Int>] {
        let total = samples.count
        guard Double(total) / Double(rate) > max(0, maximumSeconds - boundarySeconds) else { return [0..<total] }
        let size = max(1, Int((maximumSeconds * Double(rate)).rounded())), boundary = max(1, Int((boundarySeconds * Double(rate)).rounded()))
        guard total > size else { return [0..<total] }
        var ranges: [Range<Int>] = [], index = 0
        while index < total {
            if index + size >= total { ranges.append(index..<total); break }
            let searchStart = max(index, index + size - boundary), searchEnd = min(index + size, total)
            var split = index + size
            if searchEnd > searchStart {
                if searchEnd - searchStart <= window {
                    split = (searchStart + searchEnd) / 2
                } else {
                    var lowest = Float.infinity
                    split = searchStart
                    for offset in stride(from: 0, to: searchEnd - searchStart - window, by: window) {
                        var sum: Float = 0
                        for sample in samples[(searchStart + offset)..<(searchStart + offset + window)] { sum += sample * sample }
                        let energy = (sum / Float(window)).squareRoot()
                        if energy < lowest { lowest = energy; split = searchStart + offset }
                    }
                }
            }
            split = max(index + 1, min(split, total))
            ranges.append(index..<split)
            index = split
        }
        return ranges.filter { !$0.isEmpty }
    }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "language", "task", "stream", "punctuation"])
        guard operation == "transcribe", let model = graph as? CohereTranscribeModel, let tokenizer else { throw audioFailure("Cohere is not loaded.") }
        guard input["task"] == nil || input["task"]?.string == "transcribe", input["word_timestamps"]?.bool != true, input["diarize"]?.bool != true else {
            throw audioFailure("Cohere adapter supports transcription without word alignment or diarization.")
        }
        if let punctuation = input["punctuation"], punctuation.bool == nil { throw audioFailure("punctuation must be true or false.") }
        let language = input["language"]?.string ?? "en"
        let pcm = try audioInput(input, rate: model.config.sampleRate)
        var texts: [String] = [], segments: [JSONValue] = []
        for range in Self.chunks(pcm.samples, rate: model.config.sampleRate, maximumSeconds: Double(model.config.maxAudioClipS)) {
            try cancellation.check()
            let start = range.lowerBound, end = range.upperBound
            let filters = CohereTranscribeAudio.computeMelFilters(sampleRate: pcm.sampleRate, numMels: model.config.encoder.featIn)
            let features = CohereTranscribeAudio.computeFeatures(audio: MLXArray(Array(pcm.samples[start..<end])), melFilters: filters).asType(.float16)
            let (encoded, _) = model.encoder(features)
            let memory = model.bridgeProj.map { $0(encoded) } ?? encoded
            let prompt = tokenizer.buildPromptTokens(language: language, usePunctuation: input["punctuation"]?.bool ?? true)
            guard !prompt.isEmpty else { throw audioFailure("Cohere tokenizer is missing prompt tokens.") }
            let initial = model.decoder(
                inputIds: MLXArray(prompt.map(Int32.init)).reshaped(1, -1), positions: MLXArray(0..<prompt.count).reshaped(1, -1), encoderHiddenStates: memory,
                selfAttentionMask: MultiHeadAttention.createAdditiveCausalMask(prompt.count).asType(.float16), crossAttentionMask: nil, cache: nil)
            var cache = initial.1, logits = model.lmHead(initial.0[0, -1]), generated: [Int] = []
            let eos = tokenizer.encode(text: "<|endoftext|>").first
            for index in 0..<min(tokenLimit, model.config.decoder.maxSequenceLength - prompt.count) {
                try cancellation.check()
                let token = logits.argMax().item(Int.self)
                if token == eos { break }
                generated.append(token)
                let next = model.decoder(
                    inputIds: MLXArray([Int32(token)]).reshaped(1, 1), positions: MLXArray([Int32(prompt.count + index)]).reshaped(1, 1), encoderHiddenStates: memory,
                    selfAttentionMask: nil, crossAttentionMask: nil, cache: cache)
                cache = next.1; logits = model.lmHead(next.0[0, -1])
            }
            let text = tokenizer.decode(tokens: generated)
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { texts.append(trimmed) }
            segments.append(["start": .number(Double(start) / Double(pcm.sampleRate)), "end": .number(Double(end) / Double(pcm.sampleRate)), "text": .string(text)])
            if input["stream"]?.bool == true { try emit(.json(["event": "token", "text": .string(text)])) }
        }
        // As the processor reassembles chunks: Japanese and Chinese join without a space.
        return [
            "text": .string(texts.joined(separator: ["ja", "zh"].contains(language) ? "" : " ")), "language": .string(audioLanguageCode(language) ?? "en"),
            "duration": .number(Double(pcm.samples.count) / Double(pcm.sampleRate)),
            "segments": .array(segments)
        ]
    }
}
