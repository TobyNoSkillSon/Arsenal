// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXLMCommon
import ArsenalWire

/// Per-generation decoding settings from the request and the model's `runners/native/language.json`:
/// - `sampling`: the maker's sampled settings. `temperature` is the default when the request gives none; `top_p`,
///   `top_k`, `min_p` and the penalties in this block apply only when the effective temperature is above zero.
///   A nested `thinking` object overrides these keys when thinking is on.
/// - `penalties`: maker penalties applied to every generation, greedy included, keyed by a task kind
///   (`default` otherwise), e.g. TeleOCR's per-block presence/frequency settings and its client's
///   `no_repeat_ngram_size`.
struct LanguageDecoding: Equatable {
    var temperature: Float = 0
    var topP: Float = 1
    var topK = 0
    var minP: Float = 0
    var repetition: Float?
    var presence: Float?
    var frequency: Float?
    var noRepeatNgram: Int?

    static func make(spec: JSONObject, input: JSONObject, thinking: Bool, kind: String? = nil, repetitionFloor: Float? = nil) -> LanguageDecoding {
        var value = LanguageDecoding()
        var sampling = spec["sampling"]?.object ?? [:]
        if thinking, let overrides = sampling["thinking"]?.object { sampling.merge(overrides) { _, override in override } }
        let number = { (object: JSONObject, key: String) in object[key]?.number.map(Float.init) }
        value.temperature = input["temperature"]?.number.map(Float.init) ?? number(sampling, "temperature") ?? 0
        if value.temperature > 0 {
            value.topP = number(sampling, "top_p") ?? 1
            value.topK = Int(sampling["top_k"]?.number ?? 0)
            value.minP = number(sampling, "min_p") ?? 0
            value.repetition = number(sampling, "repetition_penalty")
            value.presence = number(sampling, "presence_penalty")
            value.frequency = number(sampling, "frequency_penalty")
        }
        if let penalties = spec["penalties"]?.object, let entry = (kind.flatMap { penalties[$0]?.object }) ?? penalties["default"]?.object {
            value.repetition = number(entry, "repetition_penalty") ?? value.repetition
            value.presence = number(entry, "presence_penalty") ?? value.presence
            value.frequency = number(entry, "frequency_penalty") ?? value.frequency
            value.noRepeatNgram = entry["no_repeat_ngram_size"]?.number.map { Int($0) } ?? value.noRepeatNgram
        }
        if let repetitionFloor { value.repetition = max(value.repetition ?? 1, repetitionFloor) }
        if value.repetition == 1 { value.repetition = nil }
        if value.presence == 0 { value.presence = nil }
        if value.frequency == 0 { value.frequency = nil }
        if (value.noRepeatNgram ?? 0) < 2 { value.noRepeatNgram = nil }
        return value
    }
    func sampler() -> LogitSampler {
        GenerateParameters(temperature: temperature, topP: topP, topK: topK, minP: minP).sampler()
    }
    /// Repetition penalty over the prompt and the output (Transformers and vLLM); presence and frequency penalties and
    /// the n-gram block over generated tokens only (vLLM, the TeleOCR and Qwen maker servers).
    func processor(promptTokens: Int, maxTokens: Int) -> LogitProcessor? {
        guard repetition != nil || presence != nil || frequency != nil || noRepeatNgram != nil else { return nil }
        return OutputPenalties(
            repetition: repetition.map { RepetitionContext(repetitionPenalty: $0, repetitionContextSize: max(1, promptTokens + maxTokens)) },
            presence: presence.map { PresencePenaltyContext(presencePenalty: $0, presenceContextSize: max(1, maxTokens)) },
            frequency: frequency.map { FrequencyPenaltyContext(frequencyPenalty: $0, frequencyContextSize: max(1, maxTokens)) },
            ngram: noRepeatNgram.map { NoRepeatNGram(size: $0, capacity: max(1, maxTokens)) })
    }
}

/// mlx-swift-lm's penalty contexts, with presence and frequency counting only generated tokens.
struct OutputPenalties: LogitProcessor {
    var repetition: RepetitionContext?
    var presence: PresencePenaltyContext?
    var frequency: FrequencyPenaltyContext?
    var ngram: NoRepeatNGram?

    mutating func prompt(_ prompt: MLXArray) { repetition?.prompt(prompt) }
    func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        if let repetition { logits = repetition.process(logits: logits) }
        if let presence { logits = presence.process(logits: logits) }
        if let frequency { logits = frequency.process(logits: logits) }
        if let ngram { logits = ngram.process(logits: logits) }
        return logits
    }
    mutating func didSample(token: MLXArray) {
        repetition?.didSample(token: token)
        presence?.didSample(token: token)
        frequency?.didSample(token: token)
        ngram?.didSample(token: token)
    }
}

/// MinerU's `no_repeat_ngram_size` logits processor (mineru-vl-utils `vllm_v1_no_repeat_ngram.py`), which the TeleOCR
/// client enables with size 100: over generated tokens only, a token that would complete an n-gram already generated
/// is banned. It stays on the GPU, so decoding keeps its asynchronous pipeline: `run[i]` is the length of the common
/// suffix of the output up to position i and the whole output, updated with fixed-size array operations per token.
struct NoRepeatNGram {
    let size: Int
    let capacity: Int
    private(set) var count = 0
    private var tokens: MLXArray
    private var run: MLXArray
    private let positions: MLXArray

    init(size: Int, capacity: Int) {
        precondition(size >= 2 && capacity > 0)
        self.size = size; self.capacity = capacity
        tokens = MLXArray.zeros([capacity], type: Int32.self)
        run = MLXArray.zeros([capacity], type: Int32.self)
        positions = MLXArray(0..<capacity).asType(.int32)
    }
    /// `logits` [1, vocabulary]: the next token of every earlier occurrence of the last n−1 generated tokens gets −∞.
    func process(logits: MLXArray) -> MLXArray {
        guard count >= size else { return logits }
        let next = concatenated([tokens[1...], MLXArray.zeros([1], type: Int32.self)])
        let penalty = MLX.where(run .>= Int32(size - 1), MLXArray(-Float.infinity), MLXArray(Float(0))).asType(logits.dtype)
        let ban = MLXArray.zeros([logits.dim(-1)], dtype: logits.dtype).at[next].add(penalty)
        return logits + ban.reshaped(1, -1)
    }
    mutating func didSample(token: MLXArray) {
        guard count < capacity else { return }
        let value = token.reshaped(-1)[0].asType(.int32)
        // run'[i] = run[i−1] + 1 where tokens[i] == the new token (i before it), else 0.
        let shifted = concatenated([MLXArray.zeros([1], type: Int32.self), run[..<(capacity - 1)]])
        run = MLX.where((tokens .== value) .&& (positions .< Int32(count)), shifted + 1, MLXArray(Int32(0)))
        tokens = MLX.where(positions .== Int32(count), value, tokens)
        count += 1
    }
}
