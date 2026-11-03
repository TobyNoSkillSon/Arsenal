//
//  Planner.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The 5 Hz language model (Qwen3 1.7B, acestep-5Hz-lm-1.7B) in its two
//  phases, as upstream's `_run_mlx_single_native` runs them: phase 1 writes
//  the reasoning block under the metadata constraint (no CFG), phase 2
//  writes duration × 5 audio codes with classifier-free guidance against
//  the "NO USER INPUT" prompt. Prefill is all but the last prompt token,
//  then the last on its own, as upstream does; each step is
//  logits → constraint → log-softmax → top-p → top-k → categorical.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Hub
import Tokenizers
import IntegrationKit

// The planner's knobs (upstream's `lm_*` / `use_cot_*` parameters).
struct PlannerSettings: Sendable, Equatable {
    var temperature: Float = 0.85
    var cfgScale: Float = 2.0
    var topK: Int?
    var topP: Float? = 0.9
    var repetitionPenalty: Float = 1.0
    var negativePrompt = PlannerText.noUserInput
    var useCotMetas = true
    var useCotCaption = true
    var useCotLanguage = true
    var constrainedDecoding = true
}

// What the planner wrote.
struct PlannerOutput: Sendable {
    var metadata: PlannedMetadata
    var reasoningText: String
    var codes: [Int]
    var reasoningSeconds: Double
    var codesSeconds: Double
    var reasoningTokens: Int
}

// The 5 Hz LM with its tokenizer and constraint vocabulary.
final class Planner {
    let model: Qwen3Model
    let tokenizer: any Tokenizers.Tokenizer
    let vocabulary: MetadataVocabulary
    // `[V]` additive mask: 0 on audio codes and EOS, -inf elsewhere.
    let codesAllowed: MLXArray

    var cancellation: NativeCancellation?
    init(model: Qwen3Model, tokenizer: any Tokenizers.Tokenizer, codeIDs: [String: Int]) {
        self.model = model; self.tokenizer = tokenizer
        let size = model.vocabularySize
        let eos = tokenizer.convertTokenToId("<|im_end|>") ?? 151_645
        vocabulary = MetadataVocabulary(
            vocabularySize: size, eos: eos, audioCodeIDs: codeIDs,
            encode: { tokenizer.encode(text: $0, addSpecialTokens: false) }, decode: { tokenizer.decode(tokens: $0, skipSpecialTokens: false) })
        var mask = [Float](repeating: -.infinity, count: size)
        for id in vocabulary.audioCodes { mask[id] = 0 }
        mask[eos] = 0
        codesAllowed = MLXArray(mask, [1, size]).asType(.float16)
    }

    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    // MARK: Phases

    // Phase 1: the reasoning block. `fixed` holds what the caller set (injected, not generated).
    func reason(
        caption: String, lyrics: String, fixed: MetadataConstraint.Fixed, settings: PlannerSettings,
        random: MLXRandom.RandomState
    ) throws(ACEStepError) -> (PlannedMetadata, String, Int) {
        let prompt = PlannerText.reasoningPrompt(caption: caption, lyrics: lyrics)
        let constraint = MetadataConstraint(
            vocabulary: vocabulary, enabled: settings.constrainedDecoding, fixed: fixed,
            skipCaption: !settings.useCotCaption, skipLanguage: !settings.useCotLanguage,
            stopAtReasoning: true, targetDuration: nil)
        let maxTokens = MetadataLimits.duration.upperBound * 5 + 500
        let tokens = try generate(
            prompt: encode(prompt), unconditional: nil, cfgScale: 1, settings: settings,
            constraint: constraint, maxTokens: maxTokens, codes: false, random: random, progress: nil)
        let text = tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
        return (PlannerText.parse(text).0, text, tokens.count)
    }

    // Phase 2: exactly `duration × 5` audio codes after the reasoning block.
    func codes(
        caption: String, lyrics: String, reasoning: String, duration: Double, settings: PlannerSettings,
        random: MLXRandom.RandomState, progress: ((Int, Int) -> Void)?
    ) throws(ACEStepError) -> [Int] {
        let prompt = PlannerText.codesPrompt(caption: caption, lyrics: lyrics, reasoning: reasoning)
        let unconditional = PlannerText.codesUnconditional(negative: settings.negativePrompt)
        let constraint = MetadataConstraint(
            vocabulary: vocabulary, enabled: settings.constrainedDecoding,
            fixed: .init(), skipCaption: true, skipLanguage: true,
            stopAtReasoning: false, targetDuration: duration)
        constraint.beginCodes()
        let clamped = Double(min(max(Int(duration), MetadataLimits.duration.lowerBound), MetadataLimits.duration.upperBound))
        let maxTokens = Int(clamped * 5) + 10
        let encodeStart = Date()
        let promptIDs = encode(prompt)
        let unconditionalIDs = settings.cfgScale > 1 ? encode(unconditional) : nil
        if ProcessInfo.processInfo.environment["ACESTEP_PROFILE"] != nil {
            Profile.log(String(format: "PROFILE encode %.2f s", Date().timeIntervalSince(encodeStart)))
        }
        let tokens = try generate(
            prompt: promptIDs, unconditional: unconditionalIDs,
            cfgScale: settings.cfgScale, settings: settings, constraint: constraint,
            maxTokens: maxTokens, codes: true, random: random, progress: progress)
        return tokens.compactMap { vocabulary.codeValue[$0] }
    }

    // MARK: Decoding loop

    private func prefill(_ ids: [Int], cache: [KVCache]) throws(ACEStepError) -> MLXArray {
        let all = MLXArray(ids.map(Int32.init))
        if ids.count > 1 {
            var start = 0
            while ids.count - start > 1 {
                if (try? cancellation?.check()) == nil && cancellation != nil { throw .cancelled }
                let chunk = min(2048, ids.count - start - 1)
                _ = model(all[start..<(start + chunk)].expandedDimensions(axis: 0), cache: cache)
                MLX.eval(cache.flatMap { $0.innerState() })
                start += chunk
            }
            return model(all[(ids.count - 1)...].expandedDimensions(axis: 0), cache: cache)[0..., -1, 0...]
        }
        return model(all.expandedDimensions(axis: 0), cache: cache)[0..., -1, 0...]
    }

    // Tokens sampled where a forced sequence was fed instead (parity tests): (sampled, forced) pairs.
    var teacherLog: [(Int, Int)] = []

    func generate(
        prompt: [Int], unconditional: [Int]?, cfgScale: Float, settings: PlannerSettings,
        constraint: MetadataConstraint, maxTokens: Int, codes: Bool,
        random: MLXRandom.RandomState, progress: ((Int, Int) -> Void)?, forced: [Int]? = nil
    ) throws(ACEStepError) -> [Int] {
        let cache = model.newCache(parameters: nil)
        let uncache = unconditional != nil ? (model.newCache(parameters: nil)) : []
        let prefillStart = Date()
        var last = try prefill(prompt, cache: cache)
        var lastUncond: MLXArray?
        if let unconditional { lastUncond = try prefill(unconditional, cache: uncache) }
        if ProcessInfo.processInfo.environment["ACESTEP_PROFILE"] != nil {
            MLX.eval([last] + (lastUncond.map { [$0] } ?? []))
            Profile.log(String(format: "PROFILE prefill %.2f s", Date().timeIntervalSince(prefillStart)))
        }
        var history = prompt
        var produced: [Int] = []
        let eos = vocabulary.eos
        let filter = TokenSampling.filter(temperature: settings.temperature, topP: settings.topP, topK: settings.topK)
        let profile = ProcessInfo.processInfo.environment["ACESTEP_PROFILE"] != nil
        var tForward = 0.0, tConstrain = 0.0, tSample = 0.0
        var mark = Date()
        func lap(_ into: inout Double, _ arrays: [MLXArray]) {
            guard profile else { return }
            MLX.eval(arrays)
            into += Date().timeIntervalSince(mark)
            mark = Date()
        }
        let loopStart = Date()
        defer {
            if profile {
                // swiftlint:disable:next trailing_semicolon
                Profile.log(String(format: "PROFILE loop %.2f s", Date().timeIntervalSince(loopStart)));
                Profile.log(String(format: "PROFILE forward %.2f constrain %.2f sample %.2f (%d tokens)", tForward, tConstrain, tSample, produced.count))
            }
        }
        for step in 0..<maxTokens {
            if (try? cancellation?.check()) == nil && cancellation != nil { throw .cancelled }
            lap(&tForward, [last] + (lastUncond.map { [$0] } ?? []))
            var logits = (lastUncond.map { $0 + cfgScale * (last - $0) } ?? last).asType(.float32)
            if settings.repetitionPenalty != 1 {
                let index = MLXArray(history.map(Int32.init))
                let selected = logits[0..., index]
                logits[0..., index] = MLX.where(
                    selected .> 0, selected / settings.repetitionPenalty,
                    selected * settings.repetitionPenalty)
            }
            if settings.constrainedDecoding {
                logits = constraint.constrain(logits, allowedCodes: codesAllowed)
            }
            lap(&tConstrain, [logits])
            let token = TokenSampling.sample(logits, filter: filter, temperature: settings.temperature, random: random)
            MLX.eval(token)
            lap(&tSample, [token])
            var id = token.item(Int.self)
            if let forced {
                guard step < forced.count else { break }
                teacherLog.append((id, forced[step]))
                id = forced[step]
            }
            produced.append(id)
            history.append(id)
            constraint.update(with: id)
            progress?(produced.count, maxTokens)
            if id == eos { break }
            let input = MLXArray([Int32(id)]).reshaped(1, 1)
            last = model(input, cache: cache)[0..., -1, 0...]
            if unconditional != nil { lastUncond = model(input, cache: uncache)[0..., -1, 0...] }
            if step % 256 == 0 && step > 0 { MLX.Memory.clearCache() }
        }
        return produced
    }
}

// The planner's tokenizer without its 65,535 `<|audio_code_N|>` added tokens.
//
// swift-transformers splits text on one regular expression that alternates
// every added token; with the audio codes in it, encoding the codes prompt
// took 65 s (measured). The prompts never contain code text — codes come
// back as ids — so the tokenizer is built without them; every other id is
// unchanged.
enum PlannerTokenizer {

    static func json(_ url: URL) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw ACEStepError.modelMissing("\(url.lastPathComponent) is not a JSON object")
        }
        return object
    }

    static func isCode(_ content: String) -> Bool { content.hasPrefix("<|audio_code_") }

    static func load(_ folder: URL) throws -> any Tokenizers.Tokenizer {
        var data = try json(folder.appendingPathComponent("tokenizer.json"))
        if let added = data["added_tokens"] as? [[String: Any]] {
            data["added_tokens"] = added.filter { !isCode($0["content"] as? String ?? "") }
        }
        var config = try json(folder.appendingPathComponent("tokenizer_config.json"))
        if let decoder = config["added_tokens_decoder"] as? [String: [String: Any]] {
            config["added_tokens_decoder"] = decoder.filter { !isCode($0.value["content"] as? String ?? "") }
        }
        if let extra = config["additional_special_tokens"] as? [Any] {
            config["additional_special_tokens"] = extra.filter { !isCode(($0 as? String) ?? "") }
        }
        func nsDict(_ d: [String: Any]) -> [NSString: Any] { Dictionary(uniqueKeysWithValues: d.map { ($0.key as NSString, $0.value) }) }
        return try AutoTokenizer.from(tokenizerConfig: Config(nsDict(config)), tokenizerData: Config(nsDict(data)))
    }

    // `<|audio_code_N|>` → id, from added_tokens.json.
    static func audioCodeIDs(_ folder: URL) throws -> [String: Int] {
        let added = try json(folder.appendingPathComponent("added_tokens.json"))
        var result: [String: Int] = [:]
        for (content, id) in added where isCode(content) { result[content] = (id as? Int) ?? (id as? NSNumber)?.intValue }
        return result
    }
}
