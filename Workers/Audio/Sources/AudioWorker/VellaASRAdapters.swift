// SPDX-License-Identifier: MIT
// Transcription with Vella's dictation runtime (Workers/Audio/Sources/VellaSTT, copied from Vella's worker): Parakeet
// TDT and Qwen3-ASR, with the optimized components Vella runs by default. Weights arrive through Arsenal's own
// precision tier. The shipped folders serve FP16 only and declare no derived tiers (see their READMEs);
// `vellaQwenEligible` (audio tower kept half, Vella's recipe) applies only to a folder that declares int8/int4.
import Foundation
import IntegrationKit
import Metal
import MLX
import MLXNN
import ArsenalWire
import Tokenizers
import VellaSTT

/// Vella runs its optimized paths only on Metal GPU family Apple 9 (`requiredGPUFamily` of both runtimes); other Macs
/// run stock MLX. `ARSENAL_ASR_STOCK=1` forces stock for a lab A/B.
enum VellaFastPath {
    static func available(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        environment["ARSENAL_ASR_STOCK"] != "1" && (MTLCreateSystemDefaultDevice()?.supportsFamily(.apple9) ?? false)
    }
}

/// Vella's runtime fallback (DictationService `infer`): when the optimized path throws, reports non-finite values or
/// returns an empty transcript that stock does not agree with, the request reruns on stock MLX and the model stays on
/// stock until it is reloaded. An empty optimized transcript that stock also returns empty keeps the optimized path.
final class VellaFallback {
    let component: String
    private(set) var active = false
    /// Why the model runs stock (nil while optimized).
    private(set) var stockReason: String? = "Not loaded."
    init(component: String) { self.component = component }

    /// `qualification`: the one-time self-test against stock (`VellaFastPathGate`) and its verdict key; nil skips it
    /// (a re-enable after both paths agreed at runtime).
    func enable(_ model: any FastPathCapable, allowed: Bool, qualification: (gate: VellaFastPathGate, key: String, name: String)? = nil) {
        active = false
        guard allowed else { stockReason = "This GPU runs the stock MLX path."; return }
        if let qualification, case .stock(let reason) = qualification.gate.verdict(model, component: component, key: qualification.key, name: qualification.name) {
            _ = try? withError { model.configureFastPath(enabled: false, component: component) }
            stockReason = reason; return
        }
        do {
            guard try withError({ model.configureFastPath(enabled: true, component: component) }) else {
                stockReason = "The optimized path does not support this checkpoint."; return
            }
            active = true; stockReason = nil
        } catch {
            _ = try? withError { model.configureFastPath(enabled: false, component: component) }
            stockReason = "The optimized path could not be configured."
        }
    }

    func reset() { active = false; stockReason = "Not loaded." }

    func run<T>(_ model: any FastPathCapable, text: (T) -> String, body: () throws -> T) throws -> T {
        guard active else { return try withError { try body() } }
        var checkingEmpty = false
        do {
            let output = try withError { try body() }
            guard model.fastPathFinite else { throw audioFailure("The optimized path produced non-finite values.") }
            guard text(output).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return output }
            checkingEmpty = true
        } catch let failure as WorkerFailure where failure.code == "cancelled" {
            throw failure
        } catch {}
        // Stock is authoritative from here, including an empty transcript.
        _ = try? withError { model.configureFastPath(enabled: false, component: component) }
        active = false
        let stock = try withError { try body() }
        if checkingEmpty, text(stock).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Both paths agree on silence; a failed reconfiguration leaves usable stock.
            enable(model, allowed: true)
        } else {
            stockReason = "Runtime fallback: optimized execution failed; using stock MLX."
        }
        return stock
    }
}

/// The self-test request for a Load: the verdict key covers the model's files, precision, component and the optimized
/// configuration the adapter chose (kept levers, disabled components). Random-weight lab graphs serve nothing and have
/// no transcript to compare, so they skip it.
func vellaQualification(_ context: NativeLoadContext, id: String, fallback: VellaFallback, model: any FastPathCapable)
    -> (gate: VellaFastPathGate, key: String, name: String)?
{
    guard !context.randomWeights else { return nil }
    let gate = VellaFastPathGate()
    let configuration =
        "disabled=" + model.fastPathDisabledComponents.sorted().joined(separator: ",")
        + ((model as? ParakeetModel).map { ";levers=" + $0.keptLevers.enabled.sorted().joined(separator: ",") } ?? "")
    return (gate, gate.key(model: id, precision: context.precision, weights: context.weights, component: fallback.component, configuration: configuration), id)
}

/// The folder's model id (`id = "…"` in its verified model.toml).
func manifestID(_ manifest: String) -> String? {
    for line in manifest.split(separator: "\n") {
        let text = line.trimmingCharacters(in: .whitespaces)
        if text.hasPrefix("[") { return nil }
        let parts = text.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
        if parts.count == 2, parts[0] == "id", parts[1].count >= 2, parts[1].hasPrefix("\""), parts[1].hasSuffix("\"") {
            return String(parts[1].dropFirst().dropLast())
        }
    }
    return nil
}

/// Runs an async tokenizer constructor on the engine's serialized thread, bounded like `localTokenizer`.
func blockingTokenizer(_ make: @escaping @Sendable () async throws -> any Tokenizer) throws -> any Tokenizer {
    final class Box: @unchecked Sendable { var result: Result<any Tokenizer, Error>? }
    let semaphore = DispatchSemaphore(value: 0), box = Box()
    Task.detached {
        do { box.result = .success(try await make()) } catch { box.result = .failure(error) }
        semaphore.signal()
    }
    guard semaphore.wait(timeout: .now() + 60) == .success, let result = box.result else { throw audioFailure("Local tokenizer initialization timed out.") }
    return try result.get()
}

/// Word timings from Parakeet's aligned sub-word tokens: a piece that starts with a space begins a new word.
func parakeetWords(_ tokens: [(text: String, start: Double, end: Double)]) -> [(word: String, start: Double, end: Double)] {
    var words: [(word: String, start: Double, end: Double)] = []
    for token in tokens {
        let piece = token.text
        if words.isEmpty || piece.hasPrefix(" ") {
            let word = piece.trimmingCharacters(in: .whitespaces)
            if word.isEmpty { continue }
            words.append((word, token.start, token.end))
        } else {
            words[words.count - 1].word += piece
            words[words.count - 1].end = max(words[words.count - 1].end, token.end)
        }
    }
    return words
}

private func seconds(_ value: Double, limit: Double) -> JSONValue { .number(min(max(value, 0), max(limit, 0))) }

// MARK: - Parakeet TDT

final class ParakeetAdapter: AudioAdapter {
    override var architecture: String { "parakeet_tdt" }
    override var operations: Set<String> { ["transcribe"] }
    let fallback = VellaFallback(component: "both")
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let id = try manifestID(String(contentsOf: context.folder.appendingPathComponent("model.toml"), encoding: .utf8))
        let model = try ParakeetModel.fromConfig(Data(contentsOf: context.weights.appendingPathComponent("config.json")))
        try prepare(model, context, cancellation: cancellation, map: { key, value in ParakeetModel.parameterName(checkpointKey: key).map { ($0, value) } })
        model.computeDType = .float16
        // Vella's kept levers: tail block sizing for Ultra only; the inexact GEMM components stay off (Optimized · Exact).
        model.keptLevers = KeptLevers(enabled: id == "parakeet-v3-ultra" ? ["VELLA_PARAKEET_TAILBLOCK"] : [])
        model.fastPathDisabledComponents = ["nax_gemm", "int8_gemm", "int4_gemm"]
        fallback.enable(model, allowed: VellaFastPath.available(), qualification: vellaQualification(context, id: id ?? "parakeet", fallback: fallback, model: model))
        try cancellation.check()
    }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "word_timestamps", "task"])
        guard operation == "transcribe", let model = graph as? ParakeetModel else { throw audioFailure("Parakeet is not loaded.") }
        guard input["task"] == nil || input["task"]?.string == "transcribe" else { throw audioFailure("Parakeet transcribes in the spoken language; it does not translate.") }
        if let flag = input["word_timestamps"], flag.bool == nil { throw audioFailure("word_timestamps must be a boolean.") }
        try cancellation.check()
        let pcm = try audioInput(input, rate: 16000)
        let duration = Double(pcm.samples.count) / 16000
        // Float32 samples: the log-mel's power spectrum can exceed FP16's range on loud audio (Vella computes it in BF16).
        let audio = MLXArray(pcm.samples)
        let parameters = STTGenerateParameters(maxTokens: 1024, verbose: false, chunkDuration: 30)
        let output = try fallback.run(
            model, text: { $0.text },
            body: { () -> (text: String, result: ParakeetAlignedResult?) in
                let text = model.generate(audio: audio, generationParameters: parameters).text
                return (text, model.lastResult)
            })
        try cancellation.check()
        let sentences = output.result?.sentences ?? []
        var result: JSONObject = [
            "text": .string(output.text.trimmingCharacters(in: .whitespacesAndNewlines)), "duration": .number(duration),
            "segments": .array(
                sentences.prefix(1024).map { sentence in
                    [
                        "start": seconds(sentence.start, limit: duration), "end": seconds(sentence.end, limit: duration),
                        "text": .string(sentence.text.trimmingCharacters(in: .whitespacesAndNewlines))
                    ]
                })
        ]
        if input["word_timestamps"]?.bool == true {
            let words = parakeetWords(sentences.flatMap { $0.tokens.map { (text: $0.text, start: $0.start, end: $0.end) } })
            result["words"] = .array(words.prefix(16384).map { ["word": .string($0.word), "start": seconds($0.start, limit: duration), "end": seconds($0.end, limit: duration)] })
        }
        return result
    }
    override func unload() { fallback.reset(); super.unload() }
}

// MARK: - Qwen3-ASR

/// Qwen3-ASR names languages in English; requests use BCP-47 codes.
let qwenLanguageNames: [String: String] = Dictionary(uniqueKeysWithValues: audioLanguageCodes.map { ($0.value, $0.key.capitalized) })

/// Vella keeps the audio tower half at every tier and quantizes the text model's matrices (group 64).
func vellaQwenEligible(_ name: String, _ shape: [Int]) -> Bool {
    audioEligible(name, shape) && !name.hasPrefix("audio_tower.") && !name.hasPrefix("thinker.audio_tower.")
}

final class VellaQwenASRAdapter: AudioAdapter {
    override var architecture: String { "qwen3_asr_vella" }
    override var operations: Set<String> { ["transcribe"] }
    /// Vella's per-segment budget (30-s segments, 1024 tokens).
    var tokenLimit = 1024
    let fallback = VellaFallback(component: "decoder")
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let configuration = try config(context, as: VellaSTT.Qwen3ASRConfig.self)
        let model = VellaSTT.Qwen3ASRModel(configuration)
        tokenLimit = context.randomWeights ? 3 : 1024
        let directory = context.randomWeights ? context.folder.appendingPathComponent("gate/tokenizer") : context.weights
        model.tokenizer = try blockingTokenizer { try await VellaSTT.Qwen3ASRModel.loadTokenizer(directory) }
        let tied = configuration.textConfig.tieWordEmbeddings
        try prepare(
            model, context, cancellation: cancellation,
            map: { key, value in
                // Qwen3ASRModel.sanitize, one tensor at a time: original (thinker.) checkpoints also need OIHW → OHWI.
                let original = key.hasPrefix("thinker.")
                let name = original ? String(key.dropFirst(8)) : key
                if tied, name == "lm_head.weight" { return nil }
                let tensor = original && name.contains("conv2d") && name.contains("weight") && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
                return (name, tensor)
            }, eligible: vellaQwenEligible)
        // Vella's optimized decoder (pipelined greedy decode, prefill without the discarded vocabulary projection). Its
        // optimized encoder holds the audio tower in FP32, which Arsenal does not serve.
        fallback.enable(
            model, allowed: VellaFastPath.available(),
            qualification: vellaQualification(
                context, id: (try? manifestID(String(contentsOf: context.folder.appendingPathComponent("model.toml"), encoding: .utf8))) ?? nil ?? "qwen3-asr",
                fallback: fallback, model: model))
        try cancellation.check()
    }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "language", "task", "prompt", "stream"])
        guard operation == "transcribe", let model = graph as? VellaSTT.Qwen3ASRModel, model.tokenizer != nil else { throw audioFailure("Qwen3-ASR is not loaded.") }
        guard input["task"] == nil || input["task"]?.string == "transcribe" else { throw audioFailure("Qwen3-ASR transcribes; it does not translate.") }
        var language: String?
        if let code = input["language"] {
            guard let value = code.string, let name = qwenLanguageNames[value.lowercased()] else { throw audioFailure("Unsupported transcription language.") }
            language = name
        }
        if let prompt = input["prompt"], prompt.string == nil { throw audioFailure("prompt must be text.") }
        if let stream = input["stream"], stream.bool == nil { throw audioFailure("stream must be a boolean.") }
        try cancellation.check()
        let pcm = try audioInput(input, rate: 16000)
        let duration = Double(pcm.samples.count) / 16000
        let context = input["prompt"]?.string ?? ""
        // Vella transcribes 30-s segments; longer audio splits at the quietest window, one request per segment.
        let chunks = VellaSTT.splitAudioIntoChunks(MLXArray(pcm.samples), sampleRate: 16000, chunkDuration: 30, minChunkDuration: 1)
        var texts: [String] = [], languages: [String?] = [], segments: [JSONValue] = []
        for (chunk, offset) in chunks {
            try cancellation.check()
            let output = try fallback.run(
                model, text: { $0.text },
                body: { model.generate(audio: chunk, maxTokens: tokenLimit, temperature: 0, context: context, language: language, chunkDuration: 30, minChunkDuration: 1) })
            let text = output.text.trimmingCharacters(in: .whitespacesAndNewlines)
            texts.append(text); languages.append(output.language)
            let start = Double(offset)
            segments.append(["start": seconds(start, limit: duration), "end": seconds(start + Double(chunk.dim(0)) / 16000, limit: duration), "text": .string(text)])
            if input["stream"]?.bool == true { try emit(.json(["event": "token", "text": .string(text)])) }
        }
        try cancellation.check()
        var result: JSONObject = ["text": .string(texts.filter { !$0.isEmpty }.joined(separator: " ")), "duration": .number(duration), "segments": .array(segments)]
        if let code = dominantAudioLanguage(languages) { result["language"] = .string(code) }
        return result
    }
    override func unload() { fallback.reset(); super.unload() }
}
