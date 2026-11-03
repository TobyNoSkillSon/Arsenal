import Foundation
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import MLXRandom
import Tokenizers

public final class BreezeTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    let config: BreezeTTSConfig
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "embed_text_tokens") var embedTextTokens: Embedding
    @ModuleInfo(key: "backbone_model") var backboneModel: BreezeBackbone
    @ModuleInfo(key: "depth_decoder") var depthDecoder: BreezeDepthDecoder
    @ModuleInfo(key: "text_encoder") var textEncoder: BreezeTTSTextEncoder
    @ModuleInfo(key: "text_encoder_proj") var textEncoderProjection: Linear

    var tokenizer: Tokenizers.Tokenizer?
    var audioTokenizer: Qwen3TTSSpeechTokenizer?

    public var guidanceScale: Float = 4

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 750,
            temperature: 0.9,
            topP: 1,
            topK: 50,
            repetitionPenalty: 1
        )
    }

    init(config: BreezeTTSConfig) {
        self.config = config
        _lmHead.wrappedValue = Linear(
            config.backboneConfig.hiddenSize,
            config.audioVocabSize + 1,
            bias: false
        )
        _embedTextTokens.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize,
            dimensions: config.backboneConfig.hiddenSize
        )
        _backboneModel.wrappedValue = BreezeBackbone(config: config)
        _depthDecoder.wrappedValue = BreezeDepthDecoder(config: config.depthDecoderConfig)
        _textEncoder.wrappedValue = BreezeTTSTextEncoder(config: config.textEncoderConfig)
        _textEncoderProjection.wrappedValue = Linear(
            config.textEncoderConfig.hiddenSize,
            config.backboneConfig.hiddenSize,
            bias: false
        )
    }

    static func promptText(text: String, instruction: String?) -> String {
        guard let instruction, !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "[S0]\(text)"
        }
        return "[S0]<ins_bos>\(instruction)<ins_eos>\(text)"
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter { key, _ in
            !key.hasPrefix("codec_model.")
                && !key.hasSuffix(".initialized")
                && !key.contains("rotary_emb.inv_freq")
        }
        for suffix in ["weight", "scales", "biases"] {
            if let value = result.removeValue(forKey: "text_encoder.embed_tokens." + suffix) {
                result["text_encoder.embed_tokens.table." + suffix] = value
            }
        }
        if let head = result.removeValue(forKey: "depth_decoder.codebooks_head.weight") {
            for i in 0..<head.dim(0) { result["depth_decoder.codebooks_head.heads.\(i).weight"] = head[i].transposed() }
        }
        let depthKey = "depth_decoder.model.embed_tokens.weight"
        let backboneKey = "backbone_model.embed_tokens.embed_audio_tokens.weight"
        for suffix in ["weight", "scales", "biases"] {
            let depth = String(depthKey.dropLast(6)) + suffix
            let backbone = String(backboneKey.dropLast(6)) + suffix
            if let tied = result[depth] { result[backbone] = tied }
        }
        return result
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = language
        return try generateAudio(
            text: text,
            instruction: voice,
            refAudio: refAudio,
            refText: refText,
            parameters: generationParameters,
            options: fastOptions
        ).audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
                       generationParameters: generationParameters, streamingInterval: 0)
    }

    /// Vireo: `streamingInterval > 0` with `fastOptions.stream` yields audio every `streamingInterval` seconds of
    /// codec frames (streaming codec decode); otherwise one `.audio` with the whole clip after generation (stock).
    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = language
        let options = fastOptions
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    var chunkFrames = options.stream && streamingInterval > 0
                        ? max(1, Int(streamingInterval * Double(self.sampleRate) / Double(self.audioTokenizer?.decodeUpsampleRate ?? 1920)))
                        : 0
                    if chunkFrames > 0, let lab = VireoModelSwitches.breezeStreamFrames {   // VIREO_BREEZE_STREAM_FRAMES
                        chunkFrames = lab   // lab: codec frames per streamed decode
                    }
                    let result = try generateAudio(
                        text: text,
                        instruction: voice,
                        refAudio: refAudio,
                        refText: refText,
                        parameters: generationParameters,
                        options: options,
                        chunkFrames: chunkFrames,
                        onToken: { continuation.yield(.token($0)) },
                        onAudioChunk: { continuation.yield(.audio($0)) }
                    )
                    continuation.yield(.info(result.info))
                    if !result.streamed { continuation.yield(.audio(result.audio)) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // MARK: Vireo fast path

    /// Graph-level optimisations of the generation loop (stock MLX ops only). The adapter sets them after the
    /// load-time self-test (`fastPathSelfTest`) passed; `.stock` is the upstream loop.
    public struct FastOptions: Sendable, Equatable {
        /// One host sync per codec frame instead of 17: the first-codebook sample, the 15 depth samples and the
        /// next backbone step stay lazy, and frame N+1 is queued before frame N is read. Same ops, same order,
        /// same sampler key sequence: token-exact.
        public var lazyFrames: Bool
        /// Classifier-free guidance: conditional and unconditional depth passes as one batch of two.
        public var cfgBatch: Bool
        /// Streaming codec decode (the decoder's causal conv/KV state, as upstream Qwen3-TTS streams the same
        /// codec) instead of one decode of the whole clip after generation. Changes numerics within the gate.
        public var stream: Bool
        /// Depth decoder with a KV cache: one new position per depth step instead of the whole prefix.
        /// Changes GEMM shapes, so not token-exact.
        public var depthCache: Bool
        /// With the depth KV cache: the one-token depth step through fused kernels (`VireoBreezeFused`: attention with
        /// RoPE and the KV write in one dispatch, compiled SiLU-gate).
        /// Inexact (attention sum order), bounded by the same teacher-forced check as the cache.
        public var fusedDepth: Bool
        /// With the fused step: dense FFNs through SmallMGEMM's GEMV (SiLU-gate pair, residual epilogue). Inexact
        /// (split-K sums), bounded by the same teacher-forced check. Only BF16/FP16 checkpoints have dense FFNs.
        public var smallM = false
        public init(lazyFrames: Bool, cfgBatch: Bool, stream: Bool, depthCache: Bool = false, fusedDepth: Bool = false) {
            self.lazyFrames = lazyFrames; self.cfgBatch = cfgBatch; self.stream = stream; self.depthCache = depthCache
            self.fusedDepth = fusedDepth
        }
        public static let stock = FastOptions(lazyFrames: false, cfgBatch: false, stream: false)
        public var isStock: Bool { self == .stock }
        public var label: String {
            [lazyFrames ? "lazy frames" : nil, cfgBatch ? "CFG batch" : nil, depthCache ? "depth KV cache" : nil,
             depthCache && fusedDepth ? (smallM ? "fused depth step + SmallMGEMM FFN" : "fused depth step") : nil, stream ? "streaming decode" : nil]
                .compactMap { $0 }.joined(separator: " + ")
        }
    }

    /// Child-process check of the fused depth attention kernel (`VireoBreezeFused.unitCheck`): output relative RMS
    /// and rotated-cache max |Δ| against the stock ops.
    public static func fusedDepthUnitCheck() -> (attention: Float, cache: Float) { VireoBreezeFused.unitCheck() }
    public static let fusedDepthKernelVersion = VireoBreezeFused.version
    /// Whether the depth decoder's FFN weights are dense (SmallMGEMM's dense GEMV applies).
    public var depthFFNIsDense: Bool { !(depthDecoder.model.layers.first?.mlp.gate is QuantizedLinear) }

    /// Bump when a fast-path change alters what the self-test must prove (part of the gate key).
    public static let fastPathVersion = "breeze-fast-1"
    public var fastOptions = FastOptions.stock
    /// VIREO_BREEZE_PROFILE=1 (set by the adapter): one stderr line per generation.
    public var profile = false

    /// Load-time self-test on short fixed prompts (with and without a voice description, fixed seed):
    /// 1. exact parts (lazy frames, streaming decode): tokens identical to the stock loop; streamed audio within
    ///    `minimumSNR` dB of the whole-clip decode of the same tokens;
    /// 2. inexact parts (depth KV cache, batched guidance): teacher-forced on the stock tokens, the sampling
    ///    distribution of every depth step (guidance and temperature applied) within `maximumProbabilityError`
    ///    (max |Δp| over the vocabulary) of stock's, then a finite run of the full fast configuration. Guidance
    ///    (u + 4(c − u) = 4c − 3u) amplifies a logit difference up to 7×, so guided steps get 5× the bound.
    ///    Measured on M5 Max (4-bit): see lab/perf/NOTES.md.
    /// Returns a one-line detail or throws.
    public func fastPathSelfTest(_ options: FastOptions, frames: Int = 10, minimumSNR: Double = 40,
                                 maximumProbabilityError: Double = 0.02) throws -> String {
        var parts = [String]()
        var exact = options
        exact.depthCache = false; exact.cfgBatch = false
        let text = "Checking the fast path, one two three."
        for (tag, instruction) in [("plain", nil), ("described", "A calm, clear narrator.")] as [(String, String?)] {
            let parameters = GenerateParameters(maxTokens: frames, temperature: 0.9, topP: 1, topK: 50,
                                                repetitionPenalty: 1, seed: 20_260_927)
            let stock = try generateAudio(text: text, instruction: instruction, refAudio: nil, refText: nil,
                                          parameters: parameters, options: .stock, chunkFrames: 0)
            var detail = tag
            if exact.lazyFrames || exact.stream {
                var chunks = [MLXArray]()
                let fast = try generateAudio(text: text, instruction: instruction, refAudio: nil, refText: nil,
                                             parameters: parameters, options: exact,
                                             chunkFrames: exact.stream ? 3 : 0, onAudioChunk: { chunks.append($0) })
                guard fast.frames == stock.frames else {
                    let same = zip(fast.frames, stock.frames).prefix { $0 == $1 }.count
                    throw FastPathFailure("\(tag): tokens differ from stock at frame \(same) of \(stock.frames.count)")
                }
                let a = stock.audio.asType(.float32).asArray(Float.self)
                let b = (fast.streamed ? concatenated(chunks.map { $0.reshaped([-1]) }, axis: 0) : fast.audio)
                    .asType(.float32).asArray(Float.self)
                guard a.count == b.count else { throw FastPathFailure("\(tag): \(b.count) samples, stock \(a.count)") }
                guard b.allSatisfy(\.isFinite) else { throw FastPathFailure("\(tag): non-finite samples") }
                let signal = a.reduce(0.0) { $0 + Double($1) * Double($1) }
                let noise = zip(a, b).reduce(0.0) { $0 + Double($1.0 - $1.1) * Double($1.0 - $1.1) }
                let snr = noise == 0 ? 999 : 10 * Foundation.log10(signal / Swift.max(noise, 1e-30))
                guard snr >= minimumSNR else { throw FastPathFailure(String(format: "%@: audio %.1f dB vs stock (< %.0f)", tag, snr, minimumSNR)) }
                detail += snr >= 999 ? " exact" : String(format: " tokens exact, audio %.0f dB", snr)
            }
            if options.depthCache || (options.cfgBatch && instruction != nil) {
                let error = try teacherForcedDepthError(text: text, instruction: instruction, frames: stock.frames,
                                                        temperature: parameters.temperature, options: options)
                let bound = instruction == nil ? maximumProbabilityError : 5 * maximumProbabilityError
                guard error <= bound else {
                    throw FastPathFailure(String(format: "%@: depth |Δp| %.4f (> %.2f)", tag, error, bound))
                }
                let fast = try generateAudio(text: text, instruction: instruction, refAudio: nil, refText: nil,
                                             parameters: parameters, options: options, chunkFrames: 0)
                guard fast.audio.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite) else {
                    throw FastPathFailure("\(tag): non-finite audio with \(options.label)")
                }
                detail += String(format: ", depth |Δp| %.4f", error)
            }
            parts.append(detail)
        }
        return parts.joined(separator: "; ")
    }

    /// Feeds `frames` (stock tokens) through the backbone and compares, for every depth step, the sampling
    /// distribution (guidance, masking and temperature applied) of the stock per-step recomputation with the
    /// fast path's (KV cache and/or batched guidance). Returns max |Δp| over all steps and vocabulary entries.
    func teacherForcedDepthError(text: String, instruction: String?, frames: [[Int32]], temperature: Float,
                                 options: FastOptions) throws -> Double {
        let conditionalPrompt = try promptEmbeddings(text: text, instruction: instruction, refAudio: nil, refText: nil)
        let unconditionalPrompt = instruction?.isEmpty == false
            ? try promptEmbeddings(text: text, instruction: nil, refAudio: nil, refText: nil) : nil
        let conditionalCache = backboneModel.makeCache()
        var conditional = backboneModel(inputEmbeddings: conditionalPrompt, cache: conditionalCache)[0..., -1, 0...]
        let unconditionalCache = unconditionalPrompt.map { _ in backboneModel.makeCache() }
        var unconditional = zipOptional(unconditionalPrompt, unconditionalCache).map {
            backboneModel(inputEmbeddings: $0.0, cache: $0.1)[0..., -1, 0...]
        }
        func distribution(_ logits: MLXArray) -> MLXArray {
            softmax(maskReservedTokens(logits, allowsEOS: false).asType(.float32) / temperature, axis: -1)
        }
        var worst: Float = 0
        for frame in frames {
            eval(conditional); if let unconditional { eval(unconditional) }
            var stock = [MLXArray]()
            for k in 1..<config.numCodebooks {
                let ids = MLXArray([Int32(0)] + Array(frame[0..<k])).reshaped([1, k + 1])
                var logits = depthDecoder.nextLogits(tokenIDs: ids, backboneHiddenState: conditional)
                if let unconditional {
                    let u = depthDecoder.nextLogits(tokenIDs: ids, backboneHiddenState: unconditional)
                    logits = u + 4 * (logits - u)
                }
                stock.append(distribution(logits))
            }
            var fast = [MLXArray]()
            let tokens = frame.map { MLXArray([$0]) }
            depthDecoder.model.fused = options.depthCache && options.fusedDepth
            depthDecoder.model.smallMFFN = options.smallM
            defer { depthDecoder.model.fused = false }
            if options.depthCache {
                let states = unconditional.map { concatenated([conditional, $0], axis: 0) } ?? conditional
                let rows = states.dim(0)
                let cache = depthDecoder.model.makeCache()
                var hidden = depthDecoder.model.prime(firstToken: MLX.broadcast(tokens[0].reshaped([1, 1]), to: [rows, 1]),
                                                      backboneHiddenState: states, cache: cache)
                for position in 2...config.numCodebooks {
                    let both = depthDecoder.logits(hidden, headIndex: position - 2)
                    var logits = both[0 ..< 1]
                    if rows == 2 { let u = both[1 ..< 2]; logits = u + 4 * (logits - u) }
                    fast.append(distribution(logits))
                    if position < config.numCodebooks {
                        hidden = depthDecoder.model.step(token: MLX.broadcast(tokens[position - 1].reshaped([1, 1]), to: [rows, 1]),
                                                         position: position, cache: cache)
                    }
                }
            } else if let unconditional {
                let both = concatenated([conditional, unconditional], axis: 0)
                for k in 1..<config.numCodebooks {
                    let ids = MLXArray([Int32(0)] + Array(frame[0..<k])).reshaped([1, k + 1])
                    let pair = depthDecoder.nextLogits(tokenIDs: MLX.broadcast(ids, to: [2, k + 1]), backboneHiddenState: both)
                    let c = pair[0 ..< 1], u = pair[1 ..< 2]
                    fast.append(distribution(u + 4 * (c - u)))
                }
            }
            if !fast.isEmpty {
                let difference = abs(stacked(stock) - stacked(fast)).max()
                let value = difference.item(Float.self)
                guard value.isFinite else { throw FastPathFailure("non-finite depth distribution") }
                worst = Swift.max(worst, value)
            }
            let codebooks = MLXArray(frame).reshaped([1, 1, config.numCodebooks])
            conditional = backboneModel(inputIDs: codebooks, cache: conditionalCache)[0..., -1, 0...]
            if let cache = unconditionalCache { unconditional = backboneModel(inputIDs: codebooks, cache: cache)[0..., -1, 0...] }
        }
        return Double(worst)
    }

    private struct GenerationResult {
        let audio: MLXArray
        let info: AudioGenerationInfo
        var frames: [[Int32]] = []
        var streamed = false
    }

    private func generateAudio(
        text: String,
        instruction: String?,
        refAudio: MLXArray?,
        refText: String?,
        parameters: GenerateParameters,
        options: FastOptions = .stock,
        chunkFrames: Int = 0,
        onToken: ((Int) -> Void)? = nil,
        onAudioChunk: ((MLXArray) -> Void)? = nil
    ) throws -> GenerationResult {
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Breeze text tokenizer is not loaded")
        }
        guard let audioTokenizer else {
            throw AudioGenerationError.modelNotInitialized("Breeze audio tokenizer is not loaded")
        }
        if refAudio != nil && (refText?.isEmpty != false) {
            throw AudioGenerationError.invalidInput("Breeze voice cloning needs a reference transcript")
        }
        let maxTokens = parameters.maxTokens ?? 750
        guard maxTokens > 0 else {
            throw AudioGenerationError.invalidInput("maxTokens must be positive")
        }

        let started = Date()
        depthDecoder.model.fused = options.lazyFrames && options.depthCache && options.fusedDepth
        depthDecoder.model.smallMFFN = options.smallM
        defer { depthDecoder.model.fused = false }
        let conditionalPrompt = try promptEmbeddings(
            text: text,
            instruction: instruction,
            refAudio: refAudio,
            refText: refText
        )
        let usesGuidance = instruction?.isEmpty == false
        let unconditionalPrompt = usesGuidance
            ? try promptEmbeddings(text: text, instruction: nil, refAudio: refAudio, refText: refText)
            : nil
        if profile { eval(conditionalPrompt); if let unconditionalPrompt { eval(unconditionalPrompt) } }
        let prepTime = Date().timeIntervalSince(started)

        let conditionalCache = backboneModel.makeCache()
        let conditionalHidden = backboneModel(
            inputEmbeddings: conditionalPrompt,
            cache: conditionalCache
        )[0..., -1, 0...]
        let unconditionalCache = unconditionalPrompt.map { _ in backboneModel.makeCache() }
        let unconditionalHidden = zipOptional(unconditionalPrompt, unconditionalCache).map {
            backboneModel(inputEmbeddings: $0.0, cache: $0.1)[0..., -1, 0...]
        }
        eval(conditionalHidden)
        if let unconditionalHidden { eval(unconditionalHidden) }
        let prefillTime = Date().timeIntervalSince(started)

        // Vireo: without an explicit seed the sampler draws its seed from MLX's global key, so MLXRandom.seed
        // (request seed, VIREO_SPEECH_SEED) makes Breeze reproducible like the other models; upstream seeded it
        // from system entropy, which no seed could fix.
        let seed = parameters.seed ?? Self.seedFromGlobalKey()
        let sampler = parameters.temperature > 0
            ? TopPSampler(
                temperature: parameters.temperature,
                topP: parameters.topP,
                topK: min(parameters.topK, config.codecVocabSize),
                minP: parameters.minP,
                seed: seed
            )
            : nil

        let penalty = parameters.repetitionPenalty ?? 1
        let decoder = FrameDecoder(tokenizer: audioTokenizer, numCodebooks: config.numCodebooks,
                                   chunkFrames: options.stream && onAudioChunk != nil ? chunkFrames : 0,
                                   emit: onAudioChunk)
        defer { decoder.finish() }
        var timing = LoopTiming()
        let frames: [[Int32]]
        if options.lazyFrames && penalty == 1 {
            frames = try lazyLoop(
                conditionalHidden: conditionalHidden, unconditionalHidden: unconditionalHidden,
                conditionalCache: conditionalCache, unconditionalCache: unconditionalCache,
                sampler: sampler, cfgBatch: options.cfgBatch, depthCache: options.depthCache, maxTokens: maxTokens,
                decoder: decoder, timing: &timing, onToken: onToken
            )
        } else {
            frames = try stockLoop(
                conditionalHidden: conditionalHidden, unconditionalHidden: unconditionalHidden,
                conditionalCache: conditionalCache, unconditionalCache: unconditionalCache,
                sampler: sampler, penalty: penalty, maxTokens: maxTokens,
                decoder: decoder, timing: &timing, onToken: onToken
            )
        }
        let loopEnd = Date()

        let audio: MLXArray
        var streamed = false
        if decoder.streaming {
            try decoder.flush(final: true)
            audio = MLXArray.zeros([0])
            streamed = true
        } else if frames.isEmpty {
            audio = MLXArray.zeros([0])
        } else {
            let flat = frames.flatMap { $0 }
            let codes = MLXArray(flat).reshaped([1, frames.count, config.numCodebooks])
            let decoded = audioTokenizer.decode(codes)
            var waveform = decoded.0.squeezed()
            let validSamples = decoded.1.asArray(Int32.self).first.map(Int.init) ?? waveform.dim(0)
            if validSamples >= 0 && validSamples < waveform.dim(0) {
                waveform = waveform[..<validSamples]
            }
            audio = waveform
        }
        eval(audio)

        let totalTime = Date().timeIntervalSince(started)
        let generationTime = max(totalTime - prefillTime, 0)
        if profile {
            let loop = loopEnd.timeIntervalSince(started) - prefillTime
            let perFrame = loop * 1000 / Double(max(frames.count, 1))
            var line = String(format: "[breeze profile] %@ cfg %@ prompt %d · prep %.0f ms · prefill %.0f ms · first frame %.0f ms · %d frames %.1f ms/frame",
                              options.isStock ? "stock" : options.label, usesGuidance ? "yes" : "no", conditionalPrompt.dim(1),
                              prepTime * 1000, (prefillTime - prepTime) * 1000, timing.firstFrame * 1000, frames.count, perFrame)
            if timing.backbone > 0 {
                line += String(format: " (backbone %.1f, depth %.1f, first-codebook %.1f)", timing.backbone * 1000 / Double(max(frames.count, 1)),
                               timing.depth * 1000 / Double(max(frames.count, 1)), timing.head * 1000 / Double(max(frames.count, 1)))
            }
            line += String(format: " · stream decode %.0f ms · final decode %.0f ms · first audio %@ · total %.0f ms\n",
                           decoder.decodeSeconds * 1000, Date().timeIntervalSince(loopEnd) * 1000,
                           decoder.firstEmit.map { String(format: "%.0f ms", $0.timeIntervalSince(started) * 1000) } ?? "end",
                           totalTime * 1000)
            FileHandle.standardError.write(Data(line.utf8))
        }
        return GenerationResult(
            audio: audio,
            info: AudioGenerationInfo(
                promptTokenCount: conditionalPrompt.dim(1),
                generationTokenCount: frames.count,
                prefillTime: prefillTime,
                generateTime: generationTime,
                tokensPerSecond: Double(frames.count) / max(generationTime, 0.001),
                peakMemoryUsage: Double(Memory.peakMemory) / 1_000_000_000
            ),
            frames: frames,
            streamed: streamed
        )
    }

    private struct LoopTiming {
        var firstFrame: TimeInterval = 0
        var head: TimeInterval = 0
        var depth: TimeInterval = 0
        var backbone: TimeInterval = 0
    }

    /// Upstream's loop: a host sync per sampled token (1 + 15 per frame) and per backbone step.
    private func stockLoop(
        conditionalHidden: MLXArray, unconditionalHidden: MLXArray?,
        conditionalCache: [KVCache], unconditionalCache: [KVCache]?,
        sampler: TopPSampler?, penalty: Float, maxTokens: Int,
        decoder: FrameDecoder, timing: inout LoopTiming, onToken: ((Int) -> Void)?
    ) throws -> [[Int32]] {
        var conditionalHidden = conditionalHidden
        var unconditionalHidden = unconditionalHidden
        var frames = [[Int32]]()
        let loopStart = Date()
        for _ in 0..<maxTokens {
            if Task.isCancelled { throw CancellationError() }
            let t0 = Date()
            var logits = lmHead(conditionalHidden)
            if let unconditionalHidden {
                let unconditioned = lmHead(unconditionalHidden)
                logits = unconditioned + guidanceScale * (logits - unconditioned)
            }
            logits = applyRepetitionPenalty(
                logits,
                generatedTokens: frames.map { Int($0[0]) },
                penalty: penalty
            )
            logits = maskReservedTokens(logits, allowsEOS: true)
            let first = sample(logits, sampler: sampler)
            if first == config.audioVocabSize { break }
            onToken?(first)
            let t1 = Date()

            var frame = [Int32(first)]
            var depthInputs = [Int32(0), Int32(first)]
            while frame.count < config.numCodebooks {
                let ids = MLXArray(depthInputs).reshaped([1, depthInputs.count])
                var depthLogits = depthDecoder.nextLogits(
                    tokenIDs: ids,
                    backboneHiddenState: conditionalHidden
                )
                if let unconditionalHidden {
                    let unconditioned = depthDecoder.nextLogits(
                        tokenIDs: ids,
                        backboneHiddenState: unconditionalHidden
                    )
                    depthLogits = unconditioned + guidanceScale * (depthLogits - unconditioned)
                }
                depthLogits = maskReservedTokens(depthLogits, allowsEOS: false)
                let next = sample(depthLogits, sampler: sampler)
                frame.append(Int32(next))
                depthInputs.append(Int32(next))
            }
            frames.append(frame)
            if frames.count == 1 { timing.firstFrame = Date().timeIntervalSince(loopStart) }
            let t2 = Date()

            let codebooks = MLXArray(frame).reshaped([1, 1, config.numCodebooks])
            conditionalHidden = backboneModel(inputIDs: codebooks, cache: conditionalCache)[0..., -1, 0...]
            if let cache = unconditionalCache {
                unconditionalHidden = backboneModel(inputIDs: codebooks, cache: cache)[0..., -1, 0...]
            }
            eval(conditionalHidden)
            if let unconditionalHidden { eval(unconditionalHidden) }
            timing.head += t1.timeIntervalSince(t0)
            timing.depth += t2.timeIntervalSince(t1)
            timing.backbone += Date().timeIntervalSince(t2)
            try decoder.append(frame)
        }
        return frames
    }

    /// One frame's tokens as a lazy graph: first codebook from the backbone state, then the 15 depth tokens, each
    /// sampled lazily and fed to the next depth step. `check` sums every raw logit row (finite iff all are).
    private func lazyFrame(
        conditional: MLXArray, unconditional: MLXArray?, sampler: TopPSampler?, cfgBatch: Bool, depthCache: Bool
    ) -> (tokens: MLXArray, check: MLXArray) {
        var logits = lmHead(conditional)
        var check = logits.sum()
        if let unconditional {
            let unconditioned = lmHead(unconditional)
            check = check + unconditioned.sum()
            logits = unconditioned + guidanceScale * (logits - unconditioned)
        }
        logits = maskReservedTokens(logits, allowsEOS: true)
        let first = sampleLazy(logits, sampler: sampler)
        var tokens = [first]
        if depthCache {
            // One batch row per guidance branch: [cond] or [cond, uncond].
            let states = unconditional.map { concatenated([conditional, $0], axis: 0) } ?? conditional
            let rows = states.dim(0)
            let cache = depthDecoder.model.makeCache()
            var hidden = depthDecoder.model.prime(firstToken: MLX.broadcast(first.reshaped([1, 1]), to: [rows, 1]),
                                                  backboneHiddenState: states, cache: cache)
            for position in 2...config.numCodebooks {
                let both = depthDecoder.logits(hidden, headIndex: position - 2)
                check = check + both.sum()
                var depthLogits = both[0 ..< 1]
                if rows == 2 {
                    let unconditioned = both[1 ..< 2]
                    depthLogits = unconditioned + guidanceScale * (depthLogits - unconditioned)
                }
                depthLogits = maskReservedTokens(depthLogits, allowsEOS: false)
                let next = sampleLazy(depthLogits, sampler: sampler)
                tokens.append(next)
                if position < config.numCodebooks {
                    hidden = depthDecoder.model.step(token: MLX.broadcast(next.reshaped([1, 1]), to: [rows, 1]),
                                                     position: position, cache: cache)
                }
            }
            return (concatenated(tokens, axis: 0), check)
        }
        var ids = concatenated([MLXArray.zeros([1, 1], type: Int32.self), first.reshaped([1, 1])], axis: 1)
        let both = cfgBatch ? unconditional.map { concatenated([conditional, $0], axis: 0) } : nil
        for _ in 1..<config.numCodebooks {
            var depthLogits: MLXArray
            if let both {
                let pair = depthDecoder.nextLogits(tokenIDs: MLX.broadcast(ids, to: [2, ids.dim(1)]), backboneHiddenState: both)
                check = check + pair.sum()
                let conditioned = pair[0 ..< 1]
                let unconditioned = pair[1 ..< 2]
                depthLogits = unconditioned + guidanceScale * (conditioned - unconditioned)
            } else {
                depthLogits = depthDecoder.nextLogits(tokenIDs: ids, backboneHiddenState: conditional)
                check = check + depthLogits.sum()
                if let unconditional {
                    let unconditioned = depthDecoder.nextLogits(tokenIDs: ids, backboneHiddenState: unconditional)
                    check = check + unconditioned.sum()
                    depthLogits = unconditioned + guidanceScale * (depthLogits - unconditioned)
                }
            }
            depthLogits = maskReservedTokens(depthLogits, allowsEOS: false)
            let next = sampleLazy(depthLogits, sampler: sampler)
            tokens.append(next)
            ids = concatenated([ids, next.reshaped([1, 1])], axis: 1)
        }
        return (concatenated(tokens, axis: 0), check)
    }

    /// One host sync per frame, pipelined: frame N+1 (and its backbone step) is queued before frame N is read, so
    /// graph building overlaps the GPU. After the end token one queued frame is discarded.
    private func lazyLoop(
        conditionalHidden: MLXArray, unconditionalHidden: MLXArray?,
        conditionalCache: [KVCache], unconditionalCache: [KVCache]?,
        sampler: TopPSampler?, cfgBatch: Bool, depthCache: Bool, maxTokens: Int,
        decoder: FrameDecoder, timing: inout LoopTiming, onToken: ((Int) -> Void)?
    ) throws -> [[Int32]] {
        struct Pending { let tokens: MLXArray; let check: MLXArray; let conditional: MLXArray; let unconditional: MLXArray? }
        func queue(_ conditional: MLXArray, _ unconditional: MLXArray?) -> Pending {
            let frame = lazyFrame(conditional: conditional, unconditional: unconditional, sampler: sampler,
                                  cfgBatch: cfgBatch, depthCache: depthCache)
            let codebooks = frame.tokens.reshaped([1, 1, config.numCodebooks])
            let nextConditional = backboneModel(inputIDs: codebooks, cache: conditionalCache)[0..., -1, 0...]
            let nextUnconditional = unconditionalCache.map { backboneModel(inputIDs: codebooks, cache: $0)[0..., -1, 0...] }
            asyncEval([frame.tokens, frame.check, nextConditional] + (nextUnconditional.map { [$0] } ?? []))
            return Pending(tokens: frame.tokens, check: frame.check, conditional: nextConditional, unconditional: nextUnconditional)
        }
        var frames = [[Int32]]()
        let loopStart = Date()
        var pending = queue(conditionalHidden, unconditionalHidden)
        for index in 0..<maxTokens {
            if Task.isCancelled { throw CancellationError() }
            let following = index + 1 < maxTokens ? queue(pending.conditional, pending.unconditional) : nil
            let frame = pending.tokens.asArray(Int32.self)
            guard pending.check.item(Float.self).isFinite else { throw FastPathFailure("Breeze: non-finite logits at frame \(index)") }
            if Int(frame[0]) == config.audioVocabSize { break }
            onToken?(Int(frame[0]))
            frames.append(frame)
            if frames.count == 1 { timing.firstFrame = Date().timeIntervalSince(loopStart) }
            try decoder.append(frame)
            guard let following else { break }
            pending = following
        }
        return frames
    }

    private func sampleLazy(_ logits: MLXArray, sampler: TopPSampler?) -> MLXArray {
        (sampler?.sample(logits: logits) ?? argMax(logits, axis: -1)).asType(.int32)
    }

    static func seedFromGlobalKey() -> UInt64 {
        let words = MLXRandom.randInt(low: 0, high: Int32.max, [2], type: Int32.self).asArray(Int32.self)
        return UInt64(UInt32(bitPattern: words[0])) << 32 | UInt64(UInt32(bitPattern: words[1]))
    }

    /// Streaming codec decode for the fast path: every `chunkFrames` frames the new codes go through the decoder's
    /// causal streaming state. Upstream trims the clip to (frames whose first code > 0) × 1920 samples; at any
    /// moment the final length is at least (frames so far − zero-first-code frames so far) × 1920, so only that
    /// prefix is emitted early and the clip ends exactly where the whole-clip decode ends.
    final class FrameDecoder {
        let tokenizer: Qwen3TTSSpeechTokenizer
        let numCodebooks: Int
        let chunkFrames: Int
        let emit: ((MLXArray) -> Void)?
        var streaming: Bool { chunkFrames > 0 && emit != nil }
        private var queued = [[Int32]]()
        private var frameCount = 0
        private var zeroFirst = 0
        private var decoded = [Float]()
        private var emitted = 0
        private var started = false
        var decodeSeconds: TimeInterval = 0
        var firstEmit: Date?

        init(tokenizer: Qwen3TTSSpeechTokenizer, numCodebooks: Int, chunkFrames: Int, emit: ((MLXArray) -> Void)?) {
            self.tokenizer = tokenizer; self.numCodebooks = numCodebooks; self.chunkFrames = chunkFrames; self.emit = emit
        }

        func append(_ frame: [Int32]) throws {
            guard streaming else { return }
            queued.append(frame)
            frameCount += 1
            if frame[0] <= 0 { zeroFirst += 1 }
            if queued.count >= chunkFrames { try flush(final: false) }
        }

        func flush(final: Bool) throws {
            guard streaming else { return }
            if !queued.isEmpty {
                let t = Date()
                if !started { tokenizer.decoder.resetStreamingState(); started = true }
                let codes = MLXArray(queued.flatMap { $0 }).reshaped([1, queued.count, numCodebooks]).transposed(0, 2, 1)
                let wav = tokenizer.decoder.streamingStep(codes).squeezed(axis: 1)[0].asType(.float32)
                let samples = wav.asArray(Float.self)
                guard samples.allSatisfy(\.isFinite) else { throw FastPathFailure("Breeze: non-finite streamed audio") }
                decoded += samples
                queued.removeAll(keepingCapacity: true)
                decodeSeconds += Date().timeIntervalSince(t)
            }
            let bound = Swift.min(decoded.count, (frameCount - zeroFirst) * tokenizer.decodeUpsampleRate)
            if bound > emitted, let emit {
                if firstEmit == nil { firstEmit = Date() }
                emit(MLXArray(Array(decoded[emitted ..< bound])))
                emitted = bound
            }
            if final { finish() }
        }

        func finish() {
            if started { tokenizer.decoder.resetStreamingState(); started = false }
        }
    }

    private func promptEmbeddings(
        text: String,
        instruction: String?,
        refAudio: MLXArray?,
        refText: String?
    ) throws -> MLXArray {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Breeze text tokenizer is not loaded")
        }
        var parts = [MLXArray]()
        if let refAudio, let refText {
            parts.append(textEmbeddings(for: "[S0]\(refText)", tokenizer: tokenizer))
            parts.append(try referenceAudioEmbeddings(refAudio))
            let eos = MLXArray(
                [Int32](repeating: Int32(config.codebookEOSTokenID), count: config.numCodebooks)
            ).reshaped([1, 1, config.numCodebooks])
            parts.append(backboneModel.embedTokens(eos))
        }
        parts.append(textEmbeddings(for: Self.promptText(text: text, instruction: instruction), tokenizer: tokenizer))
        return concatenated(parts, axis: 1)
    }

    private func textEmbeddings(for text: String, tokenizer: Tokenizers.Tokenizer) -> MLXArray {
        let ids = MLXArray(tokenizer.encode(text: text).map { Int32($0) }).reshaped([1, -1])
        return textEncoderProjection(textEncoder(ids))
    }

    private func referenceAudioEmbeddings(_ reference: MLXArray) throws -> MLXArray {
        guard let audioTokenizer else {
            throw AudioGenerationError.modelNotInitialized("Breeze audio tokenizer is not loaded")
        }
        let input: MLXArray
        switch reference.ndim {
        case 1: input = reference.reshaped([1, 1, reference.dim(0)])
        case 2: input = reference.expandedDimensions(axis: 1)
        case 3: input = reference
        default:
            throw AudioGenerationError.invalidInput("Reference audio must have one to three dimensions")
        }
        guard input.dim(0) == 1 else {
            throw AudioGenerationError.invalidInput("Breeze accepts one reference clip per request")
        }
        return backboneModel.embedTokens(audioTokenizer.encode(input).transposed(0, 2, 1))
    }

    private func maskReservedTokens(_ logits: MLXArray, allowsEOS: Bool) -> MLXArray {
        var result = logits
        if config.codecVocabSize < config.audioVocabSize {
            let count = config.audioVocabSize - config.codecVocabSize
            let ids = MLXArray((config.codecVocabSize..<config.audioVocabSize).map(Int32.init)).reshaped([1, count])
            result = putAlong(
                result,
                ids,
                values: MLXArray.full([1, count], values: MLXArray(-Float.infinity), dtype: result.dtype),
                axis: -1
            )
        }
        let width = allowsEOS ? config.audioVocabSize + 1 : config.audioVocabSize
        return result[0..., ..<width]
    }

    private func sample(_ logits: MLXArray, sampler: TopPSampler?) -> Int {
        let token = sampler?.sample(logits: logits) ?? argMax(logits, axis: -1)
        eval(token)
        return token.item(Int.self)
    }

    private func applyRepetitionPenalty(
        _ logits: MLXArray,
        generatedTokens: [Int],
        penalty: Float
    ) -> MLXArray {
        let unique = Array(Set(generatedTokens)).filter { $0 >= 0 && $0 < logits.dim(-1) }
        guard penalty != 1, !unique.isEmpty else { return logits }
        let ids = MLXArray(unique.map(Int32.init)).reshaped([1, -1])
        let selected = takeAlong(logits, ids, axis: -1)
        let penalized = MLX.where(selected .< 0, selected * penalty, selected / penalty)
        return putAlong(logits, ids, values: penalized, axis: -1)
    }



    public static func fromModelDirectory(_ modelDir: URL) async throws -> BreezeTTSModel {
        let data = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(BreezeTTSConfig.self, from: data)
        let intake = StoredVoiceWeights.isIntake(data)
        let model = BreezeTTSModel(config: config)
        let weights = try loadSafetensors(in: modelDir)
        let sanitized = sanitize(weights: weights)

        if intake {
            try StoredVoiceWeights.prepare(model, weights: sanitized)
        } else if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitized["\(path).scales"] != nil else { return nil }
                if let layer = config.perLayerQuantization?.quantization(layer: path) {
                    return layer.asTuple
                }
                return config.quantization?.asTuple
            }
        }
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) }),
            verify: .all
        )
        eval(model.parameters())
        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        model.audioTokenizer = try loadAudioTokenizer(
            from: modelDir.appendingPathComponent("audio_tokenizer"), intake: intake
        )
        return model
    }

    private static func loadSafetensors(in directory: URL) throws -> [String: MLXArray] {
        var result = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        for file in files where file.pathExtension == "safetensors" {
            result.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        return result
    }

    private static func loadAudioTokenizer(from directory: URL, intake: Bool = false) throws -> Qwen3TTSSpeechTokenizer {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw AudioGenerationError.modelNotInitialized("Breeze checkpoint has no audio_tokenizer directory")
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: data)
        let tokenizer = Qwen3TTSSpeechTokenizer(config: config)
        let weights = try loadSafetensors(in: directory)
        let sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights: weights)
        if intake { try StoredVoiceWeights.prepare(tokenizer, weights: sanitized) }
        try tokenizer.update(
            parameters: ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) }),
            verify: .all
        )
        eval(tokenizer.parameters())
        return tokenizer
    }
}

private func zipOptional<A, B>(_ lhs: A?, _ rhs: B?) -> (A, B)? {
    guard let lhs, let rhs else { return nil }
    return (lhs, rhs)
}
