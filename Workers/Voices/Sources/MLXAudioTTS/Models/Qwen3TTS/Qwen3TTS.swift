import Foundation
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import MLXRandom
import Tokenizers

// MARK: - Qwen3TTS Model

public final class Qwen3TTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    let config: Qwen3TTSModelConfig
    let talker: Qwen3TTSTalkerForConditionalGeneration
    var speakerEncoder: Qwen3TTSSpeakerEncoder?
    var speechTokenizer: Qwen3TTSSpeechTokenizer?
    var tokenizer: Tokenizers.Tokenizer?

    private struct ReferenceAudioContext {
        let speakerEmbedding: MLXArray?
        let refCodes: MLXArray
        let codecEmbedIcl: MLXArray
    }


    public var cancellationRequested: (() -> Bool)?
    /// Vireo fast paths (set by the adapter after its self-test), all bit-identical:
    /// pipelined decode loop; fused RoPE kernel (talker + code predictor); code-predictor rotary/mask tables and a
    /// frame-sized KV cache; fused code-predictor sampler (top-k + inverse-CDF categorical per sub-step, keys and
    /// uniform draws for all sub-steps in one dispatch).
    public struct FastOptions: Sendable, Equatable {
        public var pipeline: Bool
        public var fusedRope: Bool
        public var predictorTables: Bool
        public var fusedSampler: Bool
        public init(pipeline: Bool, fusedRope: Bool, predictorTables: Bool, fusedSampler: Bool = false) {
            self.pipeline = pipeline; self.fusedRope = fusedRope; self.predictorTables = predictorTables
            self.fusedSampler = fusedSampler
        }
        public static let stock = FastOptions(pipeline: false, fusedRope: false, predictorTables: false, fusedSampler: false)
        public var isStock: Bool { self == .stock }
        public var label: String {
            [fusedRope ? "fused RoPE" : nil, predictorTables ? "predictor tables" : nil,
             fusedSampler ? "fused predictor sampler" : nil, pipeline ? "pipelined decode" : nil]
                .compactMap { $0 }.joined(separator: " + ")
        }
    }
    public private(set) var fastOptions = FastOptions.stock
    var fastPipeline: Bool { fastOptions.pipeline }

    public func setFastOptions(_ options: FastOptions) {
        fastOptions = options
        for layer in talker.model.layers { layer.selfAttn.fusedRope = options.fusedRope }
        for layer in talker.codePredictor.model.layers { layer.selfAttn.fusedRope = options.fusedRope }
        talker.codePredictor.model.cacheTables = options.predictorTables
    }
    /// VIREO_QWEN_PROFILE=1: one stderr line per generation (prep, prefill, ms/step, activation dtype).
    static let profile = ["1", "2"].contains(ProcessInfo.processInfo.environment["VIREO_QWEN_PROFILE"])
    /// VIREO_QWEN_PROFILE=2 (lab): also sync after the talker and after the code predictor to split the step time.
    static let profileSplit = ProcessInfo.processInfo.environment["VIREO_QWEN_PROFILE"] == "2"
    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 8192,
            temperature: 0.9,
            topP: 1.0,
            topK: 50,
            repetitionPenalty: 1.05
        )
    }

    init(config: Qwen3TTSModelConfig) {
        let talkerConfig = config.talkerConfig ?? {
            let json = "{}".data(using: .utf8)!
            return try! JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: json)
        }()
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config: talkerConfig)
        self.speakerEncoder = config.ttsModelType == "base"
            ? Qwen3TTSSpeakerEncoder(config: config.speakerEncoderConfig)
            : nil
    }

    // MARK: - SpeechGenerationModel protocol

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try requireGenerationComponents()
        let settings = resolveVoiceDesignGenerationSettings(
            language: language,
            generationParameters: generationParameters
        )

        return try generateVoiceDesign(
            text: text,
            instruct: voice,
            language: settings.language,
            refAudio: refAudio,
            refText: refText,
            temperature: settings.temperature,
            topK: settings.topK,
            topP: settings.topP,
            repetitionPenalty: settings.repetitionPenalty,
            minP: settings.minP,
            maxTokens: settings.maxTokens
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: 2.0
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let settings = resolveVoiceDesignGenerationSettings(
            language: language,
            generationParameters: generationParameters
        )
        return makeGenerationStream { model, onToken, onInfo, onAudioChunk in
            _ = try model.generateVoiceDesign(
                text: text,
                instruct: voice,
                language: settings.language,
                refAudio: refAudio,
                refText: refText,
                temperature: settings.temperature,
                topK: settings.topK,
                topP: settings.topP,
                repetitionPenalty: settings.repetitionPenalty,
                minP: settings.minP,
                maxTokens: settings.maxTokens,
                streamingInterval: streamingInterval,
                onToken: onToken,
                onInfo: onInfo,
                onAudioChunk: onAudioChunk
            )
        }
    }

    public func generate(
        text: String,
        conditioning: Qwen3TTSReferenceConditioning,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try requireGenerationComponents()
        let settings = resolveVoiceDesignGenerationSettings(
            language: conditioning.resolvedLanguage,
            generationParameters: generationParameters
        )

        return try generateVoiceDesign(
            text: text,
            instruct: nil,
            language: settings.language,
            conditioning: conditioning,
            refAudio: nil,
            refText: nil,
            temperature: settings.temperature,
            topK: settings.topK,
            topP: settings.topP,
            repetitionPenalty: settings.repetitionPenalty,
            minP: settings.minP,
            maxTokens: settings.maxTokens
        )
    }

    public func generateStream(
        text: String,
        conditioning: Qwen3TTSReferenceConditioning,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            conditioning: conditioning,
            generationParameters: generationParameters,
            streamingInterval: 2.0
        )
    }

    public func generateStream(
        text: String,
        conditioning: Qwen3TTSReferenceConditioning,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let settings = resolveVoiceDesignGenerationSettings(
            language: conditioning.resolvedLanguage,
            generationParameters: generationParameters
        )
        return makeGenerationStream { model, onToken, onInfo, onAudioChunk in
            _ = try model.generateVoiceDesign(
                text: text,
                instruct: nil,
                language: settings.language,
                conditioning: conditioning,
                refAudio: nil,
                refText: nil,
                temperature: settings.temperature,
                topK: settings.topK,
                topP: settings.topP,
                repetitionPenalty: settings.repetitionPenalty,
                minP: settings.minP,
                maxTokens: settings.maxTokens,
                streamingInterval: streamingInterval,
                onToken: onToken,
                onInfo: onInfo,
                onAudioChunk: onAudioChunk
            )
        }
    }

    // MARK: - Decode chunk helper

    /// Decode a chunk of codec codes to audio waveform.
    /// - Parameters:
    ///   - codes: Codec codes [1, time, numCodeGroups]
    ///   - chunkTokens: Tokens per decode chunk (controls decode granularity)
    /// - Returns: Decoded audio waveform (1D)
    private func decodeChunk(_ codes: MLXArray, chunkTokens: Int = 300) -> MLXArray {
        guard let speechTokenizer else { return MLXArray.zeros([1]) }

        var audioChunks = [MLXArray]()
        for chunk in speechTokenizer.streamingDecode(codes, chunkTokens: chunkTokens) {
            audioChunks.append(chunk)
        }
        var audio = concatenated(audioChunks, axis: -1)[0]

        let validLen = Int((codes[0..., 0..., 0] .> 0).sum().item(Int32.self))
            * speechTokenizer.decodeUpsampleRate
        if validLen > 0, validLen < audio.dim(0) {
            audio = audio[..<validLen]
        }

        eval(audio)
        return audio
    }

    private func referenceAudioForEncoder(_ refAudio: MLXArray) -> MLXArray {
        if refAudio.ndim == 1 {
            return refAudio.reshaped(1, 1, refAudio.dim(0))
        } else if refAudio.ndim == 2 {
            return refAudio.reshaped(1, refAudio.dim(0), refAudio.dim(1))
        } else {
            return refAudio
        }
    }

    private func codecEmbedIcl(from refCodes: MLXArray, talkerConfig: Qwen3TTSTalkerConfig) -> MLXArray {
        let firstCbCodes = refCodes[0..., 0, 0...]
        var refCodecEmbed = talker.getInputEmbeddings()(firstCbCodes)
        if talkerConfig.numCodeGroups > 1 {
            for i in 0 ..< (talkerConfig.numCodeGroups - 1) {
                let codeIdx = i + 1
                if codeIdx >= refCodes.dim(1) { break }
                let cbCodes = refCodes[0..., codeIdx, 0...]
                refCodecEmbed = refCodecEmbed + talker.codePredictor.codecEmbedding[i](cbCodes)
            }
        }

        let codecBosEmbed = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecBosId)]).reshaped(1, 1)
        )
        return concatenated([codecBosEmbed, refCodecEmbed], axis: 1)
    }

    private func referenceAudioContext(for refAudio: MLXArray) -> ReferenceAudioContext {
        guard let speechTokenizer, let talkerConfig = config.talkerConfig else {
            fatalError("Speech tokenizer/config not loaded")
        }

        // Vireo: no per-model cache here. Upstream keyed one by ObjectIdentifier(refAudio), which never hit for
        // per-request arrays and could return another voice's context when a freed array's address was reused.
        // MLXSpeech caches the conditioning per saved voice (content-keyed) instead.
        let speakerEmbedding = extractSpeakerEmbedding(refAudio)
        let refCodes = speechTokenizer.encode(referenceAudioForEncoder(refAudio))
        let codecEmbedIcl = codecEmbedIcl(from: refCodes, talkerConfig: talkerConfig)
        if let speakerEmbedding {
            eval(speakerEmbedding, refCodes, codecEmbedIcl)
        } else {
            eval(refCodes, codecEmbedIcl)
        }
        return ReferenceAudioContext(
            speakerEmbedding: speakerEmbedding,
            refCodes: refCodes,
            codecEmbedIcl: codecEmbedIcl
        )
    }

    // MARK: - VoiceDesign generation

    func generateVoiceDesign(
        text: String,
        instruct: String?,
        language: String,
        conditioning: Qwen3TTSReferenceConditioning? = nil,
        refAudio: MLXArray?,
        refText: String?,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        minP: Float,
        maxTokens: Int,
        streamingInterval: Double = 2.0,
        onToken: ((Int) -> Void)? = nil,
        onInfo: ((AudioGenerationInfo) -> Void)? = nil,
        onAudioChunk: ((MLXArray) -> Void)? = nil
    ) throws -> MLXArray {
        guard let speechTokenizer, let tokenizer else {
            throw AudioGenerationError.modelNotInitialized(
                "Qwen3TTS generateVoiceDesign requires both the speech tokenizer and the text tokenizer to be loaded."
            )
        }

        let talkerConfig = config.talkerConfig!
        let profileStart = Date()

        // Prepare inputs
        let inputEmbedsInit: MLXArray
        let trailingTextHidden: MLXArray
        let ttsPadEmbed: MLXArray
        let refCodes: MLXArray?

        if let conditioning {
            let prepared = try prepareICLGenerationInputs(
                text: text,
                conditioning: conditioning
            )
            inputEmbedsInit = prepared.0
            trailingTextHidden = prepared.1
            ttsPadEmbed = prepared.2
            refCodes = prepared.3
        } else if let refAudio,
           let refText,
           speechTokenizer.hasEncoder {
            let prepared = try prepareICLGenerationInputs(
                text: text,
                refAudio: refAudio,
                refText: refText,
                language: language
            )
            inputEmbedsInit = prepared.0
            trailingTextHidden = prepared.1
            ttsPadEmbed = prepared.2
            refCodes = prepared.3
        } else {
            // CustomVoice accepts `voice` as "speaker, instruction".
            let isCVModel = config.ttsModelType == "custom_voice"
            let customVoicePrompt = isCVModel ? Self.parseCustomVoicePrompt(instruct) : nil
            let speaker: String? = isCVModel ? customVoicePrompt?.speaker : nil
            let effectiveInstruct: String? = isCVModel ? customVoicePrompt?.instruction : instruct
            let prepared = prepareGenerationInputs(
                text: text,
                language: language,
                instruct: effectiveInstruct,
                speaker: speaker
            )
            inputEmbedsInit = prepared.0
            trailingTextHidden = prepared.1
            ttsPadEmbed = prepared.2
            refCodes = nil
        }

        // Cap max tokens based on text length
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Initialize cache and timing
        let startTime = Date()
        if Self.profile { eval(inputEmbedsInit) }
        let prepMS = Date().timeIntervalSince(profileStart) * 1000
        var firstStepMS = 0.0
        var splitTalker = 0.0, splitPredictor = 0.0
        let cache = talker.makeCache()
        var generatedCodes = [MLXArray]()
        var generatedCodebookTokens = [Int]()
        let eosTokenId = talkerConfig.codecEosTokenId

        // Suppress special tokens
        let suppressTokens = (talkerConfig.vocabSize - 1024 ..< talkerConfig.vocabSize)
            .filter { $0 != eosTokenId }

        // Streaming decode state
        let codecTokenRateHz = 12.5
        let streamingChunkSize = max(1, Int(streamingInterval * codecTokenRateHz))
        var decodedTokens = 0

        var trailingIdx = 0
        var inputEmbeds = inputEmbedsInit
        let eosTokenArray = MLXArray([Int32(eosTokenId)]).reshaped(1, 1)
        let codeCache = talker.codePredictor.makeCache()
        // Vireo fast path: fused code-predictor sampler (same keys, same tokens; see VireoFusedSampler).
        let cpVocab = talker.codePredictor.config.vocabSize
        let cpSteps = talkerConfig.numCodeGroups - 1
        let fusedSampler = fastOptions.fusedSampler && VireoFusedSampler.eligible(
            temperature: temperature, topP: topP, minP: minP, vocab: cpVocab, dtype: inputEmbedsInit.dtype)
        func sampleCode(_ codeLogits: MLXArray, draws: VireoFusedSampler.Draws?, step: Int) -> MLXArray {
            guard let draws else {
                return sampleToken(codeLogits, temperature: temperature, topP: topP, topK: topK, minP: minP)
            }
            return VireoFusedSampler.sample(logits: codeLogits, draws: draws, step: step, topK: topK, temperature: temperature)
        }

        if onAudioChunk != nil {
            speechTokenizer.decoder.resetStreamingState()
        }
        defer {
            if onAudioChunk != nil {
                speechTokenizer.decoder.resetStreamingState()
            }
        }

        // Streaming: decode and yield audio chunks during generation (shared by both loops).
        func streamIfDue() {
            guard let onAudioChunk else { return }
            let newTokens = generatedCodes.count - decodedTokens
            if newTokens >= streamingChunkSize {
                let codesChunk = stacked(Array(generatedCodes[decodedTokens...]), axis: 1)
                let codesForDecoder = codesChunk.transposed(0, 2, 1)
                eval(codesForDecoder)
                let decoded = speechTokenizer.decoder.streamingStep(codesForDecoder).squeezed(axis: 1)
                let audioChunk = decoded[0]
                eval(audioChunk)

                decodedTokens = generatedCodes.count
                onAudioChunk(audioChunk)
            }
        }

        if fastPipeline {
            // Vireo fast path: step N+1 is built from step N's still-lazy tokens and queued before step N is read,
            // so graph building overlaps the GPU. Same ops and sampler key order as the loop below; the repetition
            // penalty reads the first-codebook history as a lazy array (duplicates scatter identical values).
            struct Pending { let token: MLXArray; let isEOS: MLXArray; let codes: MLXArray; let next: MLXArray; let check: MLXArray }
            func build(_ input: MLXArray, step: Int, history: MLXArray?) -> Pending {
                let (logits, hidden) = talker(input, cache: cache)
                let nextToken = sampleToken(logits, temperature: temperature, topP: topP, topK: topK,
                                            repetitionPenalty: repetitionPenalty, generatedTokenArray: history,
                                            suppressTokens: suppressTokens, eosTokenId: eosTokenId, minP: minP)
                let isEOS = nextToken .== eosTokenArray
                var codeTokens = [nextToken]
                let codeHidden = hidden[0..., (-1)..., 0...]
                for layerCache in codeCache { _ = layerCache.trim(layerCache.offset) }
                let draws = fusedSampler ? VireoFusedSampler.draw(steps: cpSteps) : nil
                for codeIdx in 0 ..< talkerConfig.numCodeGroups - 1 {
                    let codeInput = codeIdx == 0
                        ? concatenated([codeHidden, talker.getInputEmbeddings()(nextToken)], axis: 1)
                        : talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens.last!)
                    let (codeLogits, _, _) = talker.codePredictor(codeInput, cache: codeCache, generationStep: codeIdx)
                    codeTokens.append(sampleCode(codeLogits, draws: draws, step: codeIdx))
                }
                let textEmbed = step < trailingTextHidden.dim(1)
                    ? trailingTextHidden[0..., step ..< (step + 1), 0...] : ttsPadEmbed
                var codecEmbed = talker.getInputEmbeddings()(nextToken)
                for (i, code) in codeTokens.dropFirst().enumerated() {
                    codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[i](code)
                }
                let pending = Pending(token: nextToken, isEOS: isEOS, codes: concatenated(codeTokens, axis: 1),
                                      next: textEmbed + codecEmbed, check: logits.sum())
                asyncEval(pending.next, pending.isEOS, pending.codes, pending.check)
                return pending
            }
            var history: MLXArray? = nil
            var pending = build(inputEmbeds, step: 0, history: nil)
            for step in 0 ..< effectiveMaxTokens {
                try Task.checkCancellation(); if cancellationRequested?() == true { throw CancellationError() }
                let extended = history.map { concatenated([$0, pending.token], axis: 1) } ?? pending.token
                let following = step + 1 < effectiveMaxTokens ? build(pending.next, step: step + 1, history: extended) : nil
                let tokenId = Int(pending.token[0, 0].item(Int32.self))
                guard pending.check.item(Float.self).isFinite else { throw FastPathFailure("Qwen3-TTS: non-finite logits at step \(step)") }
                if step == 0 { firstStepMS = Date().timeIntervalSince(startTime) * 1000 }
                onToken?(tokenId)
                if pending.isEOS.item(Bool.self) { break }
                history = extended
                generatedCodebookTokens.append(tokenId)
                generatedCodes.append(pending.codes)
                streamIfDue()
                if step > 0, step % 50 == 0 { Memory.clearCache() }
                guard let following else { break }
                pending = following
            }
        } else {
        for step in 0 ..< effectiveMaxTokens {
            try Task.checkCancellation(); if cancellationRequested?() == true { throw CancellationError() }
            // Forward pass through talker
            let t0 = Date()
            let (logits, hidden) = talker(inputEmbeds, cache: cache)
            if Self.profileSplit { eval(logits, hidden); splitTalker += Date().timeIntervalSince(t0) }
            let t1 = Date()

            // Sample first codebook token
            let nextToken = sampleToken(
                logits,
                temperature: temperature,
                topP: topP,
                topK: topK,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedCodebookTokens,
                suppressTokens: suppressTokens,
                eosTokenId: eosTokenId,
                minP: minP
            )

            // Defer sync to the eval boundary with inputEmbeds.
            let isEOS = nextToken .== eosTokenArray

            // Generate remaining codebook tokens with code predictor
            var codeTokens = [nextToken]
            let codeHidden = hidden[0..., (-1)..., 0...]
            for layerCache in codeCache {
                _ = layerCache.trim(layerCache.offset)
            }
            let draws = fusedSampler ? VireoFusedSampler.draw(steps: cpSteps) : nil

            for codeIdx in 0 ..< talkerConfig.numCodeGroups - 1 {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    let code0Embed = talker.getInputEmbeddings()(nextToken)
                    codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](codeTokens.last!)
                }

                let (codeLogits, _, _) = talker.codePredictor(
                    codeInput, cache: codeCache, generationStep: codeIdx
                )

                let nextCode = sampleCode(codeLogits, draws: draws, step: codeIdx)
                codeTokens.append(nextCode)
            }

            let allCodes = concatenated(codeTokens, axis: 1) // [1, num_code_groups]
            if Self.profileSplit { eval(allCodes); splitPredictor += Date().timeIntervalSince(t1) }

            // Prepare next input
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.dim(1) {
                textEmbed = trailingTextHidden[0..., trailingIdx ..< (trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            // Sum all code embeddings for next step
            var codecEmbed = talker.getInputEmbeddings()(nextToken)
            for (i, code) in codeTokens.dropFirst().enumerated() {
                codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[i](code)
            }

            inputEmbeds = textEmbed + codecEmbed
            eval(inputEmbeds, isEOS)

            let tokenId = Int(nextToken[0, 0].item(Int32.self))
            if step == 0 { firstStepMS = Date().timeIntervalSince(startTime) * 1000 }
            onToken?(tokenId)
            if isEOS.item(Bool.self) {
                break
            }
            generatedCodebookTokens.append(tokenId)
            generatedCodes.append(allCodes)

            streamIfDue()

            if step > 0, step % 50 == 0 {
                Memory.clearCache()
            }
        }
        }

        try Task.checkCancellation(); if cancellationRequested?() == true { throw CancellationError() }

        guard !generatedCodes.isEmpty else {
            return MLXArray.zeros([1])
        }

        // Emit generation info
        let generateTime = Date().timeIntervalSince(startTime)
        let tokenCount = generatedCodes.count
        if Self.profile {
            let rest = generateTime * 1000 - firstStepMS
            FileHandle.standardError.write(Data(String(format: "[qwen3-tts profile] %@mode %@ prompt %d tokens dtype %@ · prep %.1f ms · first step %.1f ms · %d steps %.2f ms/step · total %.0f ms\n",
                fastOptions.isStock ? "" : fastOptions.label + " · ", conditioning != nil || refAudio != nil ? "icl" : "design", inputEmbedsInit.dim(1), "\(inputEmbedsInit.dtype)", prepMS, firstStepMS,
                tokenCount, rest / Double(max(1, tokenCount - 1)), generateTime * 1000 + prepMS).utf8))
            if Self.profileSplit {
                FileHandle.standardError.write(Data(String(format: "[qwen3-tts profile] split: talker %.2f ms/step · code predictor %.2f ms/step\n",
                    splitTalker * 1000 / Double(tokenCount), splitPredictor * 1000 / Double(tokenCount)).utf8))
            }
        }
        let info = AudioGenerationInfo(
            promptTokenCount: 0, // Not tracked for VoiceDesign
            generationTokenCount: tokenCount,
            prefillTime: 0, // Included in generateTime
            generateTime: generateTime,
            tokensPerSecond: Double(tokenCount) / generateTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
        onInfo?(info)
        try Task.checkCancellation(); if cancellationRequested?() == true { throw CancellationError() }

        // Streaming path: yield remaining tokens and return early
        if let onAudioChunk {
            if generatedCodes.count > decodedTokens {
                let codesChunk = stacked(Array(generatedCodes[decodedTokens...]), axis: 1)
                let codesForDecoder = codesChunk.transposed(0, 2, 1)
                eval(codesForDecoder)
                let decoded = speechTokenizer.decoder.streamingStep(codesForDecoder).squeezed(axis: 1)
                let audioChunk = decoded[0]
                eval(audioChunk)
                onAudioChunk(audioChunk)
            }
            // Streaming chunks already yielded; return empty (caller uses chunks)
            return MLXArray.zeros([1])
        }

        // Non-streaming path: full decode (existing behavior)
        let codes = stacked(generatedCodes, axis: 1) // [1, seq_len, num_code_groups]

        var decodeCodes = codes
        if let refCodes {
            let refCodesT = refCodes.transposed(0, 2, 1)
            decodeCodes = concatenated([refCodesT, codes], axis: 1)
        }

        var audio = decodeChunk(decodeCodes)

        if let refCodes {
            let refLen = refCodes.dim(2)
            let totalLen = decodeCodes.dim(1)
            let cut = Int(Double(refLen) / Double(max(totalLen, 1)) * Double(audio.dim(0)))
            if cut > 0, cut < audio.dim(0) {
                audio = audio[cut...]
            }
        }

        eval(audio)
        return audio
    }

    /// Vireo: the fused RoPE kernel against the stock expression on fixed inputs; differing elements (0 = exact).
    /// Run it in KernelGate's child process first: a Metal compile error or fault aborts the process.
    public static func fusedRopeUnitCheck() -> Int { VireoFusedRoPE.unitCheck() }

    /// Vireo: the fused code-predictor sampler against stock sampling (tokens and global key state); mismatches.
    /// Child process first, like the RoPE check.
    public static func fusedSamplerUnitCheck() -> (mismatches: Int, detail: String) { VireoFusedSampler.unitCheckReport() }

    /// Vireo load-time self-test for `options` (after the kernel passed `fusedRopeUnitCheck` in the child):
    /// the same short seeded generation (sampling and repetition penalty on) through the stock and the
    /// fast configuration: audio must be bit-identical (the codec decoder is deterministic, so identical samples
    /// mean identical codes). Leaves the model on stock; the adapter applies the verdict.
    public func fastPathSelfTest(_ options: FastOptions, steps: Int = 12) throws -> String {
        try requireGenerationComponents()
        func run(_ options: FastOptions) throws -> [Float] {
            setFastOptions(options)
            defer { setFastOptions(.stock) }
            MLXRandom.seed(20_260_927)
            let audio = try generateVoiceDesign(text: "Checking the fast path, one two three.", instruct: "A calm, clear narrator.",
                                                language: "english", refAudio: nil, refText: nil, temperature: 0.9, topK: 50,
                                                topP: 1.0, repetitionPenalty: 1.05, minP: 0, maxTokens: steps)
            return audio.asType(.float32).asArray(Float.self)
        }
        let stock = try run(.stock)
        let fast = try run(options)
        guard fast.allSatisfy(\.isFinite) else { throw FastPathFailure("non-finite audio") }
        guard fast == stock else {
            throw FastPathFailure("audio differs from stock (\(fast.count) vs \(stock.count) samples)")
        }
        return "\(stock.count) samples bit-identical"
    }

    static func parseCustomVoicePrompt(_ voice: String?) -> (speaker: String, instruction: String?)? {
        guard let voice = voice?.trimmingCharacters(in: .whitespacesAndNewlines),
              !voice.isEmpty else {
            return nil
        }

        guard let commaIndex = voice.firstIndex(of: ",") else {
            return (speaker: voice, instruction: nil)
        }

        let speaker = voice[..<commaIndex].trimmingCharacters(in: .whitespacesAndNewlines)
        let instructionStart = voice.index(after: commaIndex)
        let instruction = voice[instructionStart...].trimmingCharacters(in: .whitespacesAndNewlines)

        guard !speaker.isEmpty else {
            return (speaker: voice, instruction: nil)
        }

        return (
            speaker: String(speaker),
            instruction: instruction.isEmpty ? nil : String(instruction)
        )
    }

    // MARK: - Reference conditioning

    public func prepareReferenceConditioning(
        refAudio: MLXArray,
        refText: String,
        language: String?
    ) throws -> Qwen3TTSReferenceConditioning {
        try prepareReferenceConditioning(
            refAudio: refAudio,
            refText: refText,
            speakerEmbedding: nil,
            language: language ?? "auto"
        )
    }

    private struct VoiceDesignGenerationSettings {
        let language: String
        let temperature: Float
        let topK: Int
        let topP: Float
        let repetitionPenalty: Float
        let minP: Float
        let maxTokens: Int
    }

    public struct Qwen3TTSReferenceConditioning: @unchecked Sendable {
        public let speakerEmbedding: MLXArray?
        public let referenceSpeechCodes: MLXArray
        public let referenceTextTokenIDs: MLXArray
        public let resolvedLanguage: String
        public let codecLanguageID: Int?

        public init(
            speakerEmbedding: MLXArray?,
            referenceSpeechCodes: MLXArray,
            referenceTextTokenIDs: MLXArray,
            resolvedLanguage: String,
            codecLanguageID: Int?
        ) {
            self.speakerEmbedding = speakerEmbedding
            self.referenceSpeechCodes = referenceSpeechCodes
            self.referenceTextTokenIDs = referenceTextTokenIDs
            self.resolvedLanguage = resolvedLanguage
            self.codecLanguageID = codecLanguageID
        }
    }

    private func requireGenerationComponents() throws {
        guard speechTokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Speech tokenizer not loaded")
        }
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Text tokenizer not loaded")
        }
    }

    private func resolveVoiceDesignGenerationSettings(
        language: String?,
        generationParameters: GenerateParameters
    ) -> VoiceDesignGenerationSettings {
        VoiceDesignGenerationSettings(
            language: language ?? "auto",
            temperature: generationParameters.temperature,
            topK: generationParameters.topK,
            topP: generationParameters.topP,
            repetitionPenalty: generationParameters.repetitionPenalty ?? 1.05,
            minP: generationParameters.minP,
            maxTokens: generationParameters.maxTokens ?? 4096
        )
    }

    private func makeGenerationStream(
        _ body: @escaping @Sendable (
            Qwen3TTSModel,
            @escaping (Int) -> Void,
            @escaping (AudioGenerationInfo) -> Void,
            @escaping (MLXArray) -> Void
        ) throws -> Void
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                try requireGenerationComponents()
                try body(
                    self,
                    { tokenId in continuation.yield(.token(tokenId)) },
                    { info in continuation.yield(.info(info)) },
                    { chunk in eval(chunk); continuation.yield(.audio(chunk)) }
                )
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    func prepareICLGenerationInputs(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let conditioning = try prepareReferenceConditioning(
            refAudio: refAudio,
            refText: refText,
            speakerEmbedding: nil,
            language: language
        )
        return try prepareICLGenerationInputs(text: text, conditioning: conditioning)
    }

    func prepareReferenceConditioning(
        refAudio: MLXArray,
        refText: String,
        speakerEmbedding: MLXArray?,
        language: String
    ) throws -> Qwen3TTSReferenceConditioning {
        guard let tokenizer, let talkerConfig = config.talkerConfig, let speechTokenizer else {
            throw AudioGenerationError.modelNotInitialized(
                "Qwen3TTS reference conditioning requires the text tokenizer, talker config, and speech tokenizer to be loaded."
            )
        }
        guard speechTokenizer.hasEncoder else {
            throw AudioGenerationError.invalidInput(
                "Qwen3TTS reference conditioning requires a speech tokenizer encoder, but this checkpoint does not provide one."
            )
        }

        let refContext = referenceAudioContext(for: refAudio)

        let resolvedLanguage = language.lowercased() == "auto" ? "auto" : language.lowercased()

        // Reference text tokenization
        let refChatText = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refIds = MLXArray(tokenizer.encode(text: refChatText).map { Int32($0) }).reshaped(1, -1)
        let refCount = refIds.dim(1)
        let refStart = min(3, refCount)
        let refEnd = max(refStart, refCount - 2)
        let refTextIds = refIds[0..., refStart ..< refEnd]

        // Language ID
        var languageId: Int?
        if resolvedLanguage != "auto", let langMap = talkerConfig.codecLanguageId {
            languageId = langMap[resolvedLanguage]
        }

        return Qwen3TTSReferenceConditioning(
            speakerEmbedding: speakerEmbedding ?? refContext.speakerEmbedding,
            referenceSpeechCodes: refContext.refCodes,
            referenceTextTokenIDs: refTextIds,
            resolvedLanguage: resolvedLanguage,
            codecLanguageID: languageId
        )
    }

    func prepareICLGenerationInputs(
        text: String,
        conditioning: Qwen3TTSReferenceConditioning
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            throw AudioGenerationError.modelNotInitialized(
                "Qwen3TTS request assembly requires the text tokenizer and talker config to be loaded."
            )
        }

        let targetChatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(text: targetChatText).map { Int32($0) }).reshaped(1, -1)
        let targetCount = targetIds.dim(1)
        let targetStart = min(3, targetCount)
        let targetEnd = max(targetStart, targetCount - 5)
        let targetTextIds = targetIds[0..., targetStart ..< targetEnd]

        let ttsTokens = MLXArray(
            [Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId), Int32(config.ttsPadTokenId)]
        ).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0 ..< 1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1 ..< 2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2 ..< 3, 0...]

        let combinedTextIds = concatenated([conditioning.referenceTextTokenIDs, targetTextIds], axis: 1)
        var textEmbed = talker.textProjection(talker.getTextEmbeddings()(combinedTextIds))
        textEmbed = concatenated([textEmbed, ttsEosEmbed], axis: 1)
        let textLen = textEmbed.dim(1)

        let refCodes = conditioning.referenceSpeechCodes
        let codecEmbedIcl = codecEmbedIcl(from: refCodes, talkerConfig: talkerConfig)

        let codecPadEmbed = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId)]).reshaped(1, 1)
        )
        let textWithCodecPad = textEmbed + broadcast(
            codecPadEmbed,
            to: [1, textLen, codecPadEmbed.dim(-1)]
        )
        let codecWithTextPad = codecEmbedIcl + broadcast(
            ttsPadEmbed,
            to: [1, codecEmbedIcl.dim(1), ttsPadEmbed.dim(-1)]
        )

        let iclInputEmbed = concatenated([textWithCodecPad, codecWithTextPad], axis: 1)
        let trailingTextHidden = ttsPadEmbed

        let codecPrefill: [Int32] = if let langId = conditioning.codecLanguageID {
            [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        } else {
            [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        }

        var codecPrefixEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))
        let codecPrefixSuffix = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped(1, 2)
        )
        if let speakerEmbedding = conditioning.speakerEmbedding {
            // The speaker encoder runs on a Float32 mel, so its embedding is Float32. Concatenated as is, it
            // promoted the whole prompt, the KV cache and every later step to Float32 (BF16 weights cast per
            // matmul): 64 vs 21 ms per step on M5 Max (Vireo perf, 2026-09-26). Upstream mlx-audio (Python)
            // casts it to the embedding dtype for the same reason. VIREO_QWEN_F32_SPEAKER=1 keeps the old path.
            let speakerEmbed = (VireoModelSwitches.qwenF32Speaker ? speakerEmbedding : speakerEmbedding.asType(codecPrefixEmbed.dtype)).reshaped(1, 1, -1)
            codecPrefixEmbed = concatenated([codecPrefixEmbed, speakerEmbed, codecPrefixSuffix], axis: 1)
        } else {
            codecPrefixEmbed = concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        let roleEmbed = talker.textProjection(talker.getTextEmbeddings()(targetIds[0..., 0 ..< 3]))

        let padCount = codecPrefixEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0 ..< (codecPrefixEmbed.dim(1) - 1), 0...]

        let inputEmbeds = concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes)
    }

    func extractSpeakerEmbedding(_ refAudio: MLXArray) -> MLXArray? {
        guard let speakerEncoder else { return nil }

        let rawAudio: MLXArray
        if refAudio.ndim == 1 {
            rawAudio = refAudio.reshaped(1, refAudio.dim(0))
        } else if refAudio.ndim == 2 {
            if refAudio.dim(0) == 1 {
                rawAudio = refAudio
            } else {
                rawAudio = refAudio[0 ..< 1]
            }
        } else if refAudio.ndim == 3, refAudio.dim(1) == 1 {
            let squeezed = refAudio[0..., 0...]
            if squeezed.dim(0) == 1 {
                rawAudio = squeezed
            } else {
                rawAudio = squeezed[0 ..< 1]
            }
        } else {
            return nil
        }

        let batchSize = rawAudio.dim(0)
        var mels = [MLXArray]()
        mels.reserveCapacity(batchSize)

        for batch in 0 ..< batchSize {
            let waveform = rawAudio[batch]
            let mel = computeMelSpectrogram(
                audio: waveform,
                sampleRate: speakerEncoder.config.sampleRate,
                nFft: 1024,
                hopLength: 256,
                nMels: 128
            )
            mels.append(mel)
        }

        let stackedMels = stacked(mels, axis: 0)
        let embedding = speakerEncoder(stackedMels.asType(StoredVoiceWeights.halfType(speakerEncoder)))
        return embedding
    }

    func prepareGenerationInputs(
        text: String,
        language: String,
        instruct: String?,
        speaker: String? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard let tokenizer, let talkerConfig = config.talkerConfig else {
            fatalError("Tokenizer/config not loaded")
        }

        // Tokenize text with ChatML template
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = MLXArray(tokenizer.encode(text: chatText).map { Int32($0) }).reshaped(1, -1)

        // Get text embeddings
        let textEmbed = talker.textProjection(talker.getTextEmbeddings()(inputIds))

        // TTS special tokens
        let ttsTokens = MLXArray(
            [Int32(config.ttsBosTokenId), Int32(config.ttsEosTokenId), Int32(config.ttsPadTokenId)]
        ).reshaped(1, 3)
        let ttsEmbeds = talker.textProjection(talker.getTextEmbeddings()(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0 ..< 1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1 ..< 2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2 ..< 3, 0...]

        // Language ID — may be overridden by speaker dialect below
        var languageId: Int?
        if language.lowercased() != "auto", let langMap = talkerConfig.codecLanguageId {
            languageId = langMap[language.lowercased()]
        }

        // Speaker embedding (CustomVoice models only)
        var speakerEmbed: MLXArray? = nil
        if let speaker = speaker {
            if let spkIdMap = talkerConfig.spkId,
               let spkIdValue = spkIdMap[speaker.lowercased()] {
                let spkIdArray = MLXArray([Int32(spkIdValue.intValue)]).reshaped(1, 1)
                speakerEmbed = talker.getInputEmbeddings()(spkIdArray)
                voiceLog("[Qwen3TTS] CustomVoice: speaker '\(speaker)' → spkId=\(spkIdValue.intValue)")
            } else {
                voiceLog("[Qwen3TTS] CustomVoice: WARNING - speaker '\(speaker)' not found in spkId map")
            }
            if let dialectMap = talkerConfig.spkIsDialect,
               let dialectVal = dialectMap[speaker.lowercased()],
               dialectVal.isDialect,
               let dialectName = dialectVal.dialectName,
               let langMap = talkerConfig.codecLanguageId,
               let dialectLangId = langMap[dialectName] {
                voiceLog("[Qwen3TTS] CustomVoice: dialect override '\(dialectName)' langId=\(dialectLangId)")
                languageId = dialectLangId
            }
        }

        // Build codec prefix
        let codecPrefill: [Int32] = if let langId = languageId {
            [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        } else {
            [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId)
            ]
        }

        var codecEmbed = talker.getInputEmbeddings()(MLXArray(codecPrefill).reshaped(1, -1))
        let codecEmbedSuffix = talker.getInputEmbeddings()(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped(1, 2)
        )
        if let spkEmbed = speakerEmbed {
            codecEmbed = concatenated([codecEmbed, spkEmbed.reshaped([1, 1, -1]), codecEmbedSuffix], axis: 1)
            voiceLog("[Qwen3TTS] CustomVoice: codec prefix length = \(codecEmbed.dim(1))")
        } else {
            codecEmbed = concatenated([codecEmbed, codecEmbedSuffix], axis: 1)
        }

        // Instruct embedding (VoiceDesign only — not CustomVoice)
        var instructEmbed: MLXArray?
        if let instruct, !instruct.isEmpty {
            let instructText = "<|im_start|>user\n\(instruct)<|im_end|>\n"
            let instructIds = MLXArray(tokenizer.encode(text: instructText).map { Int32($0) }).reshaped(1, -1)
            instructEmbed = talker.textProjection(talker.getTextEmbeddings()(instructIds))
        }

        // Role embedding (first 3 tokens: <|im_start|>assistant\n)
        let roleEmbed = textEmbed[0..., ..<3, 0...]

        // Build pad/bos prefix
        let padCount = codecEmbed.dim(1) - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, ttsPadEmbed.dim(-1)])
        var combinedEmbed = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedEmbed = combinedEmbed + codecEmbed[0..., ..<(-1), 0...]

        // Full input embedding
        var inputEmbeds: MLXArray = if let instructEmbed {
            concatenated([instructEmbed, roleEmbed, combinedEmbed], axis: 1)
        } else {
            concatenated([roleEmbed, combinedEmbed], axis: 1)
        }

        // Add first text token (index 3) + last codec embed
        let firstTextEmbed = textEmbed[0..., 3 ..< 4, 0...] + codecEmbed[0..., (-1)..., 0...]
        inputEmbeds = concatenated([inputEmbeds, firstTextEmbed], axis: 1)

        // Trailing text (tokens 4 to -5, plus EOS)
        let trailingTextHidden = concatenated(
            [textEmbed[0..., 4 ..< (textEmbed.dim(1) - 5), 0...], ttsEosEmbed],
            axis: 1
        )

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed)
    }

    // MARK: - Token sampling

    func sampleToken(
        _ logits: MLXArray,
        temperature: Float = 0.9,
        topP: Float = 1.0,
        topK: Int = 50,
        repetitionPenalty: Float = 1.0,
        generatedTokens: [Int]? = nil,
        generatedTokenArray: MLXArray? = nil,
        suppressTokens: [Int]? = nil,
        eosTokenId: Int? = nil,
        minP: Float = 0.0
    ) -> MLXArray {
        var logitsSlice = logits[0..., (-1)..., 0...].squeezed(axis: 1) // [batch, vocab_size]

        // Suppress tokens by setting to -inf
        if let suppress = suppressTokens, !suppress.isEmpty {
            let suppressArr = MLXArray(suppress.map { Int32($0) }).reshaped(1, -1)
            let negInf = MLXArray.full([1, suppress.count], values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
            logitsSlice = putAlong(logitsSlice, suppressArr, values: negInf, axis: -1)
        }

        // Repetition penalty
        if let tokens = generatedTokens, !tokens.isEmpty, repetitionPenalty != 1.0 {
            let unique = Array(Set(tokens)).filter { $0 < logitsSlice.dim(-1) }
            if !unique.isEmpty {
                let tokenIds = MLXArray(unique.map { Int32($0) }).reshaped(1, -1)
                let selected = takeAlong(logitsSlice, tokenIds, axis: -1)
                let penalized = which(
                    selected .< 0,
                    selected * repetitionPenalty,
                    selected / repetitionPenalty
                )
                logitsSlice = putAlong(logitsSlice, tokenIds, values: penalized, axis: -1)
            }
        }

        // Vireo: the same penalty from a lazy token history [1, n] (pipelined loop). Repeated ids gather and
        // scatter identical values, so the result equals the unique-set version above.
        if let tokenIds = generatedTokenArray, tokenIds.size > 0, repetitionPenalty != 1.0 {
            let selected = takeAlong(logitsSlice, tokenIds, axis: -1)
            let penalized = which(selected .< 0, selected * repetitionPenalty, selected / repetitionPenalty)
            logitsSlice = putAlong(logitsSlice, tokenIds, values: penalized, axis: -1)
        }

        // Greedy if temperature 0
        if temperature <= 0 {
            return argMax(logitsSlice, axis: -1, keepDims: true)
        }

        // Preserve EOS logit so top-k/top-p/min-p do not permanently suppress it.
        let eosLogit: MLXArray? = if let eosTokenId, eosTokenId >= 0, eosTokenId < logitsSlice.dim(-1) {
            logitsSlice[0..., eosTokenId ..< (eosTokenId + 1)]
        } else {
            nil
        }

        // Apply top-k filtering (match mlx_lm.apply_top_k ordering and masking semantics)
        var filteredLogits = logitsSlice
        let vocabSize = logitsSlice.dim(-1)
        if topK > 0, topK < vocabSize {
            let kth = min(topK - 1, max(vocabSize - 1, 0))
            if kth >= 0 {
                let maskIdx = argPartition(-logitsSlice, kth: kth, axis: -1)[0..., topK...]
                let negInf = MLXArray.full(maskIdx.shape, values: MLXArray(-Float.infinity), dtype: logitsSlice.dtype)
                filteredLogits = putAlong(filteredLogits, maskIdx, values: negInf, axis: -1)
            }
        }

        // Apply top-p (nucleus) sampling
        if topP > 0, topP < 1.0 {
            let probs = softmax(filteredLogits, axis: -1)

            // Sort in ASCENDING order (like Python)
            let sortedIndices = argSort(filteredLogits, axis: -1)
            let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

            // Cumulative probabilities
            let cumProbs = cumsum(sortedProbs, axis: -1)

            // Rearrange cumulative probs back to original order
            // Create inverse index mapping using putAlong
            let vocabSize = sortedIndices.dim(-1)
            let arangeIndices = MLXArray(0 ..< vocabSize).reshaped(1, -1).asType(Int32.self)
            let zeros = MLXArray.zeros(sortedIndices.shape, type: Int32.self)
            let inverseIndices = putAlong(zeros, sortedIndices, values: arangeIndices, axis: -1)
            let cumProbsOrigOrder = takeAlong(cumProbs, inverseIndices, axis: -1)

            // Mask tokens where cumulative prob > (1 - top_p)
            // Keep tokens that are in the top_p nucleus
            let threshold = 1.0 - topP
            let mask = cumProbsOrigOrder .> threshold
            let negInf = MLXArray.full(filteredLogits.shape, values: MLXArray(-Float.infinity), dtype: filteredLogits.dtype)
            filteredLogits = which(mask, filteredLogits, negInf)
        }

        // Apply min-p sampling behavior (default kept at 0.0 for now)
        if minP > 0.0 {
            let scaledMinP = Float(log(Double(minP)))
            // Indices sorted in descending order (like Python `argsort(-logits)`)
            let sortedIndices = argSort(-filteredLogits, axis: -1)
            let sortedLogits = takeAlong(filteredLogits, sortedIndices, axis: -1)
            let topLogits = sortedLogits[0..., 0 ..< 1]
            let scaledMinPArray = MLXArray.full(
                topLogits.shape,
                values: MLXArray(scaledMinP),
                dtype: sortedLogits.dtype
            ) + topLogits
            let removeMask = sortedLogits .< scaledMinPArray
            let negInf = MLXArray.full(sortedLogits.shape, values: MLXArray(-Float.infinity), dtype: sortedLogits.dtype)
            let filteredSortedLogits = which(removeMask, negInf, sortedLogits)

            let invArange = MLXArray(0 ..< vocabSize).reshaped(1, -1).asType(Int32.self)
            let inverseIndices = putAlong(MLXArray.zeros(sortedIndices.shape, type: Int32.self), sortedIndices, values: invArange, axis: -1)
            filteredLogits = takeAlong(filteredSortedLogits, inverseIndices, axis: -1)
        }

        if let eosLogit, let eosTokenId {
            let eosIdx = MLXArray([Int32(eosTokenId)]).reshaped(1, 1)
            filteredLogits = putAlong(filteredLogits, eosIdx, values: eosLogit, axis: -1)
        }

        // Sample with temperature
        let token = categorical(filteredLogits / temperature)
        return token.reshaped(1, 1)
    }

    // MARK: - fromPretrained

    public static func fromModelDirectory(_ modelDir: URL) async throws -> Qwen3TTSModel {
        // Load main config
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let intake = StoredVoiceWeights.isIntake(configData)
        let config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: configData)

        let model = Qwen3TTSModel(config: config)

        // Load talker weights
        var allWeights = [String: MLXArray]()
        let fm = FileManager.default
        let modelFiles = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        for file in modelFiles where file.pathExtension == "safetensors" {
            let weights = try MLX.loadArrays(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        // Sanitize and load talker weights
        let talkerWeights = Qwen3TTSTalkerForConditionalGeneration.sanitize(weights: allWeights)
        let talkerPairs = talkerWeights.map { ($0.key, $0.value) }

        // Quantized checkpoints store packed weights and companion .scales tensors.
        // Convert talker Linear layers before loading those tensors.
        if intake {
            try StoredVoiceWeights.prepare(model.talker, weights: talkerWeights)
        } else if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model.talker) { path, _ in
                guard talkerWeights["\(path).scales"] != nil else {
                    return nil
                }

                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path) {
                    return layerQuant.asTuple
                }

                return config.quantization?.asTuple
            }
        }

        try model.talker.update(parameters: ModuleParameters.unflattened(talkerPairs), verify: .all)
        eval(model.talker.parameters())

        // Generate tokenizer.json if missing (Qwen3-TTS ships without it)
        let tokenizerDirectory = fm.temporaryDirectory.appendingPathComponent("vireo-tokenizer-" + UUID().uuidString)
        try fm.createDirectory(at: tokenizerDirectory, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tokenizerDirectory) }
        for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"] {
            let source = modelDir.appendingPathComponent(name)
            if fm.fileExists(atPath: source.path) { try fm.copyItem(at: source, to: tokenizerDirectory.appendingPathComponent(name)) }
        }
        let tokenizerJsonPath = tokenizerDirectory.appendingPathComponent("tokenizer.json")

        if !fm.fileExists(atPath: tokenizerJsonPath.path) {
            let vocabPath = modelDir.appendingPathComponent("vocab.json")
            let mergesPath = modelDir.appendingPathComponent("merges.txt")
            let hasVocab = fm.fileExists(atPath: vocabPath.path)
            let hasMerges = fm.fileExists(atPath: mergesPath.path)
            if hasVocab, hasMerges {
                do {
                    try generateTokenizerJson(
                        vocabPath: vocabPath,
                        mergesPath: mergesPath,
                        tokenizerConfigPath: modelDir.appendingPathComponent("tokenizer_config.json"),
                        outputPath: tokenizerJsonPath
                    )
                    voiceLog("Generated tokenizer.json from vocab.json + merges.txt")
                } catch {
                    voiceLog("Warning: Failed to generate tokenizer.json: \(error)")
                }
            } else {
                voiceLog("Warning: Cannot generate tokenizer.json — vocab.json: \(hasVocab), merges.txt: \(hasMerges)")
            }
        }

        // Load tokenizer
        do {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        } catch {
            voiceLog("Warning: Could not load tokenizer: \(error)")
        }

        // Load speech tokenizer — check that it's a directory, not a stale file
        let speechTokenizerPath = modelDir.appendingPathComponent("speech_tokenizer")
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: speechTokenizerPath.path, isDirectory: &isDir), isDir.boolValue {
            try loadSpeechTokenizer(model: model, path: speechTokenizerPath, intake: intake)
        } else if fm.fileExists(atPath: speechTokenizerPath.path) {
            // speech_tokenizer exists but is not a directory — stale cache
            // Remove the entire model cache so it re-downloads cleanly next time
            voiceLog("speech_tokenizer is not a directory (stale cache), clearing model cache...")
            // Do not mutate a shared HF snapshot on loader failure.
            throw AudioGenerationError.modelNotInitialized(
                "Model cache was corrupted (speech_tokenizer). It has been cleared. Please try loading again."
            )
        } else {
            voiceLog("Warning: speech_tokenizer directory not found, speech decoding unavailable")
        }

        // Load speaker encoder for base models when available
        if config.ttsModelType == "base" {
            let speakerWeights = Qwen3TTSSpeakerEncoder.sanitize(weights: allWeights)
            if !speakerWeights.isEmpty {
                if let speakerEncoder = model.speakerEncoder {
                    if intake { try StoredVoiceWeights.prepare(speakerEncoder, weights: speakerWeights) }
                    let speakerPairs = speakerWeights.map { ($0.key, $0.value) }
                    try speakerEncoder.update(parameters: ModuleParameters.unflattened(speakerPairs), verify: .all)
                    eval(speakerEncoder.parameters())
                }
            }
            if model.speakerEncoder != nil {
                voiceLog("Loaded speaker encoder")
            } else {
                voiceLog("Warning: speaker encoder config missing, skipping speaker encoder load")
            }
        }

        voiceLog("Loaded Qwen3-TTS model (\(config.ttsModelType))")
        return model
    }

    private static func loadSpeechTokenizer(model: Qwen3TTSModel, path: URL, intake: Bool = false) throws {
        // Load config — fall back to defaults if config.json is missing
        let tokenizerConfig: Qwen3TTSTokenizerConfig
        let configPath = path.appendingPathComponent("config.json")
        if let configData = try? Data(contentsOf: configPath) {
            tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: configData)
        } else {
            voiceLog("Warning: speech_tokenizer/config.json not found, using defaults")
            let defaultJson = "{}".data(using: .utf8)!
            tokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: defaultJson)
        }

        let speechTokenizer = Qwen3TTSSpeechTokenizer(config: tokenizerConfig)

        // Load weights
        var tokenizerWeights = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            let weights = try MLX.loadArrays(url: file)
            tokenizerWeights.merge(weights) { _, new in new }
        }

        if !tokenizerWeights.isEmpty {
            let sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights: tokenizerWeights)
            if intake { try StoredVoiceWeights.prepare(speechTokenizer, weights: sanitized) }
            let pairs = sanitized.map { ($0.key, $0.value) }
            try speechTokenizer.update(parameters: ModuleParameters.unflattened(pairs), verify: .all)
            eval(speechTokenizer.parameters())
        }

        model.speechTokenizer = speechTokenizer
        voiceLog("Loaded speech tokenizer decoder")
    }

    // MARK: - Generate tokenizer.json from vocab.json + merges.txt

    /// Qwen3-TTS repos ship with a slow tokenizer (vocab.json + merges.txt) but
    /// swift-transformers requires tokenizer.json (fast tokenizer format). This
    /// generates the fast tokenizer JSON from the available files.
    private static func generateTokenizerJson(
        vocabPath: URL,
        mergesPath: URL,
        tokenizerConfigPath: URL,
        outputPath: URL
    ) throws {
        // Read vocab
        let vocabData = try Data(contentsOf: vocabPath)
        let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]

        // Read merges (skip header line "#version: ...")
        let mergesText = try String(contentsOf: mergesPath, encoding: .utf8)
        let mergeLines = mergesText.components(separatedBy: .newlines)
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        // Read added_tokens from tokenizer_config.json
        var addedTokens = [[String: Any]]()
        if let configData = try? Data(contentsOf: tokenizerConfigPath),
           let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: [String: Any]] {
            for (idStr, tokenInfo) in addedTokensDecoder {
                guard let tokenId = Int(idStr),
                      let content = tokenInfo["content"] as? String else { continue }
                let entry: [String: Any] = [
                    "id": tokenId,
                    "content": content,
                    "single_word": tokenInfo["single_word"] as? Bool ?? false,
                    "lstrip": tokenInfo["lstrip"] as? Bool ?? false,
                    "rstrip": tokenInfo["rstrip"] as? Bool ?? false,
                    "normalized": tokenInfo["normalized"] as? Bool ?? false,
                    "special": tokenInfo["special"] as? Bool ?? true
                ]
                addedTokens.append(entry)
            }
            addedTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
        }

        // Build tokenizer.json
        // Qwen2 uses ByteLevel BPE with a GPT-2-style regex pre-tokenizer
        let tokenizerJson: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": addedTokens,
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    [
                        "type": "Split",
                        "pattern": [
                            "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                        ],
                        "behavior": "Isolated",
                        "invert": false
                    ] as [String: Any],
                    [
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    ] as [String: Any]
                ] as [[String: Any]]
            ] as [String: Any],
            "post_processor": NSNull(),
            "decoder": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            ] as [String: Any],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": NSNull(),
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocabDict,
                "merges": mergeLines
            ] as [String: Any]
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: tokenizerJson, options: [.sortedKeys])
        try jsonData.write(to: outputPath)
    }
}

private func voiceLog(_ message: String) { FileHandle.standardError.write(Data((message + "\n").utf8)) }
