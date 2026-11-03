import Foundation
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import MLXRandom
import NaturalLanguage

public final class KokoroModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public private(set) var storedHalf: DType?
    public let config: KokoroConfig
    public let sampleRate: Int
    public let defaultGenerationParameters: GenerateParameters

    @ModuleInfo var bert: Albert
    @ModuleInfo(key: "bert_encoder") var bertEncoder: Linear
    @ModuleInfo var predictor: KokoroProsodyPredictor
    @ModuleInfo(key: "text_encoder") var textEncoder: KokoroTextEncoder
    @ModuleInfo var decoder: KokoroDecoder

    private let modelDirectory: URL?
    private var voiceCache: [String: MLXArray] = [:]
    private let voiceCacheLock = NSLock()
    private let maxTokenCount = 510
    public private(set) var textProcessor: TextProcessor?

    /// Speed multiplier for speech rate. Higher = faster.
    public var speed: Float = 1.0

    /// Vireo: optimized path (every option stock ops only; chosen by the adapter after `fastPathSelfTest`).
    public private(set) var fastOptions = FastOptions.stock
    /// Vireo: stderr profile line per sentence (VIREO_KOKORO_PROFILE via the adapter).
    public var profile = false
    /// Vireo lab (VIREO_KOKORO_PROFILE=2): also sync before the decoder to split its time (adds one host sync).
    public var profileSplit = false
    /// Vireo: checked between sentences, so a stopped request ends early.
    public var cancellationRequested: (() -> Bool)?
    /// Vireo lab: poison one sample of the optimized path's output (fallback test; never in the self-test).
    public var injectNonFinite = false

    private init(config: KokoroConfig, modelDirectory: URL? = nil, textProcessor: TextProcessor? = nil) {
        self.config = config
        self.modelDirectory = modelDirectory
        self.textProcessor = textProcessor
        self.sampleRate = config.sampleRate
        self.defaultGenerationParameters = GenerateParameters(temperature: 0, topP: 1)

        _bert = ModuleInfo(wrappedValue: Albert(config: config.plbert, vocabSize: config.nToken))
        _bertEncoder = ModuleInfo(
            wrappedValue: Linear(config.plbert.hiddenSize, config.hiddenDim),
            key: "bert_encoder"
        )
        _predictor = ModuleInfo(wrappedValue: KokoroProsodyPredictor(
            styleDim: config.styleDim, dHid: config.hiddenDim,
            nLayers: config.nLayer, maxDur: config.maxDur, dropout: 0.0
        ))
        _textEncoder = ModuleInfo(wrappedValue: KokoroTextEncoder(
            channels: config.hiddenDim, kernelSize: config.textEncoderKernelSize,
            depth: config.nLayer, nSymbols: config.nToken
        ), key: "text_encoder")
        _decoder = ModuleInfo(wrappedValue: KokoroDecoder(config: config))
    }

    /// For testing: create model from config without loading weights
    static func testInit(config: KokoroConfig) -> KokoroModel {
        KokoroModel(config: config)
    }

    // MARK: - Forward Pass

    func callAsFunction(
        inputIds: MLXArray, refS: MLXArray, speed: Float = 1.0
    ) -> (audio: MLXArray, predDur: MLXArray) {
        forward(inputIds: inputIds, refS: refS, speed: speed, options: fastOptions)
    }

    func forward(
        inputIds: MLXArray, refS: MLXArray, speed: Float, options: FastOptions, timing: ((String) -> Void)? = nil
    ) -> (audio: MLXArray, predDur: MLXArray) {
        let seqLen = inputIds.shape[inputIds.ndim - 1]
        let inputLengths = MLXArray([Int32(seqLen)])
        var textMask = MLXArray(Array(0..<Int32(seqLen))).reshaped([1, -1])
        textMask = (textMask + 1) .> inputLengths.reshaped([-1, 1])

        let attMask = MLXArray(1) - textMask.asType(.int32)
        let (bertOut, _) = bert(inputIds, attentionMask: attMask)
        let dEn = bertEncoder(bertOut).transposed(0, 2, 1)

        let globalStyle = refS[0..., 128...]
        let acousticStyle = refS[0..., ..<128]

        let d = predictor.textEncoder(dEn, style: globalStyle, textLengths: inputLengths, mask: textMask)
        let (x, _) = predictor.lstm(d)
        let duration = predictor.durationProj(x)
        // Defensive: quantized encoders may produce NaN durations for certain inputs.
        // Cap at maxFramesPerPhoneme to prevent OOM from garbage int32 casts.
        let maxFramesPerPhoneme = 100
        let durRaw = MLX.sigmoid(duration).sum(axis: -1) / speed
        let durSafe = nanToNum(durRaw, nan: 1.0)
        let predDur = MLX.clip(MLX.round(durSafe), min: 1, max: Float(maxFramesPerPhoneme))
            .asType(.int32)[0]

        let durArray: [Int32] = predDur.asArray(Int32.self)
        timing?("durations")
        let counts = durArray.map { min(max(Int($0), 0), maxFramesPerPhoneme) }
        let frames = counts.reduce(0, +)

        // All durations collapsed to zero — return silence instead of crashing on empty concat
        guard frames > 0 else {
            let silence = MLXArray.zeros([1, 1])
            return (silence, predDur)
        }
        let predAln: MLXArray
        if options.hostAlignment {
            // Vireo: the same one-hot alignment [1, tokens, frames], built on the host in one upload (stock: one
            // `repeated` op per phoneme, a concatenation and a scatter).
            var onehot = [Float](repeating: 0, count: seqLen * frames)
            var frame = 0
            for (i, n) in counts.enumerated() {
                for _ in 0..<n { onehot[i * frames + frame] = 1; frame += 1 }
            }
            predAln = MLXArray(onehot, [1, seqLen, frames])
        } else {
            var indices = [MLXArray]()
            for (i, count) in counts.enumerated() where count > 0 {
                indices.append(MLX.repeated(MLXArray(Int32(i)), count: count))
            }
            let allIndices = MLX.concatenated(indices, axis: 0)

            let predAlnTrg = MLXArray.zeros([inputIds.shape[1], allIndices.shape[0]])
            predAlnTrg[allIndices, MLXArray(Array(0..<Int32(allIndices.shape[0])))] = MLXArray(Float(1))
            predAln = predAlnTrg.expandedDimensions(axis: 0)
        }

        let en = MLX.matmul(d.transposed(0, 2, 1), predAln.asType(d.dtype))
        let (f0Pred, nPred) = predictor.predict(en, globalStyle)

        let tEn = textEncoder(inputIds, inputLengths: inputLengths, mask: textMask)
        let asr = MLX.matmul(tEn, predAln.asType(tEn.dtype))
        if profileSplit, let timing { eval(asr, f0Pred, nPred); timing("predictor") }

        let audio = decoder(asr, f0Pred, nPred, acousticStyle)
        return (audio[0], predDur)
    }

    // MARK: - Tokenizer

    func tokenize(_ text: String) -> [Int] {
        // Iterate Unicode scalars, not Characters. Swift's `for ch in String`
        // returns extended grapheme clusters, which fuse combining marks into
        // their base character. The IPA lexicon emits French nasal vowels as
        // base vowel + U+0303 COMBINING TILDE (e.g. "ɔ" + "̃"), which Swift
        // collapses into a single Character "ɔ̃" that isn't in `config.vocab`
        // — so `compactMap` silently drops the entire nasal vowel and
        // "bonjour" comes out as "bjour". Iterating scalars preserves the
        // base + combining pair as two separate tokens, both of which exist
        // in the vocab. See https://github.com/Blaizzy/mlx-audio-swift/issues/187
        text.unicodeScalars.compactMap { config.vocab[String($0)] }
    }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        var parts: [MLXArray] = []
        for try await event in generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
                                              generationParameters: generationParameters) {
            if case .audio(let audio) = event { parts.append(audio) }
        }
        guard !parts.isEmpty else { return MLXArray.zeros([0]) }
        return parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 0)
    }

    /// Vireo: one piece of text the model speaks in one forward pass.
    struct Segment { var text: String; var phonemes: String; var tokens: [Int] }

    /// Vireo: `text` as sentences (NaturalLanguage sentence boundaries), each phonemized and tokenized; a sentence
    /// longer than the model's 510-token context is split at the clause boundary (then the space) nearest its
    /// middle, so no input is refused for length (upstream threw; mlx-audio-swift #91). Empty pieces are dropped.
    func segments(_ text: String, language: String?) throws -> [Segment] {
        var sentences: [String] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { sentences.append(sentence) }
            return true
        }
        if sentences.isEmpty, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { sentences = [text] }
        var result: [Segment] = []
        for sentence in sentences { try fit(sentence, language: language, depth: 0, into: &result) }
        return result
    }

    private func fit(_ text: String, language: String?, depth: Int, into result: inout [Segment]) throws {
        let phonemes = try textProcessor?.process(text: text, language: language) ?? text
        let tokens = tokenize(phonemes)
        if tokens.isEmpty { return }
        if tokens.count <= maxTokenCount { result.append(Segment(text: text, phonemes: phonemes, tokens: tokens)); return }
        let characters = Array(text)
        let middle = characters.count / 2
        func nearest(_ isBoundary: (Character) -> Bool) -> Int? {
            let candidates = characters.indices.filter { $0 > 0 && $0 < characters.count - 1 && isBoundary(characters[$0]) }
            return candidates.min { abs($0 - middle) < abs($1 - middle) }
        }
        guard depth < 24, let cut = nearest({ ",;:\u{2014}\u{2013}".contains($0) }) ?? nearest({ $0.isWhitespace }) else {
            // One unbroken run longer than the context: keep what fits.
            result.append(Segment(text: text, phonemes: phonemes, tokens: Array(tokens.prefix(maxTokenCount)))); return
        }
        let head = String(characters[...cut]).trimmingCharacters(in: .whitespacesAndNewlines)
        let tail = String(characters[(cut + 1)...]).trimmingCharacters(in: .whitespacesAndNewlines)
        if !head.isEmpty { try fit(head, language: language, depth: depth + 1, into: &result) }
        if !tail.isEmpty { try fit(tail, language: language, depth: depth + 1, into: &result) }
    }

    /// Vireo: the voice's style table and the language its text is phonemized in (prepared).
    func voiceAndLanguage(voice: String?, refAudio: MLXArray?, language: String?) async throws -> (MLXArray, String?) {
        let voiceName = voice ?? "af_heart"
        let voiceEmb = try refAudio ?? loadVoice(named: voiceName)
        let inferredLang = language ?? KokoroMultilingualProcessor.languageForVoice(voiceName)
        if let multilingual = textProcessor as? KokoroMultilingualProcessor, let lang = inferredLang {
            try await multilingual.prepare(for: lang)
        }
        return (voiceEmb, inferredLang)
    }

    /// Vireo: one segment through the model, evaluated (the lazy graph finishes on its producing executor before it
    /// crosses an await or stream boundary).
    func synthesize(_ tokens: [Int], voiceEmb: MLXArray, options: FastOptions, timing: ((String) -> Void)? = nil) -> MLXArray {
        var tokenArray = [Int32(0)]
        tokenArray.append(contentsOf: tokens.map { Int32($0) })
        tokenArray.append(0)
        let inputIds = MLXArray(tokenArray).reshaped([1, -1])
        let refIdx = min(tokens.count, voiceEmb.shape[0] - 1)
        let refS = voiceEmb[refIdx..<(refIdx + 1)]
        let (audio, _) = forward(inputIds: inputIds, refS: refS, speed: speed, options: options, timing: timing)
        let samples = audio.reshaped([-1])
        eval(samples)
        return samples
    }

    /// Vireo: streams one audio chunk per sentence (upstream generated the whole text in one pass and refused more
    /// than 510 phoneme tokens), so first audio is the first sentence, not the whole request.
    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let (voiceEmb, lang) = try await self.voiceAndLanguage(voice: voice, refAudio: refAudio, language: language)
                var clock = Date()
                let segments = try self.segments(text, language: lang)
                let g2p = Date().timeIntervalSince(clock) * 1000
                for (index, segment) in segments.enumerated() {
                    try Task.checkCancellation()
                    if self.cancellationRequested?() == true { break }
                    let options = self.fastOptions
                    clock = Date()
                    var marks: [String] = []
                    let timing: ((String) -> Void)? = self.profile ? { marks.append("\($0) \(String(format: "%.1f", Date().timeIntervalSince(clock) * 1000)) ms") } : nil
                    var audio = self.synthesize(segment.tokens, voiceEmb: voiceEmb, options: options, timing: timing)
                    if self.profile {
                        let ms = Date().timeIntervalSince(clock) * 1000
                        let seconds = Double(audio.shape[0]) / Double(self.sampleRate)
                        FileHandle.standardError.write(Data(("[kokoro profile] segment \(index + 1)/\(segments.count) \(segment.tokens.count) tokens"
                            + (index == 0 ? " g2p(all) \(String(format: "%.1f", g2p)) ms" : "") + " · " + marks.joined(separator: " · ")
                            + " · total \(String(format: "%.1f", ms)) ms · \(String(format: "%.2f", seconds)) s audio · \(options.label)\n").utf8))
                    }
                    if self.injectNonFinite, options != .stock, audio.shape[0] > 0 {
                        var samples = audio.asArray(Float.self); samples[samples.count / 2] = .nan; audio = MLXArray(samples)
                    }
                    continuation.yield(.audio(audio))
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    // MARK: - Vireo fast path

    /// Graph-level optimizations from stock MLX ops; each has its own switch in the adapter.
    public struct FastOptions: Equatable, Sendable {
        /// One-hot alignment built on the host (exact: same values; stock builds it with one op per phoneme).
        public var hostAlignment: Bool
        /// Weight-normalized conv weights computed once, not on every call (exact: same ops, same values).
        public var convWeightCache: Bool
        /// BiLSTM: both directions as one batch per step, gates via `addmm` and one compiled cell update.
        public var fusedLSTM: Bool
        /// Decoder (HiFi-GAN/iSTFTNet conv stack) in fp16: parameters and cached conv weights cast once, harmonic source,
        /// STFT, spectrogram head and iSTFT kept fp32. Inexact (rounding), so gated by a tolerance, not bit-identity.
        /// Needs `convWeightCache` (the fp16 conv weights are the cached ones).
        public var halfDecoder: Bool
        public init(hostAlignment: Bool, convWeightCache: Bool, fusedLSTM: Bool, halfDecoder: Bool = false) {
            self.hostAlignment = hostAlignment; self.convWeightCache = convWeightCache; self.fusedLSTM = fusedLSTM
            self.halfDecoder = halfDecoder && convWeightCache
        }
        public static let stock = FastOptions(hostAlignment: false, convWeightCache: false, fusedLSTM: false)
        public var isStock: Bool { self == .stock }
        /// The same options without the inexact fp16 decoder (what must still match stock to 60 dB).
        public var exactPart: FastOptions { FastOptions(hostAlignment: hostAlignment, convWeightCache: convWeightCache, fusedLSTM: fusedLSTM) }
        public var label: String {
            isStock ? "stock" : [hostAlignment ? "host-alignment" : nil, convWeightCache ? "conv-weight-cache" : nil,
                                 fusedLSTM ? "fused-lstm" : nil, halfDecoder ? "fp16-decoder" : nil].compactMap { $0 }.joined(separator: " + ")
        }
    }
    public static let fastPathVersion = "kokoro-fast-2"

    /// The decoder's parameters as loaded, while the fp16 decoder serves (restored exactly by `setFastOptions`).
    private var decoderOriginals: [(String, MLXArray)]?

    /// Installs a path (reversible: stock drops the cached conv weights and restores the loaded decoder parameters).
    public func setFastOptions(_ options: FastOptions) {
        fastOptions = options
        setHalfDecoder(storedHalf == nil && options.halfDecoder)
        let decoderConvs = Set(decoder.modules().compactMap { $0 as? WeightNormedConv }.map(ObjectIdentifier.init))
        for case let conv as WeightNormedConv in modules() {
            let half = storedHalf == nil && options.halfDecoder && decoderConvs.contains(ObjectIdentifier(conv))
            conv.cacheWeight(options.convWeightCache, dtype: half ? .float16 : nil)
        }
        for case let lstm as BiLSTM in modules() { lstm.fused = options.fusedLSTM }
    }

    /// fp16 copies of the decoder's float parameters (quantized AdaIN projections: their scales/biases; the packed
    /// weights stay), except the harmonic source's projection (fp32 source path) and the weight-norm `weight_g`/`weight_v`
    /// (the fp16 conv weights are the cached ones, normalized from them in fp32). The loaded arrays are kept and put
    /// back on `false`, so stock after fp16 is exactly the loaded model.
    private func setHalfDecoder(_ on: Bool) {
        if on, decoderOriginals == nil {
            var originals: [(String, MLXArray)] = []
            var half: [(String, MLXArray)] = []
            for (key, value) in decoder.parameters().flattened()
            where (value.dtype == .float32 || value.dtype == .bfloat16) && !key.hasPrefix("generator.m_source.")
                && !key.hasSuffix("weight_g") && !key.hasSuffix("weight_v") {
                // `update` replaces each parameter's contents in place: keep a separate handle to the loaded array.
                originals.append((key, value.reshaped(value.shape)))
                half.append((key, value.asType(.float16)))
            }
            eval(half.map(\.1) + originals.map(\.1))
            decoder.update(parameters: ModuleParameters.unflattened(half))
            decoderOriginals = originals
        } else if !on, let originals = decoderOriginals {
            decoder.update(parameters: ModuleParameters.unflattened(originals))
            decoderOriginals = nil
        }
        decoder.halfPrecision = on
    }

    /// Bound for the inexact fp16 decoder (waveform SNR against the same options in fp32, same seed). Calibrated on M5 Max
    /// (29 Sep): fp16 41.8-43.4 dB; the same fp32 path at another seed 19-20 dB; injected faults bf16 decoder 25.3-25.7,
    /// fp16 F0 curve 16-25, non-finite instance-norm statistics, a dropped noise branch -0.7 dB. 35 dB sits between.
    public static let halfDecoderMinSNR: Double = 35

    /// Stock vs `options` on fixed sentences and seeds (the source module draws noise): lengths must match, the exact
    /// components must agree with stock to `minSNR` dB, and the fp16 decoder (if chosen) with the same options in fp32
    /// to `halfDecoderMinSNR`; returns a one-line detail, throws `FastPathFailure` otherwise.
    public func fastPathSelfTest(_ options: FastOptions, minSNR: Double = 60) throws -> String {
        guard let processor = textProcessor else { throw FastPathFailure("no phonemizer") }
        let voiceEmb = try loadVoice(named: "af_heart")
        let texts = ["Checking the fast path, one two three.",
                     "The backup finished a few minutes ago, and the report is waiting in your downloads folder; nothing else changed overnight."]
        let previous = fastOptions
        defer { setFastOptions(previous) }
        let exact = options.exactPart
        var worst = Double.infinity, worstHalf = Double.infinity
        var identical = 0
        var total = 0
        func snr(_ a: [Float], _ reference: [Float]) -> Double {
            var signal = 0.0, noise = 0.0
            for k in reference.indices { let d = Double(a[k] - reference[k]); signal += Double(reference[k]) * Double(reference[k]); noise += d * d }
            return 10 * log10(signal / Swift.max(noise, 1e-30))
        }
        for (i, text) in texts.enumerated() {
            let tokens = tokenize(try processor.process(text: text, language: "en-us"))
            func run(_ o: FastOptions) -> [Float] {
                setFastOptions(o)
                MLXRandom.seed(20_260_927 + UInt64(i))
                return synthesize(tokens, voiceEmb: voiceEmb, options: o).asArray(Float.self)
            }
            let stock = run(.stock)
            let fast = exact.isStock ? stock : run(exact)
            guard fast.allSatisfy(\.isFinite) else { throw FastPathFailure("non-finite audio on sentence \(i + 1)") }
            guard fast.count == stock.count else { throw FastPathFailure("length differs from stock on sentence \(i + 1) (\(fast.count) vs \(stock.count) samples)") }
            total += stock.count
            if fast == stock {
                identical += stock.count
            } else {
                let db = snr(fast, stock)
                worst = Swift.min(worst, db)
                guard db >= minSNR else { throw FastPathFailure(String(format: "sentence %d: %.1f dB vs stock (bound %.0f dB)", i + 1, db, minSNR)) }
            }
            guard options.halfDecoder else { continue }
            let half = run(options)
            guard half.allSatisfy(\.isFinite) else { throw FastPathFailure("fp16 decoder: non-finite audio on sentence \(i + 1)") }
            guard half.count == fast.count else { throw FastPathFailure("fp16 decoder: length differs on sentence \(i + 1) (\(half.count) vs \(fast.count) samples)") }
            let db = snr(half, fast)
            worstHalf = Swift.min(worstHalf, db)
            guard db >= Self.halfDecoderMinSNR else {
                throw FastPathFailure(String(format: "fp16 decoder, sentence %d: %.1f dB vs fp32 (bound %.0f dB)", i + 1, db, Self.halfDecoderMinSNR))
            }
        }
        var detail = identical == total ? "\(total) samples bit-identical to stock" : String(format: "%d samples, %.1f dB vs stock (bound %.0f dB)", total, worst, minSNR)
        if options.halfDecoder { detail += String(format: "; fp16 decoder %.1f dB vs fp32 (bound %.0f dB)", worstHalf, Self.halfDecoderMinSNR) }
        return detail
    }

    /// Vireo: the first run in a process compiles Metal pipelines and fills the phonemizer; do it at load.
    public func warmUp(voice: String = "af_heart") throws -> String {
        guard let processor = textProcessor else { return "no phonemizer" }
        var clock = Date()
        let tokens = tokenize(try processor.process(text: "Ready.", language: KokoroMultilingualProcessor.languageForVoice(voice)))
        let g2p = Date().timeIntervalSince(clock)
        clock = Date()
        _ = synthesize(tokens, voiceEmb: try loadVoice(named: voice), options: fastOptions)
        return String(format: "phonemizer %.2f s, first pass %.2f s", g2p, Date().timeIntervalSince(clock))
    }

    // MARK: - Voice Loading

    public func loadVoice(named name: String) throws -> MLXArray {
        if let cached = voiceCacheLock.withLock({ voiceCache[name] }) { return cached }

        guard let dir = modelDirectory else {
            throw AudioGenerationError.invalidInput("Voice '\(name)' not available: no model directory")
        }

        let voicePath = dir.appendingPathComponent("voices/\(name).safetensors")
        guard FileManager.default.fileExists(atPath: voicePath.path) else {
            let available = availableVoices().joined(separator: ", ")
            throw AudioGenerationError.invalidInput(
                "Voice '\(name)' not found. Available: \(available)")
        }

        let voiceWeights = try MLX.loadArrays(url: voicePath)
        guard let voiceArray = voiceWeights["voice"] ?? voiceWeights.values.first else {
            throw AudioGenerationError.invalidInput("Voice '\(name)' file has no voice data")
        }

        var voice = storedHalf.map { voiceArray.asType($0) } ?? voiceArray.asType(.float32)
        if voice.ndim == 3 { voice = voice.squeezed(axis: 1) }
        voiceCacheLock.withLock { voiceCache[name] = voice }
        return voice
    }

    public func availableVoices() -> [String] {
        guard let dir = modelDirectory else { return [] }
        let voicesDir = dir.appendingPathComponent("voices")
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: voicesDir.path) else {
            return []
        }
        return files
            .filter { $0.hasSuffix(".safetensors") }
            .map { String($0.dropLast(".safetensors".count)) }
            .sorted()
    }

    public func setTextProcessor(_ processor: TextProcessor?) {
        self.textProcessor = processor
    }

    // MARK: - Loading

    public static func fromModelDirectory(
        _ modelDir: URL,
        textProcessor: TextProcessor? = nil
    ) async throws -> KokoroModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(KokoroConfig.self, from: configData)
        let model = KokoroModel(config: config, modelDirectory: modelDir, textProcessor: textProcessor)

        let weights = try loadWeights(modelDir: modelDir)
        let configObject = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        let metadata = (configObject?["arsenal_voice_intake"] ?? configObject?["mobius_voice_intake"]) as? [String: Any] // legacy-compat: read in snapshots written before the rename
        let sanitized = metadata == nil ? model.sanitize(weights: weights) : weights
        if let half = metadata?["half"] as? String { model.storedHalf = half == "fp16" ? .float16 : .bfloat16 }

        if metadata != nil {
            _ = try StoredVoiceWeights.prepare(model, weights: sanitized)
        } else if let quant = config.quantization?.asTuple {
            quantizeTree(
                module: model, weights: sanitized,
                quantization: quant
            )
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: metadata == nil ? .noUnusedKeys : .all)
        if let dtype = model.storedHalf {
            guard model.parameters().flattened().allSatisfy({ !$0.1.dtype.isFloatingPoint || $0.1.dtype == dtype }) else {
                throw AudioGenerationError.invalidInput("Stored Kokoro learned weights must remain selected half")
            }
        }

        try await model.textProcessor?.prepare()

        model.train(false)
        MLX.eval(model.parameters())
        return model
    }

    // MARK: - Weight Sanitization

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] { Self.sanitizeWeights(weights) }

    /// Vireo: the checkpoint → MLX layout mapping, without a model (tier derivation writes the MLX layout, which
    /// `sanitize` leaves alone once packed quantized weights are present).
    public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let hasPackedQuantizedWeights = weights.keys.contains {
            $0.hasSuffix(".scales") || $0.hasSuffix(".biases")
        }
        let needsConvTranspose = !hasPackedQuantizedWeights

        var result = [String: MLXArray]()
        for (k, v) in weights {
            if k.contains("position_ids") { continue }

            var nk = k

            // LSTM: remap reverse FIRST to avoid partial match with forward keys
            nk = nk.replacingOccurrences(of: ".weight_ih_l0_reverse", with: ".Wx_backward")
            nk = nk.replacingOccurrences(of: ".weight_hh_l0_reverse", with: ".Wh_backward")
            nk = nk.replacingOccurrences(of: ".bias_ih_l0_reverse", with: ".bias_ih_backward")
            nk = nk.replacingOccurrences(of: ".bias_hh_l0_reverse", with: ".bias_hh_backward")
            nk = nk.replacingOccurrences(of: ".weight_ih_l0", with: ".Wx_forward")
            nk = nk.replacingOccurrences(of: ".weight_hh_l0", with: ".Wh_forward")
            nk = nk.replacingOccurrences(of: ".bias_ih_l0", with: ".bias_ih_forward")
            nk = nk.replacingOccurrences(of: ".bias_hh_l0", with: ".bias_hh_forward")

            nk = nk.replacingOccurrences(of: ".gamma", with: ".weight")
            nk = nk.replacingOccurrences(of: ".beta", with: ".bias")
            nk = nk.replacingOccurrences(of: ".alpha1.", with: ".alpha1_")
            nk = nk.replacingOccurrences(of: ".alpha2.", with: ".alpha2_")

            var value = v
            if needsConvTranspose {
                if (nk.contains("F0_proj.weight") || nk.contains("N_proj.weight")) && v.ndim == 3 {
                    value = v.transposed(0, 2, 1)
                } else if nk.contains("noise_convs") && nk.hasSuffix(".weight") && v.ndim == 3 {
                    value = v.transposed(0, 2, 1)
                } else if nk.hasSuffix("weight_v") && v.ndim == 3 {
                    let (o, h, w) = (v.shape[0], v.shape[1], v.shape[2])
                    if !(o >= h && o >= w && h == w) {
                        value = v.transposed(0, 2, 1)
                    }
                }
            }

            result[nk] = value
        }
        return result
    }

    // MARK: - Weight Loading

    private static func loadWeights(modelDir: URL) throws -> [String: MLXArray] {
        let url = modelDir.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: url.path) {
            return try MLX.loadArrays(url: url)
        }
        var allWeights = [String: MLXArray]()
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors"
                && !$0.lastPathComponent.contains("voices") }
        for file in files {
            for (k, v) in try MLX.loadArrays(url: file) {
                allWeights[k] = v
            }
        }
        return allWeights
    }

    // MARK: - Tree-Walking Quantization

    private static func quantizeTree(
        module: Module,
        weights: [String: MLXArray],
        quantization: (groupSize: Int, bits: Int, mode: QuantizationMode),
        path: String = ""
    ) {
        var replacements = [(String, Module)]()

        for (key, child) in module.items() {
            let childPath = path.isEmpty ? key : "\(path).\(key)"

            switch child {
            case .value(.module(let childModule)):
                if childModule is Quantizable,
                   weights["\(childPath).scales"] != nil,
                   let quantized = quantizeSingle(
                       layer: childModule,
                       groupSize: quantization.groupSize,
                       bits: quantization.bits,
                       mode: quantization.mode
                   )
                {
                    replacements.append((key, quantized))
                } else {
                    quantizeTree(
                        module: childModule, weights: weights,
                        quantization: quantization, path: childPath
                    )
                }

            case .array(let items):
                for (index, item) in items.enumerated() {
                    let indexPath = "\(childPath).\(index)"
                    switch item {
                    case .value(.module(let childModule)):
                        quantizeTree(
                            module: childModule, weights: weights,
                            quantization: quantization, path: indexPath
                        )
                    case .array(let nestedItems):
                        for (nestedIndex, nestedItem) in nestedItems.enumerated() {
                            if case .value(.module(let nestedModule)) = nestedItem {
                                quantizeTree(
                                    module: nestedModule, weights: weights,
                                    quantization: quantization,
                                    path: "\(indexPath).\(nestedIndex)"
                                )
                            }
                        }
                    default:
                        break
                    }
                }

            default:
                break
            }
        }

        if !replacements.isEmpty {
            module.update(modules: ModuleChildren.unflattened(replacements))
        }
    }
}
