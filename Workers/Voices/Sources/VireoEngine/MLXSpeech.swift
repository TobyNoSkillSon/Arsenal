import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXAudioTTS
import SmallMGEMM
import VireoCore

/// Adapter from the vendored mlx-audio-swift speech models to Vireo's `SpeechModel`.
public final class MLXSpeech: SpeechModel {
    public let id: String
    var model: SpeechGenerationModel?
    var matrixDiagnosis: VoiceMatrixDiagnosis?
    public let sampleRate: Int
    /// Qwen3-TTS cloning: a saved voice's reference conditioning (speaker embedding + reference codec tokens),
    /// keyed by the clip's sample bytes, transcript and language. Lives and dies with this loaded model, so a
    /// model change or unload invalidates it; an edited voice has other bytes or text and misses.
    /// `VIREO_QWEN_REF_CACHE=0` recomputes per request (A/B).
    var intakeAssets: [String: URL] = [:]
    private var conditionings: [(key: String, value: Qwen3TTSModel.Qwen3TTSReferenceConditioning)] = []
    private static let conditioningCacheSize = 4
    /// The model-path configuration this model was loaded (and gated) with.
    let configuration: FastPath

    /// Gated fast path (FastPath.swift switches, KernelGate verdict): what serves, and why.
    public private(set) var engineLabel = "mlx"
    public private(set) var engineReason = "stock MLX; no optimized path for this model"
    /// Gate identity of the active optimized path, for a runtime fallback (engine key, version, weights directory);
    /// nil while stock serves.
    private(set) var gate: (engine: String, version: String, weights: URL)?

    init(id: String, model: SpeechGenerationModel, configuration: FastPath = .current) {
        self.id = id; self.model = model; sampleRate = model.sampleRate; self.configuration = configuration
    }

    /// An optimized path passed its self-test (verdict stored under `gate` and `configuration`): it serves from now on.
    func activate(reason: String, gate: (engine: String, version: String, weights: URL)) {
        engineLabel = "optimized"
        engineReason = reason
        self.gate = gate
    }

    public static let load: ModelLoader = { source in
        let directory = source.directory
        // One configuration selects the path, sets the vendored models' lab switches and keys the verdicts.
        let configuration = FastPath(source: source)
        configuration.applyToVendorModels()
        let quantized = Engines.quantizedDirectory(source)
        // Pocket's converted tiers hold only the FlowLM weights (QuantRules.pocket) beside the 16-bit snapshot. Qwen3, Kokoro
        // and Breeze tiers are whole checkpoints (CheckpointRules; the vendor MXFP4 snapshot for Breeze 4): `directory` is the tier.
        if quantized != nil && source.id != "pocket-tts" { throw EngineError("\(source.id) has no converted Vireo quantization") }
        let model: SpeechGenerationModel
        switch source.id {
        case "pocket-tts":
            if let quantized {
                model = try await loadPocketQuantized(base: directory, weights: quantized.appendingPathComponent(pocketWeights))
            } else {
                model = try await PocketTTSModel.fromModelDirectory(directory)
            }
        case "kokoro-82m":
            // English G2P always; the IPA lexicons only when the catalog offers a non-English voice.
            // UK voices (bf_/bm_) read misaki's British dictionaries; without them en-GB fails instead of sounding American.
            guard let english = source.assets["kokoro-english-g2p"], let british = source.assets["kokoro-british-lexicon"] else {
                throw EngineError("Missing pinned Kokoro phonemizer assets")
            }
            let configObject = (try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json")))) as? [String: Any]
            let metadata = (configObject?["arsenal_voice_intake"] ?? configObject?["mobius_voice_intake"]) as? [String: Any] ?? [:] // legacy-compat: read in snapshots written before the rename
            let recipe = metadata["recipe"] as? String ?? ""
            let keeping = recipe.hasPrefix("intake-v3:") ? try WholeVoiceConversion.keeping(recipe: recipe, model: source.id, dtype: source.precision) : []
            let precision = BARTPrecision(half: metadata["half"] as? String ?? "fp16", bits: metadata["bits"] as? Int ?? 16,
                keeping: keeping,
                truncation: metadata["source_truncation"] as? Bool ?? false)
            let processor = KokoroMultilingualProcessor(englishDirectory: english, britishDirectory: british, lexiconDirectory: source.assets["kokoro-lexicons"], precision: precision)
            model = try await KokoroModel.fromModelDirectory(directory, textProcessor: processor)
        case "breeze-tts-2": model = try await BreezeTTSModel.fromModelDirectory(directory)
        case let id where id.hasPrefix("qwen3-tts"): model = try await Qwen3TTSModel.fromModelDirectory(directory)
        default: throw EngineError("No native loader qualified for \(source.id)")
        }
        let speech = MLXSpeech(id: source.id, model: model, configuration: configuration)
        if let breeze = model as? BreezeTTSModel { speech.gateBreeze(breeze, weights: directory) }
        if let qwen = model as? Qwen3TTSModel { speech.gateQwen(qwen, weights: directory) }
        speech.intakeAssets = source.assets
        if let kokoro = model as? KokoroModel { speech.gateKokoro(kokoro, weights: directory) }
        return speech
    }

    // MARK: Fast paths (self-test at load, runtime fallback, label)

    /// Breeze TTS 2: VIREO_BREEZE_LAZY (one sync per frame, pipelined), VIREO_BREEZE_DEPTH_CACHE (depth decoder
    /// KV cache, guidance branches as one batch), VIREO_BREEZE_STREAM (streaming codec decode); each "0" turns it
    /// off. VIREO_BREEZE_FUSED (fused depth-step kernels, with the cache; child kernel check first). VIREO_BREEZE_CFG_BATCH=1
    /// (lab): batched guidance without the cache. Every choice is self-tested.
    static func breezeOptions(_ configuration: FastPath) -> BreezeTTSModel.FastOptions {
        BreezeTTSModel.FastOptions(lazyFrames: configuration.enabled("VIREO_BREEZE_LAZY"),
                                   cfgBatch: configuration.optIn("VIREO_BREEZE_CFG_BATCH"),
                                   stream: configuration.enabled("VIREO_BREEZE_STREAM"),
                                   depthCache: configuration.enabled("VIREO_BREEZE_DEPTH_CACHE"),
                                   fusedDepth: configuration.enabled("VIREO_BREEZE_FUSED"))
    }

    func gateBreeze(_ breeze: BreezeTTSModel, weights: URL) {
        breeze.profile = FastPath.profiling("VIREO_BREEZE_PROFILE")
        var options = Self.breezeOptions(configuration)
        if !options.lazyFrames { options.cfgBatch = false; options.depthCache = false }   // both live in the lazy loop
        if !options.depthCache { options.fusedDepth = false }   // the fused step extends the cached step
        // SmallMGEMM FFN: opt-in (measured on BF16 only, not yet gated on the suite).
        options.smallM = options.fusedDepth && breeze.depthFFNIsDense && configuration.optIn("VIREO_BREEZE_SMALLM")
        guard !options.isStock else {
            engineReason = configuration.forceStock ? "stock MLX (VIREO_FORCE_STOCK)" : "stock MLX (fast path switched off)"
            return
        }
        if options.smallM {
            // The family kernel's own unit self-test (GEMV classes this path uses), in the child first.
            let kernel = KernelGate.verdict(engine: Self.smallMKernelKey, kernelVersion: SmallMGEMM.revision, weights: weights,
                                            configuration: configuration)
            if !kernel.passed { options.smallM = false; breezeKernelNote += " (SmallMGEMM off: \(kernel.detail))" }
        }
        if options.fusedDepth {
            // Custom Metal kernels: checked in the child first (a compile error or fault aborts the process).
            let kernel = KernelGate.verdict(engine: Self.breezeFusedKernelKey, kernelVersion: BreezeTTSModel.fusedDepthKernelVersion,
                                            weights: weights, configuration: configuration)
            if !kernel.passed {
                options.fusedDepth = false; options.smallM = false
                breezeKernelNote += " (fused depth kernels off: \(kernel.detail))"
            }
        }
        // Inexact components fail alone: without the newest ones first, then without the fused step, each its own verdict.
        var candidates = [options]
        if options.smallM { var o = options; o.smallM = false; candidates.append(o) }
        if options.fusedDepth { var o = options; o.fusedDepth = false; o.smallM = false; candidates.append(o) }
        var failures = [String]()
        for candidate in candidates {
            var version = BreezeTTSModel.fastPathVersion + " " + candidate.label
            if candidate.fusedDepth { version += " (" + BreezeTTSModel.fusedDepthKernelVersion + ")" }
            if candidate.smallM { version += " (SmallMGEMM " + SmallMGEMM.revision + ")" }
            // The self-test samples with its own fixed seed (its sampler's key), never the global key.
            // Exact (only lazy frames): the audio must equal stock's, not just come within 40 dB (999 = no difference).
            let minimumSNR: Double = configuration.exact ? 999 : 40
            let verdict = KernelGate.inProcessVerdict(engine: id, kernelVersion: version, weights: weights, configuration: configuration) {
                try breeze.fastPathSelfTest(candidate, minimumSNR: minimumSNR)
            }
            if verdict.passed {
                breeze.fastOptions = candidate
                let note = failures.isEmpty ? "" : " (without: \(failures.joined(separator: "; ")))"
                activate(reason: "\(candidate.label) (self-test passed: \(verdict.detail))\(breezeKernelNote)\(note)", gate: (id, version, weights))
                return
            }
            failures.append("\(candidate.label): \(verdict.detail)")
        }
        engineReason = "stock MLX (fast-path self-test failed: \(failures.last ?? ""))"
    }
    private var breezeKernelNote = ""

    static let smallMKernelKey = "smallm-gemv"
    /// Child-process check of the vendored SmallMGEMM: its own `selfTest()`, every GEMV class within its bound.
    static func smallMChildSelfTest(_ weights: URL) throws -> String {
        let results = SmallMGEMM.selfTest().filter { $0.key.hasPrefix("gemv.") }
        let failures = SmallMGEMM.selfTestFailures(results)
        let worst = results.values.max() ?? .infinity
        guard !results.isEmpty, failures.isEmpty else { throw EngineError("SmallMGEMM GEMV self-test failed: \(failures.joined(separator: ", "))") }
        return String(format: "SmallMGEMM %@ GEMV self-test passed (%d classes, worst rel RMS %.1e)", SmallMGEMM.revision, results.count, worst)
    }

    static let breezeFusedKernelKey = "breeze-depth-fused"
    /// Child-process check of the fused depth attention kernel against stock ops (bounds: output relative RMS 1e-2,
    /// rotated keys/values within one bf16 step at |x| ≤ 8 (0.0625); measured 1.3e-4 and 0).
    static func breezeFusedChildSelfTest(_ weights: URL) throws -> String {
        let c = BreezeTTSModel.fusedDepthUnitCheck()
        let detail = String(format: "attention rel RMS %.1e, cache max|Δ| %.3g", c.attention, c.cache)
        guard c.attention <= 1e-2, c.cache <= 0.0625 else { throw EngineError("fused depth kernels: \(detail)") }
        return "fused depth kernels within bounds (\(detail))"
    }

    /// Qwen3-TTS (VoiceDesign and Base): VIREO_QWEN_FAST_ROPE (fused RoPE kernel), VIREO_QWEN_FAST_TABLES
    /// (code-predictor rotary/mask tables and frame-sized KV cache) and VIREO_QWEN_FAST_SAMPLER (fused code-predictor
    /// sampler kernel), default on, "0" off; VIREO_QWEN_FAST_PIPELINE=1 (lab) pipelines the decode loop (no gain
    /// measured while the step is GPU-bound). Each kernel passes its child-process check before the in-process
    /// whole-generation self-test.
    func gateQwen(_ qwen: Qwen3TTSModel, weights: URL) {
        let options = Qwen3TTSModel.FastOptions(pipeline: configuration.optIn("VIREO_QWEN_FAST_PIPELINE"),
                                                fusedRope: configuration.enabled("VIREO_QWEN_FAST_ROPE"),
                                                predictorTables: configuration.enabled("VIREO_QWEN_FAST_TABLES"),
                                                fusedSampler: configuration.enabled("VIREO_QWEN_FAST_SAMPLER"))
        guard !options.isStock else {
            engineReason = configuration.forceStock ? "stock MLX (VIREO_FORCE_STOCK)" : "stock MLX (fast path switched off)"
            return
        }
        // A custom Metal kernel can abort the process (compile error, fault): check it in the child first.
        if options.fusedRope {
            let kernel = KernelGate.verdict(engine: Self.qwenRopeKernelKey, kernelVersion: Self.qwenRopeKernelVersion, weights: weights,
                                            configuration: configuration)
            guard kernel.passed else {
                engineReason = "stock MLX (fused RoPE kernel self-test failed: \(kernel.detail))"
                return
            }
        }
        if options.fusedSampler {
            let kernel = KernelGate.verdict(engine: Self.qwenSamplerKernelKey, kernelVersion: Self.qwenSamplerKernelVersion, weights: weights,
                                            configuration: configuration)
            guard kernel.passed else {
                engineReason = "stock MLX (fused predictor sampler self-test failed: \(kernel.detail))"
                return
            }
        }
        let version = "qwen3-tts-fast-3 " + options.label
        let verdict = KernelGate.inProcessVerdict(engine: id, kernelVersion: version, weights: weights, configuration: configuration) {
            try qwen.fastPathSelfTest(options)
        }
        if verdict.passed {
            qwen.setFastOptions(options)
            activate(reason: "\(options.label) (self-test passed: \(verdict.detail))", gate: (id, version, weights))
        } else {
            engineReason = "stock MLX (fast-path self-test failed: \(verdict.detail))"
        }
    }

    /// Kokoro 82M: VIREO_KOKORO_ALIGN (one-hot alignment built on the host), VIREO_KOKORO_CONV_CACHE (weight-normalized
    /// conv weights computed once) and VIREO_KOKORO_LSTM (both BiLSTM directions per step as one batched addmm and one
    /// compiled cell update) and VIREO_KOKORO_HALF (decoder conv stack in fp16; needs the conv cache; inexact, gated by
    /// a tolerance against the same path in fp32), default on, "0" off; stock MLX ops only, so the self-test runs
    /// in-process. Then a warm-up pass (VIREO_KOKORO_WARMUP=0 skips it): the first pass in a process builds Metal
    /// pipelines and the phonemizer. VIREO_KOKORO_INJECT_NAN=1 (lab) poisons the optimized path's output to exercise the
    /// runtime fallback. VIREO_KOKORO_PROFILE=2 (lab) also splits predictor and decoder time (one extra sync).
    func gateKokoro(_ kokoro: KokoroModel, weights: URL) {
        let profileLevel = FastPath.profilingLevel("VIREO_KOKORO_PROFILE")
        kokoro.profile = profileLevel >= 1
        kokoro.profileSplit = profileLevel >= 2
        kokoro.injectNonFinite = configuration.optIn("VIREO_KOKORO_INJECT_NAN")
        defer {
            if configuration.value("VIREO_KOKORO_WARMUP") != "0" {
                let start = Date()
                let detail = (try? kokoro.warmUp()) ?? "failed"
                if kokoro.profile {
                    FileHandle.standardError.write(Data("[kokoro profile] warm-up \(String(format: "%.2f", Date().timeIntervalSince(start))) s (\(detail))\n".utf8))
                }
            }
        }
        let options = KokoroModel.FastOptions(hostAlignment: configuration.enabled("VIREO_KOKORO_ALIGN"),
                                              convWeightCache: configuration.enabled("VIREO_KOKORO_CONV_CACHE"),
                                              fusedLSTM: configuration.enabled("VIREO_KOKORO_LSTM"),
                                              halfDecoder: kokoro.storedHalf == nil && configuration.enabled("VIREO_KOKORO_HALF"))
        guard !options.isStock else {
            engineReason = configuration.forceStock ? "stock MLX (VIREO_FORCE_STOCK)" : "stock MLX (fast path switched off)"
            return
        }
        func gate(_ options: KokoroModel.FastOptions) -> (version: String, verdict: KernelGate.Verdict) {
            let version = KokoroModel.fastPathVersion + " " + options.label
            // Exact: the exact parts must match stock sample for sample (their usual bound is 60 dB).
            let bound = configuration.exact ? Double.infinity : 60
            return (version, KernelGate.inProcessVerdict(engine: id, kernelVersion: version, weights: weights, configuration: configuration) {
                try kokoro.fastPathSelfTest(options, minSNR: bound)
            })
        }
        var chosen = options
        var (version, verdict) = gate(chosen)
        var note = ""
        // The fp16 decoder has its own fallback: a failed (or runtime-failed, sticky) fp16 verdict keeps the exact parts.
        if !verdict.passed, chosen.halfDecoder {
            note = "; fp16 decoder off: \(verdict.detail)"
            chosen = chosen.exactPart
            (version, verdict) = gate(chosen)
        }
        if verdict.passed {
            kokoro.setFastOptions(chosen)
            activate(reason: "\(chosen.label) (self-test passed: \(verdict.detail)\(note))", gate: (id, version, weights))
        } else {
            engineReason = "stock MLX (fast-path self-test failed: \(verdict.detail)\(note))"
        }
    }

    static let qwenRopeKernelKey = "qwen3-tts-rope"
    static let qwenRopeKernelVersion = "vireo_qwen3tts_rope_rotate_half 2"

    static let qwenSamplerKernelKey = "qwen3-tts-sampler"
    static let qwenSamplerKernelVersion = "vireo_qwen3tts_topk_categorical 2"

    /// Child-process check of the fused code-predictor sampler (tokens and key state vs stock on tie-heavy logits).
    static func qwenSamplerChildSelfTest(_ weights: URL) throws -> String {
        let check = Qwen3TTSModel.fusedSamplerUnitCheck()
        guard check.mismatches == 0 else { throw EngineError("fused sampler differs from stock: \(check.detail)") }
        return "fused predictor sampler identical to stock (\(check.detail))"
    }

    /// Child-process check of the fused RoPE kernel (no model needed; `weights` only keys the verdict).
    static func qwenRopeChildSelfTest(_ weights: URL) throws -> String {
        let differing = Qwen3TTSModel.fusedRopeUnitCheck()
        guard differing == 0 else { throw EngineError("fused RoPE: \(differing) elements differ from stock") }
        return "fused RoPE bit-identical to stock (bf16, fp32)"
    }

    /// An optimized path failed its runtime check: stock from now on (sticky for this gate key).
    private func fallBack(_ failure: FastPathFailure) {
        (model as? BreezeTTSModel)?.fastOptions = .stock
        (model as? Qwen3TTSModel)?.setFastOptions(.stock)
        (model as? KokoroModel)?.setFastOptions(.stock)
        engineLabel = "mlx"
        engineReason = "stock MLX (fallback: \(failure.detail))"
        if let gate {
            KernelGate.recordRuntimeFailure(engine: gate.engine, kernelVersion: gate.version, weights: gate.weights, configuration: configuration,
                                            detail: failure.detail)
        }
        gate = nil
    }

    // MARK: Pocket TTS converted precisions (rule: QuantRules.pocket)

    static let pocketWeights = "model.safetensors"

    /// The model tree from the base snapshot's config (tokenizer and voice embeddings stay in the base snapshot);
    /// parameters are lazy placeholders until `update(parameters:)`.
    static func pocketSkeleton(_ base: URL) async throws -> PocketTTSModel {
        let config = try PocketTTSModelConfig.load(from: base.appendingPathComponent("config.json"))
        return try await PocketTTSModel.fromConfig(config, modelFolder: base)
    }

    /// Same steps as `PocketTTSModel.fromModelDirectory`, with the converted weights file and the quantized
    /// layers installed from it first. Costs what the base load costs.
    static func loadPocketQuantized(base: URL, weights: URL) async throws -> PocketTTSModel {
        let model = try await pocketSkeleton(base)
        var arrays = try MLX.loadArrays(url: weights)
        if let speaker = arrays.removeValue(forKey: "speaker_proj_weight") { arrays["speaker_proj.weight"] = speaker }
        guard try QuantizedWeights.prepare(model, weights: arrays) > 0 else { throw EngineError("\(weights.path) holds no quantized layers") }
        try model.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(model)
        return model
    }

    static func convertPocket(_ source: ModelSource, bits: Int, into directory: URL, report: inout QuantReport) async throws {
        let skeleton = try await pocketSkeleton(source.directory)
        let fresh = try await pocketSkeleton(source.directory)
        let rule = report.rule
        try QuantizedWeights.convertFile(source: source.directory.appendingPathComponent(pocketWeights),
                                         output: directory.appendingPathComponent(pocketWeights),
                                         skeleton: skeleton, rule: rule, bits: bits, report: &report) { saved, expected in
            var mapped = saved
            if let speaker = mapped.removeValue(forKey: "speaker_proj_weight") { mapped["speaker_proj.weight"] = speaker }
            try QuantizedWeights.strictCheck(fresh, mapped, expected: expected)
        }
    }

    /// Models that take a prose voice description and delivery instructions.
    public static func takesDescription(_ id: String) -> Bool { id.hasPrefix("qwen3-tts-voicedesign") || id == "breeze-tts-2" }

    public func setSpeed(_ speed: Float) throws {
        guard speed.isFinite, speed >= 0.25, speed <= 4 else { throw EngineError("speed must be between 0.25 and 4") }
        if let kokoro = model as? KokoroModel { kokoro.speed = speed }
        else if speed != 1 { throw EngineError("Speed control is only available for Kokoro") }
    }

    public func speak(_ request: SpeechRequest, control: GenerationControl, emit: @escaping ([Float]) async throws -> Void) async throws {
        guard let model else { throw EngineError("\(id) is unloaded") }
        // Each attempt's stream stops at its next frame or sentence when the request is stopped or the attempt is abandoned.
        let stoppable = model as? StopChecking
        func attempt() -> GenerationControl {
            let abandon = GenerationControl()
            stoppable?.cancellationRequested = { control.isCancelled || abandon.isCancelled }
            return abandon
        }
        defer { stoppable?.cancellationRequested = nil }
        var parameters = model.defaultGenerationParameters
        if let value = request.maxTokens { parameters.maxTokens = value }
        if let value = request.temperature { parameters.temperature = value }
        if let value = request.topP { parameters.topP = value }
        if let value = request.topK { parameters.topK = value }
        if let value = request.repetitionPenalty { parameters.repetitionPenalty = value }
        if let breeze = model as? BreezeTTSModel { breeze.guidanceScale = request.guidance ?? 4 }
        var prompt = request.preset
        if Self.takesDescription(id) {
            let joined = [request.description, request.style].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: ". Delivery: ")
            prompt = joined.isEmpty ? nil : joined
        }
        // The request's seed wins; VIREO_SPEECH_SEED (lab parity runs) applies only when the request has none.
        // Both paths below (cached reference conditioning and ordinary) sample after this.
        if let seed = request.seed ?? configuration.speechSeed { MLXRandom.seed(seed) }
        let reference = request.reference.map { MLXArray(Self.mono($0, rate: model.sampleRate)) }
        // Kokoro takes its language from the voice (af_… US English, ff_… French); a request language does not apply.
        let language = model is KokoroModel ? nil : (request.language ?? "english")
        if let qwen = model as? Qwen3TTSModel, let clip = request.reference, let refText = request.referenceText, let reference {
            let conditioning = try conditioning(qwen, clip: clip, audio: reference, text: refText, language: language ?? "auto")
            func clone(_ emit: @escaping ([Float]) async throws -> Void) async throws {
                let abandon = attempt()
                let stream = qwen.generateStream(text: request.text, conditioning: conditioning,
                                                 generationParameters: parameters, streamingInterval: 0.25)
                try await drain(stream.compactMap { event -> [Float]? in
                    guard case .audio(let samples) = event else { return nil }
                    return samples.asArray(Float.self)
                }, control: control, abandon: abandon, emit: emit)
            }
            var emitted = false
            do {
                try await clone { emitted = true; try await emit($0) }
            } catch let failure as FastPathFailure {
                fallBack(failure)
                guard !emitted else { throw EngineError("Generation stopped: \(failure.detail); switched to the stock path, try again") }
                try await clone(emit)
            }
            return
        }
        var emitted = false
        do {
            let abandon = attempt()
            let stream = model.generateSamplesStream(text: request.text, voice: prompt, refAudio: reference, refText: request.referenceText,
                                                     language: language,
                                                     generationParameters: parameters, streamingInterval: 0.25)
            try await drain(stream, control: control, abandon: abandon) { emitted = true; try await emit($0) }
        } catch let failure as FastPathFailure {
            fallBack(failure)
            // Nothing played yet: serve the request on stock (a fresh sample). Otherwise the caller gets an
            // error (the long-render pipeline redoes the chunk).
            guard !emitted else { throw EngineError("Generation stopped: \(failure.detail); switched to the stock path, try again") }
            let abandon = attempt()
            let stream = model.generateSamplesStream(text: request.text, voice: prompt, refAudio: reference, refText: request.referenceText,
                                                     language: language,
                                                     generationParameters: parameters, streamingInterval: 0.25)
            try await drain(stream, control: control, abandon: abandon, emit: emit)
        }
    }

    /// Never abandon the model stream while it still owns GPU work: after a stop, drain without emitting.
    /// Non-finite samples are never emitted: from an active optimized path they are a `FastPathFailure` (the caller
    /// falls back to stock, sticky, and retries if nothing was emitted); from stock they end the request.
    /// The failed stream is stopped (`abandon`, checked at its next frame or sentence) and drained to its end before the
    /// failure is thrown, so the switch to stock options and the stock retry never run beside it.
    private func drain<S: AsyncSequence>(_ stream: S, control: GenerationControl, abandon: GenerationControl,
                                         emit: @escaping ([Float]) async throws -> Void) async throws where S.Element == [Float] {
        var failure: FastPathFailure?
        do {
            for try await samples in stream {
                if control.isCancelled || failure != nil { continue }
                if let bad = samples.firstIndex(where: { !$0.isFinite }) {
                    guard gate != nil else { throw EngineError("Non-finite model output") }
                    failure = FastPathFailure("non-finite output sample \(bad) on the optimized path")
                    abandon.cancel()
                    continue
                }
                try await emit(samples)
            }
        } catch {
            // After a stop or an abandoned fast path, the stream's own ending error does not matter.
            if failure == nil && !(error is CancellationError && control.isCancelled) { throw error }
        }
        if let failure { throw failure }
    }

    private func conditioning(_ qwen: Qwen3TTSModel, clip: AudioClip, audio: MLXArray, text: String, language: String) throws
        -> Qwen3TTSModel.Qwen3TTSReferenceConditioning {
        let digest = clip.samples.withUnsafeBytes { SHA256.hash(data: $0) }.map { String(format: "%02x", $0) }.joined()
        let key = "\(digest)|\(clip.sampleRate)|\(clip.channels)|\(language)|\(text)"
        if configuration.qwenReferenceCache, let index = conditionings.firstIndex(where: { $0.key == key }) {
            let hit = conditionings.remove(at: index); conditionings.append(hit); return hit.value
        }
        let value = try qwen.prepareReferenceConditioning(refAudio: audio, refText: text, language: language)
        eval([value.referenceSpeechCodes, value.referenceTextTokenIDs] + (value.speakerEmbedding.map { [$0] } ?? []))
        if configuration.qwenReferenceCache {
            conditionings.append((key, value))
            if conditionings.count > Self.conditioningCacheSize { conditionings.removeFirst() }
        }
        return value
    }

    public func unload() async { conditionings = []; model = nil }

    /// Reference clips are stored mono at the model rate by the helper; this only guards other shapes.
    static func mono(_ audio: AudioClip, rate: Int) -> [Float] {
        guard audio.channels > 1 else { return audio.samples }
        return stride(from: 0, to: audio.samples.count, by: audio.channels).map { i in
            (0..<audio.channels).reduce(Float(0)) { $0 + audio.samples[i + $1] } / Float(audio.channels)
        }
    }
}

/// Speech models that check a stop flag between frames or sentences: a stopped request, or an abandoned fast-path stream.
public protocol StopChecking: AnyObject { var cancellationRequested: (() -> Bool)? { get set } }
extension PocketTTSModel: StopChecking {}
extension Qwen3TTSModel: StopChecking {}
extension KokoroModel: StopChecking {}
