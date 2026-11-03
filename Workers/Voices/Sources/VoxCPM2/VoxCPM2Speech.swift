// SPDX-License-Identifier: MIT
// VoxCPM2 behind the voices family's engine boundary (VireoCore `SpeechModel`). Zero-shot speech, a voice described in
// words (the maker's "(description)" prefix), cloning from a reference clip, and cloning from a clip with its
// transcript (continuation; with the clip also as isolated reference by default, the maker's "ultimate cloning").
// 48 kHz out; reference audio is read at the AudioVAE's 16 kHz.
import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import VireoCore

public struct VoxCPM2Options: Sendable, Equatable {
    public var cfgValue: Float = 2
    public var inferenceTimesteps = 10
    public var cloneMode: VoxCPM2CloneMode?
    public var retryBadcase = true
    public init() {}
}

public final class VoxCPM2Speech: SpeechModel {
    public static let modelID = "voxcpm2"
    /// Sample rate reference clips are read at (the AudioVAE encoder's).
    public static let referenceRate = 16000
    /// Text longer than this is spoken in sentence groups of about this many characters, each one generation; without
    /// a clip, groups after the first continue the first group's audio, so they keep its voice.
    static let chunkCharacters = 400
    public let id: String
    public let sampleRate: Int
    var model: VoxCPM2Model?
    let tokenizer: any Tokenizer
    /// Called with each chunk's plan as it runs (tests check that later chunks continue chunk 0).
    var onPlan: ((Int, ChunkPlan) -> Void)?

    init(id: String, model: VoxCPM2Model, tokenizer: any Tokenizer) {
        self.id = id; self.model = model; self.tokenizer = tokenizer; sampleRate = model.sampleRate
    }

    public var engineLabel: String { "mlx" }
    public var engineReason: String { "Native Swift port of VoxCPM2 (mlx-audio-swift e8450402, checked against OpenBMB/VoxCPM); stock MLX kernels" }

    /// Loads a snapshot (the FP16 Get cast, or the BF16/FP32 release cast to `dtype` in memory).
    public static func load(directory: URL, id: String = modelID, dtype: DType = .float16) async throws -> VoxCPM2Speech {
        let config = try JSONDecoder().decode(VoxCPM2Configuration.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard config.modelType == "voxcpm2" else { throw VoxCPM2Error("config.json is \(config.modelType), not voxcpm2") }
        try config.check()
        let model = VoxCPM2Model(config)
        try model.apply(weights: try VoxCPM2Model.loadWeights(directory, dtype: dtype))
        eval(model.parameters())
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        try checkTokenizer(tokenizer)
        return VoxCPM2Speech(id: id, model: model, tokenizer: tokenizer)
    }

    /// The prompt's special tokens must be the ones the network was trained with (ids 101–104).
    static func checkTokenizer(_ tokenizer: any Tokenizer) throws {
        for (token, expected) in [("<|audio_start|>", 101), ("<|audio_end|>", 102), ("<|ref_audio_start|>", 103), ("<|ref_audio_end|>", 104)] {
            if let found = tokenizer.convertTokenToId(token), found != expected { throw VoxCPM2Error("tokenizer maps \(token) to \(found), expected \(expected)") }
        }
    }

    public func unload() async {
        model = nil
        Memory.clearCache()
    }

    public func speak(_ request: SpeechRequest, control: GenerationControl, emit: @escaping ([Float]) async throws -> Void) async throws {
        var options = VoxCPM2Options()
        if let guidance = request.guidance { options.cfgValue = guidance }
        try await speak(request, options: options, control: control, emit: emit)
    }

    /// Longest text one request takes. Estimate, not a measurement: the 2 Nov real check generated about 35 characters of
    /// text per second of wall time on an M5 Max in use, so 590 s (the service's limit) is about 20,000 characters;
    /// 5,000 leaves a 4× margin for retries and slow chunks.
    public static let maxTextCharacters = 5000

    /// A voice or style description and the words to say. `instructions` become the maker's "(…)" prefix; text that
    /// already opens with a complete "(…)" description keeps it as the prefix, so it applies to every chunk.
    public static func parts(_ text: String, instructions: String?) throws -> (prefix: String, body: String) {
        var body = VoxCPM2Model.cleanText(text).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty else { throw VoxCPM2Error("text must contain something to say") }
        guard body.count <= maxTextCharacters else {
            throw VoxCPM2Error("text exceeds \(maxTextCharacters) characters, about what VoxCPM2 can speak within one request's time limit; split it")
        }
        var prefix = ""
        if let instructions = instructions?.trimmingCharacters(in: .whitespacesAndNewlines), !instructions.isEmpty {
            guard !instructions.contains("(") && !instructions.contains(")") else { throw VoxCPM2Error("instructions must not contain parentheses") }
            prefix = "(" + VoxCPM2Model.cleanText(instructions) + ")"
        } else if body.hasPrefix("("), let close = body.firstIndex(of: ")") {
            prefix = String(body[...close])
            body = String(body[body.index(after: close)...]).trimmingCharacters(in: .whitespaces)
            guard !body.isEmpty else { throw VoxCPM2Error("text must contain something to say after the description") }
        }
        return (prefix, body)
    }

    /// The text the network reads for a short request: description prefix and words.
    public static func text(_ text: String, instructions: String?) throws -> String {
        let (prefix, body) = try parts(text, instructions: instructions)
        return prefix + body
    }

    /// Resolves how a request uses its reference clip. A transcript without a clip, or continuation without a
    /// transcript, is refused.
    public static func cloneMode(hasClip: Bool, transcript: String?, requested: VoxCPM2CloneMode?) throws -> VoxCPM2CloneMode? {
        let hasText = transcript.map { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } ?? false
        guard hasClip else {
            if hasText { throw VoxCPM2Error("reference.text needs its reference.path clip") }
            if requested != nil { throw VoxCPM2Error("clone_mode needs a reference clip") }
            return nil
        }
        let mode = requested ?? (hasText ? .both : .reference)
        if mode != .reference && !hasText { throw VoxCPM2Error("clone_mode \(mode.rawValue) needs the exact transcript of the reference clip (reference.text)") }
        return mode
    }

    /// Sentence ends in the supported scripts: Latin, CJK, Devanagari, Burmese, Arabic, Urdu, Khmer, Thai/Lao spacing aside.
    static let sentenceEnds: Set<Character> = [".", "!", "?", "。", "！", "？", "…", "।", "॥", "။", "؟", "۔", "។", "︒", "؛"]

    /// Splits long text at sentence ends into groups of at most `limit` characters; a longer sentence is cut at spaces,
    /// and a longer run without spaces (Chinese, Thai) at `limit`. Short text stays whole.
    static func chunks(_ text: String, limit: Int = chunkCharacters) -> [String] {
        guard text.count > limit else { return [text] }
        var sentences: [String] = [], current = ""
        for ch in text {
            current.append(ch)
            if sentenceEnds.contains(ch) { sentences.append(current); current = "" }
        }
        if !current.trimmingCharacters(in: .whitespaces).isEmpty { sentences.append(current) }
        var pieces: [String] = []
        for sentence in sentences {
            guard sentence.count > limit else { pieces.append(sentence); continue }
            var part = ""
            // Every word after the first keeps the space before it, so no piece loses its separator: the pieces
            // concatenate back to the sentence, and a group joining two of them reads "Hi. Lorem", never "Hi.Lorem".
            for (index, word) in sentence.split(separator: " ", omittingEmptySubsequences: false).enumerated() {
                var word = index == 0 ? String(word) : " " + word
                while word.count > limit {
                    if !part.isEmpty { pieces.append(part); part = "" }
                    pieces.append(String(word.prefix(limit))); word = String(word.dropFirst(limit))
                }
                if !part.isEmpty && part.count + word.count > limit { pieces.append(part); part = "" }
                part += word
            }
            if !part.isEmpty { pieces.append(part) }
        }
        var groups: [String] = [], group = ""
        for piece in pieces {
            if !group.isEmpty && group.count + piece.count > limit { groups.append(group); group = "" }
            group += piece
        }
        if !group.isEmpty { groups.append(group) }
        return groups.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    /// Seed for chunk `index`: the request seed for the first (so a one-chunk request matches the maker's seeding),
    /// and well-separated values after it, so a chunk's retries (seed + 1, + 2) never reuse another chunk's seed.
    static func chunkSeed(_ seed: UInt64, index: Int) -> UInt64 {
        guard index > 0 else { return seed }
        var z = seed &+ UInt64(index) &* 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return (z ^ (z >> 31)) & 0xFFFF_FFFF
    }

    /// One chunk as the LM reads it: `ids`, the text tokens after any reference; `target`, the tokens of the words this
    /// chunk speaks, which set its length cap `maxLength`. `continuesFirst`: the chunk continues chunk 0's own audio.
    struct ChunkPlan: Equatable {
        var ids: [Int]
        var target: Int
        var maxLength: Int
        var continuesFirst = false
    }

    /// The maker's length cap: 6 patches per target token plus 10, within the request's cap.
    static func lengthCap(_ target: Int, generation: VoxCPM2Generation) -> Int {
        min(Int(Float(target) * generation.retryRatio + 10), generation.maxLength)
    }

    /// Plans every chunk of `body`. `alone`: each chunk on its own, after a clip's transcript (`promptText`) in the
    /// continuation layouts. `continuing` (only with `anchored`, i.e. without a clip): chunks after the first continue
    /// chunk 0, reading chunk 0's text and then their own, with chunk 0's generated patches as the audio prompt, as a
    /// clip with its transcript would be; so one voice speaks the whole text instead of a new one every chunk.
    static func plans(
        body: String, pieces: [String], prefix: String, promptText: String?, anchored: Bool, generation: VoxCPM2Generation,
        tokenize: (String) -> [Int]
    ) -> (alone: [ChunkPlan], continuing: [ChunkPlan?]) {
        let alone = pieces.map { piece -> ChunkPlan in
            let text = prefix + piece
            let target = tokenize(text).count
            return ChunkPlan(ids: tokenize((promptText ?? "") + text), target: target, maxLength: lengthCap(target, generation: generation))
        }
        guard anchored, let first = pieces.first, pieces.count > 1 else { return (alone, alone.map { _ in nil }) }
        // The chunks were cut from `body` in order, so what followed chunk 0 there (a space, or nothing in scripts
        // without spaces) joins it to each later chunk.
        let joiner = body.dropFirst(first.count).first?.isWhitespace == true ? " " : ""
        let continuing = pieces.enumerated().map { index, piece -> ChunkPlan? in
            guard index > 0 else { return nil }
            let target = tokenize(prefix + piece).count
            return ChunkPlan(ids: tokenize(prefix + first + joiner + piece), target: target, maxLength: lengthCap(target, generation: generation), continuesFirst: true)
        }
        return (alone, continuing)
    }

    /// Context tokens a chunk can occupy: the reference (patches and two markers), its text, the audio-start token,
    /// the prompt patches (a clip's, or chunk 0's for a chunk continuing it) and its own cap.
    static func contextNeed(_ plan: ChunkPlan, referencePatches: Int?, promptPatches: Int) -> Int {
        (referencePatches.map { $0 + 2 } ?? 0) + plan.ids.count + 1 + promptPatches + plan.maxLength
    }

    /// The plan chunk `index` runs: it continues chunk 0 when chunk 0 left audio to continue (`firstPatches`) and the
    /// continuation fits the context with its whole cap; otherwise it speaks on its own (always checked up front).
    static func plan(_ index: Int, alone: [ChunkPlan], continuing: [ChunkPlan?], firstPatches: Int?, context: Int) -> ChunkPlan {
        guard let firstPatches, let plan = continuing[index], contextNeed(plan, referencePatches: nil, promptPatches: firstPatches) <= context else {
            return alone[index]
        }
        return plan
    }

    /// Chunk 0's generated patches, (1, P, D) each, as a continuation prompt (T, P, D).
    static func continuationPrompt(_ patches: [MLXArray]) -> MLXArray { MLX.concatenated(patches, axis: 0) }

    /// What a full-size chunk can add to a saved voice's prompt: three tokens per character (scripts the tokenizer
    /// spells in UTF-8 bytes, such as Devanagari and Thai), the audio-start token and the length cap of that many tokens.
    static var fullChunkTokens: Int { 3 * chunkCharacters + 1 + lengthCap(3 * chunkCharacters, generation: VoxCPM2Generation()) }

    /// Prompt tokens of a saved voice in its largest layout: the clip as reference (patches and two markers) and, with a
    /// transcript, the clip again as continuation prompt plus the transcript's tokens.
    static func voicePromptTokens(patches: Int, transcriptTokens: Int?) -> Int {
        patches + 2 + (transcriptTokens.map { $0 + patches } ?? 0)
    }

    /// A clip and transcript a saved voice may keep (`clone_voice`): its prompt must leave room in the context for a
    /// full-size chunk, or every later request with the voice would fail the context check.
    public func checkVoice(samples: Int, transcript: String?) throws {
        guard let model else { throw EngineError("VoxCPM2 is not loaded") }
        let patchSamples = model.patchSize * model.audioVAE.chunkSize
        let patches = (samples + patchSamples - 1) / patchSamples
        let words = transcript.map { VoxCPM2Model.tokenize($0, tokenizer: tokenizer).count }
        let need = Self.voicePromptTokens(patches: patches, transcriptTokens: words) + Self.fullChunkTokens
        guard need <= model.config.maxLength else {
            throw VoxCPM2Error(
                "The transcript (\(words ?? 0) tokens) leaves no room in VoxCPM2's \(model.config.maxLength)-token context for speech; give only the words spoken in the clip")
        }
    }

    /// Speaks `request.text`. `request.reference` must be mono at `referenceRate`. A request the model cannot serve
    /// throws `VoxCPM2Error` before any generation.
    public func speak(_ request: SpeechRequest, options: VoxCPM2Options, control: GenerationControl, emit: @escaping ([Float]) async throws -> Void) async throws {
        guard let model else { throw EngineError("VoxCPM2 is not loaded") }
        if let preset = request.preset, !preset.isEmpty { throw VoxCPM2Error("VoxCPM2 has no preset voices; pass a reference clip, a saved voice or instructions") }
        guard (1...100).contains(options.inferenceTimesteps) else { throw VoxCPM2Error("inference_timesteps must be 1 to 100") }
        guard options.cfgValue.isFinite, (0.1...10).contains(options.cfgValue) else { throw VoxCPM2Error("cfg_value must be 0.1 to 10") }
        let mode = try Self.cloneMode(hasClip: request.reference != nil, transcript: request.referenceText, requested: options.cloneMode)
        let (prefix, body) = try Self.parts(request.text, instructions: request.description ?? request.style)
        var refFeat: MLXArray?, promptFeat: MLXArray?, promptText: String?
        if let clip = request.reference, let mode {
            // The caller reads clips at this rate; anything else is a fault in the caller, not in the request.
            guard clip.sampleRate == Self.referenceRate, clip.channels == 1 else { throw EngineError("VoxCPM2 reference audio must be mono at \(Self.referenceRate) Hz") }
            guard clip.seconds >= 0.5 else { throw VoxCPM2Error("Reference audio must be at least half a second") }
            if mode != .continuation { refFeat = model.encodeAudio(clip.samples, padLeft: false) }
            if mode != .reference {
                promptFeat = model.encodeAudio(clip.samples, padLeft: true)
                // The maker joins the transcript, as given, to the target text before tokenizing.
                promptText = request.referenceText ?? ""
            }
        }
        var generation = VoxCPM2Generation()
        generation.cfgValue = options.cfgValue; generation.inferenceTimesteps = options.inferenceTimesteps; generation.retryBadcase = options.retryBadcase
        if let cap = request.maxTokens { generation.maxLength = max(1, min(cap, 4096)) }
        let (alone, continuing) = Self.plans(
            body: body, pieces: Self.chunks(body), prefix: prefix, promptText: promptText, anchored: mode == nil, generation: generation,
            tokenize: { VoxCPM2Model.tokenize($0, tokenizer: tokenizer) })
        // Every chunk on its own must fit the LM context with its whole length cap before anything is generated; a
        // continuation of chunk 0 is used only where it fits too (`plan`).
        let worst = alone.map { Self.contextNeed($0, referencePatches: refFeat?.dim(0), promptPatches: promptFeat?.dim(0) ?? 0) }.max() ?? 0
        if worst > model.config.maxLength {
            let fix = mode == nil ? "shorten the instructions" : "shorten the reference clip or its transcript"
            throw VoxCPM2Error("Prompt and speech exceed VoxCPM2's \(model.config.maxLength)-token context (\(worst)); \(fix)")
        }
        let seed = request.seed ?? UInt64(UInt32.random(in: 0...UInt32.max))
        // Chunk 0's own audio once it has ended at its stop head; a chunk 0 cut off at its cap, or as long as the
        // maker's bad case, has no clean ending to continue, so later chunks then speak on their own.
        var first: MLXArray?
        for index in alone.indices {
            let plan = Self.plan(index, alone: alone, continuing: continuing, firstPatches: first?.dim(0), context: model.config.maxLength)
            onPlan?(index, plan)
            let prompt = plan.continuesFirst ? model.prompt(text: plan.ids, promptFeat: first) : model.prompt(text: plan.ids, promptFeat: promptFeat, refFeat: refFeat)
            let base = Self.chunkSeed(seed, index: index)
            var result: (patches: [MLXArray], stopped: Bool) = ([], false)
            // Each attempt's own random state, scoped to this task (as Fish's): draws made elsewhere in the process can't
            // land between the seed and the samples. Same sequence as MLXRandom.seed; the decode continues it.
            var random = MLXRandom.RandomState(seed: base)
            // The maker's retry_badcase: up to three attempts (seed, seed + 1, seed + 2) while a run hits its cap.
            for attempt in 0..<(generation.retryBadcase ? generation.retryMaxTimes : 1) {
                random = MLXRandom.RandomState(seed: base &+ UInt64(attempt))
                result = try withRandomState(random) {
                    try model.generatePatches(prompt, maxLength: plan.maxLength, generation: generation, isCancelled: { control.isCancelled })
                }
                // The maker's bad case: as many patches as 6× the target tokens, i.e. no stop.
                if !generation.retryBadcase || Float(result.patches.count) < Float(plan.target) * generation.retryRatio { break }
            }
            if index == 0, result.stopped, Float(result.patches.count) < Float(plan.target) * generation.retryRatio, continuing.contains(where: { $0 != nil }) {
                first = Self.continuationPrompt(result.patches)
            }
            let context = prompt.contextPatches > 0 ? prompt.feats[(prompt.feats.dim(0) - prompt.contextPatches)...] : nil
            let samples = withRandomState(random) { model.decode(context: context, patches: result.patches) }
            guard samples.allSatisfy(\.isFinite) else { throw EngineError("VoxCPM2 produced non-finite audio") }
            if control.isCancelled { return }
            try await emit(samples)
            Memory.clearCache()
        }
    }
}
