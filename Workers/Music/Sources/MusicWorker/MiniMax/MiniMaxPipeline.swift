// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXLMCommon
import Tokenizers

/// MiniMax Music 3 generation, following the maker's diffusers blocks step by step: tokenize → guided autoregressive
/// frames (global LM semantic code + depth-decoder residual codes, their hidden states kept) → 200-frame windows with a
/// 100-frame hop, each flow-matched with classifier-free guidance and the previous window's trailing latents blended in
/// → vocoded, cropped and stitched windows at 44.1 kHz stereo.
final class MiniMaxPipeline {
    static let arGuidance: Float = 1.5
    static let arTopK = 50
    static let chunkFrames = 200
    static let chunkHop = 100
    static let overlapLatents = 172
    static let cropLeftLatents = 86
    static let cropRightLatents = 344 - 86

    let c: MiniMaxConfiguration
    let lm: MiniMaxLanguageModel
    let depth: MiniMaxDepthDecoder
    let condition: MiniMaxConditionEncoder
    let transformer: MiniMaxFlowTransformer
    let vocoder: MiniMaxVocoder
    let tokenizer: (any Tokenizers.Tokenizer)?
    /// The language-model head restricted to the rows sampling can choose (semantic codes, then end of audio). The logits
    /// of those rows equal the full head's; every other row is masked out before sampling in the reference.
    private(set) var audioHead: MLXArray
    /// Parity only: when set, each depth step's guided logits (1, audio vocabulary) are appended here.
    var depthLogitsTrace: [MLXArray]?
    /// Global-model passes over fed-back frames in the last `frames` call (tests check no pass is wasted).
    private(set) var lastFeedbackPasses = 0

    init(
        configuration: MiniMaxConfiguration, lm: MiniMaxLanguageModel, depth: MiniMaxDepthDecoder, condition: MiniMaxConditionEncoder,
        transformer: MiniMaxFlowTransformer, vocoder: MiniMaxVocoder, tokenizer: (any Tokenizers.Tokenizer)?
    ) {
        c = configuration; self.lm = lm; self.depth = depth; self.condition = condition; self.transformer = transformer; self.vocoder = vocoder
        self.tokenizer = tokenizer
        audioHead = Self.restrictedHead(lm.lmHead.weight, offset: MiniMaxPrompt.audioCodeOffset, count: MiniMaxPrompt.semanticVocabulary, end: MiniMaxPrompt.audioEndToken)
        eval(audioHead)
        // The full head (200,000 rows) is never used after this; keeping only the restricted rows frees about 1.6 GB.
        try? lm.lmHead.update(parameters: ModuleParameters.unflattened(["weight": audioHead]), verify: .none)
    }

    static func restrictedHead(_ weight: MLXArray, offset: Int, count: Int, end: Int) -> MLXArray {
        concatenated([weight[offset..<(offset + count)], weight[end..<(end + 1)]], axis: 0)
    }

    struct Request {
        var caption: String
        var lyrics: String
        var duration: Double
        var seed: UInt64
        var steps: Int
        var guidance: Float
    }
    struct Output {
        var left: [Float], right: [Float]
        var frames: Int
        var ended: Bool
        var promptTokens: Int
    }

    // MARK: Prompt

    func tokens(caption: String, lyrics: String) throws -> MLXArray {
        guard let tokenizer else { throw MusicSupport.failure("MiniMax Music 3 tokenizer is not loaded.") }
        let ids = tokenizer.encode(text: MiniMaxPrompt.text(caption: caption, lyrics: lyrics), addSpecialTokens: false)
        guard ids.count <= MiniMaxPrompt.maxPromptTokens else {
            throw MusicSupport.failure("The assembled prompt has \(ids.count) tokens; the maximum is \(MiniMaxPrompt.maxPromptTokens).")
        }
        return MLXArray((ids + MiniMaxPrompt.unconditional(ids)).map(Int32.init)).reshaped(2, ids.count)
    }

    // MARK: Autoregressive stage

    /// Top-k (50) sampling over float32 logits; masked entries are -1e9, as the reference's `nan_to_num` makes them.
    static func sampleTopK(_ logits: MLXArray, key: MLXArray) -> MLXArray {
        let values = MLX.where(MLX.isNaN(logits), MLXArray(Float(-1e9)), logits).asType(.float32)
        let k = Swift.min(arTopK, values.dim(-1))
        let threshold = MLX.min(MLX.top(values, k: k, axis: -1), axis: -1, keepDims: true)
        let masked = MLX.where(values .< threshold, MLXArray(Float(-1e9)), values)
        return MLXRandom.categorical(masked, axis: -1, key: key)
    }

    /// The guided semantic step: `uncond + (cond - uncond)·1.5`, restricted to the conditional row's top 50 among the
    /// allowed entries. Logits are (2, allowed) over `audioHead` rows; returns an index into those rows.
    static func guidedSemantic(_ logits: MLXArray, key: MLXArray) -> MLXArray {
        let conditional = logits[0..<1], unconditional = logits[1..<2]
        var guided = unconditional + (conditional - unconditional) * arGuidance
        let threshold = MLX.min(MLX.top(conditional, k: Swift.min(arTopK, conditional.dim(-1)), axis: -1), axis: -1, keepDims: true)
        guided = MLX.where(conditional .< threshold, MLXArray(Float(-1e9)), guided)
        return sampleTopK(guided, key: key)
    }

    /// Semantic-code embedding plus the seven residual-code embeddings, scaled by 1/√8, for the global model's feedback.
    func embedFrame(_ codes: MLXArray) -> MLXArray {
        let semantic = lm.model.embedTokens(codes[0..., 0..<1] + Int32(MiniMaxPrompt.audioCodeOffset))
        let offsets = (MLXArray(Int32(0)..<Int32(c.codebooks - 1)) * Int32(c.audioVocabulary)).reshaped(1, -1)
        let residual = depth.audioEmbeddings(codes[0..., 1...] + offsets).sum(axis: 1, keepDims: true)
        return (semantic + residual.asType(semantic.dtype)) * Float(pow(Double(c.codebooks), -0.5))
    }

    /// Residual codes c1…c7 for one frame from the global hidden state (2, hidden) and the semantic code (2,).
    /// Returns codes (2, codebooks) and the conditional depth hidden states (1, 7·hidden).
    /// The local model's half type: the global model's BF16 states are cast to it at the boundary (MLX would otherwise
    /// promote mixed BF16/FP16 arithmetic to float32).
    var depthType: DType { depth.projection.weight.dtype }

    func depthCodes(lastHidden: MLXArray, semantic: MLXArray, key: inout MLXArray, forced: [Int32]? = nil) -> (MLXArray, MLXArray) {
        var sequence = [depth.projection(lastHidden.asType(depthType))[0..., .newAxis, 0...]]
        let codeEmbedding = lm.model.embedTokens(semantic + Int32(MiniMaxPrompt.audioCodeOffset)).asType(depthType)
        sequence.append(depth.projection(codeEmbedding)[0..., .newAxis, 0...])
        var codes = [semantic], hidden: [MLXArray] = []
        for index in 1..<c.codebooks {
            let h = depth(concatenated(sequence, axis: 1))[0..., -1, 0...]
            hidden.append(h[0..<1])
            let logits = depth.heads[index - 1](h).asType(.float32)
            let guided = logits[1..<2] + (logits[0..<1] - logits[1..<2]) * Self.arGuidance
            depthLogitsTrace?.append(guided)
            let code: MLXArray
            if let forced {
                code = MLXArray([forced[index], forced[index]])
            } else {
                let split = MLXRandom.split(key: key); key = split.0
                code = MLX.repeated(Self.sampleTopK(guided, key: split.1).asType(.int32), count: 2, axis: 0)
            }
            codes.append(code)
            if index < c.codebooks - 1 {
                let embedding = depth.audioEmbeddings(code + Int32((index - 1) * c.audioVocabulary))
                sequence.append(depth.projection(embedding)[0..., .newAxis, 0...])
            }
        }
        return (stacked(codes, axis: 1), concatenated(hidden, axis: -1))
    }

    /// Per-frame hidden states (1, frames, 8·hidden) for at most `maxFrames` frames. `forced` (teacher forcing, for parity)
    /// replaces sampling with the given frames' eight codes; the end token is then never sampled.
    func frames(
        tokens: MLXArray, maxFrames: Int, seed: UInt64, forced: [[Int32]]? = nil, cancelled: () -> Bool = { false },
        progress: (Double) -> Void = { _ in }
    ) throws -> (hidden: MLXArray, ended: Bool, codes: [[Int32]]) {
        var key = MLXRandom.key(seed)
        let cache = lm.makeCache()
        var last = lm.hidden(lm.model.embedTokens(tokens), cache: cache)[0..., -1, 0...]
        var frames: [MLXArray] = [], codesOut: [[Int32]] = [], ended = false
        let steps = forced.map { $0.count } ?? (maxFrames + 1)
        lastFeedbackPasses = 0
        for frameIndex in 0..<steps {
            if cancelled() { throw MusicSupport.failure("cancelled") }
            let semantic: MLXArray
            if let forced {
                semantic = MLXArray([forced[frameIndex][0], forced[frameIndex][0]])
            } else {
                let logits = matmul(last, audioHead.T).asType(.float32)
                let split = MLXRandom.split(key: key); key = split.0
                let index = Int(Self.guidedSemantic(logits, key: split.1).asType(.int32).item(Int32.self))
                if index == MiniMaxPrompt.semanticVocabulary { ended = true; break }
                semantic = MLXArray([Int32(index), Int32(index)])
            }
            let (codes, depthHidden) = depthCodes(lastHidden: last, semantic: semantic, key: &key, forced: forced?[frameIndex])
            if frameIndex > 0 || forced != nil {
                frames.append(concatenated([last[0..<1].asType(depthType), depthHidden], axis: -1))
            }
            codesOut.append(codes[0].asArray(Int32.self))
            progress(Double(frames.count) / Double(Swift.max(1, maxFrames)))
            // As the reference: the budget ends the loop before the frame is fed back (no unused global-model pass).
            if forced == nil ? frames.count >= maxFrames : frameIndex == steps - 1 {
                if let frame = frames.last { eval(frame) }
                break
            }
            let next = lm.hidden(embedFrame(codes), cache: cache)
            lastFeedbackPasses += 1
            last = next[0..., -1, 0...]
            eval(last); if let frame = frames.last { eval(frame) }
        }
        guard !frames.isEmpty else { throw MusicSupport.failure("MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately.") }
        return (stacked(frames, axis: 1), ended, codesOut)
    }

    // MARK: Flow matching

    /// The maker's Euler schedule: sigmas `1 - linspace(1, 1/N, N)` then 1 (flow time, 0 = noise).
    static func sigmas(_ steps: Int) -> [Float] {
        let raw: [Float] = steps == 1 ? [1] : (0..<steps).map { Float(1.0 + (1.0 / Double(steps) - 1.0) * Double($0) / Double(steps - 1)) }
        return raw.map { 1 - $0 } + [1]
    }

    /// One window: condition (1, L, dim), noise (1, L, channels). Returns the denoised latents.
    func denoise(
        condition: MLXArray, noise: MLXArray, steps: Int, guidance: Float, previousLatent: MLXArray?, previousCondition: MLXArray?,
        cancelled: () -> Bool, progress: (Int) -> Void
    ) throws -> (latents: MLXArray, condition: MLXArray) {
        var cond = condition, overlap = 0
        if let previousLatent, let previousCondition {
            overlap = Swift.min(previousLatent.dim(1), cond.dim(1))
            if overlap > 0 { cond = concatenated([previousCondition[0..., 0..<overlap], cond[0..., overlap...]], axis: 1) }
        }
        let noisePrompt = overlap > 0 ? noise[0..., 0..<overlap] : nil
        let sigma = Self.sigmas(steps)
        var x = noise
        let guided = guidance != 1
        let pairCondition = guided ? concatenated([cond, MLXArray.zeros(like: cond)], axis: 0) : cond
        for index in 0..<steps {
            if cancelled() { throw MusicSupport.failure("cancelled") }
            let t = sigma[index]
            if overlap > 0, let noisePrompt, let previousLatent {
                let blend = (1 - (1 - 1e-6) * t) * noisePrompt + t * previousLatent[0..., 0..<overlap]
                x = concatenated([blend.asType(x.dtype), x[0..., overlap...]], axis: 1)
            }
            // Guidance 1 is the conditional velocity alone, so the unconditional pass is skipped.
            let pair = guided ? concatenated([x, x], axis: 0) : x
            let time = MLXArray(guided ? [t, t] : [t]).asType(x.dtype)
            let velocities = transformer(pair, time: time, condition: pairCondition).asType(.float32)
            let conditional = velocities[0..<1]
            let velocity = guided ? velocities[1..<2] + guidance * (conditional - velocities[1..<2]) : conditional
            x = (x.asType(.float32) + (sigma[index + 1] - t) * velocity).asType(noise.dtype)
            eval(x)
            progress(index + 1)
        }
        if overlap > 0, let previousLatent { x = concatenated([previousLatent[0..., 0..<overlap], x[0..., overlap...]], axis: 1) }
        return (x, cond)
    }

    /// Window starts in frames, as `MiniMaxMusic3PrepareChunksStep`.
    static func chunkStarts(_ frames: Int) -> [Int] {
        frames <= chunkFrames ? [0] : Array(stride(from: 0, to: frames - chunkHop, by: chunkHop))
    }

    func latents(
        frameHidden: MLXArray, steps: Int, guidance: Float, key: MLXArray, cancelled: () -> Bool, progress: (Double) -> Void,
        noise: ((Int, Int) -> MLXArray)? = nil
    ) throws -> [MLXArray] {
        let starts = Self.chunkStarts(frameHidden.dim(1))
        var chunks: [MLXArray] = [], previousLatent: MLXArray?, previousCondition: MLXArray?
        var key = key
        let dtype = transformer.projIn.weight.dtype == .uint32 ? DType.float16 : transformer.projIn.weight.dtype
        for (k, start) in starts.enumerated() {
            let end = Swift.min(start + Self.chunkFrames, frameHidden.dim(1))
            let cond = condition(frameHidden[0..., start..<end].asType(dtype)).asType(dtype)
            let length = cond.dim(1)
            let initial: MLXArray
            if let noise {
                initial = noise(k, length).asType(dtype)
            } else {
                let split = MLXRandom.split(key: key); key = split.0
                initial = MLXRandom.normal([1, length, c.latent], key: split.1).asType(dtype)
            }
            let (x, used) = try denoise(
                condition: cond, noise: initial, steps: steps, guidance: guidance, previousLatent: previousLatent, previousCondition: previousCondition,
                cancelled: cancelled, progress: { step in progress((Double(k) + Double(step) / Double(steps)) / Double(starts.count)) })
            let overlapStart = Swift.max(0, length - 2 * Self.overlapLatents), overlapEnd = Swift.max(overlapStart, length - Self.overlapLatents)
            previousLatent = x[0..., overlapStart..<overlapEnd]; previousCondition = used[0..., overlapStart..<overlapEnd]
            chunks.append(x)
        }
        return chunks
    }

    /// Vocodes each window, crops the overlaps and stitches the stereo waveform.
    func decode(_ chunks: [MLXArray], cancelled: () -> Bool) throws -> (left: [Float], right: [Float]) {
        var left: [Float] = [], right: [Float] = []
        for (index, latents) in chunks.enumerated() {
            if cancelled() { throw MusicSupport.failure("cancelled") }
            let wave = vocoder(latents)
            let total = wave.dim(-1)
            let start = index == 0 ? 0 : Self.cropLeftLatents * c.outputHop
            let stop = total - (index == chunks.count - 1 ? 0 : Self.cropRightLatents * c.outputHop)
            guard stop > start else { continue }
            let piece = clip(wave[0, 0..., start..<stop].asType(.float32), min: -1, max: 1)
            eval(piece)
            left += piece[0].asArray(Float.self); right += piece[1].asArray(Float.self)
        }
        return (left, right)
    }

    func generate(_ request: Request, cancelled: () -> Bool, progress: (SongProgress) -> Void) throws -> Output {
        let tokens = try tokens(caption: request.caption, lyrics: request.lyrics)
        let maxFrames = Swift.min(Int(request.duration * c.frameRate), MiniMaxPrompt.maxAudioFrames)
        guard maxFrames > 0 else { throw MusicSupport.failure("The duration is shorter than one audio frame (1/\(Int(c.frameRate)) s).") }
        let ar = try frames(tokens: tokens, maxFrames: maxFrames, seed: request.seed, cancelled: cancelled) { progress(SongProgress(stage: .codes, fraction: $0)) }
        let chunks = try latents(
            frameHidden: ar.hidden, steps: request.steps, guidance: request.guidance, key: MLXRandom.key(request.seed &+ 0x9E37_79B9_7F4A_7C15),
            cancelled: cancelled
        ) { progress(SongProgress(stage: .diffusion, fraction: $0)) }
        progress(SongProgress(stage: .decoding, fraction: 0))
        let audio = try decode(chunks, cancelled: cancelled)
        progress(SongProgress(stage: .decoding, fraction: 1))
        return Output(left: audio.left, right: audio.right, frames: ar.hidden.dim(1), ended: ar.ended, promptTokens: tokens.dim(1))
    }
}
