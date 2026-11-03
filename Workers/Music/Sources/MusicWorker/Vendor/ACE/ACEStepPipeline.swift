//
//  ACEStepPipeline.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  One song, the way upstream's `generate_music` makes it (acestep/
//  inference.py and acestep/core/generation/handler/ at ca1e85f):
//
//    1. the planner fills what the caller left open and, when `thinking`,
//       writes duration × 5 audio codes;
//    2. codes (or a cover's source) become 25 Hz "hints" through the FSQ
//       detokenizer — text-to-music with codes runs as a cover of them,
//       with the cover instruction, exactly as upstream switches it;
//    3. the caption prompt goes through Qwen3-Embedding, the lyrics through
//       its embedding table, the timbre reference (or 30 s of the silence
//       latent) through the timbre encoder, and the three are packed;
//    4. the DiT denoises from seeded noise; the VAE decodes in 512-frame
//       windows with 64 frames of overlap thrown away each side;
//    5. peak above 1 is scaled down, then normalised to −1 dBFS, then faded.
//

import Foundation
import MLX
import MLXRandom
import IntegrationKit

// The loaded models and the generation path.
final class ACEStepPipeline {

    let configuration: ACEStepConfiguration
    let network: ACEStepNetwork
    let vae: OobleckVAE
    let textEncoder: TextEncoder
    let planner: Planner
    // `[T, 64]` latents of silence: the timbre reference when none is given, and padding.
    let silence: MLXArray
    let dtype: DType
    // The VAE executes in the selected half precision.
    let vaeDType: DType

    static let sampleRate = 48_000
    static let hop = 1_920

    var cancellation: NativeCancellation?
    init(configuration: ACEStepConfiguration, network: ACEStepNetwork, vae: OobleckVAE, textEncoder: TextEncoder, planner: Planner, silence: MLXArray) {
        self.configuration = configuration; self.network = network; self.vae = vae
        self.textEncoder = textEncoder; self.planner = planner; self.silence = silence
        dtype = .float16; vaeDType = .float16
    }

    // MARK: - Planning

    // The planner's settings from a request.
    static func plannerSettings(_ r: SongRequest) -> PlannerSettings {
        PlannerSettings(
            temperature: Float(r.plannerTemperature), cfgScale: Float(r.plannerGuidance),
            topK: r.plannerTopK > 0 ? r.plannerTopK : nil,
            topP: r.plannerTopP > 0 && r.plannerTopP < 1 ? Float(r.plannerTopP) : nil,
            repetitionPenalty: Float(r.plannerRepetitionPenalty), negativePrompt: r.plannerNegativePrompt,
            useCotMetas: r.planMetadata, useCotCaption: r.rewriteCaption, useCotLanguage: r.detectLanguage,
            constrainedDecoding: r.constrainedDecoding)
    }

    // Phase 1, as upstream gates it. Returns nil when the planner does not run.
    func plan(_ r: SongRequest, random: MLXRandom.RandomState) throws(ACEStepError) -> (PlannedMetadata, String, Double)? {
        let isDirect = r.task == .cover || r.task == .repaint
        let needCot = r.planMetadata || r.rewriteCaption || r.detectLanguage
        guard (r.thinking || needCot) && !isDirect else { return nil }
        let fixed = MetadataConstraint.Fixed(
            bpm: r.bpm.map(String.init), caption: nil, duration: r.duration.map { String(Int($0)) },
            keyscale: r.key.flatMap { $0.trimmingCharacters(in: .whitespaces).isEmpty ? nil : $0 },
            language: nil, timesignature: r.timeSignature.map(String.init))
        let hasAll = fixed.bpm != nil && fixed.keyscale != nil && fixed.timesignature != nil && fixed.duration != nil
        guard !hasAll && r.planMetadata else {
            return (
                PlannedMetadata(
                    bpm: r.bpm, duration: r.duration.map { Int($0) }, keyscale: fixed.keyscale,
                    timesignature: fixed.timesignature), "", 0
            )
        }
        let start = Date()
        let (meta, text, _) = try planner.reason(
            caption: r.caption, lyrics: r.effectiveLyrics, fixed: fixed,
            settings: Self.plannerSettings(r), random: random)
        return (meta, text, Date().timeIntervalSince(start))
    }

    // MARK: - Conditioning

    // `- bpm: …` lines, as `_dict_to_meta_string` writes them.
    static func metas(bpm: Int?, timeSignature: String?, key: String?, duration: Double?) -> String {
        var text = "- bpm: \(bpm.map(String.init) ?? "N/A")\n"
        text += "- timesignature: \(timeSignature ?? "N/A")\n"
        text += "- keyscale: \(key ?? "N/A")\n"
        if let duration { text += "- duration: \(Int(duration)) seconds\n" }
        return text
    }

    static func textPrompt(instruction: String, caption: String, metas: String) -> String {
        let line = instruction.hasSuffix(":") ? instruction : instruction + ":"
        return "# Instruction\n\(line)\n\n# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }

    static func lyricPrompt(lyrics: String, language: String) -> String {
        "# Languages\n\(language)\n\n# Lyric\n\(lyrics)<|endoftext|>"
    }

    // `[1, T, 64]` hints for 5 Hz codes.
    func hints(codes: [Int]) -> MLXArray {
        let ids = MLXArray(codes.map(Int32.init)).reshaped(1, -1)
        let quantized = network.tokenizer.quantizer.embedding(codes: ids, dtype: dtype)
        return network.detokenizer(quantized)
    }

    // `[T, 64]` silence, tiled when longer than the stored latent.
    func silence(frames: Int) -> MLXArray {
        let stored = silence.dim(0)
        if frames <= stored { return silence[..<frames] }
        let repeats = (frames + stored - 1) / stored
        return MLX.tiled(silence, repetitions: [repeats, 1])[..<frames]
    }

    // Encodes the packed conditions and the context.
    func conditions(
        caption: String, instruction: String, metas: String, lyrics: String, language: String,
        references: MLXArray, referenceOrder: [Int], source: MLXArray, chunkMask: MLXArray
    )
        -> DiffusionConditions
    {
        let textIDs = textEncoder.tokens(
            Self.textPrompt(instruction: instruction, caption: caption, metas: metas),
            maxLength: 256)
        let lyricIDs = textEncoder.tokens(Self.lyricPrompt(lyrics: lyrics, language: language), maxLength: 2048)
        var mark = Date()
        func lap(_ name: String, _ a: MLXArray) {
            guard Profile.enabled else { return }
            MLX.eval(a)
            Profile.log(String(format: "PROFILE cond %@ %.2f s", name, Date().timeIntervalSince(mark)))
            mark = Date()
        }
        lap("tokenise", MLXArray(0))
        let text = textEncoder.encode(textIDs).asType(dtype)
        lap("text encoder", text)
        let lyric = textEncoder.embed(lyricIDs).asType(dtype)
        lap("lyrics", lyric)
        let (encoder, _) = network.encoder(
            text: text, textKeep: MLXArray.ones([1, textIDs.count], dtype: .int32),
            lyrics: lyric, lyricKeep: MLXArray.ones([1, lyricIDs.count], dtype: .int32),
            references: references, referenceOrder: referenceOrder)
        lap("condition encoder", encoder)
        let context = MLX.concatenated(
            [
                source,
                MLX.broadcast(
                    chunkMask.expandedDimensions(axis: -1).asType(dtype),
                    to: source.shape)
            ], axis: -1)
        return DiffusionConditions(encoder: encoder, context: context)
    }

    // MARK: - Audio in

    // A file as `[2, samples]` at 48 kHz, clamped to ±1 (upstream's `_normalize_audio_to_stereo_48k`).
    static func loadStereo48k(_ path: String) throws(ACEStepError) -> [[Float]] {
        do {
            let pcm = try NativeMedia.audio(URL(fileURLWithPath: path), sampleRate: 48000, channels: 2, maximumFrames: 48_000 * 600)
            let frames = pcm.samples.count / 2
            return (0..<2).map { c in (0..<frames).map { Swift.max(-1, Swift.min(1, pcm.samples[$0 * 2 + c])) } }
        } catch { throw .unreadableAudio(error.localizedDescription) }
    }

    // VAE latents `[1, T, 64]` for audio `[2, S]`: windows of 30 s with 2 s either side
    // discarded, sampled from the posterior with a seeded key (upstream samples unseeded).
    func encode(_ audio: [[Float]], key: MLXArray) throws(ACEStepError) -> MLXArray {
        let frames = audio[0].count
        let nlc = MLX.stacked([MLXArray(audio[0]), MLXArray(audio[1])], axis: -1).expandedDimensions(axis: 0).asType(vaeDType)
        let chunk = Self.sampleRate * 30, overlap = Self.sampleRate * 2
        if frames <= chunk { return vae.encodeSample(nlc, key: key).asType(dtype) }
        let stride = chunk - 2 * overlap
        let steps = (frames + stride - 1) / stride
        var parts: [MLXArray] = []
        let keys = MLXRandom.split(key: key, into: steps)
        for i in 0..<steps {
            if (try? cancellation?.check()) == nil && cancellation != nil { throw .cancelled }
            let core = i * stride, coreEnd = min(core + stride, frames)
            let start = max(0, core - overlap), end = min(frames, coreEnd + overlap)
            let latent = vae.encodeSample(nlc[0..., start..<end, 0...], key: keys[i])
            MLX.eval(latent)
            let factor = Double(end - start) / Double(latent.dim(1))
            let trimStart = Int((Double(core - start) / factor).rounded(.toNearestOrEven))
            let trimEnd = Int((Double(end - coreEnd) / factor).rounded(.toNearestOrEven))
            parts.append(latent[0..., trimStart..<(latent.dim(1) - trimEnd), 0...])
        }
        return MLX.concatenated(parts, axis: 1).asType(dtype)
    }

    // Upstream's `process_reference_audio`: 30 s made of three 10 s pieces from the
    // front, middle and back (repeating short audio), offsets drawn from `seed`.
    static func referenceClip(_ audio: [[Float]], seed: UInt64) -> [[Float]] {
        let target = 30 * sampleRate, piece = 10 * sampleRate
        var audio = audio
        if audio[0].count < target {
            let times = (target + audio[0].count - 1) / audio[0].count
            audio = audio.map { Array(repeating: $0, count: times).flatMap { $0 } }
        }
        var generator = SplitMix64(seed: seed)
        let total = audio[0].count, third = total / 3
        func start(_ base: Int, _ span: Int) -> Int { base + Int(generator.next() % UInt64(max(0, span - piece) + 1)) }
        let starts = [start(0, third), start(third, third), start(2 * third, total - 2 * third)]
        return audio.map { channel in starts.flatMap { Array(channel[$0..<min($0 + piece, channel.count)]) } }
    }

    // MARK: - Audio out

    // Upstream's MLX tiled decode: 512-frame windows, 64 frames of overlap discarded.
    func decode(_ latents: MLXArray, progress: ((Double) -> Void)?) throws(ACEStepError) -> [[Float]] {
        if (try? cancellation?.check()) == nil && cancellation != nil { throw .cancelled }
        let decodeStart = Date()
        let z = latents.asType(vaeDType)
        let frames = z.dim(1)
        let vae = self.vae
        // Compiled, as upstream compiles its MLX decode: the Snake activations fuse.
        let decodeWindow = compile { vae.decode($0) }
        let chunk = 512, overlap = 64
        var audio: MLXArray
        if frames <= chunk {
            audio = decodeWindow(z)
        } else {
            let stride = chunk - 2 * overlap
            let steps = (frames + stride - 1) / stride
            var parts: [MLXArray] = []
            var factor: Double?
            for i in 0..<steps {
                if (try? cancellation?.check()) == nil && cancellation != nil { throw .cancelled }
                let core = i * stride, coreEnd = min(core + stride, frames)
                let start = max(0, core - overlap), end = min(frames, coreEnd + overlap)
                let piece = decodeWindow(z[0..., start..<end, 0...])
                MLX.eval(piece)
                let f = factor ?? Double(piece.dim(1)) / Double(end - start)
                factor = f
                let trimStart = Int((Double(core - start) * f).rounded(.toNearestOrEven))
                let trimEnd = Int((Double(end - coreEnd) * f).rounded(.toNearestOrEven))
                let trimmed = piece[0..., trimStart..<(trimEnd > 0 ? piece.dim(1) - trimEnd : piece.dim(1)), 0...]
                MLX.eval(trimmed)
                parts.append(trimmed)
                if (i + 1) % 4 == 0 { MLX.Memory.clearCache() }
                progress?(Double(i + 1) / Double(steps))
            }
            audio = MLX.concatenated(parts, axis: 1)
        }
        audio = audio.asType(.float32)
        MLX.eval(audio)
        let convertStart = Date()
        defer {
            if ProcessInfo.processInfo.environment["ACESTEP_PROFILE"] != nil {
                Profile.log(
                    String(
                        format: "PROFILE vae windows %.2f s, to host %.2f s", convertStart.timeIntervalSince(decodeStart),
                        Date().timeIntervalSince(convertStart)))
            }
        }
        let channels = audio.dim(2)
        let planar = audio[0].transposed(1, 0)
        return (0..<channels).map { planar[$0].asArray(Float.self) }
    }

    // ``SongFinishing/finish(_:sampleRate:normalize:db:fadeIn:fadeOut:)`` at this engine's rate.
    static func finish(_ channels: [[Float]], normalize: Bool, db: Double, fadeIn: Double, fadeOut: Double) -> [[Float]] {
        SongFinishing.finish(
            channels, sampleRate: Double(sampleRate), normalize: normalize, db: db, fadeIn: fadeIn,
            fadeOut: fadeOut)
    }
}

// A small seeded generator for the reference-clip offsets.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
