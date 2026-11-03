//
//  SongRequest.swift
//  MusicGeneration
//
//  Created by David Sherlock on 2026.
//
//  Everything a song is made from, as data: the words, the musical facts,
//  the diffusion and planner settings, the task and its audio inputs, and
//  the finishing. Each field is one upstream parameter (ACE-Step 1.5's
//  `GenerationParams` / `GenerationConfig`, named in each comment) with
//  upstream's default, so a request written down replays exactly. JSON keys
//  are snake_case; the README maps every upstream name to its flag and key.
//

import Foundation

// A song to make.
public struct SongRequest: Codable, Sendable, Equatable {

    // Which model makes the song (`ace-step`, or the opt-in `yue2`).
    public var engine: SongEngine
    // YuE2's own settings; nil is upstream's defaults. Ignored by ACE-Step.
    public var yue: YuE2Options?

    // MARK: Words

    // `caption` — what the music is: genre, mood, instruments, production. Up to 512 characters.
    public var caption: String
    // `lyrics` — with `[verse]`, `[chorus]` tags; ignored when `instrumental`.
    public var lyrics: String
    // `instrumental` — no vocals; the lyrics become `[Instrumental]`.
    public var instrumental: Bool
    // `vocal_language` — an ISO 639-1 code (`en`, `ja`, `yue`…) or `unknown`.
    public var language: String

    // MARK: Musical facts (nil = the planner decides)

    // `bpm` — 30–300.
    public var bpm: Int?
    // `keyscale` — `C major`, `F# minor`, `Bb major`.
    public var key: String?
    // `timesignature` — 2, 3, 4 or 6 beats per bar.
    public var timeSignature: Int?
    // `duration` — seconds, 10–600.
    public var duration: Double?

    // MARK: Diffusion

    // `inference_steps` — denoising steps (turbo: 8).
    public var steps: Int
    // `guidance_scale` — classifier-free guidance on the DiT. Upstream forces 1 for turbo models; > 1 is honoured here, with APG.
    public var guidance: Double
    // `use_adg` — adaptive dual guidance (base models upstream; not implemented, see README).
    public var adaptiveDualGuidance: Bool
    // `cfg_interval_start` — fraction of the schedule (by timestep) where guidance starts.
    public var guidanceStart: Double
    // `cfg_interval_end` — where it stops.
    public var guidanceEnd: Double
    // `shift` — timestep shift, `t' = s·t / (1 + (s−1)·t)`; 1–5.
    public var shift: Double
    // `timesteps` — a custom schedule from 1 towards 0 (≤ 20 values, each snapped to the nearest the turbo model was distilled on); overrides steps and shift.
    public var timesteps: [Double]?
    // `infer_method` — `ode` (deterministic) or `sde` (re-noises each step).
    public var method: DiffusionMethod
    // `sampler_mode` — `euler` or `heun`.
    public var sampler: SamplerKind
    // `velocity_norm_threshold` — clamp each velocity's norm to this multiple of the latent's (0 = off).
    public var velocityNormThreshold: Double
    // `velocity_ema_factor` — blend each velocity with the previous one (0 = off).
    public var velocityEMA: Double
    // `dcw_enabled` — differential correction in the wavelet domain (on for turbo).
    public var dcw: Bool
    // `dcw_mode` — `low`, `high`, `double` or `pix`.
    public var dcwMode: DCWMode
    // `dcw_scaler` — low-band (or single) strength.
    public var dcwScaler: Double
    // `dcw_high_scaler` — high-band strength (double mode).
    public var dcwHighScaler: Double
    // `dcw_wavelet` — only `haar` is implemented (upstream's MLX path is native only for Haar too).
    public var dcwWavelet: String
    // `seed` — nil picks one at random (and the result records it).
    public var seed: UInt32?
    // `retake_seed` — a second noise for variations.
    public var retakeSeed: UInt32?
    // `retake_variance` — 0 keeps the seed's noise, 1 is the retake seed's.
    public var retakeVariance: Double
    // `latent_shift` — added to the latents before decoding.
    public var latentShift: Double
    // `latent_rescale` — multiplied into the latents before decoding.
    public var latentRescale: Double

    // MARK: Planner (the 5 Hz LM)

    // `thinking` — let the planner write the audio codes the DiT starts from.
    public var thinking: Bool
    // `use_cot_metas` — let it fill bpm, key, time signature and duration you left open.
    public var planMetadata: Bool
    // `use_cot_caption` — let it rewrite the caption into the detailed form the DiT was trained on.
    public var rewriteCaption: Bool
    // `use_cot_language` — let it name the vocal language.
    public var detectLanguage: Bool
    // `lm_temperature`.
    public var plannerTemperature: Double
    // `lm_cfg_scale` — guidance on the codes.
    public var plannerGuidance: Double
    // `lm_top_k` — 0 = off.
    public var plannerTopK: Int
    // `lm_top_p` — 1 = off.
    public var plannerTopP: Double
    // `repetition_penalty` (upstream's LM call; 1 = off).
    public var plannerRepetitionPenalty: Double
    // `lm_negative_prompt` — the unconditional side of the codes' guidance.
    public var plannerNegativePrompt: String
    // `use_constrained_decoding` — hold the planner to the metadata format and the code count.
    public var constrainedDecoding: Bool

    // MARK: Task

    // `task_type`.
    public var task: SongTask
    // `src_audio` — the song to cover or repaint; a path.
    public var sourceAudio: String?
    // `reference_audio` — a recording whose timbre the song takes; a path.
    public var referenceAudio: String?
    // `audio_codes` — 5 Hz codes to start from instead of the planner's (0–63999 each).
    public var audioCodes: [Int]?
    // `instruction` — the DiT's task line; nil uses the task's own.
    public var instruction: String?
    // `repainting_start` — seconds; negative extends before the start.
    public var repaintStart: Double
    // `repainting_end` — seconds; nil is the end of the source, past it extends.
    public var repaintEnd: Double?
    // `chunk_mask_mode` — `auto` (the model decides) or `explicit` (the repaint range);
    // nil is explicit for a repaint, as upstream's own interface chooses, and auto otherwise.
    public var chunkMaskMode: ChunkMaskMode?
    // `repaint_mode` — `conservative`, `balanced` or `aggressive`.
    public var repaintMode: RepaintMode
    // `repaint_strength` — in balanced mode, 0 aggressive … 1 conservative.
    public var repaintStrength: Double
    // `repaint_latent_crossfade_frames` — 25 Hz frames blended at the repaint edges; nil
    // takes it from the repaint mode, as upstream's handler always does.
    public var repaintCrossfadeFrames: Int?
    // `repaint_wav_crossfade_sec` — seconds crossfaded when splicing the waveform back; nil
    // takes it from the repaint mode.
    public var repaintWaveCrossfade: Double?
    // `audio_cover_strength` — 0–1, how long the cover keeps to the source's codes.
    public var coverStrength: Double
    // `cover_noise_strength` — 0 starts from noise, 1 from the source itself.
    public var coverNoiseStrength: Double

    // MARK: Finishing

    // `enable_normalization` — peak-normalise.
    public var normalize: Bool
    // `normalization_db` — the peak, dBFS.
    public var normalizationDB: Double
    // `fade_in_duration` — seconds.
    public var fadeIn: Double
    // `fade_out_duration` — seconds.
    public var fadeOut: Double
    // `batch_size` — songs made from one request (seeds seed, seed+1, …).
    public var batchSize: Int

    public init(caption: String, lyrics: String = "", instrumental: Bool = false) {
        self.caption = caption
        self.lyrics = lyrics
        self.instrumental = instrumental
        engine = .aceStep
        language = "unknown"
        steps = 8
        guidance = 1
        adaptiveDualGuidance = false
        guidanceStart = 0
        guidanceEnd = 1
        shift = 1
        method = .ode
        sampler = .euler
        velocityNormThreshold = 0
        velocityEMA = 0
        dcw = true
        dcwMode = .double
        dcwScaler = 0.05
        dcwHighScaler = 0.02
        dcwWavelet = "haar"
        retakeVariance = 0
        latentShift = 0
        latentRescale = 1
        thinking = true
        planMetadata = true
        rewriteCaption = true
        detectLanguage = true
        plannerTemperature = 0.85
        plannerGuidance = 2
        plannerTopK = 0
        plannerTopP = 0.9
        plannerRepetitionPenalty = 1
        plannerNegativePrompt = "NO USER INPUT"
        constrainedDecoding = true
        task = .text2music
        repaintStart = 0
        repaintMode = .balanced
        repaintStrength = 0.5
        coverStrength = 1
        coverNoiseStrength = 0
        normalize = true
        normalizationDB = -1
        fadeIn = 0
        fadeOut = 0
        batchSize = 1
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case engine, yue
        case caption, lyrics, instrumental, language, bpm, key
        case timeSignature = "time_signature"
        case duration, steps, guidance
        case adaptiveDualGuidance = "adaptive_dual_guidance"
        case guidanceStart = "guidance_start"
        case guidanceEnd = "guidance_end"
        case shift, timesteps, method, sampler
        case velocityNormThreshold = "velocity_norm_threshold"
        case velocityEMA = "velocity_ema"
        case dcw
        case dcwMode = "dcw_mode"
        case dcwScaler = "dcw_scaler"
        case dcwHighScaler = "dcw_high_scaler"
        case dcwWavelet = "dcw_wavelet"
        case seed
        case retakeSeed = "retake_seed"
        case retakeVariance = "retake_variance"
        case latentShift = "latent_shift"
        case latentRescale = "latent_rescale"
        case thinking
        case planMetadata = "plan_metadata"
        case rewriteCaption = "rewrite_caption"
        case detectLanguage = "detect_language"
        case plannerTemperature = "planner_temperature"
        case plannerGuidance = "planner_guidance"
        case plannerTopK = "planner_top_k"
        case plannerTopP = "planner_top_p"
        case plannerRepetitionPenalty = "planner_repetition_penalty"
        case plannerNegativePrompt = "planner_negative_prompt"
        case constrainedDecoding = "constrained_decoding"
        case task
        case sourceAudio = "source_audio"
        case referenceAudio = "reference_audio"
        case audioCodes = "audio_codes"
        case instruction
        case repaintStart = "repaint_start"
        case repaintEnd = "repaint_end"
        case chunkMaskMode = "chunk_mask_mode"
        case repaintMode = "repaint_mode"
        case repaintStrength = "repaint_strength"
        case repaintCrossfadeFrames = "repaint_crossfade_frames"
        case repaintWaveCrossfade = "repaint_wave_crossfade"
        case coverStrength = "cover_strength"
        case coverNoiseStrength = "cover_noise_strength"
        case normalize
        case normalizationDB = "normalization_db"
        case fadeIn = "fade_in"
        case fadeOut = "fade_out"
        case batchSize = "batch_size"
    }

    // A request read back: every field optional in JSON, upstream's default where absent.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var r = SongRequest(caption: try c.decodeIfPresent(String.self, forKey: .caption) ?? "")
        func v<T: Decodable>(_ key: CodingKeys, _ current: T) throws -> T { try c.decodeIfPresent(T.self, forKey: key) ?? current }
        r.engine = try v(.engine, r.engine)
        r.yue = try c.decodeIfPresent(YuE2Options.self, forKey: .yue)
        r.lyrics = try v(.lyrics, r.lyrics)
        r.instrumental = try v(.instrumental, r.instrumental)
        r.language = try v(.language, r.language)
        r.bpm = try c.decodeIfPresent(Int.self, forKey: .bpm)
        r.key = try c.decodeIfPresent(String.self, forKey: .key)
        r.timeSignature = try c.decodeIfPresent(Int.self, forKey: .timeSignature)
        r.duration = try c.decodeIfPresent(Double.self, forKey: .duration)
        r.steps = try v(.steps, r.steps)
        r.guidance = try v(.guidance, r.guidance)
        r.adaptiveDualGuidance = try v(.adaptiveDualGuidance, r.adaptiveDualGuidance)
        r.guidanceStart = try v(.guidanceStart, r.guidanceStart)
        r.guidanceEnd = try v(.guidanceEnd, r.guidanceEnd)
        r.shift = try v(.shift, r.shift)
        r.timesteps = try c.decodeIfPresent([Double].self, forKey: .timesteps)
        r.method = try v(.method, r.method)
        r.sampler = try v(.sampler, r.sampler)
        r.velocityNormThreshold = try v(.velocityNormThreshold, r.velocityNormThreshold)
        r.velocityEMA = try v(.velocityEMA, r.velocityEMA)
        r.dcw = try v(.dcw, r.dcw)
        r.dcwMode = try v(.dcwMode, r.dcwMode)
        r.dcwScaler = try v(.dcwScaler, r.dcwScaler)
        r.dcwHighScaler = try v(.dcwHighScaler, r.dcwHighScaler)
        r.dcwWavelet = try v(.dcwWavelet, r.dcwWavelet)
        r.seed = try c.decodeIfPresent(UInt32.self, forKey: .seed)
        r.retakeSeed = try c.decodeIfPresent(UInt32.self, forKey: .retakeSeed)
        r.retakeVariance = try v(.retakeVariance, r.retakeVariance)
        r.latentShift = try v(.latentShift, r.latentShift)
        r.latentRescale = try v(.latentRescale, r.latentRescale)
        r.thinking = try v(.thinking, r.thinking)
        r.planMetadata = try v(.planMetadata, r.planMetadata)
        r.rewriteCaption = try v(.rewriteCaption, r.rewriteCaption)
        r.detectLanguage = try v(.detectLanguage, r.detectLanguage)
        r.plannerTemperature = try v(.plannerTemperature, r.plannerTemperature)
        r.plannerGuidance = try v(.plannerGuidance, r.plannerGuidance)
        r.plannerTopK = try v(.plannerTopK, r.plannerTopK)
        r.plannerTopP = try v(.plannerTopP, r.plannerTopP)
        r.plannerRepetitionPenalty = try v(.plannerRepetitionPenalty, r.plannerRepetitionPenalty)
        r.plannerNegativePrompt = try v(.plannerNegativePrompt, r.plannerNegativePrompt)
        r.constrainedDecoding = try v(.constrainedDecoding, r.constrainedDecoding)
        r.task = try v(.task, r.task)
        r.sourceAudio = try c.decodeIfPresent(String.self, forKey: .sourceAudio)
        r.referenceAudio = try c.decodeIfPresent(String.self, forKey: .referenceAudio)
        r.audioCodes = try c.decodeIfPresent([Int].self, forKey: .audioCodes)
        r.instruction = try c.decodeIfPresent(String.self, forKey: .instruction)
        r.repaintStart = try v(.repaintStart, r.repaintStart)
        r.repaintEnd = try c.decodeIfPresent(Double.self, forKey: .repaintEnd)
        r.chunkMaskMode = try c.decodeIfPresent(ChunkMaskMode.self, forKey: .chunkMaskMode)
        r.repaintMode = try v(.repaintMode, r.repaintMode)
        r.repaintStrength = try v(.repaintStrength, r.repaintStrength)
        r.repaintCrossfadeFrames = try c.decodeIfPresent(Int.self, forKey: .repaintCrossfadeFrames)
        r.repaintWaveCrossfade = try c.decodeIfPresent(Double.self, forKey: .repaintWaveCrossfade)
        r.coverStrength = try v(.coverStrength, r.coverStrength)
        r.coverNoiseStrength = try v(.coverNoiseStrength, r.coverNoiseStrength)
        r.normalize = try v(.normalize, r.normalize)
        r.normalizationDB = try v(.normalizationDB, r.normalizationDB)
        r.fadeIn = try v(.fadeIn, r.fadeIn)
        r.fadeOut = try v(.fadeOut, r.fadeOut)
        r.batchSize = try v(.batchSize, r.batchSize)
        self = r
    }

    // The lyrics the models see: `[Instrumental]` when instrumental or empty.
    public var effectiveLyrics: String {
        let trimmed = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        return instrumental || trimmed.isEmpty ? "[Instrumental]" : lyrics
    }

    // The JSON keys, in declaration order.
    public static var fieldNames: [String] { CodingKeys.allCases.map(\.rawValue) }
}
