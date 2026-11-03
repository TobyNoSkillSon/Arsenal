//
//  ACEStepPipeline+Generate.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The task logic: what the planner is asked, which latents the DiT starts
//  from and which instruction it reads, for text-to-music, covers and
//  repaints — in the order upstream's handler decides them.
//

import Foundation
import MLX
import MLXRandom

// The instruction upstream's interfaces give each task.
enum TaskInstruction {
    static let text2music = "Fill the audio semantic mask based on the given conditions:"
    static let cover = "Generate audio semantic tokens based on the given conditions:"
    static let repaint = "Repaint the mask area based on the given conditions:"
}

// Test hooks: tensors to inject in place of the engine's own, for parity runs.
struct ParityOverrides {
    var codes: [Int]?
    var noise: MLXArray?
    // `[1, T, 64]` in place of the VAE encoding of the source (it samples).
    var sourceLatents: MLXArray?
    // `[1, T, 64]` in place of the reference clip's encoding (random offsets, sampled).
    var referenceLatents: MLXArray?
}

extension ACEStepPipeline {

    // Every song a request makes (`batch_size` of them).
    // swiftlint:disable:next function_body_length
    func generate(
        _ request: SongRequest, overrides: ParityOverrides = .init(),
        progress: (@Sendable (SongProgress) -> Void)?, cancelled: () -> Bool
    ) throws(ACEStepError) -> [GeneratedSong] {
        try validate(request)
        var r = request
        let first = r.seed ?? UInt32.random(in: 0...UInt32.max)
        r.seed = first
        let seeds = (0..<max(1, r.batchSize)).map { first &+ UInt32($0) }
        if r.retakeVariance > 0 && r.retakeSeed == nil { r.retakeSeed = UInt32.random(in: 0...UInt32.max) }
        let retakes = r.retakeVariance > 0 ? seeds.indices.map { (r.retakeSeed ?? 0) &+ UInt32($0) } : nil
        if let key = r.key { r.key = KeyScale.normalised(key) ?? key }
        var timings = SongTimings()
        let started = Date()

        // 1. Planning.
        let randomState = MLXRandom.RandomState(seed: UInt64(first))
        progress?(SongProgress(stage: .planning, fraction: 0))
        let planned = try plan(r, random: randomState)
        timings.planning = planned?.2 ?? 0
        let meta = planned?.0
        let usesPlanner = planned != nil
        var resolved = r
        if usesPlanner, let meta {
            if resolved.bpm == nil { resolved.bpm = meta.bpm }
            if resolved.key == nil || resolved.key?.isEmpty == true { resolved.key = meta.keyscale }
            if resolved.timeSignature == nil { resolved.timeSignature = meta.timesignature.flatMap { Int($0) } }
            if resolved.duration == nil, let d = meta.duration { resolved.duration = Double(d) }
        }
        let ditCaption = usesPlanner && r.rewriteCaption ? (meta?.caption ?? r.caption) : r.caption
        let lyrics = r.effectiveLyrics

        // 2. Source audio and the latent length.
        var sourceLatents: MLXArray?
        var sourceWave: [[Float]]?
        var repaintSpan: (start: Double, end: Double)?
        var frames: Int
        let encodeKey = MLXRandom.key(UInt64(first) &+ 0x5EED)
        switch r.task {
        case .text2music:
            if let codes = r.audioCodes ?? overrides.codes, !codes.isEmpty {
                frames = codes.count * configuration.poolWindowSize
            } else {
                let seconds = resolved.duration ?? 120
                frames = Int((seconds * 10).rounded() / 10 * Double(Self.sampleRate)) / Self.hop
            }
        case .cover:
            guard let path = r.sourceAudio else { throw .invalidRequest("a cover needs source audio") }
            let wave = try Self.loadStereo48k(path)
            sourceWave = wave
            progress?(SongProgress(stage: .conditioning, fraction: 0))
            if let injected = overrides.sourceLatents { sourceLatents = injected } else { sourceLatents = try encode(wave, key: encodeKey) }
            frames = sourceLatents!.dim(1)
            resolved.duration = Double(wave[0].count) / Double(Self.sampleRate)
        case .repaint:
            guard let path = r.sourceAudio else { throw .invalidRequest("a repaint needs source audio") }
            let wave = try Self.loadStereo48k(path)
            let sourceSeconds = Double(wave[0].count) / Double(Self.sampleRate)
            let end = r.repaintEnd.flatMap { $0 < 0 ? nil : $0 } ?? sourceSeconds
            let left = max(0, -r.repaintStart), right = max(0, end - sourceSeconds)
            let leftFrames = Int(left * Double(Self.sampleRate)), rightFrames = Int(right * Double(Self.sampleRate))
            let padded = wave.map { Array(repeating: Float(0), count: leftFrames) + $0 + Array(repeating: 0, count: rightFrames) }
            sourceWave = padded
            repaintSpan = (r.repaintStart + left, end + left)
            progress?(SongProgress(stage: .conditioning, fraction: 0))
            if let injected = overrides.sourceLatents { sourceLatents = injected } else { sourceLatents = try encode(padded, key: encodeKey) }
            frames = sourceLatents!.dim(1)
            resolved.duration = sourceSeconds
        }
        let length = max(128, frames)

        // 3. Text and timbre conditions (shared by the batch).
        let conditioningStart = Date()
        let references: MLXArray
        if let injected = overrides.referenceLatents {
            references = injected.asType(dtype)
        } else if let reference = r.referenceAudio {
            let clip = Self.referenceClip(try Self.loadStereo48k(reference), seed: UInt64(first))
            references = try encode(clip, key: MLXRandom.key(UInt64(first) &+ 0x7EF))
        } else {
            references = silence(frames: configuration.timbreFixFrame).expandedDimensions(axis: 0)
        }
        let metas = Self.metas(
            bpm: resolved.bpm, timeSignature: resolved.timeSignature.map(String.init),
            key: resolved.key, duration: resolved.duration)
        let wantsCodes = r.task == .text2music && r.thinking && usesPlanner && r.audioCodes == nil
        let hasCodes = (r.audioCodes?.isEmpty == false) || overrides.codes != nil || wantsCodes
        let task: SongTask = (r.task == .text2music && hasCodes) ? .cover : r.task
        let instruction: String =
            switch (r.task, task) {
            case (.text2music, .cover): TaskInstruction.cover
            default:
                r.instruction
                    ?? (task == .cover
                        ? TaskInstruction.cover
                        : task == .repaint
                            ? TaskInstruction.repaint
                            : TaskInstruction.text2music)
            }
        let silenceSource = silence(frames: length).expandedDimensions(axis: 0)
        timings.conditioning += Date().timeIntervalSince(conditioningStart)

        var songs: [GeneratedSong] = []
        for (item, seed) in seeds.enumerated() {
            if cancelled() { throw .cancelled }
            var itemTimings = timings
            // 4. The planner's codes for this item.
            var codes = r.audioCodes ?? overrides.codes
            if codes == nil && wantsCodes {
                let codesStart = Date()
                let state = item == 0 ? randomState : MLXRandom.RandomState(seed: UInt64(seed))
                var block = meta ?? PlannedMetadata()
                if !r.planMetadata || planned?.1.isEmpty == true {
                    block = PlannedMetadata(
                        bpm: r.bpm, duration: r.duration.map { Int($0) }, keyscale: r.key,
                        timesignature: r.timeSignature.map(String.init))
                }
                let target = r.duration ?? meta?.duration.map(Double.init) ?? 120
                codes = try planner.codes(
                    caption: r.caption, lyrics: lyrics, reasoning: PlannerText.reasoningBlock(block),
                    duration: target, settings: Self.plannerSettings(r), random: state
                ) { done, total in
                    progress?(SongProgress(stage: .codes, fraction: Double(done) / Double(total), item: item))
                }
                itemTimings.codes = Date().timeIntervalSince(codesStart)
                guard let made = codes, !made.isEmpty else { throw .generationFailed("the planner wrote no audio codes") }
            }

            // 5. The DiT's source latents, context and conditions.
            let conditionStart = Date()
            var source: MLXArray
            var chunk = MLXArray.ones([1, length], dtype: dtype)
            var repaint: RepaintInputs?
            var clean: MLXArray?
            if let codes, !codes.isEmpty {
                var h = hints(codes: codes)
                if Profile.enabled {
                    MLX.eval(h)
                    Profile.log(String(format: "PROFILE cond hints %.2f s", Date().timeIntervalSince(conditionStart)))
                }
                if h.dim(1) < length { h = MLX.concatenated([h, silence(frames: length - h.dim(1)).expandedDimensions(axis: 0)], axis: 1) }
                source = h[0..., ..<length, 0...]
                clean = sourceLatents.map { padTo($0, length) }
            } else if let latents = sourceLatents {
                let padded = padTo(latents, length)
                clean = padded
                switch task {
                case .cover:
                    source = coverHints(padded, frames: length)
                case .repaint:
                    let span = repaintSpan!
                    let startFrame = min(max(0, Int(span.start * Double(Self.sampleRate)) / Self.hop), length - 1)
                    let endFrame = max(startFrame + 1, min(Int(span.end * Double(Self.sampleRate)) / Self.hop, length))
                    var mask = Array(repeating: false, count: length)
                    for i in startFrame..<endFrame { mask[i] = true }
                    let region = MLXArray(mask.map { $0 ? Int32(1) : 0 }).reshaped(1, length, 1) .> 0
                    source = MLX.where(region, silenceSource, padded)
                    let explicit = (r.chunkMaskMode ?? .explicit) == .explicit
                    if explicit { chunk = MLXArray(mask.map { $0 ? Float(1) : 0 }).reshaped(1, length).asType(dtype) }
                    let config = Self.repaintConfig(mode: r.repaintMode, strength: r.repaintStrength)
                    repaint = RepaintInputs(
                        mask: [mask], clean: padded,
                        crossfadeFrames: r.repaintCrossfadeFrames ?? config.frames,
                        injectionRatio: config.injection)
                case .text2music:
                    source = silenceSource
                }
            } else {
                source = silenceSource
            }
            let conditions = self.conditions(
                caption: ditCaption, instruction: instruction, metas: metas, lyrics: lyrics,
                language: r.language, references: references, referenceOrder: [0],
                source: source, chunkMask: chunk)
            var nonCover: DiffusionConditions?
            if task == .cover && r.coverStrength < 1 {
                nonCover = self.conditions(
                    caption: ditCaption, instruction: TaskInstruction.text2music, metas: metas,
                    lyrics: lyrics, language: r.language, references: references,
                    referenceOrder: [0], source: silenceSource, chunkMask: chunk)
            }
            MLX.eval(conditions.encoder, conditions.context)
            itemTimings.conditioning += Date().timeIntervalSince(conditionStart)

            // 6. Diffusion.
            let diffusionStart = Date()
            var settings = DiffusionSettings(
                seeds: [seed], retakeSeeds: retakes.map { [$0[item]] },
                retakeVariance: r.retakeVariance, shift: r.shift, timesteps: r.timesteps,
                steps: r.steps, guidance: r.guidance, guidanceStart: r.guidanceStart,
                guidanceEnd: r.guidanceEnd, coverStrength: task == .cover ? r.coverStrength : 1,
                method: r.method, sampler: r.sampler,
                velocityNormThreshold: r.velocityNormThreshold, velocityEMA: r.velocityEMA,
                dcw: r.dcw, dcwMode: r.dcwMode, dcwScaler: r.dcwScaler,
                dcwHighScaler: r.dcwHighScaler)
            settings.noise = item == 0 ? overrides.noise : nil
            if task == .cover, r.coverNoiseStrength > 0, let clean {
                settings.startFrom = (clean, r.coverNoiseStrength)
            }
            var latents = try DiffusionSampler.run(
                decoder: network.decoder, conditions: conditions, nonCover: nonCover,
                nullCondition: network.nullCondition, frames: length, settings: settings, repaint: repaint,
                progress: { done, total in
                    progress?(SongProgress(stage: .diffusion, fraction: Double(done) / Double(total), item: item))
                }, cancelled: cancelled)
            itemTimings.diffusion = Date().timeIntervalSince(diffusionStart)
            let finite = MLX.isNaN(latents).any().item(Bool.self) == false && MLX.isInf(latents).any().item(Bool.self) == false
            guard finite else { throw .generationFailed("the diffusion produced NaN or infinite latents") }
            if r.latentShift != 0 || r.latentRescale != 1 {
                latents = latents * Float(r.latentRescale) + Float(r.latentShift)
            }

            // 7. Decode and finish.
            let decodeStart = Date()
            var wave = try decode(latents) { fraction in
                progress?(SongProgress(stage: .decoding, fraction: fraction, item: item))
            }
            let peak = wave.map { $0.map(abs).max() ?? 0 }.max() ?? 0
            if peak > 1 { wave = wave.map { $0.map { $0 / peak } } }
            if let span = repaintSpan, r.repaintMode != .aggressive, let original = sourceWave {
                let config = Self.repaintConfig(mode: r.repaintMode, strength: r.repaintStrength)
                wave = Self.splice(
                    wave, source: original, start: span.start, end: span.end,
                    crossfade: r.repaintWaveCrossfade ?? config.wave)
            }
            let finishStart = Date()
            wave = Self.finish(wave, normalize: r.normalize, db: r.normalizationDB, fadeIn: r.fadeIn, fadeOut: r.fadeOut)
            if ProcessInfo.processInfo.environment["ACESTEP_PROFILE"] != nil {
                Profile.log(String(format: "PROFILE finish %.2f s", Date().timeIntervalSince(finishStart)))
            }
            itemTimings.decoding = Date().timeIntervalSince(decodeStart)
            itemTimings.total = Date().timeIntervalSince(started)
            var itemResolved = resolved
            itemResolved.seed = seed
            itemResolved.batchSize = 1
            if item == 0 && r.audioCodes == nil && overrides.codes == nil { itemResolved.audioCodes = nil }
            let plan = meta.map {
                SongPlan(
                    bpm: $0.bpm, caption: $0.caption, duration: $0.duration, key: $0.keyscale,
                    language: $0.language, timeSignature: $0.timesignature)
            }
            songs.append(
                GeneratedSong(
                    audio: SongAudio(channels: wave, sampleRate: Double(Self.sampleRate)),
                    resolved: itemResolved, plan: usesPlanner ? plan : nil, audioCodes: codes,
                    timings: itemTimings))
            MLX.Memory.clearCache()
        }
        return songs
    }

    // MARK: Helpers

    func validate(_ r: SongRequest) throws(ACEStepError) {
        if let bpm = r.bpm, !MetadataLimits.bpm.contains(bpm) { throw .invalidRequest("bpm is 30–300; not \(bpm)") }
        if let d = r.duration, !(10...600).contains(d) { throw .invalidRequest("duration is 10–600 seconds; not \(d)") }
        if let t = r.timeSignature, !MetadataLimits.timeSignatures.contains(t) {
            throw .invalidRequest("time signature is 2, 3, 4 or 6; not \(t)")
        }
        if let key = r.key, !key.isEmpty, KeyScale.normalised(key) == nil {
            throw .invalidRequest("key is a note A–G, optional # or b, then major or minor (\"F# minor\"); not \(key)")
        }
        if !MetadataLimits.languages.contains(r.language) {
            throw .invalidRequest("language is one of \(MetadataLimits.languages.joined(separator: ", ")); not \(r.language)")
        }
        if r.steps < 1 || r.steps > 200 { throw .invalidRequest("steps is 1–200; not \(r.steps)") }
        if r.batchSize < 1 || r.batchSize > 16 { throw .invalidRequest("batch size is 1–16; not \(r.batchSize)") }
        if r.dcwWavelet != "haar" {
            throw .invalidRequest("only the haar wavelet is implemented for DCW (upstream's MLX path is native only for haar)")
        }
        if r.adaptiveDualGuidance {
            throw .invalidRequest("adaptive dual guidance is for ACE-Step's base models; the turbo model does not support it")
        }
        if let codes = r.audioCodes, codes.contains(where: { !(0...MetadataLimits.maxAudioCode).contains($0) }) {
            throw .invalidRequest("audio codes are 0–\(MetadataLimits.maxAudioCode)")
        }
    }

    // `[1, frames, 64]`: trimmed, or padded with silence.
    func padTo(_ latents: MLXArray, _ frames: Int) -> MLXArray {
        if latents.dim(1) >= frames { return latents[0..., ..<frames, 0...] }
        return MLX.concatenated([latents, silence(frames: frames - latents.dim(1)).expandedDimensions(axis: 0)], axis: 1)
    }

    // A cover's hints: the source latents through the tokenizer and back.
    func coverHints(_ latents: MLXArray, frames: Int) -> MLXArray {
        let window = configuration.poolWindowSize
        var x = latents
        if x.dim(1) % window != 0 {
            let pad = window - x.dim(1) % window
            x = MLX.concatenated([x, silence[..<pad].expandedDimensions(axis: 0)], axis: 1)
        }
        let patches = x.reshaped(x.dim(0), x.dim(1) / window, window, x.dim(2))
        let (_, quantized) = network.tokenizer(patches)
        return network.detokenizer(quantized)[0..., ..<frames, 0...]
    }

    // Upstream's `_resolve_repaint_config`.
    static func repaintConfig(mode: RepaintMode, strength: Double) -> (injection: Double, frames: Int, wave: Double) {
        let s = min(max(strength, 0), 1)
        switch mode {
        case .aggressive: return (0, 0, 0)
        case .conservative: return (1, 25, 0.05)
        case .balanced:
            let inverse = 1 - s
            return (inverse, Int((25 * inverse).rounded(.toNearestOrEven)), 0.05 * inverse)
        }
    }

    // Upstream's `apply_repaint_waveform_splice`: the source outside the repaint, crossfaded at its edges.
    static func splice(_ generated: [[Float]], source: [[Float]], start: Double, end: Double, crossfade: Double) -> [[Float]] {
        let count = min(generated[0].count, source[0].count)
        let a = max(0, min(Int(start * Double(sampleRate)), count))
        let b = max(a, min(Int(end * Double(sampleRate)), count))
        if a == 0 && b >= count { return generated }
        var mask = [Float](repeating: 0, count: count)
        for i in a..<b { mask[i] = 1 }
        let fade = Int(crossfade * Double(sampleRate))
        if fade > 0 {
            let fs = max(a - fade, 0)
            for i in 0..<(a - fs) { mask[fs + i] = Float(i + 1) / Float(a - fs + 1) }
            let fe = min(b + fade, count)
            for i in 0..<(fe - b) { mask[b + i] = 1 - Float(i + 1) / Float(fe - b + 1) }
        }
        return generated.indices.map { c in
            var out = generated[c]
            for i in 0..<count { out[i] = mask[i] * generated[c][i] + (1 - mask[i]) * source[c][i] }
            return out
        }
    }
}
