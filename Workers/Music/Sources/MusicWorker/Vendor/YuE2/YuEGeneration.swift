import Foundation
import MLX
import MLXRandom

enum YuEGeneration {
    static func run(
        _ request: SongRequest, options o: YuE2Options, pipeline p: YuE2Pipeline, item: Int,
        cancelled: () -> Bool, progress: (@Sendable (SongProgress) -> Void)?
    ) throws(YuE2Error) -> GeneratedSong {
        let abc = YuE2Prompt.abcSettings(o)
        let semantic = YuE2Prompt.semanticSettings(o, duration: request.duration)
        try YuE2Prompt.validate(o, [abc, semantic])
        let style = YuE2Prompt.style(request), lyrics = YuE2Prompt.lyrics(request)
        let seed = UInt64(request.seed ?? 0)
        var timings = SongTimings()
        let start = Date()

        // 1. The score.
        var scoreIDs: [Int] = []
        var score: String?
        if o.planning != .off {
            if let given = o.score {
                scoreIDs = p.tokenizer.encode(given)
                score = given
            } else {
                let planned = try plan(
                    style: style, lyrics: lyrics, options: o, settings: abc, seed: seed, pipeline: p, item: item,
                    cancelled: cancelled, progress: progress)
                scoreIDs = planned.ids
                score = p.tokenizer.decode(planned.ids)
            }
        }
        timings.planning = Date().timeIntervalSince(start)

        // 2. The song tokens.
        let codesStart = Date()
        let prefix = YuE2Protocol.prefix(
            style: style, lyrics: lyrics, cot: o.planning,
            abcIDs: o.planning == .off ? nil : scoreIDs, tokenizer: p.tokenizer)
        let guidance = o.effectiveGuidance
        let negative = guidance != 1 ? YuE2Protocol.negative(cot: o.planning, abcIDs: scoreIDs, tokenizer: p.tokenizer) : nil
        var count = 0
        let tokens = try p.generate(
            prefix: prefix, negative: negative, guidance: Float(guidance), settings: semantic,
            seed: seed, phase: .semantic, legacy: o.planning == .off, cancelled: cancelled
        ) { _ in
            count += 1
            progress?(SongProgress(stage: .codes, fraction: min(1, Double(count) / Double(semantic.maxTokens)), item: item))
        }
        let codec = tokens.ids.map { $0 - YuE2Protocol.codecOffset }
        guard !codec.isEmpty else { throw .generationFailed("the model wrote no song tokens") }
        timings.codes = Date().timeIntervalSince(codesStart)

        // 3. Acoustic latents (one noise draw for the whole song, as upstream).
        let narStart = Date()
        let noise = MLXRandom.normal([codec.count, YuE2Pipeline.latentChannels], key: MLXRandom.key(seed))
        let latents = try p.synthesize(
            prefix: prefix, codec: codec, noise: noise, steps: o.odeSteps,
            cancelled: cancelled
        ) { fraction in
            progress?(SongProgress(stage: .diffusion, fraction: fraction, item: item))
        }
        timings.diffusion = Date().timeIntervalSince(narStart)

        // 4. Audio.
        let decodeStart = Date()
        var channels = try p.decode(latents, decoder: o.decoder, core: o.decoderTileFrames, cancelled: cancelled) { fraction in
            progress?(SongProgress(stage: .decoding, fraction: fraction, item: item))
        }
        channels = SongFinishing.finish(
            channels, sampleRate: 48_000, normalize: request.normalize, db: request.normalizationDB,
            fadeIn: request.fadeIn, fadeOut: request.fadeOut)
        timings.decoding = Date().timeIntervalSince(decodeStart)
        timings.total = Date().timeIntervalSince(start)

        var resolved = request
        resolved.engine = .yue2
        resolved.yue = o
        return GeneratedSong(
            audio: SongAudio(channels: channels, sampleRate: 48_000), resolved: resolved,
            plan: score.map { SongPlan(score: $0) }, audioCodes: codec, timings: timings,
            truncated: tokens.truncated)
    }

    // The score alone.
    static func score(
        _ request: SongRequest, options o: YuE2Options, pipeline p: YuE2Pipeline, item: Int,
        cancelled: () -> Bool, progress: (@Sendable (SongProgress) -> Void)?
    ) throws(YuE2Error) -> (score: String, truncated: Bool) {
        let abc = YuE2Prompt.abcSettings(o)
        try YuE2Prompt.validate(o, [abc])
        if let given = o.score { return (given, false) }
        let planned = try plan(
            style: YuE2Prompt.style(request), lyrics: YuE2Prompt.lyrics(request), options: o, settings: abc,
            seed: UInt64(request.seed ?? UInt32.random(in: 0...UInt32.max)), pipeline: p, item: item, cancelled: cancelled, progress: progress)
        return (p.tokenizer.decode(planned.ids), planned.truncated)
    }

    static func plan(
        style: String, lyrics: String, options o: YuE2Options, settings: YuE2SamplingSettings, seed: UInt64,
        pipeline p: YuE2Pipeline, item: Int, cancelled: () -> Bool,
        progress: (@Sendable (SongProgress) -> Void)?
    ) throws(YuE2Error) -> YuE2Pipeline.Tokens {
        let prefix = YuE2Protocol.prefix(style: style, lyrics: lyrics, cot: o.planning, abcIDs: nil, tokenizer: p.tokenizer)
        var count = 0
        return try p.generate(
            prefix: prefix, negative: nil, guidance: 1, settings: settings, seed: seed, phase: .abc,
            legacy: false, cancelled: cancelled
        ) { _ in
            count += 1
            progress?(SongProgress(stage: .planning, fraction: min(1, Double(count) / Double(settings.maxTokens)), item: item))
        }
    }
}
