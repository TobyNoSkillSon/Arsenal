//
//  YuE2Options.swift
//  MusicGeneration
//
//  Created by David Sherlock on 2026.
//
//  YuE2's own knobs (yue2_infer 0.1.5: `SongRequest`, `GenerationConfig`,
//  the two `Sampling` stages), with upstream's defaults. Shared ideas —
//  caption (YuE2's style tags), lyrics, seed, batch, duration — stay on
//  ``SongRequest``.
//

import Foundation

// YuE2-specific settings, under `"yue"` in a request.
public struct YuE2Options: Codable, Sendable, Equatable {
    // `cot`.
    public var planning: YuE2Planning = .full
    // `abc` — an ABC score to render instead of planning one (needs full or melody planning).
    public var score: String?
    // `cfg_scale` on the semantic tokens; nil is upstream's (1.0, or 1.01 without a score).
    public var guidance: Double?
    // `GenerationConfig.abc` — the score's sampling.
    public var abcTemperature = 0.7
    public var abcTopP = 0.9
    public var abcTopK = 30
    public var abcRepetitionPenalty = 1.005
    public var abcPenaltyWindow = 100
    public var abcMinTokens = 32
    public var abcMaxTokens = 4096
    // `GenerationConfig.semantic` — the 25 Hz song tokens' sampling.
    public var semanticTemperature = 1.0
    public var semanticTopP = 0.95
    public var semanticTopK = 100
    public var semanticRepetitionPenalty = 1.2
    public var semanticPenaltyWindow = 50
    public var semanticMinTokens = 200
    public var semanticMaxTokens = 9000
    // `ode_steps` — midpoint steps of the acoustic flow matching.
    public var odeSteps = 32
    // `vae` — the decoder.
    public var decoder: YuE2Decoder = .standard
    // `vae_core_frames` — latent frames per decoder tile (16 frames of context either side).
    public var decoderTileFrames = 1024

    public init() {}

    enum CodingKeys: String, CodingKey, CaseIterable {
        case planning, score, guidance
        case abcTemperature = "abc_temperature", abcTopP = "abc_top_p", abcTopK = "abc_top_k"
        case abcRepetitionPenalty = "abc_repetition_penalty", abcPenaltyWindow = "abc_penalty_window"
        case abcMinTokens = "abc_min_tokens", abcMaxTokens = "abc_max_tokens"
        case semanticTemperature = "semantic_temperature", semanticTopP = "semantic_top_p", semanticTopK = "semantic_top_k"
        case semanticRepetitionPenalty = "semantic_repetition_penalty", semanticPenaltyWindow = "semantic_penalty_window"
        case semanticMinTokens = "semantic_min_tokens", semanticMaxTokens = "semantic_max_tokens"
        case odeSteps = "ode_steps", decoder, decoderTileFrames = "decoder_tile_frames"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var o = YuE2Options()
        func v<T: Decodable>(_ key: CodingKeys, _ current: T) throws -> T { try c.decodeIfPresent(T.self, forKey: key) ?? current }
        o.planning = try v(.planning, o.planning)
        o.score = try c.decodeIfPresent(String.self, forKey: .score)
        o.guidance = try c.decodeIfPresent(Double.self, forKey: .guidance)
        o.abcTemperature = try v(.abcTemperature, o.abcTemperature)
        o.abcTopP = try v(.abcTopP, o.abcTopP)
        o.abcTopK = try v(.abcTopK, o.abcTopK)
        o.abcRepetitionPenalty = try v(.abcRepetitionPenalty, o.abcRepetitionPenalty)
        o.abcPenaltyWindow = try v(.abcPenaltyWindow, o.abcPenaltyWindow)
        o.abcMinTokens = try v(.abcMinTokens, o.abcMinTokens)
        o.abcMaxTokens = try v(.abcMaxTokens, o.abcMaxTokens)
        o.semanticTemperature = try v(.semanticTemperature, o.semanticTemperature)
        o.semanticTopP = try v(.semanticTopP, o.semanticTopP)
        o.semanticTopK = try v(.semanticTopK, o.semanticTopK)
        o.semanticRepetitionPenalty = try v(.semanticRepetitionPenalty, o.semanticRepetitionPenalty)
        o.semanticPenaltyWindow = try v(.semanticPenaltyWindow, o.semanticPenaltyWindow)
        o.semanticMinTokens = try v(.semanticMinTokens, o.semanticMinTokens)
        o.semanticMaxTokens = try v(.semanticMaxTokens, o.semanticMaxTokens)
        o.odeSteps = try v(.odeSteps, o.odeSteps)
        o.decoder = try v(.decoder, o.decoder)
        o.decoderTileFrames = try v(.decoderTileFrames, o.decoderTileFrames)
        self = o
    }

    // Upstream's guidance when none is given.
    public var effectiveGuidance: Double { guidance ?? (planning == .off ? 1.01 : 1.0) }

    public static var fieldNames: [String] { CodingKeys.allCases.map(\.rawValue) }
}
