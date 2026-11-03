//
//  ACEStepConfiguration.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The DiT checkpoint's config.json (acestep-v15-turbo). Field names and
//  defaults follow upstream's AceStepConfig; adapted from mere.run (MIT,
//  see THIRD_PARTY_NOTICES.md).
//

import Foundation

// The shape of ACE-Step 1.5's diffusion model, its condition encoder,
// audio tokenizer and detokenizer.
public struct ACEStepConfiguration: Decodable, Sendable, Equatable {
    public var fsqInputLevels: [Int]
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var attentionBias: Bool
    public var useSlidingWindow: Bool
    public var slidingWindow: Int
    public var layerTypes: [String]
    public var audioAcousticHiddenDim: Int
    public var poolWindowSize: Int
    public var textHiddenDim: Int
    public var inChannels: Int
    public var patchSize: Int
    public var numLyricEncoderHiddenLayers: Int
    public var numAttentionPoolerHiddenLayers: Int
    public var timbreHiddenDim: Int
    public var numTimbreEncoderHiddenLayers: Int
    public var timbreFixFrame: Int
    public var isTurbo: Bool

    enum CodingKeys: String, CodingKey {
        case fsqInputLevels = "fsq_input_levels"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case useSlidingWindow = "use_sliding_window"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case audioAcousticHiddenDim = "audio_acoustic_hidden_dim"
        case poolWindowSize = "pool_window_size"
        case textHiddenDim = "text_hidden_dim"
        case inChannels = "in_channels"
        case patchSize = "patch_size"
        case numLyricEncoderHiddenLayers = "num_lyric_encoder_hidden_layers"
        case numAttentionPoolerHiddenLayers = "num_attention_pooler_hidden_layers"
        case timbreHiddenDim = "timbre_hidden_dim"
        case numTimbreEncoderHiddenLayers = "num_timbre_encoder_hidden_layers"
        case timbreFixFrame = "timbre_fix_frame"
        case isTurbo = "is_turbo"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        fsqInputLevels = try c.decodeIfPresent([Int].self, forKey: .fsqInputLevels) ?? [8, 8, 8, 5, 5, 5]
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        useSlidingWindow = try c.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? true
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        let layers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0..<layers).map { ($0 + 1) % 2 == 1 ? "sliding_attention" : "full_attention" }
        audioAcousticHiddenDim = try c.decodeIfPresent(Int.self, forKey: .audioAcousticHiddenDim) ?? 64
        poolWindowSize = try c.decodeIfPresent(Int.self, forKey: .poolWindowSize) ?? 5
        textHiddenDim = try c.decodeIfPresent(Int.self, forKey: .textHiddenDim) ?? 1024
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 192
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 2
        numLyricEncoderHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numLyricEncoderHiddenLayers) ?? 8
        numAttentionPoolerHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numAttentionPoolerHiddenLayers) ?? 2
        timbreHiddenDim = try c.decodeIfPresent(Int.self, forKey: .timbreHiddenDim) ?? 64
        numTimbreEncoderHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numTimbreEncoderHiddenLayers) ?? 4
        timbreFixFrame = try c.decodeIfPresent(Int.self, forKey: .timbreFixFrame) ?? 750
        isTurbo = try c.decodeIfPresent(Bool.self, forKey: .isTurbo) ?? false
    }

    // The attention kind of encoder/decoder layer `index`.
    func isSliding(layer index: Int) -> Bool {
        useSlidingWindow && index < layerTypes.count && layerTypes[index] == "sliding_attention"
    }
}
