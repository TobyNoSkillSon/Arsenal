//
//  OobleckConfiguration.swift
//  MusicNetworks
//
//  Created by David Sherlock on 2026.
//

import Foundation

// The Oobleck VAE's vae/config.json.
public struct OobleckConfiguration: Decodable, Sendable, Equatable {
    public var audioChannels: Int
    public var channelMultiples: [Int]
    public var decoderChannels: Int
    public var decoderInputChannels: Int
    public var downsamplingRatios: [Int]
    public var encoderHiddenSize: Int
    public var samplingRate: Int

    enum CodingKeys: String, CodingKey {
        case audioChannels = "audio_channels"
        case channelMultiples = "channel_multiples"
        case decoderChannels = "decoder_channels"
        case decoderInputChannels = "decoder_input_channels"
        case downsamplingRatios = "downsampling_ratios"
        case encoderHiddenSize = "encoder_hidden_size"
        case samplingRate = "sampling_rate"
    }

    // Audio samples per latent frame: 1920 at 48 kHz, 25 latent frames a second.
    public var hopLength: Int { downsamplingRatios.reduce(1, *) }

    public init(
        audioChannels: Int, channelMultiples: [Int], decoderChannels: Int, decoderInputChannels: Int,
        downsamplingRatios: [Int], encoderHiddenSize: Int, samplingRate: Int
    ) {
        self.audioChannels = audioChannels
        self.channelMultiples = channelMultiples
        self.decoderChannels = decoderChannels
        self.decoderInputChannels = decoderInputChannels
        self.downsamplingRatios = downsamplingRatios
        self.encoderHiddenSize = encoderHiddenSize
        self.samplingRate = samplingRate
    }
}
