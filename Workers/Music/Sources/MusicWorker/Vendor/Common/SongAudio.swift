//
//  SongAudio.swift
//  MusicGeneration
//
//  Created by David Sherlock on 2026.
//

import Foundation

// Rendered audio: planar float channels at one sample rate.
public struct SongAudio: Sendable, Equatable {
    // One array per channel, all the same length.
    public var channels: [[Float]]
    // Samples per second (ACE-Step renders 48,000).
    public var sampleRate: Double

    public init(channels: [[Float]], sampleRate: Double) {
        self.channels = channels
        self.sampleRate = sampleRate
    }

    public var frameCount: Int { channels.first?.count ?? 0 }
    public var seconds: Double { Double(frameCount) / sampleRate }

    // The largest absolute sample.
    public var peak: Float { channels.map { $0.map(abs).max() ?? 0 }.max() ?? 0 }
}

// One finished song and how it was made.
public struct GeneratedSong: Sendable {
    // The audio.
    public var audio: SongAudio
    // The request with every value the engine resolved: the seed used, and
    // what the planner chose for anything left open.
    public var resolved: SongRequest
    // What the planner wrote into its reasoning block, if it ran.
    public var plan: SongPlan?
    // The codes the audio was rendered from, if any: ACE-Step's 5 Hz codes, YuE2's 25 Hz song tokens.
    public var audioCodes: [Int]?
    // Seconds per stage.
    public var timings: SongTimings
    // YuE2: the song tokens hit their cap before the model ended the song.
    public var truncated: Bool

    public init(
        audio: SongAudio, resolved: SongRequest, plan: SongPlan?, audioCodes: [Int]?, timings: SongTimings,
        truncated: Bool = false
    ) {
        self.audio = audio
        self.resolved = resolved
        self.plan = plan
        self.audioCodes = audioCodes
        self.timings = timings
        self.truncated = truncated
    }
}

// What the planner decided: ACE-Step's reasoning block, or YuE2's score.
public struct SongPlan: Codable, Sendable, Equatable {
    public var bpm: Int?
    public var caption: String?
    public var duration: Int?
    public var key: String?
    public var language: String?
    public var timeSignature: String?
    // YuE2: the ABC score it planned (or was given).
    public var score: String?

    public init(
        bpm: Int? = nil, caption: String? = nil, duration: Int? = nil, key: String? = nil,
        language: String? = nil, timeSignature: String? = nil, score: String? = nil
    ) {
        self.score = score
        self.bpm = bpm
        self.caption = caption
        self.duration = duration
        self.key = key
        self.language = language
        self.timeSignature = timeSignature
    }

    enum CodingKeys: String, CodingKey {
        case bpm, caption, duration, key, language, score
        case timeSignature = "time_signature"
    }
}

// Where the time went.
public struct SongTimings: Codable, Sendable, Equatable {
    public var load: Double = 0
    public var planning: Double = 0
    public var codes: Double = 0
    public var conditioning: Double = 0
    public var diffusion: Double = 0
    public var decoding: Double = 0
    public var total: Double = 0

    public init() {}
}

// Where a generation is, for a progress bar.
public struct SongProgress: Sendable, Equatable {
    public enum Stage: String, Sendable { case loading, planning, codes, conditioning, diffusion, decoding }
    public var stage: Stage
    // 0–1 within the stage.
    public var fraction: Double
    // Which song of a batch, from 0.
    public var item: Int

    public init(stage: Stage, fraction: Double, item: Int = 0) {
        self.stage = stage
        self.fraction = fraction
        self.item = item
    }
}
