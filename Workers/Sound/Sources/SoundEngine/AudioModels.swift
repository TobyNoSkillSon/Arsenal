// SPDX-License-Identifier: MIT
import Foundation
import VireoCore
public struct SoundRequest: Sendable, Equatable {
    /// `.sfx` or `.music`.
    public var kind: AudioKind
    public var prompt: String
    public var negativePrompt: String?
    public var seconds: Double
    public var seed: UInt64?
    /// Sampler steps and guidance; nil = the model's recommended default.
    public var steps: Int?
    public var guidance: Double?
    public init(kind: AudioKind, prompt: String, negativePrompt: String? = nil, seconds: Double, seed: UInt64? = nil, steps: Int? = nil, guidance: Double? = nil) {
        self.kind = kind; self.prompt = prompt; self.negativePrompt = negativePrompt; self.seconds = seconds
        self.seed = seed; self.steps = steps; self.guidance = guidance
    }
}

public protocol SoundModel: AudioModel {
    var maxSeconds: Double { get }
    /// One whole clip (stereo allowed). Report progress per sampler step.
    func generate(_ request: SoundRequest, control: GenerationControl) async throws -> AudioClip
}

public struct EnhanceRequest: Sendable, Equatable {
    /// "denoise" (default), "restore", "upscale"; a model rejects modes it cannot do.
    public var mode: String
    public var options: [String: String]
    public init(mode: String = "denoise", options: [String: String] = [:]) { self.mode = mode; self.options = options }
}

public protocol EnhanceModel: AudioModel {
    /// Input at any rate and channel count; the model resamples internally as it needs.
    /// Returns audio at the input's rate unless the mode is "upscale".
    func enhance(_ input: AudioClip, request: EnhanceRequest, control: GenerationControl) async throws -> AudioClip
}

/// One time range, in seconds of the input, where the described sound is audible (a positive "+" anchor).
public struct IsolationSpan: Sendable, Equatable {
    public var start: Double
    public var end: Double
    public init(start: Double, end: Double) { self.start = start; self.end = end }
}

