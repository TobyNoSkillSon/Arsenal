// SPDX-License-Identifier: Apache-2.0
// Vendored from faraday/SpeechLens Sources/Inference (Apache-2.0)
// @ d20e11bcafbc32435ef18a17ed56caf22fca6833; Vireo changes listed in CHANGES.md.

import Foundation

/// Runtime settings for the SE-Mamba inference pipeline.
/// This is the single source of truth for chunking and overlap parameters.
public struct ReUseSettings: Sendable, Equatable {
    public static let standard = ReUseSettings(
        validatedChunkSeconds: 5.0,
        overlapPortion: 0.05
    )
    
    /// The length of audio chunks processed by the model.
    /// The five-second default balances throughput with bounded activation memory.
    public let chunkSeconds: Double
    
    /// The portion of the chunk that overlaps with the next/previous chunk.
    /// Used for Hann-windowed overlap-add to suppress warm-up transients.
    public let overlapPortion: Double
    
    public init(chunkSeconds: Double = 5.0, overlapPortion: Double = 0.05) throws {
        guard chunkSeconds.isFinite, chunkSeconds > 0 else {
            throw ReUseSettingsError.invalidChunkSeconds
        }
        guard overlapPortion.isFinite, (0.0...0.5).contains(overlapPortion) else {
            throw ReUseSettingsError.invalidOverlapPortion
        }
        self.chunkSeconds = chunkSeconds
        self.overlapPortion = overlapPortion
    }

    private init(validatedChunkSeconds: Double, overlapPortion: Double) {
        self.chunkSeconds = validatedChunkSeconds
        self.overlapPortion = overlapPortion
    }

    public func with(chunkSeconds: Double) throws -> Self {
        try Self(chunkSeconds: chunkSeconds, overlapPortion: overlapPortion)
    }

    public func with(overlapPortion: Double) throws -> Self {
        try Self(chunkSeconds: chunkSeconds, overlapPortion: overlapPortion)
    }
}

extension ReUseSettings: Codable {
    private enum CodingKeys: String, CodingKey {
        case chunkSeconds
        case overlapPortion
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let chunkSeconds = try container.decode(Double.self, forKey: .chunkSeconds)
        let overlapPortion = try container.decode(Double.self, forKey: .overlapPortion)
        try self.init(chunkSeconds: chunkSeconds, overlapPortion: overlapPortion)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(chunkSeconds, forKey: .chunkSeconds)
        try container.encode(overlapPortion, forKey: .overlapPortion)
    }
}

public enum ReUseSettingsError: Error, LocalizedError, Sendable, Equatable {
    case invalidChunkSeconds
    case invalidOverlapPortion

    public var errorDescription: String? {
        switch self {
        case .invalidChunkSeconds:
            return "chunkSeconds must be finite and greater than 0."
        case .invalidOverlapPortion:
            return "overlapPortion must be finite and in the range 0.0...0.5."
        }
    }
}
