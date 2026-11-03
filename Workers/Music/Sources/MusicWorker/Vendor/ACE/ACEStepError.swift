//
//  ACEStepError.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//

import Foundation

// Why ACE-Step could not make a song.
public enum ACEStepError: Error, Sendable, Equatable, LocalizedError {
    // A model file is missing or unreadable.
    case modelMissing(String)
    // The request asks for something this engine cannot do.
    case invalidRequest(String)
    // An input audio file could not be read.
    case unreadableAudio(String)
    // The model produced something unusable (NaN latents, no codes).
    case generationFailed(String)
    // The Mac has too little memory.
    case insufficientMemory(physicalGB: Int, requiredGB: Int)
    // Cancelled by the caller.
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .modelMissing(let detail): "ACE-Step model: \(detail)"
        case .invalidRequest(let detail): detail
        case .unreadableAudio(let detail): "audio input: \(detail)"
        case .generationFailed(let detail): "generation failed: \(detail)"
        case .insufficientMemory(let have, let need):
            "ACE-Step needs a Mac with at least \(need) GB of memory at this precision; this one has \(have) GB"
                + (need > 16 ? " — the lite tier (--tier lite) runs on 16 GB" : "")
        case .cancelled: "cancelled"
        }
    }
}
