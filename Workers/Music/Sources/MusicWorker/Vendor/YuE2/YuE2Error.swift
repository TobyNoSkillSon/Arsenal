//
//  YuE2Error.swift
//  YuE2Music
//
//  Created by David Sherlock on 2026.
//

import Foundation

// Why YuE2 could not make a song.
public enum YuE2Error: Error, Sendable, Equatable, LocalizedError {
    // The engine was not handed ``YuE2Licence/version``.
    case licenceNotAccepted(expected: String)
    case modelMissing(String)
    case invalidRequest(String)
    case generationFailed(String)
    case insufficientMemory(physicalGB: Int, requiredGB: Int)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .licenceNotAccepted(let expected): "YuE2's weights need their licence accepted (\(expected))"
        case .modelMissing(let detail): "YuE2 model: \(detail)"
        case .invalidRequest(let detail): detail
        case .generationFailed(let detail): "generation failed: \(detail)"
        case .insufficientMemory(let have, let need): "YuE2 needs a Mac with at least \(need) GB of memory; this one has \(have) GB"
        case .cancelled: "cancelled"
        }
    }
}
