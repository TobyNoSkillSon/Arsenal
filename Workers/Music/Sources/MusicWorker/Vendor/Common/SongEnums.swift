//
//  SongEnums.swift
//  MusicGeneration
//
//  Created by David Sherlock on 2026.
//

import Foundation

// What the request asks of the model (upstream's `task_type`).
public enum SongTask: String, Codable, Sendable, CaseIterable {
    // A new song from words.
    case text2music
    // The source song re-made in the caption's style, keeping its structure.
    case cover
    // A stretch of the source redrawn (or, past its end, extended).
    case repaint
}

// `infer_method`.
public enum DiffusionMethod: String, Codable, Sendable, CaseIterable { case ode, sde }

// `sampler_mode`.
public enum SamplerKind: String, Codable, Sendable, CaseIterable { case euler, heun }

// `dcw_mode`.
public enum DCWMode: String, Codable, Sendable, CaseIterable { case low, high, double, pix }

// `chunk_mask_mode`.
public enum ChunkMaskMode: String, Codable, Sendable, CaseIterable { case auto, explicit }

// `repaint_mode`.
public enum RepaintMode: String, Codable, Sendable, CaseIterable { case conservative, balanced, aggressive }

// Which model makes the song.
public enum SongEngine: String, Codable, Sendable, CaseIterable {
    // ACE-Step 1.5 turbo (MIT) — the default.
    case aceStep = "ace-step"
    // YuE2-3B (weights CC BY-NC 4.0 + individual-creator permission) — opt-in behind its licence.
    case yue2
}

// YuE2's symbolic planning (`cot`).
public enum YuE2Planning: String, Codable, Sendable, CaseIterable {
    // A chord-annotated ABC score first (upstream's default).
    case full
    // A melody-only score (recommended for covers from a supplied score).
    case melody
    // No score.
    case off
}

// Which YuE2 decoder renders the latents.
public enum YuE2Decoder: String, Codable, Sendable, CaseIterable {
    // m-a-p/YuE2-Vae — better perceptual quality (upstream's default).
    case standard
    // m-a-p/YuE2-Vae-legacy — the decoder upstream's benchmarks used.
    case legacy
}
