//
//  SongFinishing.swift
//  MusicGeneration
//
//  Created by David Sherlock on 2026.
//
//  The last touch both engines share (ACE-Step's `normalize_audio` and
//  fades, applied to YuE2's output the same way): a peak above full scale
//  brought back to it, an optional peak normalisation, linear fades.
//

import Foundation

public enum SongFinishing {

    // Peak above 1 scaled to 1; then peak-normalised to `db` when asked; then fades (torch.linspace ramps).
    public static func finish(
        _ channels: [[Float]], sampleRate: Double, normalize: Bool, db: Double, fadeIn: Double,
        fadeOut: Double
    ) -> [[Float]] {
        var channels = channels
        var peak = channels.map { $0.map(abs).max() ?? 0 }.max() ?? 0
        if peak > 1 {
            channels = channels.map { $0.map { $0 / peak } }
            peak = 1
        }
        if normalize && db <= 0 && peak > 1e-6 {
            let gain = Float(pow(10, db / 20)) / peak
            channels = channels.map { $0.map { $0 * gain } }
        }
        let count = channels.first?.count ?? 0
        let inFrames = min(count, Int(fadeIn * sampleRate))
        let outFrames = min(count, Int(fadeOut * sampleRate))
        func ramp(_ i: Int, _ n: Int) -> Float { n == 1 ? 0 : Float(i) / Float(n - 1) }
        for c in channels.indices {
            for i in 0..<inFrames { channels[c][i] *= ramp(i, inFrames) }
            for i in 0..<outFrames { channels[c][count - outFrames + i] *= 1 - ramp(i, outFrames) }
        }
        return channels
    }
}
