//
//  KeyScale.swift
//  MusicGeneration
//
//  Created by David Sherlock on 2026.
//
//  Keys as ACE-Step writes them: a note A–G, an optional sharp or flat
//  (`#`, `b`, `♯`, `♭`), a space, `major` or `minor`. People type `Am`,
//  `f# min`, `Eb Major` — all read here into that one form. Pure.
//

import Foundation

public enum KeyScale {

    // `"F# minor"` for `F#m`, `f# min`, `F♯ Minor`; nil when it is not a key.
    public static func normalised(_ text: String) -> String? {
        var s = text.trimmingCharacters(in: .whitespaces)
        guard let first = s.first?.uppercased(), "ABCDEFG".contains(first) else { return nil }
        s.removeFirst()
        var accidental = ""
        if let next = s.first, "#b♯♭".contains(next) {
            accidental = String(next)
            s.removeFirst()
        }
        let mode = s.trimmingCharacters(in: .whitespaces).lowercased()
        switch mode {
        case "", "maj", "major", "M": return "\(first)\(accidental) major"
        case "m", "min", "minor": return "\(first)\(accidental) minor"
        default: return nil
        }
    }
}
