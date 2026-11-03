// SPDX-License-Identifier: MIT
import Foundation

/// The three audio tasks of the main menu, each its own window over the same sound kit: speech from any voice model,
/// songs, and sound effects. A model appears in every task one of its operations serves.
enum SoundTask: String, CaseIterable {
    case speak, music, sound

    /// The operations the task's window offers, its everyday one first.
    var operations: [String] {
        switch self {
        case .speak: return ["speak", "clone_voice", "design_voice"]
        case .music: return ["generate_music"]
        case .sound: return ["generate_sound"]
        }
    }
    /// The operation that places a model in this task (a model that only clones voices is not a Speak model).
    var placing: String {
        switch self {
        case .speak: return "speak"
        case .music: return "generate_music"
        case .sound: return "generate_sound"
        }
    }
    var windowTitle: String {
        switch self {
        case .speak: return "Speak"
        case .music: return "Make Music"
        case .sound: return "Make a Sound"
        }
    }
    var menuTitle: String { windowTitle + "\u{2026}" }
    /// The run button: the task's verb.
    var runTitle: String {
        switch self {
        case .speak: return "Speak"
        case .music: return "Make Music"
        case .sound: return "Make Sound"
        }
    }
    var symbol: String {
        switch self {
        case .speak: return "person.wave.2"
        case .music: return "music.note"
        case .sound: return "waveform"
        }
    }
}
