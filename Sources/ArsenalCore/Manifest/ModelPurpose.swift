// SPDX-License-Identifier: MIT
import Foundation

/// What a model does, in a few plain words, for the line under its name in the Models table and Get more models
/// ("Image generation", "Background removal", "Speech, voice cloning"). A folder declares it as `purpose`; a folder
/// without one (a user's own) gets it from its kind and the operations it declares.
public enum ModelPurpose {
    public static let maxLength = 40

    /// The folder's declared purpose, else the derived one.
    public static func text(_ m: ModelManifest, kind: KindSpec?) -> String {
        if let declared = m.purpose?.trimmingCharacters(in: .whitespacesAndNewlines), !declared.isEmpty { return declared }
        let ops = m.operations.isEmpty ? kind?.op ?? [] : m.operations
        return derive(kind: m.kind, operations: ops.map(\.name), kindTitle: kind?.title)
    }

    /// From the kind and operation names: speech with what else it does with voices, sound and music, image generation
    /// or background removal, decisions, chat, audio cleanup; else the first operation's name. Empty when that would only
    /// repeat the kind's group header.
    public static func derive(kind: String, operations: [String], kindTitle: String? = nil) -> String {
        let text = guess(kind: kind, operations: operations)
        // The group header already says it ("Decisions" under Decisions): nothing rather than a repeat.
        let header = kindTitle ?? titles[kind] ?? ""
        return text.caseInsensitiveCompare(header) == .orderedSame ? "" : text
    }
    /// The built-in kinds' group titles (kinds/*.toml).
    static let titles = [
        "decisions": "Decisions", "chat": "Chat", "images": "Images", "voices": "Voices", "sound": "Sound", "enhancement": "Enhancement", "video": "Video"
    ]

    private static func guess(kind: String, operations: [String]) -> String {
        let ops = Set(operations)
        var parts: [String] = []
        if ops.contains("speak") {
            let voice = [ops.contains("design_voice") ? "design" : nil, ops.contains("clone_voice") ? "cloning" : nil].compactMap { $0 }
            return voice.isEmpty ? "Speech" : "Speech, voice " + voice.joined(separator: " and ")
        }
        if ops.contains("generate_sound") && ops.contains("generate_music") { return "Sound and music generation" }
        if ops.contains("generate_sound") { return "Sound generation" }
        if ops.contains("generate_music") { return "Music generation" }
        if ops.contains("generate") && (kind == "images" || kind.hasSuffix("image")) { parts.append("Image generation") }
        if ops.contains("remove_background") { parts.append(parts.isEmpty ? "Background removal" : "background removal") }
        if !parts.isEmpty { return parts.joined(separator: ", ") }
        if ops.contains("decide") || ops.contains("judge") || kind == "decisions" { return "Decisions" }
        if ops.contains("chat") || kind == "chat" { return "Chat" }
        if ops.contains("enhance") || kind == "enhancement" { return "Audio cleanup" }
        switch kind {
        case "images": return "Images"
        case "voices": return "Speech"
        case "sound": return "Sound generation"
        case "video": return "Video"
        default: break
        }
        guard let first = operations.first else { return "" }
        let words = first.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}
