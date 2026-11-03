// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// A folder's `[[setting]]` declaration as the settings menu reads it (CONTRACTS §15), and the values it shows. Plain values,
// no AppKit: unit-tested directly.

extension SettingState {
    var type: String { declaration["type"]?.string ?? "text" }
    var title: String { declaration["title"]?.string ?? id }
    /// "load" (row turns red + Reload until applied) or "request" (the next request).
    var applies: String { declaration["applies"]?.string ?? "request" }
    var defaultValue: JSONValue { declaration["default"] ?? .null }
    var choices: [JSONValue] { declaration["choices"]?.array ?? [] }
    var minValue: Double? { declaration["min"]?.number }
    var maxValue: Double? { declaration["max"]?.number }
    var step: Double? { declaration["step"]?.number }
    /// The declared unit; a context length reads in tokens until its folder says so (`32K`, not `32768`).
    var unit: String? { declaration["unit"]?.string ?? (id.contains("context") ? "tokens" : nil) }
    /// The label's and the control's tooltip: the folder's sentence, then when it applies.
    var tooltip: String {
        // A switch for the app (Keep loaded, Show in menu) applies as it moves: nothing to add.
        let applies = self.applies == "load" ? "Applies after Reload." : self.applies == "app" ? "" : "Applies to the next request."
        guard let help = declaration["help"]?.string, !help.isEmpty else { return applies }
        return applies.isEmpty ? help : help + " " + applies
    }

    /// How a value reads in a control and in the red row's tooltip: `32K` for a token count, `1.25` for a number.
    func display(_ value: JSONValue) -> String {
        switch value {
        case .bool(let b): return b ? "On" : "Off"
        case .string(let s): return declaration["labels"]?[s]?.string ?? s
        case .number(let n):
            if unit == "tokens", n >= 1024, n.truncatingRemainder(dividingBy: 1024) == 0 { return "\(Int(n / 1024))K" }
            // A whole value of a fractional setting keeps one decimal ("1.0" beside "0.6"), so the column reads alike.
            if n == n.rounded(), abs(n) < 1e15 { return (step ?? 1) < 1 ? String(format: "%.1f", n) : String(Int(n)) }
            let decimals = step.map { max(0, min(4, Int((-log10($0)).rounded(.up)))) } ?? 2
            return String(format: "%.\(decimals)f", n)
        case .null: return ""
        default: return value.string ?? ""
        }
    }
    /// Which control the settings menu draws (spec §3.2): a long choice list is a pop-up menu (`list`).
    enum Control: Equatable { case toggle, segments, list, slider, number, text, editor, file, voice }
    var control: Control {
        switch type {
        case "toggle": return .toggle
        case "choice": return choices.count <= 3 && choices.allSatisfy { display($0).count <= 6 } ? .segments : .list
        case "slider": return .slider
        case "number": return .number
        case "voice": return .voice
        default: return .text
        }
    }
    /// A typed number snapped to the step and clamped to min/max; nil when it is not a number.
    func number(from text: String) -> Double? {
        let raw = text.trimmingCharacters(in: .whitespaces)
        let kilo = unit == "tokens" && raw.uppercased().hasSuffix("K")
        guard var n = Double(kilo ? String(raw.dropLast()) : raw), n.isFinite else { return nil }
        if kilo { n *= 1024 }
        if let step, step > 0 { let base = minValue ?? 0; n = base + ((n - base) / step).rounded() * step }
        if let minValue { n = max(minValue, n) }
        if let maxValue { n = min(maxValue, n) }
        // Steps such as 0.05 leave binary noise; the service checks the step exactly.
        return (n * 1e6).rounded() / 1e6
    }
}

/// One entry of a model's `voices` op.
struct VoiceChoice: Equatable, Identifiable {
    var id: String
    var name: String
    var language: String
    var title: String { language.isEmpty ? name : "\(name) \u{00b7} \(languageName)" }
    /// "en-US" → "American English", "en-GB" → "British English", else the language's English name.
    var languageName: String { Self.languageName(language) }
    /// Kokoro's voice ids carry the voice's sex second ("af_heart", "am_adam"): the voice menu separates the two.
    var group: Character? { id.count > 3 && id.dropFirst(2).first == "_" ? id.dropFirst().first : nil }
    static func languageName(_ code: String) -> String {
        switch code.lowercased() {
        case "en-us": return "American English"
        case "en-gb": return "British English"
        case "": return ""
        default: return Locale(identifier: "en").localizedString(forIdentifier: code) ?? code
        }
    }
    /// The voice pop-up's menu: names only, with a language heading over each group when the voices span more than one
    /// language (one language says nothing that tells voices apart).
    enum Row: Equatable { case heading(String), voice(VoiceChoice) }
    static func rows(_ voices: [VoiceChoice]) -> [Row] {
        var order: [String] = [], groups: [String: [VoiceChoice]] = [:]
        for v in voices {
            if groups[v.languageName] == nil { order.append(v.languageName) }
            groups[v.languageName, default: []].append(v)
        }
        guard order.count > 1 else { return voices.map { .voice($0) } }
        return order.flatMap { language in [Row.heading(language.isEmpty ? "Other" : language)] + (groups[language] ?? []).map { .voice($0) } }
    }
    /// The voice row's title, loaded or not: the voice's name ("Heart"), or Default when none is set.
    static func title(_ id: String, voices: [VoiceChoice]) -> String {
        if id.isEmpty { return "Default" }
        return voices.first { $0.id == id }?.name ?? displayName(id)
    }
    static func list(_ result: JSONObject) -> [VoiceChoice] {
        (result["voices"]?.array ?? []).compactMap { v in
            guard let id = v["id"]?.string ?? v["voice_id"]?.string else { return nil }
            return VoiceChoice(id: id, name: v["name"]?.string.flatMap { $0 == id ? nil : $0 } ?? displayName(id), language: v["language"]?.string ?? "")
        }
    }
    /// A preset id used as its own name reads as a name: Kokoro's "af_heart" (accent, sex, name) → "Heart".
    static func displayName(_ id: String) -> String {
        guard let match = id.wholeMatch(of: /[a-z]{2}_([a-z][a-z0-9]*)/) else { return id }
        return match.1.prefix(1).uppercased() + match.1.dropFirst()
    }
}

/// The Remove alert's answer for a row whose weights are on this Mac.
enum RemoveChoice { case cancel, keepWeights, deleteWeights }
