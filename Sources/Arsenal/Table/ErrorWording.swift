// SPDX-License-Identifier: MIT
import Foundation

/// Failures in the user's words. A raw system message ("The file “diarize_session.input.json” couldn’t be opened because
/// there is no such file.") never takes a visible line: the line says what happened in plain words, attached to the
/// model it belongs to, and the raw text goes to the line's hover. Arsenal's own messages pass through unchanged.
enum ErrorWording {
    /// Plain words for a raw file or system message; nil when the message is not one (Arsenal's own wording).
    static func plain(_ raw: String) -> String? {
        let m = raw.replacingOccurrences(of: "\u{2019}", with: "'").lowercased()
        func has(_ parts: String...) -> Bool { parts.contains { m.contains($0) } }
        if has("no space left", "not enough space", "volume is full", "disk is full", "out of space") { return "the disk is full" }
        if has("don't have permission", "permission denied", "operation not permitted") { return "Arsenal is not allowed to open a file it needs" }
        if has("there is no such file", "no such file or directory", "doesn't exist", "does not exist") && has("file", "folder", "directory") {
            return "a file it needs is missing"
        }
        // A download that stopped arriving: the reason fits beside Try Again; the hover keeps the server and the file.
        if has("not connected to the internet", "appears to be offline") { return "this Mac is offline" }
        if has("stopped sending data", "timed out") && has("hugging face", "download", "network", "connection", "server", "request") {
            return "the connection stalled"
        }
        if has("couldn't be removed", "couldn't be deleted") { return "a file could not be deleted" }
        if has("couldn't be saved", "couldn't be written", "couldn't be created", "couldn't be copied", "couldn't be moved") { return "a file could not be saved" }
        if has("couldn't be opened", "couldn't be read", "isn't in the correct format", "is corrupt", "is damaged") { return "a file it needs could not be read" }
        if has("the operation couldn't be completed", "nscocoaerrordomain", "nsposixerrordomain", "nsurlerrordomain") { return "something went wrong on this Mac" }
        return nil
    }

    /// The visible text for a failure: "Pyannote: a file it needs is missing" (the model's name when known), and the
    /// hover: the raw message when it was reworded, else the text itself.
    static func line(_ raw: String, model: String?) -> (text: String, help: String) {
        let words = plain(raw)
        let said = words ?? raw
        let text = model.map { "\($0): \(said)" } ?? (words == nil ? raw : said.prefix(1).uppercased() + said.dropFirst())
        return (text, words == nil ? text : raw)
    }
    /// A line someone else composed ("Local models folder: The file … couldn’t be opened …"), reworded when it carries
    /// a raw message: its short lead ("Local models folder") stays, the rest becomes plain words; the hover is the line
    /// as it came. Nil when the line has nothing raw in it.
    static func reword(_ text: String) -> (text: String, help: String)? {
        guard plain(text) != nil else { return nil }
        if let cut = text.range(of: ": ") {
            let lead = String(text[..<cut.lowerBound]), rest = String(text[cut.upperBound...])
            if lead.count <= 40, !lead.contains("."), plain(lead) == nil, let words = plain(rest) { return ("\(lead): \(words)", text) }
        }
        return (short(text), text)
    }
    /// A failure's first sentence for a row's two-line status, when it has no plain wording ("Get failed: Hash
    /// mismatch on model.safetensors."); what follows stays in the hover with the rest of the message.
    static func lead(_ text: String) -> String {
        guard let end = text.range(of: ". ") else { return text }
        return String(text[..<end.lowerBound]) + "."
    }
    /// One line's own text (a row's status, a settings card's error line): the plain words, capitalised, or the
    /// message as it is.
    static func short(_ raw: String) -> String {
        guard let words = plain(raw) else { return raw }
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}
