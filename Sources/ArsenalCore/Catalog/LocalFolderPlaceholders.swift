// SPDX-License-Identifier: MIT
import Foundation

/// The kit's placeholder rule (kit/folder_check.py, S1-todo), applied at discovery: a scaffolded folder whose text still
/// still holds the kit's placeholder word is unfinished, so Arsenal shows it as an error row naming the placeholder and never offers approval or Get.
/// Files checked: .toml, .md, .json and .swift, outside runners/swift/bin and any .build folder; files over 1 MB (a
/// vendored tokenizer vocabulary can hold the word as a token) are skipped.
enum LocalFolderPlaceholders {
    static let maxBytes: Int64 = 1_048_576
    private static let pattern = try? NSRegularExpression(pattern: "\\bTODO\\b.{0,60}")

    /// The first placeholder, as one line naming the file and, in a TOML file, the key; nil when there is none.
    /// `files`: the folder's relative paths with their sizes. model.toml is read first, so its field is the one named.
    static func first(in folder: URL, files: [(path: String, bytes: Int64)]) -> String? {
        let checked = files.filter { file in
            let parts = file.path.split(separator: "/")
            return [".toml", ".md", ".json", ".swift"].contains { file.path.hasSuffix($0) } && file.bytes <= maxBytes
                && !parts.contains(".build") && !file.path.contains("runners/swift/bin/")
        }.sorted { ($0.path == "model.toml" ? 0 : 1, $0.path) < ($1.path == "model.toml" ? 0 : 1, $1.path) }
        for file in checked {
            guard let url = try? SafePath.resolve(file.path, inside: folder), let data = try? Data(contentsOf: url) else { continue }
            if let found = placeholder(in: String(decoding: data, as: UTF8.self), path: file.path) { return found }
        }
        return nil
    }
    static func placeholder(in text: String, path: String) -> String? {
        let range = NSRange(text.startIndex..., in: text)
        guard let match = pattern?.firstMatch(in: text, range: range), let found = Range(match.range, in: text) else { return nil }
        let snippet = text[found].trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\",]"))
        let field = path.hasSuffix(".toml") ? tomlKey(before: found.lowerBound, in: text) : nil
        return "\(path)\(field.map { " \($0)" } ?? "") is still a placeholder (\"\(snippet)…\"): replace it with the real value from the model card, then refresh."
    }
    /// The key whose value holds `index` ("name", "[facets] modalities_in"): the last `key =` line at or above it.
    static func tomlKey(before index: String.Index, in text: String) -> String? {
        var section: String?, key: String?
        for line in text[..<index].split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("[") {
                section = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "[] ")); key = nil
            } else if !trimmed.hasPrefix("#"), let equals = trimmed.firstIndex(of: "="),
                trimmed[..<equals].trimmingCharacters(in: .whitespaces).range(of: "^[A-Za-z0-9_.-]+$", options: .regularExpression) != nil
            {
                key = trimmed[..<equals].trimmingCharacters(in: .whitespaces)
            }
        }
        guard let key else { return section.map { "[\($0)]" } }
        return section.map { "[\($0)] \(key)" } ?? key
    }
}
