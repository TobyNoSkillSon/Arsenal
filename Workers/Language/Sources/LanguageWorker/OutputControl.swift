// SPDX-License-Identifier: MIT
import Foundation

/// Chat-role markers and control tokens that a model can write into its answer text. Index-Translate wrote a literal
/// `<assistant>` before its German translation. Markers that lead or trail the text (with whitespace) are removed in any
/// order; `<think>…</think>` blocks are removed wherever they occur. Free-form reasoning prose is not detected here.
enum OutputControl {
    /// Role tags and turn markers of the chat formats Language models use (Qwen and Index `<|im_start|>`, Gemma
    /// `<start_of_turn>`, bare `<assistant>`-style tags). The model's own control tokens are added by the caller.
    private static let roleMarkers = [
        #"(?:<\|im_start\|>|<start_of_turn>)\s*(?:assistant|user|system|model)\b"#,
        #"<\|im_start\|>"#, #"<\|im_end\|>"#, #"<end_of_turn>"#,
        #"</?(?:assistant|user|system|model)>"#
    ]

    static func strip(_ text: String, controlTokens: [String] = []) -> String {
        let markers = (roleMarkers + controlTokens.filter { !$0.isEmpty }.map { NSRegularExpression.escapedPattern(for: $0) }).joined(separator: "|")
        var value = text.replacingOccurrences(of: #"<think>[\s\S]*?</think>"#, with: "", options: .regularExpression)
        var previous: String?
        while previous != value {
            previous = value
            value = value.trimmingCharacters(in: .whitespacesAndNewlines)
            value = value.replacingOccurrences(of: "^(?:\(markers))\\s*", with: "", options: .regularExpression)
            value = value.replacingOccurrences(of: "\\s*(?:\(markers))$", with: "", options: .regularExpression)
        }
        return value
    }

    /// The model's own special tokens: tokenizer.json's added tokens marked special that its chat template writes out.
    /// Special tokens the template never uses (table cell tokens, image pads) are not turn markers and stay out.
    static func controlTokens(directory: URL, template: String?) -> [String] {
        guard let template, let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let added = json["added_tokens"] as? [[String: Any]]
        else { return [] }
        return added.compactMap { entry in
            guard entry["special"] as? Bool == true, let content = entry["content"] as? String, !content.isEmpty, template.contains(content) else { return nil }
            return content
        }
    }
}
