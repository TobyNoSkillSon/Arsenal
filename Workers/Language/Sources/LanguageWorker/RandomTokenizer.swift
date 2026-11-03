// SPDX-License-Identifier: MIT
import Foundation
import MLXLMCommon

/// Lab-only byte tokenizer: exercises the real graph and image processor at reduced dimensions.
/// It does not qualify the checkpoint's tokenizer or chat template.
struct RandomTokenizer: MLXLMCommon.Tokenizer {
    let imageID: Int
    let startID: Int
    let endID: Int
    let imageToken: String
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let specials = ["<|image_pad|>": imageID, "<image>": imageID, "<|vision_start|>": startID, "<|vision_end|>": endID, "<|im_start|>": startID, "<|im_end|>": endID]
        var remaining = text[...]; var output: [Int] = []
        while !remaining.isEmpty {
            if let entry = specials.first(where: { remaining.hasPrefix($0.key) }) {
                output.append(entry.value); remaining = remaining.dropFirst(entry.key.count)
            } else {
                output += String(remaining.removeFirst()).utf8.map { Int($0) + 1 }
            }
        }
        return output
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { String(decoding: tokenIds.map { UInt8(clamping: $0 - 1) }, as: UTF8.self) }
    func convertTokenToId(_ token: String) -> Int? {
        if token.lowercased().contains("image") { return imageID }; return token == "<|vision_start|>" ? startID : token == "<|vision_end|>" ? endID : nil
    }
    func convertIdToToken(_ id: Int) -> String? { id == imageID ? imageToken : nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        let text = messages.map { message in
            if let content = message["content"] as? String { return content }
            return (message["content"] as? [[String: any Sendable]] ?? []).map { part in
                if part["type"] as? String == "image" { return "<|vision_start|>" + imageToken + "<|vision_end|>" }
                return part["text"] as? String ?? ""
            }.joined(separator: "\n")
        }.joined(separator: "\n")
        return encode(text: text, addSpecialTokens: false)
    }
}
