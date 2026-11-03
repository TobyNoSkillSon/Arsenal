// SPDX-License-Identifier: MIT
import Foundation
import MLXLMCommon
import Tokenizers

// Adapter to the pinned swift-transformers implementation; no remote tokenizer downloads.
struct LocalTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let inner = try await AutoTokenizer.from(modelFolder: directory)
        let path = directory.appendingPathComponent("chat_template.jinja")
        let template = FileManager.default.fileExists(atPath: path.path) ? try String(contentsOf: path, encoding: .utf8) : nil
        return LocalTokenizer(inner: inner, template: template)
    }
}
struct LocalTokenizer: MLXLMCommon.Tokenizer {
    let inner: any Tokenizers.Tokenizer
    let template: String?
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { inner.encode(text: text, addSpecialTokens: addSpecialTokens) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { inner.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { inner.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { inner.convertIdToToken(id) }
    var bosToken: String? { inner.bosToken }
    var eosToken: String? { inner.eosToken }
    var unknownToken: String? { inner.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        try inner.applyChatTemplate(
            messages: messages, chatTemplate: template.map { .literal($0) }, addGenerationPrompt: true, truncation: false, maxLength: nil,
            tools: tools, additionalContext: additionalContext)
    }
}
