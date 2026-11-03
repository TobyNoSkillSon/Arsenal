// SPDX-License-Identifier: MIT
import Foundation
import MLXLMCommon
import Tokenizers

struct LocalTokenizer: MLXLMCommon.Tokenizer {
    let inner: any Tokenizers.Tokenizer
    let template: String?
    /// The model's own turn and role markers, stripped from answer text (OutputControl).
    let controlTokens: [String]
    init(directory: URL) async throws {
        // modelFolder never fetches missing files.
        inner = try await AutoTokenizer.from(modelFolder: directory)
        template = try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        controlTokens = OutputControl.controlTokens(directory: directory, template: template)
    }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { inner.encode(text: text, addSpecialTokens: addSpecialTokens) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { inner.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { inner.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { inner.convertIdToToken(id) }
    var bosToken: String? { inner.bosToken }
    var eosToken: String? { inner.eosToken }
    var unknownToken: String? { inner.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        try inner.applyChatTemplate(
            messages: messages, chatTemplate: template.map { .literal($0) }, addGenerationPrompt: true, truncation: false, maxLength: nil, tools: tools,
            additionalContext: additionalContext)
    }
}

/// The serial worker has no cooperative-executor callers. Bridge only local processor/tokenizer work.
func localAwait<T>(_ body: @escaping () async throws -> T) throws -> T {
    let box = AwaitBox<T>(); let semaphore = DispatchSemaphore(value: 0)
    Task {
        do { box.result = .success(try await body()) } catch { box.result = .failure(error) }; semaphore.signal()
    }
    semaphore.wait(); return try box.result!.get()
}
private final class AwaitBox<T>: @unchecked Sendable { var result: Result<T, Error>? }
