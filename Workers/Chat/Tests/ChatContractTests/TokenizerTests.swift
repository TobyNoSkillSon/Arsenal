// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ChatContract
@testable import ChatEngine

struct TokenizerTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_CHAT_TOKENIZER_FIXTURE"] != nil))
    func requestSettingsUseVendorTemplate() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["ARSENAL_CHAT_TOKENIZER_FIXTURE"])
        let tokenizer = try await LocalTokenizerLoader().load(from: URL(fileURLWithPath: path))
        for thinking in [false, true] {
            for explicit in [false, true] {
                var messages = [["role": "user", "content": "Hi"]]
                if explicit { messages.insert(["role": "system", "content": "Explicit."], at: 0) }
                let request = try ChatRequest(["messages": messages, "system_prompt": "Default.", "thinking": thinking])
                let ids = try tokenizer.applyChatTemplate(
                    messages: request.messages.map {
                        ["role": $0.role, "content": $0.content] as [String: any Sendable]
                    }, tools: nil, additionalContext: ["enable_thinking": request.thinking])
                let rendered = tokenizer.decode(tokenIds: ids)
                // Thinking-on prepends the vendor's reasoning-effort instructions to the system content.
                #expect(rendered.contains("<|im_start|>system\n"))
                #expect(rendered.contains("\(explicit ? "Explicit." : "Default.")<|im_end|>"))
                #expect(rendered.hasSuffix(thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n"))
                if explicit { #expect(!rendered.contains("Default.")) }
            }
        }
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_CHAT_TOKENIZER_FIXTURE"] != nil))
    func pinnedJinjaAndGenerationPrompt() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["ARSENAL_CHAT_TOKENIZER_FIXTURE"])
        let tokenizer = try await LocalTokenizerLoader().load(from: URL(fileURLWithPath: path))
        let ids = try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": "Be brief."], ["role": "user", "content": "Hi"],
                ["role": "assistant", "content": "Hello"], ["role": "user", "content": "What is 2+2?"]
            ], tools: nil, additionalContext: ["enable_thinking": false])
        let rendered = tokenizer.decode(tokenIds: ids)
        #expect(rendered.contains("<|im_start|>system\nBe brief.<|im_end|>"))
        #expect(rendered.contains("Hello<|im_end|>"))
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        #expect(ids.count > tokenizer.encode(text: "Hi").count)
        #expect(
            ids
                == (try tokenizer.applyChatTemplate(
                    messages: [
                        ["role": "system", "content": "Be brief."], ["role": "user", "content": "Hi"],
                        ["role": "assistant", "content": "Hello"], ["role": "user", "content": "What is 2+2?"]
                    ], tools: nil, additionalContext: ["enable_thinking": false])))
    }
}
