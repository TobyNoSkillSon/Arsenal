// SPDX-License-Identifier: MIT
import Testing
import Foundation
import ChatContract

struct SettingsTests {
    let user: [[String: Any]] = [["role": "user", "content": "Hi"]]
    @Test func thinkingAndSamplingDefaults() throws {
        let base = try ChatRequest(["messages": user])
        #expect(!base.thinking); #expect(base.temperature == 0.6); #expect(base.topP == 1)
        let changed = try ChatRequest(["messages": user, "thinking": true, "temperature": 0.2, "top_p": 0.9])
        #expect(changed.thinking); #expect(changed.temperature == 0.2); #expect(changed.topP == 0.9)
        for value: Any in ["true", 1, NSNull()] {
            #expect(throws: (any Error).self) { try ChatRequest(["messages": user, "thinking": value]) }
        }
    }
    @Test func systemPromptMerge() throws {
        let merged = try ChatRequest(["messages": user, "system_prompt": "Answer in Polish."])
        #expect(merged.messages == [.init(role: "system", content: "Answer in Polish."), .init(role: "user", content: "Hi")])
        let existing = [["role": "system", "content": "Explicit."], ["role": "user", "content": "Hi"]]
        #expect(try ChatRequest(["messages": existing, "system_prompt": "Default."]).messages.first?.content == "Explicit.")
        #expect(try ChatRequest(["messages": user, "system_prompt": ""]).messages.count == 1)
        #expect(throws: (any Error).self) { try ChatRequest(["messages": user, "system_prompt": true]) }
    }
    @Test func canonicalAndLegacyContext() throws {
        #expect(try ChatRequest.contextLimit([:]) == 8192)
        #expect(try ChatRequest.contextLimit(["context_limit": 1024]) == 1024)
        #expect(try ChatRequest.contextLimit(["settings": ["context_limit": 2048], "context_limit": 1024]) == 2048)
        #expect(try ChatRequest.contextLimit(["settings": [:], "context_limit": 1024]) == 1024)
        #expect(try ChatRequest.contextLimit(["settings": ["context_limit": 131072]]) == 131072)
        #expect(try ChatRequest.contextLimit(["settings": ["context_limit": 262144]]) == 262144)
        for config: [String: Any] in [["settings": true], ["settings": ["context_limit": true]], ["context_limit": 262145]] {
            #expect(throws: (any Error).self) { try ChatRequest.contextLimit(config) }
        }
    }
    @Test func reasoningDelimiterEverySplit() {
        let source = "First work it out.\n</think>\n\n42"
        for n in 0...source.count {
            var filter = ReasoningFilter(thinking: true)
            let deltas = filter.append(String(source.prefix(n))) + filter.append(String(source.dropFirst(n))) + filter.finish()
            #expect(deltas.filter { $0.event == .reasoning }.map(\.text).joined() == "First work it out.\n")
            #expect(deltas.filter { $0.event == .token }.map(\.text).joined() == "\n\n42")
        }
    }
    @Test func noThinkingAndIncompleteReasoning() {
        var plain = ReasoningFilter(thinking: false)
        #expect(plain.append("42") == [.init(event: .token, text: "42")]); #expect(plain.finish().isEmpty)
        var cut = ReasoningFilter(thinking: true)
        #expect(cut.append("work </thi") == [.init(event: .reasoning, text: "work ")])
        #expect(cut.finish() == [.init(event: .reasoning, text: "</thi")])
    }
}
