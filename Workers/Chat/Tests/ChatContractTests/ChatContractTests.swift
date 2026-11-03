// SPDX-License-Identifier: MIT
import Testing
import ChatContract

struct ChatContractTests {
    let messages: [[String: Any]] = [["role": "system", "content": "Be brief."], ["role": "user", "content": "Hi"]]
    @Test func defaultsAndRoles() throws {
        let request = try ChatRequest(["messages": messages])
        #expect(request.messages[0].role == "system")
        #expect(request.maxTokens == 512); #expect(!request.stream)
        #expect(request.temperature == 0.6); #expect(request.topP == 1)
    }
    @Test func textBlocks() throws {
        let request = try ChatRequest(["messages": [["role": "user", "content": [["type": "text", "text": "a"], ["type": "text", "text": "b"]]]]])
        #expect(request.messages[0].content == "ab")
    }
    @Test func unsupportedFeatures() {
        for extra: [String: Any] in [["tools": [["type": "function"]]], ["response_format": ["type": "json_schema"]], ["response_format": ["type": "json_object"]]] {
            do {
                _ = try ChatRequest(["messages": messages].merging(extra) { _, new in new })
                Issue.record("Unsupported capability was accepted")
            } catch { #expect((error as? ChatFailure)?.code == "unsupported") }
        }
        #expect(throws: (any Error).self) { try ChatRequest(["messages": [["role": "tool", "content": "x"]]]) }
        #expect(throws: (any Error).self) { try ChatRequest(["messages": [["role": "user", "content": [["type": "image_url"]]]]]) }
    }
    @Test func samplingAndTypes() throws {
        let r = try ChatRequest(["messages": messages, "temperature": 0, "top_p": 0.8, "seed": 42, "stream": true, "max_tokens": 64])
        #expect(r.seed == 42); #expect(r.stream); #expect(r.temperature == 0)
        for extra: [String: Any] in [
            ["max_tokens": true], ["max_tokens": 1.5], ["max_tokens": 0], ["seed": -1], ["seed": 9_007_199_254_740_992], ["seed": true], ["temperature": -1], ["top_p": 0],
            ["stream": "yes"], ["stop": ""], ["tools": "bad"]
        ] {
            #expect(throws: (any Error).self) { try ChatRequest(["messages": messages].merging(extra) { _, new in new }) }
        }
    }
    @Test func splitStopDoesNotLeak() {
        var filter = StopFilter(stops: ["<END>"])
        #expect(filter.append("hello <E") == "hello ")
        #expect(filter.append("ND> ignored") == "")
        #expect(filter.stopped); #expect(filter.finish() == "")
    }
    @Test func earliestAndOverlappingStops() {
        var filter = StopFilter(stops: ["later", "ab", "abc"])
        #expect(filter.append("xabc later") == "x"); #expect(filter.stopped)
    }
    @Test func incompleteStopAndUnicode() {
        var filter = StopFilter(stops: ["🛑done"])
        #expect(filter.append("Cześć 🛑d") == "Cześć ")
        #expect(filter.finish() == "🛑d")
        var plain = StopFilter(stops: [])
        #expect(plain.append("日本語") == "日本語")
    }
    @Test func cancellation() { let c = Cancellation(); #expect(!c.isCancelled); c.cancel(); #expect(c.isCancelled) }
}

extension ChatContractTests {
    @Test func stopStringsUseUTF8BytesForSingleAndArrayInputs() throws {
        let boundary = String(repeating: "🛑", count: 256), over = String(repeating: "🛑", count: 257)
        for stop: Any in [boundary, [boundary]] { _ = try ChatRequest(["messages": messages, "stop": stop]) }
        for stop: Any in [over, [over]] { #expect(throws: ChatFailure.self) { try ChatRequest(["messages": messages, "stop": stop]) } }
    }
}
