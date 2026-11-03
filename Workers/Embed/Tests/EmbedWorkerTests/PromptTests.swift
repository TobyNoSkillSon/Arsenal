import Testing
@testable import EmbedWorker

@Test func instructionsApplyToQueriesOnly() {
    #expect(TextRetrievalAdapter.embeddingPrompt("hello", task: "document", instruction: "custom") == "hello")
    #expect(TextRetrievalAdapter.embeddingPrompt("hello", task: "query", instruction: "custom") == "Instruct: custom\nQuery: hello")
}
/// Qwen3-Embedding's card: "Instruct: {task}\nQuery:{query}", no space (2 Nov 2026: the space and a missing
/// <|endoftext|> pooled the wrong token, cosine 0.46 to 0.70 against Qwen's own vectors).
@Test func qwenEmbeddingQueryHasNoSpaceAfterQuery() {
    #expect(TextRetrievalAdapter.embeddingPrompt("hello", task: "query", instruction: "custom", qwen: true) == "Instruct: custom\nQuery:hello")
    #expect(TextRetrievalAdapter.embeddingPrompt("hello", task: "document", instruction: "custom", qwen: true) == "hello")
}
@Test func jinaHasBothQueryMarkersAndOrderedDocuments() {
    let prompt = TextRetrievalAdapter.jinaPrompt(query: "question", documents: ["first", "second"])
    #expect(prompt.components(separatedBy: "<|rerank_token|>").count == 3)
    #expect(prompt.components(separatedBy: "<|embed_token|>").count == 3)
    #expect(prompt.contains("<passage id=\"0\">\nfirst"))
    #expect(TextRetrievalAdapter.cosine([1, 0], [-1, 0]) == -1)
}
@Test func qwenRerankPromptKeepsTheCardsWholeLengthAndCutsOnlyTheDocument() throws {
    let prefix = Array(0..<10), suffix = Array(100..<105), pair = Array(1000..<21000)
    // Default: the whole prompt is the card's max_length, 8,192, the pair cut from its end.
    let full = try TextRetrievalAdapter.qwenRerankIDs(prefix: prefix, leading: 20, pair: pair, suffix: suffix, maxDocument: 8192, limit: 8192)
    #expect(full.count == 8192)
    #expect(Array(full.prefix(10)) == prefix && Array(full.suffix(5)) == suffix)
    #expect(Array(full[10..<8187]) == Array(pair.prefix(8177)))
    // A per-document bound keeps the instruction and query whole.
    let bounded = try TextRetrievalAdapter.qwenRerankIDs(prefix: prefix, leading: 20, pair: pair, suffix: suffix, maxDocument: 100, limit: 8192)
    #expect(bounded.count == 10 + 120 + 5)
    // Short pairs pass unchanged; a query that leaves no room for the document fails.
    #expect(try TextRetrievalAdapter.qwenRerankIDs(prefix: prefix, leading: 2, pair: [7, 8, 9], suffix: suffix, maxDocument: 8192, limit: 8192) == prefix + [7, 8, 9] + suffix)
    #expect(throws: (any Error).self) {
        try TextRetrievalAdapter.qwenRerankIDs(prefix: prefix, leading: 8177, pair: pair, suffix: suffix, maxDocument: 8192, limit: 8192)
    }
    #expect(TextRetrievalAdapter.qwenRerankLength == 8192)
}
