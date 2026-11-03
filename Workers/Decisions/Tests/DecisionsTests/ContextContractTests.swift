// SPDX-License-Identifier: MIT
import Testing
import DecisionsProtocol
@testable import VerdictEngine

struct ContextContractTests {
    @Test func pinnedDecisionBudgetsOverrideTheEncoderMaximum() throws {
        for (id, limit) in [("laya-english", 512), ("laya-multilingual", 1024), ("laya-typed-decisions", 1024)] {
            #expect(try LayaPrompt.decisionContextLimit(["max_len": limit, "max_position_embeddings": 8192]) == limit)
            #expect(DecisionsCatalog.verdict.spec(id)?.context == limit)
        }
        for config: [String: Any] in [[:], ["max_len": 0], ["max_len": 8193], ["max_len": "512"], ["max_len": true]] {
            #expect(throws: (any Error).self) { try LayaPrompt.decisionContextLimit(config) }
        }
    }
}
