// SPDX-License-Identifier: MIT
import Testing
@testable import Arsenal
import ArsenalCore

/// Rough reference figures say so where the figures are read: the baseline line and its tooltip.
@Test func quickCheckReferenceIsLabelledRough() throws {
    let object: [String: Any] = [
        "match": "same_chip", "chip": "m5-max", "date": "2026-10-08T06:20:00Z", "verification": "quick check", "metrics": ["latency_s": 0.0198],
        "suite": ["id": "arsenal-decisions-suite-v1"]
    ]
    let rough = try #require(ToolReference(object))
    #expect(rough.quickCheck && rough.baseline.hasPrefix("M5 Max \u{00b7} quick check"))
    #expect(rough.help.hasPrefix("Quick check on this chip") && rough.help.contains("Not a controlled measurement: rough figures."))
    let measured = try #require(ToolReference(["match": "same_chip", "chip": "m5-max", "verification": "Measured", "metrics": [:]]))
    #expect(!measured.quickCheck && !measured.baseline.contains("quick check"))
}
