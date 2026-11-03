// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import Testing
@testable import ArsenalBench

/// Shipped results record the GPU core count (and may name their workload); the catalog build validates every one.
@Test func shippedResultsWithGPUCoresValidateAndUnknownFieldsDoNot() throws {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let data = try Data(contentsOf: repo.appendingPathComponent("models/d1-3b/results/m5-max.json"))
    let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(root["gpu_cores"] as? Int == 40)
    #expect(try ResultValidation.validate(data: data).gpu_cores == 40)
    var unknown = root
    unknown["surprise"] = 1
    #expect(throws: BenchmarkError.self) { try ResultValidation.validate(data: JSONSerialization.data(withJSONObject: unknown)) }
    var missing = root
    missing.removeValue(forKey: "chip")
    #expect(throws: BenchmarkError.self) { try ResultValidation.validate(data: JSONSerialization.data(withJSONObject: missing)) }
}
