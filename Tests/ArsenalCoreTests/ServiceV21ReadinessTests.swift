// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

// Native runner selection and machine eligibility stay independent of the model weights.

private func runner(_ name: String, chips: [String] = ["*"], os: String = "26.0") -> TOMLValue {
    .table([
        "name": .string(name), "language": .string("swift"), "worker": .string("stub"), "protocol": .integer(1), "chips": .array(chips.map(TOMLValue.string)),
        "min_macos": .string(os), "fast_exact": .bool(true)
    ])
}
@Test func v21RunnerReadinessAndIneligibleReasons() async throws {
    let table: [String: TOMLValue] = [
        "runner": .array([runner("native"), runner("alternate"), runner("foreign", chips: ["m1"]), runner("future", os: "99.0")]),
        "selection": .table(["default": .string("native")])
    ]
    let f = try await makeLocalCatalogFixture(kinds: ["voices"], extraTable: table); defer { f.stop() }
    let s = f.service, id = "fixture-voices"
    let entry = try #require(await s.catalog().first)
    let byName = Dictionary(uniqueKeysWithValues: entry.runners.map { ($0.name, $0) })
    #expect(byName["native"]?.eligible == true && byName["native"]?.ineligibleReason == nil)
    #expect(byName["alternate"]?.eligible == true)
    #expect(byName["foreign"]?.eligible == false && byName["foreign"]?.ineligibleReason == "Built for M1")
    #expect(byName["future"]?.ineligibleReason == "Needs macOS 99.0")
    #expect(entry.sourceDtype == "bf16")

    // The selected native runner and its accepted default survive the round trip.
    try await s.setRunner(id: id, runner: "alternate")
    #expect(await catalogRow(s, id)?.runner == "alternate")
    try await s.setRunner(id: id, runner: nil)
    #expect(await catalogRow(s, id)?.runner == "native")
    await s.shutdown()
}

@Test func v21ReadinessFieldsRoundTripAndDecodeFromOlderPayloads() throws {
    let row = ModelRow(id: "m", name: "M", kind: "voices")
    #expect(try JSONDecoder().decode(ModelRow.self, from: JSONEncoder().encode(row)) == row)
    // Older payloads have no precision evidence, so decoding retains the existing pending-measurement defaults.
    var legacyRow = row; legacyRow.recommendationPending = true; legacyRow.recommendationReason = "Precision measurements are pending."
    #expect(try JSONDecoder().decode(ModelRow.self, from: Data(#"{"id":"m","name":"M","kind":"voices"}"#.utf8)) == legacyRow)
    let r = CatalogRunner(name: "alternate", language: "swift", engine: "stub", ineligibleReason: "Built for M4")
    #expect(try JSONDecoder().decode(CatalogRunner.self, from: JSONEncoder().encode(r)) == r)
    let old = try JSONDecoder().decode(
        CatalogRunner.self, from: Data(#"{"name":"n","language":"swift","engine":"voices","acceptedDefault":true,"eligible":true,"selected":true,"results":[]}"#.utf8))
    #expect(old.ineligibleReason == nil && old.acceptedDefault)
}
