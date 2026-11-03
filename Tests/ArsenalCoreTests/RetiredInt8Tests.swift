// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// Since 21 Oct int8 is no longer offered. A saved int8 preference falls back to the 16-bit format, and get, load and the
/// precision choice refuse 8-bit through the service.
@Test func retiredInt8PreferenceFallsBackToSixteenBitAndIsRefused() async throws {
    var configuration = ArsenalConfiguration()
    configuration.selections["fixture-chat"] = ModelPreferences(precision: "int8")
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], configuration: configuration); defer { f.stop() }
    let service = f.service
    let row = try #require(await catalogRow(service, "fixture-chat"))
    #expect(row.selectedPrecision == "bf16", "the saved int8 resolved to \(row.selectedPrecision)")
    await expectRetiredRefusal("int8") { try await service.get(id: "fixture-chat", precision: "int8") }
    await expectRetiredRefusal("int8") { try await service.load(id: "fixture-chat", precision: "int8", fast: false) }
    await expectRetiredRefusal("int8") { try await service.setPrecision(id: "fixture-chat", precision: "int8") }
    // With the 16-bit copy on disk, a load without a precision uses the fallback, never the saved int8.
    try await service.get(id: "fixture-chat", precision: "bf16")
    try await service.load(id: "fixture-chat", precision: nil, fast: false)
    let loaded = try #require(await service.rows().first { $0.id == "fixture-chat" })
    let bf16 = try #require(loaded.states["bf16"])
    #expect(bf16.state == .loaded, "bf16 state was \(bf16.state)")
    #expect(loaded.states["int8"]?.state != .loaded)
    await service.shutdown()
}

/// Asserts that `action` throws a refusal naming the retired precision.
private func expectRetiredRefusal(_ dtype: String, _ action: () async throws -> Void) async {
    do {
        try await action()
        Issue.record("a retired \(dtype) request was allowed")
    } catch {
        #expect(error.localizedDescription.contains("no longer offered"), "message was: \(error.localizedDescription)")
    }
}
