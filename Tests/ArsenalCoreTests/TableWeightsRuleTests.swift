// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

// The Models table holds exactly the models with weights on this Mac or a Get under way; every other model lives only in
// Get more models, and no weights on disk are ever hidden there, where they could not be deleted.

@Test func tableRuleCountsWeightsAndGetsOnly() {
    func row(_ states: [String: ModelState], withdrawn: [String: Int64] = [:]) -> ModelRow {
        var r = ModelRow(id: "m", name: "M", kind: "voices", precisions: Array(states.keys), states: states.mapValues { PrecisionState(state: $0) })
        r.withdrawnTiers = withdrawn
        return r
    }
    #expect(!row(["fp16": .notInstalled]).holdsWeightsOrGet)
    #expect(!row(["fp16": .notInstalled, "int4": .notInstalled]).holdsWeightsOrGet)
    for state in [ModelState.ready, .loaded, .loading, .updateAvailable, .downloading, .deriving, .failed] {
        #expect(row(["fp16": .notInstalled, "int4": state]).holdsWeightsOrGet, "\(state)")
    }
    // A local folder's or engine's failure is no Get and no weights: no row (it could not be removed from the table).
    for phase in ["localEngineError", "localFolderError"] {
        var r = row(["fp16": .notInstalled]); r.states["fp16"] = PrecisionState(state: .failed, phase: phase)
        #expect(!r.holdsWeightsOrGet, "\(phase)")
    }
    // An old precision the entry no longer offers is still weights on this Mac.
    #expect(row(["fp16": .notInstalled], withdrawn: ["int8": 400_000_000]).holdsWeightsOrGet)
}

@Test func stubTableShowsOnlyModelsWithWeightsAndGetJoinsIt() async throws {
    let rows = [
        ModelRow(id: "kept", name: "Kept", kind: "voices", states: ["bf16": .init(state: .ready)]),
        ModelRow(id: "absent", name: "Absent", kind: "voices", states: ["bf16": .init(state: .notInstalled)])
    ]
    let s = StubModelsService(rows: rows)
    // Both were added; only the one with weights is on the table, and the catalog says the same.
    #expect(await s.rows().map(\.id) == ["kept"])
    let catalog = await s.catalog()
    #expect(catalog.count == 2)
    #expect(catalog.first { $0.id == "kept" }?.added == true && catalog.first { $0.id == "absent" }?.added == false)
    // Get puts it on the table.
    try await s.get(id: "absent", precision: "bf16")
    #expect(await s.rows().map(\.id) == ["kept", "absent"])
    // Deleting every weight takes it off again.
    try await s.remove(id: "absent", deleteWeights: true)
    #expect(await s.rows().map(\.id) == ["kept"])
    #expect(await s.catalog().first { $0.id == "absent" }?.added == false)
}

@Test func realTableShowsOnlyDownloadedModels() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["voices", "images"], addToTable: false); defer { f.stop() }
    let s = f.service
    // Add without a download records the model, but it stays in Get more models.
    try await s.add(id: "fixture-images"); try await s.add(id: "fixture-voices")
    #expect(await s.rows().isEmpty)
    #expect(await s.catalog().allSatisfy { !$0.added })
    // A Get of a model Add never recorded joins the table, Show in menu off.
    try await s.remove(id: "fixture-voices", deleteWeights: false)
    try await s.get(id: "fixture-voices", precision: "bf16")
    #expect(await s.rows().map(\.id) == ["fixture-voices"])
    #expect(await s.rows().first?.toolsShown == false)
    #expect(await s.catalog().first { $0.id == "fixture-voices" }?.added == true)
    #expect(await s.catalog().first { $0.id == "fixture-images" }?.added == false)
    // Weights survive a restart on the table; a model without them does not appear there.
    await s.shutdown()
    let reopened = try await f.reopen(threshold: 8_000_000_000)
    #expect(await reopened.rows().map(\.id) == ["fixture-voices"])
    // Deleting its weights takes it off the table.
    try await reopened.remove(id: "fixture-voices", deleteWeights: true)
    #expect(await reopened.rows().isEmpty)
    #expect(await reopened.catalog().first { $0.id == "fixture-voices" }?.added == false)
    await reopened.shutdown()
}
