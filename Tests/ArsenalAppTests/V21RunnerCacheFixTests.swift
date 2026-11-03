// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) struct V21RunnerCacheFixTests {
    @Test func refreshInvalidatesStaleRunnerCache() async throws {
        // On the table: its weights are on this Mac (deleting them takes it off, and its runners with it).
        let r = ModelRow(id: "m", name: "M", kind: "voices", states: ["bf16": PrecisionState(state: .ready)])
        let svc = FixtureService(rows: [r], metadata: ["m": ["runner": .array([["name": "native", "language": "swift"], ["name": "new", "language": "swift"]])]])
        let c = TableController(service: svc)
        // Cache from before a folder acquired a second runner; current catalog already has both.
        c.setRunners(["m": [CatalogRunner(name: "native", language: "swift", engine: "stub")], "removed": []])
        await c.refresh()
        print("RV21 cache: actual=\(await svc.catalog(query: .init()).first?.runners.count ?? -1) cached=\(c.runners["m"]?.count ?? -1) settings=\(c.settings["m"]?.count ?? -1)")
        #expect(c.runners["removed"] == nil && c.runners["m"]?.count == 2)
        #expect(c.settings["m"]?.contains { $0.id == TableController.runnerKey } == true)
        try await svc.setRunner(id: "m", runner: "new")
        await c.refresh()
        #expect(c.runners["m"]?.first { $0.name == "new" }?.selected == true)
        #expect(c.settings["m"]?.first { $0.id == TableController.runnerKey }?.value == "new")
        try await svc.remove(id: "m", deleteWeights: true)
        await c.refresh()
        #expect(c.runners.isEmpty)
    }
}
