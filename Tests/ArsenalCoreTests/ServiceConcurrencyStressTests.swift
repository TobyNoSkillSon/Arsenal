// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The 2026-10-22 crash: a Get finishing while status polls, table edits and Load/Unload ran at once. `refreshInstalled`
/// awaited inside an optional-chained assignment to `modelRows`, so calls interleaving at that await released the row's
/// arrays twice (SIGSEGV in `_swift_release_dealloc`). Fails within 1.5 s before the fix, plain or under ASan/TSan.
/// `ARSENAL_STRESS_SECONDS` lengthens a soak; `ARSENAL_STRESS_LOOPS` (get,rows,catalog,table,load) picks the callers.
@Test func getCompletionRacesStatusTableAndLoadWithoutCorruption() async throws {
    let fixture = try await makeLocalCatalogFixture(kinds: ["voices", "decisions", "chat"]); defer { fixture.stop() }
    let s = fixture.service
    let env = ProcessInfo.processInfo.environment
    let deadline = ContinuousClock.now + .seconds(Double(env["ARSENAL_STRESS_SECONDS"] ?? "1.5") ?? 1.5)
    let on = Set((env["ARSENAL_STRESS_LOOPS"] ?? "get,rows,catalog,table,load").split(separator: ",").map(String.init))
    @Sendable func running() -> Bool { ContinuousClock.now < deadline }
    await withTaskGroup(of: Void.self) { group in
        if on.contains("get") {
            group.addTask {
                while running() {
                    for p in ["bf16", "int8", "int4"] { try? await s.get(id: "fixture-chat", precision: p) }
                    try? await s.delete(id: "fixture-chat"); try? await s.add(id: "fixture-chat")
                }
            }
        }
        if on.contains("rows") { group.addTask { while running() { _ = await s.rows(); _ = await s.memory() } } }
        if on.contains("catalog") { group.addTask { while running() { _ = await s.catalog(query: .init()) } } }
        if on.contains("table") {
            group.addTask {
                while running() { try? await s.remove(id: "fixture-voices", deleteWeights: false); try? await s.add(id: "fixture-voices") }
            }
        }
        if on.contains("load") {
            group.addTask {
                try? await s.get(id: "fixture-decisions", precision: "int4")
                while running() { try? await s.load(id: "fixture-decisions", precision: "int4", fast: false); try? await s.unload(id: "fixture-decisions") }
            }
        }
    }
    // The get loop ends on delete then Add, so the model is recorded without weights: in Get more models, and a Get after
    // the race puts it back on the table.
    #expect(await s.catalog(query: .init()).contains { $0.id == "fixture-chat" })
    try await s.get(id: "fixture-chat", precision: "bf16")
    #expect(await s.rows().contains { $0.id == "fixture-chat" })
    await s.shutdown()
}
