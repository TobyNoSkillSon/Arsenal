// SPDX-License-Identifier: MIT
import Testing
@testable import ArsenalCore
@Test func stubLifecycle() async throws {
    let s = StubModelsService(); let entries = await s.catalog(); #expect(entries.count == 6); #expect(await s.rows().isEmpty)
    for entry in entries { try await s.add(id: entry.id) }; let rows = await s.rows()
    for r in rows { try await s.get(id: r.id, precision: "bf16"); try await s.load(id: r.id); try await s.unload(id: r.id) }
    #expect(await s.memory().workersMB == 0)
}
