// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// One undecodable record must not blank the weights table: its model is skipped and reported as an error event, the other
/// models still list, and deletes still work for every model.
@Test func corruptAssetReferenceSkipsOnlyItsModelAndDeletesStillWork() async throws {
    let f = try await makeLocalCatalogFixture(kinds: ["chat", "voices"]); defer { f.stop() }
    let service = f.service
    try await service.get(id: "fixture-chat", precision: "bf16")
    try await service.get(id: "fixture-voices", precision: "bf16")
    // A damaged asset reference for voices: the file exists but is not JSON.
    let references = f.paths.directory("weights").appendingPathComponent("asset-references")
    try FileManager.default.createDirectory(at: references, withIntermediateDirectories: true)
    try Data("{ not json".utf8).write(to: references.appendingPathComponent("fixture-voices.json"))
    let events = await service.events()
    let listed = try await service.onDiskModels()
    #expect(listed.contains { $0.id == "fixture-chat" })
    #expect(listed.contains { $0.id == "fixture-voices" } == false, "the damaged model should be skipped")
    let message = await firstErrorEvent(events, id: "fixture-voices")
    #expect(message?.contains("could not be read") == true, "no footer error was reported: \(message ?? "none")")
    // Deletes still work for the other model, and for the damaged one's own files.
    #expect(try await service.deletePrecision(id: "fixture-chat", precision: "bf16") > 0)
    #expect(try await service.deleteAllWeights(id: "fixture-voices") > 0)
    #expect(try await service.onDiskModels().isEmpty)
    await service.shutdown()
}

/// The first error event for `id` on the stream, or nil when none arrives within five seconds.
private func firstErrorEvent(_ stream: AsyncStream<ServiceEvent>, id: String) async -> String? {
    await withTaskGroup(of: String?.self) { group in
        group.addTask {
            for await event in stream {
                if case .error(let eventID, let message) = event, eventID == id { return message }
            }
            return nil
        }
        group.addTask {
            try? await Task.sleep(for: .seconds(5))
            return nil
        }
        let first = await group.next() ?? nil
        group.cancelAll()
        return first
    }
}
