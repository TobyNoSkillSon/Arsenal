// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func realMenuPlacementAndSettingsCommitOnceOrRestoreAll() async throws {
    let setting: TOMLValue = .table([
        "id": .string("speed"), "title": .string("Speed"), "type": .string("number"), "default": .integer(1), "min": .decimal(0.5), "max": .integer(2),
        "applies": .string("request"),
        "maps_to": .string("request.speed")
    ])
    let f = try await makeLocalCatalogFixture(kinds: ["chat"], extraTable: ["tools": .array([.string("chat")]), "setting": .array([setting])])
    defer { f.stop(); try? FileManager.default.removeItem(at: f.paths.support.deletingLastPathComponent()) }
    // Downloaded, so it is on the table, where Show in menu and its settings are edited.
    try await f.service.get(id: "fixture-chat", precision: "bf16")
    let id = try #require(await f.service.rows().first?.id)
    try await f.service.setSettings(id: id, values: ["__tools": false, "speed": 1.5])
    #expect(await f.service.rows().first?.toolsShown == false)
    #expect(try await f.service.settings(id: id).first?.value == 1.5)
    #expect(await f.service.catalog(query: .init()).first?.row.toolsShown == false)
    let file = f.paths.support.appendingPathComponent("config.json"), saved = try Data(contentsOf: file)
    for invalid: JSONObject in [["__tools": 1, "speed": 1], ["__tools": true, "speed": 99], ["__tools": true, "unknown": 1]] {
        await #expect(throws: (any Error).self) { try await f.service.setSettings(id: id, values: invalid) }
        #expect(await f.service.rows().first?.toolsShown == false)
        #expect(try await f.service.settings(id: id).first?.value == 1.5)
        #expect(try Data(contentsOf: file) == saved)
    }
    try FileManager.default.removeItem(at: file)
    try FileManager.default.createDirectory(at: file, withIntermediateDirectories: false)
    await #expect(throws: (any Error).self) { try await f.service.setSettings(id: id, values: ["__tools": true, "speed": 2]) }
    #expect(await f.service.rows().first?.toolsShown == false)
    #expect(try await f.service.settings(id: id).first?.value == 1.5)
    try FileManager.default.removeItem(at: file); try saved.write(to: file)
    let reopened = try await f.reopen(threshold: 8_000_000_000)
    #expect(await reopened.rows().first?.toolsShown == false)
    #expect(try await reopened.settings(id: id).first?.value == 1.5)
    try await f.service.setSettings(id: id, values: ["__tools": true])
    #expect(await f.service.rows().first?.toolsShown == true)
    #expect(await f.service.catalog(query: .init()).first?.row.toolsShown == true)
}
