// SPDX-License-Identifier: MIT
import Testing
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) struct Round2FixEAppTests {
    @Test func brokenLocalRowKeepsUnloadAndErrorOnItsOwnRow() {
        var row = ModelRow(
            id: "local", name: "Local", kind: "voices", states: ["bf16": .init(state: .failed, error: "Bad model.toml", phase: "localFolderErrorLoaded")],
            verification: "Bad model.toml")
        row.local = true
        let presentation = RowPresentation(row)
        #expect(presentation.hot && presentation.action == .unload)
        #expect(presentation.status.text == "Bad model.toml" && presentation.status.help == "Bad model.toml")
        let controller = TableController(service: FixtureService(rows: []), chip: nil)
        controller.apply(rows: [row], kinds: Fixtures.fallbackKinds)
        #expect(controller.lastError == nil)
        controller.lastError = "Other model: Get failed"
        controller.apply(rows: [row], kinds: Fixtures.fallbackKinds)
        #expect(controller.lastError == "Other model: Get failed")
        row.states = ["bf16": .init(state: .loaded)]
        controller.apply(rows: [row], kinds: Fixtures.fallbackKinds)
        #expect(controller.lastError == "Other model: Get failed")
    }
    @Test func unreadableKindStillHasATableGroup() {
        let row = ModelRow(id: "broken", name: "Broken", kind: "local", states: ["bf16": .init(state: .failed, error: "Bad folder", phase: "localFolderError")])
        let controller = TableController(service: FixtureService(rows: []), chip: nil)
        controller.apply(rows: [row], kinds: Fixtures.fallbackKinds)
        #expect(controller.groups.contains { $0.rows.map(\.id) == ["broken"] })
    }
}
