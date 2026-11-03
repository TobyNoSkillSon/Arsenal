// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI
import Testing
@testable import Arsenal

/// The settings card on its own: what it says when the service refuses a change or a Reload fails, and Restore defaults'
/// reason when it is off. The table and its card sit in an unordered window far off every screen; nothing is shown, no
/// app is activated, no cursor, focus or keyboard is touched.
@MainActor @Suite(.menuState, .serialized) final class SettingsCardReviewTests {
    init() { _ = NSApplication.shared }

    static let chat: JSONObject = [
        "runner": [["name": "native", "language": "swift", "engine": "chat"]],
        "setting": [
            ["id": "context", "title": "Context window", "type": "number", "default": 262_144, "min": 1024, "max": 262_144, "unit": "tokens", "applies": "load"],
            ["id": "thinking", "title": "Reason before replying", "type": "toggle", "default": false, "applies": "request"],
            ["id": "cache", "title": "Quantized cache", "type": "toggle", "default": false, "applies": "load"],
            [
                "id": "tone", "title": "Tone", "type": "choice", "default": "amber", "choices": ["amber", "brightest", "coolest", "darkest"], "applies": "request",
                "maps_to": "request.tone"
            ]
        ]
    ]
    private var chatRow: ModelRow { get throws { try #require(PolishTableTests.rows.first { $0.kind == "chat" }) } }
    /// The other rows beside it, so the table is tall enough for the whole card.
    static var others: [ModelRow] { PolishTableTests.rows.filter { $0.kind != "chat" } }

    // MARK: Errors

    /// A refused save: the switch goes back and the settings say why.
    @Test func aRefusedSettingShowsItsReasonInTheSettings() async throws {
        let row = try chatRow
        let service = FixtureService(rows: [row] + Self.others, metadata: [row.id: Self.chat], kinds: PolishTableTests.kinds)
        let controller = TableController(service: service)
        defer { controller.stop() }
        await controller.refresh()
        let (host, content) = try await open(controller, row)
        defer { host.close() }
        let before = host.height
        let reason = "This runner cannot reason before replying."
        #expect(!said(reason, content))
        await service.fail("setSetting", with: WorkerFailure(code: "invalid_setting", message: reason))
        let cache = try #require(host.views(MenuSwitch.self).first { $0.exposed?.accessibilityLabel() == "Quantized cache" })
        flip(cache)
        try await waitUntil { controller.lastError?.contains(reason) == true }
        await host.settle()
        #expect(said(reason, content), "The card names the refusal")
        #expect(controller.settings[row.id]?.first { $0.id == "cache" }?.value == .bool(false))
        #expect(host.height > before)
        // The next change that goes through clears it.
        await service.fail("setSetting", with: nil)
        flip(cache)
        try await waitUntil { controller.settings[row.id]?.first { $0.id == "cache" }?.value == .bool(true) }
        await host.settle()
        #expect(!said(reason, content))
        // A failure from the table's own buttons while the card is closed stays in the table's footer: a card opened
        // later starts clean.
        controller.settingsCard.close()
        controller.fail(row, reason)
        let (_, reopened) = try await open(controller, row)
        #expect(!said(reason, reopened) && controller.lastError?.contains(reason) == true)
    }

    /// A failed Reload, pressed at the form's foot through the real event path, is said in the settings.
    @Test func aFailedReloadShowsItsReasonInTheSettings() async throws {
        let row = try chatRow
        let service = FixtureService(rows: [row] + Self.others, metadata: [row.id: Self.chat], kinds: PolishTableTests.kinds)
        let controller = TableController(service: service)
        defer { controller.stop() }
        await controller.refresh()
        try await service.setSetting(id: row.id, key: "context", value: .number(131_072))
        await controller.refresh()
        let loaded = try #require(controller.rows.first { $0.id == row.id })
        #expect(SettingsPane.footer(controller.settings[row.id] ?? [], loaded: controller.presentation(loaded).hot).reload)
        let (host, content) = try await open(controller, loaded)
        defer { host.close() }
        let reason = "The runner stopped while reloading."
        await service.fail("apply", with: WorkerFailure(code: "load_failed", message: reason))
        try host.press(try #require(host.views(ActionButtonView.self).first))
        try await waitUntil { controller.lastError?.contains(reason) == true }
        #expect(await service.calls.contains("apply \(row.id)"))
        await host.settle()
        #expect(said(reason, content), "The settings name the failed Reload")
    }

    /// Remove, pressed at the card's foot through the real event path, removes the old precisions: the service is asked,
    /// and the Model hover's note goes with them.
    @Test func removeWithdrawnPrecisionsClearsTheNote() async throws {
        var row = try chatRow
        row.withdrawnTiers = ["int8": 1_200_000_000]
        let service = FixtureService(rows: [row] + Self.others, metadata: [row.id: Self.chat], kinds: PolishTableTests.kinds)
        let controller = TableController(service: service)
        defer { controller.stop() }
        await controller.refresh()
        let shown = try #require(controller.rows.first { $0.id == row.id })
        #expect(ModelHelp.withdrawn(shown)?.hasPrefix("An old 8-bit copy (1.2 GB) is still on this Mac") == true)
        let (host, content) = try await open(controller, shown)
        defer { host.close() }
        let remove = try #require(host.views(ActionButtonView.self).first { $0.title == "Delete" })
        #expect(remove.toolTip?.hasSuffix("Deletes the old copy from this Mac. The precisions the model offers stay.") == true)
        try host.press(remove)
        try await waitUntil { await service.calls.contains("delete withdrawn \(row.id)") }
        try await waitUntil { controller.rows.first { $0.id == row.id }?.withdrawnTiers.isEmpty == true }
        await host.settle()
        let after = try #require(controller.rows.first { $0.id == row.id })
        #expect(ModelHelp.withdrawn(after) == nil)
        #expect(!ModelHelp.text(after, presentation: RowPresentation(after)).contains("still on this Mac"))
        #expect(!host.views(ActionButtonView.self).contains { $0.title == "Delete" }, "The footer goes with the leftovers")
    }

    /// A refused removal says why in the table's footer and keeps the note: the copy is still on disk.
    @Test func aRefusedRemovalKeepsTheNoteAndSaysWhy() async throws {
        var row = try chatRow
        row.withdrawnTiers = ["int8": 1_200_000_000]
        let service = FixtureService(rows: [row] + Self.others, metadata: [row.id: Self.chat], kinds: PolishTableTests.kinds)
        let controller = TableController(service: service)
        defer { controller.stop() }
        await controller.refresh()
        let shown = try #require(controller.rows.first { $0.id == row.id })
        let (host, _) = try await open(controller, shown)
        defer { host.close() }
        let reason = "The old copy is in use."
        await service.fail("deleteWithdrawnTiers", with: WorkerFailure(code: "remove_failed", message: reason))
        try host.press(try #require(host.views(ActionButtonView.self).first { $0.title == "Delete" }))
        try await waitUntil { controller.lastError?.contains(reason) == true }
        await controller.refresh()
        #expect(controller.rows.first { $0.id == row.id }?.withdrawnTiers == ["int8": 1_200_000_000])
    }

    // MARK: Restore defaults

    /// Off with only locked settings away from their defaults, Restore defaults says they are locked, not that
    /// everything is at its default.
    @Test func restoreDefaultsSaysWhyWhenOnlyLockedSettingsDiffer() async throws {
        var row = try chatRow
        row.states = ["int4": .init(state: .ready)]
        let service = FixtureService(rows: [row] + Self.others, metadata: [row.id: Self.chat], kinds: PolishTableTests.kinds)
        let controller = TableController(service: service)
        defer { controller.stop() }
        try await service.setSetting(id: row.id, key: "context", value: .number(131_072))
        await controller.refresh()
        let ready = try #require(controller.rows.first { $0.id == row.id })
        let (host, content) = try await open(controller, ready)
        defer { host.close() }
        // The card has no Restore glyph; the controller's restore state still says what Restore defaults would do.
        #expect(!controller.restorable(ready).isEmpty)
        #expect(SettingsPane.restore(controller, row: ready) == (true, SettingsPane.restoreHelp))
        var busy = ready
        busy.states["int4"] = PrecisionState(state: .downloading, progress: 0.2)
        controller.apply(rows: controller.rows.map { $0.id == busy.id ? busy : $0 }, kinds: PolishTableTests.kinds)
        await host.settle()
        #expect(controller.restorable(busy).isEmpty)
        let restore = SettingsPane.restore(controller, row: busy)
        #expect(!restore.enabled && restore.help == "Locked while downloading", "Off because the changed setting is locked: \(restore)")
        #expect(restore.help != SettingsPane.restoredHelp)
    }

    // MARK: Helpers

    private func open(_ controller: TableController, _ row: ModelRow) async throws -> (SettingsCardHost, NSView) {
        let host = SettingsCardHost.of(controller)
        await host.open(row)
        current = host
        return (host, host.view)
    }
    /// The open card's host: what the card says is read inside its frame (the table's footer says the same in its own
    /// line).
    private var current: SettingsCardHost?
    /// What a click on a switch does once AppKit's tracking ends: the state flips and the button sends its action.
    /// (`performClick` on an ordered-in window runs AppKit's highlight delay, which stops the test runner's run loop.)
    private func flip(_ button: NSButton) {
        button.state = button.state == .on ? .off : .on
        #expect(button.sendAction(button.action, to: button.target))
    }
    private func tooltips(_ root: NSView) -> [String] {
        guard let current, current.view === root else { return [] }
        return current.views(TooltipCell.TooltipView.self).compactMap(\.toolTip) + current.views(GlyphButton.self).compactMap(\.toolTip)
    }
    /// The settings show this text: its error line carries the whole text as its tooltip, where it is drawn.
    private func said(_ text: String, _ root: NSView) -> Bool { tooltips(root).contains { $0.contains(text) } }
    private func waitUntil(seconds: Double = 5, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            guard Date() < deadline else { Issue.record("Timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
