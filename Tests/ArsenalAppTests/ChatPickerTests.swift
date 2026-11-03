// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

/// Chat picks its model inside the window, as every tool window does: one Chat entry in the menu, a pop-up of the chat
/// models in the band, and each model keeps its own conversation when the pop-up switches between them.
@MainActor @Suite(.menuState, .serialized) final class ChatPickerTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-chat-picker-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// The loaded fixture plus a second chat model on the table.
    private func rows() -> [ModelRow] {
        var rows = Fixtures.rows(.loaded)
        var second = rows.first { $0.id == "qwen3.8-27b" }!
        second.id = "chat-two"; second.name = "Chat Two"
        rows.append(second)
        return rows
    }

    @Test func oneChatEntryAndThePopUpSwitchesConversations() async throws {
        let app = AppDelegate(service: FixtureService(rows: rows()), paths: ArsenalPaths(support: support))
        app.tools.playback = FixturePlayback()
        app.tools.present = { _ in }
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        defer { app.tools.closeAll() }
        // One Chat entry, opened by a click (no submenu of model names).
        let index = try #require(app.menu.items.firstIndex { $0.title == "Chat\u{2026}" && !$0.isHidden })
        #expect(app.menu.items[index].submenu == nil && app.menu.items.filter { $0.title == "Chat\u{2026}" }.count == 1)
        app.menu.performActionForItem(at: index)
        let first = try #require(app.tools.chats.values.first)
        // The band offers both chat models.
        #expect(Set(first.session.choices.map(\.id)) == ["qwen3.8-27b", "chat-two"])
        let firstID = first.session.model.id, otherID = firstID == "chat-two" ? "qwen3.8-27b" : "chat-two"
        first.session.draft = "Kept for later"
        first.session.switchModel(otherID)
        let other = try #require(app.tools.chats[otherID])
        #expect(other.session.model.id == otherID && other.session.choices.count == 2)
        // Back again: the first model's conversation is as it was left.
        other.session.switchModel(firstID)
        #expect(app.tools.chats[firstID]?.session.draft == "Kept for later")
    }
}
