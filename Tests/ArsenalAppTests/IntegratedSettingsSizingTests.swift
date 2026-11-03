// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

/// Real folders' settings in the card the real gear opens; nothing is shown on screen. The table the gear sits in does
/// not change: the card floats over it, sized to its form.
@MainActor @Suite(.menuState, .serialized) struct IntegratedSettingsSizingTests {
    @Test(arguments: ["qwen3.8-27b", "flux2-klein-4b", "laya-multilingual"])
    func folderSettingsOpenInACardThatFitsThem(id: String) async throws {
        let folder = SettingsRenderDelegate.repo.appendingPathComponent("models/\(id)")
        let source = try SettingsRenderDelegate.folder(folder)
        let metadata: JSONObject = [
            "setting": .array(source.settings.map { .object(SettingsRenderDelegate.declarationJSON($0)) }),
            "settings_toml": .string(try String(contentsOf: folder.appendingPathComponent(SettingsPanel.fileName), encoding: .utf8))
        ]
        var row = ModelRow(id: id, name: source.name, kind: source.kind, states: ["bf16": .init(state: .ready)])
        row.tools = source.tools
        let other = ModelRow(id: "other", name: "Other", kind: source.kind, states: ["bf16": .init(state: .ready)])
        let service = FixtureService(rows: [row, other], metadata: [id: metadata])
        let table = TableController(service: service)
        defer { table.stop() }
        await table.refresh()
        #expect(table.panels[id] == source.panel)
        table.apply(rows: table.rows, kinds: table.kinds, catalog: CatalogStatus(verified: true))
        let settings = SettingsCardHost(table)
        defer { settings.close() }
        await settings.settle()
        let host = settings.view
        let tableBefore = try await Self.render(host.hosting)
        let base = host.naturalSize
        let gear = try #require(settings.gear(id))
        // The gear's row takes the open fill and the filled gear while its card is open; the card covers its own frame.
        let band = gear.convert(gear.bounds, to: host.hosting)
        let rowBand = NSRect(x: 0, y: band.minY, width: host.hosting.bounds.width, height: band.height)
        try settings.press(gear)
        await settings.settle()
        #expect(table.settingsCard.openID == id && settings.state.id == id && gear.active)
        let card = settings.frame
        let mask = [rowBand, card.insetBy(dx: -16, dy: -16)]
        let opened = try await Self.render(host.hosting, excluding: mask)
        try settings.press(gear)
        await settings.settle()
        let closed = try await Self.render(host.hosting, excluding: mask)
        try settings.press(gear)
        await settings.settle()
        // The table did not move or grow: same size, same pixels outside the open row and the card (and its shadow).
        #expect(opened == closed && settings.frame == card && table.settingsCard.openID == id)
        #expect(host.naturalSize == base)
        // The card is as tall as its form (or the table, which then scrolls the form) and as wide as its columns.
        let natural = SettingsCard.height(table, row: try #require(table.rows.first { $0.id == id }))
        #expect(card.height == min(natural, ModelTable.height(table) - 2 * SettingsCard.inset).rounded(.up) && card.width == settings.state.geometry.width)
        // The gear again closes the card: the table is back to its exact first pixels.
        try settings.press(gear)
        await settings.settle()
        #expect(table.settingsCard.openID == nil && !gear.active)
        #expect(try await Self.render(host.hosting) == tableBefore)
        #expect(await service.calls.isEmpty)
        #expect(await service.runLog().isEmpty)
    }

    /// A load choice with many values is segments in the card, never a pop-up.
    @Test func aLongLoadChoiceIsSegmentsNotAPopUp() async throws {
        let row = ModelRow(id: "chat-sizing", name: "Chat", kind: "chat", states: ["bf16": .init(state: .ready)])
        let table = TableController(service: FixtureService(rows: []))
        defer { table.stop() }
        table.apply(rows: [row], kinds: Fixtures.kinds, catalog: CatalogStatus(verified: true))
        let context = SettingState(
            declaration: ["id": "context_limit", "title": "Context", "type": "choice", "choices": [8192, 32768, 131072, 262144], "default": 32768, "applies": "load"],
            value: .number(32768))
        table.settings[row.id] = TableController.withKeep([context], row)
        let settings = SettingsCardHost(table)
        defer { settings.close() }
        await settings.open(row)
        let segments = try #require(settings.views(MenuSegments.self).first)
        #expect(segments.segmentCount == 4 && segments.selectedSegment == 1)
        #expect(Self.views(NSPopUpButton.self, settings.view).isEmpty)
    }

    static func views<T: NSView>(_ type: T.Type, _ root: NSView) -> [T] {
        (root as? T).map { [$0] } ?? root.subviews.flatMap { views(type, $0) }
    }
    static func offscreen(_ view: NSView, size: NSSize) -> NSWindow {
        let window = NSWindow(contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.appearance = NSAppearance(named: .darkAqua)
        window.backgroundColor = .clear; window.isOpaque = false
        window.contentView = view
        return window
    }
    /// A window's or view's pixels after its layout has settled (SwiftUI updates and the window's fit included).
    static func render(_ window: NSWindow) async throws -> Data { try await render(try #require(window.contentView)) }
    /// `excluding`: rectangles of the view (controls allowed to change) whose pixels are blanked in both renders.
    static func render(_ view: NSView, excluding: [NSRect] = []) async throws -> Data {
        for _ in 0..<16 {
            view.needsLayout = true; view.layoutSubtreeIfNeeded(); await Task.yield()
        }
        let rep = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: rep)
        for excluding in excluding {
            guard let pointer = rep.bitmapData else { break }
            let scale = CGFloat(rep.pixelsWide) / view.bounds.width
            // Bitmap rows run top-down; a flipped view's y does too.
            let top = view.isFlipped ? excluding.minY : view.bounds.height - excluding.maxY
            let rows = max(0, Int(top * scale))..<min(rep.pixelsHigh, Int(((top + excluding.height) * scale).rounded(.up)))
            let columns = max(0, Int(excluding.minX * scale))..<min(rep.pixelsWide, Int((excluding.maxX * scale).rounded(.up)))
            for y in rows { memset(pointer + y * rep.bytesPerRow + columns.lowerBound * 4, 0, columns.count * 4) }
        }
        let data = Data(bytes: try #require(rep.bitmapData), count: rep.bytesPerRow * rep.pixelsHigh)
        #expect(Set(data).count > 8, "The real hosted component must render visible controls")
        return data
    }
}
