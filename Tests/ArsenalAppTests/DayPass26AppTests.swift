// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

/// The 25 Oct requests in the app: the catalog's light-green row for models on the table, and re-approval and Revoke of a
/// changed local model's code.
@MainActor @Suite(.serialized) struct DayPass26AppTests {
    private func entries() -> [CatalogEntry] {
        var loaded = BrowserTests.entry("kokoro", "Kokoro 82M", kind: "voices", params: "82M", bytes: 342_000_000, added: true, installed: true)
        loaded.row.states["bf16"] = .init(state: .loaded); loaded.loaded = true
        return [
            BrowserTests.entry("von", "Von 1.2", kind: "decisions", params: "395M", bytes: 1_600_000_000),
            BrowserTests.entry("qwen", "Qwen3.8 27B", kind: "chat", params: "27B", bytes: 55_600_000_000, added: true), loaded
        ]
    }

    /// A model on the table gets the light-green wash; a loaded one keeps the blue (and no green); others stay plain.
    @Test func modelsOnTheTableGetTheGreenRowAndLoadedOnesKeepTheBlue() throws {
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        c.apply(entries: entries(), kinds: BrowserTests.kinds)
        let list = BrowserList(); list.controller = c
        list.show(c.shown)
        var states: [String: (added: Bool, loaded: Bool)] = [:]
        for (index, line) in list.lines.enumerated() {
            guard case .item(let id) = line, let view = list.tableView(list.table, rowViewForRow: index) as? BrowserRowView else { continue }
            states[id] = (view.added, view.loaded)
        }
        #expect(states.count == 3)
        #expect(states["qwen"]?.added == true && states["qwen"]?.loaded == false)
        #expect(states["kokoro"]?.added == false && states["kokoro"]?.loaded == true)
        #expect(states["von"]?.added == false && states["von"]?.loaded == false)
        // Green, never the loaded blue, in both appearances.
        for palette in [ArsenalPalette.dark, .light, .translucent] {
            let green = try #require(palette.addedRow.usingColorSpace(.sRGB))
            #expect(green.greenComponent > green.redComponent && green.greenComponent > green.blueComponent && green.alphaComponent < 0.3)
            #expect(palette.addedRow != palette.hotRow)
        }
    }

    /// The approval prompt names the new contents and, for a changed folder, the ones approved before.
    @Test func theCodePromptShowsTheNewDigestBesideTheOldOne() {
        var row = ModelRow(id: "mine", name: "Mine", kind: "voices", precisions: ["fp16"], selectedPrecision: "fp16")
        row.local = true; row.runnerLanguage = "swift"; row.contentDigest = "b2c3d4e5f60718293a4b"
        row.approval = CodeApproval(by: .app, digest: "a1b2c3d4e5f6")
        let changed = ModelsMenu.codeAlert(row).informativeText
        #expect(changed.contains("changed since you approved contents a1b2c3d4") && changed.contains("Contents b2c3d4e5"), "\(changed)")
        row.approval = nil
        let first = ModelsMenu.codeAlert(row).informativeText
        #expect(!first.contains("changed since") && first.contains("Contents b2c3d4e5"))
    }

    /// The settings card offers Revoke for an approved local model only; it forgets the approval through the service.
    @Test func revokeShowsForApprovedLocalModelsAndAsksAgainAfter() async throws {
        var row = ModelRow(id: "mine", name: "Mine", kind: "voices", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        row.local = true; row.runner = "native"; row.contentDigest = String(repeating: "c", count: 64)
        row.approval = CodeApproval(by: .app, digest: String(repeating: "c", count: 64))
        #expect(SettingsPane.revocable(row))
        var catalog = row; catalog.local = false
        #expect(!SettingsPane.revocable(catalog))
        var pending = row; pending.needsCodeConsent = true
        #expect(!SettingsPane.revocable(pending))
        #expect(ModelHelp.revokeHelp(row).contains("cccccccc") && ModelHelp.revokeHelp(row).contains("asks again"))
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-revoke-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        let service = FixtureService(rows: [row], supportDirectory: support)
        let table = TableController(service: service)
        await table.refresh()
        table.revokeCode(try #require(table.rows.first))
        try await waitUntil { table.rows.first?.needsCodeConsent == true }
        #expect(table.rows.first?.approval == nil)
        #expect(await service.calls.contains("revoke mine"))
    }
}
