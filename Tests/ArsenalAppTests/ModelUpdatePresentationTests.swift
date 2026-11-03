// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) final class ModelUpdatePresentationTests {
    private func row(bytes: Int64 = 0) -> ModelRow {
        var row = Fixtures.figureRows().first { $0.id == "stable-audio-3-medium" }!
        row.states["fp16"] = .init(state: .updateAvailable, phase: "updateAvailable")
        row.downloadBytes["fp16"] = bytes
        row.precisionStats["fp16"] = Fixtures.stats(.sound, "fp16", speed: 0.5, memory: 6.8)
        return row
    }

    @Test(arguments: [Int64(0), Int64(3_000_000)])
    func pendingUpdateKeepsFiguresAndUsesGet(bytes: Int64) async throws {
        let row = row(bytes: bytes), p = RowPresentation(row)
        #expect(p.action == .get && p.updateAvailable)
        #expect(p.actionTitle == (bytes > 0 ? "Update · 3 MB" : "Update"))
        #expect(p.status.tone != .error && p.busyText == nil && !p.locked)
        // The update is the button's to say; the line under the name stays what the model does, the update in its hover.
        #expect(p.line.text == "Sound and music generation" && p.line.help.hasPrefix("Update available\n"))
        #expect(ModelHelp.precision(row, column: "16", presentation: p).contains("Update available"))
        #expect(!ModelHelp.action(row, presentation: p).contains("0 KB"))
        let speed = TableFigures.figure(row, precision: "fp16", column: .speed, host: "M5 Max")
        let memory = TableFigures.figure(row, precision: "fp16", column: .memory, host: "M5 Max")
        #expect(speed.value != nil && memory.value != nil)
        let item = BrowserItem(CatalogEntry(row: row, added: true), kind: KindSpec.defaults.first { $0.id == "sound" })
        #expect(item.downloadText == p.actionTitle)
        #expect(!item.downloadCheck)
        #expect(item.downloadHelp.contains("Update in the Models table"))
        #expect(ModelDetail(item).note?.contains("Update in the Models table") == true)
        #expect(item.peak == memory.value)
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-row-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let service = StubModelsService(rows: [row], supportDirectory: support)
        let controller = TableController(service: service)
        await controller.refresh(); defer { controller.stop() }
        controller.confirmGet = { _, _ in true }
        let action = RowActionView()
        action.spec = RowAction(title: p.actionTitle, enabled: true, deletable: true, help: "Update", onPerform: { controller.perform(row) }, onDelete: {})
        #expect(action.accessibilityPerformPress())
        try await waitUntil { await service.rows().first?.state == .ready }
        #expect(controller.lastError == nil)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_UPDATE_RENDERS"] != nil))
    func renderPendingUpdateDarkAndLight() async throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_UPDATE_RENDERS"]!)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let support = directory.appendingPathComponent("support")
        defer { try? FileManager.default.removeItem(at: support) }
        let previous = Appearance.shared.mode
        defer { Appearance.shared.set(previous); NSApp.appearance = nil }
        let rows = [row()]
        let controller = TableController(service: FixtureService(rows: rows, supportDirectory: support), chip: "M5 Max")
        controller.apply(rows: rows, kinds: KindSpec.defaults.filter { $0.id == "sound" })
        defer { controller.stop() }
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
            let file = directory.appendingPathComponent("update-\(mode.rawValue).png")
            await QA.renderTable(controller, to: file)
            let image = try #require(NSBitmapImageRep(data: Data(contentsOf: file)))
            #expect(image.pixelsWide > 0 && image.pixelsHigh > 0)
        }
    }
}
