// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal
import ArsenalCore

/// The table-holds-weights rule at real size: the Models table read through the service (models added without weights
/// stay out; every precision on disk shows with its size), Get more models with the same models, and Remove's delete
/// alert. `ARSENAL_TABLE_WEIGHTS_RENDERS=<dir>` writes table-weights-*.png. No screen interaction: drawn offscreen.
@MainActor @Suite(.menuState, .serialized) struct HeldWeightsRenders {
    /// Kokoro with two precisions on disk (one loaded), FLUX downloading, Qwen with only int4 on disk, and Laya added
    /// without weights: the table shows the first three.
    static func service() async -> FixtureService {
        var rows = Fixtures.rows(.loaded)
        if let i = rows.firstIndex(where: { $0.id == "flux2-klein-4b" }) {
            rows[i].states["bf16"] = PrecisionState(state: .downloading, progress: 0.23, bytes: 3_675_000_000, totalBytes: 15_980_140_843)
        }
        if let i = rows.firstIndex(where: { $0.id == "qwen3.8-27b" }) {
            rows[i].states = ["int4": PrecisionState(state: .ready)]; rows[i].selectedPrecision = "int4"
        }
        var laya = ModelRow(id: "laya-english", name: "Laya English", kind: "decisions", params: "395M", precisions: ["fp16"], selectedPrecision: "fp16")
        laya.purpose = "English decisions"
        let service = FixtureService(rows: rows + [laya])
        await service.setCatalogSizes([
            "kokoro-82m": ["fp16": 341_742_463, "int8": 181_550_000], "flux2-klein-4b": ["bf16": 15_980_140_843],
            "qwen3.8-27b": ["bf16": 55_600_000_000, "int8": 29_500_000_000, "int4": 15_600_000_000], "laya-english": ["fp16": 807_000_000]
        ])
        return service
    }

    @Test func tableReadsOnlyModelsWithWeights() async throws {
        let table = TableController(service: await Self.service(), chip: "M5 Max")
        await table.refresh()
        #expect(table.rows.map(\.id).sorted() == ["flux2-klein-4b", "kokoro-82m", "qwen3.8-27b"])
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_TABLE_WEIGHTS_RENDERS"] != nil))
    func renders() async throws {
        _ = NSApplication.shared
        NSApplication.shared.setActivationPolicy(.accessory)
        let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ARSENAL_TABLE_WEIGHTS_RENDERS"]))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let oldAppearance = Appearance.shared.mode
        defer { Appearance.shared.set(oldAppearance) }
        Appearance.shared.set(.dark)
        let service = await Self.service()

        let table = TableController(service: service, chip: "M5 Max")
        await table.refresh()
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            RenderDelegate.renderTable(table, to: directory.appendingPathComponent("table-weights-table.png")) { done.resume() }
        }

        let browser = BrowserController(service: service, chip: "M5 Max")
        await browser.refresh()
        let windowController = BrowserWindowController(controller: browser)
        let window = try #require(windowController.window)
        window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
        window.orderFrontRegardless()
        try await Task.sleep(for: .milliseconds(350))
        try PolishGetMoreRenders.capture(window, to: directory.appendingPathComponent("table-weights-get-more.png"))
        window.close()

        let kokoro = try #require(table.rows.first { $0.id == "kokoro-82m" })
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.capture(AlertMock(ModelsMenu.removeAlert(kokoro)), to: directory.appendingPathComponent("table-weights-remove-alert.png")) { done.resume() }
        }
    }
}
