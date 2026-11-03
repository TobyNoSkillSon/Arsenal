// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

/// Renders of the 26 Oct day pass: the main menu without "Get a model" rows (fresh and loaded), Get more models with rows
/// on the table in light green beside a loaded blue row (dark and light), and a local model's settings card with Revoke.
/// `ARSENAL_DAY26_RENDERS=<dir>` turns them on.
@MainActor @Suite(.menuState, .serialized) struct DayPass26Renders {
    nonisolated static var directory: URL? { ProcessInfo.processInfo.environment["ARSENAL_DAY26_RENDERS"].map { URL(fileURLWithPath: $0) } }

    @Test(.enabled(if: DayPass26Renders.directory != nil))
    func menuCatalogueAndRevoke() async throws {
        _ = NSApplication.shared
        NSApplication.shared.setActivationPolicy(.accessory)
        let directory = try #require(Self.directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let oldAppearance = Appearance.shared.mode
        let oldSidebar = UserDefaults.standard.object(forKey: BrowserRootView.openKey)
        defer {
            Appearance.shared.set(oldAppearance)
            UserDefaults.standard.set(oldSidebar, forKey: BrowserRootView.openKey)
        }
        Appearance.shared.set(.dark)
        for state in [MenuMainState.fresh, .loaded] {
            let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-day26-render-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: support) }
            let app = try await MenuMainState.app(state, support: support)
            let width = app.menu.size.width
            await withCheckedContinuation { done in
                MenuMock.render(app.menu.items, width: width, to: directory.appendingPathComponent("day26-menu-\(state.rawValue).png")) { done.resume() }
            }
        }
        var entries = BrowserTests.entries.filter { $0.id != "giant" }
        var loaded = BrowserTests.entry("pocket", "Pocket TTS", kind: "voices", params: "100M", bytes: 240_000_000, downloads: 3000, added: true, installed: true)
        loaded.row.states["bf16"] = .init(state: .loaded); loaded.loaded = true
        entries.append(loaded)
        entries.append(BrowserTests.entry("laya", "Laya · English", kind: "decisions", params: "395M", bytes: 846_000_000, downloads: 700, added: true, installed: true))
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode)
            UserDefaults.standard.set(true, forKey: BrowserRootView.openKey)
            let controller = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
            controller.apply(entries: entries, kinds: BrowserTests.kinds)
            let windowController = BrowserWindowController(controller: controller)
            let window = try #require(windowController.window)
            window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
            window.orderFrontRegardless()
            try await Task.sleep(for: .milliseconds(400))
            try PolishGetMoreRenders.capture(window, to: directory.appendingPathComponent("day26-catalogue-\(mode.rawValue).png"))
            window.close()
        }
        Appearance.shared.set(.dark)
        var row = ModelRow(
            id: "ming-tts-tiny", name: "Ming 3.0 TTS Tiny", kind: "voices", params: "0.5B", precisions: ["fp16"], states: ["fp16": .init(state: .ready)],
            selectedPrecision: "fp16")
        row.local = true; row.runner = "native"; row.contentDigest = String(repeating: "7", count: 64); row.tools = ["voice"]
        row.approval = CodeApproval(by: .app, digest: String(repeating: "7", count: 64))
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-day26-card-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        let table = TableController(service: FixtureService(rows: [row], supportDirectory: support))
        await table.refresh()
        _ = try await SettingsCardRender.render(table, row: try #require(table.rows.first), to: directory.appendingPathComponent("day26-settings-revoke.png"))
    }
}
