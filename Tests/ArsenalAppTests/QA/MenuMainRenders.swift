// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

/// The four states the main status menu is checked in: a fresh install with nothing on disk, one loaded model with a tool,
/// a last-output row, and an update row.
enum MenuMainState: String, CaseIterable, Sendable {
    case fresh, loaded, lastOutput = "last-output", update

    /// The app as the menu shows it in this state, over its own support folder (no service runs; the fixture answers).
    /// Fresh has nothing on this Mac, so an empty table; the other states have a voice, an image and a chat model downloaded.
    @MainActor static func app(_ state: MenuMainState, support: URL) async throws -> AppDelegate {
        let service = FixtureService(rows: state == .fresh ? Fixtures.rows(.firstLaunch) : MenuTests.downloaded)
        if state == .loaded {
            try await service.get(id: "kokoro-82m", precision: "fp16")
            try await service.load(id: "kokoro-82m", precision: "fp16", fast: true)
        }
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        app.tools.playback = FixturePlayback()
        app.loginStatus = { .notRegistered }
        if state == .lastOutput {
            let last = LastOutput(defaults: UserDefaults(suiteName: "arsenal-main-\(UUID().uuidString)")!)
            last.record(.text("This is my voice."), noun: "Transcript", model: "Kokoro 82M")
            app.lastOutput = last
        }
        if state == .update { app.updater = FixtureUpdater() }
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        return app
    }
}

/// `ARSENAL_MENU_MAIN_RENDERS=<dir>` (optional `ARSENAL_MENU_MAIN_LABEL=<name>` for the widths file): the main status menu
/// in each state at the width AppKit gives it (`NSMenu.size`), as a dark PNG at 2x pixels, plus a widths file that lists
/// each state's width and its rows.
@MainActor @Suite(.menuState, .serialized) struct MenuMainRenders {
    nonisolated static var directory: URL? { ProcessInfo.processInfo.environment["ARSENAL_MENU_MAIN_RENDERS"].map { URL(fileURLWithPath: $0) } }

    @Test(.enabled(if: MenuMainRenders.directory != nil))
    func mainMenuStates() async throws {
        _ = NSApplication.shared
        let directory = try #require(Self.directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let label = ProcessInfo.processInfo.environment["ARSENAL_MENU_MAIN_LABEL"] ?? "current"
        let oldAppearance = Appearance.shared.mode
        defer { Appearance.shared.set(oldAppearance) }
        Appearance.shared.set(.dark)
        var lines: [String] = []
        for state in MenuMainState.allCases {
            let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-main-render-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: support) }
            let app = try await MenuMainState.app(state, support: support)
            _ = app.menu.size
            let width = app.menu.size.width
            lines.append("\(state.rawValue)\twidth \(String(format: "%.1f", width)) pt")
            lines.append(contentsOf: app.menu.items.filter { !$0.isHidden }.map { "  " + ($0.isSeparatorItem ? "—" : $0.title) })
            // The mock draws at the menu's own width; its height follows the rows.
            let mock = MenuMock(items: app.menu.items, width: width)
            mock.setFrameSize(NSSize(width: width, height: mock.frame.height))
            await withCheckedContinuation { done in
                MenuMock.capture(mock, to: directory.appendingPathComponent("main-\(state.rawValue).png")) { done.resume() }
            }
        }
        try (lines.joined(separator: "\n") + "\n").write(
            to: directory.appendingPathComponent("main-menu-widths-\(label).txt"), atomically: true, encoding: .utf8)
    }
}
