// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

/// Renders of the 22 Oct night pass at real size and 2× pixels: Get more models scrolled under its column header, its
/// sidebar with Kind and Input & output folded, the status menu with Copy Last and Support the Developer, and the real
/// Copy Last item view at rest and highlighted. `ARSENAL_NIGHT_RENDERS=<dir>` turns them on.
@MainActor @Suite(.menuState, .serialized) struct NightPassRenders {
    nonisolated static var directory: URL? { ProcessInfo.processInfo.environment["ARSENAL_NIGHT_RENDERS"].map { URL(fileURLWithPath: $0) } }

    /// Enough rows that the list scrolls well past its header.
    static func manyEntries() -> [CatalogEntry] {
        var entries = BrowserTests.entries
        for i in 0..<24 {
            entries.append(
                BrowserTests.entry(
                    "extra-\(i)", ["Pocket TTS", "Kokoro 82M", "Breeze TTS 2", "Von 1.2"][i % 4] + " \(i)", kind: ["voices", "decisions", "chat"][i % 3],
                    params: "\(100 + i)M", bytes: Int64(300_000_000 + i * 10_000_000), downloads: Double(100 - i)))
        }
        return entries
    }

    @Test(.enabled(if: NightPassRenders.directory != nil))
    func getMoreModelsHeaderAndFoldedSidebar() async throws {
        _ = NSApplication.shared
        NSApplication.shared.setActivationPolicy(.accessory)
        let directory = try #require(Self.directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let oldAppearance = Appearance.shared.mode
        let oldSidebar = UserDefaults.standard.object(forKey: BrowserRootView.openKey)
        let oldFolded = UserDefaults.standard.object(forKey: BrowserController.foldedKey)
        defer {
            Appearance.shared.set(oldAppearance)
            UserDefaults.standard.set(oldSidebar, forKey: BrowserRootView.openKey)
            UserDefaults.standard.set(oldFolded, forKey: BrowserController.foldedKey)
        }
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode)
            for scene in ["header-scrolled", "header-scrolled-collapsed", "sidebar-folded"] {
                UserDefaults.standard.set(scene != "header-scrolled-collapsed", forKey: BrowserRootView.openKey)
                UserDefaults.standard.removeObject(forKey: BrowserController.foldedKey)
                let controller = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
                controller.apply(entries: Self.manyEntries(), kinds: BrowserTests.kinds)
                if scene == "sidebar-folded" {
                    controller.toggleFold(.kinds); controller.toggleFold(.modality)
                    controller.state.kind = "voices"
                }
                let windowController = BrowserWindowController(controller: controller)
                let window = try #require(windowController.window)
                window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
                window.orderFrontRegardless()
                try await Task.sleep(for: .milliseconds(300))
                if scene.hasPrefix("header") {
                    // Rows scrolled up under the header: none may read through it.
                    let clip = windowController.list.scrollView.contentView
                    clip.scroll(to: NSPoint(x: 0, y: 7 * 30 + 13))
                    windowController.list.scrollView.reflectScrolledClipView(clip)
                }
                try await Task.sleep(for: .milliseconds(250))
                try PolishGetMoreRenders.capture(window, to: directory.appendingPathComponent("night-\(scene)-\(mode.rawValue).png"))
                window.close()
            }
        }
    }

    @Test(.enabled(if: NightPassRenders.directory != nil))
    func menuWithCopyLastAndSupport() async throws {
        _ = NSApplication.shared
        let directory = try #require(Self.directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-night-render-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        let defaults = try #require(UserDefaults(suiteName: "arsenal-night-render-\(UUID().uuidString)"))
        let rows = Fixtures.rows(.loaded)
        let app = AppDelegate(service: FixtureService(rows: rows), paths: ArsenalPaths(support: support))
        app.table.apply(rows: rows, kinds: Fixtures.kinds)
        let last = LastOutput(defaults: defaults)
        let image = support.appendingPathComponent("2026-10-22 0312 flux2-klein-4b.png")
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: image)
        last.record(file: image, mime: "image/png", model: "FLUX.2 Klein 4B")
        app.lastOutput = last
        app.rebuildMenu()
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode)
            await withCheckedContinuation { done in
                MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("night-menu-\(mode.rawValue).png")) { done.resume() }
            }
            // The real item view, at rest (lighter row) and highlighted, on the menu's stand-in material.
            let copy = try #require(app.menu.items.first { $0.title == "Copy Last Image" })
            #expect(copy.view is ProminentMenuItemView)
            let restItem = try #require(copy.copy() as? NSMenuItem)
            let rest = ProminentMenuItemView(item: restItem)
            let litItem = try #require(copy.copy() as? NSMenuItem)
            let lit = ProminentMenuItemView(item: litItem)
            lit.forcesHighlight = true; lit.rendersSelectionFill = true
            for (state, view) in [("rest", rest), ("highlighted", lit)] {
                let container = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: ProminentMenuItemView.height + 12))
                container.wantsLayer = true
                MenuMock.applyMaterial(to: container.layer, dark: mode == .dark, picture: nil)
                view.frame = NSRect(x: 0, y: 6, width: 300, height: ProminentMenuItemView.height)
                container.addSubview(view)
                container.appearance = Appearance.shared.nsAppearance
                await withCheckedContinuation { done in
                    MenuMock.capture(container, to: directory.appendingPathComponent("night-copy-item-\(state)-\(mode.rawValue).png")) { done.resume() }
                }
            }
            withExtendedLifetime((litItem, restItem)) {}
        }
    }
}
