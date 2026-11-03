// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) struct PolishGetMoreRenders {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_GETMORE_RENDERS"] != nil))
    func browserFirstFramesAndFeatureHover() async throws {
        _ = NSApplication.shared
        NSApplication.shared.setActivationPolicy(.accessory)
        let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ARSENAL_GETMORE_RENDERS"]))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let oldAppearance = Appearance.shared.mode
        let oldSidebar = UserDefaults.standard.object(forKey: BrowserRootView.openKey)
        let oldFrame = UserDefaults.standard.object(forKey: "NSWindow Frame GetMoreModels")
        defer {
            Appearance.shared.set(oldAppearance)
            UserDefaults.standard.set(oldSidebar, forKey: BrowserRootView.openKey)
            UserDefaults.standard.set(oldFrame, forKey: "NSWindow Frame GetMoreModels")
        }
        UserDefaults.standard.set(true, forKey: BrowserRootView.openKey)
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode)
            for scenario in ["collapsed", "one-runner", "two-runners", "overridden", "minimum-width"] {
                UserDefaults.standard.set(true, forKey: BrowserRootView.openKey)
                let controller = BrowserController(service: FixtureService(rows: []), chip: "Fixture Mac")
                var entries = BrowserTests.entries
                let index = try #require(entries.firstIndex { $0.id == "kokoro" })
                entries[index].description = "Lightweight multilingual speech synthesis with natural voices."
                var native = CatalogRunner(name: "native", language: "swift", engine: "voices", acceptedDefault: true, selected: true)
                native.features = [.batching, .streaming]
                var sequential = CatalogRunner(name: "sequential-fixture", language: "swift", engine: "voices")
                sequential.features = [.sequentialOnly]
                entries[index].row.runner = "native"; entries[index].row.runnerLanguage = "swift"; entries[index].row.defaultRunner = "native"
                if scenario == "overridden" {
                    native.selected = false; sequential.selected = true
                    entries[index].row.runner = sequential.name
                }
                entries[index].runners = ["two-runners", "overridden"].contains(scenario) ? [native, sequential] : [native]
                controller.apply(entries: entries, kinds: BrowserTests.kinds)
                if scenario != "collapsed" { controller.expanded = ["kokoro"] }
                let windowController = BrowserWindowController(controller: controller)
                let window = try #require(windowController.window)
                let size = scenario == "minimum-width" ? BrowserWindowController.minSize : BrowserWindowController.defaultSize
                window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
                windowController.list.expansionChanged()
                window.orderFrontRegardless()
                try await Task.sleep(for: .milliseconds(350))
                try Self.capture(window, to: directory.appendingPathComponent("polish-getmore-\(scenario)-\(mode.rawValue).png"))
                window.close()
            }
        }
    }

    static func capture(_ window: NSWindow, to url: URL) throws {
        let image = try #require(WindowSnapshot.capture(window))
        let data = try #require(image.representation(using: .png, properties: [:]))
        try data.write(to: url)
        print("Render: \(url.lastPathComponent), \(image.pixelsWide) × \(image.pixelsHigh)")
    }
}
