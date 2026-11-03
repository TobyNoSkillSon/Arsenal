// SPDX-License-Identifier: MIT
import AppKit
import Testing
@testable import Arsenal
import ArsenalCore

/// Get more models' expanded detail with the Docs popover open on a real model README, both appearances.
@MainActor @Suite(.menuState, .serialized) struct ReadmeRenders {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_README_RENDERS"] != nil))
    func docsPopoverOverTheExpandedDetail() async throws {
        _ = NSApplication.shared
        NSApplication.shared.setActivationPolicy(.accessory)
        let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["ARSENAL_README_RENDERS"]))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let readme = try String(contentsOf: repo.appendingPathComponent("models/birefnet/README.md"), encoding: .utf8)
        let oldAppearance = Appearance.shared.mode
        defer { Appearance.shared.set(oldAppearance) }
        for mode in [RenderAppearance.dark, .light] {
            Appearance.shared.set(mode)
            let controller = BrowserController(service: FixtureService(rows: []), chip: "Fixture Mac")
            controller.apply(entries: BrowserTests.entries, kinds: BrowserTests.kinds)
            controller.expanded = ["kokoro"]
            let windowController = BrowserWindowController(controller: controller)
            let window = try #require(windowController.window)
            window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
            windowController.list.expansionChanged()
            window.orderFrontRegardless()
            try await Task.sleep(for: .milliseconds(350))
            try PolishGetMoreRenders.capture(window, to: directory.appendingPathComponent("readme-detail-\(mode.rawValue).png"))
            let button = try #require(Self.find(window.contentView))
            button.load = { readme }
            button.toggle()
            try await Task.sleep(for: .milliseconds(500))
            #expect(button.isShowingReadme)
            let ground = try #require(button.readmeContent as? FillView)
            try Self.capture(ground, to: directory.appendingPathComponent("readme-popover-\(mode.rawValue).png"))
            window.close()
        }
    }

    /// The ground's fill, then the README drawn over it: what AppKit composites inside the popover.
    static func capture(_ ground: FillView, to url: URL) throws {
        ground.layoutSubtreeIfNeeded(); ground.displayIfNeeded()
        let hosting = try #require(ground.subviews.first)
        let rep = try #require(ground.bitmapImageRepForCachingDisplay(in: ground.bounds))
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        ground.color.setFill(); ground.bounds.fill()
        NSGraphicsContext.restoreGraphicsState()
        let content = try #require(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
        hosting.cacheDisplay(in: hosting.bounds, to: content)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        content.draw(in: hosting.frame, from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
        NSGraphicsContext.restoreGraphicsState()
        try #require(rep.representation(using: .png, properties: [:])).write(to: url)
    }

    static func find(_ view: NSView?) -> ModelReadmeButton.ReadmeButton? {
        guard let view else { return nil }
        if let button = view as? ModelReadmeButton.ReadmeButton { return button }
        for sub in view.subviews { if let found = find(sub) { return found } }
        return nil
    }
}
