// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI
import Testing
@testable import Arsenal

/// 22 Oct night pass: Copy Last (with Recent below it), Support the Developer, the folded sidebar groups, Fits on any Mac, the opaque Get more
/// models header and the busy row's sweep.
@MainActor @Suite(.menuState, .serialized) struct NightPassTests {
    static func defaults() throws -> UserDefaults { try #require(UserDefaults(suiteName: "arsenal-night-\(UUID().uuidString)")) }

    @Test func lastOutputNamesCopiesAndForgetsAMovedFile() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-last-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let defaults = try Self.defaults()
        let textFile = folder.appendingPathComponent("state/last-output.txt")
        let last = LastOutput(defaults: defaults, textFile: textFile)
        var changes = 0
        last.changed = { changes += 1 }
        #expect(last.title == nil)
        let sound = folder.appendingPathComponent("take.wav")
        try Data("RIFF".utf8).write(to: sound)
        last.record(file: sound, model: "Stable Audio 3")
        #expect(last.title == "Copy Last Sound" && changes == 1)
        // Kept across launches.
        #expect(LastOutput(defaults: defaults).title == "Copy Last Sound")
        let pasteboard = NSPasteboard(name: .init("arsenal-night-\(UUID().uuidString)"))
        #expect(last.copy(to: pasteboard))
        // Text results copy as text; empty text is not a result.
        last.record(.text("Hello there"), noun: "Transcript", model: "Parakeet")
        #expect(last.title == "Copy Last Transcript" && last.copy(to: pasteboard) && pasteboard.string(forType: .string) == "Hello there")
        // The text lives in its file, not in preferences, and comes back at the next launch.
        #expect(String(data: defaults.data(forKey: LastOutput.defaultsKey) ?? Data(), encoding: .utf8)?.contains("Hello there") == false)
        let relaunched = LastOutput(defaults: defaults, textFile: textFile)
        #expect(relaunched.copy(to: pasteboard) && pasteboard.string(forType: .string) == "Hello there")
        last.record(.text("  \n"), noun: "Transcript", model: "Parakeet")
        #expect(last.title == "Copy Last Transcript" && changes == 2)
        // A file moved away offers nothing.
        let image = folder.appendingPathComponent("photo.png")
        try Data([0x89]).write(to: image)
        last.record(file: image, mime: "image/png", model: "FLUX.2")
        #expect(last.title == "Copy Last Image")
        try FileManager.default.removeItem(at: image)
        #expect(last.title == nil && !last.copy(to: pasteboard))
        #expect(LastOutput.noun(file: URL(fileURLWithPath: "/x/mesh.glb"), mime: nil) == "3D Model")
        #expect(LastOutput.noun(file: URL(fileURLWithPath: "/x/clip.mov"), mime: nil) == "Video")
    }

    @Test func theMenuOffersCopyLastAndRecentAndSupportAboveQuit() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-night-menu-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.firstLaunch)), paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        #expect(!app.menu.items.contains { $0.title.hasPrefix("Copy Last") })
        let titles = app.menu.items.map(\.title)
        // Support sits above Quit, as Vella's.
        #expect(titles.suffix(2) == [AppDelegate.supportTitle, "Quit Arsenal"])
        let supportItem = try #require(app.menu.items.dropLast().last)
        #expect(supportItem.image != nil && supportItem.keyEquivalent.isEmpty && supportItem.target === app)
        #expect(AppDelegate.supportURL.absoluteString == "https://tobynoskillson.com/#support")

        // The results group: Copy Last <noun> on the lighter row, then Recent, after the tasks and above Models, only
        // once there is a result.
        #expect(!app.menu.items.contains { $0.title == "Recent" })
        let last = LastOutput(defaults: try Self.defaults())
        app.lastOutput = last
        last.record(.text("Bonjour"), noun: "Translation", model: "Index Translate 2B")
        let copy = try #require(app.menu.items.first { $0.title == "Copy Last Translation" })
        #expect(copy.view is ProminentMenuItemView && copy.toolTip?.contains("Bonjour") == true)
        let index = try #require(app.menu.items.firstIndex(of: copy))
        let recent = try #require(app.menu.items.first { $0.title == "Recent" })
        #expect(app.menu.items[index - 1].isSeparatorItem && app.menu.items[index + 1] === recent)
        #expect(app.menu.items[index + 2].isSeparatorItem && app.menu.items[index + 3].title == "Models")
        let entry = try #require(recent.submenu?.items.first)
        #expect(entry.title == "Translation \u{00b7} Index Translate 2B" && entry.subtitle == "Bonjour")
        // A text result offers Copy and Save As…; Open and Show in Finder are for files.
        #expect(entry.submenu?.items.map(\.title) == ["Copy", "Save As\u{2026}"])
        // A new result renames Copy Last in place and joins Recent at the top; Recent keeps one text result, the newest.
        last.record(.text("Hi"), noun: "Transcript", model: "Parakeet")
        #expect(app.menu.items.contains { $0.title == "Copy Last Transcript" } && !app.menu.items.contains { $0.title == "Copy Last Translation" })
        #expect(app.menu.items.first { $0.title == "Recent" } === recent)
        #expect(recent.submenu?.items.map(\.title) == ["Transcript \u{00b7} Parakeet"])
        // With the menu open an item is renamed in place, and a lighter row's view widens to the new title and redraws.
        let menu = NSMenu()
        let live = NSMenuItem(title: "Copy Last Image", action: #selector(NSText.copy(_:)), keyEquivalent: "")
        menu.addItem(live)
        let view = ProminentMenuItemView(item: live)
        let narrow = view.frame.width
        // Shown in a window (outside one AppKit keeps no redraw state).
        let host = NSWindow(contentRect: NSRect(x: -9000, y: -9000, width: 600, height: 40), styleMask: .borderless, backing: .buffered, defer: false)
        host.isReleasedWhenClosed = false
        defer { host.close() }
        host.contentView = NSView(frame: NSRect(x: 0, y: 0, width: 600, height: 40))
        host.contentView?.addSubview(view)
        host.orderFrontRegardless()
        view.display()
        let snapshot = NSMenuItem(title: "Copy Last Extraction From A Long Document", action: #selector(NSText.copy(_:)), keyEquivalent: "")
        #expect(MenuReconciliation.apply([snapshot], to: menu, structure: false))
        #expect(menu.items.first === live && live.title == snapshot.title && live.view === view)
        #expect(view.frame.width > narrow)
        #expect(view.needsDisplay)
    }

    /// Recent lists the newest results first, at most five: each file offers Open, Copy, Save As… and Show in Finder, each
    /// targeting its own result; a file moved away leaves the list; Copy and Save As… deliver that result.
    @Test func recentListsTheNewestFiveFirstWithTheirActions() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-recent-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.firstLaunch)), paths: ArsenalPaths(support: folder.appendingPathComponent("support")))
        await app.table.refresh(); await app.refreshMenuState()
        let defaults = try Self.defaults()
        let last = LastOutput(defaults: defaults)
        app.lastOutput = last
        var files: [URL] = []
        for n in 1...7 {
            let file = folder.appendingPathComponent("take-\(n).wav")
            try Data("RIFF\(n)".utf8).write(to: file)
            files.append(file)
            last.record(file: file, model: "Model \(n)")
        }
        app.rebuildMenu()
        let recent = try #require(app.menu.recent)
        #expect(LastOutput.recentCount == 5)
        #expect(recent.items.map(\.subtitle) == ["take-7.wav", "take-6.wav", "take-5.wav", "take-4.wav", "take-3.wav"])
        #expect(recent.items.map(\.title) == (3...7).reversed().map { "Sound \u{00b7} Model \($0)" })
        for (index, item) in recent.items.enumerated() {
            // An entry opens its actions (AppKit gives a submenu's item its own submenu action).
            #expect(item.image != nil && item.submenu != nil)
            let actions = try #require(item.submenu)
            #expect(actions.items.map(\.title) == ["Open", "Copy", "Save As\u{2026}", "Show in Finder"])
            #expect(actions.items.allSatisfy { $0.target === app && $0.action != nil && $0.tag == index && $0.isEnabled })
        }
        // The same result again moves to the top, once.
        last.record(file: files[3], model: "Model 4")
        app.rebuildMenu()
        #expect(try #require(app.menu.recent).items.map(\.subtitle) == ["take-4.wav", "take-7.wav", "take-6.wav", "take-5.wav", "take-3.wav"])
        // A file moved away leaves the list.
        try FileManager.default.removeItem(at: files[6])
        app.rebuildMenu()
        #expect(try #require(app.menu.recent).items.map(\.subtitle) == ["take-4.wav", "take-6.wav", "take-5.wav", "take-3.wav"])
        // Copy and Save As… act on the entry at that position (the second: take-6).
        let pasteboard = NSPasteboard(name: .init("arsenal-recent-\(UUID().uuidString)"))
        defer { pasteboard.releaseGlobally() }
        #expect(last.copy(1, to: pasteboard))
        #expect(pasteboard.types?.isEmpty == false)
        let target = folder.appendingPathComponent("saved.wav")
        #expect(LastOutput.write(try #require(last.recent[1]).content, to: target))
        #expect(try Data(contentsOf: target) == Data("RIFF6".utf8))
        // A file kept by an older build with Copy Last only (no Recent list) heads Recent after the upgrade.
        let legacy = try Self.defaults()
        legacy.set(try JSONEncoder().encode(LastOutput.Item(content: .file(files[5]), noun: "Sound", model: "Model 6", made: Date())), forKey: LastOutput.defaultsKey)
        #expect(LastOutput(defaults: legacy).recent.map(\.content) == [.file(files[5])])
        // Kept across launches (files), newest first.
        #expect(LastOutput(defaults: defaults).recent.map(\.content) == last.recent.map(\.content))
        // Each action names its result, not its row: a row's position may shift while the menu is open.
        #expect(try #require(app.menu.recent).items.allSatisfy { $0.submenu?.items.allSatisfy { $0.representedObject is Date } == true })
        // The newest result's file gone: Copy Last goes, Recent stays with the others.
        try FileManager.default.removeItem(at: files[3])
        app.rebuildMenu()
        #expect(!app.menu.items.contains { $0.title.hasPrefix("Copy Last") && !$0.isHidden })
        #expect(try #require(app.menu.recent).items.map(\.subtitle) == ["take-6.wav", "take-5.wav", "take-3.wav"])
    }

    @Test func sidebarGroupsFoldAndRemember() throws {
        let old = UserDefaults.standard.object(forKey: BrowserController.foldedKey)
        defer { UserDefaults.standard.set(old, forKey: BrowserController.foldedKey) }
        UserDefaults.standard.removeObject(forKey: BrowserController.foldedKey)
        let c = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        #expect(c.foldedGroups.isEmpty)
        c.toggleFold(.kinds)
        #expect(c.foldedGroups == [.kinds])
        #expect(BrowserController(service: FixtureService(rows: []), chip: "M5 Max").foldedGroups == [.kinds])
        c.toggleFold(.kinds)
        #expect(c.foldedGroups.isEmpty && UserDefaults.standard.stringArray(forKey: BrowserController.foldedKey) == [])
    }

    @Test func fitsFollowsThisMacsLine() {
        let small = BrowserController(service: FixtureService(rows: []), physicalBytes: 16 * 1_073_741_824, chip: "M1")
        #expect(small.fitsHelp.contains("12 GB Arsenal uses on this 16 GB Mac"))
        let big = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        #expect(big.fitsHelp.contains("100 GB Arsenal uses on this 128 GB Mac"))
        // A model needing 13 GB fits a 128 GB Mac but not a 16 GB one (its line is 12.5 GiB).
        #expect(CatalogMatch.fits(ramMB: 13_000, physicalBytes: 128 * 1_073_741_824))
        #expect(!CatalogMatch.fits(ramMB: 13_000, physicalBytes: 16 * 1_073_741_824))
    }

    @Test func getMoreModelsHeaderHasItsOwnGroundAndTakesClicks() async throws {
        let controller = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        controller.apply(entries: BrowserTests.entries, kinds: BrowserTests.kinds)
        let windowController = BrowserWindowController(controller: controller)
        defer { windowController.window?.close() }
        let header = windowController.list.header
        header.frame = NSRect(x: 0, y: 0, width: 600, height: 30)
        header.layoutSubtreeIfNeeded()
        // The ground and titles are subviews spanning the header; a click lands on the header itself (sort, drag).
        #expect(header.subviews.count == 2 && header.subviews.allSatisfy { $0.frame == header.bounds })
        #expect(header.hitTest(NSPoint(x: 100, y: 15)) === header)
        // Opaque palettes paint the ground themselves; the translucent one shows the window's material behind it.
        #expect(Appearance.shared.palette.canvas.alphaComponent == 1 || Appearance.shared.palette.translucent)
        // A column resize invalidates a rect, not the property: the titles redraw with the new column widths.
        let window = try #require(windowController.window)
        window.setFrameOrigin(NSPoint(x: -9000, y: -9000)); window.orderFrontRegardless()
        window.displayIfNeeded()
        for _ in 0..<30 where header.titleDraws == 0 { try await Task.sleep(for: .milliseconds(100)) }
        let before = header.titleDraws
        #expect(before > 0)
        // A width change is saved as the user's column weights: put them back for the suites that follow.
        let savedWeights = UserDefaults.standard.object(forKey: BrowserColumns.weightsKey)
        defer { UserDefaults.standard.set(savedWeights, forKey: BrowserColumns.weightsKey) }
        let column = try #require(windowController.list.table.tableColumns.first { !$0.isHidden && $0.resizingMask.contains(.userResizingMask) })
        column.width += 24
        window.displayIfNeeded()
        // The redraw lands on a Core Animation commit; allow a busy machine a few seconds.
        for _ in 0..<30 where header.titleDraws == before { try await Task.sleep(for: .milliseconds(100)) }
        #expect(header.titleDraws > before)
    }

    @Test func aBusyRowSweepsAndARestingRowDrawsNothing() async throws {
        let animated = SheenView.animates
        SheenView.animates = true
        defer { SheenView.animates = animated }
        let view = SheenView(frame: NSRect(x: 0, y: 0, width: 400, height: 49))
        #expect(!view.sweeping)
        let window = NSWindow(contentRect: view.frame, styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.layoutSubtreeIfNeeded()
        view.needsLayout = true; view.layout()
        #expect(view.sweeping == !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion)
        window.close()
        // The table hosts one sweep per busy row, and none at rest.
        func sheens(_ scene: Fixtures.Scene) async throws -> (busy: Int, views: Int) {
            let rows = Fixtures.rows(scene)
            let c = TableController(service: FixtureService(rows: rows), chip: "M5 Max")
            c.apply(rows: rows, kinds: Fixtures.kinds)
            let host = NSHostingView(rootView: ModelsMenuView(controller: c))
            host.frame = NSRect(x: 0, y: 0, width: ModelTable.width(c), height: ModelTable.height(c))
            let w = NSWindow(contentRect: host.frame, styleMask: .borderless, backing: .buffered, defer: false)
            w.isReleasedWhenClosed = false
            w.contentView = host
            host.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(100))
            func count(_ v: NSView) -> Int { (v is SheenView ? 1 : 0) + v.subviews.reduce(0) { $0 + count($1) } }
            defer { w.close() }
            return (c.rows.filter { c.presentation($0).busyText != nil }.count, count(host))
        }
        let downloading = try await sheens(.downloading)
        #expect(downloading.busy > 0 && downloading.views == downloading.busy)
        let loaded = try await sheens(.loaded)
        #expect(loaded.busy == 0 && loaded.views == 0)
    }
}
