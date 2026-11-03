// SPDX-License-Identifier: MIT
import AppKit
import CryptoKit
import Foundation
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

/// Timing probes for the app's hot paths (docs/notes/pol-ui.md records the numbers). Off by default: they print
/// figures rather than assert them, and need a quiet machine to mean anything. Run with
/// `ARSENAL_PERF=1 scripts/mlock work pol-ui-perf -- scripts/test.sh --filter PerformanceProbe`.
@MainActor @Suite(.menuState, .serialized, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_PERF"] == "1"))
final class PerformanceProbe {
    private let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-pol-ui-perf-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: root) }
    private static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    /// Core's real service over every catalog folder of this checkout, dev-signed, in a private support dir.
    private func realService() async throws -> any ModelsService {
        let resources = root.appendingPathComponent("Resources")
        let fm = FileManager.default
        let kinds = resources.appendingPathComponent("kinds"), models = resources.appendingPathComponent("models")
        try fm.createDirectory(at: models, withIntermediateDirectories: true)
        try fm.copyItem(at: Self.repo.appendingPathComponent("Resources/kinds"), to: kinds)
        func inventory(_ dir: URL) throws -> [CatalogFile] {
            try SafePath.files(in: dir).map {
                CatalogFile(path: $0, sha256: try FileDigest.sha256(dir.appendingPathComponent($0)), bytes: try FileDigest.bytes(dir.appendingPathComponent($0)))
            }
        }
        let ids = try fm.contentsOfDirectory(atPath: Self.repo.appendingPathComponent("models").path).filter { !$0.hasPrefix(".") }.sorted()
        var entries: [CatalogModel] = []
        for id in ids {
            let folder = models.appendingPathComponent(id)
            try fm.copyItem(at: Self.repo.appendingPathComponent("models/\(id)"), to: folder)
            guard let manifest = try? ModelManifest.load(folder, kindsDirectory: kinds) else { try? fm.removeItem(at: folder); continue }
            entries.append(CatalogModel(manifest: manifest, files: try inventory(folder)))
        }
        let data = try CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: entries, kindFiles: try inventory(kinds)).encoded()
        let key = Curve25519.Signing.PrivateKey()
        try fm.createDirectory(at: resources.appendingPathComponent("catalog"), withIntermediateDirectories: true)
        try data.write(to: resources.appendingPathComponent("catalog/catalog.json"))
        try key.signature(for: data).write(to: resources.appendingPathComponent("catalog/catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        let started = ContinuousClock.now
        let (service, error) = await ServiceFactory.make(paths: paths, environment: [:], bundle: resources)
        openTime = ContinuousClock.now - started
        #expect(error == nil, "\(error ?? "")")
        return service
    }

    private var openTime: Duration = .zero
    private func ms(_ d: Duration) -> String { String(format: "%.2f ms", Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1000) }
    private func report(_ line: String) { print("PERF " + line) }

    /// Offscreen: a borderless window far off any screen, so drawing runs as it would on screen.
    private func offscreen(_ view: NSView) -> NSWindow {
        let window = NSWindow(contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: view.frame.size), styleMask: .borderless, backing: .buffered, defer: false)
        window.contentView = view
        window.orderFrontRegardless()
        window.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        return window
    }
    /// Lets SwiftUI and AppKit run their pending update, layout and display.
    private func settle(_ window: NSWindow) {
        RunLoop.main.run(until: Date())
        window.contentView?.layoutSubtreeIfNeeded()
        window.displayIfNeeded()
    }

    /// Launch: the service opening (catalog verified) and the first table read, as LaunchDelegate does them.
    @Test func launchToMenu() async throws {
        let clock = ContinuousClock()
        let service = try await realService()
        let opened = clock.now
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root.appendingPathComponent("support")))
        app.rebuildMenu()
        let menu = clock.now
        await app.table.refresh()
        report("launch: ServiceFactory.make (catalog verified, 15 models) \(ms(openTime)), app objects + menu \(ms(menu - opened)), first table read \(ms(clock.now - menu))")
        await service.shutdown()
    }

    @Test func tableRefreshOnTheRealService() async throws {
        let service = try await realService()
        let table = TableController(service: service)
        for entry in await service.catalog(query: CatalogQuery()) where !entry.added { try? await service.add(id: entry.id) }
        await table.refresh()
        let clock = ContinuousClock()
        let n = 20
        let total = await clock.measure { for _ in 0..<n { await table.refresh() } }
        report("table refresh (real service, \(table.rows.count) rows): \(ms(total / n)) each")
        let rows = await clock.measure { for _ in 0..<n { _ = await service.rows() } }
        report(
            "  of which service.rows(): \(ms(rows / n)); catalog(query:): \(ms(await clock.measure { for _ in 0..<n { _ = await service.catalog(query: CatalogQuery()) } } / n))")
        let browser = BrowserController(service: service)
        let browserTotal = await clock.measure { for _ in 0..<n { await browser.refresh() } }
        report("browser refresh (real service, \(browser.items.count) items): \(ms(browserTotal / n)) each")
        await service.shutdown()
    }

    /// A download's progress tick as the open menu sees it: the row's state changes, the table redraws.
    @Test func progressTickInTheOpenMenu() async throws {
        let service = try await realService()
        for entry in await service.catalog(query: CatalogQuery()) where !entry.added { try? await service.add(id: entry.id) }
        let table = TableController(service: service)
        await table.refresh()
        let menu = ModelsMenu(controller: table)
        let item = menu.modelItem()
        let view = try #require(item.submenu?.items.first?.view)
        let window = offscreen(view)
        settle(window)
        var rows = table.rows
        let id = rows[0].id, p = rows[0].precisions[0]
        let clock = ContinuousClock()
        let n = 200
        let total = clock.measure {
            for i in 0..<n {
                rows[0].states[p] = PrecisionState(state: .downloading, progress: Double(i) / Double(n), bytes: Int64(i) * 1_000_000, totalBytes: Int64(n) * 1_000_000)
                table.apply(rows: rows, kinds: table.kinds, catalog: table.catalog)
                settle(window)
            }
        }
        report("progress tick, open menu (\(rows.count) rows, \(id)): \(ms(total / n)) each (apply + SwiftUI update + layout + display)")
        let same = clock.measure { for _ in 0..<n { table.apply(rows: rows, kinds: table.kinds, catalog: table.catalog); settle(window) } }
        report("unchanged refresh, open menu: \(ms(same / n)) each")
        window.orderOut(nil)
        await service.shutdown()
    }

    /// What every service event costs while the menu is closed: the status menu is rebuilt.
    @Test func menuRebuildWhileClosed() async throws {
        let service = try await realService()
        for entry in await service.catalog(query: CatalogQuery()) where !entry.added { try? await service.add(id: entry.id) }
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root.appendingPathComponent("support")))
        await app.table.refresh(); await app.refreshMenuState()
        let clock = ContinuousClock()
        let n = 50
        let rebuild = clock.measure { for _ in 0..<n { app.rebuildMenu() } }
        report("rebuildMenu (\(app.table.rows.count) rows): \(ms(rebuild / n)) each")
        // Opening the menu after a rebuild: the table's item view lays out and draws.
        let open = clock.measure {
            for _ in 0..<10 {
                app.rebuildMenu()
                guard let view = app.menu.items.first?.submenu?.items.first?.view else { return }
                let window = offscreen(view); settle(window); window.orderOut(nil); window.contentView = nil
            }
        }
        report("rebuild + first layout/draw of the table: \(ms(open / 10)) each")
        let state = await clock.measure { for _ in 0..<n { await app.refreshMenuState() } }
        report("refreshMenuState (real service): \(ms(state / n)) each")
        // One batched admission per refresh (kinds verified once): about 15 ms on an M5 Max in use, 100 ms per row before.
        #expect(state / n < .milliseconds(50))
        let admissions = await clock.measure { for _ in 0..<n { _ = await app.tools.requestAdmissions(rows: app.table.rows, service: service) } }
        report("  of which tool admissions: \(ms(admissions / n)) each")
        let rows = app.table.rows
        let primary = await clock.measure { for row in rows { _ = await service.primaryOperationAdmission(id: row.id, preferred: nil) } }
        let refusal = await clock.measure { for row in rows { _ = await service.runRefusal(id: row.id, op: "generate") } }
        let names = await clock.measure { for row in rows { _ = await service.declaredOperationNames(id: row.id) } }
        report("    one pass over \(rows.count) rows: primaryOperationAdmission \(ms(primary)), runRefusal \(ms(refusal)), declaredOperationNames \(ms(names))")
        await service.shutdown()
    }

    /// A service event storm (a download's progress): how long until the table has caught up with the last change.
    @Test func eventStormSettles() async throws {
        let rows = Fixtures.rows(.firstLaunch)
        let service = FixtureService(rows: rows)
        let table = TableController(service: service)
        table.start()
        try await Task.sleep(for: .milliseconds(200))
        let clock = ContinuousClock()
        let n = 300
        let start = clock.now
        for i in 0..<n { await service.emit(.progress(id: rows[0].id, precision: rows[0].precisions[0], progress: Double(i) / Double(n))) }
        try await service.setKeepHot(id: rows[1].id, on: true)
        while table.rows.first(where: { $0.id == rows[1].id })?.keepHot != true { try await Task.sleep(for: .milliseconds(1)) }
        report("event storm (\(n) progress events, stub service): table current after \(ms(clock.now - start))")
        table.stop()
    }

    private static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
    }

    /// One control alone in a fresh process (ARSENAL_PERF_CONTROL names it): what showing it costs in footprint.
    @Test func controlFootprint() throws {
        guard let name = ProcessInfo.processInfo.environment["ARSENAL_PERF_CONTROL"] else { return }
        let view: NSView
        switch name {
        case "nsswitch-mini": let s = NSSwitch(); s.controlSize = .mini; view = s
        case "nsswitch-small": let s = NSSwitch(); s.controlSize = .small; view = s
        case "swiftui-switch": view = NSHostingView(rootView: Toggle("", isOn: .constant(true)).toggleStyle(.switch).controlSize(.mini).labelsHidden())
        case "nsslider": view = NSSlider(value: 0.5, minValue: 0, maxValue: 1, target: nil, action: nil)
        case "nsstepper": view = NSStepper()
        case "nssegmented": view = NSSegmentedControl(labels: ["a", "b"], trackingMode: .selectOne, target: nil, action: nil)
        case "nsbutton": view = NSButton(title: "Go", target: nil, action: nil)
        case "nssearchfield": view = NSSearchField()
        case "nstextfield": view = NSTextField(string: "x")
        case "popup": view = NSPopUpButton()
        default: view = NSView()
        }
        let before = Self.footprintMB()
        let w = NSWindow(contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: NSSize(width: 300, height: 200)), styleMask: [.titled], backing: .buffered, defer: false)
        let box = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: 200)); view.frame = NSRect(x: 20, y: 20, width: 120, height: 30); box.addSubview(view)
        w.contentView = box; w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000)); settle(w)
        report(String(format: "control %@: +%.1f MB (%.1f → %.1f)", name, Self.footprintMB() - before, before, Self.footprintMB()))
    }

    /// Which part of the browser costs the memory.
    @Test func browserMemoryParts() async throws {
        let service = try await realService()
        let browser = BrowserController(service: service)
        await browser.refresh()
        func host(_ view: NSView, _ name: String) {
            let before = Self.footprintMB()
            let w = NSWindow(
                contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), styleMask: [.titled, .resizable, .fullSizeContentView],
                backing: .buffered, defer: false)
            view.frame = NSRect(origin: .zero, size: BrowserWindowController.defaultSize)
            w.contentView = view; w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000)); settle(w)
            report(String(format: "  %@: +%.1f MB", name, Self.footprintMB() - before))
        }
        host(NSView(), "plain view")
        let list = BrowserList(); list.controller = browser; list.show(browser.shown)
        host(list.scrollView, "list")
        host(NSHostingView(rootView: NoResultsView(controller: browser)), "no-results SwiftUI")
        let effect = NSVisualEffectView(); effect.material = .sidebar; effect.blendingMode = .behindWindow; effect.state = .active
        host(effect, "visual effect view")
        host(NSHostingView(rootView: Text("x").frame(maxWidth: .infinity, maxHeight: .infinity)), "SwiftUI text")
        host(
            NSHostingView(
                rootView: Menu {
                    Button("a") {}
                } label: {
                    Text("Sort")
                }.menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden).fixedSize()), "SwiftUI Menu")
        host(NSHostingView(rootView: Toggle("", isOn: .constant(true)).toggleStyle(.checkbox).labelsHidden()), "SwiftUI checkbox")
        host(NSButton(checkboxWithTitle: "x", target: nil, action: nil), "NSButton checkbox")
        func footprint(_ label: String) throws {
            let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/footprint")
            p.arguments = [String(ProcessInfo.processInfo.processIdentifier)]
            let out = Pipe(); p.standardOutput = out; try p.run(); p.waitUntilExit()
            let text = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            try text.write(toFile: "/tmp/arsenal-pol-ui/footprint-\(label).txt", atomically: true, encoding: .utf8)
        }
        try? FileManager.default.createDirectory(atPath: "/tmp/arsenal-pol-ui", withIntermediateDirectories: true)
        try footprint("before-switch")
        let small = NSSwitch(); small.controlSize = .small
        host(small, "NSSwitch small")
        try footprint("after-switch")
        let mini = NSSwitch(); mini.controlSize = .mini
        host(mini, "NSSwitch mini")
        host(NSHostingView(rootView: Toggle("", isOn: .constant(true)).toggleStyle(.switch).controlSize(.mini).labelsHidden()), "SwiftUI switch")
        host(NSHostingView(rootView: Color.clear.contentShape(Rectangle()).gesture(WindowDragGesture())), "WindowDragGesture")
        host(NSHostingView(rootView: ScrollView(.vertical) { VStack { ForEach(0..<40) { Text("row \($0)") } } }.scrollIndicators(.never)), "SwiftUI ScrollView")
        host(NSHostingView(rootView: SearchFieldHost(field: NSSearchField(), palette: .dark).frame(width: 200, height: 30)), "search field host")
        host(
            NSHostingView(
                rootView: PillFlow(spacing: 6) {
                    ForEach(0..<20) { i in
                        Button {
                        } label: {
                            Text("pill \(i)").padding(.horizontal, 8).frame(height: 22)
                        }.buttonStyle(.plain)
                    }
                }), "PillFlow")
        host(NSHostingView(rootView: BrowserSidebar(controller: browser, search: NSSearchField(), band: 28)), "sidebar SwiftUI")
        await service.shutdown()
    }

    /// What the browser window costs in memory while open, and what closing it gives back.
    @Test func browserMemory() async throws {
        let service = try await realService()
        let browser = BrowserController(service: service)
        await browser.refresh()
        let clock = ContinuousClock()
        let bare = Self.footprintMB()
        // A plain window of the same size first: what any window costs this process (backing store, CoreAnimation).
        let plain = NSWindow(
            contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), styleMask: [.titled, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        plain.contentView = NSView(); plain.orderFrontRegardless(); plain.setFrameOrigin(NSPoint(x: -9000, y: -9000)); settle(plain)
        let afterPlain = Self.footprintMB()
        plain.orderOut(nil)
        report(String(format: "plain window: footprint %.1f MB before, %.1f MB with it", bare, afterPlain))
        let before = Self.footprintMB()
        var window: BrowserWindowController?
        let open = clock.measure {
            window = BrowserWindowController(controller: browser)
            if let w = window?.window {
                w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
                w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000)); settle(w)
            }
        }
        // The first switch on screen spikes ~200 MB of graphics memory for about a second (macOS 26 compiling its
        // control material); sample after it settles.
        try await Task.sleep(for: .seconds(2))
        let opened = Self.footprintMB()
        window?.close(); window = nil
        try await Task.sleep(for: .seconds(2))
        let closed = Self.footprintMB()
        report(String(format: "browser first open: %@; footprint %.1f MB before, %.1f MB open, %.1f MB after close + release", ms(open), before, opened, closed))
        // v2.1 kept the closed window for the session; v2.2 releases it (AppDelegate.showBrowser).
        for cycle in 1...3 {
            var w: BrowserWindowController? = BrowserWindowController(controller: browser)
            weak let released = w
            guard let win = w?.window else { break }
            win.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
            win.orderFrontRegardless(); win.setFrameOrigin(NSPoint(x: -9000, y: -9000)); settle(win)
            try await Task.sleep(for: .seconds(2))
            let open = Self.footprintMB()
            w?.close()
            try await Task.sleep(for: .seconds(2))
            let kept = Self.footprintMB()
            w = nil
            try await Task.sleep(for: .seconds(2))
            report(
                String(
                    format: "  cycle %d: %.1f MB open, %.1f MB closed and kept, %.1f MB released (freed: %@)", cycle, open, kept, Self.footprintMB(), released == nil ? "yes" : "no"
                ))
        }
        await service.shutdown()
    }

    /// The browser's list after a data refresh (an Add, a table Get's progress): rows are rebuilt.
    @Test func browserDataRefresh() async throws {
        let service = try await realService()
        let browser = BrowserController(service: service)
        await browser.refresh()
        let window = BrowserWindowController(controller: browser)
        guard let w = window.window else { return }
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
        w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        settle(w)
        let entries = await service.catalog(query: CatalogQuery()), kinds = await service.kinds()
        let clock = ContinuousClock()
        let n = 50
        let same = clock.measure { for _ in 0..<n { browser.apply(entries: entries, kinds: kinds); settle(w) } }
        report("browser data refresh, nothing changed (\(entries.count) models): \(ms(same / n)) each")
        // Items are rebuilt only for entries that changed: 0.3 ms here, 24 ms when every refresh rebuilt all 48.
        #expect(same / n < .milliseconds(5))
        var changed = entries
        let one = clock.measure {
            for i in 0..<n {
                changed[0].row.states[changed[0].row.precisions[0]] = PrecisionState(state: .downloading, progress: Double(i) / Double(n))
                browser.apply(entries: changed, kinds: kinds); settle(w)
            }
        }
        report("browser data refresh, one row changed: \(ms(one / n)) each")
        browser.expanded = Set(entries.prefix(2).map(\.id)); settle(w)
        let expanded = clock.measure { for _ in 0..<n { browser.apply(entries: entries, kinds: kinds); settle(w) } }
        report("browser data refresh, two rows expanded: \(ms(expanded / n)) each")
        let toggle = clock.measure { for _ in 0..<20 { window.list.toggle(entries[3].id); settle(w) } }
        report("expand/collapse one row: \(ms(toggle / 20)) each")
        let hover = clock.measure { for i in 0..<n { window.list.hoveredRow = i % 10; settle(w) } }
        report("hover move between rows (two expanded): \(ms(hover / n)) each")
        let resize = clock.measure {
            for i in 0..<40 { w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: NSSize(width: 1000 + CGFloat(i * 10), height: 800)), display: true); settle(w) }
        }
        report("live resize step: \(ms(resize / 40)) each")
        let search = clock.measure { for i in 0..<n { browser.state.search = i % 2 == 0 ? "ko" : ""; settle(w) } }
        report("search keystroke: \(ms(search / n)) each")
        let searchModel = clock.measure { for i in 0..<n { browser.state.search = i % 2 == 0 ? "ko" : "" } }
        report("  of which filtering (no layout/display): \(ms(searchModel / n)) each")
        var probe = browser.state
        let visible = clock.measure { for i in 0..<n { probe.search = i % 2 == 0 ? "ko" : ""; _ = probe.visible(browser.items, physicalBytes: browser.physicalBytes) } }
        report("    of which the filter itself: \(ms(visible / n)) each")
        settle(w)
        let fold = clock.measure { for i in 0..<n { if i % 2 == 0 { browser.openSections.insert(.status) } else { browser.openSections.remove(.status) }; settle(w) } }
        report("sidebar section fold (list untouched): \(ms(fold / n)) each")
        let kind = clock.measure { for i in 0..<n { browser.state.kind = i % 2 == 0 ? "voices" : nil; settle(w) } }
        report("kind filter: \(ms(kind / n)) each")
        browser.expanded = []; settle(w)
        let all = browser.shown, some = Array(all.prefix(2))
        let swap = clock.measure { for i in 0..<n { window.list.show(i % 2 == 0 ? some : all); settle(w) } }
        report("list alone, 15 ⇄ 2 rows: \(ms(swap / n)) each")
        let swapOnly = clock.measure { for i in 0..<n { window.list.show(i % 2 == 0 ? some : all) } }
        report("list alone, 15 ⇄ 2 rows, without layout/display: \(ms(swapOnly / n)) each")
        w.orderOut(nil)
        await service.shutdown()
    }
}
