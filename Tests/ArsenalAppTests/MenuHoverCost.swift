// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Foundation
import QuartzCore
import SwiftUI
import Testing

@testable import Arsenal

/// `ARSENAL_MEASURE_HOVER=1`: what the Models table costs the main thread with a table the size of the maintainer's (85
/// models in 26 groups): an action cell's hover, a scroll step, an open of the Models submenu (AppKit moves the item view
/// into a new window each time) and a table refresh, shown and with its window ordered out. The real hosted table in
/// AppKit's item viewer, in a borderless window at (-9000, -9000), never on a screen. Prints figures (debug build, no
/// accessibility client: the real menu with one costs more; Arsenal-public-work/lab/menu-lag/hoverprobe measures that in
/// the VM); asserts nothing. Not run by default. `ARSENAL_MEASURE_OPENS` and `ARSENAL_MEASURE_RENDERS` repeat those two.
@MainActor @Suite(.menuState, .serialized) struct MenuHoverCost {
    /// One turn of the main run loop without waiting.
    static func spin() { RunLoop.main.run(until: Date()) }
    static func ms(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
    static func summary(_ label: String, _ times: [Double]) -> String {
        let t = times.sorted()
        guard !t.isEmpty else { return "\(label): none" }
        func p(_ q: Double) -> Double { t[min(t.count - 1, Int(Double(t.count) * q))] }
        return String(
            format: "%@: n=%d median %.2f ms p95 %.2f ms max %.2f ms (>16.7 ms: %d)", label, t.count, p(0.5), p(0.95), t.last!,
            t.filter { $0 > 16.7 }.count)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MEASURE_HOVER"] != nil))
    func measureHoverScrollOpen() async throws {
        let environment = ProcessInfo.processInfo.environment
        let (rows, kinds) = MenuTableScaleTests.bigTable()
        let controller = TableController(service: FixtureService(rows: rows, kinds: kinds), chip: "M5 Max")
        defer { controller.stop() }
        controller.apply(rows: rows, kinds: kinds)
        await controller.refresh()
        let host = SettingsCardHost(controller)
        defer { host.close() }
        await host.settle()
        let window = try #require(host.window)
        window.orderFrontRegardless(); window.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        await host.settle()
        print("HOVER rows \(controller.rows.count) groups \(controller.groups.count) scrolls \(ModelTable.scrolls(controller))")
        func flush() { window.displayIfNeeded(); CATransaction.flush() }

        // 1. Hover enter and exit of each built row's action cell (its own tracking area's callbacks).
        var cells: [Double] = []
        for view in SettingsCardHost.views(RowActionView.self, host.view) {
            let t0 = DispatchTime.now().uptimeNanoseconds
            view.setHovered(true); view.setHovered(false)
            flush()
            cells.append(Self.ms(t0))
        }
        print("HOVER " + Self.summary("action cell enter+exit", cells))

        // 2. Scroll: 60 wheel steps down, then 60 up, at the table's centre.
        var scroll: [Double] = []
        let bounds = host.view.bounds
        let centre = host.view.convert(NSPoint(x: bounds.midX, y: bounds.midY), to: nil)
        let global = window.convertPoint(toScreen: centre)
        let root = try #require(window.contentView?.superview)
        for step in 0..<120 {
            let cg = try #require(CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1, wheel1: step < 60 ? -12 : 12, wheel2: 0, wheel3: 0))
            cg.location = CGPoint(x: global.x, y: (NSScreen.screens.first?.frame.height ?? 0) - global.y)
            let event = try #require(NSEvent(cgEvent: cg))
            let t0 = DispatchTime.now().uptimeNanoseconds
            root.hitTest(root.convert(centre, from: nil))?.scrollWheel(with: event)
            Self.spin()
            flush()
            scroll.append(Self.ms(t0))
        }
        print("HOVER " + Self.summary("scroll step", scroll))

        // 3. The Models submenu opening: the item view leaves its window and comes back, then lays out and draws.
        var opens: [Double] = []
        var passes: [Int] = []
        let viewer = try #require(window.contentView)
        for _ in 0..<(Int(environment["ARSENAL_MEASURE_OPENS"] ?? "") ?? 10) {
            window.contentView = NSView()
            Self.spin()
            let t0 = DispatchTime.now().uptimeNanoseconds
            let layouts = host.view.hostingLayoutPasses
            host.models.menuWillOpen(host.menu)
            window.contentView = viewer
            host.allocate()
            Self.spin()
            flush()
            opens.append(Self.ms(t0))
            passes.append(host.view.hostingLayoutPasses - layouts)
        }
        print("HOVER host layout passes per open: \(passes)")
        print("HOVER " + Self.summary("Models submenu open", opens) + " (\(SettingsCardHost.views(RowActionView.self, host.view).count) rows built)")

        // 4. A table refresh (one publish), shown, then with the window ordered out as a closed submenu's.
        let renders = Int(environment["ARSENAL_MEASURE_RENDERS"] ?? "") ?? 10
        func refreshes(_ label: String, display: Bool) {
            var times: [Double] = []
            for _ in 0..<renders {
                let t0 = DispatchTime.now().uptimeNanoseconds
                controller.objectWillChange.send()
                Self.spin()
                if display { host.view.layoutSubtreeIfNeeded(); flush() }
                times.append(Self.ms(t0))
            }
            print("HOVER " + Self.summary(label, times) + " (\(SettingsCardHost.views(RowActionView.self, host.view).count) rows built)")
        }
        refreshes("table refresh, shown", display: true)
        window.orderOut(nil)
        Self.spin()
        refreshes("table refresh, window ordered out", display: false)
    }
}
