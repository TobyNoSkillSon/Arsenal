// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Foundation
import SwiftUI
import Testing

@testable import Arsenal

/// (6 Nov 2026, with 85 models on the table: hovering the Models table lagged the whole menu, and moving up and down the
/// main menu stayed laggy afterwards.) AppKit makes a new window for a submenu each
/// time it opens, so every pass of the pointer over Models moved the whole table into a new window, and SwiftUI laid out
/// and walked all 85 rows (for accessibility focus too): 0.4-1 s of main thread per pass in the VM. And every refresh of
/// the table read each row's precision labels with a Swift Regex, thousands of times. Here, a table of the same size in
/// the real Models item view: only the rows in sight are built, also after reopening, and each label is read once.
@MainActor @Suite(.menuState, .serialized) final class MenuTableScaleTests {
    /// A table the size of the maintainer's: 85 models in 26 groups (the figure fixture's rows, renamed).
    static func bigTable(models: Int = 85, groups: Int = 26) -> (rows: [ModelRow], kinds: [KindSpec]) {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let all = (try? KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds"))) ?? Fixtures.kinds
        let kinds = Array(all.prefix(groups))
        let base = Fixtures.figureRows()
        let rows = (0..<models).map { i in
            var row = base[i % base.count]
            row.id = "\(row.id)-\(i)"; row.name = "\(row.name) \(i)"
            row.kind = kinds[i % kinds.count].id
            return row
        }
        return (rows, kinds)
    }

    private func open() async throws -> SettingsCardHost {
        let (rows, kinds) = Self.bigTable()
        let controller = TableController(service: FixtureService(rows: rows, kinds: kinds), chip: "M5 Max")
        controller.apply(rows: rows, kinds: kinds)
        await controller.refresh()
        let host = SettingsCardHost(controller)
        await host.settle()
        return host
    }

    /// Rows a table of `maxHeight` can show at once, with a whole group of slack (a lazy stack builds a group at a time).
    private func inSight(_ c: TableController) -> Int {
        Int(ModelTable.maxHeight / (ModelTable.rowHeight + ModelTable.rowGap)) + (c.groups.map(\.rows.count).max() ?? 0) + 2
    }

    @Test func aLongTableBuildsOnlyTheRowsInSight() async throws {
        let host = try await open(); defer { host.close() }
        let c = host.controller
        #expect(c.rows.count >= 60, "a table the size of the maintainer's")
        #expect(ModelTable.scrolls(c))
        let built = SettingsCardHost.views(RowActionView.self, host.view).count
        #expect(built > 5, "the rows in sight are there")
        #expect(built <= inSight(c), "\(built) of \(c.rows.count) rows built")
    }

    /// AppKit opens a submenu in a new window each time: the item view leaves one window for another. Each open draws only
    /// the rows in sight, and reopening builds nothing more.
    @Test func reopeningModelsDrawsOnlyTheRowsInSight() async throws {
        let host = try await open(); defer { host.close() }
        let c = host.controller
        let window = try #require(host.window), viewer = try #require(window.contentView)
        for pass in 0..<3 {
            window.contentView = NSView()
            await host.settle()
            let before = SettingsCardHost.views(RowActionView.self, host.view).reduce(0) { $0 + $1.drawPasses }
            host.models.menuWillOpen(host.menu)
            window.contentView = viewer
            await host.settle()
            window.display()
            let views = SettingsCardHost.views(RowActionView.self, host.view)
            #expect(views.count <= inSight(c), "pass \(pass): \(views.count) rows built")
            let draws = views.reduce(0) { $0 + $1.drawPasses } - before
            #expect(draws <= 2 * inSight(c), "pass \(pass): \(draws) action cells drawn")
        }
    }

    /// A row's text hovers (flame, name, status line, figures) ride on one view of the row, not a platform view each: each
    /// open walks every platform view the table holds.
    @Test func aRowsHoversRideOnOneView() async throws {
        let host = try await open(); defer { host.close() }
        let rows = SettingsCardHost.views(RowActionView.self, host.view).count
        let carriers = SettingsCardHost.views(RowTooltips.RowTooltipsView.self, host.view)
        #expect(carriers.count == rows, "one hover view per built row")
        let tips = carriers.flatMap(\.subviews).compactMap { $0 as? TooltipCell.TooltipView }
        #expect(tips.count >= 3 * rows, "\(tips.count) hovers on \(rows) rows")
        #expect(tips.allSatisfy { $0.toolTip?.isEmpty == false })
    }

    /// Each precision label is read once, whatever the number of rows and refreshes (counted per label, so labels other
    /// tests read meanwhile do not count).
    @Test func aTableRefreshReadsNoLabelAgain() async throws {
        let host = try await open(); defer { host.close() }
        let c = host.controller
        await host.settle()
        let labels = Set(c.rows.flatMap(\.precisions).map { $0.lowercased() })
        func reads() -> [String: Int] { PrecisionColumn.labelReads.withLock { all in all.filter { labels.contains($0.key) } } }
        let first = reads()
        for _ in 0..<3 {
            c.objectWillChange.send()
            await host.settle()
            for row in c.rows { _ = c.presentation(row) }
        }
        #expect(reads() == first, "a refresh read labels again")
    }

    /// A row's hovers sit over their cells: the flame at the row's start, the name after it, each figure over its column.
    @Test func aRowsHoversSitOverTheirCells() async throws {
        let host = try await open(); defer { host.close() }
        let carrier = try #require(SettingsCardHost.views(RowTooltips.RowTooltipsView.self, host.view).first)
        let frames = carrier.subviews.map(\.frame)
        typealias W = ModelTable.W
        #expect(frames.contains { abs($0.minX - W.rowPadding) < 0.5 && abs($0.width - W.icon) < 0.5 }, "the flame: \(frames)")
        #expect(frames.contains { abs($0.minX - (W.rowPadding + W.icon + W.iconGap)) < 0.5 }, "the name: \(frames)")
        for slot in 0..<TableFigures.columns.count {
            #expect(frames.contains { abs($0.minX - W.slotX(slot)) < 0.5 && abs($0.width - W.metric) < 0.5 }, "figure \(slot): \(frames)")
        }
        #expect(frames.allSatisfy { carrier.bounds.contains($0) }, "inside the row: \(frames) in \(carrier.bounds)")
    }

    /// A drag in a scrolled table lands where the pointer is: the headers built are the groups in sight, not the first ones.
    @Test func aDragSlotCountsTheGroupsOutOfSight() {
        let indices = [12, 13, 14, 15, 16, 17]
        #expect(GroupHeaderView.tableSlot(0, indices: indices) == 12)
        #expect(GroupHeaderView.tableSlot(4, indices: indices) == 16)
        #expect(GroupHeaderView.tableSlot(6, indices: indices) == 18)
        #expect(GroupHeaderView.tableSlot(2, indices: [0, 1, 2]) == 2)
        #expect(GroupHeaderView.tableSlot(3, indices: [0, 1, 2]) == 3)
        #expect(GroupHeaderView.tableSlot(0, indices: []) == 0)
    }
}
