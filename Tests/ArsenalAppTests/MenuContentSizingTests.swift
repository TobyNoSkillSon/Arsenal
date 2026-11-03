// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MenuContentSizingTests {
    /// Begin with a verified catalog and no footer. Exercise the error drained *after* the menu callback,
    /// through both subscription read paths and the real fitted Models hosting view.
    @Test(arguments: [false, true], [CGFloat(0), CGFloat(400)])
    func bufferedFooterGrowsAndShrinksAnOpenFittedMenu(rowsOnly: Bool, lostWidth: CGFloat) async throws {
        let service = RefreshReadService()
        await service.setVerifiedCatalog()
        let table = TableController(service: service)
        defer { table.stop() }
        await table.refresh()
        #expect(ModelTable.footer(table) == nil)
        let menu = ModelsMenu(controller: table)
        let submenu = try #require(menu.modelItem().submenu)
        let host = try #require(submenu.items.first?.view as? MenuTableHostingView)
        let original = host.naturalSize
        let container = NSView(frame: NSRect(origin: .zero, size: original))
        let window = hiddenWindow(size: original)
        window.contentView = container; container.addSubview(host)
        defer { window.contentView = nil; window.close() }
        func render() async throws -> NSBitmapImageRep {
            // Stand in for the native viewer's width constraint; height is the item's requested height.
            for _ in 0..<12 {
                let size = NSSize(width: host.naturalSize.width - lostWidth, height: host.naturalSize.height)
                window.setContentSize(size); container.setFrameSize(size)
                host.needsLayout = true; host.layoutSubtreeIfNeeded()
                await Task.yield()
            }
            let bitmap = try #require(host.hosting.bitmapImageRepForCachingDisplay(in: host.hosting.bounds))
            host.hosting.cacheDisplay(in: host.hosting.bounds, to: bitmap)
            return bitmap
        }
        _ = try await render()
        #expect(host.naturalSize == original)
        #expect(host.contentScale == (original.width - lostWidth) / original.width)
        // Pause the first subscription read, or the following rows-only read, so no wall-clock gate is needed.
        await service.holdNextRowsRead()
        table.start { menu.resize() }
        await service.waitForPausedRows()
        await service.releaseRowsRead()
        await table.waitUntilSettled()
        if rowsOnly {
            await service.holdNextRowsRead()
            await service.progressChanged()
            await service.waitForPausedRows()
        } else {
            await service.holdNextRowsRead()
            await service.stateChanged()
            await service.waitForPausedRows()
        }
        // rows() has already captured its snapshot. This error must be buffered by the active subscription.
        await service.reportFailure()
        await service.releaseRowsRead()
        await table.waitUntilSettled()
        #expect(table.lastError == "Fixture: Delayed read failure.")
        let bitmap = try await render()
        #expect(host.naturalSize.height == ModelTable.height(table))
        #expect(host.naturalSize.height > original.height)
        #expect(submenu.minimumWidth == host.naturalSize.width)
        #expect(!MenuDiagnostics.overflows(host.hosting.convert(host.hosting.bounds, to: container), container: container.bounds))
        // The footer text occupies the bottom band at both 1x and the safety-net scale, rather than being blank/clipped.
        let scale = CGFloat(bitmap.pixelsHigh) / host.naturalSize.height
        let top = Int((host.naturalSize.height - ModelTable.verticalPadding - ModelTable.footerHeight) * scale)
        let bottom = Int((host.naturalSize.height - ModelTable.verticalPadding) * scale)
        var ink = 0
        for y in top..<bottom {
            for x in 0..<bitmap.pixelsWide {
                if let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.sRGB), color.alphaComponent > 0.3 { ink += 1 }
            }
        }
        #expect(ink > 20, "The rendered footer must contain visible text")
        let reads = await service.counts
        table.lastError = nil
        _ = try await render()
        #expect(host.naturalSize == original, "Removing a footer shrinks the same host")
        for _ in 0..<20 { host.needsLayout = true; host.layoutSubtreeIfNeeded(); await Task.yield() }
        #expect(host.naturalSize == original)
        #expect(await service.counts == reads, "Size measurement and fitting never read the service")
        #expect(!table.isRefreshing)
    }

    /// More in the settings lane is view-local state. The host must follow geometry without knowing that state.
    /// This fixture keeps that contract testable before the separate settings lane is integrated.
    @Test func moreOpensAndClosesThroughItsHostedButtonWithoutAControllerPublish() async throws {
        let host = MenuContentView(rootView: MoreSizingFixture(), tracksContentSize: true)
        let original = NSSize(width: 360, height: 48)
        host.setNaturalSize(original)
        let window = hiddenWindow(size: original)
        let container = NSView(frame: NSRect(origin: .zero, size: original))
        window.contentView = container; container.addSubview(host)
        defer { window.contentView = nil; window.close() }
        var sizes: [NSSize] = []
        host.onNaturalSizeChange = { sizes.append($0) }
        func render() async throws -> Data {
            for _ in 0..<12 {
                let size = NSSize(width: 240, height: host.naturalSize.height)
                window.setContentSize(size); container.setFrameSize(size)
                host.needsLayout = true; host.layoutSubtreeIfNeeded(); await Task.yield()
            }
            let bitmap = try #require(host.hosting.bitmapImageRepForCachingDisplay(in: host.hosting.bounds))
            host.hosting.cacheDisplay(in: host.hosting.bounds, to: bitmap)
            #expect(!MenuDiagnostics.overflows(host.hosting.convert(host.hosting.bounds, to: container), container: container.bounds))
            return Data(bytes: try #require(bitmap.bitmapData), count: bitmap.bytesPerRow * bitmap.pixelsHigh)
        }
        let closed = try await render()
        let more = try #require(MenuTests.find(RowActionView.self, in: host.hosting))
        #expect(more.accessibilityPerformPress())
        let opened = try await render()
        #expect(host.naturalSize == NSSize(width: 360, height: 176))
        #expect(opened != closed && Set(opened).count > 8)
        let less = try #require(MenuTests.find(RowActionView.self, in: host.hosting))
        #expect(less.accessibilityPerformPress())
        let closedAgain = try await render()
        #expect(host.naturalSize == original)
        #expect(closedAgain == closed)
        #expect(sizes == [NSSize(width: 360, height: 176), original], "Exactly one resize per change; fitting cannot loop")
    }

    @Test func rowsResizeTheNativeItemAndOpeningSettingsLeavesIt() async throws {
        let table = TableController(service: FixtureService(rows: []))
        let row = ModelRow(id: "sizing", name: "Sizing", kind: "images", states: ["bf16": .init(state: .ready)])
        table.apply(rows: [], kinds: KindSpec.defaults, catalog: CatalogStatus(verified: true))
        let menu = ModelsMenu(controller: table)
        let submenu = try #require(menu.modelItem().submenu)
        let host = try #require(submenu.items.first?.view as? MenuTableHostingView)
        let window = hiddenWindow(size: host.naturalSize)
        window.contentView = host
        defer { window.contentView = nil; window.close() }
        func render() async throws -> NSSize {
            for _ in 0..<12 {
                host.needsLayout = true; host.layoutSubtreeIfNeeded(); await Task.yield()
            }
            let size = host.naturalSize
            #expect(size == NSSize(width: ModelTable.width(table), height: ModelTable.height(table)))
            #expect(submenu.minimumWidth == size.width)
            #expect(submenu.size.width >= size.width && submenu.size.height >= size.height)
            window.setContentSize(size)
            let bitmap = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: bitmap)
            #expect(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0)
            return size
        }
        let empty = try await render()
        table.apply(rows: [row], kinds: KindSpec.defaults, catalog: CatalogStatus(verified: true))
        let populated = try await render()
        #expect(populated.width > empty.width && populated.height > empty.height)
        table.settings[row.id] = TableController.withKeep(
            [
                SettingState(declaration: ["id": "choice", "title": "Choice", "type": "choice", "choices": .array((1...8).map { .string("Item \($0)") })], value: "Item 1")
            ], row)
        // The settings card floats over the table beside the row's gear: the menu's table keeps its size.
        table.openSettings(row)
        #expect(table.settingsCard.openID == row.id)
        #expect(try await render() == populated)
        table.settingsCard.close()
        #expect(try await render() == populated)
        table.apply(rows: [], kinds: KindSpec.defaults, catalog: CatalogStatus(verified: true))
        #expect(try await render() == empty)
    }

    private func hiddenWindow(size: NSSize) -> NSWindow {
        let window = NSWindow(contentRect: NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.appearance = NSAppearance(named: .darkAqua)
        window.backgroundColor = .clear; window.isOpaque = false
        return window
    }
}

// CLT lacks SDK 27's State macro plugin; use the supported property-wrapper type as BrowserRunnerFeature does.
private typealias MoreExpandedState = SwiftUI.State<Bool>
private struct MoreSizingFixture: View {
    @MoreExpandedState private var expanded = false
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            RowAction(title: "More", enabled: true, deletable: false, help: "More", onPerform: { expanded.toggle() }, onDelete: {}).frame(height: 48)
            if expanded { Text("Steps\nSeed\nGuidance\nOutput").frame(height: 128) }
        }.frame(width: 360, alignment: .leading)
    }
}
