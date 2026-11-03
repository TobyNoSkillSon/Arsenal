// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import ObjectiveC
@testable import ArsenalAPI
import SwiftUI
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MenuStabilityTests {
    @Test func windowCallbacksDismissBeforePresentation() throws {
        let table = TableController(service: FixtureService(rows: []))
        let models = ModelsMenu(controller: table)
        _ = models.modelItem()
        let row = ModelRow(id: "q", name: "Qwen", kind: "chat")
        let plan = GetPlan(id: row.id, dtype: "bf16", runner: "native", items: [])
        let method = try #require(class_getInstanceMethod(NSMenu.self, #selector(NSMenu.cancelTracking)))
        var cancellations = 0
        let block: @convention(block) (NSMenu) -> Void = { _ in cancellations += 1 }
        let replacement = imp_implementationWithBlock(block)
        let original = method_setImplementation(method, replacement)
        defer { method_setImplementation(method, original); imp_removeBlock(replacement) }
        var presentations = 0
        models.presentAlert = { _ in
            presentations += 1
            #expect(cancellations == presentations, "Dismiss before each modal")
            return .alertFirstButtonReturn
        }
        models.presentOpenPanel = { _ in
            presentations += 1
            #expect(cancellations == presentations, "Dismiss before opening a file picker")
            return nil
        }
        _ = table.confirmRemove(row); _ = table.confirmCode(row)
        _ = table.confirmGet(row, plan); table.unknownGetSize(row, plan)
        _ = table.pickFile(row, SettingState(declaration: ["id": "path", "title": "Path", "type": "path"], value: ""))
        #expect(cancellations == 5 && presentations == 5)
        table.openBrowser()
        #expect(cancellations == 6, "Opening a window ends tracking")
    }

    @Test(arguments: ToolKind.allCases, [(false, false), (true, false), (false, true), (true, true)])
    func visibilityUpdatesKeepOtherItemsAndHostedFrames(tool: ToolKind, route: (open: Bool, mcp: Bool)) async throws {
        let (open, mcp) = route
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/visibility-\(UUID())")
        defer { try? FileManager.default.removeItem(at: support) }
        var row = ModelRow(
            id: "changed", name: "Changed", kind: tool == .voice ? "voices" : tool == .image || tool == .background ? "images" : tool == .sound ? "sound" : "chat",
            states: ["bf16": .init(state: .loaded)]
        )
        row.tools = [tool.rawValue]
        var neighbour = ModelRow(id: "neighbour", name: "Neighbour", kind: "chat", states: ["bf16": .init(state: .loaded)])
        neighbour.tools = ["chat"]
        let service = FixtureService(
            rows: [row, neighbour], metadata: [row.id: ["tools": [.string(tool.rawValue)]]], supportDirectory: support,
            operations: tool == .background ? [row.id: [CutoutFixture.operation]] : [:])
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        let api = APIHost(service: service, support: support.appendingPathComponent("api"), port: 0)
        await api.start()
        let client = ArsenalClient(port: try #require(api.port))
        let router = APIRouter(service: service, support: support)
        defer { api.stop() }
        let status = AppDelegate.makeStatusItem(visible: false)
        app.status = status
        let statusLength = status.length, statusFrame = status.button?.frame
        defer { NSStatusBar.system.removeStatusItem(status) }
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        app.table.start { await app.refreshMenuState() }
        await app.table.waitUntilSettled()
        if open { app.menuWillOpen(app.menu) }
        let models = try #require(app.menu.items.first { $0.title == "Models" })
        let activity = try #require(app.menu.items.first { $0.title == "Activity" })
        // By task: the neighbour's Chat… entry; the changed model's entry, or its choice inside Chat… when both chat.
        let chatKey = AppDelegate.taskKey(.tool(.chat))
        let neighbourItem = try #require(app.menu.items.first { ($0.representedObject as? String) == chatKey })
        let task = app.tools.tasks(try #require(app.table.rows.first { $0.id == row.id })).first
        func changedItem() -> NSMenuItem? {
            guard let task else { return nil }
            let key = AppDelegate.taskKey(task)
            if let choice = app.menu.everyItem.first(where: { ($0.representedObject as? [String]) == [key, row.id] }) { return choice }
            return app.menu.items.first { ($0.representedObject as? String) == key }
        }
        let changed = changedItem()
        #expect((changed == nil) == (tool == .search), "\(tool): every tool but Search has an entry")
        let structure = app.menu.everyItem.map(ObjectIdentifier.init)
        let modelHost = try #require(models.submenu?.items.first?.view as? MenuTableHostingView)
        let activityHost = try #require(activity.submenu?.items.first?.view as? ActivityHostingView)
        let modelFrames = MotionFrames(host: modelHost), activityFrames = MotionFrames(host: activityHost)
        defer { modelFrames.close(); activityFrames.close() }
        await modelFrames.settle(); await activityFrames.settle()
        let modelBefore = try modelFrames.capture(), activityBefore = try activityFrames.capture()
        for shown in [false, true, false, true] {
            if mcp {
                _ = try await router.callMCPTool(
                    MCPNames.tool("settings"), arguments: ["model": "changed", "set": ["show_in_menu": .bool(shown)]], requester: "Menu regression", server: nil, progress: nil)
            } else {
                _ = try await client.send("POST", "/arsenal/v1/models/changed/tools", json: ["on": shown])
            }
            for _ in 0..<12 {
                await modelFrames.tick(); await activityFrames.tick()
                let m = try modelFrames.capture(), a = try activityFrames.capture()
                #expect(m.bytes == modelBefore.bytes && m.size == modelBefore.size)
                #expect(a.bytes == activityBefore.bytes && a.size == activityBefore.size)
                #expect(app.menu.items.contains { $0 === models } && app.menu.items.contains { $0 === activity })
                #expect(app.menu.items.contains { $0 === neighbourItem }, "An unrelated tool must never detach")
                #expect(status.length == statusLength && status.button?.frame == statusFrame && !status.isVisible)
            }
            // Open or closed, the menu follows at once: the model's entry stays in place and is shown or hidden, and no
            // item anywhere in the menu tree is added, removed or replaced.
            if let changed {
                #expect(changedItem() === changed)
                // Chat is one entry for every chat model (the window picks the model): the neighbour keeps it shown.
                let shared = task == .tool(.chat)
                #expect(changed.isHidden == (shared ? false : !shown), "\(tool) shown \(shown): \(changed.title)")
            }
            #expect(app.menu.everyItem.map(ObjectIdentifier.init) == structure, "\(app.menu.everyItem.map { ($0.isHidden ? "h:" : "") + ($0.isSeparatorItem ? "—" : $0.title) })")
            #expect(!neighbourItem.isHidden)
            #expect(!app.structurePending)
        }
        if open { app.menuDidClose(app.menu) }
        #expect(app.menu.items.contains { $0 === models } && app.menu.items.contains { $0 === neighbourItem })
    }

    @Test func displacedModelsAndActivityFitTheLoggedAllocation() throws {
        let table = RenderDelegate.controller(.init(name: "offset", rows: Fixtures.rows(.loaded)))
        let models = ModelsMenu(controller: table), activity = ActivityMenu(controller: table)
        let modelHost = try #require(models.modelItem().submenu?.items.first?.view as? MenuTableHostingView)
        let activityHost = try #require(activity.item().submenu?.items.first?.view as? ActivityHostingView)
        // Preserve the captured native viewer allocation. An untracked NSMenu otherwise observes
        // frame changes and normalizes the item to y=0 before the offscreen host can see the bug.
        // Detach the item association, retaining the actual production hosts/content/geometry hook.
        models.modelItem().submenu?.items.first?.view = nil
        activity.item().submenu?.items.first?.view = nil
        func check<Content: View>(_ host: MenuContentView<Content>) {
            host.setNaturalSize(NSSize(width: 971, height: 512))
            let window = NSWindow(contentRect: NSRect(x: -9000, y: -9000, width: 971, height: 522), styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            let container = NSView(frame: NSRect(origin: .zero, size: window.frame.size))
            window.contentView = container; container.addSubview(host)
            defer { window.contentView = nil; window.close() }
            for y in [-80.5, -55.5, -20.0] {
                host.setFrameOrigin(NSPoint(x: 0, y: y))
                host.needsLayout = true; host.layoutSubtreeIfNeeded()
                #expect(!MenuDiagnostics.overflows(host.hosting.convert(host.hosting.bounds, to: nil), container: NSRect(origin: .zero, size: window.frame.size)))
                let content = host.hosting.convert(host.hosting.bounds, to: nil)
                #expect(host.contentScale == 1)
                #expect(content.minY >= 0 && window.frame.height - content.maxY <= 10, "Only the native menu's 10-pt total padding may remain: \(content)")
            }
        }
        check(modelHost); check(activityHost)
    }

    @Test func provisionalTenPointWindowNeverCollapsesContent() throws {
        let host = MenuContentView(rootView: Text("Models").frame(width: 971, height: 512))
        host.setNaturalSize(NSSize(width: 971, height: 512))
        let window = NSWindow(contentRect: NSRect(x: -9000, y: -9000, width: 10, height: 10), styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let container = NSView(frame: NSRect(origin: .zero, size: window.frame.size))
        window.contentView = container; container.addSubview(host)
        defer { window.contentView = nil; window.close() }
        host.needsLayout = true; host.layoutSubtreeIfNeeded()
        #expect(host.contentScale == 1 && host.hosting.frame.size == host.naturalSize)
        window.setContentSize(host.naturalSize); container.setFrameSize(host.naturalSize)
        host.needsLayout = true; host.layoutSubtreeIfNeeded()
        #expect(host.contentScale == 1 && host.hosting.frame.origin == .zero)
    }

    @Test func nativeActionsKeepNativePresentation() async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/native-\(UUID())")
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.loaded), supportDirectory: support), paths: ArsenalPaths(support: support))
        let playback = FixturePlayback(); app.tools.playback = playback
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        let settings = try #require(app.menu.settings)
        let mute = try #require(settings.items.first { $0.title == "Mute Agent Speech" })
        #expect(mute.view == nil && mute.state == .off)
        #expect(NSApp.sendAction(try #require(mute.action), to: mute.target, from: mute))
        #expect(playback.muted && mute.title == "Mute Agent Speech" && mute.state == .on)
        #expect(app.menu.settings === settings && settings.items.contains { $0 === mute } && mute.view == nil)
        let keep = try #require(settings.items.first { $0.title == MenuContent.keepLoadedTitle }?.submenu)
        #expect(keep.items.filter { $0.action != nil }.allSatisfy { $0.view == nil })
    }

    /// A model not on this Mac yet: its task entry opens like any other (enabled, no note, no "not installed — Get" row),
    /// its window's model choice carries the Download problem with the size, and the window's Download reaches the
    /// table's own Get. Once on the Mac a request loads it: no Load step.
    @Test(arguments: ToolKind.allCases)
    func missingWeightsHaveAnEnabledGetAction(tool: ToolKind) async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/missing-\(UUID())")
        defer { try? FileManager.default.removeItem(at: support) }
        var row = ModelRow(id: "missing", name: "Missing", kind: tool == .voice ? "voices" : tool == .image || tool == .background ? "images" : tool == .sound ? "sound" : "chat")
        row.tools = [tool.rawValue]
        row.sizeBytes = 3_400_000_000
        // On the table through an old copy the catalog no longer offers; the selected precision is not on this Mac.
        row.withdrawnTiers = ["int8": 1_700_000_000]
        let service = FixtureService(
            rows: [row], metadata: [row.id: ["tools": [.string(tool.rawValue)]]], supportDirectory: support,
            operations: tool == .background ? [row.id: [CutoutFixture.operation]] : [:])
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        // The table's Get confirms a download of this size; the person says yes.
        var prompts: [String] = []
        app.modelsMenu.presentAlert = { alert in
            prompts.append(alert.messageText); return .alertFirstButtonReturn
        }
        let current = try #require(app.table.rows.first)
        let tasks = app.tools.tasks(current)
        #expect(!app.menu.items.contains { $0.title.contains("not installed") || $0.title.contains("\u{2014} Get") })
        guard let task = tasks.first else {
            // Search serves no menu task (agents use it); nothing else may be left without an entry.
            #expect(tool == .search, "\(tool) serves no task")
            #expect(!app.menu.items.contains { ($0.representedObject as? String)?.hasPrefix("tool:") == true && !$0.isHidden })
            return
        }
        let item = try #require(app.menu.items.first { ($0.representedObject as? String) == AppDelegate.taskKey(task) })
        #expect(item.isEnabled && !item.isHidden && item.action != nil && item.badge == nil && item.view == nil)
        let problem = try #require(app.tools.readiness(current, task: task)?.problem)
        #expect(problem.action == .download && problem.actionTitle == "Download 3.4 GB")
        // The window's model choice carries the same problem, so the window offers Download.
        #expect(app.tools.models(app.table.rows, for: task).first { $0.id == row.id }?.unavailable == problem)
        // The window's Download runs the Models table's Get.
        app.tools.prepareModel(current)
        let deadline = ContinuousClock.now + .seconds(3)
        while await service.rows().first?.state != .ready {
            try #require(ContinuousClock.now < deadline, "The window's Download must reach the service")
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(await service.approvedGets.first?.id == row.id)
        await app.table.refresh(); await app.refreshMenuState()
        // On this Mac, not loaded: the entry still just opens, and nothing stands between it and a request.
        let ready = try #require(app.table.rows.first)
        #expect(app.tools.readiness(ready, task: task) == nil)
        let again = try #require(app.menu.items.first { ($0.representedObject as? String) == AppDelegate.taskKey(task) })
        #expect(again.isEnabled && again.badge == nil && !app.menu.items.contains { $0.title.contains("Load") })
    }
}
