// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import ObjectiveC
import SwiftUI
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MenuRound2Tests {
    /// Audit the old settings-form paths with the real controller callbacks. Ordinary settings
    /// writes do not call cancelTracking; Reload requiring consent and file picking do.
    @Test func settingsAndTableActionsOnlyDismissForWindows() async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/callbacks-\(UUID())")
        defer { try? FileManager.default.removeItem(at: support) }
        // On the table through a failed int4 Get (no weights yet), so the selected bf16 still starts at Get.
        let row = ModelRow(
            id: "chat", name: "Chat", kind: "chat", precisions: ["bf16", "int4"], states: ["int4": PrecisionState(state: .failed, error: "Get failed.")])
        let metadata: JSONObject = [
            "setting": [["id": "speed", "title": "Speed", "type": "number", "default": 1, "min": 0.5, "max": 2, "applies": "request", "maps_to": "request.speed"]]
        ]
        let service = FixtureService(rows: [row], metadata: [row.id: metadata], supportDirectory: support)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        app.table.start { await app.refreshMenuState() }
        await app.table.waitUntilSettled()
        app.menuWillOpen(app.menu)
        let host = try #require(app.modelsMenu.modelItem().submenu?.items.first?.view as? MenuTableHostingView)
        let frames = MotionFrames(host: host)
        defer { frames.close() }
        await frames.settle()
        let method = try #require(class_getInstanceMethod(NSMenu.self, #selector(NSMenu.cancelTracking)))
        var cancellations = 0, windows = 0
        let replacement = imp_implementationWithBlock({ (_: NSMenu) in cancellations += 1 } as @convention(block) (NSMenu) -> Void)
        let original = method_setImplementation(method, replacement)
        defer { method_setImplementation(method, original); imp_removeBlock(replacement) }
        app.modelsMenu.presentAlert = { _ in
            windows += 1; return .alertSecondButtonReturn
        }
        app.modelsMenu.presentOpenPanel = { _ in
            windows += 1; return nil
        }
        func settle() async throws {
            for _ in 0..<12 { await frames.tick() }
            await app.table.waitUntilSettled()
        }
        for (key, value): (String, JSONValue) in [(TableController.keepKey, true), (TableController.toolsKey, false), ("speed", 1.5)] {
            app.table.set(try #require(app.table.rows.first), key, value)
            try await settle()
            #expect(cancellations == 0 && windows == 0, "\(key) has no dismissal handler")
        }
        for title in ["Get", "Load", "Unload"] {
            let action = try #require(MenuTests.find(RowActionView.self, in: host))
            #expect(action.spec?.title.hasPrefix(title) == true)
            #expect(action.accessibilityPerformPress())
            try await settle()
            #expect(cancellations == 0 && windows == 0, "\(title) without a window acts in place")
            // Once the weights are on disk, their Precision line is chosen in place too.
            if title == "Get" {
                let lines = try #require(MenuTests.find(PrecisionLinesView.self, in: host))
                #expect(lines.lines.map(\.dtype) == ["bf16"])
                // The one line is the chosen precision already: the press acts in place and writes nothing.
                #expect(!lines.press(.pill(0)))
                try await settle()
                #expect(cancellations == 0 && windows == 0, "a Precision line acts in place")
                #expect(app.table.rows.first?.selectedPrecision == "bf16")
            }
        }
        var refreshes = 0
        app.modelsMenu.refreshModels = {
            refreshes += 1; Task { try? await service.refresh() }
        }
        let refresh = try #require(MenuTests.find(RefreshControlView.self, in: host))
        #expect(refresh.accessibilityPerformPress())
        try await settle()
        #expect(refreshes == 1 && cancellations == 0 && windows == 0)
        // The old settings footer calls reloadFromSettings -> perform -> start -> confirmCode.
        var changed = try #require(app.table.rows.first)
        changed.states["int8"] = .init(state: .loaded)
        changed.needsReload = true; changed.needsCodeConsent = true; changed.contentDigest = String(repeating: "a", count: 64)
        app.table.apply(rows: [changed], kinds: app.table.kinds)
        // The old footer closes its local form, then invokes this shared row-action path.
        app.table.perform(changed)
        #expect(cancellations == 1 && windows == 1, "Reload dismisses for the code-approval window")
        let path = SettingState(declaration: ["id": "path", "title": "Path", "type": "path"], value: "")
        app.table.chooseFile(changed, path)
        #expect(cancellations == 2 && windows == 2, "File picking dismisses for NSOpenPanel")
        try await settle()
    }

    @Test(arguments: [false, true])
    func nestedViewerRepairPreservesContainment(flipped: Bool) throws {
        let table = RenderDelegate.controller(.init(name: "nested", rows: Fixtures.rows(.loaded)))
        let models = ModelsMenu(controller: table)
        let item = try #require(models.modelItem().submenu?.items.first)
        let host = try #require(item.view as? MenuTableHostingView)
        host.setNaturalSize(NSSize(width: 971, height: 512))
        let window = NSWindow(contentRect: NSRect(x: -9000, y: -9000, width: 971, height: 522), styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let container = NSView(frame: NSRect(origin: .zero, size: window.frame.size))
        let viewer: NSView = flipped ? FlippedViewer() : NSView()
        let inner: NSView = flipped ? FlippedViewer() : NSView()
        viewer.frame = NSRect(x: 0, y: -80.5, width: 971, height: 512)
        inner.frame = viewer.bounds
        window.contentView = container; container.addSubview(viewer); viewer.addSubview(inner); inner.addSubview(host)
        defer { window.contentView = nil; window.close() }
        host.setFrameOrigin(.zero)
        host.needsLayout = true; host.layoutSubtreeIfNeeded()
        #expect(item.view === host, "Retain the production menu-item association")
        #expect(host.contentScale == 1)
        let drawn = host.hosting.convert(host.hosting.bounds, to: nil)
        #expect(drawn.minY == 5 && drawn.maxY == 517, "Repair the displaced viewer, not just its child: \(drawn)")
        #expect(host.availableRect().size == host.naturalSize)
        for ancestor in [host, inner, viewer, container] {
            #expect(!MenuDiagnostics.overflows(host.hosting.convert(host.hosting.bounds, to: ancestor), container: ancestor.bounds))
        }
        let action = try #require(MenuTests.find(RowActionView.self, in: host))
        let point = action.convert(NSPoint(x: action.bounds.midX, y: action.bounds.midY), to: container)
        let hit = try #require(container.hitTest(point))
        #expect(hit === action || hit.isDescendant(of: action))
    }

    @Test(arguments: ["bf16", "int4"])
    func residentVoiceRecoveryNeverDispatchesUnload(dtype: String) async throws {
        let support = URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/recovery-\(UUID())")
        defer { try? FileManager.default.removeItem(at: support) }
        var row = ModelRow(id: "voice", name: "Voice", kind: "voices", precisions: ["bf16", "int4"], states: [dtype: .init(state: .loaded)], selectedPrecision: dtype)
        row.local = true; row.needsCodeConsent = true; row.contentDigest = String(repeating: "a", count: 64)
        let service = FixtureService(rows: [row], supportDirectory: support)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        defer { app.table.stop(); app.tools.closeAll() }
        var prompts = 0
        app.modelsMenu.presentAlert = { alert in
            prompts += 1
            return alert.messageText.hasPrefix("Get ") ? .alertFirstButtonReturn : .alertSecondButtonReturn
        }
        await app.table.refresh(); await app.refreshMenuState()
        let actual = try #require(app.table.rows.first)
        #expect(actual.residentDType != nil && actual.needsCodeConsent)
        #expect(app.table.presentation(actual).action == .unload)
        // Speak… stays an entry that opens its window, noting the approval; the window's Approve runs the table's flow.
        let item = try #require(app.menu.items.first { ($0.representedObject as? String) == AppDelegate.taskKey(.sound(.speak)) })
        #expect(item.title == "Speak\u{2026}" && item.isEnabled && item.badge?.stringValue == "Approve")
        let problem = try #require(app.tools.readiness(actual, task: .sound(.speak))?.problem)
        #expect(problem.action == .approve)
        app.tools.prepareModel(actual)
        for _ in 0..<100 {
            let called = await service.calls.contains(where: { $0.hasPrefix("load ") || $0.hasPrefix("unload ") })
            let resident = await service.rows().first?.residentDType != nil
            if called && resident { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(prompts == 1)
        let calls = await service.calls
        #expect(calls.contains("approve voice aaaa app"))
        #expect(!calls.contains("unload voice"))
        #expect(calls.contains("load voice " + dtype))
        #expect(await service.rows().first?.residentDType != nil)
    }
}

private final class FlippedViewer: NSView { override var isFlipped: Bool { true } }
