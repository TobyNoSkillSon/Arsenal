// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ServiceManagement
import Testing
import ArsenalCore
@testable import Arsenal

/// Arsenal 1.0 felt laggy in its menu while models ran (3 Nov 2026). Profiled in the VM with a 39-model table: the main thread's
/// work was mostly `SMAppService.status`, a synchronous round trip to the system's login-item service (60–150 ms each while
/// models ran), asked by every rebuild of the status menu; and every table refresh rebuilt the menu twice, the second time
/// after re-reading voices that had not changed. Now the status is read once and refreshed off the main thread, and an
/// unchanged voice read changes nothing.
@MainActor @Suite(.menuState, .serialized) struct MenuResponsivenessTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-menu-responsive-\(UUID().uuidString)")

    @Test func rebuildingTheMenuNeverAsksTheSystemForLaunchAtLogin() async throws {
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: MenuTests.downloaded), paths: ArsenalPaths(support: support))
        var reads = 0
        app.loginStatus = {
            reads += 1
            return .enabled
        }
        await app.table.refresh(); await app.refreshMenuState()
        for _ in 0..<25 { app.rebuildMenu() }
        await app.refreshMenuState()
        #expect(reads == 1, "\(reads) reads")
        let settings = try #require(app.menu.items.first { $0.title == "Settings" }?.submenu)
        #expect(settings.items.first { $0.title == AppDelegate.loginTitle }?.state == .on)
        // A change is read again once, and shows.
        app.loginStatus = {
            reads += 1
            return .notRegistered
        }
        app.rebuildMenu(); app.rebuildMenu()
        #expect(reads == 2)
        #expect(app.menu.items.first { $0.title == "Settings" }?.submenu?.items.first { $0.title == AppDelegate.loginTitle }?.state == .off)
        // Toggling reads the state it changes, then the menu reads the result once (registration is a stub: a test never
        // registers a login item).
        var registered = 0
        app.registerLogin = { registered += 1 }
        app.unregisterLogin = { Issue.record("unregistered") }
        try app.changeLogin(); app.rebuildMenu(); app.rebuildMenu()
        #expect(reads == 4 && registered == 1)
    }

    /// A background reading that began before a toggle never puts the older state back.
    @Test func aStaleBackgroundReadingIsDropped() async throws {
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: []), paths: ArsenalPaths(support: support))
        app.registerLogin = {}; app.unregisterLogin = {}
        var state = SMAppService.Status.notRegistered
        app.loginStatus = { state }
        app.rebuildMenu()
        #expect(app.loginState == .notRegistered)
        let gate = DispatchSemaphore(value: 0)
        app.refreshLoginStateInBackground(reader: {
            gate.wait()
            return .notRegistered
        })
        state = .enabled
        try app.changeLogin()
        #expect(app.loginState == .enabled)
        gate.signal()
        await app.loginRead?.value
        #expect(app.loginState == .enabled)
        // A reading with nothing newer is taken, and the menu follows it.
        app.refreshLoginStateInBackground(reader: { .requiresApproval })
        await app.loginRead?.value
        #expect(app.loginState == .requiresApproval)
        #expect(app.menu.items.first { $0.title == "Settings" }?.submenu?.items.contains { $0.title == AppDelegate.loginApprovalTitle } == true)
    }

    @Test func anUnchangedVoiceReadDoesNotRebuildTheMenu() async throws {
        defer { try? FileManager.default.removeItem(at: support) }
        let row = ModelRow(id: "kokoro-82m", name: "Kokoro", kind: "voices", states: ["bf16": .init(state: .loaded)])
        let metadata: [String: JSONObject] = [row.id: ["setting": [["id": "voice", "type": "voice", "default": "af_heart", "applies": "request"]]]]
        let service = StubModelsService(rows: [row], metadata: metadata, supportDirectory: support)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start(); defer { api.stop() }
        let client = ArsenalClient(port: try #require(api.port))
        let center = ToolCenter(client: { client }, playback: FixturePlayback())
        var changes = 0
        center.onChange = { changes += 1 }
        let rows = await service.rows()
        await center.refreshVoices(rows, service: service)
        #expect(changes == 1 && center.currentVoice[row.id] == "af_heart")
        for _ in 0..<5 { await center.refreshVoices(rows, service: service) }
        #expect(changes == 1, "\(changes) changes")
        try await client.setSettings(row.id, ["voice": "bm_lewis"])
        await center.refreshVoices(rows, service: service)
        #expect(changes == 2 && center.currentVoice[row.id] == "bm_lewis")
    }

    @Test func supportTheDeveloperHasAQuietWarmAccent() async throws {
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: []), paths: ArsenalPaths(support: support))
        app.loginStatus = { .notRegistered }
        app.rebuildMenu()
        let item = try #require(app.menu.items.first { $0.title == AppDelegate.supportTitle })
        let view = try #require(item.view as? ProminentMenuItemView)
        #expect(view.accent == AppDelegate.supportAccent)
        #expect(item.image === Symbol.menu("heart.fill"))
        // Highlighted: the soft wash, not the system selection; with Increase Contrast, the system selection.
        view.forcesHighlight = true
        view.viewWillDraw()
        #expect(view.subviews.compactMap { $0 as? NSVisualEffectView }.allSatisfy { $0.isHidden })
        view.increaseContrast = { true }
        view.viewWillDraw()
        #expect(view.subviews.compactMap { $0 as? NSVisualEffectView }.contains { !$0.isHidden })
        // Copy Last keeps the plain native look.
        #expect(ProminentMenuItemView(item: NSMenuItem(title: "Copy", action: nil, keyEquivalent: "")).accent == nil)
    }
    @Test func aRebuildWithNothingChangedLeavesEveryItemsImageAlone() async throws {
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: MenuTests.downloaded), paths: ArsenalPaths(support: support))
        app.loginStatus = { .enabled }
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        func images(_ menu: NSMenu) -> [ObjectIdentifier?] {
            menu.items.flatMap { item in [item.image.map(ObjectIdentifier.init)] + (item.submenu.map(images) ?? []) }
        }
        let before = images(app.menu)
        #expect(before.compactMap { $0 }.count > 5)
        app.rebuildMenu(); app.rebuildMenu()
        #expect(images(app.menu) == before)
    }
}
