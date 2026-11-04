// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import ServiceManagement
import Testing
@testable import Arsenal

/// The menu's end matches Vella's: Support the Developer on Vella's lighter row above Quit, and hints
/// on the app items. Launch at Login is on from the first launch (MenuLoginDefaultTests); building the menu never changes it.
@MainActor @Suite(.menuState, .serialized) final class VellaMenuTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-vella-menu-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    private func app(login: SMAppService.Status = .notRegistered) async -> AppDelegate {
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.firstLaunch)), paths: ArsenalPaths(support: support))
        app.loginStatus = { login }
        app.registerLogin = { Issue.record("registered Launch at Login") }
        app.unregisterLogin = { Issue.record("unregistered Launch at Login") }
        app.openLoginSettings = { Issue.record("opened Login Items") }
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        return app
    }
    /// An item of the main menu or of its Settings submenu (Launch at Login and the models folder live there).
    private func item(_ app: AppDelegate, _ title: String) throws -> NSMenuItem {
        let settings = app.menu.items.first { $0.title == "Settings" }?.submenu?.items ?? []
        return try #require((app.menu.items + settings).first { $0.title == title })
    }

    @Test func theEndOfTheMenuIsVellas() async throws {
        let app = await app()
        let tail = app.menu.items.suffix(6).map { $0.isSeparatorItem ? "—" : $0.title }
        #expect(tail == ["Activity", "Settings", "—", "About Arsenal", AppDelegate.supportTitle, "Quit Arsenal"])
        let settings = try #require(app.menu.items.first { $0.title == "Settings" }?.submenu)
        #expect(settings.items.suffix(2).map(\.title) == [AppDelegate.loginTitle, MenuContent.openFiles])
        let support = try item(app, AppDelegate.supportTitle)
        #expect(support.view is ProminentMenuItemView && support.image != nil && support.toolTip == nil)
        let quit = try #require(app.menu.items.last)
        #expect(quit.keyEquivalent == "q" && quit.keyEquivalentModifierMask == .command && quit.toolTip == AppDelegate.quitHelp)
        // Hints on the items Vella hints, each its own sentence (Support the Developer, the family item, has none).
        for title in [MenuContent.openFiles, AppDelegate.loginTitle, "Quit Arsenal"] {
            #expect(try item(app, title).toolTip?.hasSuffix(".") == true, "\(title)")
        }
    }

    @Test func supportOpensTheSitesSupportSection() async throws {
        let app = await app()
        var opened: [URL] = []
        app.openExternalURL = {
            opened.append($0); return true
        }
        let support = try item(app, AppDelegate.supportTitle)
        (support.view as? ProminentMenuItemView)?.perform()
        try await Task.sleep(for: .milliseconds(50))
        #expect(opened == [URL(string: "https://tobynoskillson.com/#support")!])
    }

    @Test func launchAtLoginRebuildsNeverRegisterAndOnlyTheUserChangesIt() async throws {
        let app = await app()
        let login = try item(app, AppDelegate.loginTitle)
        #expect(login.state == .off && login.isEnabled && login.image != nil && login.toolTip == AppDelegate.loginHelp)
        // Building the menu, again and again, never registers anything (the fakes record any call).
        app.rebuildMenu(); app.rebuildMenu()

        var calls: [String] = []
        app.registerLogin = { calls.append("register") }
        app.unregisterLogin = { calls.append("unregister") }
        app.openLoginSettings = { calls.append("settings") }
        try app.changeLogin()
        app.loginStatus = { .enabled }; try app.changeLogin()
        app.loginStatus = { .requiresApproval }; try app.changeLogin()
        app.loginStatus = { .notFound }; try app.changeLogin()
        #expect(calls == ["register", "unregister", "settings", "register"])
    }

    @Test func launchAtLoginShowsItsStateAndWhenItWaitsForApproval() async throws {
        let on = await app(login: .enabled)
        #expect(try item(on, AppDelegate.loginTitle).state == .on)
        let waiting = await app(login: .requiresApproval)
        let login = try item(waiting, AppDelegate.loginApprovalTitle)
        #expect(login.state == .off && login.toolTip == AppDelegate.loginApprovalHelp)
        // The lighter Support row follows checkmarks in its own menu (Vella's state column): Launch at Login's checkmark
        // is in Settings now, so the main menu keeps Support at the edge; a checkmark beside it shifts it right.
        let supportItem = try item(on, AppDelegate.supportTitle)
        let support = try #require(supportItem.view as? ProminentMenuItemView)
        #expect(!on.menu.items.contains { $0.state != .off } && support.leading == 0)
        let about = try item(on, "About Arsenal")
        about.state = .on
        defer { about.state = .off }
        #expect(support.leading == ProminentMenuItemView.stateColumn)
    }

    /// While the menu is open, the approval state and the hints change in place: same items, Support keeps its view.
    @Test func anOpenMenuChangesLoginAndHintsInPlace() async throws {
        let app = await app()
        let login = try item(app, AppDelegate.loginTitle)
        let support = try item(app, AppDelegate.supportTitle)
        let view = try #require(support.view as? ProminentMenuItemView)
        let settingsItem = try item(app, "Settings")
        let settings = try #require(settingsItem.submenu)
        app.menuWillOpen(app.menu)
        defer { app.menuDidClose(app.menu) }
        app.loginStatus = { .requiresApproval }
        app.rebuildMenu()
        // Settings and its submenu keep their identity; Login changes in place inside it.
        #expect(app.menu.items.contains { $0 === settingsItem } && settingsItem.submenu === settings)
        #expect(settings.items.contains { $0 === login } && login.title == AppDelegate.loginApprovalTitle)
        #expect(login.toolTip == AppDelegate.loginApprovalHelp && !app.structurePending)
        app.loginStatus = { .enabled }
        app.rebuildMenu()
        #expect(login.title == AppDelegate.loginTitle && login.state == .on)
        #expect(app.menu.items.contains { $0 === support } && support.view === view)
        // The row's hint is the item's, read by the view (AppKit shows a view's hint, not its item's) and by VoiceOver.
        // Support has none (the family item); a hint set on its item is still the one the view shows.
        #expect(view.hint == nil)
        support.toolTip = "A hint"
        #expect(view.hint == "A hint" && view.accessibilityHelp() == "A hint")
        #expect(view.view(view, stringForToolTip: 0, point: .zero, userData: nil) == "A hint")
        support.toolTip = nil
    }

    @Test func aRefusedChangeSaysWhereToLook() {
        let error = NSError(domain: "SMAppServiceErrorDomain", code: 1, userInfo: [NSLocalizedDescriptionKey: "Operation not permitted"])
        let alert = AppDelegate.loginFailureAlert(error)
        #expect(alert.messageText == "Launch at Login not changed")
        #expect(alert.informativeText == "Operation not permitted. See Login Items & Extensions in System Settings.")
    }
}

/// `ARSENAL_VELLA_MENU_RENDERS=<dir>`: the status menu at real size (2× pixels) with Launch at Login off and on, and the
/// real Support item view at rest and highlighted, for comparison with Vella's docs/images/menu-current.png.
@MainActor @Suite(.menuState, .serialized) struct VellaMenuRenders {
    nonisolated static var directory: URL? { ProcessInfo.processInfo.environment["ARSENAL_VELLA_MENU_RENDERS"].map { URL(fileURLWithPath: $0) } }

    @Test(.enabled(if: VellaMenuRenders.directory != nil))
    func menuEnd() async throws {
        _ = NSApplication.shared
        let directory = try #require(Self.directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-vella-render-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        let oldAppearance = Appearance.shared.mode
        defer { Appearance.shared.set(oldAppearance) }
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.firstLaunch)), paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        Appearance.shared.set(.dark)
        for (name, status) in [("off", SMAppService.Status.notRegistered), ("on", .enabled), ("approval", .requiresApproval)] {
            app.loginStatus = { status }
            app.rebuildMenu()
            await withCheckedContinuation { done in
                MenuMock.render(app.menu.items, width: 374, to: directory.appendingPathComponent("arsenal-menu-login-\(name).png")) { done.resume() }
            }
        }
        app.loginStatus = { .notRegistered }; app.rebuildMenu()
        let item = try #require(app.menu.items.first { $0.title == AppDelegate.supportTitle })
        for state in ["rest", "highlighted"] {
            let twin = try #require(item.copy() as? NSMenuItem)
            let view = ProminentMenuItemView(item: twin)
            if state == "highlighted" { view.forcesHighlight = true; view.rendersSelectionFill = true }
            let container = NSView(frame: NSRect(x: 0, y: 0, width: 374, height: ProminentMenuItemView.height + 12))
            container.wantsLayer = true
            MenuMock.applyMaterial(to: container.layer, dark: true, picture: nil)
            view.frame = NSRect(x: 0, y: 6, width: 374, height: ProminentMenuItemView.height)
            container.addSubview(view)
            container.appearance = Appearance.shared.nsAppearance
            await withCheckedContinuation { done in
                MenuMock.capture(container, to: directory.appendingPathComponent("arsenal-support-item-\(state).png")) { done.resume() }
            }
            withExtendedLifetime(twin) {}
        }
    }
}
