// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import ServiceManagement
import Testing
@testable import Arsenal

/// Launch at Login is on from the first launch, and the user's choice after that. Each launch is a fresh
/// AppDelegate over the same support folder and the same fake login item; no test touches SMAppService.
@MainActor @Suite(.menuState, .serialized) final class MenuLoginDefaultTests {
    /// The login item as the app sees it, and every call the app makes to it.
    @MainActor final class LoginItem {
        var status: SMAppService.Status
        var refuses: Bool
        private(set) var registers = 0, unregisters = 0
        init(status: SMAppService.Status = .notRegistered, refuses: Bool = false) { self.status = status; self.refuses = refuses }
        func register() throws {
            registers += 1
            if refuses { throw Refusal() }
            status = .enabled
        }
        func unregister() throws { unregisters += 1; status = .notRegistered }
    }
    struct Refusal: LocalizedError { var errorDescription: String? { "Operation not permitted" } }

    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-login-default-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// One launch of Arsenal: a fresh app over the shared support folder, the fake login item behind every seam. The owed mark
    /// is set first, as the launch delegate sets it before the service writes.
    private func launch(_ item: LoginItem) async -> AppDelegate {
        if AppDelegate.isFreshInstall(support: support) { AppDelegate.markLoginDefaultOwed(support: support) }
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.firstLaunch)), paths: ArsenalPaths(support: support))
        app.loginStatus = { item.status }
        app.registerLogin = { try item.register() }
        app.unregisterLogin = { try item.unregister() }
        app.openLoginSettings = { Issue.record("opened Login Items") }
        await app.table.refresh(); await app.refreshMenuState()
        return app
    }
    /// What an install that has run before leaves in its support folder: its settings file.
    private func existingData() throws {
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: support.appendingPathComponent("config.json").path, contents: Data("{}".utf8))
    }
    private func loginItem(_ app: AppDelegate) throws -> NSMenuItem {
        app.rebuildMenu()
        // Launch at Login lives in Settings, and only there.
        #expect(!app.menu.items.contains { $0.title == AppDelegate.loginTitle || $0.title == AppDelegate.loginApprovalTitle })
        return try #require(app.menu.settings?.items.first { $0.title == AppDelegate.loginTitle || $0.title == AppDelegate.loginApprovalTitle })
    }

    @Test func theFirstLaunchTurnsItOnOnceAndRecordsIt() async throws {
        let item = LoginItem()
        let first = await launch(item)
        #expect(!first.loginChoiceRecorded)
        first.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 1 && item.status == .enabled && first.loginChoiceRecorded)
        #expect(try loginItem(first).state == .on)
        // Later launches see the record and never register again.
        for _ in 0..<2 {
            let next = await launch(item)
            next.applyLaunchAtLoginDefault(applies: true)
        }
        #expect(item.registers == 1)
    }

    @Test func aUserWhoTurnedItOffStaysOff() async throws {
        let item = LoginItem()
        let first = await launch(item)
        first.applyLaunchAtLoginDefault(applies: true)
        try first.changeLogin()
        #expect(item.status == .notRegistered && item.unregisters == 1)
        for _ in 0..<2 {
            let next = await launch(item)
            next.applyLaunchAtLoginDefault(applies: true)
            #expect(try loginItem(next).state == .off)
        }
        #expect(item.registers == 1 && item.status == .notRegistered)
    }

    /// An install that already had data (settings, models, a migrated Arsenal profile) keeps its login state: its first launch
    /// only records that, and later launches leave it alone, whichever state the item is in.
    @Test func anExistingInstallKeepsItsLoginStateAndNothingIsRegistered() async throws {
        try existingData()
        let item = LoginItem()
        let first = await launch(item)
        first.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 0 && item.status == .notRegistered && first.loginChoiceRecorded)
        #expect(try loginItem(first).state == .off)
        let next = await launch(item)
        next.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 0 && item.status == .notRegistered)
    }

    /// A fresh install marked owed at launch, then cut short before its login decision (it quit while the catalog verified), and
    /// the service's first files written: the next launch still owes and makes the default.
    @Test func aFreshInstallCutShortBeforeItsServiceOpensStillGetsTheDefault() async throws {
        AppDelegate.markLoginDefaultOwed(support: support)
        try existingData()
        let item = LoginItem()
        let next = await launch(item)
        next.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 1 && item.status == .enabled && next.loginChoiceRecorded)
    }

    @Test func anExistingInstallThatAlreadyHasTheItemOnKeepsItOn() async throws {
        try existingData()
        let item = LoginItem(status: .enabled)
        let app = await launch(item)
        app.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 0 && item.unregisters == 0 && app.loginChoiceRecorded)
        #expect(try loginItem(app).state == .on)
    }

    /// The fresh check reads only what was in the support folder before the launch: the instance lock does not count.
    @Test func freshMeansNothingInTheSupportFolderButTheInstanceLock() throws {
        #expect(AppDelegate.isFreshInstall(support: support), "a missing folder is fresh")
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: support.appendingPathComponent(".instance.lock").path, contents: nil)
        #expect(AppDelegate.isFreshInstall(support: support))
        try existingData()
        #expect(!AppDelegate.isFreshInstall(support: support))
    }

    /// A user's own choice counts as the record, even when the default has not run yet.
    @Test func aChoiceMadeBeforeTheDefaultRunsIsKept() async throws {
        let item = LoginItem()
        let first = await launch(item)
        try first.changeLogin()
        #expect(item.registers == 1 && first.loginChoiceRecorded)
        try first.changeLogin()
        #expect(item.status == .notRegistered)
        let next = await launch(item)
        next.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 1 && item.status == .notRegistered)
    }

    @Test func aLoginItemThatIsAlreadyOnIsRecordedWithoutRegistering() async throws {
        let item = LoginItem(status: .enabled)
        let app = await launch(item)
        app.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 0 && app.loginChoiceRecorded)
        #expect(try loginItem(app).state == .on)
    }

    /// Approval is the system's to give: the default records the registration and the menu keeps its approval path.
    @Test func aLoginItemWaitingForApprovalIsKeptAsItIs() async throws {
        let item = LoginItem(status: .requiresApproval)
        let app = await launch(item)
        app.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 0 && app.loginChoiceRecorded)
        #expect(try loginItem(app).title == AppDelegate.loginApprovalTitle)
    }

    /// A refused first registration records no choice and keeps the default owed, so the next launch tries again.
    @Test func aRefusedFirstRegistrationTriesAgainNextLaunch() async throws {
        let item = LoginItem(refuses: true)
        let first = await launch(item)
        first.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 1 && item.status == .notRegistered && !first.loginChoiceRecorded)
        item.refuses = false
        let next = await launch(item)
        next.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 2 && item.status == .enabled && next.loginChoiceRecorded)
    }

    /// A fresh install on macOS 27 reports .notFound before its first registration (seen in a blank VM); it is registered.
    @Test func aFreshInstallTheSystemDoesNotKnowYetIsRegistered() async throws {
        let item = LoginItem(status: .notFound)
        let app = await launch(item)
        app.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 1 && item.status == .enabled && app.loginChoiceRecorded)
    }

    /// A build the system cannot register (no bundle identifier, no helper) refuses: the default stays owed, no choice recorded.
    @Test func aLoginItemTheSystemCannotRegisterIsLeftOwed() async throws {
        let item = LoginItem(status: .notFound, refuses: true)
        let app = await launch(item)
        app.applyLaunchAtLoginDefault(applies: true)
        #expect(item.registers == 1 && item.unregisters == 0 && !app.loginChoiceRecorded)
    }

    /// The launch path as applicationDidFinishLaunching runs it: the default is applied, then the menu is built. The user
    /// turns the item off through the menu's own item, as a click does, and a later launch keeps it off.
    @Test func aLaunchSequenceRegistersOnceAndTheUsersOffChoiceSurvivesLaterLaunches() async throws {
        let item = LoginItem()
        let first = await launch(item)
        first.applyLaunchAtLoginDefault(applies: true)
        first.rebuildMenu()
        #expect(item.registers == 1 && item.status == .enabled && first.loginChoiceRecorded)
        #expect(try loginItem(first).state == .on)
        let settings = try #require(first.menu.settings)
        let toggle = try #require(settings.items.first { $0.title == AppDelegate.loginTitle })
        settings.performActionForItem(at: settings.index(of: toggle))
        #expect(item.status == .notRegistered && item.unregisters == 1)
        let second = await launch(item)
        second.applyLaunchAtLoginDefault(applies: true)
        second.rebuildMenu()
        #expect(item.registers == 1 && item.status == .notRegistered, "a later launch does not re-enable it")
        #expect(try loginItem(second).state == .off)
    }

    /// The first-run default runs only for a release app with no `--smoke` flag and no support-folder override (a lab or
    /// smoke copy): debug builds, smoke runs and overridden folders never register a login item.
    @Test func theFirstRunDefaultRunsOnlyForAReleaseAppWithItsOwnFolder() {
        #expect(AppDelegate.launchDefaultApplies(debugBuild: false, arguments: ["Arsenal"], environment: [:]))
        #expect(!AppDelegate.launchDefaultApplies(debugBuild: true, arguments: ["Arsenal"], environment: [:]), "a debug build never registers")
        #expect(!AppDelegate.launchDefaultApplies(debugBuild: false, arguments: ["Arsenal", "--smoke"], environment: [:]), "smoke never registers")
        let overridden = ["ARSENAL_SUPPORT_DIR": "/tmp/x"]
        #expect(!AppDelegate.launchDefaultApplies(debugBuild: false, arguments: ["Arsenal"], environment: overridden), "an overridden folder never registers")
    }
}
