// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Synchronization
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct MenuCloseLogTests {
    private func support() -> URL { URL(fileURLWithPath: "/private/tmp/arsenal-r12-menu-stability/close-\(UUID())") }
    private func records(_ log: MenuDiagnostics) throws -> [[String: Any]] {
        try String(contentsOf: log.url, encoding: .utf8).split(separator: "\n").map {
            try #require(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
        }.filter { ($0["record"] as? String) == "close" }
    }

    @Test func everyProductionMenuLogsOneCloseAndRepeatedOpensGetDistinctIDs() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.loaded), supportDirectory: support), paths: ArsenalPaths(support: support))
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        let log = app.menuDiagnostics
        var time = 100.0
        log.trackingRecorder.uptime = { time }; log.trackingRecorder.currentEvent = { nil }
        // Every submenu the app builds, nested ones included: Models, Activity, Settings, and Keep Loaded, Memory and the
        // agents' Voice inside Settings.
        let submenus = app.menu.everyItem.compactMap(\.submenu)
        #expect(submenus.count == 6)
        let unknown = NSMenuItem(title: "PRIVATE-MENU-TITLE", action: nil, keyEquivalent: "")
        unknown.submenu = NSMenu(title: "PRIVATE-MENU-TITLE")
        app.menu.addItem(unknown); log.observeSubmenus(in: app.menu)
        let menus = [app.menu] + submenus + [try #require(unknown.submenu)]
        for menu in menus {
            menu.delegate?.menuWillOpen?(menu); menu.delegate?.menuWillOpen?(menu)
            time += 2
            menu.delegate?.menuDidClose?(menu); menu.delegate?.menuDidClose?(menu)
            await log.flush()
        }
        let first = try records(log)
        #expect(first.count == 8)
        #expect(Set(first.compactMap { $0["menu"] as? String }) == Set(["Main", "Models", "Activity", "Settings", "Voice", "Keep Hot", "Memory", "Submenu"]))
        #expect(first.allSatisfy { ($0["openSeconds"] as? Double) == 2 && ($0["structuralChange"] as? Bool) == false })
        #expect(Set(first.compactMap { $0["menuID"] as? Int }).count == 8)
        let voice = try #require(submenus.first { ($0.delegate as? MenuTrackingDelegate)?.name == .voice })
        voice.delegate?.menuWillOpen?(voice); time += 3; voice.delegate?.menuDidClose?(voice)
        await log.flush()
        let last = try #require(records(log).last)
        #expect(last["openSeconds"] as? Double == 3)
        #expect(last["menuID"] as? Int == first.first { ($0["menu"] as? String) == "Voice" }?["menuID"] as? Int)
        #expect(Set(try records(log).compactMap { $0["openID"] as? Int }).count == 9)
        #expect(try !String(contentsOf: log.url, encoding: .utf8).contains("PRIVATE-MENU-TITLE"))
    }

    @Test func closeCapturesCurrentEventButDoesNotInventItsCause() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support), menu = NSMenu()
        log.observeMainMenu(menu)
        var time = 20.0
        let event = try #require(
            NSEvent.mouseEvent(
                with: .leftMouseUp, location: NSPoint(x: 123, y: 456), modifierFlags: [], timestamp: 21, windowNumber: 0, context: nil, eventNumber: 1, clickCount: 1, pressure: 0))
        log.trackingRecorder.uptime = { time }; log.trackingRecorder.currentEvent = { event }
        menu.delegate?.menuWillOpen?(menu); time = 22; menu.delegate?.menuDidClose?(menu)
        await log.flush()
        let close = try #require(records(log).first)
        #expect(close["openSeconds"] as? Double == 2 && close["cause"] as? String == "unknown")
        let captured = try #require(close["event"] as? [String: Any])
        #expect(captured["type"] as? Int == Int(NSEvent.EventType.leftMouseUp.rawValue))
        #expect(captured["location"] as? [Double] == [123, 456])
        #expect(captured["duringOpen"] as? Bool == true && close["eventEvidence"] as? String == "current-at-menuDidClose")
        // The exposed currentEvent can be stale. Record that fact, rather than treating it as the ending event.
        time = 30; menu.delegate?.menuWillOpen?(menu); time = 31; menu.delegate?.menuDidClose?(menu)
        await log.flush()
        #expect((try records(log).last?["event"] as? [String: Any])?["duringOpen"] as? Bool == false)
        print("sample close-log: " + String(data: try JSONSerialization.data(withJSONObject: close, options: [.sortedKeys]), encoding: .utf8)!)
    }

    @Test func nativeActionAfterDidCloseAndDeactivationAreRecordedSeparately() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.loaded), supportDirectory: support), paths: ArsenalPaths(support: support))
        defer { app.table.stop(); app.tools.closeAll() }
        let playback = FixturePlayback(); app.tools.playback = playback
        await app.table.refresh(); await app.refreshMenuState()
        let log = app.menuDiagnostics
        log.trackingRecorder.currentEvent = { nil }
        app.menuWillOpen(app.menu); app.menuDidClose(app.menu)
        // Use AppKit's action dispatch, not a fabricated action notification. It legitimately follows close.
        // Mute Agent Speech lives in Settings now: its action is dispatched from that submenu.
        let settings = try #require(app.menu.settings)
        settings.performActionForItem(at: try #require(settings.items.firstIndex { $0.title == "Mute Agent Speech" }))
        #expect(playback.muted)
        await log.flush()
        let action = try #require(records(log).first)
        #expect(action["cause"] as? String == "item-action" && action["itemAction"] as? String == "toggleMute")
        #expect(action["structuralChange"] as? Bool == false, "The action's post-close rebuild is outside the open interval")
        app.menuWillOpen(app.menu)
        NotificationCenter.default.post(name: NSApplication.didResignActiveNotification, object: NSApp)
        app.menuDidClose(app.menu); await log.flush()
        let deactivated = try #require(records(log).last)
        #expect(deactivated["cause"] as? String == "app-deactivation" && deactivated["appDeactivated"] as? Bool == true)
        app.menuWillOpen(app.menu); app.menuDidClose(app.menu)
        NotificationCenter.default.post(name: NSApplication.didResignActiveNotification, object: NSApp)
        await log.flush()
        let afterClose = try #require(records(log).last)
        #expect(afterClose["cause"] as? String == "unknown" && afterClose["appDeactivated"] as? Bool == false)
        #expect(afterClose["appDeactivatedAfterClose"] as? Bool == true)
    }

    @Test func everyWindowCancellationPathRecordsItsFixedCallSite() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let app = AppDelegate(service: FixtureService(rows: Fixtures.rows(.loaded), supportDirectory: support), paths: ArsenalPaths(support: support))
        defer { app.table.stop(); app.tools.closeAll() }
        await app.table.refresh(); await app.refreshMenuState()
        let menu = try #require(app.modelsMenu.modelItem().submenu)
        let row = try #require(app.table.rows.first)
        let plan = GetPlan(id: row.id, dtype: "bf16", runner: "native", items: [])
        app.modelsMenu.presentAlert = { _ in .alertFirstButtonReturn }
        app.modelsMenu.presentOpenPanel = { _ in nil }; app.modelsMenu.showBrowser = {}
        let sites: [MenuCancellationSite] = [.removeConfirmation, .codeConfirmation, .filePicker, .modelBrowser, .getConfirmation, .unknownGetSize, .memoryWarning]
        for site in sites {
            app.menuWillOpen(app.menu); menu.delegate?.menuWillOpen?(menu)
            switch site {
            case .removeConfirmation: _ = app.table.confirmRemove(row)
            case .codeConfirmation: _ = app.table.confirmCode(row)
            case .filePicker: _ = app.table.pickFile(row, SettingState(declaration: ["id": "path", "title": "PRIVATE-SETTING-TEXT", "type": "path"], value: ""))
            case .modelBrowser: app.table.openBrowser()
            case .getConfirmation: _ = app.table.confirmGet(row, plan)
            case .unknownGetSize: app.table.unknownGetSize(row, plan)
            case .memoryWarning: app.modelsMenu.dismiss(.memoryWarning)
            }
            menu.delegate?.menuDidClose?(menu); app.menuDidClose(app.menu)
            await app.menuDiagnostics.flush()
            let pair = try records(app.menuDiagnostics).suffix(2)
            #expect(pair.count == 2)
            #expect(pair.allSatisfy { ($0["cause"] as? String) == "cancelTracking" && ($0["cancelTrackingSites"] as? [String]) == [site.rawValue] }, "\(site.rawValue): \(pair)")
        }
        // Seven Models/Main pairs.
        let all = try records(app.menuDiagnostics)
        #expect(all.count == 14)
        #expect(try !String(contentsOf: app.menuDiagnostics.url, encoding: .utf8).contains("PRIVATE-SETTING-TEXT"))
    }

    @Test func structuralCallsAreRecordedEvenWhenTheFinalIdentityListIsUnchanged() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support), menu = NSMenu()
        let item = NSMenuItem(title: "PRIVATE-TITLE", action: NSSelectorFromString("PRIVATE-SELECTOR:"), keyEquivalent: "")
        item.representedObject = "PRIVATE-REPRESENTED-OBJECT"
        menu.addItem(item); log.observeMainMenu(menu)
        menu.delegate?.menuWillOpen?(menu)
        item.title = "PRIVATE-NEW-TITLE"; item.isEnabled = false
        menu.delegate?.menuDidClose?(menu); await log.flush()
        #expect(try records(log).last?["structuralChange"] as? Bool == false)
        menu.delegate?.menuWillOpen?(menu)
        menu.removeItem(item); menu.insertItem(item, at: 0)
        menu.delegate?.menuDidClose?(menu); await log.flush()
        #expect(try records(log).last?["structuralChange"] as? Bool == true)
        #expect(try records(log).last?["structuralCalls"] as? Int == 2)
        menu.delegate?.menuWillOpen?(menu)
        menu.removeAllItems() // AppKit explicitly omits item-removal notifications for removeAllItems.
        menu.delegate?.menuDidClose?(menu); await log.flush()
        #expect(try records(log).last?["structuralChange"] as? Bool == true)
        #expect(try !String(contentsOf: log.url, encoding: .utf8).contains("PRIVATE-"))
    }

    @Test func closeWritesReturnWhileDiskIsBlockedAndRemainBoundedOffMain() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let accesses = Mutex<[Bool]>([]), gate = DispatchSemaphore(value: 0)
        let log = MenuDiagnostics(
            support: support,
            onDiskAccess: {
                let first = accesses.withLock { calls in
                    calls.append(Thread.isMainThread); return calls.count == 1
                }
                if first { _ = gate.wait(timeout: .now() + 5) }
            })
        defer { gate.signal() }
        let menu = NSMenu(); log.observeMainMenu(menu)
        for _ in 0..<100 { menu.delegate?.menuWillOpen?(menu); menu.delegate?.menuDidClose?(menu); log.trackingRecorder.writeClosed() }
        try await waitUntil { accesses.withLock { !$0.isEmpty } }
        #expect(!FileManager.default.fileExists(atPath: log.url.path))
        gate.signal(); await log.flush()
        let closes = try records(log)
        #expect(closes.count == 100)
        #expect(closes.compactMap { $0["openID"] as? Int } == Array(1...100))
        #expect(try Data(contentsOf: log.url).count <= MenuDiagnostics.maximumBytes)
        #expect(accesses.withLock { $0.allSatisfy { !$0 } })
    }

    @Test func diagnosticDelegateForwardsOriginalLifecycleAndOptionalMethods() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support), menu = NSMenu(), original = DelegateProbe()
        menu.delegate = original; log.observeMainMenu(menu)
        log.observeMainMenu(menu) // Re-observation must not stack wrappers.
        menu.delegate?.menuWillOpen?(menu); menu.delegate?.menuNeedsUpdate?(menu); menu.delegate?.menuDidClose?(menu)
        await log.flush()
        #expect(original.opens == 1 && original.closes == 1 && original.updates == 1)
        #expect(try records(log).count == 1)
    }

    @Test func nativeTrackingEndRetainsAnActionDeliveredAfterAQueueTurn() async throws {
        let support = support()
        defer { try? FileManager.default.removeItem(at: support) }
        let log = MenuDiagnostics(support: support), menu = NSMenu(), original = DelegateProbe()
        menu.delegate = original
        let item = NSMenuItem(title: "PRIVATE-NATIVE-ACTION", action: #selector(DelegateProbe.privateAction(_:)), keyEquivalent: "")
        let child = NSMenu(), childDelegate = DelegateProbe()
        child.delegate = childDelegate
        let root = NSMenuItem(title: "PRIVATE-VOICE-ROOT", action: nil, keyEquivalent: "")
        root.identifier = MenuDiagnosticName.voice.itemIdentifier; root.submenu = child
        item.target = original; menu.addItem(item); menu.addItem(root)
        log.observeMainMenu(menu); log.observeSubmenus(in: menu)
        log.trackingRecorder.currentEvent = { nil }
        NotificationCenter.default.post(name: NSMenu.didBeginTrackingNotification, object: menu)
        menu.delegate?.menuWillOpen?(menu)
        NotificationCenter.default.post(name: NSMenu.didBeginTrackingNotification, object: child)
        child.delegate?.menuWillOpen?(child); child.delegate?.menuDidClose?(child); menu.delegate?.menuDidClose?(menu)
        NotificationCenter.default.post(name: NSMenu.didEndTrackingNotification, object: child)
        await log.flush()
        #expect(!FileManager.default.fileExists(atPath: log.url.path), "A child's end must not prematurely end the root tracking cycle")
        for _ in 0..<5 { await Task.yield() }
        menu.performActionForItem(at: 0)
        NotificationCenter.default.post(name: NSMenu.didEndTrackingNotification, object: menu)
        await log.flush()
        let close = try #require(records(log).first)
        #expect(try records(log).count == 2 && original.actions == 1)
        #expect(close["cause"] as? String == "item-action" && close["itemAction"] as? String == "other")
        #expect(close["trackingEndObserved"] as? Bool == true)
        #expect(try !String(contentsOf: log.url, encoding: .utf8).contains("PRIVATE-"))
    }
}

@MainActor private final class DelegateProbe: NSObject, NSMenuDelegate {
    var opens = 0, closes = 0, updates = 0, actions = 0
    func menuWillOpen(_ menu: NSMenu) { opens += 1 }
    func menuDidClose(_ menu: NSMenu) { closes += 1 }
    func menuNeedsUpdate(_ menu: NSMenu) { updates += 1 }
    @objc func privateAction(_ sender: NSMenuItem) { actions += 1 }
}
