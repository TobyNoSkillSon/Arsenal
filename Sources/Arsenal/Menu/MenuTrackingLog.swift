// SPDX-License-Identifier: MIT
import AppKit

/// Code-owned call sites, never a caller-supplied string or model/setting name.
enum MenuCancellationSite: String {
    case removeConfirmation, codeConfirmation, filePicker, modelBrowser, getConfirmation, unknownGetSize, memoryWarning
}

struct MenuTrackingStructure: Equatable {
    let item: ObjectIdentifier, submenu: ObjectIdentifier?, view: ObjectIdentifier?
    let hidden: Bool
}

extension NSMenu {
    @MainActor func cancelTrackingFrom(_ site: MenuCancellationSite) {
        (delegate as? MenuTrackingDelegate)?.recorder?.cancelled(self, at: site)
        cancelTracking()
    }
}

/// Event-driven evidence only. Delegate forwarding leaves each menu's existing behavior intact.
@MainActor final class MenuTrackingRecorder: NSObject {
    private weak var diagnostics: MenuDiagnostics?
    private let delegates = NSMapTable<NSMenu, MenuTrackingDelegate>.weakToStrongObjects()
    private var nextMenuID = 0, nextOpenID = 0
    private var open: [ObjectIdentifier: Session] = [:]
    private var closing: [Session] = []
    private var nativeTracking: [ObjectIdentifier: ObjectIdentifier] = [:]
    private var terminating = false
    var uptime: () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
    var currentEvent: () -> NSEvent? = { NSApp.currentEvent }

    @MainActor private final class Session {
        let menu: NSMenu, root: NSMenu, name: MenuDiagnosticName, menuID: Int, openID: Int, started: TimeInterval
        let initialStructure: [MenuTrackingStructure]
        var ended: TimeInterval?
        var event: [String: Any]?
        var trackingEndEvent: [String: Any]?
        var trackingEndObserved = false
        var cancellations: [MenuCancellationSite] = []
        var action: String?
        var deactivated = false, deactivatedAfterClose = false, structuralChange = false
        var structuralCalls = 0
        init(menu: NSMenu, delegate: MenuTrackingDelegate, openID: Int, started: TimeInterval) {
            self.menu = menu; root = MenuTrackingRecorder.root(menu); name = delegate.name; menuID = delegate.menuID; self.openID = openID; self.started = started
            initialStructure = MenuTrackingRecorder.structure(menu)
        }
    }

    init(diagnostics: MenuDiagnostics) {
        self.diagnostics = diagnostics
        super.init()
        let center = NotificationCenter.default
        for name in [NSMenu.didAddItemNotification, NSMenu.didRemoveItemNotification, NSMenu.didChangeItemNotification] {
            center.addObserver(self, selector: #selector(menuChanged(_:)), name: name, object: nil)
        }
        center.addObserver(self, selector: #selector(itemAction(_:)), name: NSMenu.willSendActionNotification, object: nil)
        center.addObserver(self, selector: #selector(trackingBegan(_:)), name: NSMenu.didBeginTrackingNotification, object: nil)
        center.addObserver(self, selector: #selector(trackingEnded(_:)), name: NSMenu.didEndTrackingNotification, object: nil)
        center.addObserver(self, selector: #selector(deactivated), name: NSApplication.didResignActiveNotification, object: nil)
    }
    deinit { NotificationCenter.default.removeObserver(self) }

    func observe(_ menu: NSMenu, name: MenuDiagnosticName) {
        if let existing = delegates.object(forKey: menu) {
            if menu.delegate !== existing { existing.original = menu.delegate; menu.delegate = existing }
            return
        }
        nextMenuID += 1
        let proxy = MenuTrackingDelegate(recorder: self, original: menu.delegate, name: name, menuID: nextMenuID)
        proxy.lastStructure = Self.shallowStructure(menu)
        delegates.setObject(proxy, forKey: menu)
        menu.delegate = proxy
    }

    func identifiers(_ menu: NSMenu) -> [String: Int] {
        guard let delegate = delegates.object(forKey: menu) else { return [:] }
        var ids = ["menuID": delegate.menuID]
        if let session = open[ObjectIdentifier(menu)] { ids["openID"] = session.openID }
        return ids
    }

    func begin(_ menu: NSMenu) {
        guard !terminating, open[ObjectIdentifier(menu)] == nil, let delegate = delegates.object(forKey: menu) else { return }
        synchronize(menu)
        nextOpenID += 1
        open[ObjectIdentifier(menu)] = Session(menu: menu, delegate: delegate, openID: nextOpenID, started: uptime())
        if delegate.original == nil { diagnostics?.recordAfterOpen(menu: menu, name: delegate.name) }
    }

    func end(_ menu: NSMenu) {
        guard !terminating, let session = open.removeValue(forKey: ObjectIdentifier(menu)) else { return }
        session.ended = uptime()
        session.structuralChange = session.structuralChange || Self.structure(menu) != session.initialStructure
        session.event = eventData(for: session)
        closing.append(session)
        // Native action notifications may follow menuDidClose. A native root close waits for didEndTracking,
        // then one main-queue turn; a submenu hidden while its parent stays open can be written immediately.
        DispatchQueue.main.async { [weak self] in self?.writeClosed() }
    }

    private func eventData(for session: Session) -> [String: Any]? {
        guard let event = currentEvent() else { return nil }
        // AppKit exposes currentEvent, not a guaranteed causal ending event. No characters or key codes.
        let point = event.window.map { $0.convertPoint(toScreen: event.locationInWindow) } ?? event.locationInWindow
        return [
            "type": event.type.rawValue, "location": [point.x, point.y], "coordinates": "AppKit-screen",
            "duringOpen": event.timestamp >= session.started && event.timestamp <= (session.ended ?? uptime())
        ]
    }

    @objc private func trackingBegan(_ notification: Notification) {
        guard let menu = notification.object as? NSMenu, delegates.object(forKey: menu) != nil else { return }
        nativeTracking[ObjectIdentifier(menu)] = ObjectIdentifier(Self.root(menu))
    }
    @objc private func trackingEnded(_ notification: Notification) {
        guard let menu = notification.object as? NSMenu, delegates.object(forKey: menu) != nil else { return }
        nativeTracking.removeValue(forKey: ObjectIdentifier(menu))
        for session in Array(open.values) + closing where session.menu === menu || session.root === menu {
            session.trackingEndObserved = true; session.trackingEndEvent = eventData(for: session)
        }
        DispatchQueue.main.async { [weak self] in self?.writeClosed() }
    }

    func cancelled(_ menu: NSMenu, at site: MenuCancellationSite) {
        for session in related(to: menu) where session.ended == nil {
            if !session.cancellations.contains(site) { session.cancellations.append(site) }
        }
    }

    @objc private func itemAction(_ notification: Notification) {
        guard let menu = notification.object as? NSMenu, let item = notification.userInfo?["MenuItem"] as? NSMenuItem else { return }
        // Whitelist code selectors: contributed titles, representedObject and saved voice IDs stay private.
        let selector = item.action.map(NSStringFromSelector) ?? ""
        let allowed = [
            "prepareTool:", "openTool:", "chooseVoice:", "toggleMute", "stopSpeaking", "updateClicked", "runAction:", "openFiles", "editHuggingFaceToken", "toggleLogin", "about",
            "quit", "chooseIdle:",
            "chooseSwap:", "chooseWhenFull:", "unloadAll"
        ]
        let action = allowed.contains(selector) ? selector : "other"
        for session in related(to: menu) { session.action = action }
    }

    @objc private func deactivated() {
        for session in open.values { session.deactivated = true }
        for session in closing { session.deactivatedAfterClose = true }
    }

    @objc private func menuChanged(_ notification: Notification) {
        guard let menu = notification.object as? NSMenu, let delegate = delegates.object(forKey: menu) else { return }
        guard open.values.contains(where: { $0.root === Self.root(menu) }) else { return }
        let structure = Self.shallowStructure(menu)
        let changed = notification.name != NSMenu.didChangeItemNotification || structure != delegate.lastStructure
        delegate.lastStructure = structure
        guard changed else { return }
        // Only changes made before didClose count. An item's action can update the menu after it closes.
        for session in open.values where session.root === Self.root(menu) {
            session.structuralChange = true; session.structuralCalls += 1
        }
    }

    private func related(to menu: NSMenu) -> [Session] {
        let root = Self.root(menu)
        return (Array(open.values) + closing).filter { $0.root === root }
    }
    private static func root(_ menu: NSMenu) -> NSMenu { menu.supermenu.map(root) ?? menu }
    private func synchronize(_ menu: NSMenu) {
        delegates.object(forKey: menu)?.lastStructure = Self.shallowStructure(menu)
        for item in menu.items { if let sub = item.submenu { synchronize(sub) } }
    }
    private static func entry(_ item: NSMenuItem) -> MenuTrackingStructure {
        .init(item: ObjectIdentifier(item), submenu: item.submenu.map(ObjectIdentifier.init), view: item.view.map(ObjectIdentifier.init), hidden: item.isHidden)
    }
    private static func shallowStructure(_ menu: NSMenu) -> [MenuTrackingStructure] { menu.items.map(entry) }
    private static func structure(_ menu: NSMenu) -> [MenuTrackingStructure] {
        menu.items.flatMap { item in [entry(item)] + (item.submenu.map(structure) ?? []) }
    }

    func writeClosed(until deadline: DispatchTime? = nil) {
        guard !terminating else { return }
        let records = closing; closing = []
        for session in records {
            if let deadline, DispatchTime.now() >= deadline { break }
            if deadline == nil, nativeTracking.values.contains(ObjectIdentifier(session.root)), open[ObjectIdentifier(session.root)] == nil {
                closing.append(session); continue
            }
            let cause = !session.cancellations.isEmpty ? "cancelTracking" : session.action != nil ? "item-action" : session.deactivated ? "app-deactivation" : "unknown"
            var record: [String: Any] = [
                "date": ISO8601DateFormatter().string(from: Date()), "record": "close", "menu": session.name.rawValue,
                "menuID": session.menuID, "openID": session.openID,
                "openSeconds": max(0, (session.ended ?? session.started) - session.started), "cause": cause,
                "cancelTrackingSites": session.cancellations.map(\.rawValue), "appDeactivated": session.deactivated,
                "appDeactivatedAfterClose": session.deactivatedAfterClose,
                "trackingEndObserved": session.trackingEndObserved,
                "structuralChange": session.structuralChange, "structuralCalls": session.structuralCalls,
                "eventEvidence": session.event == nil ? "unavailable" : "current-at-menuDidClose"
            ]
            if let event = session.event { record["event"] = event }
            if let event = session.trackingEndEvent { record["trackingEndEvent"] = event }
            if let action = session.action { record["itemAction"] = action }
            do { try diagnostics?.append(record) } catch { diagnostics?.report(error) }
        }
    }

    func terminate(until deadline: DispatchTime) { writeClosed(until: deadline); terminating = true }
}

/// Only lifecycle methods are intercepted. Other optional delegate selectors forward unchanged.
@MainActor final class MenuTrackingDelegate: NSObject, NSMenuDelegate {
    weak var recorder: MenuTrackingRecorder?
    weak var original: (any NSMenuDelegate)?
    let name: MenuDiagnosticName, menuID: Int
    var lastStructure: [MenuTrackingStructure] = []
    init(recorder: MenuTrackingRecorder, original: (any NSMenuDelegate)?, name: MenuDiagnosticName, menuID: Int) {
        self.recorder = recorder; self.original = original; self.name = name; self.menuID = menuID
    }
    func menuWillOpen(_ menu: NSMenu) { recorder?.begin(menu); original?.menuWillOpen?(menu) }
    func menuDidClose(_ menu: NSMenu) { recorder?.end(menu); original?.menuDidClose?(menu) }
    override nonisolated func responds(to selector: Selector!) -> Bool {
        super.responds(to: selector) || MainActor.assumeIsolated { original?.responds(to: selector) == true }
    }
    // This reference leaves the actor only for synchronous Objective-C forwarding, on AppKit's caller.
    private struct ForwardTarget: @unchecked Sendable { let value: any NSMenuDelegate }
    override nonisolated func forwardingTarget(for selector: Selector!) -> Any? {
        let target = MainActor.assumeIsolated { original?.responds(to: selector) == true ? original.map { ForwardTarget(value: $0) } : nil }
        return target?.value ?? super.forwardingTarget(for: selector)
    }
}
