// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore

/// `Arsenal --smoke`: the app's real objects on the stub service in a private support dir; checks the status item, the
/// menu and the table's first layout, prints one line, exits.
@MainActor final class SmokeDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    var exitCode: Int32 = 1
    private var app: AppDelegate?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-smoke-\(ProcessInfo.processInfo.processIdentifier)")
        let paths = ArsenalPaths(support: support)
        let app = AppDelegate(service: FixtureService(rows: []), paths: paths)
        self.app = app
        app.applicationDidFinishLaunching(notification)
        Task { @MainActor in
            await app.table.refresh()
            await app.refreshMenuState()
            app.rebuildMenu()
            // The API the app hosts: up within 5 s and answering /arsenal/v1/models with the table's rows.
            var apiRows: Int?
            for _ in 0..<50 where apiRows == nil {
                if let port = app.api.port { apiRows = await Self.apiRowCount(port: port) }
                if apiRows == nil { try? await Task.sleep(for: .milliseconds(100)) }
            }
            var line = Self.check(app)
            let apiOK = apiRows == app.table.rows.count
            line = (line.ok && apiOK, line.text + ", API port \(app.api.port.map(String.init) ?? "none") \(apiRows.map { "\($0) rows" } ?? "not answering")")
            if !apiOK { line.text = line.text.replacingOccurrences(of: "smoke ok", with: "smoke FAILED") }
            print(line.text)
            app.api.stop()
            if let status = app.status { NSStatusBar.system.removeStatusItem(status) }
            try? FileManager.default.removeItem(at: support)
            exitCode = line.ok ? 0 : 1
            AppRun.stop()
        }
    }

    /// Rows the control API lists (`GET /arsenal/v1/models`), whatever its JSON wraps them in.
    static func apiRowCount(port: Int) async -> Int? {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/models")!, timeoutInterval: 2)
        request.setValue("127.0.0.1:\(port)", forHTTPHeaderField: "Host")
        guard let (data, response) = try? await URLSession.shared.data(for: request), (response as? HTTPURLResponse)?.statusCode == 200,
            let object = try? JSONSerialization.jsonObject(with: data)
        else { return nil }
        if let array = object as? [Any] { return array.count }
        if let dict = object as? [String: Any] { return ((dict["models"] ?? dict["data"] ?? dict["rows"]) as? [Any])?.count }
        return nil
    }

    /// The fresh install's menu in order, separators and hidden items left out. No rows, so no task has an entry (no "Get
    /// a model" placeholders) and nothing is Recent. Strict: a changed title, a missing row or a moved one fails the smoke.
    static let expectedMenuTitles = ["Models", "Activity", "Settings", "About Arsenal", AppDelegate.supportTitle, "Quit Arsenal"]
    /// The fresh install's Settings submenu in order, separators left out (no voice model is loaded).
    static let expectedSettingsTitles = [
        MenuContent.keepLoadedTitle, "Memory", "Mute Agent Speech", HuggingFaceToken.title, AppDelegate.loginTitle, MenuContent.openFiles
    ]
    /// True when the menu's titles are exactly the expected list.
    static func menuMatches(_ titles: [String]) -> Bool { titles == expectedMenuTitles }
    /// True when the Settings submenu's titles are exactly the expected list; Launch at Login may show its approval title.
    static func settingsMatch(_ titles: [String]) -> Bool {
        titles.map { $0 == AppDelegate.loginApprovalTitle ? AppDelegate.loginTitle : $0 } == expectedSettingsTitles
    }

    static func check(_ app: AppDelegate) -> (ok: Bool, text: String) {
        // What the menu shows: a hidden item (Stop Speaking while nothing speaks) is not part of it.
        let titles = app.menu.items.filter { !$0.isSeparatorItem && !$0.isHidden }.map(\.title)
        let settings = app.menu.items.first { $0.title == "Settings" }?.submenu?.items.filter { !$0.isSeparatorItem && !$0.isHidden }.map(\.title) ?? []
        let tableItem = app.menu.items.first { $0.title == "Models" }?.submenu?.items.first
        let view = tableItem?.view
        view.map { v in
            let window = NSWindow(contentRect: v.frame, styleMask: .borderless, backing: .buffered, defer: false)
            window.contentView = v; v.layoutSubtreeIfNeeded()
        }
        var actions = 0, headers = 0, pinnedRows = 0, refreshControls = 0
        func walk(_ v: NSView) {
            if v is RowActionView { actions += 1 }
            if v is RefreshControlView { refreshControls += 1 }
            if v is GroupHeaderView { headers += 1 }
            if v is PinnedRowView { pinnedRows += 1 }
            v.subviews.forEach(walk)
        }
        if let view { walk(view) }
        let groups = app.table.groups.count, rows = app.table.rows.count
        let ok =
            app.status?.button != nil
            && menuMatches(titles) && settingsMatch(settings)
            && app.menu.items.first { $0.title == "Models" }?.submenu?.items.count == 1
            && view != nil && rows == 0 && groups == 0 && actions == 0 && headers == 0 && pinnedRows == 1 && refreshControls == 1
        let size = view.map { "\(Int($0.frame.width))x\(Int($0.frame.height))" } ?? "none"
        return (
            ok,
            "smoke \(ok ? "ok" : "FAILED"): status item \(app.status?.button != nil ? "yes" : "no"), menu \(titles.joined(separator: " | ")), settings \(settings.joined(separator: " | ")), table \(size), \(groups) groups, \(rows) rows, \(actions) action cells, \(pinnedRows) pinned Get more models row"
        )
    }
}

/// `Arsenal --smoke` reports through the process's exit code.
@MainActor protocol ExitCoding { var exitCode: Int32 { get } }

@MainActor enum AppRun {
    /// Ends `NSApp.run()` after an async run; `stop` takes effect after the next event, so one is posted.
    static func stop() {
        NSApp.stop(nil)
        NSApp.postEvent(
            NSEvent.otherEvent(
                with: .applicationDefined, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0,
                context: nil, subtype: 0, data1: 0, data2: 0)!, atStart: false)
    }
}
