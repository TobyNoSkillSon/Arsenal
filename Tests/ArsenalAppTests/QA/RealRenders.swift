// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore

/// `Arsenal --render-real <dir>`: the real signed catalog (ARSENAL_CATALOG, ARSENAL_SUPPORT_DIR as the app uses them)
/// rendered as its table and menu show it, with every tooltip. Opens the service only: no Get, no worker, no GPU.
@MainActor final class RealRenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private var app: AppDelegate?
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.appearance = NSAppearance(named: .darkAqua)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            let paths = ArsenalPaths()
            let (service, error) = await ServiceFactory.make(paths: paths)
            let app = AppDelegate(service: service, paths: paths)
            self.app = app
            await app.table.refresh(); await app.refreshMenuState()
            if let error { print(error) } else { exitCode = 0 }
            let name = ProcessInfo.processInfo.environment["ARSENAL_RENDER_NAME"] ?? "first-launch"
            try? RenderDelegate.tooltips(app.table).write(to: directory.appendingPathComponent("real-\(name)-tooltips.txt"), atomically: true, encoding: .utf8)
            print(
                "real catalog: \(app.table.rows.count) models in \(app.table.groups.count) groups: "
                    + app.table.groups.map { "\($0.kind.title) \($0.rows.count)" }.joined(separator: ", "))
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                RenderDelegate.renderTable(app.table, to: directory.appendingPathComponent("real-table-\(name).png")) { done.resume() }
            }
            app.rebuildMenu()
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("real-menu-\(name).png")) { done.resume() }
            }
            QA.stopApp()
        }
    }
}
