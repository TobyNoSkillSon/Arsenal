// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore

/// `Arsenal --real-cycle-v21 <model> <dir>`: the v2.1 browser path through the real service: the browser's Add, the
/// table's Get and Load, then Remove from the browser (a loaded model unloads first; its weights are kept, as the
/// alert's Keep weights). Renders the browser and the table at each step and prints one line per step. GPU: run it only
/// under `scripts/mlock gpulock` (alarm ≤ 180 s).
@MainActor final class RealCycleV21Delegate: NSObject, NSApplicationDelegate, ExitCoding {
    let model: String, directory: URL
    var exitCode: Int32 = 1
    private var app: AppDelegate!
    private var browser: BrowserController!
    private var window: BrowserWindowController!
    private let log = QALog()
    init(model: String, directory: URL) { self.model = model; self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.appearance = Appearance.shared.nsAppearance
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            let (service, error) = await ServiceFactory.make(paths: ArsenalPaths())
            app = AppDelegate(service: service, paths: ArsenalPaths())
            do {
                if let error { throw QAFailure(error) }
                try await cycle(); exitCode = 0
            } catch { log.say("FAILED: \(TableController.message(error))") }
            window?.close()
            await service.shutdown()
            log.write(to: directory.appendingPathComponent("v21-real-cycle.txt"))
            QA.stopApp()
        }
    }

    private var table: TableController { app.table }
    private var row: ModelRow? { table.rows.first { $0.id == model } }
    private var item: BrowserItem? { browser.items.first { $0.id == model } }

    private func cycle() async throws {
        await table.refresh()
        browser = BrowserController(service: app.service)
        browser.confirmRemove = { _ in .keepWeights }
        await browser.refresh()
        window = BrowserWindowController(controller: browser)
        guard let first = item else { throw QAFailure("\(model) is not in the catalog") }
        log.say("opened: \(table.rows.count) rows in the table; browser \(browser.count); \(first.name) · \(first.downloadText) · button \(first.added ? "Remove" : "Add")")
        try await capture("before")

        browser.add(model)
        try await QA.wait(timeout: 10) {
            await self.table.refresh(); return self.row != nil
        }
        await browser.refresh()
        log.say("added from the browser: table has \(table.rows.map(\.name).joined(separator: ", ")); browser button \(item?.added == true ? "Remove" : "Add")")

        table.confirmGet = { _, _ in true }
        table.confirmCode = { _ in true }
        if table.presentation(row!).action == .get {
            log.say("get: \(table.presentation(row!).actionTitle)")
            table.perform(row!)
            try await QA.follow(table, model: model, log: log, step: "get", timeout: 150, until: { $0.action == .load && $0.busyText == nil })
            try await Task.sleep(for: .seconds(1))
        }
        table.perform(row!)
        try await QA.follow(table, model: model, log: log, step: "load", timeout: 120, until: { $0.action == .unload })
        let loaded = table.presentation(row!)
        log.say("loaded \(loaded.loadedPrecision ?? "?"), \(loaded.fast ? "Fast" : "Exact"); runner \(row!.runner), resident \(row!.residentRunner ?? "-")")
        await browser.refresh()
        browser.expanded = [model]
        if let d = browser.detail(model) {
            log.say(
                "browser detail: sizes " + d.sizes.map { "\($0.dtype) \($0.size) \($0.note)" }.joined(separator: " | ") + "; runner "
                    + d.runners.map { "\($0.name) [\($0.engine)] \($0.condition)" }.joined(separator: " | ") + (d.note.map { "; note \($0)" } ?? ""))
        }
        try await capture("loaded")
        try await renderTable("loaded")

        // Remove from the browser: the service unloads it first; Keep weights.
        browser.remove(model)
        try await QA.wait(timeout: 30) {
            await self.table.refresh(); return self.row == nil
        }
        await browser.refresh()
        log.say(
            "removed from the browser: table has \(table.rows.isEmpty ? "no rows" : table.rows.map(\.name).joined(separator: ", ")); browser button \(item?.added == true ? "Remove" : "Add"); weights kept: \(item?.weightsPresent == true)"
        )
        try await capture("removed")
        try await renderTable("removed")
        if item?.added != false { throw QAFailure("the browser still shows the model as added") }
        try QA.noWorkersLeft(after: "Remove", log: log)
        log.say("done")
    }

    private func renderTable(_ name: String) async throws {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            RenderDelegate.renderTable(table, to: directory.appendingPathComponent("v21-real-table-\(name).png")) { done.resume() }
        }
    }
    private func capture(_ name: String) async throws {
        guard let w = window.window else { return }
        window.applyAppearance()
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
        w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await Task.sleep(for: .milliseconds(600))
        guard let rep = WindowSnapshot.capture(w) else { return }
        try rep.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("v21-real-browser-\(name).png"))
        w.orderOut(nil)
    }
}
