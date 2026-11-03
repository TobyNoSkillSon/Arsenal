// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore

/// `Arsenal --real-cycle-v2 <model> <dir> <change>`: the v2 path through the real service, driven by the app's own
/// controllers: the browser's Add, then the table's Get and Load, a change that turns the loaded row red
/// (`dtype:int8`, a dtype cell; or `setting:<key>=<number>`, a load-time setting), the row's Reload, and Unload.
/// Renders each state and prints one line per step. GPU: run it only under `scripts/mlock gpulock` (alarm ≤ 180 s);
/// rehearse off the lock with ARSENAL_HELPERS pointing at stub-worker wrappers.
@MainActor final class RealCycleV2Delegate: NSObject, NSApplicationDelegate, ExitCoding {
    let model: String, directory: URL, change: String
    var exitCode: Int32 = 1
    private var app: AppDelegate!
    private let log = QALog()
    private var rendered: Set<String> = []
    init(model: String, directory: URL, change: String) { self.model = model; self.directory = directory; self.change = change }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.appearance = NSAppearance(named: .darkAqua)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            let (service, error) = await ServiceFactory.make(paths: ArsenalPaths())
            app = AppDelegate(service: service, paths: ArsenalPaths())
            do {
                if let error { throw QAFailure(error) }
                try await cycle(); exitCode = 0
            } catch { log.say("FAILED: \(TableController.message(error))") }
            await service.shutdown()
            log.write(to: directory.appendingPathComponent("real-v2-cycle.txt"))
            QA.stopApp()
        }
    }

    private var table: TableController { app.table }
    private var row: ModelRow? { table.rows.first { $0.id == model } }

    private func cycle() async throws {
        await table.refresh()
        log.say("opened: \(table.rows.count) rows in the table; \(table.catalog.map(FooterText.catalog) ?? "no catalog status")")
        try await render("day-one")

        // Add, as the browser's button does.
        let browser = BrowserController(service: app.service)
        await browser.refresh()
        guard let item = browser.items.first(where: { $0.id == model }) else { throw QAFailure("\(model) is not in the catalog") }
        log.say("browser: \(browser.count); \(item.name) · \(item.downloadText) · \(item.ramText) · \(item.status.rawValue)")
        browser.add(model)
        try await QA.wait(timeout: 10) {
            await self.table.refresh(); return self.row != nil
        }
        log.say("added: table has \(table.rows.map(\.name).joined(separator: ", "))")
        try await render("added")

        // Get and Load through the row's button.
        table.confirmGet = { _, _ in true }
        table.confirmCode = { _ in true }
        if table.presentation(row!).action == .get {
            log.say("get: \(table.presentation(row!).actionTitle)")
            table.perform(row!)
            try await QA.follow(table, model: model, log: log, step: "get", timeout: 150, until: { $0.action == .load && $0.busyText == nil })
            // The service can show the row Ready a moment before its Get task has finished ("Wait for Get to finish").
            try await Task.sleep(for: .seconds(1))
        }
        // A dtype change reloads in place only when that dtype is on this Mac: core's Get unloads a loaded model, so
        // the other dtype is made first (as a user would: click int8, Get, then fp16, Load).
        if change.hasPrefix("dtype:"), let column = PrecisionColumn.of(String(change.dropFirst(6))), let dtype = PrecisionColumn.dtype(row!, column),
            row!.states[dtype]?.state != .ready
        {
            let base = table.presentation(row!).selected
            table.select(row!, column: column)
            log.say("get \(dtype) before loading: \(table.presentation(row!).actionTitle)")
            table.perform(row!)
            try await QA.follow(table, model: model, log: log, step: "get \(dtype)", timeout: 60, until: { $0.action == .load && $0.busyText == nil })
            try await Task.sleep(for: .seconds(1))
            table.select(row!, column: PrecisionColumn.of(base) ?? "16")
        }
        table.perform(row!)
        try await QA.follow(table, model: model, log: log, step: "load", timeout: 120, until: { $0.action == .unload })
        let loaded = table.presentation(row!)
        log.say("loaded \(loaded.loadedPrecision ?? "?"), \(loaded.fast ? "Fast" : "Exact")")
        try await render("loaded")

        // The change that turns the loaded row red.
        if change.hasPrefix("dtype:") {
            let dtype = String(change.dropFirst(6))
            guard let column = PrecisionColumn.of(dtype) else { throw QAFailure("unknown dtype \(dtype)") }
            table.select(row!, column: column)
        } else if change.hasPrefix("setting:"), let eq = change.firstIndex(of: "=") {
            let key = String(change[change.index(change.startIndex, offsetBy: 8)..<eq]), value = Double(change[change.index(after: eq)...]) ?? 0
            table.set(row!, key, .number(value))
            try await QA.wait(timeout: 10) {
                await self.table.refresh(); return self.row?.needsReload == true
            }
        } else {
            throw QAFailure("change must be dtype:<d> or setting:<key>=<n>")
        }
        var p = table.presentation(row!)
        guard p.pending else { throw QAFailure("the row did not turn red") }
        log.say(
            "red: pending \(table.pendingChanges(row!).joined(separator: ", ")); button \(p.actionTitle); tooltip: \(ModelHelp.action(row!, presentation: p, changes: table.pendingChanges(row!)).replacingOccurrences(of: "\n", with: " / "))"
        )
        try await render("red")
        // A dtype not on this Mac yet is made first (Get, derive), the row stays red; then Reload.
        if p.action == .get {
            table.perform(row!)
            try await QA.follow(table, model: model, log: log, step: "get \(p.selected)", timeout: 60, until: { $0.action == .reload && $0.busyText == nil })
            try await Task.sleep(for: .seconds(1))
            try await render("red-ready")
        }
        p = table.presentation(row!)
        table.perform(row!)
        try await QA.follow(table, model: model, log: log, step: "reload", timeout: 120, until: { !$0.pending && $0.action == .unload })
        let after = table.presentation(row!)
        let settings = (table.settings[model] ?? []).filter { $0.applies == "load" }.map { "\($0.id)=\($0.display($0.value))\($0.pending ? " (pending)" : "")" }
        log.say(
            "reloaded: loaded \(after.loadedPrecision ?? "?"), \(after.fast ? "Fast" : "Exact"); needsReload \(row!.needsReload); load settings \(settings.isEmpty ? "none" : settings.joined(separator: ", "))"
        )
        try await render("reloaded")

        table.perform(row!)
        try await QA.follow(table, model: model, log: log, step: "unload", timeout: 60, until: { $0.action == .load })
        try await render("after-unload")
        try QA.noWorkersLeft(after: "unload", log: log)
        log.say("done")
    }

    private func render(_ name: String) async throws {
        guard rendered.insert(name).inserted else { return }
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            RenderDelegate.renderTable(table, to: directory.appendingPathComponent("real-v2-\(name).png")) { done.resume() }
        }
    }
}
