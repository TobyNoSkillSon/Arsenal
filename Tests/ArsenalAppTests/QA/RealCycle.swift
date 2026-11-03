// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore

/// `Arsenal --real-cycle <model> <dir>`: one small model through the real service the way a user drives it — the
/// table's own action (`TableController.perform`, what the row's button calls) for Get, Load and Unload; the run through
/// a menu action the loaded model contributes, else through the API the app hosts. Renders the table in each state
/// it passes through and prints one line per step. GPU: run it only under `scripts/mlock gpulock`.
@MainActor final class RealCycleDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    private var app: AppDelegate!
    let model: String
    let directory: URL
    var exitCode: Int32 = 1
    private var rendered: Set<String> = []
    private let log = QALog()

    init(model: String, directory: URL) { self.model = model; self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.appearance = NSAppearance(named: .darkAqua)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            log.say("opening the catalog")
            let paths = ArsenalPaths()
            let (service, openError) = await ServiceFactory.make(paths: paths)
            app = AppDelegate(service: service, paths: paths)
            do {
                if let openError { throw QAFailure(openError) }
                try await cycle(); exitCode = 0
            } catch { log.say("FAILED: \(TableController.message(error))") }
            app.api.stop()
            log.write(to: directory.appendingPathComponent("real-cycle.txt"))
            QA.stopApp()
        }
    }

    private var row: ModelRow? { app.table.rows.first { $0.id == model } }

    private func cycle() async throws {
        let table = app.table
        table.start { [weak self] in await self?.app.refreshMenuState() }
        log.say("starting the API")
        await app.api.start()
        guard let port = app.api.port else { throw QAFailure(app.api.error ?? "API did not start") }
        await table.refresh()
        // v2: the table starts empty; Add puts the model in it.
        if row == nil { try await app.service.add(id: model); await table.refresh() }
        guard let first = row else { throw QAFailure("\(model) is not in the catalog") }
        let catalog = table.catalog.map(FooterText.catalog) ?? "no catalog status"
        log.say("opened: \(table.rows.count) models in \(table.groups.count) groups; \(catalog); API on port \(port)")
        log.say("row: \(first.name) · \(table.presentation(first).actionTitle) · \(table.presentation(first).status.text)")
        try await render("before-get")

        // Get, exactly as the row's button: Get · size, then progress / deriving in the row, then Load.
        table.confirmGet = { _, _ in true }
        if table.presentation(first).action == .get {
            table.perform(first)
            try await QA.follow(app.table, model: model, log: log, step: "get", timeout: 900, until: { $0.action == .load && $0.busyText == nil }, each: renderBusy)
            try await render("after-get")
        } else {
            log.say("get: already on this Mac (\(table.presentation(first).actionTitle)); a fresh support dir exercises Get")
        }
        // Load.
        table.perform(row!)
        try await QA.follow(app.table, model: model, log: log, step: "load", timeout: 300, until: { $0.action == .unload }, each: renderBusy)
        await app.refreshMenuState()
        let declared = await app.service.menuActions()
        log.say("actions the service declares: " + (declared.isEmpty ? "none" : declared.map { "\($0.title) (\($0.modelID))" }.joined(separator: ", ")))
        try await render("loaded")
        app.rebuildMenu()
        try await renderMenu("menu-loaded")
        log.say("menu: " + app.menu.items.filter { !$0.isSeparatorItem }.map { $0.title + ($0.subtitle.map { " (\($0))" } ?? "") }.joined(separator: " | "))

        // Run: a contributed menu action if there is one, else the hosted API.
        if let action = app.actions.first(where: { $0.modelID == model }) {
            app.runner.readClipboard = { "Arsenal loaded this voice from its own table." }
            // Silent on the development Mac: the speech file is made, the shared queue is muted.
            try? PlaybackQueue.shared.setMuted(true)
            app.runner.run(action)
            try await QA.wait(timeout: 120) { self.app.runner.running == nil && self.app.runner.lastResult?.hasSuffix("\u{2026}") == false }
            log.say("menu action \(action.title): \(app.runner.lastResult ?? "no result")")
            if app.runner.lastResult?.contains("failed") == true { throw QAFailure(app.runner.lastResult) }
        } else {
            log.say("API: " + (try await decideThroughAPI(port: port)))
        }
        app.rebuildMenu()
        try await renderMenu("menu-after-run")

        // Unload, as the row's button; the runner process must be gone.
        table.perform(row!)
        try await QA.follow(app.table, model: model, log: log, step: "unload", timeout: 120, until: { $0.action == .load }, each: renderBusy)
        await app.refreshMenuState()
        try await render("after-unload")
        log.say("memory after unload: " + MenuContent.memoryLines(app.memory, rows: table.rows).joined(separator: "; "))
        // Unload = the runner process exits: no worker is left under this app.
        try QA.noWorkersLeft(after: "unload", log: log)
        log.say("done")
    }

    /// One System One decision through the API the app hosts (the first golden case of the decisions worker's fixtures).
    private func decideThroughAPI(port: Int) async throws -> String {
        let body =
            #"{"model":"\#(model)","state":{"text":"Please refund my invoice, order 000."},"questions":{"refund":{"type":"noul","instructions":"Does this message ask for a refund?","criteria":{"false":"not a refund request","true":"requests money back"}}}}"#
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/systemone")!, timeoutInterval: 120)
        request.httpMethod = "POST"; request.httpBody = Data(body.utf8)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let text = String(decoding: data, as: UTF8.self)
        guard status == 200 else { throw QAFailure("POST /v1/systemone → \(status): \(text.prefix(300))") }
        return "POST /v1/systemone → 200 \(text.prefix(300))"
    }

    /// The first downloading / deriving / loading state a step passes through.
    private func renderBusy(_ r: ModelRow) async throws {
        for state in [ModelState.downloading, .deriving, .loading] where r.states.values.contains(where: { $0.state == state }) { try await render(state.rawValue) }
    }
    private func render(_ name: String) async throws {
        guard rendered.insert(name).inserted else { return }
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            RenderDelegate.renderTable(app.table, to: directory.appendingPathComponent("real-\(name).png")) { done.resume() }
        }
    }
    private func renderMenu(_ name: String) async throws {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("real-\(name).png")) { done.resume() }
        }
    }
}
