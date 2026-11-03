// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore

/// `Arsenal --render-v22 <dir>`: the v2.2 surfaces at real size (2× pixels) from the real catalog in the three
/// appearances: the browser filtered per dtype, the collapsed sidebar's toggle, the table without its catalog footer,
/// the status menu with Open Arsenal Files / Copy Skill / Refresh and a model's own section, and the Get pop-up from a
/// real Get plan. No Get, no worker, no GPU. Writes v22-*.png and v22-renders.txt.
@MainActor final class V22RenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private let log = QALog()
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            do { try await run(); exitCode = 0 } catch { log.say("render-v22 FAILED: \(TableController.message(error))") }
            log.write(to: directory.appendingPathComponent("v22-renders.txt"))
            QA.stopApp()
        }
    }

    private func set(_ mode: RenderAppearance) async {
        Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
        try? await Task.sleep(for: .milliseconds(100))
    }

    private func run() async throws {
        let paths = ArsenalPaths()
        let (service, error) = await ServiceFactory.make(paths: paths)
        if let error { throw QAFailure(error) }
        let app = AppDelegate(service: service, paths: paths)
        for id in ["kokoro-82m", "qwen3.8-27b"] { try? await service.add(id: id) }

        // The browser: dtype int4 and Size 1–8 GB hold at one dtype; each row shows that dtype's Download and RAM.
        let browser = BrowserController(service: service)
        await browser.refresh()
        log.say("browser: \(browser.count); dtypes offered: " + BrowserFacet.dtype.values(browser.items).joined(separator: ", "))
        UserDefaults.standard.set(true, forKey: BrowserRootView.openKey)
        browser.openSections.formUnion([.dtype, .size, .ram])
        browser.state.toggle(.dtype, "int4"); browser.state.toggle(.size, "1\u{2013}8 GB")
        log.say("filter int4 · 1–8 GB: " + browser.shown.map { "\($0.name) at \($0.matchedDtype ?? "-") \($0.downloadText) \($0.ramText)" }.joined(separator: "; "))
        if browser.shown.isEmpty {
            browser.state.values[.size] = ["8\u{2013}32 GB"]
            log.say("filter int4 · 8–32 GB: " + browser.shown.map { "\($0.name) at \($0.matchedDtype ?? "-") \($0.downloadText) \($0.ramText)" }.joined(separator: "; "))
        }
        if let first = browser.shown.first { browser.expanded = [first.id] }
        for mode in [RenderAppearance.dark, .light, .translucent] {
            await set(mode)
            let window = BrowserWindowController(controller: browser)
            if mode == .translucent {
                try await capture(window, "browser-filtered-translucent", backdrop: V21RenderDelegate.darkPicture())
            } else {
                try await capture(window, "browser-filtered-\(mode.rawValue)")
            }
            window.close()
        }
        await set(.dark)
        browser.expanded = []
        UserDefaults.standard.set(false, forKey: BrowserRootView.openKey)
        let collapsed = BrowserWindowController(controller: browser)
        try await capture(collapsed, "browser-collapsed", size: NSSize(width: 1100, height: 420))
        collapsed.close()
        UserDefaults.standard.set(true, forKey: BrowserRootView.openKey)

        // The table: no footer while the catalog is verified.
        await app.table.refresh()
        log.say("table: \(app.table.rows.map(\.name).joined(separator: ", ")); footer: \(ModelTable.footer(app.table)?.text ?? "none")")
        for mode in [RenderAppearance.dark, .light, .translucent] {
            await set(mode)
            let url = directory.appendingPathComponent("v22-table-\(mode.rawValue).png")
            if mode == .translucent {
                await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                    RenderDelegate.renderTable(app.table, to: url, backdrop: V21RenderDelegate.blurred(V21RenderDelegate.darkPicture(), size: NSSize(width: 1100, height: 700))) {
                        done.resume()
                    }
                }
            } else {
                await QA.renderTable(app.table, to: url)
            }
            log.say("wrote v22-table-\(mode.rawValue).png")
        }

        // The status menu, with a loaded model's own section (as a folder's [[menu.section]] would add it).
        await set(.dark)
        await app.refreshMenuState()
        let kokoro = app.table.rows.first { $0.id == "kokoro-82m" }
        let speak = LoadedAction(id: "kokoro-82m-speak", modelID: "kokoro-82m", title: "Speak Clipboard", op: "speak")
        let transcribe = LoadedAction(id: "kokoro-82m-section-0-read", modelID: "kokoro-82m", title: "Read Selection Aloud", op: "speak")
        if kokoro != nil {
            var rows = app.table.rows
            if let i = rows.firstIndex(where: { $0.id == "kokoro-82m" }) { rows[i].states[rows[i].selectedPrecision] = .init(state: .loaded) }
            app.table.apply(rows: rows, kinds: app.table.kinds, catalog: app.table.catalog)
            app.previewState(
                actions: [speak, transcribe], memory: await service.memory(),
                sections: [
                    LoadedMenuSection(id: "kokoro-82m", modelID: "kokoro-82m", title: "Kokoro 82M", items: [speak]),
                    LoadedMenuSection(id: "kokoro-82m-section-0", modelID: "kokoro-82m", title: "Reading", items: [transcribe])
                ])
        }
        for mode in [RenderAppearance.dark, .light] {
            await set(mode)
            app.rebuildMenu()
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("v22-menu-\(mode.rawValue).png")) { done.resume() }
            }
            log.say("wrote v22-menu-\(mode.rawValue).png: " + app.menu.items.map { $0.isSeparatorItem ? "—" : $0.title }.joined(separator: " | "))
        }

        // The Get pop-up from real plans: every model whose Get needs more than its weights, and one weights-only plan.
        await set(.dark)
        var rendered = 0
        for item in browser.items {
            for dtype in item.dtypes {
                let plan: GetPlan
                do { plan = try await service.getPlan(id: item.id, dtype: dtype) } catch {
                    log.say("plan \(item.id) \(dtype): refused: \(TableController.message(error))"); continue
                }
                let lines = plan.items.map { "\($0.name) [\($0.kind)] \($0.downloadBytes.map(formatBytes) ?? "unknown")" }
                log.say("plan \(item.id) \(dtype) runner \(plan.runner): \(lines.joined(separator: "; ")) · asks \(TableController.asks(plan))")
                guard rendered < 2, TableController.asks(plan), TableController.downloads(plan).contains(where: { $0.kind != "weights" }) else { continue }
                for mode in [RenderAppearance.dark, .light] {
                    await set(mode)
                    try renderAlert(ModelsMenu.getAlert(item.entry.row, plan: plan), "get-plan-\(item.id)-\(mode.rawValue)")
                }
                rendered += 1
            }
        }
        await set(.dark)
        await service.shutdown()
    }

    /// The alert as laid out by AppKit, redrawn from its own views: macOS 26 draws an alert's glass, labels and buttons
    /// in the window server, which an offscreen display cache cannot capture. Icon, texts (their strings, fonts and
    /// frames) and buttons (frames, titles, the default one filled) come from the real alert; the glass is a flat fill.
    private func renderAlert(_ alert: NSAlert, _ name: String) throws {
        let window = alert.window
        window.appearance = Appearance.shared.nsAppearance
        alert.layout()
        guard let content = window.contentView else { return }
        content.layoutSubtreeIfNeeded()
        let size = window.frame.size, dark = Appearance.shared.palette.dark
        let text = dark ? NSColor(white: 0.95, alpha: 1) : NSColor(white: 0.1, alpha: 1)
        let out = NSImage(size: size)
        out.lockFocus()
        NSBezierPath(roundedRect: NSRect(origin: .zero, size: size), xRadius: 22, yRadius: 22).addClip()
        (dark ? NSColor(white: 0.18, alpha: 1) : NSColor(white: 0.97, alpha: 1)).setFill()
        NSRect(origin: .zero, size: size).fill()
        func rect(_ v: NSView) -> NSRect { v.convert(v.bounds, to: nil) }
        func walk(_ v: NSView) {
            if let field = v as? NSTextField, !field.stringValue.isEmpty, rect(field).width > 20 {
                let style = NSMutableParagraphStyle(); style.alignment = field.alignment
                NSAttributedString(string: field.stringValue, attributes: [.font: field.font ?? .systemFont(ofSize: 13), .foregroundColor: text, .paragraphStyle: style])
                    .draw(with: rect(field), options: [.usesLineFragmentOrigin])
                return
            }
            if let image = (v as? NSImageView)?.image { image.draw(in: rect(v)); return }
            if let button = v as? NSButton {
                let r = rect(button), primary = button.keyEquivalent == "\r"
                (primary ? NSColor.controlAccentColor : (dark ? NSColor(white: 1, alpha: 0.14) : NSColor(white: 0, alpha: 0.08))).setFill()
                NSBezierPath(roundedRect: r, xRadius: r.height / 2, yRadius: r.height / 2).fill()
                let title = NSAttributedString(string: button.title, attributes: [.font: NSFont.systemFont(ofSize: 13, weight: .medium), .foregroundColor: primary ? .white : text])
                let t = title.size()
                title.draw(at: NSPoint(x: r.midX - t.width / 2, y: r.midY - t.height / 2))
                return
            }
            v.subviews.forEach(walk)
        }
        walk(content)
        out.unlockFocus()
        guard let tiff = out.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else { return }
        try png.write(to: directory.appendingPathComponent("v22-\(name).png"))
        log.say("wrote v22-\(name).png: \(alert.messageText) / \(alert.informativeText.replacingOccurrences(of: "\n", with: " | "))")
    }

    private func capture(_ controller: BrowserWindowController, _ name: String, size: NSSize = BrowserWindowController.defaultSize, backdrop: NSImage? = nil) async throws {
        guard let window = controller.window else { return }
        controller.applyAppearance()
        window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
        window.orderFrontRegardless()
        window.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await Task.sleep(for: .milliseconds(700))
        guard var rep = WindowSnapshot.capture(window) else { return }
        if let backdrop, let composed = V21RenderDelegate.compose(rep, over: V21RenderDelegate.blurred(backdrop, size: window.frame.size)) { rep = composed }
        try rep.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("v22-\(name).png"))
        window.orderOut(nil)
        log.say("wrote v22-\(name).png")
    }
}

/// `Arsenal --real-cycle-v22 <model> <dir>`: the v2.2 path through the real service: the browser's per-dtype filter,
/// Add, the table's Get after its plan (and the pop-up's decision), Load, then Remove from the browser. GPU: run it only
/// under `scripts/mlock gpulock` (alarm ≤ 180 s).
@MainActor final class RealCycleV22Delegate: NSObject, NSApplicationDelegate, ExitCoding {
    let model: String, directory: URL
    var exitCode: Int32 = 1
    private var app: AppDelegate!
    private let log = QALog()
    init(model: String, directory: URL) { self.model = model; self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            let (service, error) = await ServiceFactory.make(paths: ArsenalPaths())
            app = AppDelegate(service: service, paths: ArsenalPaths())
            do {
                if let error { throw QAFailure(error) }
                try await cycle(); exitCode = 0
            } catch { log.say("FAILED: \(TableController.message(error))") }
            await service.shutdown()
            log.write(to: directory.appendingPathComponent("v22-real-cycle.txt"))
            QA.stopApp()
        }
    }

    private var table: TableController { app.table }
    private var row: ModelRow? { table.rows.first { $0.id == model } }

    private func cycle() async throws {
        let browser = BrowserController(service: app.service)
        browser.confirmRemove = { _ in .keepWeights }
        await browser.refresh()
        browser.state.toggle(.dtype, "int4"); browser.state.toggle(.size, "1\u{2013}8 GB")
        log.say(
            "browser filter int4 · 1–8 GB: "
                + (browser.shown.isEmpty
                    ? "no model" : browser.shown.map { "\($0.name) at \($0.matchedDtype ?? "-") · \($0.downloadText) · \($0.ramText)" }.joined(separator: "; ")))
        browser.state = BrowserState()
        browser.state.toggle(.size, "< 1 GB")
        log.say("browser filter Size < 1 GB: " + browser.shown.map { "\($0.name) at \($0.matchedDtype ?? "-")" }.joined(separator: "; "))
        guard browser.items.contains(where: { $0.id == model }) else { throw QAFailure("\(model) is not in the catalog") }
        browser.add(model)
        try await QA.wait(timeout: 10) {
            await self.table.refresh(); return self.row != nil
        }
        log.say("added from the browser: table has \(table.rows.map(\.name).joined(separator: ", "))")

        var asked: GetPlan?
        table.confirmGet = { _, plan in
            asked = plan; return true
        }
        table.confirmCode = { _ in true }
        if table.presentation(row!).action == .get {
            let p = table.presentation(row!)
            let plan = try await app.service.getPlan(id: model, dtype: p.selected)
            log.say(
                "get plan \(p.selected): " + plan.items.map { "\($0.name) [\($0.kind)] \($0.downloadBytes.map(formatBytes) ?? "size unknown")" }.joined(separator: "; ")
                    + " · pop-up: \(TableController.asks(plan) ? "yes" : "no (nothing beyond small weights)")")
            table.perform(row!)
            try await QA.follow(table, model: model, log: log, step: "get", timeout: 150, until: { $0.action == .load && $0.busyText == nil })
            log.say("pop-up shown: \(asked.map { _ in "yes" } ?? "no")")
            try await Task.sleep(for: .seconds(1))
        }
        table.perform(row!)
        try await QA.follow(table, model: model, log: log, step: "load", timeout: 120, until: { $0.action == .unload })
        let loaded = table.presentation(row!)
        log.say("loaded \(loaded.loadedPrecision ?? "?"); runner \(row!.runner)")
        await QA.renderTable(table, to: directory.appendingPathComponent("v22-real-table-loaded.png"))

        await browser.refresh()
        browser.remove(model)
        try await QA.wait(timeout: 30) {
            await self.table.refresh(); return self.row == nil
        }
        log.say("removed from the browser: table has \(table.rows.isEmpty ? "no rows" : table.rows.map(\.name).joined(separator: ", "))")
        try QA.noWorkersLeft(after: "Remove", log: log)
        log.say("done")
    }
}
