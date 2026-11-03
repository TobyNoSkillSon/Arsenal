// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore
import SwiftUI

/// `ARSENAL_QA="render-dropin <dir>"`: the generic operation tool on the layers fixture (a new kind, `local.layers-fixture`)
/// at real size, 2× pixels. The run states come from a real run through the app's own API over the stub service (form,
/// staged image, the runner's progress, four PNGs fetched from their job URLs); the blocked, JSON and appearance states
/// are previews. Then the menu with the tool section, the Models submenu's table row and activity card, and the model's
/// details. Writes dropin-*.png and dropin-renders.txt. No worker, no GPU.
@MainActor final class DropinRenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private let log = QALog()
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        ToolWindow.keepOffscreenFrames = true
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            do { try await run(); exitCode = 0 } catch { log.say("render-dropin FAILED: \(TableController.message(error))") }
            log.write(to: directory.appendingPathComponent("dropin-renders.txt"))
            await set(.dark)
            QA.stopApp()
        }
    }

    private func set(_ mode: RenderAppearance) async {
        Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
        try? await Task.sleep(for: .milliseconds(100))
    }

    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-dropin-render-\(UUID().uuidString)", isDirectory: true)

    private func run() async throws {
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        await set(.dark)
        try await realRun()
        try await previews()
        try await menuAndTable()
    }

    private func window(_ session: OperationSession) -> ToolWindowController {
        ToolWindowController(
            title: "render", autosave: "", size: OperationView.size, minSize: OperationView.size,
            content: OperationView(session: session, symbol: LayersFixture.kind.icon ?? "square.stack.3d.up"))
    }

    // MARK: A real run through the API

    private func realRun() async throws {
        let service = FixtureService(rows: Fixtures.rows(.loaded) + [LayersFixture.row], kinds: Fixtures.kinds + [LayersFixture.kind])
        await service.setDelay("separate", .milliseconds(450))
        let support = scratch.appendingPathComponent("support")
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        guard let port = api.port else { throw QAFailure("the API did not start") }
        let session = OperationSession(
            model: ToolModel(LayersFixture.row), client: { ArsenalClient(port: port) }, inbox: support.appendingPathComponent("inbox"),
            results: scratch.appendingPathComponent("results"))
        defer { session.discard() }
        let controller = window(session)
        session.prepare(primary: "separate")
        try await until { !session.loading }
        // First frame: the form as the schema declares it, nothing chosen, Run off.
        try await capture(controller, "tool-empty-dark", keep: true)
        log.say("form: " + (session.form?.fields.map { "\($0.title) \($0.control)" }.joined(separator: "; ") ?? "none") + "; can run \(session.canRun)")
        let photo = try V3RenderDelegate.picture(seed: 2, in: scratch)
        session.addFiles([photo], to: "image")
        try await capture(controller, "tool-input-dark", keep: true)
        session.start()
        // Mid-run: the runner's own label, step, total and unit.
        try await until(seconds: 10) { session.run?.step != nil && (session.run?.step ?? 0) >= 2 }
        try await capture(controller, "tool-running-dark", keep: true)
        log.say("progress: \(session.run?.line(model: session.model?.name ?? "") ?? "none")")
        try await until(seconds: 15) { !session.running }
        if let problem = session.problem { throw QAFailure("the run failed: \(problem.message)") }
        try await capture(controller, "tool-result-dark", keep: true)
        let r = session.result
        log.say(
            "result: "
                + (r?.files.map { "\($0.order) \($0.label) \($0.width ?? 0)×\($0.height ?? 0) [\($0.extras.map { "\($0.key)=\($0.value)" }.joined(separator: ","))]" }.joined(
                    separator: "; ") ?? "none"))
        log.say("metadata: " + (r?.json?.replacingOccurrences(of: "\n", with: " ") ?? r?.fields.map { "\($0.key)=\($0.value)" }.joined(separator: ", ") ?? "none"))
        // Again over the last result, with the scrim.
        session.set("layers", .text("6"))
        session.start()
        try await until(seconds: 10) { (session.run?.step ?? 0) >= 3 }
        try await capture(controller, "tool-rerunning-dark", keep: true)
        try await until(seconds: 15) { !session.running }
        try await capture(controller, "tool-result6-dark", keep: true)
        for mode in [RenderAppearance.light, .translucent] {
            await set(mode)
            try await capture(controller, "tool-result6-\(mode.rawValue)", keep: true, translucent: mode == .translucent)
        }
        await set(.dark)
        // The smallest window the tool allows: six layers still fit with their labels.
        try await capture(controller, "tool-result6-min-dark", keep: true, size: OperationView.minSize)
        controller.close()
    }

    // MARK: Previews

    private func previews() async throws {
        var row = LayersFixture.row
        row.states = ["fp16": PrecisionState(state: .ready)]; row.needsCodeConsent = true
        let blocked = OperationSession(model: ToolModel(row), client: { nil }, inbox: scratch)
        blocked.preview(operations: [LayersFixture.descriptor], blocked: .init(message: "Layer Fixture\u{2019}s folder changed since you approved it.", action: .approve))
        try await capture(window(blocked), "tool-blocked-dark")
        let failed = OperationSession(model: ToolModel(LayersFixture.row), client: { nil }, inbox: scratch)
        failed.preview(
            operations: [LayersFixture.descriptor], values: ["image": .files([try V3RenderDelegate.picture(seed: 1, in: scratch)])],
            problem: OperationProblem.from(
                ToolError.from(
                    status: 409,
                    body: try WireJSON.encode([
                        "code": "insufficient_memory",
                        "details": .object(LoadMemoryRefusal(model: "layer-fixture", name: "Layer Fixture", precision: "bf16", neededGB: 2.1, freeGB: 1.4).details)
                    ])), model: "Layer Fixture"))
        try await capture(window(failed), "tool-error-dark")
        // A JSON operation with a union input (two operations: the pills).
        let measure = OperationDescriptor(
            name: "measure", title: "Measure",
            inputSchema: [
                "type": "object", "required": ["image"],
                "properties": [
                    "image": ["type": "object", "properties": ["path": ["type": "string"]], "x-arsenal-file": ["accept": ["image/png"]]],
                    "region": ["title": "Region", "anyOf": [["type": "string", "enum": ["all"]], ["type": "object", "properties": ["x": ["type": "integer"]]]], "default": "all"],
                    "scale": ["title": "Scale", "type": "number", "minimum": 0.1, "maximum": 4, "default": 1],
                    "mode": ["title": "Mode", "enum": ["fast", "fine"], "default": "fine"]
                ]
            ], output: "json", stream: false)
        let json = OperationSession(model: ToolModel(LayersFixture.row), client: { nil }, inbox: scratch)
        json.preview(
            operations: [LayersFixture.descriptor, measure], selected: "measure", values: ["image": .files([try V3RenderDelegate.picture(seed: 0, in: scratch)])],
            result: OperationResult(
                operation: "measure",
                fields: [
                    Pair(key: "depth_mean", value: "0.4182"), Pair(key: "depth_units", value: "relative"), Pair(key: "nearest", value: "0.0312"),
                    Pair(key: "pixels", value: "1048576")
                ], seconds: 0.18))
        try await capture(window(json), "tool-json-dark")
    }

    // MARK: Menu, Models submenu, details

    private func menuAndTable() async throws {
        var layers = LayersFixture.row
        layers.checks = ModelChecksStatus(
            protocolStatus: .pass, parity: .notRun, task: .notRun, implementationFingerprint: "7d1c0a9e44f2b6", cases: ["protocol": 6],
            evidence: ["results/2026-10-08-fixture/report.json"])
        let service = FixtureService(rows: Fixtures.rows(.loaded) + [layers], metadata: LayersFixture.metadata, kinds: Fixtures.kinds + [LayersFixture.kind])
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: scratch.appendingPathComponent("menu-support")))
        await app.table.refresh(); await app.refreshMenuState()
        app.tools.playback = FixturePlayback()
        app.rebuildMenu()
        log.say("menu: " + app.menu.items.map { $0.isSeparatorItem ? "—" : $0.title }.joined(separator: " | "))
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(app.menu.items, width: 300, to: directory.appendingPathComponent("dropin-menu-dark.png")) { done.resume() }
        }
        let status = RowPresentation.status(layers, selected: "fp16", chip: "M5 Max")
        log.say("row status: \(status.text) (\(status.tone)); tooltip: \(status.help.replacingOccurrences(of: "\n", with: " / "))")
        let feed = ActivityFeed(
            models: [
                .init(
                    id: LayersFixture.id, kind: LayersFixture.kind.id, state: "step",
                    requests: [
                        .init(
                            id: "l", requester: "Arsenal", op: "separate", phase: "step", title: "Separate Layers", label: "Separating layers", unit: "layers",
                            promptTokens: nil, cachedTokens: nil, generatedTokens: nil, step: 2, total: 4, startedAt: Date(timeIntervalSinceNow: -1), tps: nil),
                        .init(
                            id: "m", requester: "Claude Code", op: "separate", phase: "step", title: "Separate Layers", label: "Separating layers", unit: "layers",
                            promptTokens: nil, cachedTokens: nil, generatedTokens: nil, step: 1, total: 6, startedAt: Date(timeIntervalSinceNow: -0.4), tps: nil)
                    ], queued: 0, batch: nil, memoryMB: 2_140)
            ], machine: .init(cpuBusyPct: 12, gpuBusyPct: 41, cpuTempC: 46, gpuTempC: 52, memoryUsedMB: 40_100, memoryTotalMB: 137_439, arsenalMB: 300))
        log.say(
            "activity: "
                + feed.models[0].requests.map { "\($0.requester) \u{00b7} \(ActivityText.doing($0, .default) ?? "-"): \(ActivityText.detail($0, .default))" }.joined(
                    separator: "; "))
        // Where the submenus open: AppKit opens a submenu to the left of the main menu when it fits between the
        // screen's left edge and the menu; a wider one is pushed over the main menu (A 09:48 screenshot).
        let screens = NSScreen.screens.map { "\(Int($0.frame.width)) × \(Int($0.frame.height)) pt" }.joined(separator: ", ")
        log.say(
            "submenu widths: Models \(Int(ModelsMenuView.width(app.table))) pt (before: \(Int(ModelsMenuView.width(app.table) + 304)) pt with the panel docked), Activity \(Int(ActivityPanel.width)) pt; screens \(screens)"
        )
        // The Models submenu: the table alone.
        let table = NSHostingView(rootView: ModelsMenuView(controller: app.table))
        table.frame = NSRect(x: 0, y: 0, width: ModelsMenuView.width(app.table), height: ModelsMenuView.height(app.table))
        table.appearance = Appearance.shared.nsAppearance
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.capture(table, to: directory.appendingPathComponent("dropin-models-dark.png")) { done.resume() }
        }
        try await activityPanels(app, layers: feed.models[0])
        // The model's details in the browser: Checks and the tool's entry.
        let detail = ModelDetail(BrowserItem(CatalogEntry(row: layers, added: true), kind: LayersFixture.kind), chip: "M5 Max")
        log.say(
            "detail: " + detail.about.map { "\($0.key)=\($0.value)\($0.note.map { " (\($0))" } ?? "")" }.joined(separator: "; ") + "; actions "
                + detail.actions.map(\.title).joined(separator: ", "))
        let width: CGFloat = 1100
        let host = NSHostingView(
            rootView: ModelDetailView(detail: detail, palette: Appearance.shared.palette, width: width, licenceAmber: false, choose: { _ in }).padding(20)
                .background(Appearance.shared.palette.canvas.swiftUI))
        host.frame = NSRect(x: 0, y: 0, width: width + 40, height: 240)
        host.appearance = Appearance.shared.nsAppearance
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.capture(host, to: directory.appendingPathComponent("dropin-detail-dark.png")) { done.resume() }
        }
    }

    // MARK: Activity

    /// The Activity submenu's panel (v3: The user's feedback and Fable's review, 8 Oct): first frame before the first reading,
    /// nothing loaded, busy with four kinds in the three appearances, everything idle, a Mac whose temperature sensors
    /// Arsenal does not know, long model names, and this Mac's live readings. Fixture memory is one consistent machine:
    /// 29 GB of other apps and Arsenal's 0.3 GB, plus whatever is loaded. Sizes are logged.
    private func activityPanels(_ app: AppDelegate, layers: ActivityFeed.Model) async throws {
        let activity = app.activityMenu.activity
        func render(_ name: String) async {
            let size = await QA.renderActivity(activity, to: directory.appendingPathComponent("dropin-activity-v3-\(name).png"))
            log.say("wrote dropin-activity-v3-\(name).png (\(Int(size.width)) × \(Int(size.height)))")
        }
        func machine(_ models: [ActivityFeed.Model], cpu: Double, gpu: Double, cpuT: Double?, gpuT: Double?) -> ActivityFeed.Machine {
            let loaded = models.reduce(0) { $0 + ($1.memoryMB ?? 0) }
            return .init(
                cpuBusyPct: cpu, gpuBusyPct: gpu, cpuTempC: cpuT, gpuTempC: gpuT, memoryUsedMB: 29_000 + 310 + loaded, memoryTotalMB: 137_439, arsenalMB: 310)
        }
        activity.rows = Dictionary(app.table.rows.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        var busy = V3RenderDelegate.busyFeed()
        // Kokoro speaking for an agent; the layers fixture separating for two clients.
        busy.models[2].requests = [.init(id: "v", requester: "Claude Code", op: "speak", phase: "step", startedAt: Date(timeIntervalSinceNow: -1))]
        busy.models[2].state = "step"
        busy.models.append(layers)
        busy.machine = machine(busy.models, cpu: 38, gpu: 94, cpuT: 52.4, gpuT: 71.2)
        // First frame: the loaded models as cards before the first reading (dashes, never zeros).
        var first = busy
        first.machine = .init()
        for i in first.models.indices { first.models[i].requests = []; first.models[i].queued = 0; first.models[i].state = "idle" }
        activity.preview(first, sampled: false)
        await render("first-frame-dark")
        activity.preview(ActivityFeed(models: [], machine: machine([], cpu: 4, gpu: 2, cpuT: 41, gpuT: 37)))
        await render("empty-dark")
        for mode in [RenderAppearance.dark, .light, .translucent] {
            await set(mode)
            activity.preview(busy)
            await render("busy-\(mode.rawValue)")
        }
        await set(.dark)
        var idle = first
        idle.machine = machine(idle.models, cpu: 5, gpu: 3, cpuT: 42, gpuT: 39)
        activity.preview(idle)
        await render("idle-dark")
        activity.preview(ActivityFeed(models: [layers], machine: machine([layers], cpu: 12, gpu: 41, cpuT: nil, gpuT: nil)))
        await render("no-sensors-dark")
        // Long names: the name gives way in its middle, the figures stay whole.
        let saved = activity.rows
        for (id, name) in [
            ("qwen3.8-27b", "Qwen3.8 27B Instruct Q4 (32K context)"), ("flux2-klein-4b", "FLUX.2 Klein 4B Distilled Turbo"),
            ("kokoro-82m", "Kokoro 82M Multilingual"), (LayersFixture.id, "Layer Fixture with a Rather Long Name")
        ] {
            activity.rows[id]?.name = name
        }
        activity.preview(busy)
        await render("long-names-dark")
        activity.rows = saved
        // Live: this Mac's own readings through the real sampler (nothing loaded in this harness).
        let live = ActivityAggregator()
        let stream = await live.stream { await live.snapshot(rows: [], memory: .init()) }
        var latest: ActivitySnapshot?
        let reader = Task { for await snapshot in stream { latest = snapshot } }
        try? await Task.sleep(for: .milliseconds(1800))
        reader.cancel()
        if let latest {
            activity.preview(ActivityFeed(latest))
            let m = latest.machine
            log.say(
                "live reading: cpu \(m.cpuBusyPct.map { String(format: "%.0f%%", $0) } ?? "-") \(m.cpuTempC.map { String(format: "%.1f °C", $0) } ?? "-"), gpu "
                    + "\(m.gpuBusyPct.map { String(format: "%.0f%%", $0) } ?? "-") \(m.gpuTempC.map { String(format: "%.1f °C", $0) } ?? "-"), memory "
                    + "\(m.memoryUsedMB.map { String(format: "%.0f", $0) } ?? "-") of \(m.memoryTotalMB.map { String(format: "%.0f", $0) } ?? "-") MB, Arsenal \(m.arsenalMB.map { String(format: "%.0f", $0) } ?? "-") MB"
            )
            await render("live-dark")
        }
        log.say(
            "busy: chat total \(ActivityText.chatSpeed(busy.models[0]).map { String(format: "%.1f", $0) } ?? "-") tok/s; lines "
                + busy.models.map { m in "\(m.id): " + ActivityText.lines(m).map { "\($0.name)=\(ActivityText.detail($0.request, activity.section(m)))" }.joined(separator: ", ") }
                .joined(separator: "; "))
    }

    // MARK: Capture

    private func until(seconds: Double = 5, _ condition: @MainActor () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() > deadline { throw QAFailure("timed out") }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    /// The window's frame view (traffic lights included) at 2× pixels; `keep` leaves the window open for the next state.
    private func capture(
        _ controller: ToolWindowController, _ name: String, keep: Bool = false, translucent: Bool = false, size: NSSize = OperationView.size
    ) async throws {
        guard let window = controller.window, let root = window.contentView?.superview else { return }
        controller.applyAppearance()
        if window.frame.size != size { window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false) }
        if !window.isVisible {
            window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
            window.orderFrontRegardless()
            window.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        }
        try await Task.sleep(for: .milliseconds(500))
        root.layoutSubtreeIfNeeded(); window.displayIfNeeded()
        let scale: CGFloat = 2
        guard
            let out = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: Int(root.bounds.width * scale), pixelsHigh: Int(root.bounds.height * scale), bitsPerSample: 8,
                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return }
        out.size = root.bounds.size
        root.cacheDisplay(in: root.bounds, to: out)
        var rep = out
        if translucent, let composed = V21RenderDelegate.compose(rep, over: V21RenderDelegate.blurred(V21RenderDelegate.darkPicture(), size: size)) { rep = composed }
        try rep.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("dropin-\(name).png"))
        if !keep { window.orderOut(nil); controller.close() }
        log.say("wrote dropin-\(name).png (\(Int(root.bounds.width)) × \(Int(root.bounds.height)))")
    }
}
