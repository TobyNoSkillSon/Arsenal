// SPDX-License-Identifier: MIT
import AppKit
@testable import Arsenal
import ArsenalCore
import SwiftUI
import UniformTypeIdentifiers

/// `ARSENAL_QA="render-background <dir>"`: Remove Background at its real size, 2× pixels, dark and light. The opening
/// frame, the first frame, a cut being made and its result come from a real run through the app's own API over the stub
/// service's fixture runner (BiRefNet's schema); loading, an earlier cut-out under a new cut, a failure, a model not on
/// this Mac, a model with its own options and two operations, and the minimum size are previews. Then the menu with
/// the entry. Writes background-*.png and background-renders.txt. No worker, no GPU.
@MainActor final class BackgroundRenderDelegate: NSObject, NSApplicationDelegate, ExitCoding {
    let directory: URL
    var exitCode: Int32 = 1
    private let log = QALog()
    init(directory: URL) { self.directory = directory }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        ToolWindow.keepOffscreenFrames = true
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task {
            do { try await run(); exitCode = 0 } catch { log.say("render-background FAILED: \(TableController.message(error))") }
            log.write(to: directory.appendingPathComponent("background-renders.txt"))
            await set(.dark)
            QA.stopApp()
        }
    }

    private func set(_ mode: RenderAppearance) async {
        Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
        try? await Task.sleep(for: .milliseconds(100))
    }

    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-background-render-\(UUID().uuidString)", isDirectory: true)

    private func run() async throws {
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        await set(.dark)
        try await realRun()
        try await previews()
        try await menu()
    }

    private func window(_ session: BackgroundSession) -> ToolWindowController {
        ToolWindowController(
            title: "Remove Background", autosave: "", size: BackgroundView.size, minSize: BackgroundView.minSize, content: BackgroundView(session: session))
    }

    /// A portrait-like photo: a sky-to-sand gradient with a figure (head and shoulders) in the middle.
    static func photo(_ url: URL, width: Int = 1200, height: Int = 900) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard
            let c = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
            let gradient = CGGradient(
                colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                colors: [CGColor(red: 0.93, green: 0.82, blue: 0.62, alpha: 1), CGColor(red: 0.45, green: 0.66, blue: 0.9, alpha: 1)] as CFArray, locations: [0, 1])
        else { throw QAFailure("could not draw the photo") }
        c.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: 0, y: height), options: [])
        let w = CGFloat(width), h = CGFloat(height)
        c.setFillColor(CGColor(red: 0.85, green: 0.3, blue: 0.25, alpha: 1))
        c.fillEllipse(in: CGRect(x: w * 0.3, y: -h * 0.25, width: w * 0.4, height: h * 0.75))
        c.setFillColor(CGColor(red: 0.96, green: 0.78, blue: 0.64, alpha: 1))
        c.fillEllipse(in: CGRect(x: w * 0.41, y: h * 0.42, width: w * 0.18, height: h * 0.3))
        c.setFillColor(CGColor(red: 0.25, green: 0.18, blue: 0.14, alpha: 1))
        c.fillEllipse(in: CGRect(x: w * 0.4, y: h * 0.62, width: w * 0.2, height: h * 0.14))
        guard let image = c.makeImage(), let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw QAFailure("could not encode the photo")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw QAFailure("could not encode the photo") }
        return url
    }

    // MARK: A real run through the API

    private var made: [Cutout] = []

    private func realRun() async throws {
        let support = scratch.appendingPathComponent("support")
        let service = FixtureService(
            rows: [CutoutFixture.row], metadata: CutoutFixture.metadata, supportDirectory: support, operations: [CutoutFixture.id: [CutoutFixture.operation]])
        await service.setDelay(CutoutFixture.operation.name, .milliseconds(1500))
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        guard let port = api.port else { throw QAFailure("the API did not start") }
        let session = BackgroundSession(
            models: [ModelChoice(model: ToolModel(CutoutFixture.row))], selected: CutoutFixture.id, client: { ArsenalClient(port: port) },
            inbox: support.appendingPathComponent("inbox"), outputDirectory: scratch.appendingPathComponent("Pictures"), scratch: scratch.appendingPathComponent("window"))
        defer { session.close() }
        let controller = window(session)
        session.prepare()
        try await capture(controller, "opening-dark", keep: true, settle: 30)
        log.say("opening: loading \(session.loading), operations \(session.operations.count), can run \(session.canRun)")
        try await until { !session.loading }
        try await both(controller, "empty", keep: true)
        log.say("empty: operations \(session.operations.map(\.name)), fields \(session.optionFields.map(\.key)), can run \(session.canRun)")
        session.setInput([try Self.photo(scratch.appendingPathComponent("photos/Portrait on the beach.jpg"))])
        try await until { session.run?.phase != nil }
        try await both(controller, "busy", keep: true)
        log.say("busy: \(session.run?.line(model: "BiRefNet") ?? "none")")
        try await until(seconds: 10) { !session.running }
        if let problem = session.problem { throw QAFailure("the cut failed: \(problem.message)") }
        try await both(controller, "result", keep: true)
        log.say("result: " + (session.current.map { "\($0.url.lastPathComponent) \($0.width)×\($0.height)" } ?? "none"))
        log.say("figures: " + session.figures().map { "\($0.title)=\($0.value ?? "-")" }.joined(separator: ", "))
        made = session.cutouts
        // A second, portrait image: the strip appears.
        session.setInput([try Self.photo(scratch.appendingPathComponent("photos/Standing.jpg"), width: 800, height: 1100)])
        try await until(seconds: 10) { session.cutouts.count == 2 && !session.running }
        try await both(controller, "two-results", keep: true)
        made = session.cutouts
        // The same run's mask (Cut-out | Mask).
        session.showsMask = true
        try await both(controller, "mask", keep: true)
        session.showsMask = false
        try await capture(controller, "min-dark", keep: true, size: BackgroundView.minSize)
        controller.close()
    }

    // MARK: Previews

    private func previews() async throws {
        let models = [ModelChoice(model: ToolModel(CutoutFixture.row))]
        let photo = try Self.photo(scratch.appendingPathComponent("photos/Portrait.jpg"))

        // The image chosen, nothing cut yet (as after ×-ing a result, or before the model can run).
        let chosen = BackgroundSession(models: models, selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        chosen.preview(operations: [CutoutFixture.descriptor], input: photo)
        try await both(window(chosen), "image")

        // Loading the model first.
        let loading = BackgroundSession(models: models, selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        var load = ToolRun(model: CutoutFixture.id, operation: CutoutFixture.operation.name)
        load.phase = "loading"
        loading.preview(operations: [CutoutFixture.descriptor], input: photo, run: load)
        try await both(window(loading), "loading")

        // A new cut over an earlier one.
        if let last = made.last {
            let again = BackgroundSession(models: models, selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
            var running = ToolRun(model: CutoutFixture.id, operation: CutoutFixture.operation.name)
            running.phase = "step"; running.label = "Removing background"; running.memoryMB = 1_380
            again.preview(operations: [CutoutFixture.descriptor], input: last.original, run: running, cutouts: made, shown: last.id)
            try await both(window(again), "busy-over-result")
        }

        // A failure, in place.
        let failed = BackgroundSession(models: models, selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        failed.preview(
            operations: [CutoutFixture.descriptor], input: photo,
            problem: OperationProblem.from(
                ToolError.from(
                    status: 409,
                    body: try WireJSON.encode([
                        "code": "insufficient_memory",
                        "details": .object(LoadMemoryRefusal(model: "birefnet", name: "BiRefNet", precision: "bf16", neededGB: 1.4, freeGB: 0.6).details)
                    ])), model: "BiRefNet"))
        try await both(window(failed), "error")

        // Not on this Mac.
        let blocked = BackgroundSession(
            models: [
                ModelChoice(
                    model: ToolModel(CutoutFixture.row), unavailable: .init(message: "BiRefNet isn\u{2019}t on this Mac yet.", action: .download, actionTitle: "Download 424 MB"))
            ],
            selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        blocked.preview(operations: [CutoutFixture.descriptor], input: photo)
        try await both(window(blocked), "blocked")

        // A model with its own options and a second operation.
        var properties = CutoutFixture.input["properties"]?.object ?? [:]
        properties["resolution"] = ["enum": [512, 1024, 2048], "default": 1024, "title": "Resolution", "description": "The model's working size"]
        properties["threshold"] = ["type": "number", "minimum": 0, "maximum": 1, "default": 0.5, "title": "Threshold", "description": "Mask cut-off"]
        properties["refine_edges"] = ["type": "boolean", "default": true, "title": "Refine edges"]
        var input = CutoutFixture.input
        input["properties"] = .object(properties)
        let ops = [
            OperationDescriptor(name: "remove_background", title: "Remove Background", inputSchema: input, output: "file", stream: true),
            OperationDescriptor(name: "matte", title: "Alpha Matte", inputSchema: CutoutFixture.input, output: "file", stream: false)
        ]
        let options = BackgroundSession(models: models, selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        options.preview(operations: ops, input: made.first?.original ?? photo, cutouts: made.isEmpty ? [] : [made[0]], shown: made.first?.id)
        try await both(window(options), "options")
        log.say("options: pills \(options.operations.map(\.title)), fields \(options.optionFields.map { "\($0.key) \($0.control)" })")
    }

    // MARK: Menu

    private func menu() async throws {
        let support = scratch.appendingPathComponent("menu-support")
        let service = FixtureService(
            rows: SoundFixture.rows + [CutoutFixture.row], metadata: SoundFixture.metadata.merging(CutoutFixture.metadata) { a, _ in a }, supportDirectory: support,
            operations: SoundFixture.allOperations.merging([CutoutFixture.id: [CutoutFixture.operation]]) { a, _ in a })
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: support))
        await app.table.refresh(); await app.refreshMenuState()
        app.tools.playback = FixturePlayback()
        app.rebuildMenu()
        log.say("menu: " + app.menu.items.filter { !$0.isHidden }.map { $0.isSeparatorItem ? "—" : $0.title }.joined(separator: " | "))
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            MenuMock.render(app.menu.items.filter { !$0.isHidden }, width: 320, to: directory.appendingPathComponent("background-menu-dark.png")) { done.resume() }
        }
        app.table.stop()
    }

    // MARK: Capture

    /// The state in dark, then light; the window closes after unless kept.
    private func both(_ controller: ToolWindowController, _ name: String, keep: Bool = false) async throws {
        try await capture(controller, name + "-dark", keep: true)
        await set(.light)
        try await capture(controller, name + "-light", keep: keep)
        await set(.dark)
    }

    private func until(seconds: Double = 5, _ condition: @MainActor () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() > deadline { throw QAFailure("timed out") }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    /// The window's frame view (traffic lights included) at 2× pixels; `keep` leaves the window open for the next state.
    private func capture(_ controller: ToolWindowController, _ name: String, keep: Bool = false, size: NSSize = BackgroundView.size, settle: Int = 500) async throws {
        guard let window = controller.window, let root = window.contentView?.superview else { return }
        controller.applyAppearance()
        if window.frame.size != size { window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false) }
        if !window.isVisible {
            window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
            window.orderFrontRegardless()
            window.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        }
        try await Task.sleep(for: .milliseconds(settle))
        root.layoutSubtreeIfNeeded(); window.displayIfNeeded()
        let scale: CGFloat = 2
        guard
            let out = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: Int(root.bounds.width * scale), pixelsHigh: Int(root.bounds.height * scale), bitsPerSample: 8,
                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return }
        out.size = root.bounds.size
        root.cacheDisplay(in: root.bounds, to: out)
        try out.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("background-\(name).png"))
        if !keep { window.orderOut(nil); controller.close() }
        log.say("wrote background-\(name).png (\(Int(root.bounds.width)) × \(Int(root.bounds.height)))")
    }
}
