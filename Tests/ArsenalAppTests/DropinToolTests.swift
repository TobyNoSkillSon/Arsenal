// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

/// The generic operation tool (drop-in models): its form from a schema, its descriptors from the API, its window's
/// controls clicked through the real event path on never-shown windows (as V3ToolTests does), the layers fixture end to
/// end through the app's own API, and the table, detail, menu and activity words for new kinds.
@MainActor @Suite(.menuState, .serialized) final class DropinToolTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        ControlFrames.recording = true
    }

    // MARK: Form

    @Test func theFixtureSchemaBecomesAFileWellAStepperAndASwitch() throws {
        let form = OperationForm(schema: LayersFixture.input)
        #expect(!form.wholeJSON)
        #expect(form.fields.map(\.key) == ["image", "layers", "fill_hidden"])
        #expect(form.fields[0].control == .file(FileAccept(mimes: ["image/png", "image/jpeg"]), multiple: false, maxItems: 1, wrapped: true))
        #expect(form.fields[1].control == .integer(min: 2, max: 8, stepper: true))
        #expect(form.fields[2].control == .toggle)
        #expect(form.initialValue(form.fields[1], setting: nil) == .text("4"))
        // A model setting of the same name wins over the schema default.
        #expect(form.initialValue(form.fields[1], setting: .number(6)) == .text("6"))
        // No image: the required field says so, nothing is staged.
        #expect(throws: OperationForm.Problem(key: "image", message: "Add image first.")) {
            try form.input(
                ["layers": .text("4")],
                stage: { _ in
                    Issue.record("staged"); return ""
                })
        }
        let input = try form.input(
            ["image": .files([URL(fileURLWithPath: "/tmp/a.png")]), "layers": .text("3"), "fill_hidden": .toggle(true)], stage: { "/inbox/x/" + $0.lastPathComponent })
        #expect(input == ["image": ["path": "/inbox/x/a.png"], "layers": 3, "fill_hidden": true])
        try OperationSchemaValidator.validate(.object(input), schema: .object(LayersFixture.input))
        #expect(throws: OperationForm.Problem(key: "layers", message: "Layers must be a whole number.")) {
            try form.input(["image": .files([URL(fileURLWithPath: "/tmp/a.png")]), "layers": .text("three")], stage: { $0.path })
        }
    }

    @Test func controlsFollowTheSchemaAndUnionsBecomeJSON() throws {
        let schema: JSONObject = [
            "type": "object", "required": ["prompt", "mode"],
            "$defs": ["strength": ["type": "number", "minimum": 0, "maximum": 1, "default": 0.5]],
            "properties": [
                "prompt": ["type": "string"],
                "name": ["type": "string", "maxLength": 40],
                "mode": ["enum": ["fast", "fine"]],
                "strength": ["$ref": "#/$defs/strength", "title": "Strength"],
                "seed": ["type": "integer"],
                "box": ["type": "object", "properties": ["x": ["type": "number"]]],
                "either": ["anyOf": [["type": "string"], ["type": "number"]]],
                "frames": ["type": "array", "maxItems": 3, "items": ["type": "string", "x-arsenal-file": ["accept": ["image/*"]]]],
                "stream": ["type": "boolean"], "out_dir": ["type": "string"]
            ]
        ]
        let form = OperationForm(schema: schema)
        let byKey = Dictionary(uniqueKeysWithValues: form.fields.map { ($0.key, $0) })
        #expect(byKey["stream"] == nil && byKey["out_dir"] == nil)
        #expect(byKey["prompt"]?.control == .text(multiline: true))
        #expect(byKey["name"]?.control == .text(multiline: false))
        #expect(byKey["mode"]?.control == .choice(["fast", "fine"]))
        #expect(byKey["strength"]?.control == .number(min: 0, max: 1) && byKey["strength"]?.title == "Strength")
        #expect(byKey["seed"]?.control == .integer(min: nil, max: nil, stepper: false))
        #expect(byKey["box"]?.control == .json && byKey["either"]?.control == .json)
        #expect(byKey["frames"]?.control == .file(FileAccept(mimes: ["image/*"]), multiple: true, maxItems: 3, wrapped: false))
        // Content first: files, then the long text.
        #expect(form.fields.prefix(2).map(\.key) == ["frames", "prompt"])
        let values: [String: FieldValue] = [
            "prompt": .text("hi"), "mode": .choice(1), "strength": .text("0.25"), "box": .text(#"{"x": 2}"#),
            "frames": .files([URL(fileURLWithPath: "/a.png"), URL(fileURLWithPath: "/b.png")])
        ]
        let input = try form.input(values, stage: { $0.path })
        #expect(input == ["prompt": "hi", "mode": "fine", "strength": 0.25, "box": ["x": 2], "frames": ["/a.png", "/b.png"]])
        #expect(throws: OperationForm.Problem(key: "box", message: "Box is not valid JSON.")) {
            var bad = values; bad["box"] = .text("{x:"); _ = try form.input(bad, stage: { $0.path })
        }
        // A union at the top: the whole input is one JSON field.
        let whole = OperationForm(schema: ["oneOf": [["type": "object"], ["type": "array"]]])
        #expect(whole.wholeJSON && whole.fields.map(\.control) == [.json])
        #expect(try whole.input(["": .text(#"{"a": 1}"#)], stage: { $0.path }) == ["a": 1])
    }

    @Test func fileAcceptMatchesByType() {
        let images = FileAccept(mimes: ["image/png", "image/jpeg"])
        #expect(images.accepts(URL(fileURLWithPath: "/x/a.PNG")) && images.accepts(URL(fileURLWithPath: "/x/b.jpeg")))
        #expect(!images.accepts(URL(fileURLWithPath: "/x/c.txt")) && !images.accepts(URL(fileURLWithPath: "/x/d")))
        #expect(FileAccept(mimes: ["image/*"]).accepts(URL(fileURLWithPath: "/x/e.webp")))
        #expect(images.title == "PNG or JPEG" && images.images)
    }

    // MARK: The API's descriptors and a real run

    private func api(_ configure: (FixtureService) async -> Void = { _ in }) async throws -> (FixtureService, APIHost, ArsenalClient, URL) {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-dropin-\(UUID().uuidString)")
        let service = FixtureService(rows: Fixtures.rows(.loaded) + [LayersFixture.row], kinds: Fixtures.kinds + [LayersFixture.kind])
        await configure(service)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        return (service, api, ArsenalClient(port: try #require(api.port)), support)
    }

    @Test func theOperationDescriptorsComeFromTheAPI() async throws {
        let (_, api, client, support) = try await api()
        defer { api.stop(); try? FileManager.default.removeItem(at: support) }
        let ops = try await client.operations(LayersFixture.id)
        let op = try #require(ops.first)
        #expect(ops.count == 1)
        #expect(op.name == "separate" && op.title == "Separate Layers" && op.output == "file" && !op.stream)
        #expect(op.inputSchema["properties"]?["image"]?["x-arsenal-file"]?["accept"] == ["image/png", "image/jpeg"])
        #expect(OperationForm(schema: op.inputSchema).fields.map(\.key) == ["image", "layers", "fill_hidden"])
        // A built-in model's operations read the same way.
        #expect(try await client.operations("qwen3.8-27b").contains { $0.name == "chat" && $0.output == "text-stream" })
    }

    /// The whole path: the window's session stages the dropped image in the inbox, runs `separate` through the app's
    /// own API with JSON negotiation, follows the runner's progress, fetches the four files from their job URLs (hash
    /// checked) and shows them in order with their labels and depth ranges; Save All writes them numbered.
    @Test func theLayersFixtureRunsEndToEndThroughTheAppsOwnAPI() async throws {
        let (service, api, client, support) = try await api { await $0.setDelay("separate", .milliseconds(150)) }
        defer { api.stop(); try? FileManager.default.removeItem(at: support) }
        let source = support.appendingPathComponent("photo.png")
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        try LayersFixture.png(layer: 0, of: 2, size: 64).write(to: source)
        let session = OperationSession(
            model: ToolModel(LayersFixture.row), client: { client }, inbox: support.appendingPathComponent("inbox"), results: support.appendingPathComponent("results"))
        session.prepare(primary: "separate")
        try await waitUntil { !session.loading }
        #expect(session.problem == nil && session.operation == "separate")
        #expect(!session.canRun || session.form != nil)
        session.addFiles([source], to: "image")
        session.set("layers", .text("4"))
        session.start()
        #expect(session.running)
        var labels: Set<String> = []
        try await waitUntil(seconds: 15) {
            if let line = session.run?.line(model: "x") { labels.insert(line) }
            return !session.running
        }
        print("progress seen: \(labels.sorted())")
        #expect(session.problem == nil, "\(session.problem?.message ?? "")")
        let result = try #require(session.result)
        // The same inputs would make the same layers: Run steps back from the blue until an option changes.
        #expect(session.isFresh)
        #expect(result.files.map(\.label) == ["Near", "Middle", "Far", "Background"])
        #expect(result.files.map(\.order) == [0, 1, 2, 3])
        #expect(result.files.allSatisfy { $0.isImage && $0.width == 256 && $0.height == 256 && NSImage(contentsOf: $0.url) != nil })
        #expect(result.files[0].extras == [Pair(key: "depth_min", value: "0"), Pair(key: "depth_max", value: "0.25")])
        let metadata = try WireJSON.decode(Data(try #require(result.json).utf8))
        #expect(metadata["dimensions"] == ["width": 256, "height": 256])
        #expect(metadata["depth_units"] == "relative" && metadata["order"] == "near-to-far")
        #expect(labels.contains { $0.hasPrefix("Separating layers") && $0.hasSuffix("layers") })
        // The runner received a path inside the inbox, never the original file.
        #expect(await service.runLog().contains("\(LayersFixture.id) separate"))
        let staged = try FileManager.default.contentsOfDirectory(atPath: support.appendingPathComponent("inbox").path)
        #expect(staged.isEmpty)
        let folder = support.appendingPathComponent("saved")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        session.chooseFolder = { folder }
        session.saveAll()
        session.set("layers", .text("3"))
        #expect(!session.isFresh)
        #expect(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() == ["1 Near.png", "2 Middle.png", "3 Far.png", "4 Background.png"])
        session.discard()
        #expect(!FileManager.default.fileExists(atPath: support.appendingPathComponent("results").path))
    }

    @Test func cancelEndsTheRequestAndTheWindowKeepsWorking() async throws {
        let (service, api, client, support) = try await api { await $0.setDelay("separate", .seconds(5)) }
        defer { api.stop(); try? FileManager.default.removeItem(at: support) }
        let source = support.appendingPathComponent("photo.png")
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        try LayersFixture.png(layer: 0, of: 2, size: 32).write(to: source)
        let session = OperationSession(model: ToolModel(LayersFixture.row), client: { client }, inbox: support.appendingPathComponent("inbox"))
        defer { session.discard() }
        session.prepare()
        try await waitUntil { !session.loading }
        session.addFiles([source], to: "image")
        session.start()
        try await waitUntil(seconds: 10) { await service.runLog().contains("\(LayersFixture.id) separate") }
        session.stop()
        #expect(!session.running && session.problem == nil && session.result == nil)
        try await waitUntil(seconds: 10) {
            let subscribers = await service.activitySubscribers()
            return !session.hasRunningWork && subscribers == 0
        }
        #expect(session.canRun)
    }

    @Test func refusalsNameTheFixAndUnsupportedFilesAreRefusedInPlace() async throws {
        let consent = OperationProblem.from(ToolError("x", status: 409, code: "code_consent"), model: "Layer Fixture")
        // Fixed in the window: Approve… and Download start the table's own flows; memory offers Load Anyway.
        #expect(consent.action == .approve && consent.message == "Layer Fixture\u{2019}s folder changed since it was approved.")
        let missing = OperationProblem.from(ToolError("x", status: 404, code: "not_installed"), model: "Layer Fixture")
        #expect(missing.action == .download && missing.message == "Layer Fixture isn\u{2019}t on this Mac yet.")
        let memory = OperationProblem.from(ToolError("Not enough memory: needs 4 GB", status: 507, code: "insufficient_memory"), model: "M")
        let words = "Not enough memory: needs 4 GB"
        #expect(memory == .init(message: words, action: .loadAnyway, detail: words))
        var row = LayersFixture.row
        #expect(await FixtureService(rows: [row], metadata: LayersFixture.metadata).runRefusal(id: row.id, op: "separate") == nil)
        row.states = ["fp16": PrecisionState(state: .notInstalled)]
        #expect(await FixtureService(rows: [row], metadata: LayersFixture.metadata).runRefusal(id: row.id, op: "separate")?.code == "not_installed")
        row.states = ["fp16": PrecisionState(state: .ready)]; row.needsCodeConsent = true
        #expect(await FixtureService(rows: [row], metadata: LayersFixture.metadata).runRefusal(id: row.id, op: "separate")?.code == "code_consent")
        let session = OperationSession(model: ToolModel(LayersFixture.row), client: { nil }, inbox: URL(fileURLWithPath: "/nonexistent"))
        session.preview(operations: [LayersFixture.descriptor])
        session.addFiles([URL(fileURLWithPath: "/tmp/notes.txt")], to: "image")
        #expect(session.problem?.field == "image" && session.value("image").files.isEmpty)
    }

    // MARK: The window

    private var scope: UUID?
    private func open(_ session: OperationSession) async throws -> (ToolWindowController, NSWindow) {
        let size = OperationView.size
        let wc = ToolWindowController(title: "test", autosave: "", size: size, minSize: size, content: OperationView(session: session, symbol: LayersFixture.kind.icon!))
        let w = try #require(wc.window)
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
        w.orderFrontRegardless()
        w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        try await settle(w)
        scope = wc.scope
        return (wc, w)
    }
    private func settle(_ w: NSWindow) async throws {
        w.contentView?.layoutSubtreeIfNeeded(); w.displayIfNeeded()
        try await Task.sleep(for: .milliseconds(150))
    }
    private func frame(_ id: String) -> CGRect? { scope.flatMap { ControlFrames.frames[$0]?[id] } }
    private func centre(_ id: String, _ w: NSWindow) throws -> NSPoint {
        let f = try #require(frame(id), "\(id) was not laid out")
        #expect(f.width >= 14 && f.height >= 14, "\(id) is too small to tap: \(f.size)")
        return NSPoint(x: f.midX, y: try #require(w.contentView).bounds.height - f.midY)
    }
    private func expectReachable(_ id: String, _ w: NSWindow, scrolled: Bool = false) throws {
        let p = try centre(id, w)
        let host = try #require(w.contentView)
        let frameView = try #require(host.superview)
        let view = frameView.hitTest(frameView.convert(p, from: nil))
        #expect(view === host || (scrolled && view?.isDescendant(of: host) == true), "\(id): the click reaches \(view.map { String(describing: type(of: $0)) } ?? "nothing")")
    }
    private func tap(_ id: String, _ w: NSWindow, scrolled: Bool = false) async throws {
        try expectReachable(id, w, scrolled: scrolled)
        let p = try centre(id, w)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            let e = try #require(
                NSEvent.mouseEvent(
                    with: type, location: p, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: w.windowNumber, context: nil,
                    eventNumber: 0, clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0))
            w.sendEvent(e)
        }
        try await settle(w)
    }

    static func results(in folder: URL, count: Int = 4) throws -> OperationResult {
        let reply = try LayersFixture.run(params: ["layers": .number(Double(count))], directory: folder)
        let files = reply["files"]!.array!.compactMap(\.object).map { f in
            ResultFile(
                id: f["id"]!.string!, label: f["label"]!.string!, order: Int(f["order"]!.number!), mime: "image/png", url: URL(fileURLWithPath: f["path"]!.string!),
                bytes: 1, width: 256, height: 256,
                extras: [Pair(key: "depth_max", value: ToolFiles.scalar(f["depth_max"]!)), Pair(key: "depth_min", value: ToolFiles.scalar(f["depth_min"]!))])
        }
        return OperationResult(
            operation: "separate", files: files, fields: [Pair(key: "depth_units", value: "relative"), Pair(key: "order", value: "near-to-far")], seconds: 1.21)
    }

    /// First frame: the form, the well and Run are laid out at real size; Run stays off-limits without the image; a
    /// failed run is a line with no crash; the result's Save and Save All take a plain tap; Cancel stops a run.
    @Test func runSaveAndCancelTakeAPlainTapAtRealSize() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-dropin-window-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let session = OperationSession(
            model: ToolModel(LayersFixture.row), client: { nil }, inbox: folder.appendingPathComponent("inbox"), results: folder.appendingPathComponent("results"))
        session.preview(operations: [LayersFixture.descriptor])
        let (wc, w) = try await open(session)
        defer { wc.close() }
        for id in ["run", "well-image", "field-layers", "field-fill_hidden"] { #expect(frame(id) != nil, "\(id) missing from the first frame") }
        try expectReachable("well-image", w, scrolled: true)
        // Without the API the run fails in place.
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let photo = folder.appendingPathComponent("photo.png")
        try LayersFixture.png(layer: 1, of: 3, size: 64).write(to: photo)
        session.addFiles([photo], to: "image")
        try await settle(w)
        try expectReachable("remove-image-photo.png", w, scrolled: true)
        try await tap("run", w)
        try await waitUntil { !session.running }
        #expect(session.problem?.message == ChatSession.noAPI)
        // A result: four tiles, each with Save; Save All.
        session.preview(operations: [LayersFixture.descriptor], result: try Self.results(in: folder.appendingPathComponent("layers")))
        try await settle(w)
        var saved: [String] = []
        session.chooseFile = { name in
            saved.append(name); return folder.appendingPathComponent("out-" + name)
        }
        try await tap("save-layer-2", w, scrolled: true)
        #expect(saved == ["Far.png"] && FileManager.default.fileExists(atPath: folder.appendingPathComponent("out-Far.png").path))
        let all = folder.appendingPathComponent("all")
        try FileManager.default.createDirectory(at: all, withIntermediateDirectories: true)
        session.chooseFolder = { all }
        try await tap("save-all", w)
        #expect(try FileManager.default.contentsOfDirectory(atPath: all.path).count == 4)
        // Running: Cancel replaces Run and stops it.
        session.preview(
            operations: [LayersFixture.descriptor],
            run: ToolRun(model: LayersFixture.id, operation: "separate", phase: "step", label: "Separating layers", step: 2, total: 4, unit: "layers"))
        try await settle(w)
        #expect(frame("stop") != nil)
        try await tap("stop", w)
        #expect(!session.running)
    }

    @Test func aBlockedModelOffersModelsAndKeepsRunOff() async throws {
        var row = LayersFixture.row
        row.states = ["fp16": PrecisionState(state: .notInstalled)]
        let session = OperationSession(model: ToolModel(row), client: { nil }, inbox: URL(fileURLWithPath: "/nonexistent"))
        session.preview(
            operations: [LayersFixture.descriptor], blocked: .init(message: "Layer Fixture isn\u{2019}t on this Mac yet.", action: .download, actionTitle: "Download 120 MB"))
        var prepared = 0
        session.prepareModel = { _ in prepared += 1 }
        let (wc, w) = try await open(session)
        defer { wc.close() }
        #expect(!session.canRun)
        // The run button is the Download, as in every window.
        try await tap("run", w)
        #expect(prepared == 1)
    }

    @Test func progressAndResultWords() {
        #expect(
            ToolRun(model: LayersFixture.id, operation: "s", phase: "step", label: "Separating layers", step: 2, total: 4, unit: "layers").line(model: "M")
                == "Separating layers \u{00b7} 2 of 4 layers")
        #expect(ToolRun(model: LayersFixture.id, operation: "s", phase: "loading").line(model: "Layer Fixture") == "Loading Layer Fixture")
        #expect(ToolRun(model: LayersFixture.id, operation: "s").line(model: "M") == "Starting")
        #expect(ToolRun(model: LayersFixture.id, operation: "s", step: 3).line(model: "M") == "Step 3")
        #expect(FileGrid.columns(1) == 1 && FileGrid.columns(4) == 2 && FileGrid.columns(6) == 3 && FileGrid.columns(8) == 4)
        let file = ResultFile(id: "layer-0", label: "Near / front", order: 0, mime: "image/png", url: URL(fileURLWithPath: "/x/a.png"), bytes: 1, extras: [])
        #expect(file.suggestedName == "Near - front.png")
    }

    // MARK: Menu, table, details, activity

    @Test func hidingTheMenuItemKeepsAnOpenOperationUsable() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-hidden-operation-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let row = LayersFixture.row
        let service = FixtureService(rows: [row], metadata: LayersFixture.metadata, kinds: Fixtures.kinds + [LayersFixture.kind], supportDirectory: root)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root))
        await app.api.start(); defer { app.api.stop(); app.table.stop(); app.tools.closeAll() }
        app.tools.present = { _ in }
        await app.table.refresh(); await app.refreshMenuState()
        app.tools.open(.operation, row)
        let session = try #require(app.tools.operations[row.id]?.session)
        try await waitUntil { !session.loading }
        #expect(session.blocked == nil && session.descriptor?.name == "separate")
        try await service.setToolsShown(id: row.id, on: false)
        await app.table.refresh(); await app.refreshMenuState()
        // Hidden, not removed: Show in menu changes no structure.
        #expect(!app.menu.items.contains { $0.representedObject as? [String] == ["operation", row.id] && !$0.isHidden })
        #expect(session.blocked == nil && session.descriptor?.name == "separate")
    }

    @Test func aLoadedNewKindGetsItsToolInTheMenuAndDetails() throws {
        let center = ToolCenter(client: { nil }, playback: FixturePlayback())
        center.kinds = { [LayersFixture.kind] }
        let sections = MenuContent.toolSections([LayersFixture.row], sections: [])
        #expect(sections.map(\.tools) == [[.operation]])
        #expect(center.menuTitle(.operation, LayersFixture.row) == "Separate Layers\u{2026}")
        #expect(center.symbol(.operation, LayersFixture.row) == "square.3.layers.3d")
        #expect(center.menuTitle(.chat, LayersFixture.row) == "Chat\u{2026}")
        let entry = CatalogEntry(row: LayersFixture.row, added: true)
        let detail = ModelDetail(BrowserItem(entry, kind: LayersFixture.kind))
        #expect(detail.actions == [.docs, .openTool("operation", title: "Separate Layers\u{2026}")])
        var notHere = LayersFixture.row
        notHere.states = ["fp16": PrecisionState(state: .notInstalled)]
        #expect(ModelDetail(BrowserItem(CatalogEntry(row: notHere, added: true), kind: LayersFixture.kind)).actions == [.docs])
    }

    @Test func checksShowTheirCompactStateWithEvidenceOnRequest() {
        var row = LayersFixture.row
        row.checks = ModelChecksStatus(
            protocolStatus: .pass, parity: .notRun, task: .notRun, implementationFingerprint: "abc123def456789", cases: ["protocol": 6],
            evidence: ["results/2026-10-08/report.json"])
        var line = RowPresentation.status(row, selected: "fp16", chip: "M5 Max")
        #expect(line.text == "Local \u{00b7} Quick check passed" && line.tone == .busy)
        #expect(line.help.contains("Protocol passed \u{00b7} Parity not run \u{00b7} Task not run") && line.help.contains("Tested implementation abc123def456"))
        row.checks?.stale = true
        line = RowPresentation.status(row, selected: "fp16", chip: nil)
        #expect(line.text == "Local \u{00b7} Stale" && line.tone == .estimate && line.help.contains("no longer describe what runs"))
        row.checks?.stale = false; row.checks?.parity = .fail; row.local = false
        #expect(RowPresentation.status(row, selected: "fp16", chip: nil).text == "Failed")
        let detail = ModelDetail(BrowserItem(CatalogEntry(row: row, added: true), kind: LayersFixture.kind))
        // No harness vocabulary on a visible line: the failure is the row's Status, with the breakdown on its hover.
        #expect(!detail.about.contains { $0.key == "Checks" })
        #expect(BrowserItem.statusHelp(.failedChecks, row).hasPrefix("Protocol passed \u{00b7} Parity failed"))
    }

    /// A new revision waits quietly for idle; rollback is a detail action only.
    @Test func aPendingRevisionIsAQuietLineAndRollbackLivesInDetails() {
        var row = LayersFixture.row
        row.revision = ModelRevision(active: "r1", pending: "r2", previous: nil)
        #expect(RowPresentation.status(row, selected: "fp16", chip: nil).text == "Update ready")
        #expect(
            ModelDetail(BrowserItem(CatalogEntry(row: row, added: true), kind: LayersFixture.kind)).actions == [
                .docs, .openTool("operation", title: "Separate Layers\u{2026}")
            ])
        row.revision = ModelRevision(active: "r2", pending: nil, previous: "r1", swapping: true)
        #expect(RowPresentation.status(row, selected: "fp16", chip: nil).text == "Updating\u{2026}")
        row.revision = ModelRevision(active: "r2", pending: nil, previous: "r1")
        row.checks = nil; row.local = false
        #expect(RowPresentation.status(row, selected: "fp16", chip: nil).text != "Update ready \u{00b7} switches when idle")
        let detail = ModelDetail(BrowserItem(CatalogEntry(row: row, added: true), kind: LayersFixture.kind))
        #expect(detail.actions.dropFirst().first == .rollback && detail.actions.dropFirst().first?.title == "Use the previous revision")
        row.revision = ModelRevision(active: "r1", pending: "r2", previous: nil, error: "The new revision failed to load: hello timed out.")
        let failed = RowPresentation.status(row, selected: "fp16", chip: nil)
        #expect(failed.tone == .error && failed.help.contains("hello timed out"))
    }

    @Test func aNewKindsUnitFollowsItsNumber() {
        var row = LayersFixture.row
        row.badge = .measured; row.metrics = ["fps": 23.4, "rtf": 42.0]
        #expect(MetricText(row, KindColumn(id: "fps", title: "Frames", unit: "fps")).value == "23.4 fps")
        #expect(MetricText(row, KindColumn(id: "rtf", title: "Speed", unit: "\u{00d7} real time")).value == "42.0")
        #expect(MetricText(row, KindColumn(id: "missing", title: "Missing", unit: "px")).value == "\u{2014}")
    }

    @Test func activityNamesTheOperationAndItsUnit() {
        let json: [String: Any] = [
            "models": [
                [
                    "id": LayersFixture.id, "kind": LayersFixture.kind.id, "state": "busy",
                    "requests": [
                        [
                            "id": "r1", "requester": "Arsenal", "op": "separate", "phase": "step", "title": "Separate Layers", "label": "Separating layers", "unit": "layers",
                            "step": 2, "total": 4
                        ]
                    ]
                ]
            ]
        ]
        let feed = ActivityFeed(json: json)
        let request = try? #require(feed.models.first?.requests.first)
        #expect(request?.title == "Separate Layers" && request?.label == "Separating layers" && request?.unit == "layers")
        #expect(ActivityText.detail(request!, .default) == "2 / 4")
        #expect(ActivityText.doing(request!, .default) == "Separating layers")
        #expect(ActivityText.showsBar(request!, .default))
        // Built-in sections keep their own form.
        var image = request!; image.label = nil; image.unit = nil; image.title = nil
        #expect(ActivityText.detail(image, .image) == "2 / 4 steps")
        var plain = request!; plain.label = "Working"
        #expect(ActivityText.detail(plain, .default) == "2 / 4 layers")
        #expect(ActivitySection(row: LayersFixture.row, kind: LayersFixture.kind.id) == .default)
    }

    private func waitUntil(seconds: Double = 5, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            if Date() > deadline { Issue.record("timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}

extension DropinToolTests {
    @Test func constantPropertiesAreHiddenAndSentWithoutEditableValues() throws {
        let schema: JSONObject = [
            "type": "object", "required": ["prompt", "guidance", "empty", "none"],
            "properties": [
                "prompt": ["type": "string"], "guidance": ["type": "number", "const": 1, "default": 99],
                "empty": ["type": "string", "const": ""], "none": ["const": .null],
                "stream": ["const": false], "out_dir": ["const": "/tmp/ignored"],
                "flag": ["$ref": "#/$defs/fixed"]
            ], "$defs": ["fixed": ["type": "boolean", "const": true]], "additionalProperties": false
        ]
        let form = OperationForm(schema: schema)
        #expect(!form.wholeJSON && form.fields.map(\.key) == ["prompt"])
        let input = try form.input(["prompt": .text("A kettle"), "guidance": .text("99")], stage: { $0.path })
        #expect(input == ["prompt": "A kettle", "guidance": 1, "empty": "", "none": .null, "flag": true])
        try OperationSchemaValidator.validate(.object(input), schema: .object(schema))
    }
}
