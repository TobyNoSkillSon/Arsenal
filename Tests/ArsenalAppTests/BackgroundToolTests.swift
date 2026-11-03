// SPDX-License-Identifier: MIT
import AppKit
import ImageIO
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import Arsenal
import ArsenalCore

/// Remove Background: BiRefNet's folder contract as the window reads it, images dropped (files and bytes), chosen and
/// pasted, each cut end to end through the app's own API over the stub service's fixture runner, inputs the operation
/// cannot take as they are converted to upright PNGs, a model's own options and operations shown from its schema, the
/// window's controls clicked through the real event path on a never-shown window, and the menu entry opening the one
/// window. No model, no GPU; no panel, Finder window or menu is shown.
@MainActor @Suite(.menuState, .serialized) final class BackgroundToolTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        ControlFrames.recording = true
    }
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-background-tests-\(UUID().uuidString)", isDirectory: true)
    deinit { try? FileManager.default.removeItem(at: scratch) }
    private static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    // MARK: Fixtures

    /// A photo-like image: a warm gradient with a darker disc in the middle, `type` PNG, JPEG or TIFF, with an EXIF
    /// orientation when given.
    static func image(_ url: URL, width: Int = 300, height: Int = 200, type: UTType = .jpeg, orientation: Int? = nil) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let context = try #require(
            CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let colours = [CGColor(red: 0.95, green: 0.62, blue: 0.3, alpha: 1), CGColor(red: 0.35, green: 0.55, blue: 0.85, alpha: 1)] as CFArray
        let gradient = try #require(CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: colours, locations: [0, 1]))
        context.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: width, y: height), options: [])
        context.setFillColor(CGColor(red: 0.2, green: 0.25, blue: 0.3, alpha: 1))
        context.fillEllipse(in: CGRect(x: width / 3, y: height / 4, width: width / 3, height: height / 2))
        let cg = try #require(context.makeImage())
        let dest = try #require(CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, cg, orientation.map { [kCGImagePropertyOrientation: $0] as CFDictionary })
        #expect(CGImageDestinationFinalize(dest))
        return url
    }

    /// Alpha at a point (top-left origin) of a PNG.
    static func alpha(_ url: URL, x: Int, y: Int) throws -> UInt8 {
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        var pixel = [UInt8](repeating: 0, count: 4)
        let context = try #require(
            CGContext(
                data: &pixel, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.draw(image, in: CGRect(x: -x, y: -(image.height - 1 - y), width: image.width, height: image.height))
        return pixel[3]
    }

    private struct Lab {
        let service: FixtureService
        let api: APIHost
        let session: BackgroundSession
        let support: URL
        @MainActor func stop() { session.discard(); api.stop() }
    }

    /// The stub service with BiRefNet's row and operation (or `operations`), the app's API over it, and a window session
    /// reading through that API.
    private func lab(_ name: String, operations: [ModelOperation] = [CutoutFixture.operation]) async throws -> Lab {
        let support = scratch.appendingPathComponent(name)
        let service = FixtureService(rows: [CutoutFixture.row], metadata: CutoutFixture.metadata, supportDirectory: support, operations: [CutoutFixture.id: operations])
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        let port = try #require(api.port)
        let session = BackgroundSession(
            models: [ModelChoice(model: ToolModel(CutoutFixture.row))], selected: CutoutFixture.id, client: { ArsenalClient(port: port) },
            inbox: support.appendingPathComponent("inbox"), outputDirectory: support.appendingPathComponent("Pictures"), scratch: support.appendingPathComponent("window"))
        session.prepare()
        try await waitUntil { !session.loading }
        return Lab(service: service, api: api, session: session, support: support)
    }

    // MARK: Contract

    @Test func biRefNetsFolderDeclaresTheToolAndTheFixtureKeepsItsSchema() throws {
        let folder = Self.repo.appendingPathComponent("models/birefnet")
        let model = try ModelManifest.load(folder, kindsDirectory: Self.repo.appendingPathComponent("Resources/kinds"))
        #expect(model.tools == [ToolKind.background.rawValue])
        let declared = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/remove_background.input.json")))
        #expect(declared == CutoutFixture.input, "the stub runner must take exactly what BiRefNet's folder declares")
        let schema = try WireJSON.decode(Data(contentsOf: Self.repo.appendingPathComponent("Resources/schemas/model.schema.json")))
        #expect(schema["properties"]?["tools"]?["items"]?["enum"]?.array?.contains("background") == true)
        // Only the image: the window shows the well and nothing else.
        let op = CutoutFixture.descriptor
        #expect(BackgroundSession.takesImage(op))
        #expect(BackgroundSession.imageField(op)?.key == "image" && BackgroundSession.imageField(op)?.accept.mimes == ["image/png", "image/jpeg"])
        #expect(OperationForm(schema: op.inputSchema).fields.map(\.key) == ["image"])
        #expect(ToolKind.background.title == "Remove Background\u{2026}")
    }

    // MARK: Drop, pick, paste

    @Test func aDroppedFileIsCutOutThroughTheAPIAndSavedWithTransparency() async throws {
        let lab = try await lab("drop")
        defer { lab.stop() }
        #expect(lab.session.operations.map(\.name) == ["remove_background"] && lab.session.optionFields.isEmpty)
        let photo = try Self.image(scratch.appendingPathComponent("photos/beach walk.jpg"))
        // Finder drags a file URL.
        #expect(BackgroundView.receive([NSItemProvider(object: photo as NSURL)], session: lab.session))
        try await waitUntil { lab.session.cutouts.count == 1 && !lab.session.running }
        #expect(lab.session.problem == nil)
        let cutout = try #require(lab.session.cutouts.first)
        #expect(lab.session.input == photo && lab.session.current == cutout && cutout.original == photo)
        // Saved as a PNG with the model's name and the image's; transparent outside the subject, opaque inside.
        #expect(cutout.url.deletingLastPathComponent() == lab.support.appendingPathComponent("Pictures"))
        #expect(cutout.url.pathExtension == "png" && cutout.url.lastPathComponent.hasSuffix(" birefnet beach walk.png"))
        #expect(cutout.width == 300 && cutout.height == 200)
        // The mask is saved beside it: "<cut-out name> mask.png", grey, the same size.
        let mask = try #require(cutout.mask)
        #expect(mask.deletingLastPathComponent() == cutout.url.deletingLastPathComponent())
        #expect(mask.lastPathComponent == cutout.url.deletingPathExtension().lastPathComponent + " mask.png")
        #expect(PixelSize.read(mask) == PixelSize(width: 300, height: 200))
        #expect(try Self.alpha(cutout.url, x: 2, y: 2) == 0)
        #expect(try Self.alpha(cutout.url, x: 150, y: 100) == 255)
        // The request: one run of remove_background with the image staged in the inbox as it was (a JPEG is taken).
        #expect(await lab.service.runLog() == ["birefnet remove_background"])
        let path = try #require(await lab.service.runInputs.first?["image"]?["path"]?.string)
        #expect(path.hasPrefix(lab.support.appendingPathComponent("inbox").path) && path.hasSuffix("beach walk.jpg"))
        #expect(!FileManager.default.fileExists(atPath: path), "the staged copy is removed after the run")
        let figures = lab.session.figures()
        // Figures of the run only (the size is under the well).
        #expect(figures.map(\.title) == ["Time", "Peak memory"])
        // After a finished run no figure is a bare dash: a missing reading says so.
        #expect(figures[0].value != nil && figures[1].value != nil)
        // Nothing to set and nothing changed: running again would make the same cut-out.
        #expect(lab.session.isFresh)
    }

    @Test func droppedBytesAndAPickedTIFFAreSentAsPNG() async throws {
        let lab = try await lab("bytes")
        defer { lab.stop() }
        // An image dragged from a browser or Photos: bytes, no file.
        let png = try Data(contentsOf: try Self.image(scratch.appendingPathComponent("bytes/source.png"), type: .png))
        let provider = NSItemProvider(item: png as NSData, typeIdentifier: UTType.png.identifier)
        provider.suggestedName = "from safari"
        #expect(BackgroundView.receive([provider], session: lab.session))
        try await waitUntil { lab.session.cutouts.count == 1 && !lab.session.running }
        #expect(lab.session.cutouts[0].original.lastPathComponent == "from safari.png")
        #expect(lab.session.cutouts[0].original.path.hasPrefix(lab.support.appendingPathComponent("window").path))
        // Chosen from disk: a TIFF, which BiRefNet does not take, goes as a PNG.
        let tiff = try Self.image(scratch.appendingPathComponent("disk/scan.tiff"), width: 120, height: 90, type: .tiff)
        var asked = 0
        lab.session.chooseImage = {
            asked += 1; return tiff
        }
        lab.session.pick()
        try await waitUntil { lab.session.cutouts.count == 2 && !lab.session.running }
        #expect(asked == 1 && lab.session.problem == nil)
        let sent = try #require(await lab.service.runInputs.last?["image"]?["path"]?.string)
        #expect(sent.hasSuffix("scan.png"))
        #expect(lab.session.cutouts[1].width == 120 && lab.session.cutouts[1].height == 90)
        // Not an image: a line, no run.
        let text = scratch.appendingPathComponent("disk/notes.txt")
        try Data("hello".utf8).write(to: text)
        lab.session.setInput([text])
        #expect(lab.session.problem?.message.hasPrefix("notes.txt is not an image") == true)
        #expect(await lab.service.runLog().count == 2)
    }

    @Test func aRotatedPhotoIsStagedUpright() throws {
        // EXIF 6: stored 300 × 200, seen 200 × 300.
        let photo = try Self.image(scratch.appendingPathComponent("rotated/phone.jpg"), orientation: 6)
        #expect(PixelSize.read(photo) == PixelSize(width: 200, height: 300))
        let accept = FileAccept(mimes: ["image/png", "image/jpeg"])
        let staged = try ToolFiles.stage(photo, accept: accept, inbox: scratch.appendingPathComponent("rotated-inbox"))
        #expect(staged.path.hasSuffix("phone.png"))
        #expect(PixelSize.read(URL(fileURLWithPath: staged.path)) == PixelSize(width: 200, height: 300))
        // An upright JPEG is copied as it is.
        let upright = try Self.image(scratch.appendingPathComponent("rotated/upright.jpg"))
        let copy = try ToolFiles.stage(upright, accept: accept, inbox: scratch.appendingPathComponent("rotated-inbox"))
        #expect(copy.path.hasSuffix("upright.jpg"))
        #expect(try Data(contentsOf: URL(fileURLWithPath: copy.path)) == Data(contentsOf: upright))
    }

    @Test func pasteTakesACopiedFileOrCopiedImageAndTheWindowAnswersCommandV() async throws {
        let lab = try await lab("paste")
        defer { lab.stop() }
        let board = NSPasteboard(name: .init("arsenal-background-test-\(UUID().uuidString)"))
        defer { board.releaseGlobally() }
        // Nothing usable: not taken.
        board.clearContents(); board.setString("words", forType: .string)
        #expect(!lab.session.paste(board))
        // A file copied in Finder.
        let photo = try Self.image(scratch.appendingPathComponent("paste/copied.jpg"))
        board.clearContents(); board.writeObjects([photo as NSURL])
        #expect(lab.session.paste(board))
        try await waitUntil { lab.session.cutouts.count == 1 && !lab.session.running }
        #expect(lab.session.cutouts[0].original == photo)
        // A copied picture (PNG bytes).
        board.clearContents(); board.setData(try Data(contentsOf: try Self.image(scratch.appendingPathComponent("paste/shot.png"), type: .png)), forType: .png)
        #expect(lab.session.paste(board))
        try await waitUntil { lab.session.cutouts.count == 2 && !lab.session.running }
        #expect(lab.session.cutouts[1].original.lastPathComponent == "Pasted image.png")
        // ⌘V reaches the window's delegate only where a tool takes pastes.
        let plain = ToolWindowController(title: "t", autosave: "", size: BackgroundView.size, minSize: BackgroundView.minSize, content: EmptyView())
        defer { plain.close() }
        #expect(!plain.responds(to: #selector(NSText.paste(_:))))
        var pasted = 0
        plain.onPaste = { _ in
            pasted += 1; return true
        }
        #expect(plain.responds(to: #selector(NSText.paste(_:))))
        #expect(NSApp.sendAction(#selector(NSText.paste(_:)), to: plain, from: nil) && pasted == 1)
    }

    /// Cut-out | Mask shows either file of the same run; Copy and Show in Finder act on the one shown. No Save: the
    /// cut-out and its mask are saved on arrival.
    @Test func copyActsOnTheShownCutOutOrMaskWithTransparencyKept() async throws {
        let lab = try await lab("copy")
        defer { lab.stop() }
        lab.session.setInput([try Self.image(scratch.appendingPathComponent("copy/cat.jpg"))])
        try await waitUntil { lab.session.cutouts.count == 1 && !lab.session.running }
        let cutout = try #require(lab.session.current)
        let board = NSPasteboard(name: .init("arsenal-background-copy-\(UUID().uuidString)"))
        defer { board.releaseGlobally() }
        lab.session.copyPasteboard = board
        #expect(!lab.session.showsMask && lab.session.shownFile(cutout) == cutout.url)
        lab.session.copy(cutout)
        #expect(board.data(forType: .png) == (try Data(contentsOf: cutout.url)))
        #expect(board.data(forType: .tiff) != nil)
        #expect(board.string(forType: .fileURL) == cutout.url.absoluteString)
        #expect(try Self.alpha(cutout.url, x: 1, y: 1) == 0)
        lab.session.showsMask = true
        let mask = try #require(cutout.mask)
        #expect(lab.session.shownFile(cutout) == mask)
        lab.session.copy(cutout)
        #expect(board.data(forType: .png) == (try Data(contentsOf: mask)))
        #expect(board.string(forType: .fileURL) == mask.absoluteString)
        #expect(await lab.service.runLog().count == 1, "the mask is the same run's")
        // A run without a mask: the cut-out is all there is to show.
        var maskless = cutout
        maskless.mask = nil
        #expect(lab.session.shownFile(maskless) == cutout.url)
    }

    // MARK: A model's own options

    /// "each model when it's reusing that tool, it only enables the options for the tool that the model
    /// supports". A model with two operations and two more inputs: both operations are pills, both inputs are controls,
    /// and the values reach the service.
    @Test func aModelsExtraInputsAndOperationsAppearFromItsSchema() async throws {
        var properties = try #require(CutoutFixture.input["properties"]?.object)
        properties["resolution"] = ["enum": [512, 1024, 2048], "default": 1024, "title": "Resolution"]
        properties["threshold"] = ["type": "number", "minimum": 0, "maximum": 1, "default": 0.5, "description": "Mask cut-off"]
        var input = CutoutFixture.input
        input["properties"] = .object(properties)
        let removal = ModelOperation(
            name: "remove_background", title: "Remove background", inputSchema: input, output: "file", stream: true, policy: CutoutFixture.operation.policy, schemaVersion: 3)
        let matte: JSONObject = [
            "type": "object", "required": ["image"], "additionalProperties": false,
            "properties": [
                "image": CutoutFixture.input["properties"]!["image"]!, "edge": ["type": "integer", "minimum": 0, "maximum": 8, "default": 2, "title": "Edge"]
            ]
        ]
        let ops = [
            removal, ModelOperation(name: "matte", title: "Alpha Matte", inputSchema: matte, output: "file", stream: false, schemaVersion: 3),
            // Not an image operation: never offered here.
            ModelOperation(name: "describe", title: "Describe", inputSchema: ["type": "object"], output: "json", stream: false, schemaVersion: 3)
        ]
        let lab = try await lab("options", operations: ops)
        defer { lab.stop() }
        let session = lab.session
        #expect(session.operations.map(\.name) == ["remove_background", "matte"])
        #expect(session.operation == "remove_background")
        #expect(Set(session.optionFields.map(\.key)) == ["resolution", "threshold"] && session.optionFields.count == 2)
        #expect(session.value("resolution") == .choice(1))
        // The window draws them (the cut's tuning under Advanced): the pills and both controls, reachable.
        session.advancedOpen = true
        let (wc, w) = try await open(session)
        defer { wc.close() }
        for id in ["op-remove_background", "op-matte", "field-resolution", "field-threshold", "well-image", "remove"] { #expect(frame(id) != nil, "\(id) not laid out") }
        try expectReachable("field-resolution", w)
        try await tap("op-matte", w)
        #expect(session.operation == "matte" && session.optionFields.map(\.key) == ["edge"])
        #expect(BackgroundView.runTitle(session.descriptor) == "Alpha Matte" && BackgroundView.runTitle(CutoutFixture.descriptor) == "Remove Background")
        try await tap("op-remove_background", w)
        #expect(session.operation == "remove_background")
        // The values go with the image.
        session.set("resolution", .choice(2)); session.set("threshold", .text("0.7"))
        session.setInput([try Self.image(scratch.appendingPathComponent("options/dog.jpg"))])
        try await waitUntil { session.cutouts.count == 1 && !session.running }
        let sent = try #require(await lab.service.runInputs.last)
        #expect(sent["resolution"] == 2048 && sent["threshold"] == 0.7 && sent["image"]?["path"] != nil)
        // Changing an option after the cut: running again would make something new, so the button is blue again.
        #expect(session.isFresh)
        session.set("threshold", .text("0.4"))
        #expect(!session.isFresh)
    }

    // MARK: The window's controls

    private func open(_ session: BackgroundSession) async throws -> (ToolWindowController, NSWindow) {
        let wc = ToolWindowController(title: "test", autosave: "", size: BackgroundView.size, minSize: BackgroundView.size, content: BackgroundView(session: session))
        let w = try #require(wc.window)
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BackgroundView.size), display: false)
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
    private var scope: UUID?
    private func frame(_ id: String) -> CGRect? { scope.flatMap { ControlFrames.frames[$0]?[id] } }
    private func centre(_ id: String, _ w: NSWindow) throws -> NSPoint {
        let f = try #require(frame(id), "\(id) was not laid out")
        #expect(f.width >= 14 && f.height >= 14, "\(id) is too small to tap: \(f.size)")
        return NSPoint(x: f.midX, y: try #require(w.contentView).bounds.height - f.midY)
    }
    /// The hit view at the control's centre is the SwiftUI host, or (over a drop place) a view inside it.
    private func expectReachable(_ id: String, _ w: NSWindow) throws {
        let p = try centre(id, w)
        let host = try #require(w.contentView)
        let frameView = try #require(host.superview)
        let view = frameView.hitTest(frameView.convert(p, from: nil))
        #expect(view === host || view?.isDescendant(of: host) == true, "\(id): the click reaches \(view.map { String(describing: type(of: $0)) } ?? "nothing")")
    }
    private func tap(_ id: String, _ w: NSWindow) async throws {
        try expectReachable(id, w)
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

    /// First frame: the well and Remove Background (off without an image). A tap on the well asks for an image and the
    /// cut starts; Cut-out | Mask and Copy take a plain tap; Show in Finder is reachable (not tapped: it would open
    /// Finder); the strip picks an earlier cut-out and puts its image back in the well; × clears the image.
    @Test func wellMaskCopyAndStripTakeAPlainTapAtRealSize() async throws {
        let lab = try await lab("window")
        defer { lab.stop() }
        let session = lab.session
        let (wc, w) = try await open(session)
        defer { wc.close() }
        for id in ["model", "reveal", "well-image", "remove"] { #expect(frame(id) != nil, "\(id) missing from the first frame") }
        #expect(!session.canRun)
        let first = try Self.image(scratch.appendingPathComponent("window/first.jpg"))
        let second = try Self.image(scratch.appendingPathComponent("window/second.png"), width: 160, height: 240, type: .png)
        var choices = [first, second]
        session.chooseImage = { choices.isEmpty ? nil : choices.removeFirst() }
        try await tap("well-image", w)
        try await waitUntil { session.cutouts.count == 1 && !session.running }
        try await settle(w)
        for id in ["copy", "reveal", "remove-image-first.jpg", "view-cutout", "view-mask"] { try expectReachable(id, w) }
        // One Show in Finder: the band's, which shows the cut-out on view.
        #expect(frame("reveal-result") == nil)
        #expect(frame("save") == nil, "no Save: the cut-out is saved on arrival")
        let board = NSPasteboard(name: .init("arsenal-background-window-\(UUID().uuidString)"))
        defer { board.releaseGlobally() }
        session.copyPasteboard = board
        try await tap("copy", w)
        #expect(board.data(forType: .png) == (try Data(contentsOf: try #require(session.current).url)))
        try await tap("view-mask", w)
        #expect(session.showsMask)
        try await tap("copy", w)
        #expect(board.data(forType: .png) == (try Data(contentsOf: try #require(session.current?.mask))))
        try await tap("view-cutout", w)
        #expect(!session.showsMask)
        // A second image through the well; then the first cut-out from the strip.
        try await tap("well-image", w)
        try await waitUntil { session.cutouts.count == 2 && !session.running }
        try await settle(w)
        #expect(session.input == second)
        try await tap("cutout-\(session.cutouts[0].id)", w)
        #expect(session.current?.id == session.cutouts[0].id && session.input == first && session.isFresh)
        // Remove Background runs the image again.
        try await tap("remove", w)
        try await waitUntil { session.cutouts.count == 3 && !session.running }
        #expect(session.cutouts[2].original == first)
        try await tap("remove-image-first.jpg", w)
        #expect(session.input == nil && session.current == nil && !session.canRun && session.cutouts.count == 3)
    }

    @Test func stopEndsTheCutAndAFailureIsALineInPlace() async throws {
        let lab = try await lab("stop")
        defer { lab.stop() }
        await lab.service.setDelay(CutoutFixture.operation.name, .seconds(5))
        let session = lab.session
        let (wc, w) = try await open(session)
        defer { wc.close() }
        session.setInput([try Self.image(scratch.appendingPathComponent("stop/slow.jpg"))])
        try await waitUntil { session.running }
        try await settle(w)
        try await tap("stop", w)
        #expect(!session.running && session.cutouts.isEmpty && session.problem == nil)
        // Without the API: the reason, with Try Again.
        let offline = BackgroundSession(
            models: [ModelChoice(model: ToolModel(CutoutFixture.row))], selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        offline.prepare()
        try await waitUntil { !offline.loading }
        #expect(offline.problem == .init(message: ChatSession.noAPI, action: .retry))
        // A model not on this Mac: listed with its reason, Remove Background off.
        let blocked = BackgroundSession(
            models: [
                ModelChoice(
                    model: ToolModel(CutoutFixture.row), unavailable: .init(message: "BiRefNet isn\u{2019}t on this Mac yet.", action: .download, actionTitle: "Download 424 MB"))
            ],
            selected: CutoutFixture.id, client: { nil }, inbox: scratch, outputDirectory: scratch)
        blocked.preview(operations: [CutoutFixture.descriptor], input: try Self.image(scratch.appendingPathComponent("stop/x.jpg")))
        #expect(!blocked.canRun && blocked.blocked?.action == .download && blocked.preparesInstead?.actionTitle == "Download 424 MB")
    }

    // MARK: Menu

    @Test func theMenuEntryOpensTheOneWindowWithBiRefNetChosen() async throws {
        let root = scratch.appendingPathComponent("menu")
        let service = FixtureService(
            rows: [CutoutFixture.row], metadata: CutoutFixture.metadata, supportDirectory: root, operations: [CutoutFixture.id: [CutoutFixture.operation]])
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root))
        await app.api.start(); defer { app.api.stop(); app.table.stop(); app.tools.closeAll() }
        app.tools.present = { _ in }
        app.tools.backgroundDirectory = scratch.appendingPathComponent("menu-pictures")
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        let items = app.menu.items.filter { ($0.representedObject as? String) == AppDelegate.taskKey(.tool(.background)) && !$0.isHidden }
        // One entry for the task; the model is a choice inside the window, and its name heads nothing in the menu.
        #expect(items.map(\.title) == ["Remove Background\u{2026}"])
        let item = try #require(items.first)
        #expect(!app.menu.items.contains { $0.title == CutoutFixture.row.name })
        #expect(item.isEnabled && item.view == nil && item.image != nil && item.submenu == nil)
        app.menu.performActionForItem(at: app.menu.index(of: item))
        let open = try #require(app.tools.background)
        #expect(open.session.selected == CutoutFixture.id && open.window.window?.title == "Remove Background")
        try await waitUntil { !open.session.loading }
        #expect(open.session.operations.map(\.name) == ["remove_background"])
        // Again: the same window.
        app.menu.performActionForItem(at: app.menu.index(of: item))
        #expect(app.tools.background?.window === open.window)
        #expect(open.window.responds(to: #selector(NSText.paste(_:))))
        #expect(TableController.toolsSetting(CutoutFixture.row)?.declaration["title"]?.string == "Show Remove Background in menu")
    }

    private func waitUntil(seconds: Double = 8, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            if Date() > deadline { Issue.record("timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
