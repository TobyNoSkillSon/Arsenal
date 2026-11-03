// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ImageIO
import ArsenalCore
import UniformTypeIdentifiers

// Remove Background, as the tool kit composes it: one window for every model that declares the background tool (as Make
// a Sound is for sound); the chosen model's image operations (from the API's descriptors: `remove_background` and any other that
// takes one image and returns files), the chosen one's inputs as its schema declares them (the image is the well; every
// other property gets the generic tool's control, so a model's own options appear without a change here); the image to
// cut out (dropped, chosen or pasted); one run at a time through the universal operation route (JSON negotiation, its
// files fetched from their job URLs); its progress from the activity stream (C3); and this window's cut-outs, each saved
// to ~/Pictures/Arsenal as a PNG with its transparency as it arrives. An image the operation cannot take as it is (HEIC,
// TIFF, WebP, a rotated photo) is sent as an upright PNG.

/// One cut-out of this window.
struct Cutout: Identifiable, Equatable {
    let id = UUID()
    /// The image it was cut from: the user's own file, or a pasted or dragged-in image kept in the window's folder.
    var original: URL
    /// The saved PNG, its background transparent.
    var url: URL
    /// The model's mask (its alpha as grey), saved beside the cut-out as `… mask.png`; nil when the model returned none.
    var mask: URL?
    var model: String
    var modelName: String
    var width: Int
    var height: Int
    /// From Remove Background to the saved file, by this window's clock.
    var seconds: Double
    /// The operation and the other inputs it ran with: Remove Background runs again only when something differs.
    var operation = "remove_background"
    var values: [String: FieldValue] = [:]
    /// A video's selection: `url` is the model's mask video (white where the subject is), never an alpha matte.
    var isVideo: Bool { FileAccept(mimes: ["video/*"]).accepts(url) }
}

@MainActor final class BackgroundSession: ToolSession {
    /// The operation a model declaring the background tool offers (models/birefnet/ops/remove_background.*.json).
    static let operation = "remove_background"

    /// The image the next run cuts out, shown in the well.
    @Published private(set) var input: URL?
    /// Its upright pixel size.
    @Published private(set) var inputSize: PixelSize?
    @Published private(set) var cutouts: [Cutout] = []
    /// The cut-out on show: the current image's, or one picked from the strip.
    @Published private(set) var shown: UUID?
    /// An image is dragged over the window's drop places (the well and the result light up).
    @Published var dropTargeted = false
    /// The result shows the model's mask instead of the cut-out (Cut-out | Mask; only when the run returned a mask).
    @Published var showsMask = false
    /// The image the run in progress cuts out.
    private var runOriginal: URL?

    /// The panels (tests answer them).
    var chooseImage: () -> URL? = {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }
    /// Where Copy writes (tests use their own pasteboard).
    var copyPasteboard: NSPasteboard = .general

    static var picturesDirectory: URL { ToolFiles.picturesDirectory }

    override init(
        models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL = BackgroundSession.picturesDirectory,
        scratch: URL? = nil
    ) {
        super.init(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }

    /// Images by every model of the tool; video only through a model's own `track_video` (it returns mask videos).
    override func offers(_ op: OperationDescriptor) -> Bool { Self.takesImage(op) || op.name == Self.video }
    override var preferredOperations: [String] { [Self.operation] }
    /// The video operation a model of this window may declare (perception/ops/track_video).
    static let video = "track_video"
    var videoMode: Bool { operation == Self.video }
    /// Image | Video when the model takes both; otherwise Cut-out | Alpha Matte (the operations), never the result pane's
    /// Cut-out | Mask (the views of one run) word for word, else the operation's own title.
    override func pillTitle(_ op: OperationDescriptor) -> String {
        if operations.contains(where: { $0.name == Self.video }) {
            return op.name == Self.video ? "Video" : op.name == Self.operation ? "Image" : super.pillTitle(op)
        }
        return Self.pillName(op) ?? super.pillTitle(op)
    }
    /// "Cut-out" for the tool's own operation, "Alpha Matte" for an alpha matte.
    static func pillName(_ op: OperationDescriptor) -> String? {
        if op.name == operation { return "Cut-out" }
        if op.name.lowercased().contains("matte") || op.title.lowercased().contains("matte") { return "Alpha Matte" }
        return nil
    }
    override func runsAsJob(_ op: OperationDescriptor) -> Bool { op.name == Self.video }
    /// A video's words are what to keep, an option beside the points, not a main text the run waits for.
    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        op.name == Self.video ? ToolForm(schema: op.inputSchema, operation: op.name, output: op.output, main: [:], titles: ["text": "Describe"]) : super.makeForm(op)
    }
    override var idleLine: String { videoMode ? "Tracking" : "Removing background" }
    override var resultNoun: String { "cut" }
    override var taskTitle: String { "Remove Background" }
    /// The cut's tuning (working size, mask cut-off, edge refinement) is expert: under Advanced with the samplers.
    override func advancedKeys(_ op: String?) -> Set<String> { super.advancedKeys(op).union(["resolution", "threshold", "refine_edges", "refine_foreground"]) }
    override var runVerb: String { BackgroundView.runTitle(descriptor) }
    override var notReadyReason: String { input == nil ? (videoMode ? "Add a video first" : "Add an image first") : "Click what to keep, or describe it" }
    override func noOperations(_ name: String) -> String { "\(name) declares no operation that takes an image." }

    /// The key of the image (or video) the well holds.
    var imageKey: String? { videoMode ? "video" : descriptor.flatMap(Self.imageField)?.key }
    /// Every other input the operation declares, in the kit's order (a video's points and boxes are placed on its frame;
    /// its session and frame updates are not typed).
    override var optionFields: [ToolField] {
        form?.fields.filter { $0.key != imageKey && !$0.isMain && !$0.isSpatial && !["session_id", "prompt_updates"].contains($0.key) } ?? []
    }
    var current: Cutout? { cutouts.first { $0.id == shown } }
    /// A video needs to be told what to keep: a point, a box or words.
    override var readyToRun: Bool {
        guard input != nil else { return false }
        guard videoMode else { return true }
        return ["points", "boxes", "text"].contains { !value($0).text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }
    override var shownFile: URL? { (current ?? cutouts.last).map(shownFile) }
    override func requestValues() -> [String: FieldValue] {
        var all = values
        if let imageKey, let input { all[imageKey] = .files([input]) }
        return all
    }
    /// The image's own problem is the window's line, not a field's.
    override func fieldProblemKey(_ key: String) -> String? { key == imageKey ? nil : key }

    /// An operation the window can run: it takes one image and answers with files.
    static func takesImage(_ op: OperationDescriptor) -> Bool { op.returnsFiles && imageField(op) != nil }

    /// The operation's image input: its file field and what it accepts.
    static func imageField(_ op: OperationDescriptor) -> (key: String, accept: FileAccept)? {
        for field in OperationForm(schema: op.inputSchema).fields {
            if case .file(let accept, false, _, _) = field.control, accept.images || accept.mimes.isEmpty { return (field.key, accept) }
        }
        return nil
    }

    // MARK: The image

    /// Images dropped or chosen: the first readable image becomes the one to cut out, and the cut starts when the model
    /// can run. Nothing changes while a cut runs.
    func setInput(_ urls: [URL]) {
        guard !running else { return }
        let video = FileAccept(mimes: ["video/*"])
        if let movie = urls.first(where: video.accepts) {
            // A video switches to the model's video operation, when it has one.
            guard operations.contains(where: { $0.name == Self.video }) else {
                problem = .init(message: "\(model?.name ?? "This model") removes backgrounds from images only.", action: nil); return
            }
            if !videoMode { select(Self.video) }
            input = movie; inputSize = VideoInfo.cachedSize(movie); shown = nil; problem = nil
            for key in ["points", "boxes"] { values[key] = .text("") }
            Task {
                _ = await VideoInfo.load(movie); if input == movie { inputSize = VideoInfo.cachedSize(movie) }
            }
            return
        }
        if videoMode, let op = operations.first(where: { Self.takesImage($0) }) { select(op.name) }
        guard let url = urls.first(where: ToolFiles.isImageFile) ?? urls.first else { return }
        guard ToolFiles.isImageFile(url), let size = PixelSize.read(url) else {
            problem = .init(message: "\(url.lastPathComponent) is not an image Arsenal can read. Use a PNG, JPEG, HEIC, TIFF or WebP.", action: nil)
            return
        }
        input = url; inputSize = size; shown = nil; problem = nil
        if canRun { start() }
    }

    /// Image bytes dragged in or pasted without a file: kept in the window's folder as a PNG, then used as a file.
    func addImage(_ data: Data, name: String? = nil) {
        guard !running else { return }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) > 0,
            let url = try? write(source, name: name ?? "Pasted image")
        else {
            problem = .init(message: "That is not an image Arsenal can read.", action: nil); return
        }
        setInput([url])
    }

    /// ⌘V: an image file copied in Finder, else image data (a copied picture). False when the pasteboard holds neither.
    @discardableResult func paste(_ pasteboard: NSPasteboard) -> Bool {
        guard !running else { return false }
        let files = (pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
        if let file = files.first(where: ToolFiles.isImageFile) { setInput([file]); return true }
        for type in [NSPasteboard.PasteboardType.png, .tiff, NSPasteboard.PasteboardType(UTType.jpeg.identifier), NSPasteboard.PasteboardType(UTType.heic.identifier)] {
            if let data = pasteboard.data(forType: type) { addImage(data); return true }
        }
        return false
    }

    func pick() {
        guard !running, let url = chooseImage() else { return }
        setInput([url])
    }

    /// × on the well: no image; the cut-outs stay in the strip.
    func clear() {
        guard !running else { return }
        input = nil; inputSize = nil; shown = nil
        if problem?.action == nil { problem = nil }
    }

    /// A cut-out from the strip: it is shown, and (unless a cut is running) its image becomes the one in the well.
    func show(_ id: UUID) {
        guard let cutout = cutouts.first(where: { $0.id == id }) else { return }
        shown = id
        if !running, input != cutout.original { input = cutout.original; inputSize = PixelSize.read(cutout.original) }
    }

    /// Bytes without a file: the window's folder, as an upright PNG named after where they came from.
    private func write(_ source: CGImageSource, name: String) throws -> URL {
        let folder = scratch.appendingPathComponent("inputs", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let stem = ToolFiles.safe(URL(fileURLWithPath: name).deletingPathExtension().lastPathComponent)
        var url = folder.appendingPathComponent((stem.isEmpty ? "Pasted image" : stem) + ".png")
        var n = 2
        while FileManager.default.fileExists(atPath: url.path) { url = folder.appendingPathComponent("\(stem.isEmpty ? "Pasted image" : stem) \(n).png"); n += 1 }
        try ToolFiles.upright(source, to: url)
        return url
    }

    // MARK: Running

    override func start() {
        let original = input
        super.start()
        if running { runOriginal = original; shown = nil }
    }

    /// The cut-out saved to ~/Pictures/Arsenal as `<date> <time> <model> <image name>.png`, its mask beside it.
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        if op.name == Self.video {
            guard let original = runOriginal, let masks = result.files.first(where: { $0.mime.hasPrefix("video/") }) else {
                throw Undelivered(.init(message: "\(model.name) answered without a mask video.", action: nil))
            }
            let url = try ToolFiles.save(
                masks.url, into: ToolFiles.override ?? outputDirectory, model: model.id, tag: original.deletingPathExtension().lastPathComponent + " mask", what: "mask video")
            cutouts.append(
                Cutout(
                    original: original, url: url, mask: nil, model: model.id, modelName: model.name, width: inputSize?.width ?? 0, height: inputSize?.height ?? 0,
                    seconds: result.seconds, operation: op.name, values: values))
            if self.input == original { shown = cutouts.last?.id }
            return
        }
        let images = result.files.filter(\.isImage)
        guard let original = runOriginal, let foreground = images.first(where: { $0.id == "foreground" }) ?? images.first else {
            throw Undelivered(.init(message: "\(model.name) answered without an image.", action: nil))
        }
        let url = try ToolFiles.save(foreground.url, into: outputDirectory, model: model.id, tag: original.deletingPathExtension().lastPathComponent, ext: "png", what: "cut-out")
        let mask = try images.first { $0.id == "mask" && $0.id != foreground.id }.map { try save($0.url, beside: url) }
        let size = PixelSize.read(url)
        cutouts.append(
            Cutout(
                original: original, url: url, mask: mask, model: model.id, modelName: model.name, width: size?.width ?? foreground.width ?? 0,
                height: size?.height ?? foreground.height ?? 0, seconds: result.seconds, operation: op.name, values: values))
        if self.input == original { shown = cutouts.last?.id }
    }

    /// The mask beside its cut-out: `<cut-out name> mask.png`.
    private func save(_ fetched: URL, beside cutout: URL) throws -> URL {
        let target = cutout.deletingLastPathComponent().appendingPathComponent(cutout.deletingPathExtension().lastPathComponent + " mask.png")
        do {
            if FileManager.default.fileExists(atPath: target.path) { try FileManager.default.removeItem(at: target) }
            try FileManager.default.moveItem(at: fetched, to: target)
            return target
        } catch {
            throw WordedError(line: "Couldn\u{2019}t save the mask in \(outputDirectory.lastPathComponent).", raw: error.localizedDescription)
        }
    }

    // MARK: The result

    /// The cut-out or its mask, whichever the result shows (the mask only when the run returned one).
    func shownFile(_ cutout: Cutout) -> URL { showsMask ? cutout.mask ?? cutout.url : cutout.url }

    /// Copy what is shown: the PNG (transparency kept), a TIFF for apps that only take that, and the saved file for Finder.
    func copy(_ cutout: Cutout) { copy(shownFile(cutout), to: copyPasteboard) }

    /// The shown cut-out was made from the image in the well, by this model and operation with these inputs: running
    /// again would make the same thing, so Remove Background steps back from the primary blue.
    override var isFresh: Bool {
        guard !running, let c = current, let input else { return false }
        return c.original == input && c.model == selected && c.operation == operation && c.values == values
    }

    /// Render harness and tests: a state without the API.
    func preview(
        operations: [OperationDescriptor], selected op: String? = nil, values: [String: FieldValue] = [:], input: URL? = nil, run: ToolRun? = nil,
        cutouts: [Cutout] = [], shown: UUID? = nil, problem: OperationProblem? = nil
    ) {
        previewOptions(operations: operations, selected: op, values: values)
        self.input = input; inputSize = input.flatMap(PixelSize.read)
        self.run = run; self.cutouts = cutouts; self.shown = shown; self.problem = problem
    }
}
