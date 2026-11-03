// SPDX-License-Identifier: MIT
import AppKit
import CoreImage
import ArsenalCore
import SwiftUI

// Select Objects: one window for every model that declares `segment` or `track_video`. A picture (or a video) with
// points to include or leave out and boxes placed on it, and a description when the model takes text; the selection as
// a tinted mask over the picture, the cut-out it makes, or the mask itself; several masks with their scores when the
// model returns several. Once a selection is shown, each new point or box selects again at once.
//
// A model that declares `segment_session` keeps the picture's encoding between clicks: the window creates the session
// with the first selection and releases it when the picture, the model or the window changes. A video goes to
// `track_video` as a job (progress, Stop cancels it); prompts placed on a later frame are sent as that frame's prompt
// updates, and a model that declares `video_session` keeps its tracking state for the video the same way.

@MainActor final class SelectSession: CapabilitySession {
    /// The picture's or video's session on the loaded model, when the model keeps one.
    private(set) var sessionID: String?
    private var sessionSubject: URL?
    /// The staged copy of a video whose session is open: the model reads its frames from this path on every
    /// `track_video`, so it stays until the session is released.
    private(set) var sessionFolder: URL?
    /// The mask on show among several (an index into the result's masks).
    @Published var maskIndex = 0
    /// Overlay (the mask tinted over the picture), the cut-out, or the mask alone.
    @Published var view: Shown = .overlay
    enum Shown: String, CaseIterable { case overlay = "Selection", cutout = "Cut-out", mask = "Mask" }
    /// The video frame prompts are placed on.
    @Published var frame = 0
    /// A selection asked for while one is running: it runs when that one ends.
    private var pending = false
    private var debounce: Task<Void, Never>?
    private var retried = false

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .select, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }

    override var managedKeys: Set<String> { ["session_id", "prompt_updates"] }
    override func pillTitle(_ op: OperationDescriptor) -> String { op.name == "track_video" ? "Video" : op.name == "segment" ? "Image" : super.pillTitle(op) }
    /// The history shows the pictures selections were made in; Copy takes what is shown (cut-out or mask).
    /// The history shows each selection as its cut-out (the pictures alone would all look the same).
    override func historyPicture(_ o: Outcome) -> URL? {
        guard let picture = o.inputs.first(where: ToolFiles.isImageFile) else { return o.inputs.first }
        guard o.operation == "segment", let mask = o.files.first(where: \.isImage) else { return picture }
        let best = Self.bestMask(o)
        let masks = o.files.filter(\.isImage)
        return cutout(picture, mask: masks.indices.contains(best) ? masks[best].url : mask.url) ?? picture
    }
    /// Copy keeps the selection on show, then takes what is shown (cut-out or mask).
    override func copyFile(_ o: Outcome) -> URL? {
        if o.id == draft?.id { keepDraft() }
        return o.id == current?.id || draft == nil ? shownResultFile() : o.files.first?.url
    }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["text": "Describe", "multimask": "Show alternatives"] }
    override var resultNoun: String { "selection" }
    override var idleLine: String { descriptor?.name == "track_video" ? "Tracking" : "Selecting" }
    override func runTitle(_ op: OperationDescriptor?) -> String { op?.name == "track_video" ? "Track" : "Select" }
    override func managedFilled(_ key: String) -> Bool { key == "session_id" && sessionID != nil }
    /// A picture or video and something to select: a point, a box or a description.
    override var readyToRun: Bool {
        guard super.readyToRun || !subject.isEmpty, !subject.isEmpty else { return false }
        return hasPrompts || filled("text")
    }
    override var promptTime: Double {
        guard let video = subject.first, let rate = VideoInfo.cachedRate(video) else { return 0 }
        return Double(frame) / rate
    }
    var declaresSession: Bool { declared.contains(descriptor?.name == "track_video" ? "video_session" : "segment_session") }

    // MARK: Sessions

    override func didSetSubject() {
        super.didSetSubject()
        frame = 0; maskIndex = 0
        release()
    }
    override func choose(_ id: String) {
        if id != selected { release() }
        super.choose(id)
    }
    override func select(_ name: String?, carry text: String? = nil) {
        if name != operation { release() }
        super.select(name, carry: text)
    }
    override func close() {
        debounce?.cancel()
        release()
        super.close()
    }

    /// Closes the model's session for the old picture (it is scoped to the loaded model; a stale one is refused there).
    func release() {
        let folder = sessionFolder
        sessionFolder = nil
        guard let id = sessionID, let client = client() else {
            sessionID = nil; sessionSubject = nil
            if let folder { try? FileManager.default.removeItem(at: folder) }
            return
        }
        let op = sessionSubject.map { FileAccept(mimes: ["video/*"]).accepts($0) } == true ? "video_session" : "segment_session"
        let model = selected
        sessionID = nil; sessionSubject = nil
        Task {
            _ = try? await client.send("POST", "/arsenal/v1/models/\(model)/ops/\(op)", json: ["action": "close", "session_id": id], timeout: 30)
            // Removed after the close, so a model still reading the video is not cut off.
            if let folder { try? FileManager.default.removeItem(at: folder) }
        }
    }

    /// Creates the session for the subject before the first selection, when the model keeps one.
    override func start() {
        // Select with the clicked selection on show and nothing changed since: keep it rather than run again.
        if !quietRun, let draft, draft.values == requestValues(), draft.model == selected, draft.operation == operation {
            keepDraft(); return
        }
        guard canRun, declaresSession, sessionID == nil, let subject = subject.first, let client = client(), let model else { super.start(); return }
        let video = FileAccept(mimes: ["video/*"]).accepts(subject)
        let op = video ? "video_session" : "segment_session"
        let accept = video ? FileAccept(mimes: ["video/*"]) : FileAccept(mimes: ["image/png", "image/jpeg"])
        run = ToolRun(model: model.id, operation: op)
        Task {
            var folder: URL?, kept = false
            defer { if let folder, !kept { try? FileManager.default.removeItem(at: folder) } }
            do {
                let (staged, path) = try ToolFiles.stage(subject, accept: accept, inbox: inbox)
                folder = staged
                let (data, _) = try await client.send(
                    "POST", "/arsenal/v1/models/\(model.id)/ops/\(op)", json: ["action": "create", (video ? "video" : "image"): ["path": path]], timeout: 600,
                    accept: "application/json")
                guard let id = try JSONDecoder().decode(JSONValue.self, from: data)["session_id"]?.string else { throw ToolError("\(model.name) did not open a session.") }
                guard self.subject.first == subject else {
                    _ = try? await client.send("POST", "/arsenal/v1/models/\(model.id)/ops/\(op)", json: ["action": "close", "session_id": id], timeout: 30)
                    return
                }
                sessionID = id; sessionSubject = subject
                // An image session holds the picture's encoding; a video session reads the staged file on each track.
                if video {
                    if let old = sessionFolder { try? FileManager.default.removeItem(at: old) }
                    sessionFolder = folder; kept = true
                }
                run = nil
                super.start()
            } catch {
                finish(OperationProblem.from(error, model: model.name, task: stoppedNoun))
            }
        }
    }

    /// The request as Select sends it: the session instead of the picture once there is one; prompts placed on a later
    /// video frame as that frame's prompt updates.
    override func prepareInput(_ input: JSONObject, op: OperationDescriptor) -> JSONObject {
        var input = input
        if let sessionID {
            input["session_id"] = .string(sessionID)
            input[op.name == "track_video" ? "video" : "image"] = nil
        }
        if op.name == "track_video", frame > 0, op.inputSchema["properties"]?["prompt_updates"] != nil {
            var update: JSONObject = ["frame_index": .number(Double(frame)), "object_id": 1, "action": "add"]
            for key in ["points", "boxes", "text"] where input[key] != nil { update[key] = input[key]; input[key] = nil }
            input["prompt_updates"] = [.object(update)]
        }
        return input
    }

    // MARK: Interaction

    /// Once a selection is shown (the model is loaded), each change to the prompts selects again.
    override func promptsChanged() {
        guard descriptor?.name == "segment", current != nil else { return }
        debounce?.cancel()
        debounce = Task {
            try? await Task.sleep(for: .milliseconds(180))
            guard !Task.isCancelled else { return }
            if running { pending = true } else { startQuiet() }
        }
    }
    override func didFinish() {
        // A session the model no longer knows (it was unloaded): the next selection opens a new one, once.
        if problem != nil, sessionID != nil, !retried {
            retried = true; sessionID = nil; problem = nil
            if let folder = sessionFolder { try? FileManager.default.removeItem(at: folder) }
            sessionFolder = nil
            start(); return
        }
        retried = false
        if pending { pending = false; startQuiet() }
    }
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        try super.deliver(result, op: op, model: model, input: input)
        maskIndex = Self.bestMask(current)
    }
    override func check(_ outcome: Outcome) throws {
        guard !outcome.files.isEmpty else { throw Undelivered(.init(message: "\(outcome.modelName) selected nothing.", action: nil)) }
    }

    // MARK: Result

    var masks: [ResultFile] { current?.files.filter(\.isImage) ?? [] }
    var shownMask: ResultFile? { masks.indices.contains(maskIndex) ? masks[maskIndex] : masks.first }
    override var canvasMasks: [URL] {
        guard let current, current.inputs == subject, let mask = shownMask, current.operation == "segment" else { return [] }
        return [mask.url]
    }
    override var shownFile: URL? { shownMask?.url ?? current?.files.first?.url }
    static func bestMask(_ o: Outcome?) -> Int {
        let scores = o?.reply["scores"]?.array?.compactMap(\.number) ?? []
        return scores.enumerated().max { $0.element < $1.element }?.offset ?? 0
    }
    func score(_ index: Int) -> Double? {
        let scores = current?.reply["scores"]?.array?.compactMap(\.number) ?? []
        return scores.indices.contains(index) ? scores[index] : nil
    }

    /// The picture with only the selection kept (its mask as the alpha), made once per mask.
    func cutout(_ picture: URL, mask: URL) -> URL? {
        let url = scratch.appendingPathComponent("cutouts", isDirectory: true).appendingPathComponent(mask.deletingPathExtension().lastPathComponent + " cut-out.png")
        if FileManager.default.fileExists(atPath: url.path) { return url }
        guard let image = CIImage(contentsOf: picture, options: [.applyOrientationProperty: true]), var alpha = CIImage(contentsOf: mask) else { return nil }
        alpha = alpha.transformed(by: CGAffineTransform(scaleX: image.extent.width / max(1, alpha.extent.width), y: image.extent.height / max(1, alpha.extent.height)))
        let blend = image.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: CIImage.empty(), kCIInputMaskImageKey: alpha])
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try CIContext().writePNGRepresentation(of: blend.cropped(to: image.extent), to: url, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
            return url
        } catch { return nil }
    }
    /// What the result shows now as a file (Copy and drag): the cut-out, the mask, or the mask over nothing.
    func shownResultFile() -> URL? {
        guard let current, let mask = shownMask else { return current?.files.first?.url }
        if view == .cutout, let picture = current.inputs.first, ToolFiles.isImageFile(picture) { return cutout(picture, mask: mask.url) ?? mask.url }
        return mask.url
    }

    /// Tests: a session as the model would have opened it.
    func preview(session id: String?, for subject: URL?) { sessionID = id; sessionSubject = subject }
}

struct SelectView: View {
    @ObservedObject var session: SelectSession
    @Environment(\.palette) private var p

    var body: some View {
        CapabilityView(session: session, below: AnyView(FrameSlider(session: session))) {
            OutcomePanel(
                session: session, what: session.view == .cutout ? "cut-out" : "mask", copyText: nil, actions: { o in actions(o) },
                content: { o in
                    content(o).padding(.horizontal, 16).padding(.bottom, 16)
                })
        }
    }

    @ViewBuilder private func actions(_ o: Outcome) -> some View {
        if o.operation == "segment" {
            ForEach(SelectSession.Shown.allCases, id: \.self) { v in
                Pill(title: v.rawValue, on: session.view == v, enabled: !session.running, help: v.rawValue) { session.view = v }.controlID("view-\(v.rawValue)")
            }
            if session.masks.count > 1 {
                Spacer(minLength: 6)
                ForEach(session.masks.indices, id: \.self) { i in
                    Pill(title: session.score(i).map { String(format: "%.2f", $0) } ?? "\(i + 1)", on: session.maskIndex == i, enabled: !session.running, help: "Mask \(i + 1)") {
                        session.maskIndex = i
                    }.controlID("mask-\(i)")
                }
            }
        }
    }

    @ViewBuilder private func content(_ o: Outcome) -> some View {
        if o.operation == "track_video" {
            if let video = o.file("video/") {
                VideoResult(url: video.url)
            } else {
                ImageStrip(
                    items: o.files.filter(\.isImage).map { .init(id: UUID(), url: $0.url, label: $0.label, controlID: "frame-\($0.id)") }, current: nil, select: { _ in })
            }
        } else if let picture = o.inputs.first, let mask = session.shownMask {
            switch session.view {
            case .overlay:
                let pixels = PixelSize.read(picture).map { CGSize(width: $0.width, height: $0.height) } ?? CGSize(width: 1, height: 1)
                GeometryReader { g in
                    let box = Letterbox(container: g.size, pixels: pixels)
                    ZStack(alignment: .topLeading) {
                        Thumbnail(url: picture, fit: true).frame(width: box.rect.width, height: box.rect.height).offset(x: box.rect.minX, y: box.rect.minY)
                        MaskOverlay(mask: mask.url, box: box, colour: SpatialColours.colour(session.maskIndex))
                    }
                }
                .onDrag {
                    // Dragging the selection out keeps it.
                    session.keepDraft()
                    return NSItemProvider(contentsOf: session.shownMask?.url ?? mask.url) ?? NSItemProvider()
                }
            case .cutout:
                if let cut = session.cutout(picture, mask: mask.url) {
                    FittedImage(
                        url: cut, size: PixelSize.read(picture), draggable: true, label: "Cut-out of \(picture.lastPathComponent)",
                        willDrag: {
                            session.keepDraft(); return session.shownResultFile()
                        })
                }
            case .mask:
                FittedImage(
                    url: mask.url, size: PixelSize.read(mask.url), draggable: true, label: "Mask of \(picture.lastPathComponent)",
                    willDrag: {
                        session.keepDraft(); return session.shownResultFile()
                    })
            }
        }
    }
}

/// The frame of a video prompts are placed on, under the video in the well (only for a video).
struct FrameSlider: View {
    @ObservedObject var session: SelectSession
    @Environment(\.palette) private var p
    var body: some View {
        if let video = session.subject.first, FileAccept(mimes: ["video/*"]).accepts(video), let length = VideoInfo.cachedLength(video), let rate = VideoInfo.cachedRate(video) {
            let frames = max(1, Int(length * rate) - 1)
            HStack(spacing: 8) {
                ToolSlider(value: Binding(get: { Double(session.frame) }, set: { session.frame = Int($0.rounded()) }), range: 0...Double(frames), title: "Frame")
                Text("Frame \(session.frame)").font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t2.swiftUI).frame(width: 78, alignment: .trailing)
            }
            .disabled(session.running)
            .windowHelp("The frame the points and boxes are placed on")
        }
    }
}
