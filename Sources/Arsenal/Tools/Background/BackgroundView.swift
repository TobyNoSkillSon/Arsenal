// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI
import UniformTypeIdentifiers

// Remove Background, composed from the tool kit: the band (the model's pop-up of every model that removes backgrounds,
// its image operations as pills, Show in Finder), the live figures, then the image on the left (dropped, chosen or
// pasted), any other option the operation declares under it, and Remove Background or Stop at their foot; on the right
// the cut-out over a checkerboard with Copy and Show in Finder, this window's earlier cut-outs in the strip under it.
// The image stays on the left, so the two read side by side.

struct BackgroundView: View {
    @ObservedObject var session: BackgroundSession
    @Environment(\.palette) private var p
    static let size = ToolLayout.size, minSize = ToolLayout.minSize
    static let wellHeight: CGFloat = 260
    static let emptyHelp = "PNG, JPEG, HEIC, TIFF or WebP; \u{2318}V pastes one"
    static let chosenHelp = "Drop or click to choose another image"

    var body: some View {
        ToolFrame(figures: session.figures(), model: session.model?.model) {
            SessionBand(session: session)
        } controls: {
            // Empty, the well takes the column (a large drop place); with an image it keeps the image's own height in every
            // state, so a failure's line below it never squeezes it.
            OptionsColumn(
                session: session, run: RunButton(session: session, title: Self.runTitle(session.descriptor), id: "remove"), leadFills: session.input == nil
            ) {
                image
            }
        } result: {
            BackgroundResult(session: session)
        }
    }

    /// The image to cut out: the kit's file well, which is also the button to choose one; with nothing else to set
    /// (BiRefNet) it takes the column, larger beside its cut-out.
    @ViewBuilder private var image: some View {
        if session.videoMode, let video = session.input {
            BackgroundVideoLead(session: session, video: video)
        } else {
            still
        }
    }

    /// Accepts videos too when the model tracks a selection through one.
    private var accept: FileAccept {
        FileAccept(mimes: session.operations.contains { $0.name == BackgroundSession.video } ? ["image/*", "video/*"] : ["image/*"])
    }

    @ViewBuilder private var still: some View {
        // Titled as the operation's own file fields are.
        Text(session.form?.fields.first { $0.key == session.imageKey }?.title ?? "Image").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).lineLimit(1)
            .padding(.bottom, -8)
        FileWell(
            key: session.imageKey ?? "image", title: "Image", files: session.input.map { [$0] } ?? [], accept: accept,
            help: accept.mimes.count > 1 ? Self.emptyHelp + "; a video selects through it" : Self.emptyHelp,
            chosenHelp: Self.chosenHelp, targeted: $session.dropTargeted, player: session.player, enabled: !session.running,
            height: session.optionFields.isEmpty ? nil : Self.wellHeight, imageSize: session.inputSize,
            add: { session.setInput($0) }, remove: { _ in session.clear() }, choose: { session.chooseImage().map { [$0] } ?? [] },
            receive: { Self.receive($0, session: session) }
        )
        .modifier(WellAspect(ratio: Self.wellAspect(session.optionFields.isEmpty ? session.inputSize : nil)))
        if let input = session.input {
            HStack(spacing: 8) {
                Text(input.lastPathComponent).font(.system(size: 12.5)).lineLimit(1).truncationMode(.middle).windowHelp(input.path)
                Spacer(minLength: 4)
                if let size = session.inputSize {
                    Text(size.title).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t3.swiftUI).lineLimit(1).fixedSize()
                }
            }
        }
    }

    /// The well's shape around a chosen image (its picture inset 8 on each side); nil: the well's own height.
    static func wellAspect(_ size: PixelSize?) -> CGFloat? {
        guard let size, size.width > 0, size.height > 0 else { return nil }
        return min(3, max(0.5, CGFloat(size.width) / CGFloat(size.height)))
    }

    /// Files or image data dropped on the well or the result: the first file, else the first image's bytes.
    @discardableResult static func receive(_ providers: [NSItemProvider], session: BackgroundSession) -> Bool {
        guard !session.running else { return false }
        if let provider = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in session.setInput([url]) }
            }
            return true
        }
        if let provider = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.image.identifier) }) {
            let name = provider.suggestedName
            _ = provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data else { return }
                Task { @MainActor in session.addImage(data, name: name) }
            }
            return true
        }
        return false
    }
    static let dropTypes: [UTType] = [.fileURL, .image]
    /// "Remove Background" for the tool's own operation, else the operation's title ("Alpha Matte").
    static func runTitle(_ op: OperationDescriptor?) -> String {
        guard let op, op.name != BackgroundSession.operation else { return "Remove Background" }
        // A video's selection runs as Select Objects names it.
        return op.name == BackgroundSession.video ? "Track Mask" : op.title
    }
}

/// The shown cut-out over a checkerboard, its actions above it; this window's cut-outs in the strip under the panel.
struct BackgroundResult: View {
    @ObservedObject var session: BackgroundSession
    @Environment(\.palette) private var p

    var body: some View {
        VStack(spacing: 12) {
            ResultPanel(session: session, symbol: ToolKind.background.symbol, empty: session.current == nil, highlighted: session.dropTargeted) {
                VStack(spacing: 0) {
                    if let cutout = session.current {
                        ResultActions(
                            what: cutout.isVideo ? "mask video" : session.showsMask && cutout.mask != nil ? "matte" : "cut-out", enabled: !session.running,
                            copy: { session.copy(cutout) }, leading: { views(cutout) }, save: { session.saveAs(session.shownFile(cutout)) })
                    }
                    dropZone
                }
            }
            ImageStrip(
                items: session.cutouts.map { .init(id: $0.id, url: $0.url, label: "Cut-out of \($0.original.lastPathComponent)", controlID: "cutout-\($0.id)") },
                current: session.current?.id, select: { session.show($0) }
            ).historySlot(!session.cutouts.isEmpty)
        }
    }

    /// Cut-out | Mask when the run returned a mask (the same run, no new one), as Select Objects names them.
    @ViewBuilder private func views(_ cutout: Cutout) -> some View {
        if cutout.isVideo {
            // The model's mask video, named as what it is (white where the selection is), never as a cut-out.
            Text("Mask").font(.system(size: 12, weight: .medium)).foregroundStyle(p.t2.swiftUI)
        } else if cutout.mask != nil {
            Pill(title: "Cut-out", on: !session.showsMask, enabled: !session.running, help: "The image with its background removed") {
                session.showsMask = false
            }.controlID("view-cutout")
            Pill(title: "Mask", on: session.showsMask, enabled: !session.running, help: "What the model kept, in white") { session.showsMask = true }
                .controlID("view-mask")
        }
    }

    /// The cut-out (or the empty panel): a drop place too, outside the action row so its hovers stay its own.
    private var dropZone: some View {
        ZStack {
            if let cutout = session.current, cutout.isVideo {
                // The model's mask video: white where the selection is. Not an alpha matte.
                VideoResult(url: cutout.url).padding(.horizontal, 16).padding(.bottom, 16).accessibilityLabel("Mask video of \(cutout.original.lastPathComponent)")
            } else if let cutout = session.current {
                FittedImage(
                    url: session.shownFile(cutout), size: PixelSize(width: cutout.width, height: cutout.height), draggable: true,
                    label: (session.showsMask && cutout.mask != nil ? "Mask of " : "Cut-out of ") + cutout.original.lastPathComponent
                ).padding(.horizontal, 16).padding(.bottom, 16)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onDrop(of: BackgroundView.dropTypes, isTargeted: $session.dropTargeted) { BackgroundView.receive($0, session: session) }
    }
}

/// A video in Remove Background: its first frame with the points and boxes that say what to keep, the operation's
/// own words field among the options; × clears it, a drop replaces it.
struct BackgroundVideoLead: View {
    @ObservedObject var session: BackgroundSession
    let video: URL
    @BackgroundFrame private var frame: NSImage?
    @BackgroundMode private var mode: PromptMode = .include
    @Environment(\.palette) private var p

    var body: some View {
        let points = session.form?.fields.contains { $0.key == "points" && $0.isSpatial } ?? false
        let boxes = session.form?.fields.contains { $0.key == "boxes" && $0.isSpatial } ?? false
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 2) {
                Text("Video").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI)
                Spacer(minLength: 4)
                ForEach(([points ? PromptMode.include : nil, points ? .exclude : nil, boxes ? .box : nil] as [PromptMode?]).compactMap { $0 }) { m in
                    Button {
                        mode = m
                    } label: {
                        Image(systemName: m.symbol).font(.system(size: 12)).frame(width: 24, height: 22)
                            .background((mode == m ? p.accent.withAlphaComponent(0.2) : .clear).swiftUI, in: RoundedRectangle(cornerRadius: 5)).contentShape(Rectangle())
                    }.buttonStyle(.plain).foregroundStyle((mode == m ? p.accentText : p.t2).swiftUI).windowHelp(m.help).accessibilityLabel(m.help).controlID("prompt-\(m.rawValue)")
                }
            }
            ZStack(alignment: .topTrailing) {
                PromptCanvas(
                    image: frame ?? VideoInfo.cachedFrame(video),
                    pixels: session.inputSize.map { CGSize(width: $0.width, height: $0.height) } ?? frame?.size ?? CGSize(width: 16, height: 9),
                    points: Binding(get: { SpatialValue.points(session.value("points").text) }, set: { session.set("points", .text(SpatialValue.text($0))) }),
                    boxes: Binding(get: { SpatialValue.boxes(session.value("boxes").text) }, set: { session.set("boxes", .text(SpatialValue.text($0))) }),
                    takesPoints: points, takesBoxes: boxes, mode: mode, enabled: !session.running
                ) { _ in EmptyView() }
                .padding(8).controlID("canvas")
                if !session.running {
                    Button {
                        session.clear()
                    } label: {
                        Image(systemName: "xmark.circle.fill").font(.system(size: 14)).symbolRenderingMode(.palette)
                            .foregroundStyle(Color.white, Color.black.opacity(0.6)).frame(width: 26, height: 26).contentShape(Rectangle())
                    }.buttonStyle(.plain).windowHelp("Remove").accessibilityLabel("Remove \(video.lastPathComponent)").controlID("remove-video")
                }
            }
            .frame(maxHeight: .infinity)
            .background(p.sidebar.swiftUI, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder((session.dropTargeted ? p.accent : p.raisedBorder).swiftUI, lineWidth: session.dropTargeted ? 2 : 1))
            .onDrop(of: BackgroundView.dropTypes, isTargeted: $session.dropTargeted) { BackgroundView.receive($0, session: session) }
            SubjectCaption(url: video)
        }
        .task(id: video) { frame = await VideoInfo.load(video) }
    }
}

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias BackgroundFrame = SwiftUI.State<NSImage?>
private typealias BackgroundMode = SwiftUI.State<PromptMode>

/// Fits the well to its image's shape when it has one; otherwise leaves its size alone.
private struct WellAspect: ViewModifier {
    let ratio: CGFloat?
    func body(content: Content) -> some View {
        if let ratio { content.aspectRatio(ratio, contentMode: .fit) } else { content }
    }
}
