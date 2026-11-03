// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI
import UniformTypeIdentifiers

// A capability window, composed from the tool kit as Make an Image and Remove Background are: the band (the model's
// pop-up of every model that declares the window's operations, its operations as pills, Show in Finder), the live
// figures, then on the left what the run works on (one well for the operation's files, with points and boxes placed on
// the picture when the operation takes them), the operation's other options and the run button; on the right the
// window's own result view, its earlier results under it.

struct CapabilityView<Result: View>: View {
    @ObservedObject var session: CapabilitySession
    /// The subject well takes the column when nothing else is set (the picture beside its result).
    var subjectFills = true
    /// Under the well (Select's frame slider).
    var below: AnyView?
    /// The well's height when the options leave room (nil: the kit's).
    var subjectHeight: CGFloat?
    @ViewBuilder let result: Result
    static var size: NSSize { ToolLayout.size }
    static var minSize: NSSize { ToolLayout.minSize }

    var body: some View {
        ToolFrame(figures: session.figures(), model: session.model?.model) {
            SessionBand(session: session)
        } controls: {
            // The well fills the column only when nothing else is set; otherwise it keeps its height and the run button
            // stays at the column's foot (a window whose only options are a chooser, as Smooth Video's, included).
            let fills = subjectFills && !session.subjectFields.isEmpty && session.optionFields.isEmpty && session.optionGroups.isEmpty
            OptionsColumn(
                session: session, run: RunButton(session: session, title: session.runVerb, id: "run"),
                leadFills: fills, groups: AnyView(OptionGroupsView(session: session, place: .options))
            ) {
                OptionGroupsView(session: session, place: .beforeSubject)
                SubjectLead(session: session, fills: fills, height: subjectHeight)
                if let below { below }
                OptionGroupsView(session: session, place: .afterSubject)
            }
        } result: {
            result
        }
    }
}

/// What the run works on: the operation's files in one well, titled as the operation names them; points and boxes on
/// the picture when the operation takes them.
struct SubjectLead: View {
    @ObservedObject var session: CapabilitySession
    var fills = false
    var height: CGFloat?
    @Environment(\.palette) private var p
    static let height: CGFloat = 220
    /// A clip needs only its row and envelope.
    static let clipHeight: CGFloat = 120

    var body: some View {
        if !session.subjectFields.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(title).font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).lineLimit(1)
                    Spacer(minLength: 4)
                    if !session.spatialFields.isEmpty, !session.subject.isEmpty { PromptTools(session: session) }
                }
                let fixed = height ?? (session.subjectAccepts.allSatisfy(\.audio) ? Self.clipHeight : Self.height)
                SubjectWell(session: session).frame(height: fills ? nil : fixed).frame(maxHeight: fills ? .infinity : nil)
                if let file = session.subject.first, session.subject.count == 1 { SubjectCaption(url: file) }
                FieldProblem(session: session, keys: session.subjectFields.map(\.key) + session.spatialFields.map(\.key))
            }
        }
    }

    /// The field's own title when there is one well's worth ("Audio", "Image"), else what the alternatives make.
    private var title: String {
        let fields = session.subjectFields
        if fields.count == 1 { return fields[0].title }
        let accepts = session.subjectAccepts
        // One picture or several views of it: the single field's name.
        if accepts.allSatisfy(\.images),
            let single = fields.first(where: {
                if case .generic(.file(_, false, _, _)) = $0.control { return true }; return false
            })
        {
            return single.title
        }
        if accepts.contains(where: \.video) { return accepts.contains(where: \.images) ? (session.takesFrames ? "Video or frames" : "Image or video") : "Video" }
        if accepts.contains(where: \.document) { return "Document or image" }
        return fields.first?.title ?? "Input"
    }
}

/// The file's name and what it is (pixels, length, pages), under the well.
struct SubjectCaption: View {
    let url: URL
    @Environment(\.palette) private var p
    var body: some View {
        HStack(spacing: 8) {
            Text(url.lastPathComponent).font(.system(size: 12.5)).lineLimit(1).truncationMode(.middle).windowHelp(url.path)
            Spacer(minLength: 4)
            if let detail = Self.detail(url) {
                Text(detail).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t3.swiftUI).lineLimit(1).fixedSize()
            }
        }
    }
    @MainActor static func detail(_ url: URL) -> String? {
        if ToolFiles.isImageFile(url) { return PixelSize.read(url)?.title }
        if FileAccept(mimes: ["video/*"]).accepts(url) {
            return [VideoInfo.cachedSize(url)?.title, VideoInfo.cachedLength(url).map(AudioShape.clock)].compactMap { $0 }.joined(separator: " \u{00b7} ")
        }
        if url.pathExtension.lowercased() == "pdf" { return DocumentInfo.first(url).map { $0.count == 1 ? "1 page" : "\($0.count) pages" } }
        return nil
    }
}

/// Include, exclude and box, as the operation takes them; a clear button once something is placed.
struct PromptTools: View {
    @ObservedObject var session: CapabilitySession
    @Environment(\.palette) private var p
    var body: some View {
        let modes = Self.modes(session)
        HStack(spacing: 2) {
            if modes.count > 1 {
                ForEach(modes) { mode in
                    Button {
                        session.promptMode = mode
                    } label: {
                        Image(systemName: mode.symbol).font(.system(size: 12)).frame(width: 24, height: 22)
                            .background((session.promptMode == mode ? p.accent.withAlphaComponent(0.2) : .clear).swiftUI, in: RoundedRectangle(cornerRadius: 5))
                            .contentShape(Rectangle())
                    }.buttonStyle(.plain).foregroundStyle((session.promptMode == mode ? p.accentText : p.t2).swiftUI).windowHelp(mode.help)
                        .accessibilityLabel(mode.help).controlID("prompt-\(mode.rawValue)")
                }
            }
            if session.hasPrompts {
                Button {
                    session.clearPrompts()
                } label: {
                    Image(systemName: "arrow.counterclockwise").font(.system(size: 11)).frame(width: 24, height: 22).contentShape(Rectangle())
                }.buttonStyle(.plain).foregroundStyle(p.t2.swiftUI).windowHelp("Clear the points and boxes").accessibilityLabel("Clear the points and boxes")
                    .controlID("prompt-clear")
            }
        }
        .disabled(session.running)
    }
    static func modes(_ session: CapabilitySession) -> [PromptMode] {
        var modes: [PromptMode] = []
        if session.pointsKey != nil { modes += [.include, .exclude] }
        if session.boxesKey != nil { modes.append(.box) }
        return modes
    }
}

/// The one well for the operation's files: empty, a drop place that is also the button to choose; filled, the picture
/// (with its points and boxes), the clip, the video's first frame, the document's first page, or several pictures.
struct SubjectWell: View {
    @ObservedObject var session: CapabilitySession
    @Environment(\.palette) private var p

    var body: some View {
        let files = session.subject
        ZStack(alignment: .topTrailing) {
            Group {
                if files.isEmpty {
                    Button {
                        session.pickSubject()
                    } label: {
                        empty.contentShape(Rectangle())
                    }
                    .buttonStyle(.plain).disabled(session.running).windowHelp(help).accessibilityLabel("Choose \(noun)").controlID("well-subject")
                } else if files.count > 1 {
                    tiles(files)
                } else if let file = files.first {
                    filled(file)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            if !files.isEmpty, !session.running {
                Button {
                    session.clearSubject()
                } label: {
                    Image(systemName: "xmark.circle.fill").font(.system(size: 14)).symbolRenderingMode(.palette)
                        .foregroundStyle(Color.white, Color.black.opacity(0.6)).frame(width: 26, height: 26).contentShape(Rectangle())
                }.buttonStyle(.plain).windowHelp("Remove").accessibilityLabel("Remove \(noun)").controlID("remove-subject")
            }
        }
        .background(files.isEmpty ? Color.clear : p.sidebar.swiftUI, in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8).strokeBorder(
                files.isEmpty ? .clear : (session.dropTargeted ? p.accent : p.raisedBorder).swiftUI, lineWidth: session.dropTargeted ? 2 : 1)
        )
        .onDrop(of: [.fileURL, .image], isTargeted: $session.dropTargeted) { session.receive($0) }
    }

    private var accepts: [FileAccept] { session.subjectAccepts }
    private var noun: String { session.subjectNoun }
    private var symbol: String {
        if accepts.allSatisfy(\.audio) { return "waveform.badge.plus" }
        if accepts.contains(where: \.video) { return "film" }
        if accepts.contains(where: \.document) { return "doc.badge.plus" }
        return "photo.badge.plus"
    }
    private var help: String {
        let types = accepts.map(\.title).joined(separator: " or ")
        let paste = accepts.contains { $0.images || $0.document } ? "; \u{2318}V pastes one" : ""
        return "Drop \(noun) here, or click to choose (\(types))\(paste)"
    }

    private var empty: some View {
        VStack(spacing: 8) {
            Image(systemName: symbol).font(.system(size: 26, weight: .light))
            Text("Drop \(noun), or click to choose").font(.system(size: 12))
        }
        .foregroundStyle((session.dropTargeted ? p.accentText : p.t3).swiftUI)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background((session.dropTargeted ? p.accent.withAlphaComponent(0.12) : .clear).swiftUI, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder((session.dropTargeted ? p.accent : p.raisedBorder).swiftUI, style: StrokeStyle(lineWidth: 1, dash: [4, 3])))
    }

    @ViewBuilder private func filled(_ url: URL) -> some View {
        if ToolFiles.isImageFile(url) {
            if session.spatialFields.isEmpty {
                Button {
                    session.pickSubject()
                } label: {
                    FittedImage(url: url, size: PixelSize.read(url), label: url.lastPathComponent).padding(8).contentShape(Rectangle())
                }.buttonStyle(.plain).disabled(session.running).windowHelp("Drop or click to choose another").controlID("well-subject")
            } else {
                SubjectCanvas(session: session, image: Thumbnail.image(url), pixels: PixelSize.read(url).map { CGSize(width: $0.width, height: $0.height) }).padding(8)
            }
        } else if FileAccept(mimes: ["video/*"]).accepts(url) {
            if session.spatialFields.isEmpty {
                VideoFrame(url: url).padding(8)
            } else {
                VideoPromptCanvas(session: session, url: url).padding(8)
            }
        } else if FileAccept(mimes: ["audio/*"]).accepts(url) {
            ClipWell(url: url, player: session.player).padding(10)
        } else if let page = DocumentInfo.first(url) {
            Button {
                session.pickSubject()
            } label: {
                Image(nsImage: page.image).resizable().scaledToFit().shadow(color: .black.opacity(0.25), radius: 2).padding(10).contentShape(Rectangle())
            }.buttonStyle(.plain).disabled(session.running).windowHelp("Drop or click to choose another").controlID("well-subject")
        } else {
            VStack(spacing: 4) {
                Image(systemName: "doc").font(.system(size: 18)).foregroundStyle(p.t2.swiftUI)
                Text(url.lastPathComponent).font(.system(size: 11.5)).lineLimit(1).truncationMode(.middle).foregroundStyle(p.t2.swiftUI)
            }
        }
    }

    private func tiles(_ files: [URL]) -> some View {
        ScrollView(.vertical) {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 72), spacing: 6)], spacing: 6) {
                ForEach(files, id: \.self) { url in
                    Group {
                        if ToolFiles.isImageFile(url) { Thumbnail(url: url) } else { VideoFrame(url: url, showsLength: false) }
                    }
                    .frame(height: 72).frame(maxWidth: .infinity).clipShape(RoundedRectangle(cornerRadius: 6)).accessibilityLabel(url.lastPathComponent)
                }
            }.padding(8)
        }
    }
}

/// A clip in the well: Play, its length and its envelope.
struct ClipWell: View {
    let url: URL
    @ObservedObject var player: SoundPlayer
    @Environment(\.palette) private var p
    var body: some View {
        let shape = AudioCache.shape(url)
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Button {
                    player.toggle(url)
                } label: {
                    Image(systemName: player.playing == url ? "stop.fill" : "play.fill").font(.system(size: 12)).foregroundStyle(Color.white)
                        .frame(width: 28, height: 28).background(p.accent.swiftUI, in: Circle()).contentShape(Circle())
                }.buttonStyle(.plain).accessibilityLabel(player.playing == url ? "Stop the clip" : "Play the clip").controlID("play-subject")
                Text(shape.map { AudioShape.clock($0.seconds) } ?? "\u{2014}").font(.system(size: 13, weight: .medium).monospacedDigit())
                Spacer(minLength: 0)
            }
            if let shape { WaveformView(shape: shape, url: url, player: player).frame(maxHeight: 64) }
        }
    }
}

/// Envelopes of clips the window shows, read once per file.
@MainActor enum AudioCache {
    private static let shapes = MediaCache<URL, AudioShape>(limit: 16 * MediaCaches.mb) { $0.peaks.count * 4 + 64 }
    static func shape(_ url: URL) -> AudioShape? {
        if let hit = shapes[url] { return hit }
        guard let shape = AudioShape.read(url) else { return nil }
        shapes.set(shape, for: url)
        return shape
    }
}

/// The subject picture with the operation's points and boxes placed on it.
struct SubjectCanvas: View {
    @ObservedObject var session: CapabilitySession
    let image: NSImage?
    let pixels: CGSize?
    var body: some View {
        PromptCanvas(
            image: image, pixels: pixels ?? image?.size ?? CGSize(width: 1, height: 1), points: session.pointsBinding, boxes: session.boxesBinding,
            takesPoints: session.pointsKey != nil, takesBoxes: session.boxesKey != nil, mode: session.promptMode, enabled: !session.running
        ) { box in
            ForEach(Array(session.canvasMasks.enumerated()), id: \.offset) { i, mask in MaskOverlay(mask: mask, box: box, colour: SpatialColours.colour(i)) }
        }
        .controlID("canvas")
    }
}

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias FrameState = SwiftUI.State<NSImage?>

/// A video's first frame as the picture prompts are placed on (frame 0, or the frame the window names).
struct VideoPromptCanvas: View {
    @ObservedObject var session: CapabilitySession
    let url: URL
    @FrameState private var frame: NSImage?
    var body: some View {
        let pixels = VideoInfo.cachedSize(url).map { CGSize(width: $0.width, height: $0.height) } ?? frame?.size
        let time = session.promptTime
        SubjectCanvas(session: session, image: frame ?? VideoInfo.cachedFrame(url, at: time) ?? VideoInfo.cachedFrame(url), pixels: pixels)
            .task(id: "\(url.path)@\(time)") {
                if VideoInfo.cachedFrame(url) == nil { _ = await VideoInfo.load(url) }
                frame = await VideoInfo.frame(url, at: time)
            }
    }
}

extension CapabilitySession {
    /// What the well takes, as a person says it: "a clip", "an image", "a video or two frames".
    var subjectNoun: String {
        let accepts = subjectAccepts
        if accepts.isEmpty { return "a file" }
        if accepts.allSatisfy(\.audio) { return "a clip" }
        if accepts.allSatisfy(\.video) { return "a video" }
        if accepts.contains(where: \.video) { return takesFrames ? "a video or two frames" : accepts.contains(where: \.images) ? "an image or a video" : "a video" }
        if accepts.contains(where: \.document) { return "a document" }
        return "an image"
    }
    /// The operation takes several pictures as an alternative to a video (Smooth Video's two frames).
    var takesFrames: Bool {
        subjectFields.contains { f in
            if case .generic(.file(let a, true, _, _)) = f.control { return a.images }; return false
        }
    }
    var pointsKey: String? {
        spatialFields.first {
            if case .points = $0.control { return true }; return false
        }?.key
    }
    var boxesKey: String? {
        spatialFields.first {
            if case .boxes = $0.control { return true }; return false
        }?.key
    }
    var hasPrompts: Bool { spatialFields.contains { !value($0.key).text.isEmpty } }
    func clearPrompts() { for f in spatialFields { set(f.key, .text("")) } }
    var pointsBinding: Binding<[SpatialPoint]> {
        Binding(
            get: { self.pointsKey.map { SpatialValue.points(self.value($0).text) } ?? [] },
            set: { list in if let key = self.pointsKey { self.set(key, .text(SpatialValue.text(list))); self.promptsChanged() } })
    }
    var boxesBinding: Binding<[SpatialBox]> {
        Binding(
            get: { self.boxesKey.map { SpatialValue.boxes(self.value($0).text) } ?? [] },
            set: { list in if let key = self.boxesKey { self.set(key, .text(SpatialValue.text(list))); self.promptsChanged() } })
    }
}
