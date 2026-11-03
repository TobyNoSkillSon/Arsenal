// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// The tool kit's result side, the same in every tool window: a panel on the sidebar ground with the tool's symbol while
// it is empty, the run's progress over it (on an opaque card above a scrim once there is a result, so nothing runs
// through the words), the result's actions (Copy, Show in Finder) above it, and this window's earlier results under it
// in a strip whose place is kept from the first frame. Results drag out as their files.

struct ResultPanel<Content: View>: View {
    @ObservedObject var session: ToolSession
    /// The tool's symbol, faint while there is no result.
    let symbol: String
    let empty: Bool
    /// A drag is over the panel (a drop place too).
    var highlighted = false
    @ViewBuilder let content: Content
    @Environment(\.palette) private var p
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10).fill(p.sidebar.swiftUI)
            if empty, !session.running {
                Image(systemName: symbol).font(.system(size: 40, weight: .light)).foregroundStyle(p.t4.swiftUI).accessibilityHidden(true)
            }
            // A failed or declined run leaves the earlier result at full strength: only the new run failed, and the line by
            // the run button says so.
            content.controlID("result-content")
            if let run = session.run, session.running, !session.quietRun {
                ProgressOverlay(
                    run: run, over: !empty, line: run.line(model: session.models.first { $0.id == run.model }?.name ?? run.model, idle: session.idleLine),
                    reduceMotion: reduceMotion)
            }
        }
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(highlighted && !session.running ? p.accent.swiftUI : .clear, lineWidth: 2).allowsHitTesting(false))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// While a run is made: the step bar (or a spinner) and its words, in the accent on the empty panel; over a result, the
/// same on an opaque card above a scrim.
struct ProgressOverlay: View {
    let run: ToolRun
    let over: Bool
    let line: String
    let reduceMotion: Bool
    @Environment(\.palette) private var p

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10).fill(Color.black.opacity(over ? (p.dark ? 0.45 : 0.16) : 0))
            VStack(spacing: 10) {
                if let fraction = run.fraction {
                    ZStack(alignment: .leading) {
                        Capsule().fill(p.t4.swiftUI)
                        GeometryReader { g in Capsule().fill(p.accent.swiftUI).frame(width: max(5, g.size.width * fraction)) }
                    }.frame(width: 220, height: 5)
                } else {
                    ProgressView().controlSize(.small).colorScheme(p.dark ? .dark : .light)
                }
                Text(line).font(.system(size: 13, weight: .semibold).monospacedDigit()).foregroundStyle(p.t1.swiftUI)
            }
            .padding(.horizontal, 24).padding(.vertical, 18)
            .background(over ? p.raised.swiftUI : .clear, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(over ? p.raisedBorder.swiftUI : .clear, lineWidth: 1))
            .animation(reduceMotion ? nil : .smooth(duration: 0.3), value: run.fraction)
        }
        .allowsHitTesting(false)
    }
}

/// A result's action: a borderless symbol button (Copy, Show in Finder, Save).
struct ResultIcon: View {
    let symbol: String
    let help: String
    let id: String
    var enabled = true
    let action: () -> Void
    @Environment(\.palette) private var p
    var body: some View {
        Button(action: action) {
            Image(systemName: symbol).font(.system(size: 13)).frame(width: 28, height: 26).contentShape(Rectangle())
        }.buttonStyle(.plain).foregroundStyle(p.t2.swiftUI).disabled(!enabled).windowHelp(help).accessibilityLabel(help).controlID(id)
    }
}

/// The shown result's line of actions: the tool's own controls on the left (Cut-out | Mask), Save As… and Copy on the
/// right, acting on the file shown. Show in Finder is the band's (one folder per window: the shown result, else the
/// tool's folder).
struct ResultActions<Leading: View>: View {
    init(
        what: String, enabled: Bool = true, copy: @escaping () -> Void, @ViewBuilder leading: () -> Leading, trailing: AnyView? = nil, showsCopy: Bool = true,
        save: (() -> Void)? = nil
    ) {
        self.what = what; self.enabled = enabled; self.copy = copy; self.leading = leading(); self.trailing = trailing; self.showsCopy = showsCopy
        self.save = save
    }
    /// Save As… for a result that is a file.
    var save: (() -> Void)?
    /// False where each item has its own Copy (Separate Audio's stems): a header Copy would not say what it copies.
    var showsCopy = true
    let what: String
    var enabled = true
    let copy: () -> Void
    @ViewBuilder let leading: Leading
    /// A window's own actions beside Copy (Save as text, Open).
    var trailing: AnyView?
    static var height: CGFloat { 40 }

    var body: some View {
        HStack(spacing: 6) {
            // The view pills give way to the actions in a narrow window: drawn as they are when they fit, else they
            // scroll within their own room, so Save and Copy never leave the toolbar.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 6) { leading }.fixedSize()
                ScrollView(.horizontal) { HStack(spacing: 6) { leading } }.scrollIndicators(.never)
            }
            .layoutPriority(-1)
            Spacer(minLength: 8)
            Group {
                if let trailing { trailing }
                if let save { SaveAsIcon(enabled: enabled, action: save) }
                if showsCopy { ResultIcon(symbol: "doc.on.doc", help: "Copy the \(what)", id: "copy", enabled: enabled, action: copy) }
            }.fixedSize()
        }
        .padding(.leading, 16).padding(.trailing, 10)
        .frame(height: Self.height)
    }
}

extension ResultActions where Leading == EmptyView {
    init(what: String, enabled: Bool = true, copy: @escaping () -> Void, save: (() -> Void)? = nil) {
        self.init(what: what, enabled: enabled, copy: copy, leading: { EmptyView() }, trailing: nil, save: save)
    }
}

/// Save As… on a result: a copy where the user chooses (the file's own name and folder suggested).
struct SaveAsIcon: View {
    var enabled = true
    var help = "Save As\u{2026}"
    var id = "save-as"
    let action: () -> Void
    var body: some View { ResultIcon(symbol: "square.and.arrow.down", help: help, id: id, enabled: enabled, action: action) }
}

/// This window's earlier results as thumbnails over a checkerboard, the shown one outlined. No hover: inside a scroll
/// view AppKit's hit test stops at SwiftUI's container, so a hover there could not be shown reliably (HoverReachTests).
struct ImageStrip: View {
    struct Item: Identifiable {
        var id: UUID
        var url: URL
        var label: String
        var controlID: String
    }
    let items: [Item]
    let current: UUID?
    var enabled = true
    let select: (UUID) -> Void
    @Environment(\.palette) private var p
    static let height: CGFloat = 66

    var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(items) { item in
                    Button {
                        select(item.id)
                    } label: {
                        // The checkerboard only under the picture itself: a letterbox stays the tile's plain ground.
                        FittedImage(url: item.url, size: nil, label: item.label)
                            .frame(width: 64, height: 64).background(p.canvas.swiftUI).clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(
                                RoundedRectangle(cornerRadius: 6).strokeBorder((current == item.id ? p.accent : p.raisedBorder).swiftUI, lineWidth: current == item.id ? 2 : 1)
                            )
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain).disabled(!enabled).accessibilityLabel(item.label).controlID(item.controlID)
                }
            }
        }.scrollIndicators(.never).frame(height: Self.height)
    }
}

extension View {
    /// A history's place, kept from the first frame so the result above never changes size as results arrive: drawn and
    /// usable from the first result, so it is plain that results are kept.
    func historySlot(_ shown: Bool) -> some View { opacity(shown ? 1 : 0).allowsHitTesting(shown) }
}

/// An image at the largest size that fits, the checkerboard exactly under it (its transparent parts read as empty; an
/// opaque image covers it); draggable out as its file when asked.
struct FittedImage: View {
    let url: URL
    let size: PixelSize?
    var draggable = false
    let label: String
    var help: String?
    /// The file a drag takes, asked when the drag starts (Select keeps its selection first); nil: `url`.
    var willDrag: (() -> URL?)?
    /// The checkerboard behind transparency; off for a rendered object (a 3D splat's preview), whose empty parts are background.
    var backdrop = true
    @Environment(\.palette) private var p

    var body: some View {
        GeometryReader { g in
            let pixels = size.map { CGSize(width: $0.width, height: $0.height) } ?? Thumbnail.image(url)?.size ?? CGSize(width: 1, height: 1)
            let aspect = pixels.width > 0 && pixels.height > 0 ? pixels.width / pixels.height : 1
            let w = min(g.size.width, g.size.height * aspect), h = w / aspect
            image.frame(width: w, height: h).frame(width: g.size.width, height: g.size.height)
        }
    }

    @ViewBuilder private var image: some View {
        // One layer: dimmed (a run going), the picture dims as a whole and its checkerboard never shows through.
        let drawn = ZStack {
            if backdrop { Checkerboard() }
            Thumbnail(url: url, fit: true)
        }
        .compositingGroup()
        .background(p.canvas.swiftUI)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .accessibilityLabel(label)
        .windowHelp(help)
        if draggable { drawn.onDrag { NSItemProvider(contentsOf: willDrag?() ?? url) ?? NSItemProvider() } } else { drawn }
    }
}

/// A transparent image's backdrop: quiet squares, so a layer's empty parts read as empty, not black.
struct Checkerboard: View {
    @Environment(\.palette) private var p
    var body: some View {
        Canvas(rendersAsynchronously: false) { context, size in
            let cell: CGFloat = 8
            let light = p.dark ? Color.white.opacity(0.06) : Color.black.opacity(0.05)
            var path = Path()
            for row in 0..<Int((size.height / cell).rounded(.up)) {
                for col in 0..<Int((size.width / cell).rounded(.up)) where (row + col) % 2 == 0 {
                    path.addRect(CGRect(x: CGFloat(col) * cell, y: CGFloat(row) * cell, width: cell, height: cell))
                }
            }
            context.fill(path, with: .color(light))
        }
    }
}

/// An image file, read once per window session: the figures tick ten times a second while a run is made, and the
/// result area must not reread its file on each tick. What is kept is a display copy (MediaCaches.displayImage): no
/// tool window shows a picture larger than the screen, so before and after at 24 MP each stay cached side by side
/// (the budget holds at least three screen-sized copies on a display larger than this Mac's).
struct Thumbnail: View {
    let url: URL
    var fit = false
    @MainActor static let cache = MediaCache<URL, NSImage>(
        limit: max(128 * MediaCaches.mb, 3 * Int(MediaCaches.displayPixels.width * MediaCaches.displayPixels.height) * 4), cost: MediaCaches.bytes)
    @MainActor static func image(_ url: URL) -> NSImage? {
        if let image = cache[url] { return image }
        guard let image = MediaCaches.displayImage(url, fitting: MediaCaches.displayPixels) else { return nil }
        cache.set(image, for: url)
        return image
    }
    var body: some View {
        if FileAccept(mimes: ["video/*"]).accepts(url) {
            VideoFrame(url: url, showsLength: false)
        } else if let image = Self.image(url) {
            Image(nsImage: image).resizable().aspectRatio(contentMode: fit ? .fit : .fill)
        } else {
            Color.gray.opacity(0.2)
        }
    }
}
