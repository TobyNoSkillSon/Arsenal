// SPDX-License-Identifier: MIT
import AppKit
import CoreImage
import ArsenalCore
import SwiftUI

// The tool kit's spatial pieces: points and boxes placed on an image (or a video's frame) by clicking and dragging,
// kept normalized to the picture's original extent (the letterbox around a fitted picture is never counted), and a
// run's masks, boxes and points drawn back over the same picture at the same mapping.

/// A point prompt: normalized x and y, a foreground (include) or background (exclude) label.
struct SpatialPoint: Codable, Equatable {
    var x: Double
    var y: Double
    var label: String?
    var excludes: Bool { label == "background" }
}

/// A box prompt or result: normalized corners.
struct SpatialBox: Codable, Equatable {
    var xmin: Double
    var ymin: Double
    var xmax: Double
    var ymax: Double
    var rect: CGRect { CGRect(x: xmin, y: ymin, width: xmax - xmin, height: ymax - ymin) }
}

/// The JSON a points or boxes field holds.
enum SpatialValue {
    static func points(_ text: String) -> [SpatialPoint] {
        (try? JSONDecoder().decode([SpatialPoint].self, from: Data(text.utf8))) ?? []
    }
    static func boxes(_ text: String) -> [SpatialBox] {
        (try? JSONDecoder().decode([SpatialBox].self, from: Data(text.utf8))) ?? []
    }
    static func text<T: Encodable>(_ list: [T]) -> String {
        guard !list.isEmpty else { return "" }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return (try? encoder.encode(list)).flatMap { String(data: $0, encoding: .utf8) } ?? ""
    }
}

/// Where a picture of `pixels` sits fitted inside `container` (centred, aspect kept), and the mapping between view
/// points and normalized picture coordinates.
struct Letterbox: Equatable {
    var rect: CGRect
    /// `top`: the picture sits at the top of its place (beside a list that starts at the top) rather than centred.
    init(container: CGSize, pixels: CGSize, top: Bool = false) {
        let aspect = pixels.width > 0 && pixels.height > 0 ? pixels.width / pixels.height : 1
        let w = min(container.width, container.height * aspect), h = w / aspect
        rect = CGRect(x: (container.width - w) / 2, y: top ? 0 : (container.height - h) / 2, width: w, height: h)
    }
    /// A view point as normalized picture coordinates, clamped to the picture; nil when it is in the letterbox.
    func normalized(_ p: CGPoint) -> CGPoint? {
        guard rect.width > 0, rect.height > 0, rect.insetBy(dx: -1, dy: -1).contains(p) else { return nil }
        return CGPoint(x: min(1, max(0, (p.x - rect.minX) / rect.width)), y: min(1, max(0, (p.y - rect.minY) / rect.height)))
    }
    func view(_ n: CGPoint) -> CGPoint { CGPoint(x: rect.minX + n.x * rect.width, y: rect.minY + n.y * rect.height) }
    func view(_ box: CGRect) -> CGRect {
        CGRect(x: rect.minX + box.minX * rect.width, y: rect.minY + box.minY * rect.height, width: box.width * rect.width, height: box.height * rect.height)
    }
}

/// One detection as a window draws it: label, score, a normalized box and/or point.
struct Detection: Equatable, Identifiable {
    var id: Int
    var label: String
    var score: Double
    var box: CGRect?
    var point: CGPoint?

    /// The reply's detections, normalized whatever its coordinate system: screenshot pixels are divided by the
    /// reply's own image size (or the picture's when the reply has none).
    static func list(_ reply: JSONObject, imageSize: PixelSize?) -> [Detection] {
        let pixels = reply["coordinate_system"]?.string == "pixels"
        let w = reply["image_size"]?["width"]?.number ?? imageSize.map { Double($0.width) } ?? 1
        let h = reply["image_size"]?["height"]?.number ?? imageSize.map { Double($0.height) } ?? 1
        let sx = pixels ? 1 / max(1, w) : 1, sy = pixels ? 1 / max(1, h) : 1
        return (reply["detections"]?.array ?? []).enumerated().compactMap { i, value in
            guard let d = value.object else { return nil }
            var box: CGRect?
            if let b = d["box"]?.object, let x0 = b["xmin"]?.number, let y0 = b["ymin"]?.number, let x1 = b["xmax"]?.number, let y1 = b["ymax"]?.number {
                box = CGRect(x: x0 * sx, y: y0 * sy, width: (x1 - x0) * sx, height: (y1 - y0) * sy)
            }
            let point = d["point"]?.object.flatMap { p in p["x"]?.number.flatMap { x in p["y"]?.number.map { CGPoint(x: x * sx, y: $0 * sy) } } }
            return Detection(id: i, label: d["label"]?.string ?? "", score: d["score"]?.number ?? 0, box: box, point: point)
        }
    }
}

/// Colours for things a window tells apart (speakers, labels, masks): distinct hues that read on dark and light.
enum SpatialColours {
    static let all: [NSColor] = [.systemBlue, .systemOrange, .systemGreen, .systemPink, .systemPurple, .systemTeal, .systemYellow, .systemRed, .systemIndigo, .systemMint]
    static func colour(_ i: Int) -> Color { Color(nsColor: all[((i % all.count) + all.count) % all.count]) }
    /// The same colour for the same name, in the order names first appear.
    static func colour(_ name: String, in names: [String]) -> Color { colour(names.firstIndex(of: name) ?? 0) }
}

/// How a click on the picture places a prompt.
enum PromptMode: String, CaseIterable, Identifiable {
    case include, exclude, box
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .include: return "plus.circle"
        case .exclude: return "minus.circle"
        case .box: return "rectangle.dashed"
        }
    }
    var help: String {
        switch self {
        case .include: return "Click what to select (\u{2325}-click excludes)"
        case .exclude: return "Click what to leave out"
        case .box: return "Drag a box around what to select"
        }
    }
}

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias CanvasDrag = SwiftUI.State<PromptDrag?>
struct PromptDrag: Equatable { var start: CGPoint; var end: CGPoint }

/// A picture with prompts placed on it: a click adds a point (⌥-click a background point), a drag draws a box, a click
/// on a point or inside a box's corner removes it. Results (masks, detections) are drawn under the prompts.
struct PromptCanvas<Overlay: View>: View {
    let image: NSImage?
    let pixels: CGSize
    @Binding var points: [SpatialPoint]
    @Binding var boxes: [SpatialBox]
    var takesPoints = true
    var takesBoxes = true
    var mode: PromptMode = .include
    var enabled = true
    @ViewBuilder let overlay: (Letterbox) -> Overlay
    @CanvasDrag private var drag: PromptDrag?
    @Environment(\.palette) private var p

    var body: some View {
        GeometryReader { g in
            let box = Letterbox(container: g.size, pixels: pixels)
            ZStack(alignment: .topLeading) {
                ZStack {
                    Checkerboard()
                    if let image { Image(nsImage: image).resizable().scaledToFit() }
                }
                .frame(width: box.rect.width, height: box.rect.height)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .offset(x: box.rect.minX, y: box.rect.minY)
                overlay(box)
                prompts(box)
                if let drag {
                    let r = CGRect(
                        x: min(drag.start.x, drag.end.x), y: min(drag.start.y, drag.end.y), width: abs(drag.end.x - drag.start.x), height: abs(drag.end.y - drag.start.y))
                    Rectangle().strokeBorder(p.accent.swiftUI, style: StrokeStyle(lineWidth: 1.5, dash: [5, 3]))
                        .frame(width: r.width, height: r.height).offset(x: r.minX, y: r.minY).allowsHitTesting(false)
                }
            }
            .frame(width: g.size.width, height: g.size.height, alignment: .topLeading)
            .contentShape(Rectangle())
            .gesture(enabled ? gesture(box) : nil)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Image: \(points.count) points, \(boxes.count) boxes")
    }

    @ViewBuilder private func prompts(_ box: Letterbox) -> some View {
        ForEach(Array(boxes.enumerated()), id: \.offset) { _, b in
            let r = box.view(b.rect)
            Rectangle().strokeBorder(Color.black.opacity(0.45), lineWidth: 4)
                .frame(width: max(2, r.width), height: max(2, r.height)).offset(x: r.minX, y: r.minY).allowsHitTesting(false)
            Rectangle().strokeBorder(p.accent.swiftUI, lineWidth: 2).background(p.accent.withAlphaComponent(0.08).swiftUI)
                .frame(width: max(2, r.width), height: max(2, r.height)).offset(x: r.minX, y: r.minY).allowsHitTesting(false)
        }
        ForEach(Array(points.enumerated()), id: \.offset) { _, point in
            let at = box.view(CGPoint(x: point.x, y: point.y))
            Circle().fill((point.excludes ? p.red : p.green).swiftUI)
                .overlay(Image(systemName: point.excludes ? "minus" : "plus").font(.system(size: 8, weight: .heavy)).foregroundStyle(Color.white))
                .overlay(Circle().strokeBorder(Color.white, lineWidth: 1.5))
                .shadow(color: .black.opacity(0.4), radius: 1.5)
                .frame(width: 16, height: 16).offset(x: at.x - 8, y: at.y - 8).allowsHitTesting(false)
        }
    }

    private func gesture(_ box: Letterbox) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { v in
                guard takesBoxes, mode == .box || !takesPoints || hypot(v.translation.width, v.translation.height) > 6 else { return }
                drag = PromptDrag(start: v.startLocation, end: v.location)
            }
            .onEnded { v in
                defer { drag = nil }
                let moved = hypot(v.translation.width, v.translation.height) > 6
                if moved, takesBoxes {
                    guard let a = box.normalized(v.startLocation) ?? box.normalized(clamp(v.startLocation, box)),
                        let b = box.normalized(clamp(v.location, box))
                    else { return }
                    let r = SpatialBox(xmin: min(a.x, b.x), ymin: min(a.y, b.y), xmax: max(a.x, b.x), ymax: max(a.y, b.y))
                    if r.xmax - r.xmin > 0.005, r.ymax - r.ymin > 0.005 { boxes.append(r) }
                    return
                }
                guard let n = box.normalized(v.location) else { return }
                // A click on a prompt removes it.
                if let i = points.firstIndex(where: { hypot(box.view(CGPoint(x: $0.x, y: $0.y)).x - v.location.x, box.view(CGPoint(x: $0.x, y: $0.y)).y - v.location.y) < 9 }) {
                    points.remove(at: i); return
                }
                if let i = boxes.firstIndex(where: { b in
                    let r = box.view(b.rect)
                    return [CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.maxX, y: r.maxY)].contains { hypot($0.x - v.location.x, $0.y - v.location.y) < 9 }
                }) {
                    boxes.remove(at: i); return
                }
                guard takesPoints else { return }
                let excludes = mode == .exclude || NSEvent.modifierFlags.contains(.option)
                points.append(SpatialPoint(x: n.x, y: n.y, label: excludes ? "background" : "foreground"))
            }
    }
    private func clamp(_ p: CGPoint, _ box: Letterbox) -> CGPoint {
        CGPoint(x: min(box.rect.maxX, max(box.rect.minX, p.x)), y: min(box.rect.maxY, max(box.rect.minY, p.y)))
    }
}

/// A mask file tinted over the picture: white parts in `colour`, black parts clear (the grey becomes the alpha of a
/// template image, made once per mask).
struct MaskOverlay: View {
    let mask: URL
    let box: Letterbox
    var colour: Color
    @MainActor private static let alphas = MediaCache<String, NSImage>(limit: 64 * MediaCaches.mb, cost: MediaCaches.bytes)
    /// The mask's grey as alpha (`edge`: only its outline, a few pixels wide), made once per mask at display size (a
    /// mask is drawn over the picture, never larger than the screen; a full 24 MP alpha would not fit its cache).
    @MainActor static func alpha(_ url: URL, edge: Bool = false) -> NSImage? {
        let key = url.path + (edge ? "#edge" : "")
        if let hit = alphas[key] { return hit }
        guard var grey = CIImage(contentsOf: url) else { return nil }
        let shown = MediaCaches.fitted(grey.extent.size, in: MediaCaches.displayPixels)
        if shown.width < grey.extent.width {
            grey = grey.applyingFilter(
                "CILanczosScaleTransform",
                parameters: [kCIInputScaleKey: shown.height / grey.extent.height, kCIInputAspectRatioKey: (shown.width / shown.height) / (grey.extent.width / grey.extent.height)])
            grey = grey.cropped(to: CGRect(origin: grey.extent.origin, size: shown))
        }
        let extent = grey.extent
        if edge {
            let radius = max(1.5, min(extent.width, extent.height) / 300)
            grey = grey.clampedToExtent().applyingFilter("CIMorphologyGradient", parameters: [kCIInputRadiusKey: radius]).cropped(to: extent)
        }
        let alpha = grey.applyingFilter("CIMaskToAlpha")
        guard let cg = CIContext().createCGImage(alpha, from: extent) else { return nil }
        let image = MediaCaches.picture(cg)
        image.isTemplate = true
        alphas.set(image, for: key)
        return image
    }
    var body: some View {
        // The fill in the selection's colour, its outline white over a dark halo: it reads on a sky the same colour.
        ZStack {
            if let image = Self.alpha(mask) {
                Image(nsImage: image).renderingMode(.template).resizable().foregroundStyle(colour.opacity(0.5))
            }
            if let edge = Self.alpha(mask, edge: true) {
                Image(nsImage: edge).renderingMode(.template).resizable().foregroundStyle(Color.white)
                    .shadow(color: .black.opacity(0.7), radius: 1)
            }
        }
        .frame(width: box.rect.width, height: box.rect.height)
        .offset(x: box.rect.minX, y: box.rect.minY)
        .allowsHitTesting(false)
    }
}

/// Detections over the picture: each box in its label's colour with its label and score, each point as a ring. The
/// chosen one is drawn heavier, the rest lighter.
struct DetectionOverlay: View {
    let detections: [Detection]
    let box: Letterbox
    var chosen: Int?

    /// A box's label as it reads: its label and score.
    static func text(_ d: Detection) -> String { "\(d.label) \(Int((d.score * 100).rounded()))%" }
    static let font = NSFont.monospacedDigitSystemFont(ofSize: 10.5, weight: .semibold)
    /// The label's whole width: its text and the tag's 4 pt padding each side.
    static func natural(_ d: Detection) -> CGFloat {
        ceil((text(d) as NSString).size(withAttributes: [.font: font]).width) + 8 + 2
    }
    static let tagHeight: CGFloat = 16
    /// Where each box's label goes: whole, from its box's left edge (kept on the picture), and wider than its box when it
    /// needs to be. Above the box first (inside its top edge when the box starts at the picture's top); a label that
    /// would cover one placed before it takes the next free line (inside the box's top edge, then one higher). Only when no line is
    /// free is it cut to its box's width, where it was.
    static func tags(_ boxes: [(id: Int, box: CGRect, width: CGFloat)], picture: CGRect) -> [Int: CGRect] {
        var placed: [Int: CGRect] = [:]
        for b in boxes {
            let top = b.box.minY
            let lines = [top - 17, top + 2, top - 34, top + 19].filter { $0 >= picture.minY && $0 + tagHeight <= picture.maxY }
            func at(_ y: CGFloat, width: CGFloat) -> CGRect {
                let w = min(width, picture.width)
                return CGRect(x: max(picture.minX, min(b.box.minX, picture.maxX - w)), y: y, width: w, height: tagHeight)
            }
            let free = lines.lazy.map { at($0, width: b.width) }.first { r in !placed.values.contains { $0.insetBy(dx: -1, dy: -1).intersects(r) } }
            placed[b.id] = free ?? at(lines.first ?? max(picture.minY, top + 2), width: max(30, min(b.width, b.box.width)))
        }
        return placed
    }
    var body: some View {
        let labels = detections.map(\.label).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
        let tags = Self.tags(detections.compactMap { d in d.box.map { (d.id, box.view($0), Self.natural(d)) } }, picture: box.rect)
        ZStack(alignment: .topLeading) {
            ForEach(detections) { d in
                let colour = SpatialColours.colour(d.label, in: labels)
                let heavy = chosen == nil || chosen == d.id
                if let b = d.box {
                    let r = box.view(b)
                    // A dark rim under the colour: the box reads on a light picture as on a dark one.
                    Rectangle().strokeBorder(Color.black.opacity(heavy ? 0.45 : 0.25), lineWidth: heavy ? 4 : 3)
                        .frame(width: max(2, r.width), height: max(2, r.height)).offset(x: r.minX, y: r.minY)
                    Rectangle().strokeBorder(colour.opacity(heavy ? 1 : 0.45), lineWidth: heavy ? 2 : 1.5)
                        .frame(width: max(2, r.width), height: max(2, r.height)).offset(x: r.minX, y: r.minY)
                    if let tag = tags[d.id] {
                        Text(Self.text(d)).font(.system(size: 10.5, weight: .semibold).monospacedDigit())
                            .lineLimit(1).truncationMode(.tail).fixedSize(horizontal: tag.width >= Self.natural(d), vertical: false)
                            .foregroundStyle(Color.white).padding(.horizontal, 4).frame(width: tag.width, height: Self.tagHeight, alignment: .leading)
                            .background(colour.opacity(heavy ? 0.9 : 0.5), in: RoundedRectangle(cornerRadius: 3))
                            .offset(x: tag.minX, y: tag.minY)
                    }
                }
                if let pt = d.point {
                    let at = box.view(pt)
                    Circle().strokeBorder(Color.black.opacity(0.55), lineWidth: 4.5).frame(width: 20, height: 20).offset(x: at.x - 10, y: at.y - 10)
                    Circle().strokeBorder(colour, lineWidth: 2.5).background(Circle().fill(Color.white.opacity(0.35)))
                        .frame(width: 18, height: 18).offset(x: at.x - 9, y: at.y - 9)
                    if d.box == nil {
                        let title = d.label.isEmpty ? "\(Int((d.score * 100).rounded()))%" : d.label
                        let width = min(160, CGFloat(title.count) * 6.4 + 10)
                        // Beside the ring, on whichever side keeps it on the picture.
                        let x = at.x + 11 + width > box.rect.maxX ? at.x - 11 - width : at.x + 11
                        Text(title).font(.system(size: 10.5, weight: .semibold)).lineLimit(1).truncationMode(.tail)
                            .foregroundStyle(Color.white).padding(.horizontal, 4).frame(width: width, height: 16, alignment: .leading)
                            .background(colour.opacity(0.9), in: RoundedRectangle(cornerRadius: 3))
                            .offset(x: max(box.rect.minX, x), y: min(box.rect.maxY - 16, max(box.rect.minY, at.y - 8)))
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }
}
