// SPDX-License-Identifier: MIT
import AVFoundation
import AVKit
import AppKit
import ImageIO
import PDFKit
import SwiftUI

// The tool kit's media: a video's first frame and length, a document's first page and page count, a video result
// that plays only when asked (inline controls, nothing moves at rest), the result over its original split by a handle
// the pointer moves, a depth map in colour with its near and far ends, and pictures read from files once per window.

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias MediaState<T> = SwiftUI.State<T>

/// A video's first frame and length, read once per file.
@MainActor enum VideoInfo {
    private static let frames = MediaCache<URL, NSImage>(limit: 64 * MediaCaches.mb, cost: MediaCaches.bytes)
    /// A few numbers per file, dropped with the frames.
    private static let lengths = MediaCache<URL, Double>(limit: MediaCaches.mb) { _ in 64 }
    private static let sizes = MediaCache<URL, PixelSize>(limit: MediaCaches.mb) { _ in 64 }
    private static let rates = MediaCache<URL, Double>(limit: MediaCaches.mb) { _ in 64 }
    private static let stills = MediaCache<String, NSImage>(limit: 64 * MediaCaches.mb, cost: MediaCaches.bytes)
    static func cachedFrame(_ url: URL) -> NSImage? { frames[url] }
    /// Frames per second (its nominal rate; 30 when the file does not say).
    static func cachedRate(_ url: URL) -> Double? { rates[url] }
    static func cachedFrame(_ url: URL, at seconds: Double) -> NSImage? { seconds <= 0 ? frames[url] : stills["\(url.path)@\(seconds)"] }

    /// The frame at a time (the first frame at 0).
    static func frame(_ url: URL, at seconds: Double) async -> NSImage? {
        if seconds <= 0 { return await load(url) }
        let key = "\(url.path)@\(seconds)"
        if let hit = stills[key] { return hit }
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1600, height: 1600)
        generator.requestedTimeToleranceBefore = .zero; generator.requestedTimeToleranceAfter = .zero
        guard let (cg, _) = try? await generator.image(at: CMTime(seconds: seconds, preferredTimescale: 600)) else { return nil }
        let image = MediaCaches.picture(cg)
        stills.set(image, for: key)
        return image
    }
    static func cachedLength(_ url: URL) -> Double? { lengths[url] }
    static func cachedSize(_ url: URL) -> PixelSize? { sizes[url] }

    /// The first frame (upright, as the video plays), its length and display size.
    static func load(_ url: URL) async -> NSImage? {
        if let frame = frames[url] { return frame }
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1600, height: 1600)
        guard let (cg, _) = try? await generator.image(at: .zero) else { return nil }
        let image = MediaCaches.picture(cg)
        if let duration = try? await asset.load(.duration) { lengths.set(duration.seconds, for: url) }
        if let track = try? await asset.loadTracks(withMediaType: .video).first, let rate = try? await track.load(.nominalFrameRate), rate > 0 {
            rates.set(Double(rate), for: url)
        }
        if let track = try? await asset.loadTracks(withMediaType: .video).first, let size = try? await track.load(.naturalSize),
            let transform = try? await track.load(.preferredTransform)
        {
            let r = CGRect(origin: .zero, size: size).applying(transform)
            sizes.set(PixelSize(width: Int(abs(r.width).rounded()), height: Int(abs(r.height).rounded())), for: url)
        }
        frames.set(image, for: url)
        return image
    }
}

/// A video's first frame, fitted, with its length in the corner.
struct VideoFrame: View {
    let url: URL
    var showsLength = true
    @MediaState private var frame: NSImage?
    @Environment(\.palette) private var p

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            if let image = frame ?? VideoInfo.cachedFrame(url) {
                Image(nsImage: image).resizable().scaledToFit()
            } else {
                Image(systemName: "film").font(.system(size: 26, weight: .light)).foregroundStyle(p.t3.swiftUI).frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if showsLength, let length = VideoInfo.cachedLength(url) {
                Label(AudioShape.clock(length), systemImage: "film").font(.system(size: 11, weight: .medium).monospacedDigit())
                    .foregroundStyle(Color.white).padding(.horizontal, 6).frame(height: 20)
                    .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 5)).padding(6)
            }
        }
        .task(id: url) { frame = await VideoInfo.load(url) }
    }
}

/// A document's first page (a PDF's, or the image itself) and its page count, read once per file.
@MainActor enum DocumentInfo {
    private static let pages = MediaCache<URL, (image: NSImage, count: Int)>(limit: 64 * MediaCaches.mb) { MediaCaches.bytes($0.image) }
    static func first(_ url: URL) -> (image: NSImage, count: Int)? {
        if let hit = pages[url] { return hit }
        var found: (NSImage, Int)?
        if url.pathExtension.lowercased() == "pdf" {
            guard let doc = PDFDocument(url: url), let page = doc.page(at: 0) else { return nil }
            let bounds = page.bounds(for: .mediaBox)
            let scale = 900 / max(bounds.width, bounds.height, 1)
            found = (page.thumbnail(of: NSSize(width: bounds.width * scale, height: bounds.height * scale), for: .mediaBox), doc.pageCount)
        } else if let image = Thumbnail.image(url) {
            found = (image, 1)
        }
        if let found { pages.set(found, for: url) }
        return found
    }
}

/// A video result at rest: its first frame with its length and a play button; a click swaps in the player, playing.
struct VideoResult: View {
    let url: URL
    @MediaState private var playing: URL?
    @Environment(\.palette) private var p
    var body: some View {
        if playing == url {
            VideoPlayerView(url: url, autoplay: true)
        } else {
            ZStack {
                VideoFrame(url: url)
                Button {
                    playing = url
                } label: {
                    Image(systemName: "play.fill").font(.system(size: 20)).foregroundStyle(Color.white)
                        .frame(width: 54, height: 54).background(p.accent.swiftUI, in: Circle()).shadow(color: .black.opacity(0.35), radius: 3).contentShape(Circle())
                }.buttonStyle(.plain).accessibilityLabel("Play").controlID("play-video")
            }
            .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
        }
    }
}

/// A video that plays when asked: AppKit's inline controls.
struct VideoPlayerView: NSViewRepresentable {
    let url: URL
    var autoplay = false
    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        view.videoGravity = .resizeAspect
        view.player = AVPlayer(url: url)
        if SoundPlayer.silent { view.player?.volume = 0 }
        if autoplay { view.player?.play() }
        return view
    }
    func updateNSView(_ view: AVPlayerView, context: Context) {
        if (view.player?.currentItem?.asset as? AVURLAsset)?.url != url {
            view.player?.pause()
            view.player = AVPlayer(url: url)
            if SoundPlayer.silent { view.player?.volume = 0 }
        }
    }
    static func dismantleNSView(_ view: AVPlayerView, coordinator: ()) { view.player?.pause(); view.player = nil }
}

/// The result over its original, split by a handle: left of it the original, right of it the result. The handle moves
/// with the pointer (drag or click); at rest nothing redraws.
struct BeforeAfter: View {
    let before: URL
    let after: URL
    let afterSize: PixelSize?
    var beforeTitle = "Original"
    var afterTitle = "Result"
    @MediaState private var split: CGFloat = 0.5
    @Environment(\.palette) private var p

    var body: some View {
        GeometryReader { g in
            let pixels = afterSize.map { CGSize(width: $0.width, height: $0.height) } ?? Thumbnail.image(after)?.size ?? CGSize(width: 1, height: 1)
            let aspect = pixels.width > 0 && pixels.height > 0 ? pixels.width / pixels.height : 1
            let w = min(g.size.width, g.size.height * aspect), h = w / aspect
            ZStack(alignment: .topLeading) {
                Thumbnail(url: after, fit: true).frame(width: w, height: h)
                Thumbnail(url: before, fit: true).frame(width: w, height: h)
                    .mask(alignment: .leading) { Rectangle().frame(width: w * split) }
                Rectangle().fill(Color.white).frame(width: 2, height: h).shadow(color: .black.opacity(0.4), radius: 1).offset(x: w * split - 1)
                Circle().fill(Color.white).frame(width: 22, height: 22).shadow(color: .black.opacity(0.35), radius: 1.5)
                    .overlay(Image(systemName: "arrow.left.and.right").font(.system(size: 9, weight: .bold)).foregroundStyle(Color.black.opacity(0.7)))
                    .offset(x: w * split - 11, y: h / 2 - 11)
                corner(beforeTitle).padding(6)
                corner(afterTitle).frame(width: w - 12, alignment: .trailing).padding(6)
            }
            .frame(width: w, height: h)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { split = min(1, max(0, $0.location.x / max(1, w))) })
            .frame(width: g.size.width, height: g.size.height)
            .accessibilityElement()
            .accessibilityLabel("\(beforeTitle) and \(afterTitle) compared")
            .accessibilityValue("\(Int(split * 100)) percent \(beforeTitle.lowercased())")
            .accessibilityAdjustableAction { split = min(1, max(0, split + ($0 == .increment ? 0.1 : -0.1))) }
            .controlID("before-after")
        }
    }
    private func corner(_ title: String) -> some View {
        Text(title).font(.system(size: 11, weight: .medium)).foregroundStyle(Color.white).padding(.horizontal, 6).frame(height: 20)
            .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 5))
    }
}

// MARK: Depth

/// A depth map as the eye reads it: near warm, far cool (a perceptual ramp), or the file's own grey. Computed once per
/// file and kept.
@MainActor enum DepthImage {
    private static let cache = MediaCache<String, (image: NSImage, low: Double, high: Double)>(limit: 96 * MediaCaches.mb) { MediaCaches.bytes($0.image) }

    /// The map's values, its range, and the coloured or grey picture. `nearIsHigh`: larger values are nearer
    /// (relative disparity); metric depth is nearer when smaller.
    static func render(_ url: URL, colour: Bool, nearIsHigh: Bool) -> (image: NSImage, low: Double, high: Double)? {
        let key = "\(url.path)|\(colour)|\(nearIsHigh)"
        if let hit = cache[key] { return hit }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let w = cg.width, h = cg.height
        // Read as 16-bit grey whatever the file holds (8-bit greys widen, 16-bit keep their precision).
        var grey = [UInt16](repeating: 0, count: w * h)
        guard
            let context = CGContext(
                data: &grey, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 2, space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue)
        else { return nil }
        context.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var lo = UInt16.max, hi = UInt16.min
        for v in grey { lo = min(lo, v); hi = max(hi, v) }
        let span = Double(max(1, Int(hi) - Int(lo)))
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0..<(w * h) {
            var t = Double(Int(grey[i]) - Int(lo)) / span
            if !nearIsHigh { t = 1 - t }
            let (r, g, b) = colour ? ramp(t) : (t, t, t)
            rgba[i * 4] = UInt8(max(0, min(255, r * 255))); rgba[i * 4 + 1] = UInt8(max(0, min(255, g * 255))); rgba[i * 4 + 2] = UInt8(max(0, min(255, b * 255)))
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
            let out = CGImage(
                width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        else { return nil }
        // Near and far come from every pixel; the picture kept is display-sized (MediaCaches.displayImage).
        let shown = MediaCaches.displayCopy(out, fitting: MediaCaches.displayPixels)
        let result = (MediaCaches.picture(shown), Double(lo) / 65535, Double(hi) / 65535)
        cache.set(result, for: key)
        return result
    }

    /// Near (1) to far (0): warm yellow through orange and magenta to deep blue, even in lightness.
    static func ramp(_ t: Double) -> (Double, Double, Double) {
        let stops: [(Double, Double, Double)] = [(0.05, 0.03, 0.25), (0.33, 0.07, 0.5), (0.72, 0.15, 0.47), (0.96, 0.45, 0.22), (0.99, 0.86, 0.36)]
        let x = max(0, min(1, t)) * Double(stops.count - 1)
        let i = min(stops.count - 2, Int(x)), f = x - Double(i)
        let a = stops[i], b = stops[i + 1]
        return (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f, a.2 + (b.2 - a.2) * f)
    }
}

/// The ramp as a bar, near on the left.
struct DepthLegend: View {
    let near: String
    let far: String
    @Environment(\.palette) private var p
    var body: some View {
        HStack(spacing: 8) {
            Text(near).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t2.swiftUI)
            LinearGradient(
                colors: (0...8).map { i in
                    let c = DepthImage.ramp(1 - Double(i) / 8); return Color(red: c.0, green: c.1, blue: c.2)
                }, startPoint: .leading, endPoint: .trailing
            )
            .frame(height: 6).clipShape(Capsule())
            Text(far).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t2.swiftUI)
        }
    }
}
