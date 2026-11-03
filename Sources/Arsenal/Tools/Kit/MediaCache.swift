// SPDX-License-Identifier: MIT
import AppKit
import ImageIO

// What the tool windows decode once and keep (thumbnails, video frames, document pages, depth maps, 3D stills, mask
// alphas, envelopes): each cache is held to a byte budget, the least recently used entry going first, and every cache
// is emptied when the last tool window closes, so a closed window leaves no pictures behind in Arsenal's memory.

/// One kind of decoded media, by key, within `limit` bytes.
@MainActor final class MediaCache<Key: Hashable, Value> {
    private var items: [Key: (value: Value, bytes: Int)] = [:]
    /// Least recently used first.
    private var order: [Key] = []
    private(set) var bytes = 0
    let limit: Int
    private let cost: @MainActor (Value) -> Int

    init(limit: Int, cost: @escaping @MainActor (Value) -> Int) {
        self.limit = limit; self.cost = cost
        MediaCaches.register { [weak self] in self?.removeAll() }
    }

    var count: Int { items.count }

    subscript(key: Key) -> Value? {
        guard let hit = items[key] else { return nil }
        if order.last != key, let i = order.firstIndex(of: key) { order.remove(at: i); order.append(key) }
        return hit.value
    }

    /// Keeps `value`, dropping the least recently used entries until the cache is within its budget. A value larger
    /// than the whole budget is not kept (the caller still has it).
    func set(_ value: Value, for key: Key) {
        remove(key)
        let size = max(1, cost(value))
        guard size <= limit else { return }
        while bytes + size > limit, let oldest = order.first { remove(oldest) }
        items[key] = (value, size); order.append(key); bytes += size
    }

    func remove(_ key: Key) {
        guard let old = items.removeValue(forKey: key) else { return }
        bytes -= old.bytes
        if let i = order.firstIndex(of: key) { order.remove(at: i) }
    }

    func removeAll() { items = [:]; order = []; bytes = 0 }
}

/// Every tool-window cache, emptied together.
@MainActor enum MediaCaches {
    private static var clears: [@MainActor () -> Void] = []
    static func register(_ clear: @escaping @MainActor () -> Void) { clears.append(clear) }
    /// Empties every cache (the last tool window closed).
    static func clear() { for clear in clears { clear() } }

    /// What a decoded picture holds in memory: its largest bitmap at four bytes a pixel.
    nonisolated static func bytes(_ image: NSImage) -> Int {
        let pixels = image.representations.map { $0.pixelsWide * $0.pixelsHigh }.max() ?? 0
        return pixels > 0 ? pixels * 4 : Int(image.size.width * image.size.height) * 4
    }

    nonisolated static let mb = 1_000_000

    // MARK: Display copies

    /// The most pixels a tool window can show a picture at: the largest screen at its backing scale, since no window is
    /// larger. A copy fitted inside it looks the same as the whole picture. Tests may set `displayOverride`.
    static var displayPixels: CGSize {
        if let displayOverride { return displayOverride }
        let screens = NSScreen.screens.map { CGSize(width: $0.frame.width * $0.backingScaleFactor, height: $0.frame.height * $0.backingScaleFactor) }
        guard let w = screens.map(\.width).max(), let h = screens.map(\.height).max(), w > 0, h > 0 else { return fallbackDisplay }
        return CGSize(width: w, height: h)
    }
    static var displayOverride: CGSize?
    /// No screen (a headless run): a 16-inch Retina panel.
    nonisolated static let fallbackDisplay = CGSize(width: 3456, height: 2234)

    /// `size` scaled down to fit inside `bounds`, aspect kept, never scaled up; whole pixels, at least one.
    nonisolated static func fitted(_ size: CGSize, in bounds: CGSize) -> CGSize {
        guard size.width > 0, size.height > 0, bounds.width > 0, bounds.height > 0 else { return size }
        let scale = min(1, bounds.width / size.width, bounds.height / size.height)
        return CGSize(width: max(1, (size.width * scale).rounded()), height: max(1, (size.height * scale).rounded()))
    }

    /// An image file decoded no larger than it can appear inside `bounds` (upright, as PixelSize reads it), instead of
    /// its whole bitmap: a 24 MP photo is held at the screen's size, so two of them (before and after) fit their cache.
    /// Its size is the copy's own pixels: views draw it resizable, take only its aspect, and read the file's pixel size
    /// from PixelSize. Files ImageIO cannot read (vector formats NSImage draws) come back whole.
    nonisolated static func displayImage(_ url: URL, fitting bounds: CGSize) -> NSImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0, let full = PixelSize.read(url) else {
            return NSImage(contentsOf: url)
        }
        let shown = fitted(CGSize(width: full.width, height: full.height), in: bounds)
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true, kCGImageSourceThumbnailMaxPixelSize: Int(max(shown.width, shown.height))
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return NSImage(contentsOf: url) }
        return picture(cg)
    }

    /// A decoded picture as an NSImage whose one representation is its bitmap, so `bytes` counts its own pixels.
    /// (`NSImage(cgImage:size:)` reports pixels at the screen's backing scale: four times the cost on a Retina Mac.)
    nonisolated static func picture(_ cg: CGImage) -> NSImage {
        let image = NSImage(size: NSSize(width: cg.width, height: cg.height))
        image.addRepresentation(NSBitmapImageRep(cgImage: cg))
        return image
    }

    /// A picture already in memory, redrawn no larger than it can appear inside `bounds` (itself when it fits).
    nonisolated static func displayCopy(_ image: CGImage, fitting bounds: CGSize) -> CGImage {
        let shown = fitted(CGSize(width: image.width, height: image.height), in: bounds)
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        guard Int(shown.width) < image.width || Int(shown.height) < image.height,
            let context = CGContext(
                data: nil, width: Int(shown.width), height: Int(shown.height), bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return image }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(origin: .zero, size: shown))
        return context.makeImage() ?? image
    }
}
