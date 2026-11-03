// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import IntegrationKit

/// 8-bit pixels as the providers' scripts read them (cv2.imread / PIL convert): the file's own values with no
/// colour-space conversion, alpha dropped (not blended), grey replicated to RGB. EXIF orientation is applied, as
/// cv2.imread does (LaMa's PIL loader would not; an upright photo is what a user means).
enum PhotoIO {
    struct Pixels {
        var bytes: [UInt8]; let width: Int; let height: Int; let colorSpace: CGColorSpace
        /// The file's EXIF orientation (1 when untagged), already applied to `bytes`.
        var orientation = 1
    }
    static let maxPixels = 16_000_000

    /// Raw channels (1 grey or 3 RGB) in stored order, the colour space, and the EXIF orientation (nil when the file has none).
    private static func decode(_ url: URL) throws -> (bytes: [UInt8], channels: Int, width: Int, height: Int, space: CGColorSpace, orientation: Int?) {
        _ = try NativeMedia.image(url, maxPixels: maxPixels) // bounds and validity first
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        else { throw MediaBridge.error("Cannot decode the image.") }
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (props?[kCGImagePropertyOrientation] as? Int).flatMap { (1...8).contains($0) ? $0 : nil }
        let grey = image.colorSpace?.model == .monochrome
        let space = grey ? CGColorSpace(name: CGColorSpace.sRGB)! : image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        if let raw = rawChannels(image, grey: grey) { return (raw, grey ? 1 : 3, image.width, image.height, space, orientation) }
        // Other layouts (16-bit, float, CMYK, indexed): drawn into 8-bit RGB, where any alpha is composited.
        let w = image.width, h = image.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ok = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let ctx = CGContext(
                    data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            ctx.interpolationQuality = .none
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h)); return true
        }
        guard ok else { throw MediaBridge.error("Cannot read the image's pixels.") }
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) { out[3 * i] = rgba[4 * i]; out[3 * i + 1] = rgba[4 * i + 1]; out[3 * i + 2] = rgba[4 * i + 2] }
        return (out, 3, w, h, space, orientation)
    }

    /// The decoder's own 8-bit samples (grey, grey+alpha, RGB, RGBA/ARGB in either alpha position), alpha dropped.
    static func rawChannels(_ image: CGImage, grey: Bool) -> [UInt8]? {
        let info = image.bitmapInfo
        guard image.bitsPerComponent == 8, !info.contains(.floatComponents),
            [CGBitmapInfo.byteOrderDefault, .byteOrder32Big].contains(CGBitmapInfo(rawValue: info.rawValue & CGBitmapInfo.byteOrderMask.rawValue)),
            let data = image.dataProvider?.data as Data?
        else { return nil }
        let color = grey ? 1 : 3, stride = image.bitsPerPixel / 8
        let alpha = image.alphaInfo
        let first = [.first, .premultipliedFirst, .noneSkipFirst].contains(alpha)
        let hasExtra = alpha != .none && alpha != .alphaOnly
        guard stride == color + (hasExtra ? 1 : 0), image.bytesPerRow >= image.width * stride, data.count >= image.bytesPerRow * (image.height - 1) + image.width * stride
        else { return nil }
        let offset = first && hasExtra ? 1 : 0
        var out = [UInt8](repeating: 0, count: image.width * image.height * color)
        data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            for y in 0..<image.height {
                let row = y * image.bytesPerRow
                for x in 0..<image.width {
                    let p = row + x * stride + offset, o = (y * image.width + x) * color
                    for c in 0..<color { out[o + c] = bytes[p + c] }
                }
            }
        }
        return out
    }

    /// EXIF orientation 1...8 applied to `channels`-interleaved pixels.
    static func orient(_ src: [UInt8], channels: Int, width: Int, height: Int, orientation: Int) -> (bytes: [UInt8], width: Int, height: Int) {
        guard orientation != 1 else { return (src, width, height) }
        let swap = orientation >= 5
        let ow = swap ? height : width, oh = swap ? width : height
        var out = [UInt8](repeating: 0, count: src.count)
        for y in 0..<oh {
            for x in 0..<ow {
                // Source coordinate of output pixel (x, y) for each EXIF orientation.
                let (sx, sy): (Int, Int)
                switch orientation {
                case 2: (sx, sy) = (width - 1 - x, y)
                case 3: (sx, sy) = (width - 1 - x, height - 1 - y)
                case 4: (sx, sy) = (x, height - 1 - y)
                case 5: (sx, sy) = (y, x)
                case 6: (sx, sy) = (y, height - 1 - x)
                case 7: (sx, sy) = (width - 1 - y, height - 1 - x)
                default: (sx, sy) = (width - 1 - y, x)
                }
                let o = (y * ow + x) * channels, i = (sy * width + sx) * channels
                for c in 0..<channels { out[o + c] = src[i + c] }
            }
        }
        return (out, ow, oh)
    }

    /// RGB bytes in the image's own RGB space (sRGB for grey, which is replicated).
    static func rgb(_ url: URL) throws -> Pixels {
        let d = try decode(url)
        var bytes = d.bytes
        if d.channels == 1 {
            bytes = [UInt8](repeating: 0, count: d.width * d.height * 3)
            for i in 0..<(d.width * d.height) { bytes[3 * i] = d.bytes[i]; bytes[3 * i + 1] = d.bytes[i]; bytes[3 * i + 2] = d.bytes[i] }
        }
        let o = orient(bytes, channels: 3, width: d.width, height: d.height, orientation: d.orientation ?? 1)
        return Pixels(bytes: o.bytes, width: o.width, height: o.height, colorSpace: d.space, orientation: d.orientation ?? 1)
    }

    /// One 8-bit channel: grey images as stored; colour images by PIL's "L" conversion (ITU-R 601-2 luma).
    /// A mask with no EXIF orientation of its own takes `photo`'s, so a mask drawn on the photo's stored pixels lines up
    /// with the upright photo; an untagged mask already drawn upright (it has the upright size, not the stored one) stays as is.
    static func mask(_ url: URL, photo: Pixels? = nil) throws -> (bytes: [UInt8], width: Int, height: Int) {
        let d = try decode(url)
        var grey = d.bytes
        if d.channels == 3 {
            grey = (0..<(d.width * d.height)).map { i in
                UInt8((Int(d.bytes[3 * i]) * 19595 + Int(d.bytes[3 * i + 1]) * 38470 + Int(d.bytes[3 * i + 2]) * 7471 + 0x8000) >> 16)
            }
        }
        var orientation = d.orientation ?? 1
        if d.orientation == nil, let photo, photo.orientation != 1 {
            let swap = photo.orientation >= 5
            if d.width == (swap ? photo.height : photo.width) && d.height == (swap ? photo.width : photo.height) { orientation = photo.orientation }
        }
        let o = orient(grey, channels: 1, width: d.width, height: d.height, orientation: orientation)
        return (o.bytes, o.width, o.height)
    }

    /// An opaque RGB PNG tagged with `space`.
    static func png(_ rgb: [UInt8], width: Int, height: Int, space: CGColorSpace, at url: URL) throws {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) { rgba[4 * i] = rgb[3 * i]; rgba[4 * i + 1] = rgb[3 * i + 1]; rgba[4 * i + 2] = rgb[3 * i + 2] }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
            let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent),
            let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw MediaBridge.error("Cannot encode PNG.") }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw MediaBridge.error("PNG encoding failed.") }
    }
}
