// SPDX-License-Identifier: MIT
// Reference images for edits, prepared as the vendor pipeline and mflux do (Flux3ReferenceImages): EXIF orientation
// applied, RGB, Pillow's LANCZOS resample (reimplemented here bit for bit for 8-bit RGB: two passes, horizontal first,
// 22-bit fixed-point coefficients, an 8-bit intermediate), then a reflect-padded canvas (one reference) or a centred
// crop to multiples of 16 (several references). Like Pillow, colour values are taken as stored (no ICC conversion).
import CoreGraphics
import Foundation
import ImageIO

/// 8-bit RGB pixels, row-major.
public struct RGBImage: Equatable, Sendable {
    public var width: Int, height: Int
    public var pixels: [UInt8]
    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3)
        self.width = width; self.height = height; self.pixels = pixels
    }

    /// Decodes an image file as Pillow's `ImageOps.exif_transpose(Image.open(path)).convert("RGB")` does: the stored
    /// 8-bit colour values (16-bit samples keep their high byte), alpha dropped without compositing, grey repeated into
    /// RGB, no colour management, then the EXIF orientation applied. Palette, CMYK and other layouts are drawn into an
    /// RGB context instead (colour-managed: they may differ from Pillow by a few levels).
    public static func read(_ url: URL) throws -> RGBImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0,
            let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        else { throw Flux3Error("\(url.lastPathComponent) is not a readable image.") }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        guard let decoded = stored(image) ?? drawn(image) else { throw Flux3Error("\(url.lastPathComponent) could not be converted to RGB.") }
        return decoded.oriented(orientation)
    }

    /// The stored samples of an 8- or 16-bit RGB or grey image, with or without alpha; nil for any other layout.
    static func stored(_ image: CGImage) -> RGBImage? {
        guard let space = image.colorSpace, [.rgb, .monochrome].contains(space.model), let data = image.dataProvider?.data,
            !image.bitmapInfo.contains(.floatComponents), [8, 16].contains(image.bitsPerComponent)
        else { return nil }
        let colours = space.model == .rgb ? 3 : 1, alpha = image.alphaInfo
        let components = colours + (alpha == .none ? 0 : 1), bytes = image.bitsPerComponent / 8
        guard image.bitsPerPixel == components * bytes * 8, ![.alphaOnly].contains(alpha) else { return nil }
        let alphaFirst = [.first, .premultipliedFirst, .noneSkipFirst].contains(alpha)
        let premultiplied = [.premultipliedFirst, .premultipliedLast].contains(alpha)
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let reversed = bytes == 1 && components == 4 && order == .byteOrder32Little
        let little16 = bytes == 2 && order == .byteOrder16Little
        let (width, height, row) = (image.width, image.height, image.bytesPerRow)
        guard CFDataGetLength(data) >= row * (height - 1) + width * components * bytes else { return nil }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        let base = CFDataGetBytePtr(data)!
        for y in 0..<height {
            for x in 0..<width {
                let pixel = base + y * row + x * components * bytes
                func sample(_ i: Int) -> Int {
                    let j = reversed ? components - 1 - i : i
                    if bytes == 1 { return Int(pixel[j]) }
                    let hi = little16 ? pixel[2 * j + 1] : pixel[2 * j]
                    return Int(hi)
                }
                let first = alpha != .none && alphaFirst ? 1 : 0
                let a = [.first, .last, .premultipliedFirst, .premultipliedLast].contains(alpha) ? sample(alphaFirst ? 0 : colours) : 255
                for c in 0..<3 {
                    var v = sample(first + (colours == 3 ? c : 0))
                    if premultiplied { v = a == 0 ? 0 : Swift.min(255, (v * 255 + a / 2) / a) }
                    rgb[(y * width + x) * 3 + c] = UInt8(v)
                }
            }
        }
        return RGBImage(width: width, height: height, pixels: rgb)
    }

    /// Any other layout, drawn into an RGB context (colour-managed) and un-premultiplied.
    static func drawn(_ image: CGImage) -> RGBImage? {
        let (width, height) = (image.width, image.height)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let ok = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else { return nil }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            let a = Int(rgba[i * 4 + 3])
            for c in 0..<3 { rgb[i * 3 + c] = a == 0 ? 0 : UInt8(Swift.min(255, (Int(rgba[i * 4 + c]) * 255 + a / 2) / a)) }
        }
        return RGBImage(width: width, height: height, pixels: rgb)
    }

    /// Pillow `exif_transpose` for EXIF orientations 2-8 (1 and unknown values leave the image as stored).
    public func oriented(_ orientation: Int) -> RGBImage {
        guard (2...8).contains(orientation) else { return self }
        let swap = orientation >= 5
        let (w, h) = swap ? (height, width) : (width, height)
        var out = [UInt8](repeating: 0, count: pixels.count)
        for y in 0..<h {
            for x in 0..<w {
                let (sx, sy): (Int, Int)
                switch orientation {
                case 2: (sx, sy) = (width - 1 - x, y) // FLIP_LEFT_RIGHT
                case 3: (sx, sy) = (width - 1 - x, height - 1 - y) // ROTATE_180
                case 4: (sx, sy) = (x, height - 1 - y) // FLIP_TOP_BOTTOM
                case 5: (sx, sy) = (y, x) // TRANSPOSE
                case 6: (sx, sy) = (y, height - 1 - x) // ROTATE_270 (90° clockwise)
                case 7: (sx, sy) = (width - 1 - y, height - 1 - x) // TRANSVERSE
                default: (sx, sy) = (width - 1 - y, x) // 8: ROTATE_90 (90° counter-clockwise)
                }
                for c in 0..<3 { out[(y * w + x) * 3 + c] = pixels[(sy * width + sx) * 3 + c] }
            }
        }
        return RGBImage(width: w, height: h, pixels: out)
    }

    // MARK: Pillow LANCZOS

    static let precisionBits = 32 - 8 - 2

    static func lanczos(_ x: Double) -> Double {
        func sinc(_ v: Double) -> Double { v == 0 ? 1 : Foundation.sin(v * .pi) / (v * .pi) }
        return (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3) : 0
    }

    /// Pillow `precompute_coeffs` + `normalize_coeffs_8bpc`: per output pixel, (first input index, fixed-point taps).
    static func coefficients(input: Int, output: Int) -> [(start: Int, taps: [Int32])] {
        let scale = Double(input) / Double(output), filterScale = Swift.max(scale, 1)
        let support = 3.0 * filterScale, ss = 1.0 / filterScale
        return (0..<output).map { xx in
            let center = (Double(xx) + 0.5) * scale
            let xmin = Swift.max(Int(center - support + 0.5), 0)
            let xmax = Swift.min(Int(center + support + 0.5), input) - xmin
            var weights = (0..<Swift.max(xmax, 0)).map { lanczos((Double($0 + xmin) - center + 0.5) * ss) }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            let one = Double(1 << precisionBits)
            return (xmin, weights.map { $0 < 0 ? Int32(-0.5 + $0 * one) : Int32(0.5 + $0 * one) })
        }
    }

    static func clip8(_ v: Int64) -> UInt8 {
        let s = v >> Int64(precisionBits)
        return s < 0 ? 0 : s > 255 ? 255 : UInt8(s)
    }

    /// One Pillow resample pass over RGB8 along rows (`horizontal`) or columns, rounded to 8 bits.
    static func pass(_ src: [UInt8], width: Int, height: Int, to size: Int, horizontal: Bool) -> [UInt8] {
        let k = coefficients(input: horizontal ? width : height, output: size)
        let (ow, oh) = horizontal ? (size, height) : (width, size)
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        let half = Int64(1) << Int64(precisionBits - 1)
        src.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                for (i, (start, taps)) in k.enumerated() {
                    taps.withUnsafeBufferPointer { t in
                        for j in 0..<(horizontal ? oh : ow) {
                            let (x, y) = horizontal ? (i, j) : (j, i)
                            var acc = (half, half, half)
                            for n in 0..<t.count {
                                let p = (horizontal ? (y * width + start + n) : ((start + n) * width + x)) * 3, w = Int64(t[n])
                                acc.0 += Int64(s[p]) * w; acc.1 += Int64(s[p + 1]) * w; acc.2 += Int64(s[p + 2]) * w
                            }
                            let q = (y * ow + x) * 3
                            o[q] = clip8(acc.0); o[q + 1] = clip8(acc.1); o[q + 2] = clip8(acc.2)
                        }
                    }
                }
            }
        }
        return out
    }

    /// Pillow `Image.resize(size, LANCZOS)` for RGB8: horizontal pass first, then vertical, each rounded to 8 bits.
    public func resizedLanczos(width w: Int, height h: Int) -> RGBImage {
        if w == width && h == height { return self }
        var current = self
        if w != width { current = RGBImage(width: w, height: height, pixels: Self.pass(pixels, width: width, height: height, to: w, horizontal: true)) }
        if h != height {
            current = RGBImage(width: w, height: h, pixels: Self.pass(current.pixels, width: w, height: height, to: h, horizontal: false))
        }
        return current
    }

    /// numpy `np.pad(..., mode="reflect")` (or "symmetric" for a side of 16 pixels or fewer) to `width` × `height`,
    /// with the image at (left, top).
    public func padded(width w: Int, height h: Int, left: Int, top: Int) -> RGBImage {
        let symmetric = Swift.min(width, height) <= Flux3Canvas.ratio
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let sy = Flux3Canvas.padIndex(y - top, size: height, symmetric: symmetric)
            for x in 0..<w {
                let sx = Flux3Canvas.padIndex(x - left, size: width, symmetric: symmetric)
                for c in 0..<3 { out[(y * w + x) * 3 + c] = pixels[(sy * width + sx) * 3 + c] }
            }
        }
        return RGBImage(width: w, height: h, pixels: out)
    }

    public func cropped(x: Int, y: Int, width w: Int, height h: Int) -> RGBImage {
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for row in 0..<h {
            let from = ((y + row) * width + x) * 3
            out.replaceSubrange((row * w * 3)..<((row + 1) * w * 3), with: pixels[from..<(from + w * 3)])
        }
        return RGBImage(width: w, height: h, pixels: out)
    }
}

/// Prepared references: the images to encode, and for a single padded reference the box of the image inside it.
public struct Flux3References {
    public var images: [RGBImage]
    public var box: (x: Int, y: Int, width: Int, height: Int)?

    public static func prepare(_ sources: [RGBImage], cropToFit: Bool = true) throws -> Flux3References {
        guard !sources.isEmpty else { throw Flux3Error("An edit needs at least one reference image.") }
        guard sources.count <= Flux3Canvas.maxReferences else { throw Flux3Error("At most \(Flux3Canvas.maxReferences) reference images are supported.") }
        if sources.count > 1 {
            return Flux3References(
                images: try sources.map { image in
                    let (scaled, crop) = try Flux3Canvas.multi(width: image.width, height: image.height)
                    return image.resizedLanczos(width: scaled.width, height: scaled.height).cropped(x: crop.x, y: crop.y, width: crop.width, height: crop.height)
                }, box: nil)
        }
        let image = sources[0]
        let fit = Flux3Canvas.single(width: image.width, height: image.height)
        let resized = image.resizedLanczos(width: fit.fit.width, height: fit.fit.height)
        if fit.canvas == fit.fit { return Flux3References(images: [resized], box: nil) }
        let left = (fit.canvas.width - resized.width) / 2, top = (fit.canvas.height - resized.height) / 2
        return Flux3References(
            images: [resized.padded(width: fit.canvas.width, height: fit.canvas.height, left: left, top: top)],
            box: cropToFit ? (left, top, resized.width, resized.height) : nil)
    }
}
