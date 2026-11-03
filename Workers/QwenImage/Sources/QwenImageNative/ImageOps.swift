// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// Interleaved 8-bit sRGB RGBA, straight (not premultiplied) alpha: what PIL's `convert("RGBA")` holds.
public struct RGBAImage: Equatable, Sendable {
    public let width: Int, height: Int
    public var bytes: [UInt8]
    public init(width: Int, height: Int, bytes: [UInt8]) throws {
        guard width > 0, height > 0, bytes.count == width * height * 4 else { throw QwenImageError("Invalid RGBA image buffer") }
        self.width = width; self.height = height; self.bytes = bytes
    }
    /// Whether the image is meant to be transparent: at least 64 pixels under half opacity. The VAE's alpha for an opaque
    /// image is 249–255 rather than exactly 255 (measured on a 1024² generation), so near-opaque noise does not count;
    /// such an image is saved as RGB.
    public var transparent: Bool {
        var count = 0
        for i in stride(from: 3, to: bytes.count, by: 4) where bytes[i] < 128 {
            count += 1
            if count >= 64 { return true }
        }
        return false
    }
}

public enum ImageOps {
    static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    /// Decodes a PNG/JPEG into sRGB RGBA bytes, colour-managed and unpremultiplied (alpha kept, as PIL reads it).
    public static func read(_ url: URL, maxPixels: Int = 16_000_000) throws -> RGBAImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int,
            width > 0, height > 0, width <= maxPixels / height,
            let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        else { throw QwenImageError("The image is unreadable or larger than 16 megapixels.") }
        // Core Graphics draws only premultiplied RGBA; undo the premultiplication afterwards.
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let drawn = bytes.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4, space: sRGB,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return true
        }
        guard drawn else { throw QwenImageError("Could not decode the image.") }
        return try RGBAImage(width: image.width, height: image.height, bytes: unpremultiply(bytes))
    }

    /// PIL rgba2rgbA: c·a/255 with MULDIV255 rounding.
    static func premultiply(_ bytes: [UInt8]) -> [UInt8] {
        var out = bytes
        for i in stride(from: 0, to: bytes.count, by: 4) {
            let a = Int(bytes[i + 3])
            for c in 0..<3 {
                let t = Int(bytes[i + c]) * a + 128
                out[i + c] = UInt8((t + (t >> 8)) >> 8)
            }
        }
        return out
    }
    /// PIL rgbA2rgba: c·255/a, clipped; alpha 0 or 255 copies the colour.
    static func unpremultiply(_ bytes: [UInt8]) -> [UInt8] {
        var out = bytes
        for i in stride(from: 0, to: bytes.count, by: 4) {
            let a = Int(bytes[i + 3])
            guard a != 0, a != 255 else { continue }
            for c in 0..<3 { out[i + c] = UInt8(Swift.min(255, Int(bytes[i + c]) * 255 / a)) }
        }
        return out
    }

    /// `Image.resize((w, h), LANCZOS)` for an RGBA image: Pillow resamples premultiplied ("RGBa") and converts back.
    public static func resize(_ image: RGBAImage, width: Int, height: Int) throws -> RGBAImage {
        guard width != image.width || height != image.height else { return image }
        var pixels = premultiply(image.bytes), w = image.width, h = image.height
        if width != w { pixels = PillowResample.horizontal(pixels, width: w, height: h, to: width); w = width }
        if height != h { pixels = PillowResample.vertical(pixels, width: w, height: h, to: height); h = height }
        return try RGBAImage(width: w, height: h, bytes: unpremultiply(pixels))
    }

    /// PIL `white.paste(img, mask=alpha)`: the prompt encoder sees the image composited over white.
    public static func overWhite(_ image: RGBAImage) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: image.width * image.height * 3)
        for p in 0..<(image.width * image.height) {
            let a = Int(image.bytes[p * 4 + 3])
            for c in 0..<3 {
                // paste blends 8-bit: out = (src·a + dst·(255−a)) / 255, rounded (PIL's BLEND with MULDIV255).
                let t = Int(image.bytes[p * 4 + c]) * a + 255 * (255 - a) + 128
                rgb[p * 3 + c] = UInt8((t + (t >> 8)) >> 8)
            }
        }
        return rgb
    }

    /// VaeImageProcessor.preprocess: bytes/255 → [-1, 1], [1, H, W, 4] float32.
    public static func vaeInput(_ image: RGBAImage) -> MLXArray {
        MLXArray(image.bytes, [1, image.height, image.width, 4]).asType(.float32) / 127.5 - 1
    }

    /// VaeImageProcessor.postprocess: [1, H, W, 4] in [-1, 1] → RGBA bytes (×255, round half to even like numpy).
    public static func bytes(_ decoded: MLXArray) throws -> RGBAImage {
        let image = decoded[0]
        let (h, w) = (image.dim(0), image.dim(1))
        let values = clip(image.asType(.float32) / 2 + 0.5, min: 0, max: 1) * 255
        let rounded = MLX.round(values).asType(.uint8)
        return try RGBAImage(width: w, height: h, bytes: rounded.asArray(UInt8.self))
    }

    /// PNG: RGBA for a transparent image (`RGBAImage.transparent`), otherwise RGB.
    public static func writePNG(_ image: RGBAImage, to url: URL) throws {
        let alpha = image.transparent
        let channels = alpha ? 4 : 3
        var data = [UInt8](); data.reserveCapacity(image.width * image.height * channels)
        if alpha {
            data = image.bytes
        } else {
            for i in stride(from: 0, to: image.bytes.count, by: 4) { data.append(contentsOf: image.bytes[i..<(i + 3)]) }
        }
        let info: CGBitmapInfo = alpha ? CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue) : CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard let provider = CGDataProvider(data: Data(data) as CFData),
            let cg = CGImage(
                width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 8 * channels, bytesPerRow: image.width * channels, space: sRGB,
                bitmapInfo: info, provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw QwenImageError("Could not encode the PNG.") }
        CGImageDestinationAddImage(destination, cg, nil)
        guard CGImageDestinationFinalize(destination) else { throw QwenImageError("Could not write the PNG.") }
    }
}

/// PORTED-FROM: Pillow 12 `src/libImaging/Resample.c` (precompute_coeffs, normalize_coeffs_8bpc, 8bpc passes), the
/// antialiased Lanczos (a = 3) behind `Image.resize`; reimplemented for four interleaved channels.
enum PillowResample {
    static let precisionBits = 22
    static func sinc(_ x: Double) -> Double { x == 0 ? 1 : sin(x * .pi) / (x * .pi) }
    static func lanczos(_ x: Double) -> Double { x >= -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0 }

    struct Plan { let start: [Int], count: [Int], kernel: Int, weights: [Int] }
    static func plan(inSize: Int, outSize: Int) -> Plan {
        let scale = Double(inSize) / Double(outSize), filterScale = Swift.max(scale, 1), support = 3 * filterScale
        let kernel = Int(support.rounded(.up)) * 2 + 1
        var start = [Int](repeating: 0, count: outSize), count = [Int](repeating: 0, count: outSize)
        var weights = [Int](repeating: 0, count: outSize * kernel)
        for o in 0..<outSize {
            let center = (Double(o) + 0.5) * scale
            let first = Swift.max(0, Int(center - support + 0.5))
            let taps = Swift.max(0, Swift.min(inSize, Int(center + support + 0.5)) - first)
            var raw = [Double](repeating: 0, count: taps), total = 0.0
            for t in 0..<taps { raw[t] = lanczos((Double(t + first) - center + 0.5) / filterScale); total += raw[t] }
            for t in 0..<taps {
                let value = total != 0 ? raw[t] / total : raw[t], scaled = value * Double(1 << precisionBits)
                weights[o * kernel + t] = value < 0 ? Int(scaled - 0.5) : Int(scaled + 0.5)
            }
            start[o] = first; count[o] = taps
        }
        return Plan(start: start, count: count, kernel: kernel, weights: weights)
    }
    static func clip8(_ v: Int) -> UInt8 { UInt8(clamping: v >> precisionBits) }

    static func horizontal(_ src: [UInt8], width: Int, height: Int, to out: Int) -> [UInt8] {
        let p = plan(inSize: width, outSize: out), rounding = 1 << (precisionBits - 1)
        var dst = [UInt8](repeating: 0, count: out * height * 4)
        src.withUnsafeBufferPointer { s in
            dst.withUnsafeMutableBufferPointer { d in
                let d = d
                DispatchQueue.concurrentPerform(iterations: height) { y in
                    for x in 0..<out {
                        var acc = [rounding, rounding, rounding, rounding]
                        for t in 0..<p.count[x] {
                            let k = p.weights[x * p.kernel + t], o = (y * width + p.start[x] + t) * 4
                            for c in 0..<4 { acc[c] += Int(s[o + c]) * k }
                        }
                        for c in 0..<4 { d[(y * out + x) * 4 + c] = clip8(acc[c]) }
                    }
                }
            }
        }
        return dst
    }
    static func vertical(_ src: [UInt8], width: Int, height: Int, to out: Int) -> [UInt8] {
        let p = plan(inSize: height, outSize: out), rounding = 1 << (precisionBits - 1)
        var dst = [UInt8](repeating: 0, count: width * out * 4)
        src.withUnsafeBufferPointer { s in
            dst.withUnsafeMutableBufferPointer { d in
                let d = d
                DispatchQueue.concurrentPerform(iterations: out) { y in
                    for x in 0..<width {
                        var acc = [rounding, rounding, rounding, rounding]
                        for t in 0..<p.count[y] {
                            let k = p.weights[y * p.kernel + t], o = ((p.start[y] + t) * width + x) * 4
                            for c in 0..<4 { acc[c] += Int(s[o + c]) * k }
                        }
                        for c in 0..<4 { d[(y * width + x) * 4 + c] = clip8(acc[c]) }
                    }
                }
            }
        }
        return dst
    }
}
