// SPDX-License-Identifier: MIT
// The maker's resize: transformers Sam2ImageProcessorFast resizes 8-bit RGB to 1024x1024 with torchvision's antialiased
// bilinear, the same triangle filter as PIL's Image.resize(BILINEAR). Copied from the CLIPSeg runner.
import CoreGraphics
import Foundation
import ImageIO

/// The image's 8-bit RGB values as stored, like PIL's Image.convert("RGB"): an RGB image is drawn in its own colour
/// space, so a Display P3 photo keeps its raw values instead of being converted to sRGB. Other colour models go to sRGB.
func decodeRGB(_ url: URL, maxPixels: Int = 64_000_000) throws -> (bytes: [UInt8], width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw RunnerError.invalid("\(url.lastPathComponent) is not a readable image.")
    }
    let w = image.width, h = image.height
    guard w * h <= maxPixels else { throw RunnerError.invalid("The image is \(w)x\(h); at most \(maxPixels / 1_000_000) megapixels.") }
    let own = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let context = CGContext(
        data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
        space: own ?? CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw RunnerError.failed("Could not draw the image.") }
    context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) {
        rgb[3 * i] = rgba[4 * i]
        rgb[3 * i + 1] = rgba[4 * i + 1]
        rgb[3 * i + 2] = rgba[4 * i + 2]
    }
    return (rgb, w, h)
}

/// PIL's Image.resize(BILINEAR) on 8-bit RGB: triangle filter widened by the scale when shrinking, fixed-point
/// coefficients (22 bits), horizontal pass then vertical pass, each rounded to 8 bits (libImaging/Resample.c).
func pilBilinear(_ src: [UInt8], width: Int, height: Int, outWidth: Int, outHeight: Int) -> [UInt8] {
    let precision = 22
    func coefficients(_ inSize: Int, _ outSize: Int) -> [(start: Int, weights: [Int])] {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1), support = filterScale
        return (0..<outSize).map { xx in
            let center = (Double(xx) + 0.5) * scale
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), inSize) - xmin
            var k = (0..<xmax).map { x -> Double in
                let t = abs((Double(x + xmin) - center + 0.5) / filterScale)
                return t < 1 ? 1 - t : 0
            }
            let total = k.reduce(0, +)
            if total != 0 { k = k.map { $0 / total } }
            return (xmin, k.map { $0 < 0 ? Int(-0.5 + $0 * Double(1 << precision)) : Int(0.5 + $0 * Double(1 << precision)) })
        }
    }
    func clip8(_ v: Int) -> UInt8 { let s = v >> precision; return UInt8(s < 0 ? 0 : (s > 255 ? 255 : s)) }
    var horizontal = src
    var midWidth = width
    if outWidth != width {
        let k = coefficients(width, outWidth)
        horizontal = [UInt8](repeating: 0, count: height * outWidth * 3)
        for y in 0..<height {
            for (x, c) in k.enumerated() {
                for ch in 0..<3 {
                    var sum = 1 << (precision - 1)
                    for (i, w) in c.weights.enumerated() { sum += Int(src[(y * width + c.start + i) * 3 + ch]) * w }
                    horizontal[(y * outWidth + x) * 3 + ch] = clip8(sum)
                }
            }
        }
        midWidth = outWidth
    }
    guard outHeight != height else { return horizontal }
    let k = coefficients(height, outHeight)
    var out = [UInt8](repeating: 0, count: outHeight * midWidth * 3)
    for (y, c) in k.enumerated() {
        for x in 0..<midWidth {
            for ch in 0..<3 {
                var sum = 1 << (precision - 1)
                for (i, w) in c.weights.enumerated() { sum += Int(horizontal[((c.start + i) * midWidth + x) * 3 + ch]) * w }
                out[(y * midWidth + x) * 3 + ch] = clip8(sum)
            }
        }
    }
    return out
}
