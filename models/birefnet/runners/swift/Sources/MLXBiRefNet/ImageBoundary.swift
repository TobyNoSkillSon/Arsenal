// SPDX-License-Identifier: MIT
// Pillow-compatible 8-bit resampling. Geometry, coefficient normalization and rounding follow
// Pillow 12.3.0 src/libImaging/Resample.c (LICENSE-Pillow). No model computation runs here.
import CoreGraphics
import Foundation

enum ImageBoundary {
    enum Filter { case bilinear, bicubic }
    private struct Coefficients { let start: Int; let weights: [Int64] }
    private static func coefficients(input: Int, output: Int, filter: Filter) -> [Coefficients] {
        let scale = Double(input) / Double(output)
        let filterScale = max(1, scale)
        let support = (filter == .bilinear ? 1.0 : 2.0) * filterScale
        return (0..<output).map { i in
            let center = (Double(i) + 0.5) * scale
            let start = max(0, Int(center - support + 0.5))
            let end = min(input, Int(center + support + 0.5))
            let values = (start..<end).map { j -> Double in
                let x = abs((Double(j) - center + 0.5) / filterScale)
                if filter == .bilinear { return max(0, 1 - x) }
                if x < 1 { return ((1.5 * x - 2.5) * x) * x + 1 }
                if x < 2 { return (((x - 5) * x + 8) * x - 4) * -0.5 }
                return 0
            }
            let sum = values.reduce(0, +)
            return Coefficients(start: start, weights: values.map {
                let w = sum == 0 ? $0 : $0 / sum
                return Int64((w < 0 ? -0.5 : 0.5) + w * Double(1 << 22))
            })
        }
    }
    static func resize(_ bytes: [UInt8], width: Int, height: Int, channels: Int,
                       targetWidth: Int, targetHeight: Int, filter: Filter) -> [UInt8] {
        precondition(bytes.count == width * height * channels && targetWidth > 0 && targetHeight > 0)
        func byte(_ sum: Int64) -> UInt8 { UInt8(clamping: sum >> 22) }
        var horizontal = bytes
        if width != targetWidth {
            let weights = coefficients(input: width, output: targetWidth, filter: filter)
            horizontal = [UInt8](repeating: 0, count: targetWidth * height * channels)
            for y in 0..<height {
                for x in 0..<targetWidth {
                    let c = weights[x]
                    for channel in 0..<channels {
                        var sum: Int64 = 1 << 21
                        for (i, weight) in c.weights.enumerated() {
                            sum += Int64(bytes[(y * width + c.start + i) * channels + channel]) * weight
                        }
                        horizontal[(y * targetWidth + x) * channels + channel] = byte(sum)
                    }
                }
            }
        }
        if height == targetHeight { return horizontal }
        let weights = coefficients(input: height, output: targetHeight, filter: filter)
        var result = [UInt8](repeating: 0, count: targetWidth * targetHeight * channels)
        for y in 0..<targetHeight {
            let c = weights[y]
            for x in 0..<targetWidth {
                for channel in 0..<channels {
                    var sum: Int64 = 1 << 21
                    for (i, weight) in c.weights.enumerated() {
                        sum += Int64(horizontal[((c.start + i) * targetWidth + x) * channels + channel]) * weight
                    }
                    result[(y * targetWidth + x) * channels + channel] = byte(sum)
                }
            }
        }
        return result
    }

    /// ImageIO's 8-bit RGB/RGBA decoder exposes straight colors. Read those before any draw can
    /// premultiply them, including the RGB hidden under alpha zero (Pillow convert("RGB") behavior).
    static func straightRGB(_ image: CGImage) -> [UInt8]? {
        let components = image.bitsPerPixel / 8
        let rgb = image.colorSpace?.model == .rgb
        let gray = image.colorSpace?.model == .monochrome
        if image.bitsPerComponent == 8, (rgb && [3,4].contains(components)) || (gray && [1,2].contains(components)),
           let raw = image.dataProvider?.data {
            let data = raw as Data
            guard data.count >= image.bytesPerRow * image.height else { return nil }
            let first = [.first, .premultipliedFirst, .noneSkipFirst].contains(image.alphaInfo)
            let premultiplied = [.premultipliedFirst, .premultipliedLast].contains(image.alphaInfo)
            let little = image.bitmapInfo.intersection(.byteOrderMask) == .byteOrder32Little && components == 4
            var result = [UInt8](repeating: 0, count: image.width * image.height * 3)
            for y in 0..<image.height {
                for x in 0..<image.width {
                    let base = y * image.bytesPerRow + x * components
                    func component(_ c: Int) -> UInt8 { data[base + (little ? components - 1 - c : c)] }
                    let alpha = components == (rgb ? 4 : 2) ? Int(component(first ? 0 : components - 1)) : 255
                    for channel in 0..<3 {
                        let value = Int(component((first ? 1 : 0) + (rgb ? channel : 0)))
                        result[(y * image.width + x) * 3 + channel] = premultiplied
                            ? UInt8(clamping: alpha == 0 ? 0 : (value * 255 + alpha / 2) / alpha) : UInt8(value)
                    }
                }
            }
            return result
        }
        // Color-managed fallback for other bit depths/color spaces. Unpremultiply before resizing.
        guard let context = CGContext(data: nil, width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.setBlendMode(.copy)
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let converted = context.makeImage() else { return nil }
        return straightRGB(converted)
    }
}
