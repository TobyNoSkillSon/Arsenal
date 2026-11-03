// SPDX-License-Identifier: MIT
// Image decoding and Pillow-exact resampling, so preprocessing matches the maker's processor (PIL Image.resize
// on 8-bit pixels: separable filter, horizontal pass then vertical, 22-bit fixed-point weights, rounding per pass).
import CoreGraphics
import Foundation
import ImageIO

struct Pixels {
    /// Interleaved 8-bit samples, row-major, `channels` per pixel.
    var data: [UInt8]
    var width: Int
    var height: Int
    var channels: Int

    /// Decodes any ImageIO image at its own size as 8-bit sRGB RGB (no resampling; transparent pixels over black).
    static func load(_ url: URL) throws -> Pixels {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RunnerError.invalid("\(url.lastPathComponent) is not a readable image.")
        }
        let w = image.width, h = image.height
        guard w > 0, h > 0, w * h <= 100_000_000 else { throw RunnerError.invalid("\(url.lastPathComponent) has an unsupported size \(w)x\(h).") }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let context = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw RunnerError.failed("Could not draw the image.") }
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0 ..< (w * h) {
            rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return Pixels(data: rgb, width: w, height: h, channels: 3)
    }

    /// PIL convert("L") then convert("RGB"): ITU-R 601-2 luma in 16-bit fixed point, copied to three channels.
    func grayRGB() -> Pixels {
        var out = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0 ..< (width * height) {
            let r = Int(data[i * 3]), g = Int(data[i * 3 + 1]), b = Int(data[i * 3 + 2])
            let l = UInt8((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16)
            out[i * 3] = l; out[i * 3 + 1] = l; out[i * 3 + 2] = l
        }
        return Pixels(data: out, width: width, height: height, channels: 3)
    }

    enum Filter {
        case bilinear, bicubic
        var support: Double { self == .bilinear ? 1 : 2 }
        func callAsFunction(_ value: Double) -> Double {
            let x = abs(value)
            switch self {
            case .bilinear: return x < 1 ? 1 - x : 0
            case .bicubic:
                let a = -0.5
                if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
                return 0
            }
        }
    }

    private static let precisionBits = 32 - 8 - 2

    /// Pillow's precompute_coeffs + normalize_coeffs_8bpc for one axis.
    private static func coefficients(inSize: Int, outSize: Int, filter: Filter) -> (bounds: [(Int, Int)], kernel: [Int32], ksize: Int) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = filter.support * filterScale
        let ksize = Int(support.rounded(.up)) * 2 + 1
        var bounds: [(Int, Int)] = []
        var kernel = [Int32](repeating: 0, count: outSize * ksize)
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            let ss = 1 / filterScale
            var xmin = Int(center - support + 0.5)  // C truncation toward zero
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var weights = [Double](repeating: 0, count: xmax)
            var total = 0.0
            for x in 0 ..< xmax {
                let w = filter((Double(x + xmin) - center + 0.5) * ss)
                weights[x] = w
                total += w
            }
            for x in 0 ..< xmax {
                let w = total != 0 ? weights[x] / total : weights[x]
                let scaled = w * Double(1 << precisionBits)
                kernel[xx * ksize + x] = Int32(w < 0 ? Int(-0.5 + scaled) : Int(0.5 + scaled))
            }
            bounds.append((xmin, xmax))
        }
        return (bounds, kernel, ksize)
    }

    @inline(__always) private static func clip8(_ value: Int) -> UInt8 {
        if value >= ((1 << precisionBits) << 8) { return 255 }
        if value <= 0 { return 0 }
        return UInt8(value >> precisionBits)
    }

    /// PIL Image.resize((width, height), resample) on 8-bit channels.
    func resized(width outW: Int, height outH: Int, filter: Filter) -> Pixels {
        var current = self
        let c = channels
        if outW != width {
            let (bounds, kernel, ksize) = Self.coefficients(inSize: width, outSize: outW, filter: filter)
            var out = [UInt8](repeating: 0, count: outW * height * c)
            current.data.withUnsafeBufferPointer { src in
                for y in 0 ..< height {
                    for xx in 0 ..< outW {
                        let (xmin, xmax) = bounds[xx]
                        for ch in 0 ..< c {
                            var ss = 1 << (Self.precisionBits - 1)
                            for x in 0 ..< xmax {
                                ss += Int(src[(y * width + x + xmin) * c + ch]) * Int(kernel[xx * ksize + x])
                            }
                            out[(y * outW + xx) * c + ch] = Self.clip8(ss)
                        }
                    }
                }
            }
            current = Pixels(data: out, width: outW, height: height, channels: c)
        }
        if outH != height {
            let (bounds, kernel, ksize) = Self.coefficients(inSize: height, outSize: outH, filter: filter)
            let w = current.width
            var out = [UInt8](repeating: 0, count: w * outH * c)
            current.data.withUnsafeBufferPointer { src in
                for yy in 0 ..< outH {
                    let (ymin, ymax) = bounds[yy]
                    for x in 0 ..< w {
                        for ch in 0 ..< c {
                            var ss = 1 << (Self.precisionBits - 1)
                            for y in 0 ..< ymax {
                                ss += Int(src[((y + ymin) * w + x) * c + ch]) * Int(kernel[yy * ksize + y])
                            }
                            out[(yy * w + x) * c + ch] = Self.clip8(ss)
                        }
                    }
                }
            }
            current = Pixels(data: out, width: w, height: outH, channels: c)
        }
        return current
    }

    /// [1, H, W, C] float32: value / 255, then (x - mean) / std per channel.
    func normalized(mean: [Float], std: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: data.count)
        for i in 0 ..< data.count {
            let ch = i % channels
            // The maker rescales in float32 (x * (1/255)) and then normalizes.
            let v = Float(data[i]) * Float(1.0 / 255.0)
            out[i] = (v - mean[ch]) / std[ch]
        }
        return out
    }
}
