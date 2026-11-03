// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import MLX

/// Interleaved 8-bit RGB in sRGB: the pixel basis that the makers' PIL/torchvision processors read
/// (`Image.open(...).convert("RGB")`, uint8 tensors).
struct RGBImage {
    let width: Int
    let height: Int
    var bytes: [UInt8]
    init(width: Int, height: Int, bytes: [UInt8]) {
        precondition(width > 0 && height > 0 && bytes.count == width * height * 3)
        self.width = width; self.height = height; self.bytes = bytes
    }
}

enum ImagePixels {
    private static let context = CIContext(options: [.cacheIntermediates: false])
    static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    /// Colour-managed render into sRGB bytes. Core Image decodes tagged files (Display P3 screenshots, Adobe RGB photos)
    /// into its linear working space, so an untagged render (`colorSpace: nil`, as the pinned Qwen3-VL image path does
    /// without `toSRGB()`) yields linear light; rendering into sRGB restores the encoded values PIL reads. Alpha is
    /// dropped without compositing, like PIL's convert("RGB"); fully transparent pixels have no recoverable colour.
    static func rgb(_ image: CIImage) -> RGBImage {
        let extent = image.extent
        let width = max(1, Int(extent.width.rounded())), height = max(1, Int(extent.height.rounded()))
        let bounds = CGRect(x: extent.minX, y: extent.minY, width: CGFloat(width), height: CGFloat(height))
        let opaque = image.unpremultiplyingAlpha().settingAlphaOne(in: bounds)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        rgba.withUnsafeMutableBytes { buffer in
            context.render(opaque, toBitmap: buffer.baseAddress!, rowBytes: width * 4, bounds: bounds, format: .RGBA8, colorSpace: sRGB)
        }
        context.clearCaches()
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for index in 0..<(width * height) {
            rgb[index * 3] = rgba[index * 4]; rgb[index * 3 + 1] = rgba[index * 4 + 1]; rgb[index * 3 + 2] = rgba[index * 4 + 2]
        }
        return RGBImage(width: width, height: height, bytes: rgb)
    }

    /// An opaque sRGB-tagged image at the origin; rendering it back through `rgb` returns the same bytes.
    static func ciImage(_ image: RGBImage) -> CIImage {
        var rgba = [UInt8](repeating: 255, count: image.width * image.height * 4)
        for index in 0..<(image.width * image.height) {
            rgba[index * 4] = image.bytes[index * 3]; rgba[index * 4 + 1] = image.bytes[index * 3 + 1]; rgba[index * 4 + 2] = image.bytes[index * 3 + 2]
        }
        return CIImage(
            bitmapData: Data(rgba), bytesPerRow: image.width * 4, size: CGSize(width: image.width, height: image.height), format: .RGBA8, colorSpace: sRGB)
    }

    /// HF rescale-and-normalize: (byte / 255 − mean) / std in float32, as `[1, 3, H, W]`.
    static func normalized(_ image: RGBImage, mean: [Float], std: [Float]) -> MLXArray {
        let raw = MLXArray(image.bytes, [image.height, image.width, 3]).asType(.float32) / Float(255)
        let values = (raw - MLXArray(mean)) / MLXArray(std)
        return values.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }
}

/// PORTED-FROM: Pillow 12 `src/libImaging/Resample.c` (precompute_coeffs, normalize_coeffs_8bpc,
/// ImagingResampleHorizontal_8bpc/Vertical_8bpc, ImagingResampleInner) — the antialiased resampler behind
/// `Image.resize`. MIT-CMU (HPND) licensed; reimplemented, no code vendored. Downscales widen the filter support by the
/// scale factor (antialiasing), which Core Image's bicubic/Lanczos scale transforms do not do.
enum PILResample {
    enum Filter {
        case bilinear, bicubic, lanczos

        var support: Double {
            switch self {
            case .bilinear: return 1
            case .bicubic: return 2
            case .lanczos: return 3
            }
        }

        func weight(_ x: Double) -> Double {
            switch self {
            case .bilinear:
                let x = abs(x)
                return x < 1 ? 1 - x : 0
            case .bicubic:
                // Keys cubic with a = -0.5 (Pillow's BICUBIC).
                let a = -0.5, x = abs(x)
                if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
                return 0
            case .lanczos:
                guard x >= -3, x < 3 else { return 0 }
                return Self.sinc(x) * Self.sinc(x / 3)
            }
        }

        private static func sinc(_ x: Double) -> Double {
            if x == 0 { return 1 }
            let value = x * Double.pi
            return sin(value) / value
        }
    }

    /// Normalized taps per output index: `start[i] ..< start[i] + count[i]` source indices with `weights[i * kernel + k]`.
    struct Coefficients {
        let start: [Int]
        let count: [Int]
        let kernel: Int
        let weights: [Double]
    }

    static func coefficients(inSize: Int, outSize: Int, filter: Filter) -> Coefficients {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = filter.support * filterScale
        let kernel = Int(support.rounded(.up)) * 2 + 1
        var start = [Int](repeating: 0, count: outSize), count = [Int](repeating: 0, count: outSize)
        var weights = [Double](repeating: 0, count: outSize * kernel)
        for output in 0..<outSize {
            let center = (Double(output) + 0.5) * scale
            // C casts truncate toward zero, like Swift's Int(Double).
            let first = max(0, Int(center - support + 0.5))
            let taps = min(inSize, Int(center + support + 0.5)) - first
            var total = 0.0
            for tap in 0..<max(taps, 0) {
                let value = filter.weight((Double(tap + first) - center + 0.5) / filterScale)
                weights[output * kernel + tap] = value
                total += value
            }
            if total != 0 { for tap in 0..<max(taps, 0) { weights[output * kernel + tap] /= total } }
            start[output] = first; count[output] = max(taps, 0)
        }
        return Coefficients(start: start, count: count, kernel: kernel, weights: weights)
    }

    private static let precisionBits = 22

    /// normalize_coeffs_8bpc: fixed point with 22 fractional bits, rounded half away from zero.
    private static func fixed(_ coefficients: Coefficients) -> [Int] {
        coefficients.weights.map { value in
            let scaled = value * Double(1 << precisionBits)
            return value < 0 ? Int(scaled - 0.5) : Int(scaled + 0.5)
        }
    }

    /// Image.resize(size, filter) for RGB: a horizontal pass, then a vertical pass, each rounded to 8 bits.
    static func resize(_ image: RGBImage, width: Int, height: Int, filter: Filter) -> RGBImage {
        precondition(width > 0 && height > 0)
        var current = image
        if width != current.width { current = horizontal(current, width: width, filter: filter) }
        if height != current.height { current = vertical(current, height: height, filter: filter) }
        return current
    }

    private static func clip8(_ sum: Int) -> UInt8 { UInt8(clamping: sum >> precisionBits) }

    private static func horizontal(_ image: RGBImage, width: Int, filter: Filter) -> RGBImage {
        let plan = Self.coefficients(inSize: image.width, outSize: width, filter: filter), weights = fixed(plan)
        let kernel = plan.kernel, rounding = 1 << (precisionBits - 1), sourceWidth = image.width
        var output = [UInt8](repeating: 0, count: width * image.height * 3)
        image.bytes.withUnsafeBufferPointer { source in
            output.withUnsafeMutableBufferPointer { target in
                let target = target
                DispatchQueue.concurrentPerform(iterations: image.height) { row in
                    let sourceRow = row * sourceWidth * 3, targetRow = row * width * 3
                    for x in 0..<width {
                        var r = rounding, g = rounding, b = rounding
                        let first = plan.start[x]
                        for tap in 0..<plan.count[x] {
                            let k = weights[x * kernel + tap], offset = sourceRow + (first + tap) * 3
                            r += Int(source[offset]) * k; g += Int(source[offset + 1]) * k; b += Int(source[offset + 2]) * k
                        }
                        target[targetRow + x * 3] = clip8(r); target[targetRow + x * 3 + 1] = clip8(g); target[targetRow + x * 3 + 2] = clip8(b)
                    }
                }
            }
        }
        return RGBImage(width: width, height: image.height, bytes: output)
    }

    private static func vertical(_ image: RGBImage, height: Int, filter: Filter) -> RGBImage {
        let plan = Self.coefficients(inSize: image.height, outSize: height, filter: filter), weights = fixed(plan)
        let kernel = plan.kernel, rounding = 1 << (precisionBits - 1), width = image.width
        var output = [UInt8](repeating: 0, count: width * height * 3)
        image.bytes.withUnsafeBufferPointer { source in
            output.withUnsafeMutableBufferPointer { target in
                let target = target
                DispatchQueue.concurrentPerform(iterations: height) { y in
                    let first = plan.start[y], taps = plan.count[y], targetRow = y * width * 3
                    for x in 0..<width {
                        var r = rounding, g = rounding, b = rounding
                        for tap in 0..<taps {
                            let k = weights[y * kernel + tap], offset = ((first + tap) * width + x) * 3
                            r += Int(source[offset]) * k; g += Int(source[offset + 1]) * k; b += Int(source[offset + 2]) * k
                        }
                        target[targetRow + x * 3] = clip8(r); target[targetRow + x * 3 + 1] = clip8(g); target[targetRow + x * 3 + 2] = clip8(b)
                    }
                }
            }
        }
        return RGBImage(width: width, height: height, bytes: output)
    }
}
