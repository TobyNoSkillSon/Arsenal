// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import PerceptionSupport

/// RF-DETR's input as the checkpoints' own processor makes it (transformers RfDetrImageProcessor, resample BILINEAR):
/// 8-bit RGB scaled by 1/255 in float32, *then* resized with PyTorch's bilinear interpolation (align_corners false,
/// half-pixel centres, antialias off), then normalised with the ImageNet mean and standard deviation. Arsenal used to
/// resize the 8-bit image with CoreGraphics' high-quality filter first, which moved scores by up to 0.06.
public enum RFDETRPreprocess {
    public static let mean: [Float] = [0.485, 0.456, 0.406]
    public static let std: [Float] = [0.229, 0.224, 0.225]

    /// The image's stored 8-bit RGB at its own size, without resampling, in its own RGB colour space (as PIL decodes it).
    /// An input-only profile (some scanner and camera ICC profiles) cannot be a drawing target; its stored values are
    /// relabelled sRGB and drawn without conversion, which is what PIL hands the processor.
    public static func rgb(_ image: CGImage) throws -> (bytes: [UInt8], width: Int, height: Int) {
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let own = image.colorSpace, own.model == .rgb else { return try rgb(image, spaces: [srgb]) }
        if own.supportsOutput { return try rgb(image, spaces: [own, srgb]) }
        return try rgb(image.copy(colorSpace: srgb) ?? image, spaces: [srgb])
    }

    /// Draws into the first space CoreGraphics accepts as an 8-bit RGBX target.
    static func rgb(_ image: CGImage, spaces: [CGColorSpace]) throws -> (bytes: [UInt8], width: Int, height: Int) {
        let width = image.width, height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { pixels in
            spaces.contains { space in
                guard
                    let context = CGContext(
                        data: pixels.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                else { return false }
                context.interpolationQuality = .none
                context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
                return true
            }
        }
        guard drawn else { throw PerceptionIO.error("Cannot read image pixels.") }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) { for c in 0..<3 { rgb[i * 3 + c] = rgba[i * 4 + c] } }
        return (rgb, width, height)
    }

    /// Source taps and weights along one axis (PyTorch upsample_bilinear2d, align_corners false, no scale factor).
    static func taps(input: Int, output: Int) -> [(Int, Int, Float, Float)] {
        let scale = Float(input) / Float(output)
        return (0..<output).map { index in
            let source = max((Float(index) + 0.5) * scale - 0.5, 0)
            let lower = min(Int(source), input - 1)
            let upper = min(lower + 1, input - 1)
            let weight = source - Float(lower)
            return (lower, upper, 1 - weight, weight)
        }
    }

    /// Normalised pixels, row-major [size, size, 3] float32 (channels last, as the MLX graph reads them).
    public static func pixels(_ rgb: [UInt8], width: Int, height: Int, size: Int) -> [Float] {
        let factor = Float(0.00392156862745098)
        let rows = taps(input: height, output: size), columns = taps(input: width, output: size)
        var out = [Float](repeating: 0, count: size * size * 3)
        for (y, row) in rows.enumerated() {
            for (x, column) in columns.enumerated() {
                for c in 0..<3 {
                    func value(_ yy: Int, _ xx: Int) -> Float { Float(rgb[(yy * width + xx) * 3 + c]) * factor }
                    let top = column.2 * value(row.0, column.0) + column.3 * value(row.0, column.1)
                    let bottom = column.2 * value(row.1, column.0) + column.3 * value(row.1, column.1)
                    out[(y * size + x) * 3 + c] = ((row.2 * top + row.3 * bottom) - mean[c]) / std[c]
                }
            }
        }
        return out
    }
}
