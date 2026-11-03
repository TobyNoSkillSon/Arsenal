// SPDX-License-Identifier: MIT
// IDOL's input preparation (idol/preprocess.py) for an RGBA cutout: resize the short side to 1024, erode the alpha by
// one pixel, crop a square of side margin × the object's bounding box, resize to 1024 and composite on black. Pillow's
// LANCZOS is the shared Pillow-compatible resampler; Pillow resizes RGBA premultiplied, so the colour planes here are
// carried premultiplied by the (eroded) alpha, which is also what the final composite on black keeps.
import CoreGraphics
import Foundation
import MLXTrellis

public enum IdolImage {
    /// Crop margin by input kind: renders and generated images of one object (1.2) or photos (1.7).
    public static let cropMargin: [String: Double] = ["synthetic": 1.2, "real": 1.7]

    /// RGB bytes [size × size × 3] of the composite the encoders see.
    public static func prepare(_ image: CGImage, size: Int = 1024, margin: Double) throws -> [UInt8] {
        guard image.width > 0, image.height > 0, image.width <= 16384, image.height <= 16384,
            let context = CGContext(
                data: nil, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw invalid() }
        context.setBlendMode(.copy)
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let data = context.data else { throw invalid() }
        let source = Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: UInt8.self), count: image.width * image.height * 4))
        guard stride(from: 3, to: source.count, by: 4).contains(where: { source[$0] < 255 }) else { throw invalid() }

        let scale = Double(size) / Double(min(image.width, image.height))
        let width = max(1, Int((Double(image.width) * scale).rounded(.toNearestOrEven)))
        let height = max(1, Int((Double(image.height) * scale).rounded(.toNearestOrEven)))
        let rgba = TrellisResample.resize(source, width: image.width, height: image.height, channels: 4, targetWidth: width, targetHeight: height, filter: .lanczos)

        // erode: alpha = 3×3 minimum (Pillow MinFilter(3)); the colour is re-weighted by the eroded alpha
        var eroded = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var m: UInt8 = 255
                for dy in -1...1 {
                    for dx in -1...1 {
                        let yy = min(max(y + dy, 0), height - 1), xx = min(max(x + dx, 0), width - 1)
                        m = min(m, rgba[(yy * width + xx) * 4 + 3])
                    }
                }
                eroded[y * width + x] = m
            }
        }
        var minX = width, minY = height, maxX = -1, maxY = -1
        for y in 0..<height {
            for x in 0..<width where eroded[y * width + x] > 0 {
                minX = min(minX, x); minY = min(minY, y); maxX = max(maxX, x); maxY = max(maxY, y)
            }
        }
        guard maxX >= minX else { throw IdolError("No object found in the image: its alpha channel is empty.") }
        let cx = Double(minX + maxX) / 2, cy = Double(minY + maxY) / 2
        let half = Double(max(maxX - minX, maxY - minY)) / 2 * margin
        let x0 = Int(cx - half), y0 = Int(cy - half), x1 = Int(cx + half), y1 = Int(cy + half) // Python int(): toward zero
        let cropW = max(1, x1 - x0), cropH = max(1, y1 - y0)
        // a very wide object with a large margin would need a crop of tens of thousands of pixels a side
        guard cropW <= 8 * size, cropH <= 8 * size else {
            throw IdolError("The object is too wide for its crop margin; use a smaller crop_margin or a less elongated picture.")
        }
        var crop = [UInt8](repeating: 0, count: cropW * cropH * 3)
        for y in 0..<cropH {
            let sy = y0 + y
            guard sy >= 0, sy < height else { continue }
            for x in 0..<cropW {
                let sx = x0 + x
                guard sx >= 0, sx < width else { continue }
                let i = sy * width + sx, a = Int(rgba[i * 4 + 3]), e = Int(eroded[i])
                for c in 0..<3 {
                    // premultiplied by the resized alpha → by the eroded alpha
                    crop[(y * cropW + x) * 3 + c] = a == 0 ? 0 : UInt8(min(255, (Int(rgba[i * 4 + c]) * e + a / 2) / a))
                }
            }
        }
        return TrellisResample.resize(crop, width: cropW, height: cropH, channels: 3, targetWidth: size, targetHeight: size, filter: .lanczos)
    }

    static func invalid() -> IdolError {
        IdolError("IDOL needs an image with the background removed (a transparent background). Run remove_background first.")
    }
}
