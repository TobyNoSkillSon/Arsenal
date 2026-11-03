// SPDX-License-Identifier: MIT
// Microsoft TRELLIS.2 RGBA crop/composite and DINOv3 Pillow LANCZOS preprocessing.
import CoreGraphics
import Foundation
import MLX
import ArsenalWire

public enum TrellisImage {
    public static func pixels(_ image: CGImage, size: Int) throws -> MLXArray {
        guard image.colorSpace?.model == .rgb, [.last, .first, .premultipliedLast, .premultipliedFirst].contains(image.alphaInfo),
            image.width > 0, image.height > 0, image.width <= 16384, image.height <= 16384,
            let context = CGContext(
                data: nil, width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw invalid() }
        context.setBlendMode(.copy)
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let data = context.data else { throw invalid() }
        let source = Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: UInt8.self), count: image.width * image.height * 4))
        guard stride(from: 3, to: source.count, by: 4).contains(where: { source[$0] < 255 }) else { throw invalid() }
        // Pillow resizes RGBA in premultiplied space; RGB*alpha compositing later recovers these bytes.
        let scale = min(1, 1024 / Double(max(image.width, image.height)))
        let width = max(1, Int(Double(image.width) * scale)), height = max(1, Int(Double(image.height) * scale))
        let rgba = TrellisResample.resize(
            source, width: image.width, height: image.height, channels: 4,
            targetWidth: width, targetHeight: height, filter: .lanczos)
        var minX = width, minY = height, maxX = -1, maxY = -1
        for y in 0..<height {
            for x in 0..<width where rgba[(y * width + x) * 4 + 3] > 204 {
                minX = min(minX, x); minY = min(minY, y); maxX = max(maxX, x); maxY = max(maxY, y)
            }
        }
        guard maxX >= minX, maxY >= minY else { throw invalid() }
        let half = max(maxX - minX, maxY - minY) / 2
        // PIL crop rounds floating endpoints to nearest-even. Preserve its odd-boundary behavior.
        let x0 = Int((Double(minX + maxX) / 2 - Double(half)).rounded(.toNearestOrEven))
        let y0 = Int((Double(minY + maxY) / 2 - Double(half)).rounded(.toNearestOrEven))
        let x1 = Int((Double(minX + maxX) / 2 + Double(half)).rounded(.toNearestOrEven))
        let y1 = Int((Double(minY + maxY) / 2 + Double(half)).rounded(.toNearestOrEven))
        let cropWidth = max(1, x1 - x0), cropHeight = max(1, y1 - y0)
        var crop = [UInt8](repeating: 0, count: cropWidth * cropHeight * 3)
        for y in 0..<cropHeight {
            for x in 0..<cropWidth {
                let sx = x0 + x, sy = y0 + y
                guard sx >= 0, sx < width, sy >= 0, sy < height else { continue }
                for c in 0..<3 { crop[(y * cropWidth + x) * 3 + c] = rgba[(sy * width + sx) * 4 + c] }
            }
        }
        let rgb = TrellisResample.resize(
            crop, width: cropWidth, height: cropHeight, channels: 3,
            targetWidth: size, targetHeight: size, filter: .lanczos)
        let mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]
        var values = [Float](repeating: 0, count: size * size * 3)
        for pixel in 0..<(size * size) {
            for channel in 0..<3 {
                values[channel * size * size + pixel] = (Float(rgb[pixel * 3 + channel]) / 255 - mean[channel]) / std[channel]
            }
        }
        return MLXArray(values, [1, 3, size, size]).asType(.float16)
    }
    private static func invalid() -> WorkerFailure {
        WorkerFailure(
            code: "invalid_request", message: "TRELLIS.2 requires a background-removed RGBA image with visible foreground and transparent background. Run remove_background first.")
    }
}
