// swiftlint:disable all
// SPDX-License-Identifier: MIT
// Pillow-compatible 8-bit resampling. Geometry, coefficient normalization and rounding follow
// Pillow 12.3.0 src/libImaging/Resample.c (LICENSE-Pillow). No model computation runs here.
import CoreGraphics
import Foundation

public enum TrellisResample {
    public enum Filter: Sendable { case bilinear, bicubic, lanczos }
    private struct Coefficients { let start: Int; let weights: [Int64] }
    private static func coefficients(input: Int, output: Int, filter: Filter) -> [Coefficients] {
        let scale = Double(input) / Double(output)
        let filterScale = max(1, scale)
        let support = (filter == .bilinear ? 1.0 : filter == .bicubic ? 2.0 : 3.0) * filterScale
        return (0..<output).map { i in
            let center = (Double(i) + 0.5) * scale
            let start = max(0, Int(center - support + 0.5))
            let end = min(input, Int(center + support + 0.5))
            let values = (start..<end).map { j -> Double in
                let x = abs((Double(j) - center + 0.5) / filterScale)
                if filter == .bilinear { return max(0, 1 - x) }
                if filter == .lanczos {
                    if x == 0 { return 1 }; if x >= 3 { return 0 }
                    return sin(.pi * x) * sin(.pi * x / 3) / (.pi * x * .pi * x / 3)
                }
                if x < 1 { return ((1.5 * x - 2.5) * x) * x + 1 }
                if x < 2 { return (((x - 5) * x + 8) * x - 4) * -0.5 }
                return 0
            }
            let sum = values.reduce(0, +)
            return Coefficients(
                start: start,
                weights: values.map {
                    let w = sum == 0 ? $0 : $0 / sum
                    return Int64((w < 0 ? -0.5 : 0.5) + w * Double(1 << 22))
                })
        }
    }
    public static func resize(
        _ bytes: [UInt8], width: Int, height: Int, channels: Int,
        targetWidth: Int, targetHeight: Int, filter: Filter
    ) -> [UInt8] {
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

}
// swiftlint:enable all
