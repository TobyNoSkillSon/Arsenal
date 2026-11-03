// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// Paddle's NeoX-style rotation, 2D vision frequencies and sectioned 3D text positions.
enum PaddlePositions {
    static func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cos + rotated * sin
    }
    static func apply(_ x: MLXArray, positions: MLXArray, base: Float, sections: [Int]) -> MLXArray {
        let dim = x.dim(-1), count = x.dim(2)
        let frequencies = MLXArray((0..<(dim / 2)).map { 1 / pow(base, Float(2 * $0) / Float(dim)) })
        let angles = positions.asType(.float32)[0..., 0..., .newAxis] * frequencies
        var pieces: [MLXArray] = [], offset = 0
        for (axis, size) in sections.enumerated() {
            pieces.append(angles[axis, 0..., offset..<(offset + size)]); offset += size
        }
        let half = concatenated(pieces, axis: -1)
        let full = concatenated([half, half], axis: -1).reshaped(1, 1, count, dim)
        return rotate(x, cos: cos(full).asType(x.dtype), sin: sin(full).asType(x.dtype))
    }
    static func imagePrompt(count: Int, imageStart: Int, height: Int, width: Int) -> ([[Int]], Int) {
        let imageCount = height * width
        var positions = Array(repeating: [Int](), count: 3)
        for index in 0..<count {
            for axis in 0..<3 {
                let value: Int
                if index < imageStart {
                    value = index
                } else if index < imageStart + imageCount {
                    let patch = index - imageStart
                    value = imageStart + (axis == 0 ? 0 : axis == 1 ? patch / width : patch % width)
                } else {
                    value = imageStart + max(height, width) + index - imageStart - imageCount
                }
                positions[axis].append(value)
            }
        }
        return (positions, (positions.flatMap { $0 }.max() ?? -1) + 1)
    }
}
