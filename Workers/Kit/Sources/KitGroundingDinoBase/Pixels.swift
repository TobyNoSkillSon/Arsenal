// SPDX-License-Identifier: MIT
// PIL-exact bilinear resize on 8-bit RGB, copied from the clipseg-rd64-refined runner. torch's uint8 antialiased
// bilinear (torchvision resize on a uint8 tensor, the GroundingDinoImageProcessor path) follows the same algorithm.
import Foundation

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
