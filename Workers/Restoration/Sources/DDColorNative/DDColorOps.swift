// SPDX-License-Identifier: MIT
// Ported from xocialize/mlx-ddcolor-swift `Sources/DDColor/DDColorOps.swift` (MIT); see PORTED-FROM.md.
import Foundation
import MLX

enum DDColorOps {
    /// Eval-mode BatchNorm over the channel (last) axis, eps 1e-5.
    static func batchNorm(_ x: MLXArray, mean: MLXArray, variance: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        (x - mean) * (MLX.rsqrt(variance + 1e-5) * weight) + bias
    }
    /// Exact erf GELU (ConvNeXt's nn.GELU()).
    static func gelu(_ x: MLXArray) -> MLXArray { 0.5 * x * (1 + MLX.erf(x / 1.4142135623730951)) }
    static func relu(_ x: MLXArray) -> MLXArray { MLX.maximum(x, 0) }

    /// NHWC PixelShuffle(r): torch splits input channels as (c_out, rh, rw), c_out outermost.
    static func pixelShuffle(_ x: MLXArray, _ r: Int) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped([b, h, w, c / (r * r), r, r]).transposed(0, 1, 4, 2, 5, 3).reshaped([b, h * r, w * r, c / (r * r)])
    }

    /// fastai anti-checkerboard blur: ReplicationPad2d((1,0,1,0)) then AvgPool2d(2, stride 1).
    static func blur(_ x: MLXArray) -> MLXArray {
        let p = MLX.padded(x, widths: [.init(0), .init((1, 0)), .init((1, 0)), .init(0)], mode: .edge)
        let h = x.dim(1), w = x.dim(2)
        let s = p[0..., 0..<h, 0..<w, 0...] + p[0..., 1...h, 0..<w, 0...] + p[0..., 0..<h, 1...w, 0...] + p[0..., 1...h, 1...w, 0...]
        return s * 0.25
    }

    /// nn.MultiheadAttention with a packed (3C, C) projection, one batch, no mask. Softmax runs in float32 internally.
    static func attention(q: MLXArray, k: MLXArray, v: MLXArray, inWeight: MLXArray, inBias: MLXArray, outWeight: MLXArray, outBias: MLXArray, heads: Int)
        -> MLXArray
    {
        let c = q.dim(-1), hd = c / heads
        func project(_ x: MLXArray, _ i: Int) -> MLXArray {
            let y = MLX.matmul(x, inWeight[(i * c)..<((i + 1) * c)].transposed()) + inBias[(i * c)..<((i + 1) * c)]
            return y.reshaped([x.dim(0), heads, hd]).transposed(1, 0, 2)
        }
        let qh = project(q, 0), kh = project(k, 1), vh = project(v, 2)
        let scores = MLX.matmul(qh, kh.transposed(0, 2, 1)) * (1 / Float(hd).squareRoot())
        let out = MLX.matmul(MLX.softmax(scores, axis: -1, precise: true), vh).transposed(1, 0, 2).reshaped([q.dim(0), c])
        return MLX.matmul(out, outWeight.transposed()) + outBias
    }

    /// DETR PositionEmbeddingSine(128, normalize=True) flattened to (H*W, 256), computed on the host in float32.
    static func positionEmbedding(h: Int, w: Int, features: Int = 128, temperature: Float = 10000) -> MLXArray {
        let scale = 2 * Float.pi, eps: Float = 1e-6
        let dimT = (0..<features).map { powf(temperature, 2 * Float($0 / 2) / Float(features)) }
        var out = [Float](repeating: 0, count: h * w * 2 * features)
        for y in 0..<h {
            let ye = Float(y + 1) / (Float(h) + eps) * scale
            for x in 0..<w {
                let xe = Float(x + 1) / (Float(w) + eps) * scale, base = (y * w + x) * 2 * features
                for j in 0..<features {
                    out[base + j] = j % 2 == 0 ? sinf(ye / dimT[j]) : cosf(ye / dimT[j])
                    out[base + features + j] = j % 2 == 0 ? sinf(xe / dimT[j]) : cosf(xe / dimT[j])
                }
            }
        }
        return MLXArray(out, [h * w, 2 * features])
    }
}
