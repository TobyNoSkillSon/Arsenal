// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
// Image processing utilities for MoGe-2 Swift port
// bilinear_resize and pad_replicate for NHWC tensors

import Foundation
import MLX
import MLXNN

/// Replicate (edge) padding for NHWC tensors.
///
/// - Parameters:
///   - x: Input tensor (B, H, W, C)
///   - pad: Number of pixels to pad on each side of the spatial dims
/// - Returns:
///   Padded tensor (B, H+2*pad, W+2*pad, C)
public func padReplicate(_ x: MLXArray, _ pad: Int) -> MLXArray {
    if pad == 0 { return x }
    // Pad only spatial axes (1 and 2), leave batch and channel unchanged.
    let widths: [IntOrPair] = [
        IntOrPair(0), IntOrPair(pad), IntOrPair(pad), IntOrPair(0)
    ]
    return padded(x, widths: widths, mode: .edge)
}

/// Native transposed convolution for the community checkpoint's unflipped OHWI weights.
public func convTranspose2d(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray?,
    stride: Int = 2
) -> MLXArray {
    var out = MLX.convTransposed2d(x, weight, stride: .init(stride))
    if let bias { out = out + bias }
    return out
}

/// Bilinear interpolation resize for NHWC tensors. Matches PyTorch's
/// `F.interpolate(..., mode='bilinear', align_corners=False)`.
///
/// - Parameters:
///   - x: Input (B, H, W, C)
///   - targetH: Target height
///   - targetW: Target width
/// - Returns:
///   Resized tensor (B, targetH, targetW, C)
public func bilinearResize(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    let H = x.dim(1)
    let W = x.dim(2)
    if H == targetH && W == targetW { return x }

    // MLX's `Upsample` computes the output size as `Int(scale * Float(N))`,
    // which truncates. A scale derived from a Float division can land a hair
    // below the exact ratio (e.g. 700/556 makes 556 × scale ≈ 699.9999 → 699),
    // dropping one row/column and mis-shaping the ViT token grid downstream.
    // `nextUp` nudges the scale past the rounding boundary.
    let sh = (Float(targetH) / Float(H)).nextUp
    let sw = (Float(targetW) / Float(W)).nextUp
    let up = Upsample(scaleFactor: .array([sh, sw]), mode: .linear(alignCorners: false))
    var out = up(x)
    // Defensive: if the adjusted scale over/undershoots by one, crop to target.
    let outH = out.dim(1)
    let outW = out.dim(2)
    if outH != targetH || outW != targetW {
        out = out[0..., 0..<targetH, 0..<targetW, 0...]
    }
    // MLX's Upsample interpolates with float32 index weights, which silently promotes a
    // half activation to float32 and with it every downstream learned layer.
    return out.asType(x.dtype)
}

/// One-axis weights of PyTorch's antialiased bilinear resize
/// (`F.interpolate(mode="bilinear", align_corners=False, antialias=True)`,
/// ATen `_compute_weights_aa` with the triangle filter): a `[outSize, inSize]`
/// row-normalized matrix. Downscaling widens the triangle support by the scale;
/// upscaling reduces to ordinary edge-clamped bilinear interpolation.
public func antialiasedBilinearWeights(inSize: Int, outSize: Int) -> [Float] {
    precondition(inSize > 0 && outSize > 0)
    let scale = Float(inSize) / Float(outSize)
    let support: Float = scale >= 1 ? scale : 1
    let invScale: Float = scale >= 1 ? 1 / scale : 1
    var weights = [Float](repeating: 0, count: outSize * inSize)
    for i in 0..<outSize {
        let center = scale * (Float(i) + 0.5)
        // C++ truncation toward zero, then clamping, exactly as ATen.
        let xmin = max(Int(center - support + 0.5), 0)
        let xmax = min(Int(center + support + 0.5), inSize)
        var total: Float = 0
        var row = [Float](repeating: 0, count: max(0, xmax - xmin))
        for j in 0..<row.count {
            let distance = abs((Float(j + xmin) - center + 0.5) * invScale)
            let w: Float = distance < 1 ? 1 - distance : 0
            row[j] = w
            total += w
        }
        for j in 0..<row.count {
            weights[i * inSize + xmin + j] = total != 0 ? row[j] / total : row[j]
        }
    }
    return weights
}

/// Antialiased bilinear resize for NHWC tensors, matching the DINOv2 encoder's
/// `F.interpolate(image, size, mode="bilinear", align_corners=False, antialias=True)`.
/// Separable weight matrices are applied in float32 and the result keeps the input dtype.
public func antialiasedBilinearResize(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    let B = x.dim(0), H = x.dim(1), W = x.dim(2), C = x.dim(3)
    if H == targetH && W == targetW { return x }
    let rows = MLXArray(antialiasedBilinearWeights(inSize: H, outSize: targetH), [targetH, H])
    let cols = MLXArray(antialiasedBilinearWeights(inSize: W, outSize: targetW), [targetW, W])
    let source = x.asType(.float32)
    // Rows: [targetH, H] x [B, H, W*C] -> [B, targetH, W*C]
    let vertical = MLX.matmul(rows, source.reshaped([B, H, W * C]))
    // Columns: [targetW, W] x [B*targetH, W, C] -> [B*targetH, targetW, C]
    let horizontal = MLX.matmul(cols, vertical.reshaped([B * targetH, W, C]))
    return horizontal.reshaped([B, targetH, targetW, C]).asType(x.dtype)
}

public func mogeBicubicResample(_ x: MLXArray, outH: Int, outW: Int, offset: Float = 0.1) -> MLXArray {
    let B = x.dim(0), inH = x.dim(1), inW = x.dim(2), C = x.dim(3)
    // Per-axis scales
    let sy = (Float(outH) + offset) / Float(inH)
    let sx = (Float(outW) + offset) / Float(inW)

    // Compute float sample coordinates per output index using PyTorch's mapping
    var ys = [Float](repeating: 0, count: outH)
    for i in 0..<outH { ys[i] = (Float(i) + 0.5) / sy - 0.5 }
    var xs = [Float](repeating: 0, count: outW)
    for i in 0..<outW { xs[i] = (Float(i) + 0.5) / sx - 0.5 }

    // Cubic kernel (a = -0.75, matches PyTorch default for bicubic).
    func cubicWeights(_ t: Float) -> [Float] {
        // Returns weights for [floor-1, floor, floor+1, floor+2] given fractional t in [0,1).
        // Distances from sample points: |1+t|, |t|, |1-t|, |2-t|.
        let a: Float = -0.75
        // For |x| in [1, 2]:  W(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a
        // For |x| in [0, 1]:  W(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1
        let d_1 = 1 + t // in [1, 2)
        let d0 = t // in [0, 1)
        let d1 = 1 - t // in (0, 1]
        let d2 = 2 - t // in (1, 2]
        let w_1 = a * d_1 * d_1 * d_1 - 5 * a * d_1 * d_1 + 8 * a * d_1 - 4 * a
        let w0 = (a + 2) * d0 * d0 * d0 - (a + 3) * d0 * d0 + 1
        let w1 = (a + 2) * d1 * d1 * d1 - (a + 3) * d1 * d1 + 1
        let w2 = a * d2 * d2 * d2 - 5 * a * d2 * d2 + 8 * a * d2 - 4 * a
        return [w_1, w0, w1, w2]
    }

    // Build per-output-index integer indices and weights for both axes.
    func build(_ coords: [Float], inSize: Int) -> (idx: [[Int]], wts: [[Float]]) {
        var idx = [[Int]]()
        var wts = [[Float]]()
        for c in coords {
            let f = floor(c)
            let t = c - f
            let base = Int(f)
            // Border-replicate (PyTorch default bicubic uses border reflection,
            // but for align_corners=False it uses zero-pad? Actually torch
            // bicubic_2d uses border replication for out-of-range.)
            let ii = [base - 1, base, base + 1, base + 2].map { max(0, min(inSize - 1, $0)) }
            idx.append(ii)
            wts.append(cubicWeights(t))
        }
        return (idx, wts)
    }

    let (yIdx, yWts) = build(ys, inSize: inH)
    let (xIdx, xWts) = build(xs, inSize: inW)

    // Output buffer: gather rows then convolve.
    // First, compute interpolation along H axis: for each output y, gather
    // 4 input rows at indices yIdx[y] and weight by yWts[y]. Result: [B, outH, inW, C].
    var hRows = [MLXArray]()
    for j in 0..<outH {
        let i0 = yIdx[j][0], i1 = yIdx[j][1], i2 = yIdx[j][2], i3 = yIdx[j][3]
        let w0 = yWts[j][0], w1 = yWts[j][1], w2 = yWts[j][2], w3 = yWts[j][3]
        let row =
            x[0..., i0..<(i0 + 1), 0..., 0...] * MLXArray(Float(w0))
            + x[0..., i1..<(i1 + 1), 0..., 0...] * MLXArray(Float(w1))
            + x[0..., i2..<(i2 + 1), 0..., 0...] * MLXArray(Float(w2))
            + x[0..., i3..<(i3 + 1), 0..., 0...] * MLXArray(Float(w3))
        hRows.append(row)
    }
    var hStack = MLX.concatenated(hRows, axis: 1) // [B, outH, inW, C]

    // Now interpolate along W axis.
    var wCols = [MLXArray]()
    for j in 0..<outW {
        let i0 = xIdx[j][0], i1 = xIdx[j][1], i2 = xIdx[j][2], i3 = xIdx[j][3]
        let w0 = xWts[j][0], w1 = xWts[j][1], w2 = xWts[j][2], w3 = xWts[j][3]
        let col =
            hStack[0..., 0..., i0..<(i0 + 1), 0...] * MLXArray(Float(w0))
            + hStack[0..., 0..., i1..<(i1 + 1), 0...] * MLXArray(Float(w1))
            + hStack[0..., 0..., i2..<(i2 + 1), 0...] * MLXArray(Float(w2))
            + hStack[0..., 0..., i3..<(i3 + 1), 0...] * MLXArray(Float(w3))
        wCols.append(col)
    }
    return MLX.concatenated(wCols, axis: 2).asType(x.dtype) // [B, outH, outW, C]
}

// swiftlint:enable all
