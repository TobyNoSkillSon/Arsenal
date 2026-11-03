// SPDX-License-Identifier: LicenseRef-S-Lab-1.0 (Swift port of GSeanCDAT/GIMM-VFI; see LICENSE-upstream-GIMMNative)
//
//  Spatial ops for the GIMM-VFI-R port, NHWC throughout. Precision follows the maker's autocast mixed-precision mode (torch.autocast float16):
//  convolutions and matmuls take FP16 inputs and weights; normalisation, sampling, warping and splatting compute in
//  float32 and return float32, as autocast does; elementwise ops follow MLX's (= torch's) type promotion.
//

import Foundation
import MLX
import MLXNN

/// The stored FP16 tensors under the maker's own state-dict names (convolution kernels transposed to OHWI at Get).
public struct GIMMParams {
    public let w: [String: MLXArray]
    public init(_ w: [String: MLXArray]) { self.w = w }
    subscript(_ key: String) -> MLXArray {
        guard let value = w[key] else { preconditionFailure("GIMM-VFI-R weight missing: \(key) (Load checks the inventory)") }
        return value
    }
}

/// torch Conv2d with padding k // 2 (every convolution in this model), FP16 in and out like autocast; `reflect` is
/// padding_mode="reflect" (one pixel).
func conv(_ x: MLXArray, _ p: GIMMParams, _ key: String, stride: Int = 1, reflect: Bool = false) -> MLXArray {
    let weight = p[key + ".weight"]
    var input = x.asType(weight.dtype)
    var pad = (weight.dim(1) / 2, weight.dim(2) / 2)
    if reflect {
        input = reflectPad1(input); pad = (0, 0)
    }
    var y = conv2d(input, weight, stride: IntOrPair(stride), padding: IntOrPair(pad))
    if let bias = p.w[key + ".bias"] {
        // MLX += mutates aliases of the array.
        // swiftlint:disable:next shorthand_operator
        y = y + bias
    }
    return y
}

func prelu(_ x: MLXArray, _ weight: MLXArray) -> MLXArray { MLX.maximum(x, 0) + weight * MLX.minimum(x, 0) }
func lrelu(_ x: MLXArray) -> MLXArray { leakyRelu(x, negativeSlope: 0.1) }

/// `convrelu` of the AMT blocks: Conv2d then PReLU, keys `<key>.0.*` and `<key>.1.weight`.
func convPReLU(_ x: MLXArray, _ p: GIMMParams, _ key: String) -> MLXArray { prelu(conv(x, p, key + ".0"), p[key + ".1.weight"]) }

/// Eval-mode BatchNorm2d in float32 (eps 1e-5).
func batchNorm(_ x: MLXArray, _ p: GIMMParams, _ key: String) -> MLXArray {
    let f = DType.float32
    return (x.asType(f) - p[key + ".running_mean"].asType(f)) * MLX.rsqrt(p[key + ".running_var"].asType(f) + 1e-5) * p[key + ".weight"].asType(f)
        + p[key + ".bias"].asType(f)
}

/// InstanceNorm2d without affine parameters, per sample and channel, biased variance, float32.
func instanceNorm(_ x: MLXArray) -> MLXArray {
    let v = x.asType(.float32)
    let mean = MLX.mean(v, axes: [1, 2], keepDims: true)
    let variance = MLX.mean(MLX.square(v - mean), axes: [1, 2], keepDims: true)
    return (v - mean) * MLX.rsqrt(variance + 1e-5)
}

/// One-pixel reflect padding of H and W.
func reflectPad1(_ x: MLXArray) -> MLXArray {
    let h = x.dim(1), w = x.dim(2)
    var y = concatenated([x[0..., 1..<2], x, x[0..., (h - 2)..<(h - 1)]], axis: 1)
    y = concatenated([y[0..., 0..., 1..<2], y, y[0..., 0..., (w - 2)..<(w - 1)]], axis: 2)
    return y
}

/// torch PixelShuffle (channel split C, r, r) for NHWC.
func pixelShuffle(_ x: MLXArray, _ r: Int = 2) -> MLXArray {
    let n = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3) / (r * r)
    return x.reshaped([n, h, w, c, r, r]).transposed(0, 1, 4, 2, 5, 3).reshaped([n, h * r, w * r, c])
}

/// torch F.interpolate(mode="bilinear", align_corners=False, scale_factor=s) along one axis: output floor(n * s),
/// source (d + 0.5) / s - 0.5 clamped at zero, computed in float32 and returned in the input's dtype.
func resizeAxis(_ x: MLXArray, axis: Int, scale: Double) -> MLXArray {
    let n = x.dim(axis), out = Int((Double(n) * scale).rounded(.down))
    if out == n && scale == 1 { return x }
    var i0 = [Int32](), i1 = [Int32](), l1 = [Float]()
    i0.reserveCapacity(out); i1.reserveCapacity(out); l1.reserveCapacity(out)
    let inverse = Float(1 / scale)
    for d in 0..<out {
        let src = Swift.max(0, (Float(d) + 0.5) * inverse - 0.5)
        let f = Swift.min(Int(src), n - 1)
        i0.append(Int32(f)); i1.append(Int32(Swift.min(f + 1, n - 1))); l1.append(src - Float(f))
    }
    var shape = [Int](repeating: 1, count: x.ndim); shape[axis] = out
    let lambda = MLXArray(l1).reshaped(shape)
    let v = x.asType(.float32)
    let y = take(v, MLXArray(i0), axis: axis) * (1 - lambda) + take(v, MLXArray(i1), axis: axis) * lambda
    return y.asType(x.dtype)
}

func resize(_ x: MLXArray, _ scale: Double) -> MLXArray { resizeAxis(resizeAxis(x, axis: 1, scale: scale), axis: 2, scale: scale) }

/// torch grid_sample(bilinear, align_corners=True) at pixel coordinates `ix`, `iy` ([N, gH, gW], float32).
/// `zeros`: padding_mode="zeros" (outside corners contribute nothing); otherwise "border" (indices clamp to the edge).
/// Returns float32 [N, gH, gW, C].
func sample(_ input: MLXArray, ix: MLXArray, iy: MLXArray, zeros: Bool) -> MLXArray {
    let n = input.dim(0), h = input.dim(1), w = input.dim(2), c = input.dim(3)
    let gh = ix.dim(1), gw = ix.dim(2)
    let flat = input.asType(.float32).reshaped([n, h * w, c])
    let x0 = MLX.floor(ix), y0 = MLX.floor(iy)
    let fx = ix - x0, fy = iy - y0
    var out: MLXArray?
    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let cx = x0 + Float(dx), cy = y0 + Float(dy)
        var weight = (dx == 1 ? fx : 1 - fx) * (dy == 1 ? fy : 1 - fy)
        if zeros {
            let inside = (cx .>= 0) .&& (cx .<= Float(w - 1)) .&& (cy .>= 0) .&& (cy .<= Float(h - 1))
            weight = MLX.where(inside, weight, MLXArray(Float(0)))
        }
        let xi = MLX.clip(cx, min: 0, max: Float(w - 1)).asType(.int32), yi = MLX.clip(cy, min: 0, max: Float(h - 1)).asType(.int32)
        let index = broadcast((yi * Int32(w) + xi).reshaped([n, gh * gw, 1]), to: [n, gh * gw, c])
        let term = takeAlong(flat, index, axis: 1).reshaped([n, gh, gw, c]) * weight.expandedDimensions(axis: -1)
        out = out.map { $0 + term } ?? term
    }
    return out!
}

/// fi_utils.warp: backward warp by a pixel flow ([N, H, W, 2], x then y), border padding, float32 out.
func warp(_ input: MLXArray, _ flow: MLXArray) -> MLXArray {
    let n = flow.dim(0), h = flow.dim(1), w = flow.dim(2)
    let f = flow.asType(.float32)
    let gx = broadcast(MLXArray((0..<w).map { Float($0) }).reshaped([1, 1, w]), to: [n, h, w])
    let gy = broadcast(MLXArray((0..<h).map { Float($0) }).reshaped([1, h, 1]), to: [n, h, w])
    return sample(input, ix: gx + f[0..., 0..., 0..., 0], iy: gy + f[0..., 0..., 0..., 1], zeros: false)
}

/// RAFT coords_grid: [N, H, W, 2] float32 holding (x, y).
func coordsGrid(_ n: Int, _ h: Int, _ w: Int) -> MLXArray {
    let gx = broadcast(MLXArray((0..<w).map { Float($0) }).reshaped([1, 1, w, 1]), to: [n, h, w, 1])
    let gy = broadcast(MLXArray((0..<h).map { Float($0) }).reshaped([1, h, 1, 1]), to: [n, h, w, 1])
    return concatenated([gx, gy], axis: -1)
}

/// The maker's softmax-splatting forward (`softsplat_out`, CuPy) for strMode "linear-zeroeps": every source pixel adds
/// `input * metric` and `metric` bilinearly at (x + flow_x, y + flow_y) in float32 (the maker casts its inputs to float32),
/// skipping a pixel whose target is not finite and any corner outside the image; the sums are divided by the summed
/// metric, with exact zeros replaced by one. input [N, H, W, C], flow [N, H, W, 2], metric [N, H, W, 1]; float32 out.
public func softsplatLinear(_ input: MLXArray, flow: MLXArray, metric: MLXArray) -> MLXArray {
    let n = input.dim(0), h = input.dim(1), w = input.dim(2), c = input.dim(3)
    let m = metric.asType(.float32)
    let source = concatenated([input.asType(.float32) * m, m], axis: -1).reshaped([n * h * w, c + 1])
    let f = flow.asType(.float32)
    let gx = MLXArray((0..<w).map { Float($0) }).reshaped([1, 1, w]), gy = MLXArray((0..<h).map { Float($0) }).reshaped([1, h, 1])
    var fx = gx + f[0..., 0..., 0..., 0], fy = gy + f[0..., 0..., 0..., 1]
    let finite = MLX.isFinite(fx) .&& MLX.isFinite(fy)
    fx = MLX.where(finite, fx, MLXArray(Float(0))); fy = MLX.where(finite, fy, MLXArray(Float(0)))
    let x0 = MLX.floor(fx), y0 = MLX.floor(fy)
    let base = MLXArray((0..<n).map { Int32($0 * h * w) }).reshaped([n, 1, 1])
    var out = MLXArray.zeros([n * h * w, c + 1], dtype: .float32)
    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let cx = x0 + Float(dx), cy = y0 + Float(dy)
        let weight = (dx == 1 ? fx - x0 : x0 + 1 - fx) * (dy == 1 ? fy - y0 : y0 + 1 - fy)
        let valid = finite .&& (cx .>= 0) .&& (cx .< Float(w)) .&& (cy .>= 0) .&& (cy .< Float(h))
        let xi = MLX.clip(cx, min: 0, max: Float(w - 1)).asType(.int32), yi = MLX.clip(cy, min: 0, max: Float(h - 1)).asType(.int32)
        let index = (base + yi * Int32(w) + xi).reshaped([n * h * w])
        let value = source * MLX.where(valid, weight, MLXArray(Float(0))).reshaped([n * h * w, 1])
        out = out.at[index].add(value)
    }
    out = out.reshaped([n, h, w, c + 1])
    let norm = out[0..., 0..., 0..., c..<(c + 1)]
    return out[0..., 0..., 0..., 0..<c] / MLX.where(norm .== 0, MLXArray(Float(1)), norm)
}
