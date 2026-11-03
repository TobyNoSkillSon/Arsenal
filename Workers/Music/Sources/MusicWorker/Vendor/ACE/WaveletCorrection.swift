//
//  WaveletCorrection.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  DCW — differential correction in the wavelet domain (upstream's
//  dcw_correction_mlx.py, native Haar path): after each sampler step, push
//  the latent's low and/or high Haar band away from the predicted clean
//  sample, with strengths that depend on the step's timestep.
//

import Foundation
import MLX

enum WaveletCorrection {

    static func haar(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var x = x
        if x.dim(1) % 2 == 1 {
            x = MLX.concatenated([x, MLXArray.zeros([x.dim(0), 1, x.dim(2)], dtype: x.dtype)], axis: 1)
        }
        let even = x[0..., .stride(from: 0, by: 2), 0...]
        let odd = x[0..., .stride(from: 1, by: 2), 0...]
        let s = Float(1 / 2.0.squareRoot())
        return ((even + odd) * s, (even - odd) * s)
    }

    static func inverse(_ low: MLXArray, _ high: MLXArray, length: Int) -> MLXArray {
        let s = Float(1 / 2.0.squareRoot())
        let even = (low + high) * s
        let odd = (low - high) * s
        let stacked = MLX.stacked([even, odd], axis: 2)
        return stacked.reshaped(even.dim(0), -1, even.dim(2))[0..., ..<length, 0...]
    }

    // `x` after the correction toward/away from `denoised` at timestep `t`.
    static func apply(
        _ x: MLXArray, denoised: MLXArray, t: Double, mode: DCWMode, scaler: Double,
        highScaler: Double
    ) -> MLXArray {
        let low = Float(t * scaler)
        let high = Float((1 - t) * scaler)
        let doubleHigh = Float((1 - t) * highScaler)
        switch mode {
        case .pix:
            return scaler == 0 ? x : x + Float(scaler) * (x - denoised)
        case .low where low == 0, .high where high == 0:
            return x
        case .double where low == 0 && doubleHigh == 0:
            return x
        default:
            break
        }
        var (xl, xh) = haar(x)
        let (yl, yh) = haar(denoised)
        switch mode {
        // swiftlint:disable:next shorthand_operator
        case .low: xl = xl + low * (xl - yl)
        // swiftlint:disable:next shorthand_operator
        case .high: xh = xh + high * (xh - yh)
        case .double:
            // swiftlint:disable:next shorthand_operator
            if low != 0 { xl = xl + low * (xl - yl) }
            // swiftlint:disable:next shorthand_operator
            if doubleHigh != 0 { xh = xh + doubleHigh * (xh - yh) }
        case .pix: break
        }
        return inverse(xl, xh, length: x.dim(1))
    }
}
