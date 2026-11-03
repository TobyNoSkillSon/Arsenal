// SPDX-License-Identifier: MIT
// Ported from xocialize/mlx-lama-swift `Sources/LaMa/{LaMaModel,LaMaOps}.swift` (MIT, see LICENSE-upstream-LaMaNative and
// PORTED-FROM.md), itself a port of advimman/lama (Apache-2.0). Arsenal runs the generator on BatchNorm-folded weights
// (LaMaFold), so every block is conv + bias; see PORTED-FROM.md for why.
import Foundation
import MLX
import MLXFFT

/// Big-LaMa FFCResNetGenerator over folded half weights: `(1,H,W,4)` = concat[image*(1-mask), mask] (H, W multiples
/// of 8) to the predicted RGB `(1,H,W,3)` in [0,1]. Compositing with the source is the caller's.
public final class LaMaModel: @unchecked Sendable {
    let w: [String: MLXArray]
    public let dtype: DType
    public var capture = false
    public private(set) var captured: [String: MLXArray] = [:]

    public init(weights: [String: MLXArray]) throws {
        try LaMaFold.validate(weights)
        w = weights; dtype = weights["model.34.weight"]!.dtype
    }

    private func a(_ k: String) -> MLXArray { w[k]! }
    private func conv(_ x: MLXArray, _ k: String, bias: Bool, stride: Int = 1) -> MLXArray {
        let y = MLX.conv2d(x, a(k + ".weight"), stride: .init(stride), padding: .init(0))
        return bias ? y + a(k + ".bias") : y
    }
    private func relu(_ x: MLXArray) -> MLXArray { MLX.maximum(x, 0) }
    private func keep(_ name: String, _ x: MLXArray?) { if capture, let x { captured[name] = x } }

    /// Reflection padding over H and W (no edge repeat), as nn.ReflectionPad2d / padding_mode="reflect".
    static func reflectPad(_ x: MLXArray, _ p: Int) -> MLXArray {
        if p == 0 { return x }
        // MLX.take does not bound-check; the adapter keeps every side at 16 pixels or more (2 at the deepest stage).
        precondition(x.dim(1) > p && x.dim(2) > p, "reflection padding needs more pixels than the pad")
        func idx(_ n: Int) -> MLXArray {
            MLXArray(Array(stride(from: p, to: 0, by: -1)).map(Int32.init) + (0..<n).map(Int32.init) + Array(stride(from: n - 2, to: n - 2 - p, by: -1)).map(Int32.init))
        }
        return MLX.take(MLX.take(x, idx(x.dim(1)), axis: 1), idx(x.dim(2)), axis: 2)
    }

    /// FourierUnit with norm="ortho": rFFT, real/imag interleaved per channel, folded 1x1 conv + ReLU, inverse rFFT.
    /// The transforms run in float32 (MLX has no half FFT; the provider's own half path does the same) and the
    /// spectral conv in the model dtype.
    private func fourierUnit(_ x: MLXArray, _ p: String) -> MLXArray {
        let (b, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let spectrum = MLXFFT.rfftn(x.asType(.float32), axes: [1, 2], norm: .ortho)
        let w2 = spectrum.dim(2)
        let interleaved = MLX.stacked([spectrum.realPart(), spectrum.imaginaryPart()], axis: -1).reshaped([b, h, w2, 2 * c]).asType(dtype)
        let y = relu(conv(interleaved, p + ".conv_layer", bias: true)).asType(.float32).reshaped([b, h, w2, c, 2])
        let complex = y[0..., 0..., 0..., 0..., 0] + y[0..., 0..., 0..., 0..., 1].asImaginary()
        return MLXFFT.irfftn(complex, s: [h, wd], axes: [1, 2], norm: .ortho).asType(dtype)
    }

    private func spectralTransform(_ xg: MLXArray, _ p: String) -> MLXArray {
        let x = relu(conv(xg, p + ".conv1.0", bias: true))
        return conv(x + fourierUnit(x, p + ".fu"), p + ".conv2", bias: false)
    }

    /// FFC_BN_ACT with its BatchNorms folded: bn_l's affine lives in convl2l's bias, bn_g's in convl2g's.
    private func ffc(_ xl: MLXArray, _ xg: MLXArray?, _ p: String, global: Bool, toGlobal: Bool, k: Int, stride: Int) -> (MLXArray, MLXArray?) {
        let pad = (k - 1) / 2
        let pl = Self.reflectPad(xl, pad)
        var outL = conv(pl, p + ".ffc.convl2l", bias: true, stride: stride)
        // swiftlint:disable:next shorthand_operator
        if global, let xg { outL = outL + conv(Self.reflectPad(xg, pad), p + ".ffc.convg2l", bias: false, stride: stride) }
        var outG: MLXArray?
        if toGlobal {
            var g = conv(pl, p + ".ffc.convl2g", bias: true, stride: stride)
            // swiftlint:disable:next shorthand_operator
            if global, let xg { g = g + spectralTransform(xg, p + ".ffc.convg2g") }
            outG = relu(g)
        }
        return (relu(outL), outG)
    }

    private func convT(_ x: MLXArray, _ k: String) -> MLXArray {
        MLX.convTransposed2d(x, a(k + ".weight"), stride: .init(2), padding: .init(1), outputPadding: .init(1)) + a(k + ".bias")
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        captured = [:]
        // model.0 is ReflectionPad2d(3) before the unpadded 7x7 FFC of model.1: the same as ffc's own reflect pad of 3.
        var (xl, xg) = ffc(input.asType(dtype), nil, "model.1", global: false, toGlobal: false, k: 7, stride: 1)
        keep("model.1.0", xl)
        (xl, xg) = ffc(xl, nil, "model.2", global: false, toGlobal: false, k: 3, stride: 2); keep("model.2.0", xl)
        (xl, xg) = ffc(xl, nil, "model.3", global: false, toGlobal: false, k: 3, stride: 2); keep("model.3.0", xl)
        (xl, xg) = ffc(xl, nil, "model.4", global: false, toGlobal: true, k: 3, stride: 2); keep("model.4.0", xl); keep("model.4.1", xg)
        for i in 5...22 {
            let p = "model.\(i)"
            var (l, g) = ffc(xl, xg, p + ".conv1", global: true, toGlobal: true, k: 3, stride: 1)
            (l, g) = ffc(l, g, p + ".conv2", global: true, toGlobal: true, k: 3, stride: 1)
            // swiftlint:disable:next shorthand_operator
            xl = xl + l; xg = xg! + g!
            if [5, 13, 22].contains(i) { keep(p + ".0", xl); keep(p + ".1", xg) }
        }
        var x = MLX.concatenated([xl, xg!], axis: -1); keep("model.23", x)
        // Folded ConvTranspose + BatchNorm (model.25/28/31 are the BatchNorm outputs, before their ReLU).
        for (t, b) in [(24, 25), (27, 28), (30, 31)] {
            x = convT(x, "model.\(t)"); keep("model.\(b)", x); x = relu(x)
        }
        let out = MLX.sigmoid(conv(Self.reflectPad(x, 3), "model.34", bias: true)); keep("predicted", out)
        return out
    }
}
