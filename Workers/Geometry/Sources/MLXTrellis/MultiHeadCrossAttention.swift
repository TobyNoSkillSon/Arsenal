// swiftlint:disable all
// Vendored inference/geometry source; see PORTED-FROM.md.
import Foundation
import MLX
import MLXNN
import MLXFast

/// Multi-head cross-attention: queries from x, keys/values from context (the
/// DINOv3 cond tokens). QK-RMS-norm, NO RoPE. Matches trellis2 cross-attn.
/// Parity target: goldens/xattn_*.
public struct MultiHeadCrossAttention {
    public let toQB, toKvB, qGamma, kGamma, toOutB: MLXArray
    public let toQW, toKvW, toOutW: Linear
    public let numHeads: Int, headDim: Int

    public init(
        toQW: Linear, toQB: MLXArray, toKvW: Linear, toKvB: MLXArray,
        qGamma: MLXArray, kGamma: MLXArray, toOutW: Linear, toOutB: MLXArray, numHeads: Int
    ) {
        self.toQW = toQW; self.toQB = toQB; self.toKvW = toKvW; self.toKvB = toKvB
        self.qGamma = qGamma; self.kGamma = kGamma; self.toOutW = toOutW; self.toOutB = toOutB
        self.numHeads = numHeads; self.headDim = qGamma.dim(1)
    }

    private func qkNorm(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {
        trellisQKNorm(x, gamma: gamma, numHeads: numHeads, headDim: headDim)
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1), Lc = context.dim(1)
        var q = (toQW(x.asType(.float16)) + toQB).reshaped([B, L, numHeads, headDim])
        let kv = (toKvW(context.asType(.float16)) + toKvB).reshaped([B, Lc, 2, numHeads, headDim])
        var k = kv[0..., 0..., 0]
        let v = kv[0..., 0..., 1]
        q = qkNorm(q, qGamma); k = qkNorm(k, kGamma)
        let qt = q.transposed(0, 2, 1, 3), kt = k.transposed(0, 2, 1, 3), vt = v.transposed(0, 2, 1, 3)
        let scale = Float(1.0 / Double(headDim).squareRoot())
        let cast = TRELLIS2Config.sdpaCastDtype
        let (qs, ks, vs) = cast.map { (qt.asType($0), kt.asType($0), vt.asType($0)) } ?? (qt, kt, vt)
        let oRaw = MLXFast.scaledDotProductAttention(queries: qs, keys: ks, values: vs, scale: scale, mask: nil)
        let o = oRaw.asType(.float16)
        let oo = o.transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])
        return toOutW(oo.asType(.float16)) + toOutB
    }
}

// swiftlint:enable all
