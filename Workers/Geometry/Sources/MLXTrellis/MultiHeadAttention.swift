// swiftlint:disable all
// Vendored inference/geometry source; see PORTED-FROM.md.
import Foundation
import MLX
import MLXNN
import MLXFast

/// Multi-head self-attention with QK-RMS-norm + RoPE + SDPA — matches trellis2
/// `MultiHeadAttention` (full mode). Order: to_qkv → QK-RMS-norm → RoPE → SDPA → to_out.
/// QK "RMS norm" is actually L2-normalize per head-dim × learned per-head gamma × √headDim
/// (no mean, no eps beyond the normalize floor). Parity target: goldens/attn_*.
// Accumulate squared norms in float to avoid FP16 overflow, then return native half activations.
func trellisQKNorm(_ x: MLXArray, gamma: MLXArray, numHeads: Int, headDim: Int) -> MLXArray {
    let f = x.asType(.float32)
    let denominator = MLX.maximum(MLX.sqrt((f * f).sum(axis: -1, keepDims: true)), 1e-6)
    return
        (f / denominator * gamma.asType(.float32).reshaped([1, 1, numHeads, headDim])
        * Float(Double(headDim).squareRoot())).asType(x.dtype)
}

public struct MultiHeadAttention {
    public let toQkvB, qGamma, kGamma, toOutB: MLXArray
    public let toQkvW, toOutW: Linear
    public let numHeads: Int, headDim: Int

    public init(
        toQkvW: Linear, toQkvB: MLXArray, qGamma: MLXArray, kGamma: MLXArray,
        toOutW: Linear, toOutB: MLXArray, numHeads: Int
    ) {
        self.toQkvW = toQkvW; self.toQkvB = toQkvB
        self.qGamma = qGamma; self.kGamma = kGamma
        self.toOutW = toOutW; self.toOutB = toOutB
        self.numHeads = numHeads
        self.headDim = qGamma.dim(1)
    }

    private func qkNorm(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {
        trellisQKNorm(x, gamma: gamma, numHeads: numHeads, headDim: headDim)
    }

    public func callAsFunction(_ h: MLXArray, phases: MLXArray) -> MLXArray {
        let B = h.dim(0), L = h.dim(1)
        let qkv = toQkvW(h.asType(.float16)) + toQkvB // [B,L,3*H*Dh]
        let r = qkv.reshaped([B, L, 3, numHeads, headDim])
        var q = r[0..., 0..., 0] // [B,L,H,Dh]
        var k = r[0..., 0..., 1]
        let v = r[0..., 0..., 2]
        q = qkNorm(q, qGamma); k = qkNorm(k, kGamma)
        q = RoPE.apply(q, phasesCosSin: phases); k = RoPE.apply(k, phasesCosSin: phases)
        // [B,L,H,Dh] -> [B,H,L,Dh] for SDPA
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
