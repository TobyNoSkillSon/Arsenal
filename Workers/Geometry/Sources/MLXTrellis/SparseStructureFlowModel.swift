// swiftlint:disable all
// Vendored inference/geometry source; see PORTED-FROM.md.
import Foundation
import MLX
import MLXNN

/// Dense 3D DiT for sparse-structure generation (SparseStructureFlowModel).
/// forward: patchify(16³ voxel grid → 4096 tokens) → input Linear → t_embedder
/// + shared adaLN(SiLU→Linear→9216) → 30× ModulatedTransformerCrossBlock →
/// final affine-less LayerNorm → out Linear → unpatchify. Parity: goldens/ssdit_*.
public struct SparseStructureFlowModel {
    public let inputB, adaLNB, outB: MLXArray
    public let inputW, adaLNW, outW: Linear
    public let tEmbedder: TimestepEmbedder
    public let blocks: [ModulatedTransformerCrossBlock]
    public let resolution: Int
    public let numHeads: Int

    public init(weights w: TrellisWeights, numBlocks: Int = 30, resolution: Int = 16, numHeads: Int = 12) {
        func g(_ k: String) -> MLXArray { w[k]! }
        func m(_ k: String) -> Linear { w.matrix(k) }
        let inW = m("input_layer.weight")
        let C = inW.weight.dim(0)
        inputW = inW; inputB = g("input_layer.bias")
        adaLNW = m("adaLN_modulation.1.weight"); adaLNB = g("adaLN_modulation.1.bias")
        outW = m("out_layer.weight"); outB = g("out_layer.bias")
        tEmbedder = TimestepEmbedder(
            w0: m("t_embedder.mlp.0.weight"), b0: g("t_embedder.mlp.0.bias"),
            w2: m("t_embedder.mlp.2.weight"), b2: g("t_embedder.mlp.2.bias"))
        self.resolution = resolution; self.numHeads = numHeads
        blocks = (0..<numBlocks).map { i in
            let p = "blocks.\(i)."
            return ModulatedTransformerCrossBlock(
                modulation: g(p + "modulation"), norm2W: g(p + "norm2.weight"), norm2B: g(p + "norm2.bias"),
                selfAttn: MultiHeadAttention(
                    toQkvW: m(p + "self_attn.to_qkv.weight"), toQkvB: g(p + "self_attn.to_qkv.bias"),
                    qGamma: g(p + "self_attn.q_rms_norm.gamma"), kGamma: g(p + "self_attn.k_rms_norm.gamma"),
                    toOutW: m(p + "self_attn.to_out.weight"), toOutB: g(p + "self_attn.to_out.bias"), numHeads: numHeads),
                crossAttn: MultiHeadCrossAttention(
                    toQW: m(p + "cross_attn.to_q.weight"), toQB: g(p + "cross_attn.to_q.bias"),
                    toKvW: m(p + "cross_attn.to_kv.weight"), toKvB: g(p + "cross_attn.to_kv.bias"),
                    qGamma: g(p + "cross_attn.q_rms_norm.gamma"), kGamma: g(p + "cross_attn.k_rms_norm.gamma"),
                    toOutW: m(p + "cross_attn.to_out.weight"), toOutB: g(p + "cross_attn.to_out.bias"), numHeads: numHeads),
                mlp: FeedForwardNet(
                    w0: m(p + "mlp.mlp.0.weight"), b0: g(p + "mlp.mlp.0.bias"),
                    w2: m(p + "mlp.mlp.2.weight"), b2: g(p + "mlp.mlp.2.bias")),
                channels: C)
        }
    }

    /// x: [B, Cin, R, R, R]; t: [B]; cond: [B, N, D]; phases: [L, headDim/2, 2].
    public func callAsFunction(_ x: MLXArray, t: MLXArray, cond: MLXArray, phases: MLXArray, checkpoint: () throws -> Void = {}) rethrows -> MLXArray {
        let B = x.dim(0), Cin = x.dim(1)
        var h = x.reshaped([B, Cin, -1]).transposed(0, 2, 1) // patchify -> [B, L, Cin]
        h = inputW(h.asType(.float16)) + inputB // [B, L, C]
        var tEmb = tEmbedder(t) // [B, C]
        tEmb = adaLNW(tEmb * MLX.sigmoid(tEmb).asType(.float16)) + adaLNB // SiLU→Linear -> [B, 6C]
        for blk in blocks {
            // Each block is materialized so cancellation is observed inside a sampling step.
            try checkpoint(); h = blk(h, mod: tEmb, context: cond, phases: phases); MLX.eval(h)
        }
        try checkpoint()
        h = layerNorm32(h) // final (no affine)
        h = outW(h.asType(.float16)) + outB // [B, L, Cout]
        let R = resolution
        return h.transposed(0, 2, 1).reshaped([B, h.dim(2), R, R, R]) // unpatchify
    }
}

// swiftlint:enable all
