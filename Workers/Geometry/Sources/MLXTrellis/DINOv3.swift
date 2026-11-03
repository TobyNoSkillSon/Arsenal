// swiftlint:disable all
// Vendored inference/geometry source; see PORTED-FROM.md.
import Foundation
import MLX
import MLXNN
import MLXFast

/// DINOv3 ViT-L/16 image-conditioning encoder (facebook/dinov3-vitl16-pretrain-lvd1689m,
/// as wrapped by trellis2 DinoV3FeatureExtractor).
///
/// forward: patch-conv (16×16 stride-16) → prepend [cls, 4×register] → 24 encoder
/// layers (affine LN eps 1e-5 → MHA with axial-2D RoPE on patch tokens only + LayerScale,
/// then affine LN → GELU-exact MLP + LayerScale, both residual) → take layer-23 output
/// (BEFORE the model's final norm) → affine-less LayerNorm. Output [1, 1029, 1024]
/// = 1 CLS + 4 register + 1024 patch tokens, D=1024.
///
/// Differs from the DiT ports: RoPE here is the GPT-NeoX split-half `rotate_half`
/// (NOT interleaved), attention has NO QK-norm, and k_proj has NO bias. Parity: goldens/dino_*.
public struct DINOv3 {
    // config
    let hidden: Int, numHeads: Int, headDim: Int, numLayers: Int
    let patchSize = 16, numPrefix = 5 // 1 cls + 4 register
    let ropeBase: Float = 100.0
    let lnEps: Float = 1e-5

    // embeddings
    let patchW, patchB, clsToken, registerTokens: MLXArray
    // per-layer weights
    let norm1W, norm1B, norm2W, norm2B: [MLXArray]
    let qW, kW, vW, oW: [Linear]
    let qB, vB, oB: [MLXArray]
    let ls1, ls2: [MLXArray] // LayerScale lambda1
    let upW, downW: [Linear]
    let upB, downB: [MLXArray]

    public init(weights raw: TrellisWeights, hidden: Int = 1024, numHeads: Int = 16, numLayers: Int = 24) {
        self.hidden = hidden; self.numHeads = numHeads; self.headDim = hidden / numHeads; self.numLayers = numLayers
        func g(_ k: String) -> MLXArray { raw[k]! }
        func m(_ k: String) -> Linear { raw.matrix(k) }
        // Conv weight [Cout,Cin,kH,kW] -> MLX conv2d expects [Cout,kH,kW,Cin].
        patchW = g("embeddings.patch_embeddings.weight").transposed(0, 2, 3, 1)
        patchB = g("embeddings.patch_embeddings.bias")
        clsToken = g("embeddings.cls_token").reshaped([1, 1, hidden])
        registerTokens = g("embeddings.register_tokens").reshaped([1, 4, hidden])

        var n1W = [MLXArray](), n1B = [MLXArray](), n2W = [MLXArray](), n2B = [MLXArray]()
        var qw = [Linear](), qb = [MLXArray](), kw = [Linear](), vw = [Linear](), vb = [MLXArray]()
        var ow = [Linear](), ob = [MLXArray](), l1 = [MLXArray](), l2 = [MLXArray]()
        var uw = [Linear](), ub = [MLXArray](), dw = [Linear](), db = [MLXArray]()
        for i in 0..<numLayers {
            let p = "layer.\(i)."
            n1W.append(g(p + "norm1.weight")); n1B.append(g(p + "norm1.bias"))
            n2W.append(g(p + "norm2.weight")); n2B.append(g(p + "norm2.bias"))
            qw.append(m(p + "attention.q_proj.weight")); qb.append(g(p + "attention.q_proj.bias"))
            kw.append(m(p + "attention.k_proj.weight")) // key_bias=false
            vw.append(m(p + "attention.v_proj.weight")); vb.append(g(p + "attention.v_proj.bias"))
            ow.append(m(p + "attention.o_proj.weight")); ob.append(g(p + "attention.o_proj.bias"))
            l1.append(g(p + "layer_scale1.lambda1")); l2.append(g(p + "layer_scale2.lambda1"))
            uw.append(m(p + "mlp.up_proj.weight")); ub.append(g(p + "mlp.up_proj.bias"))
            dw.append(m(p + "mlp.down_proj.weight")); db.append(g(p + "mlp.down_proj.bias"))
        }
        norm1W = n1W; norm1B = n1B; norm2W = n2W; norm2B = n2B
        qW = qw; qB = qb; kW = kw; vW = vw; vB = vb; oW = ow; oB = ob
        ls1 = l1; ls2 = l2; upW = uw; upB = ub; downW = dw; downB = db
    }

    // --- axial-2D RoPE (matches DINOv3ViTRopePositionEmbedding, float32) ---
    /// Returns (cos, sin) each [numPatches, headDim] for an H×W patch grid.
    public func ropeCosSin(_ nH: Int, _ nW: Int) -> (MLXArray, MLXArray) {
        let nf = headDim / 4 // head_dim/4 frequencies
        let step = 4.0 / Float(headDim)
        var invFreq = [Float](repeating: 0, count: nf)
        for i in 0..<nf { invFreq[i] = 1.0 / powf(ropeBase, Float(i) * step) }
        let twoPi = Float(2.0 * Double.pi)
        // patch center coords in [-1,1]: coord_y from H, coord_x from W (meshgrid ij).
        var angles = [Float](repeating: 0, count: nH * nW * headDim)
        var idx = 0
        for h in 0..<nH {
            let cy = 2.0 * ((Float(h) + 0.5) / Float(nH)) - 1.0
            for w in 0..<nW {
                let cx = 2.0 * ((Float(w) + 0.5) / Float(nW)) - 1.0
                // [y*f0..y*f_{nf-1}, x*f0..x*f_{nf-1}] then tile(2)
                for t in 0..<2 {
                    for f in 0..<nf { angles[idx + t * (2 * nf) + f] = twoPi * cy * invFreq[f] }
                    for f in 0..<nf { angles[idx + t * (2 * nf) + nf + f] = twoPi * cx * invFreq[f] }
                }
                idx += headDim
            }
        }
        let a = MLXArray(angles).reshaped([nH * nW, headDim])
        return (MLX.cos(a), MLX.sin(a))
    }

    /// Apply RoPE to patch tokens only (prefix cls+register untouched).
    /// x: [B, H, L, Dh]; cos/sin: [Lp, Dh].
    private func applyRope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
        let L = x.dim(2), Dh = x.dim(3), half = Dh / 2
        let prefix = x[0..., 0..., 0..<numPrefix, 0...]
        let patches = x[0..., 0..., numPrefix..<L, 0...] // [B,H,Lp,Dh]
        let c = cos.asType(x.dtype).reshaped([1, 1, cos.dim(0), Dh])
        let s = sin.asType(x.dtype).reshaped([1, 1, sin.dim(0), Dh])
        let x1 = patches[0..., 0..., 0..., 0..<half]
        let x2 = patches[0..., 0..., 0..., half...]
        let rot = MLX.concatenated([-x2, x1], axis: -1) // rotate_half
        let rotated = patches * c + rot * s
        return MLX.concatenated([prefix, rotated], axis: 2)
    }

    private func attention(_ h: MLXArray, layer i: Int, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let B = h.dim(0), L = h.dim(1)
        var q = qW[i](h.asType(.float16)) + qB[i]
        var k = kW[i](h.asType(.float16)) // no bias
        let v = vW[i](h.asType(.float16)) + vB[i]
        q = q.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let vt = v.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        q = applyRope(q, cos, sin)
        k = applyRope(k, cos, sin)
        let scale = 1.0 / Float(headDim).squareRoot()
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: vt, scale: scale, mask: nil)
        let merged = o.transposed(0, 2, 1, 3).reshaped([B, L, hidden])
        return oW[i](merged.asType(.float16)) + oB[i]
    }

    /// The MLP output is returned in `dtype`. Layer 0's down projection alone reaches about 5e4 on an ordinary product
    /// photo, within 1.3x of FP16's limit, so the FP16 matmul runs on input scaled by an exact 1/8 and is rescaled wider.
    private func mlp(_ h: MLXArray, layer i: Int, dtype: DType) -> MLXArray {
        let up = upW[i](h.asType(.float16)) + upB[i]
        let act = geluExact(up)
        return downW[i]((act * Float(0.125)).asType(.float16)).asType(dtype) * 8 + downB[i].asType(dtype)
    }

    /// Patch embed + prepend cls/register tokens -> [1, 1+numReg+numPatch, 1024].
    func embed(_ pixels: MLXArray) -> MLXArray {
        let nH = pixels.dim(2) / patchSize, nW = pixels.dim(3) / patchSize
        // patch embed: NCHW -> NHWC conv2d(stride patchSize) -> [1, nH, nW, 1024]
        let nhwc = pixels.transposed(0, 2, 3, 1)
        var pe = conv2d(nhwc, patchW, stride: IntOrPair(patchSize), padding: IntOrPair(0))
        pe = pe.reshaped([1, nH * nW, hidden]) + patchB // token = h*nW + w
        let B = pe.dim(0)
        let cls = MLX.broadcast(clsToken, to: [B, 1, hidden])
        let reg = MLX.broadcast(registerTokens, to: [B, 4, hidden])
        return MLX.concatenated([cls, reg, pe], axis: 1) // [1, 1029, 1024]
    }

    /// One encoder layer. Matmuls run in FP16; LayerScale and the residual add run in `h0`'s dtype. The trained DINOv3-L
    /// residual stream exceeds FP16's 65,504 range from layer 0 on ordinary photos (MLP output about 5e4 times
    /// layer_scale2 gives about 1.5e5), so the encoder carries it in float32; in FP16 it becomes inf, the final LayerNorm
    /// turns that into NaN, and the NaN condition leaves the sparse-structure decoder with no positive logit.
    func block(_ h0: MLXArray, layer i: Int, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let dtype = h0.dtype
        var h = h0
        let a = attention(layerNorm32(h, weight: norm1W[i], bias: norm1B[i], eps: lnEps).asType(.float16), layer: i, cos: cos, sin: sin)
        h = h + a.asType(dtype) * ls1[i].asType(dtype) // LayerScale
        let m = mlp(layerNorm32(h, weight: norm2W[i], bias: norm2B[i], eps: lnEps).asType(.float16), layer: i, dtype: dtype)
        return h + m * ls2[i].asType(dtype)
    }

    /// x: [1, 3, H, W] normalized pixels. Returns [1, 1+numReg+numPatch, 1024].
    public func callAsFunction(_ pixels: MLXArray, checkpoint: () throws -> Void = {}) throws -> MLXArray {
        let nH = pixels.dim(2) / patchSize, nW = pixels.dim(3) / patchSize
        var h = embed(pixels).asType(.float32) // float32 residual stream; see `block`
        let (cos, sin) = ropeCosSin(nH, nW)
        for i in 0..<numLayers {
            try checkpoint(); h = block(h, layer: i, cos: cos, sin: sin); MLX.eval(h)
        }
        // hidden_states[-1] (pre model's final norm) then affine-less LayerNorm, returned as the FP16 condition.
        return layerNorm32(h, eps: lnEps).asType(.float16)
    }

    /// Intermediates for localization gating: (patch_embed, after_blk0, after_blk23, cond).
    public func forwardStages(_ pixels: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let nH = pixels.dim(2) / patchSize, nW = pixels.dim(3) / patchSize
        let emb = embed(pixels)
        let (cos, sin) = ropeCosSin(nH, nW)
        var h = emb.asType(.float32)
        var afterBlk0 = emb
        for i in 0..<numLayers {
            h = block(h, layer: i, cos: cos, sin: sin)
            if i == 0 { afterBlk0 = h }
        }
        let cond = layerNorm32(h, eps: lnEps).asType(.float16)
        return (emb, afterBlk0, h, cond)
    }
}

/// Exact (erf-based) GELU — ACT2FN["gelu"], NOT the tanh approximation.
func geluExact(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1 + MLX.erf(x / Float(2).squareRoot()))
}

// swiftlint:enable all
