// SPDX-License-Identifier: MIT
// FLUX.2 Klein 4B MMDiT (5 double-stream + 20 single-stream blocks, 24 heads × 128, joint dim 7680, 4-axis RoPE θ 2000).
// Adapted from xocialize/flux2-klein-swift (MIT, commit d8a9fa2), itself an op-for-op port of mflux's flux2 transformer;
// the diffusers Flux2Transformer2DModel is the reference it is gated against (gate/). Changes for Arsenal: RoPE tables
// and timestep handling are computed once per image by the caller, no host syncs inside the forward, and the activation
// precision is an explicit choice (see `Flux2Transformer.callAsFunction`).
import Foundation
import MLX
import MLXFast
import MLXNN

public enum KleinConfig {
    public static let inChannels = 128
    public static let numLayers = 5
    public static let numSingleLayers = 20
    public static let numHeads = 24
    public static let headDim = 128
    public static let innerDim = 3072
    public static let jointAttentionDim = 7680
    public static let mlpRatio: Float = 3.0
    public static let ropeTheta: Double = 2000
    public static let axesDimsRope = [32, 32, 32, 32]
    public static let timestepGuidanceChannels = 256
}

/// Real (cos, sin) tables per axis, concatenated over the 4 axes: [N, headDim / 2].
public enum Flux2Rope {
    public static func tables(_ ids: MLXArray, dtype: DType = .bfloat16) -> (cos: MLXArray, sin: MLXArray) {
        let pos = ids.asType(dtype)
        var cosParts: [MLXArray] = [], sinParts: [MLXArray] = []
        for (axis, dim) in KleinConfig.axesDimsRope.enumerated() {
            let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float16($0) / Float16(dim) }).asType(dtype)
            let omega = 1 / MLX.pow(MLXArray(Float16(KleinConfig.ropeTheta)).asType(dtype), scale)
            let out = pos[0..., axis][0..., .newAxis] * omega[.newAxis, 0...]
            cosParts.append(MLX.cos(out)); sinParts.append(MLX.sin(out))
        }
        return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))
    }
}

/// Interleaved-pair rotation in half precision: x [B, H, S, D], cos/sin [S, D/2].
func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let shape = x.shape
    let x2 = x.reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
    let real = x2[.ellipsis, 0], imag = x2[.ellipsis, 1]
    let c = cos.asType(x.dtype).reshaped(1, 1, cos.dim(0), cos.dim(1)), s = sin.asType(x.dtype).reshaped(1, 1, sin.dim(0), sin.dim(1))
    return stacked([real * c - imag * s, imag * c + real * s], axis: -1).reshaped(shape).asType(x.dtype)
}

public final class Flux2TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(embeddingDim: Int) {
        _linear1.wrappedValue = Linear(KleinConfig.timestepGuidanceChannels, embeddingDim, bias: false)
        _linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
    }
    /// diffusers Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0): [cos | sin].
    static func sinusoid(_ t: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let freqs = MLX.exp(-Float(Foundation.log(10000.0)) * MLXArray(0..<half).asType(t.dtype) / Float(half))
        let args = t[0..., .newAxis] * freqs[.newAxis, 0...]
        return concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    }
    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let t = Self.sinusoid(timestep, dim: KleinConfig.timestepGuidanceChannels)
        return linear2(silu(linear1(t)))
    }
}

/// SiLU → Linear(dim → dim·3·sets): `sets` tuples of (shift, scale, gate), each [B, 1, dim].
public final class Flux2Modulation: Module {
    let sets: Int
    @ModuleInfo var linear: Linear
    init(dim: Int, sets: Int) {
        self.sets = sets
        _linear.wrappedValue = Linear(dim, dim * 3 * sets, bias: false)
    }
    func callAsFunction(_ temb: MLXArray) -> [(shift: MLXArray, scale: MLXArray, gate: MLXArray)] {
        let parts = split(linear(silu(temb))[0..., .newAxis, 0...], parts: 3 * sets, axis: -1)
        return (0..<sets).map { (parts[3 * $0], parts[3 * $0 + 1], parts[3 * $0 + 2]) }
    }
}

public final class Flux2FeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    init(dim: Int) {
        let inner = Int(Float(dim) * KleinConfig.mlpRatio)
        _linearIn.wrappedValue = Linear(dim, inner * 2, bias: false)
        _linearOut.wrappedValue = Linear(inner, dim, bias: false)
    }
    static func swiglu(_ x: MLXArray) -> MLXArray { let p = split(x, parts: 2, axis: -1); return silu(p[0]) * p[1] }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linearOut(Self.swiglu(linearIn(x))) }
}

/// [B, S, H·D] → [B, H, S, D] with per-head RMSNorm in half precision.
func heads(_ x: MLXArray, norm: RMSNorm?) -> MLXArray {
    let (b, s) = (x.dim(0), x.dim(1))
    let h = x.reshaped(b, s, KleinConfig.numHeads, KleinConfig.headDim).transposed(0, 2, 1, 3)
    guard let norm else { return h }
    return norm(h)
}

func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
    let out = MLXFast.scaledDotProductAttention(
        queries: q, keys: k, values: v, scale: 1 / Float(KleinConfig.headDim).squareRoot(), mask: .none)
    return out.transposed(0, 2, 1, 3).reshaped(out.dim(0), -1, KleinConfig.innerDim)
}

public final class Flux2Attention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "add_q_proj") var addQ: Linear
    @ModuleInfo(key: "add_k_proj") var addK: Linear
    @ModuleInfo(key: "add_v_proj") var addV: Linear
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear
    init(dim: Int) {
        let inner = KleinConfig.innerDim, hd = KleinConfig.headDim
        _toQ.wrappedValue = Linear(dim, inner, bias: false); _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false); _toOut.wrappedValue = Linear(inner, dim, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: hd, eps: 1e-5); _normK.wrappedValue = RMSNorm(dimensions: hd, eps: 1e-5)
        _normAddedQ.wrappedValue = RMSNorm(dimensions: hd, eps: 1e-5); _normAddedK.wrappedValue = RMSNorm(dimensions: hd, eps: 1e-5)
        _addQ.wrappedValue = Linear(dim, inner, bias: false); _addK.wrappedValue = Linear(dim, inner, bias: false)
        _addV.wrappedValue = Linear(dim, inner, bias: false); _toAddOut.wrappedValue = Linear(inner, dim, bias: false)
    }
    /// Joint attention over [txt, img]; returns (img, txt).
    func callAsFunction(_ x: MLXArray, encoder e: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let q = applyRope(concatenated([heads(addQ(e), norm: normAddedQ), heads(toQ(x), norm: normQ)], axis: 2), cos: cos, sin: sin)
        let k = applyRope(concatenated([heads(addK(e), norm: normAddedK), heads(toK(x), norm: normK)], axis: 2), cos: cos, sin: sin)
        let v = concatenated([heads(addV(e), norm: nil), heads(toV(x), norm: nil)], axis: 2)
        let out = attention(q, k, v), txt = e.dim(1)
        return (toOut(out[0..., txt..., 0...]), toAddOut(out[0..., ..<txt, 0...]))
    }
}

public final class Flux2TransformerBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo(key: "norm1_context") var norm1Context: LayerNorm
    @ModuleInfo var attn: Flux2Attention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var ff: Flux2FeedForward
    @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm
    @ModuleInfo(key: "ff_context") var ffContext: Flux2FeedForward
    init(dim: Int) {
        func ln() -> LayerNorm { LayerNorm(dimensions: dim, eps: 1e-6, affine: false) }
        _norm1.wrappedValue = ln(); _norm1Context.wrappedValue = ln(); _norm2.wrappedValue = ln(); _norm2Context.wrappedValue = ln()
        _attn.wrappedValue = Flux2Attention(dim: dim)
        _ff.wrappedValue = Flux2FeedForward(dim: dim); _ffContext.wrappedValue = Flux2FeedForward(dim: dim)
    }
    typealias Mod = (shift: MLXArray, scale: MLXArray, gate: MLXArray)
    func callAsFunction(_ x: MLXArray, encoder e: MLXArray, img: [Mod], txt: [Mod], cos: MLXArray, sin: MLXArray, trace: ((String, MLXArray) -> Void)? = nil) -> (
        MLXArray, MLXArray
    ) {
        var x = x, e = e
        let (a, ea) = attn((1 + img[0].scale) * norm1(x) + img[0].shift, encoder: (1 + txt[0].scale) * norm1Context(e) + txt[0].shift, cos: cos, sin: sin)
        x = x + img[0].gate * a // swiftlint:disable:this shorthand_operator
        e = e + txt[0].gate * ea // swiftlint:disable:this shorthand_operator
        x = x + img[1].gate * ff((1 + img[1].scale) * norm2(x) + img[1].shift) // swiftlint:disable:this shorthand_operator
        e = e + txt[1].gate * ffContext((1 + txt[1].scale) * norm2Context(e) + txt[1].shift) // swiftlint:disable:this shorthand_operator
        // The vendor's Flux2TransformerBlock clamps the fp16 context residual.
        // It can exceed fp16's finite range; the next norm must never receive infinity.
        trace?("pre-clamp", e)
        if e.dtype == .float16 { e = clip(e, min: -65504, max: 65504) }
        return (x, e)
    }
}

public final class Flux2ParallelSelfAttention: Module {
    @ModuleInfo(key: "to_qkv_mlp_proj") var toQKVMLP: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear
    let mlpHidden: Int
    init(dim: Int) {
        mlpHidden = Int(Float(dim) * KleinConfig.mlpRatio)
        _toQKVMLP.wrappedValue = Linear(dim, KleinConfig.innerDim * 3 + mlpHidden * 2, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: KleinConfig.headDim, eps: 1e-5)
        _normK.wrappedValue = RMSNorm(dimensions: KleinConfig.headDim, eps: 1e-5)
        _toOut.wrappedValue = Linear(KleinConfig.innerDim + mlpHidden, dim, bias: false)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let inner = KleinConfig.innerDim
        let parts = split(toQKVMLP(x), indices: [inner, 2 * inner, 3 * inner], axis: -1)
        let q = applyRope(heads(parts[0], norm: normQ), cos: cos, sin: sin)
        let k = applyRope(heads(parts[1], norm: normK), cos: cos, sin: sin)
        let out = attention(q, k, heads(parts[2], norm: nil))
        return toOut(concatenated([out, Flux2FeedForward.swiglu(parts[3])], axis: -1))
    }
}

public final class Flux2SingleTransformerBlock: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var attn: Flux2ParallelSelfAttention
    init(dim: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _attn.wrappedValue = Flux2ParallelSelfAttention(dim: dim)
    }
    func callAsFunction(_ x: MLXArray, mod: Flux2TransformerBlock.Mod, cos: MLXArray, sin: MLXArray, trace: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let h = x + mod.gate * attn((1 + mod.scale) * norm(x) + mod.shift, cos: cos, sin: sin)
        // Same fp16 range guard as the vendor's Flux2SingleTransformerBlock.
        trace?("pre-clamp", h)
        return h.dtype == .float16 ? clip(h, min: -65504, max: 65504) : h
    }
}

public final class AdaLayerNormContinuous: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo var norm: LayerNorm
    let dim: Int
    init(dim: Int) {
        self.dim = dim
        _linear.wrappedValue = Linear(dim, dim * 2, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    }
    func callAsFunction(_ x: MLXArray, _ temb: MLXArray) -> MLXArray {
        let emb = linear(silu(temb))
        let scale = emb[0..., ..<dim], shift = emb[0..., dim...]
        return norm(x) * (1 + scale)[0..., .newAxis, 0...] + shift[0..., .newAxis, 0...]
    }
}

public final class Flux2Transformer: Module {
    @ModuleInfo(key: "time_guidance_embed") var timeEmbed: Flux2TimeGuidanceEmbed
    @ModuleInfo(key: "double_stream_modulation_img") var modImg: Flux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var modTxt: Flux2Modulation
    @ModuleInfo(key: "single_stream_modulation") var modSingle: Flux2Modulation
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [Flux2TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleBlocks: [Flux2SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public override init() {
        let dim = KleinConfig.innerDim
        _timeEmbed.wrappedValue = Flux2TimeGuidanceEmbed(embeddingDim: dim)
        _modImg.wrappedValue = Flux2Modulation(dim: dim, sets: 2)
        _modTxt.wrappedValue = Flux2Modulation(dim: dim, sets: 2)
        _modSingle.wrappedValue = Flux2Modulation(dim: dim, sets: 1)
        _xEmbedder.wrappedValue = Linear(KleinConfig.inChannels, dim, bias: false)
        _contextEmbedder.wrappedValue = Linear(KleinConfig.jointAttentionDim, dim, bias: false)
        _blocks.wrappedValue = (0..<KleinConfig.numLayers).map { _ in Flux2TransformerBlock(dim: dim) }
        _singleBlocks.wrappedValue = (0..<KleinConfig.numSingleLayers).map { _ in Flux2SingleTransformerBlock(dim: dim) }
        _normOut.wrappedValue = AdaLayerNormContinuous(dim: dim)
        _projOut.wrappedValue = Linear(dim, KleinConfig.inChannels, bias: false)
    }

    /// One double-stream block with independent reference inputs (lab fidelity probe).
    public func componentBlock(
        _ x: MLXArray, encoder: MLXArray,
        img: [(MLXArray, MLXArray, MLXArray)], txt: [(MLXArray, MLXArray, MLXArray)],
        cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        blocks[0](x, encoder: encoder, img: img, txt: txt, cos: cos, sin: sin)
    }

    /// x [B, imgLen, 128], encoder [B, txtLen, 7680], timestep [B] on the 0…1000 scale, cos/sin [txt+img, 64].
    /// `activations` is the precision of the residual stream. The timestep embedding is always computed in half precision;
    /// half precision activations reproduce what bf16 weights do under MLX promotion (the stream stays half precision), bf16 keeps it in the
    /// weights' dtype like the vendor's bf16 pipeline. The gate (gate/) decides which one ships.
    public func callAsFunction(
        _ x: MLXArray, encoder: MLXArray, timestep: MLXArray, cos: MLXArray, sin: MLXArray, activations: DType, trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let temb = timeEmbed(timestep.asType(activations)).asType(activations)
        trace?("time", temb)
        var x = xEmbedder(x.asType(activations)), e = contextEmbedder(encoder.asType(activations))
        trace?("image-projection", x); trace?("context-projection", e)
        let img = modImg(temb), txt = modTxt(temb)
        for (i, block) in blocks.enumerated() {
            (x, e) = block(x, encoder: e, img: img, txt: txt, cos: cos, sin: sin, trace: trace.map { f in { n, a in f("double-\(i)-" + n, a) } }); trace?("double-\(i)-image", x)
            trace?("double-\(i)-text", e)
        }
        let txtLen = e.dim(1)
        var h = concatenated([e, x], axis: 1)
        let single = modSingle(temb)[0]
        for (i, block) in singleBlocks.enumerated() {
            h = block(h, mod: single, cos: cos, sin: sin, trace: trace.map { f in { n, a in f("single-\(i)-" + n, a) } }); trace?("single-\(i)", h)
        }
        return projOut(normOut(h[0..., txtLen..., 0...], temb))
    }
}

/// diffusers `time_guidance_embed` (guidance_embeds = false for Klein): only the timestep embedder.
public final class Flux2TimeGuidanceEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: Flux2TimestepEmbedding
    init(embeddingDim: Int) { _timestepEmbedder.wrappedValue = Flux2TimestepEmbedding(embeddingDim: embeddingDim) }
    func callAsFunction(_ t: MLXArray) -> MLXArray { timestepEmbedder(t) }
}
