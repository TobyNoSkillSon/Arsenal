import Foundation
import MLX
import MLXNN
import MLXFast

// Adapted from soniqo/speech-swift `Sources/StableAudio3MusicGen/DiTMedium.swift`
// (Apache-2.0, commit 2af9c6c). Vireo changes:
//  * every projection is a plain `Linear`, so the official F16 weights
//    (`stabilityai/stable-audio-3-optimized/MLX/dit_medium_f16.npz`) load as-is;
//    int8/int4 bundles are handled by `quantize(model:)` before the update
//    (see `sa3LoadDiTMedium`), instead of hard-wiring `QuantizedLinear`;
//  * dimensions come from `DiTMediumConfig`, so tests can build a tiny DiT on CPU;
//  * `DiTMediumConfig.small` covers the small DiTs (`dit_mlx.py`: sm-sfx, sm-music), which
//    differ only in size and in plain (not differential) self- and cross-attention;
//  * the zero local-add conditioning is built at the *input* length, not a
//    `tLat` baked at load time (upstream crashed for any length other than the
//    load hint; Stability's Python reference uses the input length too), and in
//    fp32 like the reference (upstream used fp16, so activations never promoted).

/// Shape of an SA3 DiT. `.medium` mirrors `dit_mlx_medium.py`, `.small` mirrors `dit_mlx.py`.
public struct DiTMediumConfig: Sendable, Equatable {
    public var ioChannels = 256
    public var embedDim = 1536
    public var depth = 24
    public var numHeads = 24
    public var headDim = 64
    public var ropeDims = 32
    public var condTokenDim = 768
    public var globalCondDim = 768
    public var localAddCondDim = 257
    public var numMemoryTokens = 64
    public var ffInner = 6144
    public var timestepFeatDim = 256
    public var normEps: Float = 1e-5
    public var qkNormEps: Float = 1e-6
    public var ropeBase: Float = 10000.0
    /// Differential attention (medium: 5-way QKV, 2-way Q and 3-way KV in cross-attention, two SDPAs subtracted).
    /// The small DiTs use plain attention (3-way QKV, Q, 2-way KV, one SDPA); every other layer is the same.
    public var differential = true

    public static let medium = DiTMediumConfig()
    /// sa3-sm-sfx / sa3-sm-music (`dit_mlx.py`): embed 1024, depth 20, 16 heads, ff 4096, plain attention.
    public static let small: DiTMediumConfig = {
        var c = DiTMediumConfig()
        c.embedDim = 1024; c.depth = 20; c.numHeads = 16; c.ffInner = 4096; c.differential = false
        return c
    }()
    public init() {}
}

// MARK: - Differential attention (5-way QKV split)

/// `to_qkv(x) → q, k, v, q_diff, k_diff`; two SDPAs, subtract. RoPE on the first
/// `ropeDims` of each head; per-head RMS QK-norm. Plain (`!differential`): `q, k, v`, one SDPA.
public final class DiffSelfAttentionMedium: Module {
    @ModuleInfo(key: "to_qkv") public var toQKV: Linear
    @ModuleInfo(key: "to_out") public var toOut: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    let c: DiTMediumConfig
    public let scale: Float

    public init(_ c: DiTMediumConfig) {
        self.c = c
        let E = c.embedDim
        self._toQKV.wrappedValue = Linear(E, (c.differential ? 5 : 3) * E, bias: false)
        self._toOut.wrappedValue = Linear(E, E, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.qkNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.qkNormEps)
        self.scale = 1.0 / Float(c.headDim).squareRoot()
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let H = c.numHeads, D = c.headDim
        let parts = MLX.split(toQKV(x), parts: c.differential ? 5 : 3, axis: -1)
        func heads(_ t: MLXArray) -> MLXArray { t.reshaped([B, T, H, D]).transposed(0, 2, 1, 3) }
        func rope(_ t: MLXArray) -> MLXArray {
            MLXFast.RoPE(t, dimensions: c.ropeDims, traditional: false, base: c.ropeBase, scale: 1.0, offset: 0)
        }
        let q = rope(qNorm(heads(parts[0])))
        let k = rope(kNorm(heads(parts[1])))
        let v = heads(parts[2])
        guard c.differential else {
            let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
            return toOut(out.transposed(0, 2, 1, 3).reshaped([B, T, c.embedDim]))
        }
        let qDiff = rope(qNorm(heads(parts[3])))
        let kDiff = rope(kNorm(heads(parts[4])))
        let outMain = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let outDiff = MLXFast.scaledDotProductAttention(queries: qDiff, keys: kDiff, values: v, scale: scale, mask: nil)
        return toOut((outMain - outDiff).transposed(0, 2, 1, 3).reshaped([B, T, c.embedDim]))
    }
}

public final class DiffCrossAttentionMedium: Module {
    @ModuleInfo(key: "to_q") public var toQ: Linear        // E → 2E
    @ModuleInfo(key: "to_kv") public var toKV: Linear      // E → 3E
    @ModuleInfo(key: "to_out") public var toOut: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    let c: DiTMediumConfig
    public let scale: Float

    public init(_ c: DiTMediumConfig) {
        self.c = c
        let E = c.embedDim
        self._toQ.wrappedValue = Linear(E, (c.differential ? 2 : 1) * E, bias: false)
        self._toKV.wrappedValue = Linear(E, (c.differential ? 3 : 2) * E, bias: false)
        self._toOut.wrappedValue = Linear(E, E, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.qkNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.qkNormEps)
        self.scale = 1.0 / Float(c.headDim).squareRoot()
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let B = x.dim(0), Tx = x.dim(1), Tc = context.dim(1)
        let H = c.numHeads, D = c.headDim
        func qh(_ t: MLXArray) -> MLXArray { t.reshaped([B, Tx, H, D]).transposed(0, 2, 1, 3) }
        func ch(_ t: MLXArray) -> MLXArray { t.reshaped([B, Tc, H, D]).transposed(0, 2, 1, 3) }
        guard c.differential else {
            let kv = MLX.split(toKV(context), parts: 2, axis: -1)
            let out = MLXFast.scaledDotProductAttention(queries: qNorm(qh(toQ(x))), keys: kNorm(ch(kv[0])), values: ch(kv[1]),
                                                        scale: scale, mask: nil)
            return toOut(out.transposed(0, 2, 1, 3).reshaped([B, Tx, c.embedDim]))
        }
        let qParts = MLX.split(toQ(x), parts: 2, axis: -1)
        let kvParts = MLX.split(toKV(context), parts: 3, axis: -1)
        let q = qNorm(qh(qParts[0])), qDiff = qNorm(qh(qParts[1]))
        let k = kNorm(ch(kvParts[0])), kDiff = kNorm(ch(kvParts[1]))
        let v = ch(kvParts[2])
        let outMain = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let outDiff = MLXFast.scaledDotProductAttention(queries: qDiff, keys: kDiff, values: v, scale: scale, mask: nil)
        return toOut((outMain - outDiff).transposed(0, 2, 1, 3).reshaped([B, Tx, c.embedDim]))
    }
}

// MARK: - GeGLU feed-forward

/// `ff.0.proj(x) → split → value * silu(gate)`.
public final class GLUWrapMedium: Module {
    @ModuleInfo public var proj: Linear
    public init(_ c: DiTMediumConfig) {
        self._proj.wrappedValue = Linear(c.embedDim, 2 * c.ffInner, bias: true)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(proj(x), parts: 2, axis: -1)
        return parts[0] * silu(parts[1])
    }
}

/// Checkpoint keys `ff.ff.{0,2}` are rewritten to `ff.ff.{glu,out}` by the loader.
public final class FeedForwardMedium: Module {
    public final class Inner: Module {
        @ModuleInfo public var glu: GLUWrapMedium
        @ModuleInfo public var out: Linear
        public init(_ c: DiTMediumConfig) {
            self._glu.wrappedValue = GLUWrapMedium(c)
            self._out.wrappedValue = Linear(c.ffInner, c.embedDim, bias: true)
            super.init()
        }
    }
    @ModuleInfo public var ff: Inner
    public init(_ c: DiTMediumConfig) {
        self._ff.wrappedValue = Inner(c)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { ff.out(ff.glu(x)) }
}

/// Per-layer local-conditioning MLP: 257 → E → E with SiLU between.
/// Checkpoint keys `to_local_embed.seq.{0,2}` → `to_local_embed.seq.{inProj,outProj}`.
public final class LocalEmbedSeqMedium: Module {
    public final class Seq: Module {
        @ModuleInfo public var inProj: Linear
        @ModuleInfo public var outProj: Linear
        public init(_ c: DiTMediumConfig) {
            self._inProj.wrappedValue = Linear(c.localAddCondDim, c.embedDim, bias: true)
            self._outProj.wrappedValue = Linear(c.embedDim, c.embedDim, bias: true)
            super.init()
        }
    }
    @ModuleInfo public var seq: Seq
    public init(_ c: DiTMediumConfig) {
        self._seq.wrappedValue = Seq(c)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { seq.outProj(silu(seq.inProj(x))) }
}

// MARK: - Transformer block

public final class TransformerBlockMedium: Module {
    @ModuleInfo(key: "pre_norm") public var preNorm: RMSNorm
    @ModuleInfo(key: "self_attn") public var selfAttn: DiffSelfAttentionMedium
    @ModuleInfo(key: "cross_attend_norm") public var crossAttendNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") public var crossAttn: DiffCrossAttentionMedium
    @ModuleInfo(key: "ff_norm") public var ffNorm: RMSNorm
    @ModuleInfo public var ff: FeedForwardMedium
    @ModuleInfo(key: "to_local_embed") public var toLocalEmbed: LocalEmbedSeqMedium
    @ParameterInfo(key: "to_scale_shift_gate") public var toScaleShiftGate: MLXArray  // (6E,)

    public init(_ c: DiTMediumConfig) {
        let E = c.embedDim
        self._preNorm.wrappedValue = RMSNorm(dimensions: E, eps: c.normEps)
        self._selfAttn.wrappedValue = DiffSelfAttentionMedium(c)
        self._crossAttendNorm.wrappedValue = RMSNorm(dimensions: E, eps: c.normEps)
        self._crossAttn.wrappedValue = DiffCrossAttentionMedium(c)
        self._ffNorm.wrappedValue = RMSNorm(dimensions: E, eps: c.normEps)
        self._ff.wrappedValue = FeedForwardMedium(c)
        self._toLocalEmbed.wrappedValue = LocalEmbedSeqMedium(c)
        self._toScaleShiftGate.wrappedValue = MLXArray.zeros([6 * E])
        super.init()
    }

    public var activationTrace: ((String, MLXArray) -> Void)?
    public func callAsFunction(_ xIn: MLXArray, context: MLXArray,
                               globalCond: MLXArray, localEmbPadded: MLXArray) -> MLXArray {
        let ss = (toScaleShiftGate.asType(globalCond.dtype) + globalCond).expandedDimensions(axis: 1)
        let p = MLX.split(ss, parts: 6, axis: -1)
        let one = MLXArray(Float(1.0)).asType(ss.dtype)

        var x = xIn
        var h = preNorm(x) * (one + p[0]) + p[1]
        h = selfAttn(h) * MLX.sigmoid(one - p[2])
        x = h + x

        x = x + crossAttn(crossAttendNorm(x), context: context)
        x = x + localEmbPadded

        h = ffNorm(x) * (one + p[3]) + p[4]
        h = ff(h) * MLX.sigmoid(one - p[5])
        return h + x
    }
}

// MARK: - Top transformer + DiT

/// Two linears with SiLU between. Checkpoint `X.{0,2}` → `X.{inProj,outProj}`.
public final class TwoLinearMLP: Module {
    @ModuleInfo public var inProj: Linear
    @ModuleInfo public var outProj: Linear
    public init(inDim: Int, hidden: Int, outDim: Int, bias: Bool) {
        self._inProj.wrappedValue = Linear(inDim, hidden, bias: bias)
        self._outProj.wrappedValue = Linear(hidden, outDim, bias: bias)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { outProj(silu(inProj(x))) }
}

public final class ContinuousTransformerMedium: Module {
    @ModuleInfo(key: "project_in") public var projectIn: Linear
    @ModuleInfo(key: "project_out") public var projectOut: Linear
    @ParameterInfo(key: "memory_tokens") public var memoryTokens: MLXArray   // (M, E)
    @ModuleInfo(key: "global_cond_embedder") public var globalCondEmbedder: TwoLinearMLP
    @ModuleInfo public var layers: [TransformerBlockMedium]

    let c: DiTMediumConfig

    public init(_ c: DiTMediumConfig) {
        self.c = c
        let E = c.embedDim
        self._projectIn.wrappedValue = Linear(c.ioChannels, E, bias: false)
        self._projectOut.wrappedValue = Linear(E, c.ioChannels, bias: false)
        self._memoryTokens.wrappedValue = MLXArray.zeros([c.numMemoryTokens, E])
        self._globalCondEmbedder.wrappedValue = TwoLinearMLP(inDim: E, hidden: E, outDim: 6 * E, bias: true)
        self._layers.wrappedValue = (0..<c.depth).map { _ in TransformerBlockMedium(c) }
        super.init()
    }

    public var activationTrace: ((String, MLXArray) -> Void)?
    public func callAsFunction(_ xIn: MLXArray, context: MLXArray,
                               globalEmbed: MLXArray, localAddCond: MLXArray) -> MLXArray {
        let B = xIn.dim(0), E = c.embedDim, M = c.numMemoryTokens
        var x = projectIn(xIn)
        let mem = MLX.broadcast(memoryTokens.expandedDimensions(axis: 0).asType(x.dtype), to: [B, M, E])
        x = MLX.concatenated([mem, x], axis: 1)
        let g = globalCondEmbedder(globalEmbed)                           // [B, 6E]
        let pad = MLXArray.zeros([B, M, E], dtype: x.dtype)
        for (i, layer) in layers.enumerated() {
            let localEmb = layer.toLocalEmbed(localAddCond)                   // [B, T, E]
            x = layer(x, context: context, globalCond: g,
                      localEmbPadded: MLX.concatenated([pad, localEmb], axis: 1)); activationTrace?("layer-\(i)", x)
        }
        return projectOut(x[0..., M..., 0...])
    }
}

public final class DiTMedium: Module {
    @ModuleInfo(key: "preprocess_conv") public var preprocessConv: Conv1d
    @ModuleInfo(key: "postprocess_conv") public var postprocessConv: Conv1d
    @ModuleInfo(key: "to_cond_embed") public var toCondEmbed: TwoLinearMLP
    @ModuleInfo(key: "to_global_embed") public var toGlobalEmbed: TwoLinearMLP
    @ModuleInfo(key: "to_timestep_embed") public var toTimestepEmbed: TwoLinearMLP
    @ModuleInfo public var transformer: ContinuousTransformerMedium

    public let config: DiTMediumConfig
    /// dtype of the zero local conditioning when none is given. Stability's reference
    /// (`dit_mlx_medium.py`) builds it with `mx.zeros(...)`, i.e. float32, so from the
    /// first block on the residual stream is promoted to fp32 even with fp16 weights.
    /// Kept for parity; fp16 activations are a performance option to validate later.
    public var localCondDType: DType = .float32
    /// Vireo fast path candidate (inexact): timestep features in the latent dtype, so with an fp16 zero local
    /// conditioning the whole residual stream stays fp16 (set together with `localCondDType = .float16`).
    public var halfTimestepFeatures = false

    /// Vireo fast path (exact): stock promotes the residual stream to fp32, so every fp16 weight that meets it is
    /// cast to fp32 inside every call. `promoteWeights()` stores those weights (and biases) as fp32 once: the GEMMs
    /// see the same fp32 operands, bit for bit. Modules whose inputs stay fp16 (text/global conditioning, the input
    /// projection, cross-attention K/V over the fp16 text) and norms/tokens are untouched. Reversible.
    public private(set) var promotedWeights = false
    private var savedHalf: [(Module, ModuleParameters)] = []

    func promotedModules() -> [Module] {
        var list: [Module] = [toTimestepEmbed, postprocessConv, transformer.globalCondEmbedder, transformer.projectOut]
        for layer in transformer.layers {
            list += [layer.selfAttn.toQKV, layer.selfAttn.toOut, layer.crossAttn.toQ, layer.crossAttn.toOut,
                     layer.ff.ff.glu.proj, layer.ff.ff.out, layer.toLocalEmbed]
        }
        return list
    }

    public func promoteWeights() {
        guard !promotedWeights else { return }
        for module in promotedModules() {
            let params = module.parameters()
            let float = params.mapValues { $0.dtype == .float16 || $0.dtype == .bfloat16 ? $0.asType(.float32) : $0 }
            savedHalf.append((module, params))
            module.update(parameters: float)
            eval(module.parameters())
        }
        promotedWeights = true
    }

    public func restoreWeights() {
        for (module, params) in savedHalf { module.update(parameters: params) }
        savedHalf = []
        promotedWeights = false
    }

    public init(_ c: DiTMediumConfig = .medium) {
        self.config = c
        let IO = c.ioChannels, E = c.embedDim
        self._preprocessConv.wrappedValue = Conv1d(inputChannels: IO, outputChannels: IO, kernelSize: 1, bias: false)
        self._postprocessConv.wrappedValue = Conv1d(inputChannels: IO, outputChannels: IO, kernelSize: 1, bias: false)
        self._toCondEmbed.wrappedValue = TwoLinearMLP(inDim: c.condTokenDim, hidden: E, outDim: E, bias: false)
        self._toGlobalEmbed.wrappedValue = TwoLinearMLP(inDim: c.globalCondDim, hidden: E, outDim: E, bias: false)
        self._toTimestepEmbed.wrappedValue = TwoLinearMLP(inDim: c.timestepFeatDim, hidden: E, outDim: E, bias: true)
        self._transformer.wrappedValue = ContinuousTransformerMedium(c)
        super.init()
    }

    /// `x` [B, 256, T] channels-first; `t` [B] current sigma; `crossAttnCondRaw`
    /// [B, 257, 768]; `globalCondRaw` [B, 768]; `localAddCond` optional
    /// [B, T, 257] (inpainting). Returns the velocity [B, 256, T].
    public func callAsFunction(_ x: MLXArray, t: MLXArray,
                               crossAttnCondRaw: MLXArray, globalCondRaw: MLXArray,
                               localAddCond: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0), T = x.dim(2)
        let context = toCondEmbed(crossAttnCondRaw)
        let globalPre = toGlobalEmbed(globalCondRaw)
        var tf = expoFourierFeatures(t, dim: config.timestepFeatDim, minFreq: 0.5, maxFreq: 10_000)
        if halfTimestepFeatures { tf = tf.asType(x.dtype) }
        // No cast: the reference feeds fp32 features, so global_embed promotes to fp32.
        let globalEmbed = globalPre + toTimestepEmbed(tf)

        let xLC = x.transposed(0, 2, 1)                  // MLX Conv1d is [B, T, C]
        let xPP = preprocessConv(xLC) + xLC
        let local = localAddCond ?? MLXArray.zeros([B, T, config.localAddCondDim], dtype: localCondDType)
        let h = transformer(xPP, context: context, globalEmbed: globalEmbed, localAddCond: local)
        return (postprocessConv(h) + h).transposed(0, 2, 1)
    }
}
