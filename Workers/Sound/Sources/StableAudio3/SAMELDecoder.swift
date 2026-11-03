import Foundation
import MLX
import MLXNN
import MLXFast

// Adapted from soniqo/speech-swift `SAMELDecoder.swift` (Apache-2.0, commit 2af9c6c).
// Vireo change: dimensions come from `SAMELConfig` (tiny configs for CPU parity
// tests against Stability's Python reference); behaviour unchanged.

/// SAME-L decoder shape; `.large` mirrors `same_l_decoder.py`. The 16× stride
/// (17-token groups) is fixed by the SWA layout.
public struct SAMELConfig: Sendable, Equatable {
    public var latentDim = SAMELDims.latentDim
    public var dim = SAMELDims.dim
    public var numHeads = SAMELDims.numHeads
    public var headDim = SAMELDims.headDim
    public var ropeDims = SAMELDims.ropeDims
    public var numBlocks = SAMELDims.numBlocks
    public var ffInner = SAMELDims.ffInner
    public var sinStartBlock = SAMELDims.sinStartBlock
    public var outChannels = SAMELDims.outChannels
    public static let large = SAMELConfig()
    public init() {}
}

// MARK: - Static SWA mask (17 × 51 band)

/// Per-block-group SWA: a query group of 17 attends to 3 consecutive KV
/// groups (51 KV tokens via padding + strided view). Mask shape: (17, 51).
@inline(__always)
func sameLSWAMask() -> MLXArray {
    let block = SAMELDims.subChunkSize        // 17
    let q  = MLXArray(0..<block).reshaped([block, 1])
    let kv = MLXArray(0..<(3 * block)).reshaped([1, 3 * block])
    // valid = (kv >= q) && (kv <= q + 2*block)
    let lower: MLXArray = kv .>= q
    let upper: MLXArray = kv .<= (q + 2 * block)
    let valid: MLXArray = lower * upper                  // boolean AND via mult
    return MLX.where(valid,
                     MLXArray(Float(0.0)),
                     MLXArray(Float(-1e9)))
}

// MARK: - DyT normalisation (γ * tanh(α * x) + β)

public final class DyT: Module {
    @ParameterInfo public var alpha: MLXArray    // scalar (shape [1])
    @ParameterInfo public var gamma: MLXArray    // (dim,)
    @ParameterInfo public var beta:  MLXArray    // (dim,)

    public init(dim: Int) {
        self._alpha.wrappedValue = MLXArray([Float(1.0)])
        self._gamma.wrappedValue = MLXArray.ones([dim])
        self._beta.wrappedValue  = MLXArray.zeros([dim])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        gamma * MLX.tanh(alpha * x) + beta
    }
}

// MARK: - Differential sliding-window attention

public final class DiffSWA: Module {
    @ModuleInfo(key: "to_qkv") public var toQKV: Linear      // FP32 in SAME-L
    @ModuleInfo(key: "to_out") public var toOut: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: DyT
    @ModuleInfo(key: "k_norm") public var kNorm: DyT

    public let scale: Float
    let c: SAMELConfig
    /// Vireo (VIREO_SA3_SWA_VIEWS): with a 4-D mask (the decoder's precombined [G, 1, 17, 51] window mask, one clip),
    /// the windows go to SDPA as strided views (see `swaViews`).
    public var viewWindows = false

    public init(_ c: SAMELConfig) {
        self.c = c
        let D = c.dim
        self._toQKV.wrappedValue = Linear(D, 5 * D, bias: false)
        self._toOut.wrappedValue = Linear(D, D, bias: false)
        self._qNorm.wrappedValue = DyT(dim: c.headDim)
        self._kNorm.wrappedValue = DyT(dim: c.headDim)
        self.scale = 1.0 / Float(c.headDim).squareRoot()
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?, fullAttention: Bool) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let H = c.numHeads, D = c.headDim
        let qkv = toQKV(x)
        let parts = MLX.split(qkv, parts: 5, axis: -1)
        var q1 = parts[0], k1 = parts[1]
        let v0 = parts[2]
        var q2 = parts[3], k2 = parts[4]

        func toHeads(_ t: MLXArray) -> MLXArray {
            t.reshaped([B, T, H, D]).transposed(0, 2, 1, 3)
        }
        q1 = toHeads(q1); k1 = toHeads(k1)
        let v = toHeads(v0)
        q2 = toHeads(q2); k2 = toHeads(k2)

        q1 = qNorm(q1); k1 = kNorm(k1)
        q2 = qNorm(q2); k2 = kNorm(k2)

        q1 = MLXFast.RoPE(q1, dimensions: c.ropeDims, traditional: false,
                           base: SAMELDims.ropeBase, scale: 1.0, offset: 0)
        k1 = MLXFast.RoPE(k1, dimensions: c.ropeDims, traditional: false,
                           base: SAMELDims.ropeBase, scale: 1.0, offset: 0)
        q2 = MLXFast.RoPE(q2, dimensions: c.ropeDims, traditional: false,
                           base: SAMELDims.ropeBase, scale: 1.0, offset: 0)
        k2 = MLXFast.RoPE(k2, dimensions: c.ropeDims, traditional: false,
                           base: SAMELDims.ropeBase, scale: 1.0, offset: 0)

        if viewWindows, !fullAttention, T > SAMELDims.subChunkSize, B == 1, let windowMask = mask, windowMask.ndim == 4 {
            return toOut(swaViews(q1: q1, k1: k1, v: v, q2: q2, k2: k2, windowMask: windowMask))
        }
        let out: MLXArray
        if fullAttention || T <= SAMELDims.subChunkSize {
            out = diffSDPA(q1: q1, k1: k1, v: v, q2: q2, k2: k2, mask: nil)
        } else {
            out = swa(q1: q1, k1: k1, v: v, q2: q2, k2: k2, mask: mask)
        }
        let outF = out.transposed(0, 2, 1, 3).reshaped([B, T, c.dim])
        return toOut(outF)
    }

    private func diffSDPA(q1: MLXArray, k1: MLXArray, v: MLXArray,
                           q2: MLXArray, k2: MLXArray, mask: MLXArray?) -> MLXArray {
        let Q = MLX.concatenated([q1, q2], axis: 1)
        let K = MLX.concatenated([k1, k2], axis: 1)
        let V = MLX.concatenated([v, v], axis: 1)
        let out = MLXFast.scaledDotProductAttention(queries: Q, keys: K, values: V,
                                                     scale: scale, mask: mask)
        let halves = MLX.split(out, parts: 2, axis: 1)
        return halves[0] - halves[1]
    }

    private func swa(q1: MLXArray, k1: MLXArray, v: MLXArray,
                      q2: MLXArray, k2: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = q1.dim(0), H = q1.dim(1), T = q1.dim(2), D = q1.dim(3)
        let block = SAMELDims.subChunkSize
        let G = T / block
        let W = 3 * block

        // Pad along seq dim by `block` on each side. MLXArray.padded uses
        // `widths` of [(before, after), ...] per axis.
        let pad: [IntOrPair] = [
            IntOrPair((0, 0)), IntOrPair((0, 0)),
            IntOrPair((block, block)), IntOrPair((0, 0)),
        ]
        let k1p = MLX.padded(k1, widths: pad)
        let k2p = MLX.padded(k2, widths: pad)
        let vp  = MLX.padded(v,  widths: pad)

        let Tp = T + 2 * block
        let winStrides = [H * Tp * D, Tp * D, block * D, D, 1]
        let winShape   = [B, H, G, W, D]
        let k1w = MLX.asStrided(k1p, winShape, strides: winStrides)
        let k2w = MLX.asStrided(k2p, winShape, strides: winStrides)
        let vw  = MLX.asStrided(vp,  winShape, strides: winStrides)

        var q1g = q1.reshaped([B, H, G, block, D])
        var q2g = q2.reshaped([B, H, G, block, D])

        // Boundary mask suppresses zero-padded KV at sequence edges.
        let gIdx = MLXArray(0..<G).reshaped([G, 1])
        let wIdx = MLXArray(0..<W).reshaped([1, W])
        let paddedPos = gIdx * block + wIdx
        let inRange: MLXArray = (paddedPos .>= MLXArray(block)) * (paddedPos .< MLXArray(T + block))
        let boundary = MLX.where(
            inRange,
            MLXArray(Float(0.0)),
            MLXArray(Float(-1e9))
        ).expandedDimensions(axis: 1).asType(q1.dtype)             // [G, 1, W]

        let combinedRaw: MLXArray
        if let m = mask {
            combinedRaw = m + boundary   // broadcast [17,51] + [G,1,W] = [G,17,W]
        } else {
            combinedRaw = boundary
        }
        let combined = MLX.broadcast(combinedRaw.expandedDimensions(axis: 0),
                                      to: [B, G, block, W])
                          .reshaped([B * G, 1, block, W])

        q1g = q1g.transposed(0, 2, 1, 3, 4).reshaped([B * G, H, block, D])
        q2g = q2g.transposed(0, 2, 1, 3, 4).reshaped([B * G, H, block, D])
        let k1g = k1w.transposed(0, 2, 1, 3, 4).reshaped([B * G, H, W, D])
        let k2g = k2w.transposed(0, 2, 1, 3, 4).reshaped([B * G, H, W, D])
        let vg  = vw.transposed(0, 2, 1, 3, 4).reshaped([B * G, H, W, D])

        let Q = MLX.concatenated([q1g, q2g], axis: 1)
        let K = MLX.concatenated([k1g, k2g], axis: 1)
        let V = MLX.concatenated([vg, vg], axis: 1)
        let out = MLXFast.scaledDotProductAttention(queries: Q, keys: K, values: V,
                                                     scale: scale, mask: combined)
        let halves = MLX.split(out, parts: 2, axis: 1)
        let diff = halves[0] - halves[1]
        return diff.reshaped([B, G, H, block, D])
                    .transposed(0, 2, 1, 3, 4)
                    .reshaped([B, H, T, D])
    }
}

extension DiffSWA {
    /// Vireo: the same sliding-window differential attention without the copies of `swa` (one clip). Stock gathers
    /// every query group's 51 keys by transposing a strided window view into a contiguous [G, H, 51, D] copy (3× K and
    /// V), then concatenates the two attention passes' heads (2× again). Here each query group is an SDPA batch entry
    /// whose Q, K and V are strided views (batch stride 17 rows, head stride one head), the two passes are two SDPA
    /// calls, and the output is written once in [1, T, H·D]. Same kernel, same per-row arithmetic; the mask is the
    /// decoder's precombined band + boundary mask. Returns [1, T, dim].
    func swaViews(q1: MLXArray, k1: MLXArray, v: MLXArray, q2: MLXArray, k2: MLXArray, windowMask: MLXArray) -> MLXArray {
        let H = q1.dim(1), T = q1.dim(2), D = q1.dim(3)
        let block = SAMELDims.subChunkSize, G = T / block, W = 3 * block, Tp = T + 2 * block
        let pad: [IntOrPair] = [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((block, block)), IntOrPair((0, 0))]
        func queries(_ q: MLXArray) -> MLXArray { MLX.asStrided(q, [G, H, block, D], strides: [block * D, T * D, D, 1]) }
        func windows(_ k: MLXArray) -> MLXArray { MLX.asStrided(MLX.padded(k, widths: pad), [G, H, W, D], strides: [block * D, Tp * D, D, 1]) }
        let vw = windows(v)
        let o1 = MLXFast.scaledDotProductAttention(queries: queries(q1), keys: windows(k1), values: vw, scale: scale, mask: windowMask)
        let o2 = MLXFast.scaledDotProductAttention(queries: queries(q2), keys: windows(k2), values: vw, scale: scale, mask: windowMask)
        return (o1 - o2).transposed(0, 2, 1, 3).reshaped([1, T, H * D])
    }
}

/// Vireo: `swa`'s combined mask (band + sequence-edge boundary) as [G, 1, 17, 51] for one clip of T tokens.
func sameLWindowMask(tokens T: Int, dtype: DType) -> MLXArray {
    let block = SAMELDims.subChunkSize, G = T / block, W = 3 * block
    let gIdx = MLXArray(0..<G).reshaped([G, 1])
    let wIdx = MLXArray(0..<W).reshaped([1, W])
    let paddedPos = gIdx * block + wIdx
    let inRange: MLXArray = (paddedPos .>= MLXArray(block)) * (paddedPos .< MLXArray(T + block))
    let boundary = MLX.where(inRange, MLXArray(Float(0.0)), MLXArray(Float(-1e9))).expandedDimensions(axis: 1).asType(dtype)
    return (sameLSWAMask().asType(dtype) + boundary).reshaped([G, 1, block, W])
}

// MARK: - GeGLU / sin-gated feed-forward

public final class FeedForwardSAML: Module {
    @ModuleInfo(key: "glu_proj") public var gluProj: Linear
    @ModuleInfo(key: "proj_out") public var projOut: Linear
    public let useSin: Bool

    public init(_ c: SAMELConfig, useSin: Bool) {
        self.useSin = useSin
        self._gluProj.wrappedValue = Linear(c.dim, c.ffInner * 2, bias: true)
        self._projOut.wrappedValue = Linear(c.ffInner, c.dim, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = gluProj(x)
        let parts = MLX.split(p, parts: 2, axis: -1)
        let value = parts[0], gate = parts[1]
        let activated: MLXArray
        if useSin {
            activated = value * MLX.sin(gate * MLXArray(Float(Double.pi)).asType(gate.dtype))
        } else {
            activated = value * silu(gate)
        }
        return projOut(activated)
    }
}

public final class TransformerBlockSAML: Module {
    @ModuleInfo(key: "pre_norm") public var preNorm: DyT
    @ModuleInfo public var attn: DiffSWA
    @ModuleInfo(key: "ff_norm") public var ffNorm: DyT
    @ModuleInfo public var ff: FeedForwardSAML

    public init(_ c: SAMELConfig, blockIdx: Int) {
        self._preNorm.wrappedValue = DyT(dim: c.dim)
        self._attn.wrappedValue = DiffSWA(c)
        self._ffNorm.wrappedValue = DyT(dim: c.dim)
        self._ff.wrappedValue = FeedForwardSAML(c, useSin: blockIdx >= c.sinStartBlock)
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray, mask: MLXArray?, fullAttention: Bool) -> MLXArray {
        var x = xIn
        x = x + attn(preNorm(x), mask: mask, fullAttention: fullAttention)
        x = x + ff(ffNorm(x))
        return x
    }
}

// MARK: - SAME-L decoder

/// 426 M params, FP32. Input [B, 256, T_lat] → output [B, 512, T_lat*16].
public final class SAMELDecoder: Module {
    @ParameterInfo(key: "running_std") public var runningStd: MLXArray   // (1,)
    @ModuleInfo(key: "project_in") public var projectIn: Linear
    @ParameterInfo(key: "new_tokens") public var newTokens: MLXArray     // (1, 1, dim)
    @ModuleInfo public var blocks: [TransformerBlockSAML]
    /// Stored as Linear; the checkpoint Conv1d weight is reshaped (out, in, 1) → (out, in) at load.
    @ModuleInfo public var mapping: Linear

    public let config: SAMELConfig
    /// Vireo (VIREO_SA3_SWA_VIEWS): sliding-window attention through strided views (`DiffSWA.swaViews`).
    public var viewWindows = false {
        didSet { for b in blocks { b.attn.viewWindows = viewWindows } }
    }

    public init(_ c: SAMELConfig = .large) {
        self.config = c
        self._runningStd.wrappedValue = MLXArray([Float(1.0)])
        self._projectIn.wrappedValue = Linear(c.latentDim, c.dim, bias: true)
        self._newTokens.wrappedValue = MLXArray.zeros([1, 1, c.dim])
        self._blocks.wrappedValue = (0..<c.numBlocks).map { TransformerBlockSAML(c, blockIdx: $0) }
        self._mapping.wrappedValue = Linear(c.dim, c.outChannels, bias: true)
        super.init()
    }

    public var activationTrace: ((String, MLXArray) -> Void)?
    public func callAsFunction(_ latents: MLXArray, fullAttention: Bool = false) -> MLXArray {
        let B = latents.dim(0)
        let tLat = latents.dim(2)

        // Softnorm bottleneck decode (scalar) — runningStd is shape [1].
        let decoderInput = latents.asType(runningStd.dtype)
        var x = decoderInput * runningStd

        // [B, 256, T_lat] → [B, T_lat, 256] → [B, T_lat, DIM]
        x = projectIn(x.transposed(0, 2, 1))

        // Single new_token broadcast to 16 positions per latent slot, with the
        // original latent at position 0 of each 17-group.
        let xE = x.expandedDimensions(axis: 2)                                   // [B, T_lat, 1, DIM]
        let nt = MLX.broadcast(
            newTokens.expandedDimensions(axis: 0).asType(x.dtype),
            to: [B, tLat, SAMELDims.sinPerPos, config.dim])
        x = MLX.concatenated([xE, nt], axis: 2)                                  // [B, T_lat, 17, DIM]
        x = x.reshaped([B, tLat * SAMELDims.subChunkSize, config.dim])

        let mask = fullAttention ? nil
            : viewWindows && B == 1 && x.dim(1) > SAMELDims.subChunkSize ? sameLWindowMask(tokens: x.dim(1), dtype: x.dtype)
            : sameLSWAMask().asType(x.dtype)
        for (i, blk) in blocks.enumerated() {
            x = blk(x, mask: mask, fullAttention: fullAttention); activationTrace?("block-\(i)", x)
        }

        // Drop the original latent slot at index 0 of each 17-group; keep 16.
        x = x.reshaped([B, tLat, SAMELDims.subChunkSize, config.dim])
        x = x[0..., 0..., 1..., 0...]
        x = x.reshaped([B, tLat * SAMELDims.sinPerPos, config.dim])

        // [B, T_lat*16, DIM] → [B, T_lat*16, 512] → [B, 512, T_lat*16]
        return mapping(x).transposed(0, 2, 1)
    }
}

// MARK: - Chunked decode for long sequences

/// Uniform-kernel chunked decode. Mirrors `decode_chunked` in the upstream
/// Python — no zero-padding, three segments (first/interior/last) each see
/// `kernel = chunk + 2*overlap` real latents.
public func sameLDecodeChunked(_ model: SAMELDecoder, latents: MLXArray,
                                chunkSize: Int, overlap: Int) -> MLXArray {
    sa3DecodeChunked({ model($0) }, latents: latents, chunkSize: chunkSize, overlap: overlap)
}

/// `decode_chunked` of `same_l_decoder.py` / `same_s_decoder.py` (identical apart from SAME-S's even-window
/// assertion, which the callers' chunk/overlap choices satisfy).
public func sa3DecodeChunked(_ model: (MLXArray) -> MLXArray, latents: MLXArray,
                             chunkSize: Int, overlap: Int) -> MLXArray {
    let T = latents.dim(2)
    let kernel = chunkSize + 2 * overlap
    if T <= kernel { return model(latents) }

    var pieces: [MLXArray] = []

    // 1) First decode covers output positions [0, chunk + overlap)
    let firstOut = model(latents[0..., 0..., 0..<kernel])
    let validFirst = chunkSize + overlap
    pieces.append(firstOut[0..., 0..., 0..<(validFirst * SAMELDims.sinPerPos)])
    var i = validFirst

    // 2) Interior: stride by chunk
    while i + chunkSize + overlap <= T {
        let lo = i - overlap
        let hi = i + chunkSize + overlap
        let out = model(latents[0..., 0..., lo..<hi])
        let pieceLo = overlap * SAMELDims.sinPerPos
        let pieceHi = (overlap + chunkSize) * SAMELDims.sinPerPos
        pieces.append(out[0..., 0..., pieceLo..<pieceHi])
        i += chunkSize
    }

    // 3) Last decode covers remaining (T - i) output positions
    let remaining = T - i
    if remaining > 0 {
        let lastOut = model(latents[0..., 0..., (T - kernel)..<T])
        let tail = remaining * SAMELDims.sinPerPos
        let total = lastOut.dim(2)
        pieces.append(lastOut[0..., 0..., (total - tail)..<total])
    }
    return MLX.concatenated(pieces, axis: -1)
}

