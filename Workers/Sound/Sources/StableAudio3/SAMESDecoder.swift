import Foundation
import MLX
import MLXNN
import MLXFast

// Vireo port of Stability's SAME-S decoder (`stable-audio-3/optimized/mlx/models/defs/same_s_decoder.py`, MIT,
// commit 779434a), the codec of the small DiTs (sm-sfx, sm-music). Differences from SAME-L: 6 blocks of 768 wide
// differential attention without a sliding window, run over fixed chunks of 34 internal tokens (two latents with
// their 16 new tokens each): blocks 0–2 on aligned chunks, blocks 3–5 on chunks shifted by 17 (the ends padded with
// copies of the first/last 17 tokens, as the reference does); SiLU feed-forward only; a kernel-3 convolution maps to
// the 512 patch channels. Intake snapshots store half/packed weights; the input follows that stored dtype.

/// SAME-S decoder shape; `.small` mirrors `same_s_decoder.py`.
public struct SAMESConfig: Sendable, Equatable {
    public var latentDim = 256
    public var dim = 768
    public var numHeads = 12
    public var headDim = 64
    public var ropeDims = 32
    public var numBlocks = 6
    public var ffInner = 2304
    public var outChannels = 512
    /// Latents per attention chunk (the internal chunk is `chunkLatents * 17` tokens).
    public var chunkLatents = 2
    public static let small = SAMESConfig()
    public init() {}
}

/// Full (non-windowed) differential attention: `to_qkv → q, k, v, q_diff, k_diff`, DyT QK-norm, RoPE on the
/// first `ropeDims`, `SDPA(q, k, v) − SDPA(q_diff, k_diff, v)`.
public final class SAMESAttention: Module {
    @ModuleInfo(key: "to_qkv") public var toQKV: Linear
    @ModuleInfo(key: "to_out") public var toOut: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: DyT
    @ModuleInfo(key: "k_norm") public var kNorm: DyT

    public let scale: Float
    let c: SAMESConfig

    public init(_ c: SAMESConfig) {
        self.c = c
        self._toQKV.wrappedValue = Linear(c.dim, 5 * c.dim, bias: false)
        self._toOut.wrappedValue = Linear(c.dim, c.dim, bias: false)
        self._qNorm.wrappedValue = DyT(dim: c.headDim)
        self._kNorm.wrappedValue = DyT(dim: c.headDim)
        self.scale = 1.0 / Float(c.headDim).squareRoot()
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let parts = MLX.split(toQKV(x), parts: 5, axis: -1)
        func heads(_ t: MLXArray) -> MLXArray { t.reshaped([B, T, c.numHeads, c.headDim]).transposed(0, 2, 1, 3) }
        func rope(_ t: MLXArray) -> MLXArray {
            MLXFast.RoPE(t, dimensions: c.ropeDims, traditional: false, base: 10000.0, scale: 1.0, offset: 0)
        }
        let q = rope(qNorm(heads(parts[0])))
        let k = rope(kNorm(heads(parts[1])))
        let v = heads(parts[2])
        let qDiff = rope(qNorm(heads(parts[3])))
        let kDiff = rope(kNorm(heads(parts[4])))
        let outMain = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let outDiff = MLXFast.scaledDotProductAttention(queries: qDiff, keys: kDiff, values: v, scale: scale, mask: nil)
        return toOut((outMain - outDiff).transposed(0, 2, 1, 3).reshaped([B, T, c.dim]))
    }
}

public final class SAMESFeedForward: Module {
    @ModuleInfo(key: "glu_proj") public var gluProj: Linear
    @ModuleInfo(key: "proj_out") public var projOut: Linear
    public init(_ c: SAMESConfig) {
        self._gluProj.wrappedValue = Linear(c.dim, c.ffInner * 2, bias: true)
        self._projOut.wrappedValue = Linear(c.ffInner, c.dim, bias: true)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(gluProj(x), parts: 2, axis: -1)
        return projOut(parts[0] * silu(parts[1]))
    }
}

public final class SAMESBlock: Module {
    @ModuleInfo(key: "pre_norm") public var preNorm: DyT
    @ModuleInfo public var attn: SAMESAttention
    @ModuleInfo(key: "ff_norm") public var ffNorm: DyT
    @ModuleInfo public var ff: SAMESFeedForward

    public init(_ c: SAMESConfig) {
        self._preNorm.wrappedValue = DyT(dim: c.dim)
        self._attn.wrappedValue = SAMESAttention(c)
        self._ffNorm.wrappedValue = DyT(dim: c.dim)
        self._ff.wrappedValue = SAMESFeedForward(c)
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        let x = xIn + attn(preNorm(xIn))
        return x + ff(ffNorm(x))
    }
}

/// ~53 M params, half/packed weights. Input [B, 256, T_lat] (T_lat even) → output [B, 512, T_lat*16].
public final class SAMESDecoder: Module {
    @ParameterInfo(key: "running_std") public var runningStd: MLXArray   // (1,)
    @ModuleInfo(key: "project_in") public var projectIn: Linear
    @ParameterInfo(key: "new_tokens") public var newTokens: MLXArray     // (1, 1, dim)
    @ModuleInfo public var blocks: [SAMESBlock]
    /// Weight-normalised Conv1d in training, stored fused; the checkpoint's PyTorch layout [out, in, k] is permuted to
    /// MLX's [out, k, in] at load.
    @ModuleInfo public var mapping: Conv1d

    public let config: SAMESConfig

    public init(_ c: SAMESConfig = .small) {
        self.config = c
        self._runningStd.wrappedValue = MLXArray([Float(1.0)])
        self._projectIn.wrappedValue = Linear(c.latentDim, c.dim, bias: true)
        self._newTokens.wrappedValue = MLXArray.zeros([1, 1, c.dim])
        self._blocks.wrappedValue = (0..<c.numBlocks).map { _ in SAMESBlock(c) }
        self._mapping.wrappedValue = Conv1d(inputChannels: c.dim, outputChannels: c.outChannels, kernelSize: 3, padding: 1, bias: true)
        super.init()
    }

    public var activationTrace: ((String, MLXArray) -> Void)?

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let latents = input.asType(runningStd.dtype)
        let B = latents.dim(0), tLat = latents.dim(2), D = config.dim
        let group = SAMELDims.subChunkSize                                        // 17: the latent + 16 new tokens
        let chunk = config.chunkLatents * group                                   // 34
        let shift = chunk / 2                                                     // 17
        precondition(tLat % config.chunkLatents == 0, "SAME-S decodes an even number of latents (got \(tLat))")

        var x = projectIn((latents * runningStd.asType(latents.dtype)).transposed(0, 2, 1))   // [B, T, D]
        let nt = MLX.broadcast(newTokens.expandedDimensions(axis: 0).asType(x.dtype), to: [B, tLat, SAMELDims.sinPerPos, D])
        x = MLX.concatenated([x.expandedDimensions(axis: 2), nt], axis: 2)        // [B, T, 17, D]
        let internalT = tLat * group
        x = x.reshaped([B * (internalT / chunk), chunk, D])

        let half = config.numBlocks / 2
        for (i, block) in blocks[..<half].enumerated() {
            x = block(x); activationTrace?("block-\(i)", x)
        }
        x = x.reshaped([B, internalT, D])

        // Second half on chunks shifted by 17: the ends are padded with copies of the first/last 17 tokens.
        x = MLX.concatenated([x[0..., ..<shift, 0...], x, x[0..., (internalT - shift)..., 0...]], axis: 1)
        x = x.reshaped([B * ((internalT + chunk) / chunk), chunk, D])
        for (i, block) in blocks[half...].enumerated() {
            x = block(x); activationTrace?("block-\(i + half)", x)
        }
        x = x.reshaped([B, internalT + chunk, D])[0..., shift..<(shift + internalT), 0...]

        // Drop the latent slot of each 17-group, keep the 16 new-token positions.
        x = x.reshaped([B * tLat, group, D])[0..., 1..., 0...].reshaped([B, tLat * SAMELDims.sinPerPos, D])
        return mapping(x).transposed(0, 2, 1)                                     // [B, 512, T*16]
    }
}
