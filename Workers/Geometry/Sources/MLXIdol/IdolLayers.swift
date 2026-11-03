// SPDX-License-Identifier: MIT
// IDOL's building blocks (speridlabs/idol-sdk idol/models/layers.py), in MLX. Module keys are the checkpoint's own
// names, so a checkpoint loads without renaming. Weights are half; where the reference computes in float32 (its
// LayerNorm32, RMS norm, rotary angles, embeddings) the activations here are float32 too.
import Foundation
import MLX
import MLXFast
import MLXNN

/// LayerNorm computed in float32 and returned in the input's dtype (the reference's LayerNorm32 / F.layer_norm).
func layerNorm32(_ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float) -> MLXArray {
    MLXFast.layerNorm(x.asType(.float32), weight: weight?.asType(.float32), bias: bias?.asType(.float32), eps: eps).asType(x.dtype)
}

/// An affine LayerNorm whose parameters are named `weight` and `bias`, computed in float32.
final class IdolLayerNorm: Module {
    let weight: MLXArray, bias: MLXArray
    let eps: Float
    init(_ channels: Int, eps: Float) {
        weight = MLXArray.ones([channels]); bias = MLXArray.zeros([channels]); self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layerNorm32(x, weight: weight, bias: bias, eps: eps) }
}

/// F.normalize(x.float(), dim=-1) * gamma * sqrt(dim), back to x's dtype. gamma: [heads, dim].
final class MultiHeadRMSNorm: Module {
    let gamma: MLXArray
    let scale: Float
    init(dim: Int, heads: Int) { gamma = MLXArray.ones([heads, dim]); scale = Float(dim).squareRoot() }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let norm = MLX.maximum(MLX.sqrt((f * f).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        return (f / norm * gamma.asType(.float32) * scale).asType(x.dtype)
    }
}

/// nn.Sequential(Linear, GELU(tanh), Linear): checkpoint keys `0` and `2`, here `l0` and `l2` (see `IdolWeights.moduleKey`).
final class GeluMLP: Module {
    @ModuleInfo(key: "l0") var fc1: Linear
    @ModuleInfo(key: "l2") var fc2: Linear
    init(_ input: Int, _ hidden: Int, _ output: Int) {
        _fc1.wrappedValue = Linear(input, hidden); _fc2.wrappedValue = Linear(hidden, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

/// FeedForwardNet: `mlp.0`, `mlp.2`.
final class FeedForwardNet: Module {
    @ModuleInfo var mlp: GeluMLP
    init(_ channels: Int, ratio: Int = 4) { _mlp.wrappedValue = GeluMLP(channels, channels * ratio, channels) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { mlp(x) }
}

/// nn.Sequential(SiLU, Linear): checkpoint key `1`, here `l1`.
final class SiLULinear: Module {
    @ModuleInfo(key: "l1") var linear: Linear
    init(_ input: Int, _ output: Int) { _linear.wrappedValue = Linear(input, output) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(silu(x)) }
}

/// nn.Sequential(Linear, SiLU, Linear): checkpoint keys `0` and `2`, here `l0` and `l2`.
final class SiLUMLP: Module {
    @ModuleInfo(key: "l0") var fc1: Linear
    @ModuleInfo(key: "l2") var fc2: Linear
    init(_ input: Int, _ output: Int) { _fc1.wrappedValue = Linear(input, output); _fc2.wrappedValue = Linear(output, output) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(silu(fc1(x))) }
}

/// Embeds one scalar per batch element (diffusion time, octree resolution). The float32 sinusoid meets half weights,
/// so the MLP computes in float32 as the reference does.
final class SinusoidalEmbedder: Module {
    @ModuleInfo var mlp: SiLUMLP
    let frequencies: Int, maxPeriod: Double, twoPi: Bool
    init(_ hidden: Int, frequencies: Int = 256, maxPeriod: Double = 10000, twoPi: Bool = false) {
        _mlp.wrappedValue = SiLUMLP(frequencies, hidden)
        self.frequencies = frequencies; self.maxPeriod = maxPeriod; self.twoPi = twoPi
    }
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let half = frequencies / 2
        let freqs = (0..<half).map { Float(exp(-log(maxPeriod) * Double(Float($0)) / Double(half))) }
        var args = t.asType(.float32).reshaped([-1, 1]) * MLXArray(freqs).reshaped([1, half])
        if twoPi { args *= Float(2 * Double.pi) }
        return mlp(MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1))
    }
}

enum PositionEmbedding {
    /// The denoiser's fixed latent positions (PcdAbsolutePositionEmbedder): x in [0, 1], frequencies 2^0..2^15 then
    /// 2^(16·i/res_dim). The angles reach about 4e5 rad, so this runs once on the host: each angle is formed in float32
    /// with the reference's operation order and its sine taken in double, which pins every value to the float32
    /// reference instead of a GPU's large-argument range reduction. Returns [count, channels] float32.
    static func latentTable(_ positions: [Float], channels: Int, maxResolution: Int = 16) -> [Float] {
        let inChannels = 3, freqDim = channels / inChannels / 2
        let resDim = max(0, freqDim - maxResolution)
        var exponents = (0..<maxResolution).map { Float($0) }
        exponents += (0..<resDim).map { Float($0) / Float(resDim) * Float(maxResolution) }
        let freqs = exponents.prefix(freqDim).map { Float(pow(2.0, Double($0))) }
        let points = positions.count / inChannels
        var table = [Float](repeating: 0, count: points * channels)
        for p in 0..<points {
            for c in 0..<inChannels {
                let x = positions[p * inChannels + c], base = p * channels + c * 2 * freqDim
                for (f, freq) in freqs.enumerated() {
                    let angle = (x * freq) * 2 * Float.pi // torch.outer(x, freqs) * 2 * torch.pi, in float32
                    table[base + f] = Float(sin(Double(angle)))
                    table[base + freqDim + f] = Float(cos(Double(angle)))
                }
            }
        }
        return table
    }

    /// The decoders' query points (PcdAbsolutePositionEmbedderv2): x [N, 3] in [0, 1] float32 -> [N, channels].
    static func points(_ x: MLXArray, channels: Int, maxResolution: Float = 10) -> MLXArray {
        let n = x.dim(0), freqDim = channels / 3 / 2
        let logs = (0..<freqDim).map { freqDim == 1 ? Float(0) : Float($0) * maxResolution / Float(freqDim - 1) }
        let freqs = MLXArray(logs.map { Float(pow(2.0, Double($0))) }).reshaped([1, 1, freqDim])
        let angles = x.asType(.float32).expandedDimensions(axis: -1) * freqs * Float.pi
        var embed = MLX.concatenated([MLX.sin(angles), MLX.cos(angles)], axis: -1).reshaped([n, -1])
        if embed.dim(1) < channels { embed = MLX.padded(embed, widths: [IntOrPair(0), IntOrPair((0, channels - embed.dim(1)))]) }
        return embed
    }
}

/// x * tanh(f) + x * (f - tanh(f)), numerically x * f (the reference's straight-through clamp, identical forward).
func clampMul(_ x: MLXArray, _ f: MLXArray) -> MLXArray {
    let t = MLX.tanh(f)
    return x * t + x * (f - t)
}

/// RePo: each token's features predict a 3D offset per head, turned into rotary angles. Returns (cos, sin) float32
/// [B, L, heads, headDim / 2].
final class RePo3DRotaryEmbedding: Module {
    @ModuleInfo var norm: IdolLayerNorm
    @ModuleInfo(key: "gate_map") var gateMap: Linear
    @ModuleInfo(key: "content_map") var contentMap: Linear
    @ModuleInfo(key: "final_map") var finalMap: Linear
    @ParameterInfo(key: "freqs_0") var freqs0: MLXArray
    @ParameterInfo(key: "freqs_1") var freqs1: MLXArray
    @ParameterInfo(key: "freqs_2") var freqs2: MLXArray
    let heads: Int
    init(channels: Int, heads: Int, headDim: Int, hiddenRatio: Double = 0.125, maxFreq: Float = 16) {
        let hidden = Int(Double(channels) * hiddenRatio)
        self.heads = heads
        _norm.wrappedValue = IdolLayerNorm(channels, eps: 1e-5)
        _gateMap.wrappedValue = Linear(channels, hidden, bias: false)
        _contentMap.wrappedValue = Linear(channels, hidden, bias: false)
        _finalMap.wrappedValue = Linear(hidden, 3 * heads, bias: false)
        let d0 = 2 * (headDim / 6), dims = [d0, d0, headDim - 2 * d0]
        func linspace(_ n: Int) -> MLXArray {
            MLXArray((0..<n).map { n == 1 ? Float(1) : 1 + Float($0) * (maxFreq - 1) / Float(n - 1) })
        }
        _freqs0.wrappedValue = linspace(dims[0] / 2); _freqs1.wrappedValue = linspace(dims[1] / 2); _freqs2.wrappedValue = linspace(dims[2] / 2)
    }
    func callAsFunction(_ hidden: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let h = norm(hidden.asType(.float32))
        let out = finalMap(silu(gateMap(h)) * contentMap(h)).asType(.float32)
        let (b, l) = (out.dim(0), out.dim(1))
        let delta = out.reshaped([b, l, heads, 3])
        func angle(_ i: Int, _ f: MLXArray) -> MLXArray {
            clampMul(delta[.ellipsis, i..<i + 1], f.asType(.float32)) * Float.pi
        }
        let a = MLX.concatenated([angle(0, freqs0), angle(1, freqs1), angle(2, freqs2)], axis: -1)
        return (MLX.cos(a), MLX.sin(a))
    }
}

/// Rotates interleaved pairs (x0, x1) of x [B, L, H, D] by (cos, sin) [B, L, H, D/2] in float32 (torch.view_as_complex).
func applyRotary(_ x: MLXArray, _ rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
    let shape = x.shape
    let pairs = x.asType(.float32).reshaped(shape.dropLast() + [shape.last! / 2, 2])
    let x0 = pairs[.ellipsis, 0], x1 = pairs[.ellipsis, 1]
    let out = MLX.stacked([x0 * rope.cos - x1 * rope.sin, x0 * rope.sin + x1 * rope.cos], axis: -1)
    return out.reshaped(shape).asType(x.dtype)
}

/// q, k, v [B, L, H, D] -> [B, L, H·D] through the fused attention kernel.
func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
    let (b, l, h, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let o = MLXFast.scaledDotProductAttention(
        queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
        scale: 1 / Float(d).squareRoot(), mask: nil)
    return o.transposed(0, 2, 1, 3).reshaped([b, l, h * d])
}

/// The denoiser's self attention: rotary embeddings, then RMS-normalised queries and keys.
final class RopeMultiHeadAttention: Module {
    @ModuleInfo var qkv: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MultiHeadRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MultiHeadRMSNorm
    @ModuleInfo var out: Linear
    let heads: Int, headDim: Int
    init(_ channels: Int, heads: Int) {
        self.heads = heads; headDim = channels / heads
        _qkv.wrappedValue = Linear(channels, channels * 3)
        _qNorm.wrappedValue = MultiHeadRMSNorm(dim: headDim, heads: heads); _kNorm.wrappedValue = MultiHeadRMSNorm(dim: headDim, heads: heads)
        _out.wrappedValue = Linear(channels, channels)
    }
    func callAsFunction(_ x: MLXArray, rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let parts = qkv(x).reshaped([b, l, 3, heads, headDim])
        let q = qNorm(applyRotary(parts[0..., 0..., 0], rope)), k = kNorm(applyRotary(parts[0..., 0..., 1], rope))
        return out(attention(q, k, parts[0..., 0..., 2]))
    }
}

/// The decoders' self or cross attention with RMS-normalised queries and keys.
final class MultiHeadAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear?
    @ModuleInfo(key: "to_kv") var toKV: Linear?
    @ModuleInfo(key: "to_qkv") var toQKV: Linear?
    @ModuleInfo(key: "q_rms_norm") var qNorm: MultiHeadRMSNorm
    @ModuleInfo(key: "k_rms_norm") var kNorm: MultiHeadRMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear
    let heads: Int, headDim: Int
    init(_ channels: Int, heads: Int, context: Int? = nil) {
        self.heads = heads; headDim = channels / heads
        if let context {
            _toQ.wrappedValue = Linear(channels, channels); _toKV.wrappedValue = Linear(context, channels * 2); _toQKV.wrappedValue = nil
        } else {
            _toQ.wrappedValue = nil; _toKV.wrappedValue = nil; _toQKV.wrappedValue = Linear(channels, channels * 3)
        }
        _qNorm.wrappedValue = MultiHeadRMSNorm(dim: headDim, heads: heads); _kNorm.wrappedValue = MultiHeadRMSNorm(dim: headDim, heads: heads)
        _toOut.wrappedValue = Linear(channels, channels)
    }
    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        if let toQ, let toKV, let context {
            let q = qNorm(toQ(x).reshaped([b, l, heads, headDim]))
            let kv = toKV(context).reshaped([b, context.dim(1), 2, heads, headDim])
            return toOut(attention(q, kNorm(kv[0..., 0..., 0]), kv[0..., 0..., 1]))
        }
        let parts = toQKV!(x).reshaped([b, l, 3, heads, headDim])
        return toOut(attention(qNorm(parts[0..., 0..., 0]), kNorm(parts[0..., 0..., 1]), parts[0..., 0..., 2]))
    }
}

/// Splits [B, 6·C] modulation into six [B, 1, C] pieces.
func sixWay(_ mod: MLXArray) -> [MLXArray] {
    let c = mod.dim(1) / 6
    return (0..<6).map { mod[0..., ($0 * c)..<(($0 + 1) * c)].expandedDimensions(axis: 1) }
}

/// The denoiser block: RoPE self attention + FFN, modulated by the time (shared adaLN plus this block's shift table);
/// without modulation (the condition refiner) the two LayerNorms are affine.
final class UnifiedTransformerBlock: Module {
    @ModuleInfo var norm1: IdolLayerNorm?
    @ModuleInfo var norm2: IdolLayerNorm?
    @ModuleInfo var attn: RopeMultiHeadAttention
    @ModuleInfo var mlp: FeedForwardNet
    @ParameterInfo(key: "shift_table") var shiftTable: MLXArray?
    let modulation: Bool
    init(_ channels: Int, heads: Int, modulation: Bool) {
        self.modulation = modulation
        _norm1.wrappedValue = modulation ? nil : IdolLayerNorm(channels, eps: 1e-6)
        _norm2.wrappedValue = modulation ? nil : IdolLayerNorm(channels, eps: 1e-6)
        _attn.wrappedValue = RopeMultiHeadAttention(channels, heads: heads)
        _mlp.wrappedValue = FeedForwardNet(channels)
        _shiftTable.wrappedValue = modulation ? MLXArray.zeros([1, 6 * channels]) : nil
    }
    func callAsFunction(_ x: MLXArray, mod: MLXArray?, rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        guard modulation, let mod, let shiftTable else {
            let h = x + attn(norm1!(x), rope: rope)
            return h + mlp(norm2!(h))
        }
        let m = sixWay(mod + shiftTable.asType(mod.dtype))
        var h = layerNorm32(x, eps: 1e-6) * (1 + m[1]) + m[0]
        var out = x + attn(h, rope: rope) * m[2]
        h = layerNorm32(out, eps: 1e-6) * (1 + m[4]) + m[3]
        out += mlp(h) * m[5]
        return out
    }
}

/// The Gaussian decoder's block: self attention, cross attention to the latent, FFN.
final class TransformerCrossBlock: Module {
    @ModuleInfo var norm2: IdolLayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: MultiHeadAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiHeadAttention
    @ModuleInfo var mlp: FeedForwardNet
    init(_ channels: Int, context: Int, heads: Int) {
        _norm2.wrappedValue = IdolLayerNorm(channels, eps: 1e-6)
        _selfAttn.wrappedValue = MultiHeadAttention(channels, heads: heads)
        _crossAttn.wrappedValue = MultiHeadAttention(channels, heads: heads, context: context)
        _mlp.wrappedValue = FeedForwardNet(channels)
    }
    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var h = x + selfAttn(layerNorm32(x, eps: 1e-6))
        h += crossAttn(norm2(h), context: context)
        return h + mlp(layerNorm32(h, eps: 1e-6))
    }
}

/// The octree decoder's block: cross attention to the latent + FFN, modulated by the octree level.
final class ModulatedCrossOnlyBlock: Module {
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiHeadAttention
    @ModuleInfo var mlp: FeedForwardNet
    init(_ channels: Int, context: Int, heads: Int) {
        _crossAttn.wrappedValue = MultiHeadAttention(channels, heads: heads, context: context)
        _mlp.wrappedValue = FeedForwardNet(channels)
    }
    func callAsFunction(_ x: MLXArray, mod: MLXArray, context: MLXArray) -> MLXArray {
        let m = sixWay(mod)
        var h = layerNorm32(x, eps: 1e-6) * (1 + m[1]) + m[0]
        var out = x + crossAttn(h, context: context) * m[2]
        h = layerNorm32(out, eps: 1e-6) * (1 + m[4]) + m[3]
        out += mlp(h) * m[5]
        return out
    }
}
